use super::*;
use crate::solver::core::{
    cones::{CompositeCone, Cone},
    kktsolvers::{direct::*, *},
    traits::{KKTSystem, Settings},
    StepDirection,
};

use crate::algebra::*;

// We require Send/Sync here to allow pyo3 builds to share
// solver objects between threads.

type BoxedKKTSolver<T> = Box<dyn KKTSolver<T> + Send + Sync>;

/// Standard-form solver type implementing the [`KKTSystem`](crate::solver::core::traits::KKTSystem) trait
pub struct DefaultKKTSystem<T> {
    kktsolver: BoxedKKTSolver<T>,

    // solution vector for constant part of KKT solves
    x1: Vec<T>,
    z1: Vec<T>,

    // solution vector for general KKT solves
    x2: Vec<T>,
    z2: Vec<T>,

    // P*x2, and P*ξ with ξ = x/τ for the current iterate (cached
    // across the affine and combined solves of one iteration)
    Px2: Vec<T>,
    Pξ: Vec<T>,
    Pξ_valid: bool,
    // P*x for the current iterate, supplied by the solver when available
    Px_hint: Option<Vec<T>>,
    // q + 2Pξ, and the Δτ denominator, for the current iteration
    qp: Vec<T>,
    tau_den: T,

    // work vectors for assembling/dissambling vectors
    workx: Vec<T>,
    workz: Vec<T>,
    work_conic: Vec<T>,
}

impl<T> DefaultKKTSystem<T>
where
    T: FloatT,
{
    /// Create a new KKT system solver
    pub fn new(
        data: &DefaultProblemData<T>,
        cones: &CompositeCone<T>,
        settings: &DefaultSettings<T>,
    ) -> Self {
        let (m, n) = (data.m, data.n);

        //here we allow scope for different KKT solvers, e.g.
        //direct vs indirect, different QR based direct methods
        //etc.   For now, we only have direct / LDL based
        // The bound eliminating solver is used by the "auto" method for
        // problems with only zero / nonnegative cones and at least one
        // bound row.  An explicit "qdldl" selects the full quasidefinite
        // KKT system factored by QDLDL.
        let use_boundelim = settings.direct_kkt_solver
            && settings.direct_solve_method == "auto"
            && BoundElimKKTSolver::<T>::is_applicable(&data.A, cones);

        let kktsolver: BoxedKKTSolver<T> = if use_boundelim {
            Box::new(BoundElimKKTSolver::<T>::new(
                &data.P,
                &data.A,
                cones,
                m,
                n,
                settings.core(),
            ))
        } else if settings.direct_kkt_solver {
            Box::new(DirectLDLKKTSolver::<T>::new(
                &data.P,
                &data.A,
                cones,
                m,
                n,
                settings.core(),
            ))
        } else {
            panic!("Indirect and other solve strategies not yet supported.");
        };

        //the LHS constant part of the reduced solve
        let x1 = vec![T::zero(); n];
        let z1 = vec![T::zero(); m];

        //the LHS for other solves
        let x2 = vec![T::zero(); n];
        let z2 = vec![T::zero(); m];

        //workspace compatible with (x,z)
        let workx = vec![T::zero(); n];
        let workz = vec![T::zero(); m];

        //additional conic workspace vector compatible with s and z
        let work_conic = vec![T::zero(); m];

        Self {
            kktsolver,
            x1,
            z1,
            x2,
            z2,
            Px2: vec![T::zero(); n],
            Pξ: vec![T::zero(); n],
            Pξ_valid: false,
            Px_hint: None,
            qp: vec![T::zero(); n],
            tau_den: T::zero(),
            workx,
            workz,
            work_conic,
        }
    }
}

impl<T> HasLinearSolverInfo for DefaultKKTSystem<T>
where
    T: FloatT,
{
    fn linear_solver_info(&self) -> LinearSolverInfo {
        self.kktsolver.linear_solver_info()
    }
}

impl<T> KKTSystem<T> for DefaultKKTSystem<T>
where
    T: FloatT,
{
    type D = DefaultProblemData<T>;
    type V = DefaultVariables<T>;
    type C = CompositeCone<T>;
    type SE = DefaultSettings<T>;

    fn update(
        &mut self,
        data: &DefaultProblemData<T>,
        cones: &CompositeCone<T>,
        settings: &DefaultSettings<T>,
    ) -> bool {
        // update the linear solver with new cones
        let is_success = self.kktsolver.update(cones, settings.core());

        if !is_success {
            return is_success;
        }

        // calculate KKT solution for constant terms
        self.solve_constant_rhs(data, settings.core())

        //PJG is_success should be a Result in rust
    }

    fn solve(
        &mut self,
        lhs: &mut DefaultVariables<T>,
        rhs: &DefaultVariables<T>,
        data: &DefaultProblemData<T>,
        variables: &DefaultVariables<T>,
        cones: &mut CompositeCone<T>,
        step_direction: StepDirection,
        settings: &DefaultSettings<T>,
    ) -> bool {
        let (x1, z1) = (&mut self.x1, &mut self.z1);
        let (workx, workz) = (&mut self.workx, &mut self.workz);

        // solve for (x1,z1)
        // -----------
        workx.copy_from(&rhs.x);

        // compute the vector c in the step equation HₛΔz + Δs = -c,
        // with shortcut in affine case
        let Δs_const_term = &mut self.work_conic;

        match step_direction {
            StepDirection::Affine => {
                Δs_const_term.copy_from(&variables.s);
            }
            StepDirection::Combined => {
                cones.Δs_from_Δz_offset(Δs_const_term, &rhs.s, &mut lhs.z, &variables.z);
            }
        }

        workz.waxpby(T::one(), Δs_const_term, -T::one(), &rhs.z);

        // ---------------------------------------------------
        // this solves the variable part of reduced KKT system
        self.kktsolver.setrhs(workx, workz);
        let is_success = self.kktsolver.solve(Some(x1), Some(z1), settings.core());
        if !is_success {
            return false;
        }

        self.finish_solve(lhs, rhs, data, variables, cones);

        // we don't check the validity of anything
        // after the KKT solve, so just return is_success
        // without further validation
        is_success
    }

    fn set_Px_hint(&mut self, Px: Option<&[T]>) {
        match Px {
            Some(Px) => match &mut self.Px_hint {
                Some(v) => {
                    v.copy_from(Px);
                }
                None => self.Px_hint = Some(Px.to_vec()),
            },
            None => self.Px_hint = None,
        }
    }

    fn update_and_solve_affine(
        &mut self,
        lhs: &mut DefaultVariables<T>,
        rhs: &DefaultVariables<T>,
        data: &DefaultProblemData<T>,
        variables: &DefaultVariables<T>,
        cones: &mut CompositeCone<T>,
        settings: &DefaultSettings<T>,
    ) -> (bool, bool) {
        if !self.kktsolver.update(cones, settings.core()) {
            return (false, false);
        }

        // constant part [-q; b] and the affine step part [rhs.x; s - rhs.z]
        // are solved together with the new factors
        self.workx.axpby(-T::one(), &data.q, T::zero());
        self.work_conic.copy_from(&variables.s);
        self.workz
            .waxpby(T::one(), &self.work_conic, -T::one(), &rhs.z);

        let is_success = self.kktsolver.solve_pair(
            [(&self.workx, &data.b), (&rhs.x, &self.workz)],
            [(&mut self.x2, &mut self.z2), (&mut self.x1, &mut self.z1)],
            settings.core(),
        );
        if !is_success {
            return (true, false);
        }
        self.after_constant_solve(data);
        self.finish_solve(lhs, rhs, data, variables, cones);
        (true, true)
    }

    fn solve_initial_point(
        &mut self,
        variables: &mut DefaultVariables<T>,
        data: &DefaultProblemData<T>,
        settings: &DefaultSettings<T>,
    ) -> bool {
        let mut is_success;

        if data.P.nnz() == 0 {
            // LP initialization
            // solve with [0;b] as a RHS to get (x,-s) initializers
            // zero out any sparse cone variables at end
            self.workx.fill(T::zero());
            self.workz.copy_from(&data.b);
            self.kktsolver.setrhs(&self.workx, &self.workz);
            is_success = self.kktsolver.solve(
                Some(&mut variables.x),
                Some(&mut variables.s),
                settings.core(),
            );
            variables.s.negate();

            if !is_success {
                return is_success;
            }

            // solve with [-q;0] as a RHS to get z initializer
            // zero out any sparse cone variables at end
            self.workx.axpby(-T::one(), &data.q, T::zero());
            self.workz.fill(T::zero());

            self.kktsolver.setrhs(&self.workx, &self.workz);
            is_success = self
                .kktsolver
                .solve(None, Some(&mut variables.z), settings.core());
        } else {
            //QP initialization
            self.workx.scalarop_from(|q| -q, &data.q);
            self.workz.copy_from(&data.b);
            self.kktsolver.setrhs(&self.workx, &self.workz);
            is_success = self.kktsolver.solve(
                Some(&mut variables.x),
                Some(&mut variables.z),
                settings.core(),
            );
            variables.s.scalarop_from(|z| -z, &variables.z);
        }
        is_success
    }
}

impl<T> DefaultKKTSystem<T>
where
    T: FloatT,
{
    // given (x1,z1) from the variable part of the KKT solve, and the
    // constant term Δs_const_term in self.work_conic, complete the step
    fn finish_solve(
        &mut self,
        lhs: &mut DefaultVariables<T>,
        rhs: &DefaultVariables<T>,
        data: &DefaultProblemData<T>,
        variables: &DefaultVariables<T>,
        cones: &mut CompositeCone<T>,
    ) {
        let (x1, z1) = (&self.x1, &self.z1);
        let (x2, z2) = (&self.x2, &self.z2);
        let (workx, workz) = (&mut self.workx, &mut self.workz);
        let Δs_const_term = &self.work_conic;

        // solve for Δτ.
        // -----------
        // The quadratic forms below are all evaluated through the cached
        // products P*x2 (computed once per KKT update) and P*ξ (computed
        // once per iteration), so each solve needs only dot products:
        //   2 ξ'P x1,  (ξ-x2)'P(ξ-x2) - x2'Px2 = ξ'Pξ - 2 ξ'Px2
        // Everything except the x1/z1 terms is fixed for the iteration, so
        // is computed with P*ξ on the first (affine) solve after an update:
        // the denominator, and qp = q + 2Pξ for the numerator.
        if !self.Pξ_valid {
            let ξ = &mut *workx;
            ξ.axpby(T::recip(variables.τ), &variables.x, T::zero());
            match &self.Px_hint {
                Some(Px) => {
                    self.Pξ.axpby(T::recip(variables.τ), Px, T::zero());
                }
                None => data.P_mul(&mut self.Pξ, ξ),
            }
            let two: T = (2.).as_T();
            let ξPξ = ξ.dot(&self.Pξ);
            let ξPx2 = ξ.dot(&self.Px2);
            self.tau_den = variables.κ / variables.τ - data.q.dot(x2) - data.b.dot(z2) + ξPξ
                - two * ξPx2;
            self.qp.waxpby(T::one(), &data.q, two, &self.Pξ);
            self.Pξ_valid = true;
        }

        let tau_num = rhs.τ - rhs.κ / variables.τ + self.qp.dot(x1) + data.b.dot(z1);
        let tau_den = self.tau_den;

        // solve for (Δx,Δz)
        // -----------
        lhs.τ = tau_num / tau_den;
        lhs.x.waxpby(T::one(), x1, lhs.τ, x2);
        lhs.z.waxpby(T::one(), z1, lhs.τ, z2);

        // solve for Δs
        // -------------
        //  compute the linear term HₛΔz, where Hs = WᵀW for symmetric
        //  cones and Hs = μH(z) for asymmetric cones
        cones.mul_Hs(&mut lhs.s, &lhs.z, workz);
        lhs.s.axpby(-T::one(), Δs_const_term, -T::one()); // lhs.s = -(lhs.s+Δs_const_term);

        // solve for Δκ
        // --------------
        lhs.κ = -(rhs.κ + variables.κ * lhs.τ) / variables.τ;

    }

    fn solve_constant_rhs(
        &mut self,
        data: &DefaultProblemData<T>,
        settings: &DefaultSettings<T>,
    ) -> bool {
        self.workx.axpby(-T::one(), &data.q, T::zero()); //workx .= -q
        self.kktsolver.setrhs(&self.workx, &data.b);
        let is_success =
            self.kktsolver
                .solve(Some(&mut self.x2), Some(&mut self.z2), settings.core());

        self.after_constant_solve(data);
        is_success
    }

    fn after_constant_solve(&mut self, data: &DefaultProblemData<T>) {
        data.P_mul(&mut self.Px2, &self.x2);
        self.Pξ_valid = false;
    }

    pub(crate) fn update_P(&mut self, P: &CscMatrix<T>) {
        self.kktsolver.update_P(P);
    }

    pub(crate) fn update_A(&mut self, A: &CscMatrix<T>) {
        self.kktsolver.update_A(A);
    }
}
