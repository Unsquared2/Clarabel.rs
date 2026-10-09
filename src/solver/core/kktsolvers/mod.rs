#![allow(non_snake_case)]
use super::{cones::CompositeCone, CoreSettings};
use crate::algebra::*;

pub mod direct;

pub trait KKTSolver<T: FloatT>: HasLinearSolverInfo {
    fn update(&mut self, cones: &CompositeCone<T>, settings: &CoreSettings<T>) -> bool;
    fn setrhs(&mut self, x: &[T], z: &[T]);
    fn solve(
        &mut self,
        x: Option<&mut [T]>,
        z: Option<&mut [T]>,
        settings: &CoreSettings<T>,
    ) -> bool;
    fn update_P(&mut self, P: &CscMatrix<T>);
    fn update_A(&mut self, A: &CscMatrix<T>);

    /// Solve for two right hand sides with the current factorization.
    /// Solvers able to share work between them may override this.
    fn solve_pair(
        &mut self,
        rhs: [(&[T], &[T]); 2],
        lhs: [(&mut [T], &mut [T]); 2],
        settings: &CoreSettings<T>,
    ) -> bool {
        let mut ok = true;
        for ((rx, rz), (lx, lz)) in rhs.into_iter().zip(lhs) {
            self.setrhs(rx, rz);
            ok = ok && self.solve(Some(lx), Some(lz), settings);
        }
        ok
    }
}

pub trait HasLinearSolverInfo {
    fn linear_solver_info(&self) -> LinearSolverInfo;
}
#[repr(C)]
#[derive(Debug, Default, Clone)]
/// Linear subsolver information.
///
pub struct LinearSolverInfo {
    /// Name of the linear solver that was used
    pub name: String,
    /// Number of threads used by solver
    pub threads: usize,
    /// Whether the solver used a direct factorisation method
    pub direct: bool,
    /// Number of nonzeros in the linear system
    pub nnzA: usize, // nnz in A for A = LDL^T
    /// Number of nonzeros in the factored system
    pub nnzL: usize, // nnz in L for A = LDL^T
}
