use nalgebra::DMatrix;

/// Numerical rank: eigenvalues above this count, in the normalized frame.
pub const RANK_TOL: f64 = 1e-5;

/// One symmetric coefficient matrix as (row, col, value) triplets.
pub type SparseCoeff = Vec<(usize, usize, f64)>;
/// Linear functional over multiple PSD blocks: list of (block_idx, coeff).
pub type BlockLinearForm = Vec<(usize, SparseCoeff)>;

#[derive(Default)]
pub struct SdpProblem {
    pub block_dims: Vec<usize>,
    pub objective: BlockLinearForm,
    pub eq_lhs: Vec<BlockLinearForm>,
    pub eq_rhs: Vec<f64>,
}

pub struct SdpSolution {
    pub blocks: Vec<DMatrix<f64>>, // X_i (dim × dim per block)
    pub objective_value: f64,
}

/// Solve the SDP with MOSEK (Dense and the alternative solver).
pub fn solve_sdp(prob: &SdpProblem) -> SdpSolution {
    crate::mosek::solve_mosek(prob)
}

/// Flatten a dense symmetric matrix into full-symmetric COO triplets (nonzeros).
pub fn dense_to_coeff(m: &DMatrix<f64>) -> SparseCoeff {
    let mut out = Vec::new();
    for c in 0..m.ncols() {
        for r in 0..m.nrows() {
            let v = m[(r, c)];
            if v != 0.0 {
                out.push((r, c, v));
            }
        }
    }
    out
}

pub use sdp_star::{Clique, Constraint, StarSdp, SymTriplets};

pub struct StarSolution {
    pub blocks: Vec<DMatrix<f64>>, // X_i (dim × dim per clique)
    /// dual objective: a lower bound on the optimum (weak duality)
    pub objective_value: f64,
    /// ! rank 1 certifies only an optimal SDP solution, so certificates require this
    pub converged: bool,
}

/// Solve an SDP on a chordal star graph (cliques overlapping only on a shared hub) with
/// SDP* (`sdp-star`).
pub fn solve_star(prob: &StarSdp) -> StarSolution {
    let sol = sdp_star::solve(prob, &sdp_star::Options::default()).expect("SDP*: invalid problem");
    StarSolution {
        converged: sol.converged(),
        objective_value: sol.dual_objective,
        blocks: sol.x,
    }
}

/// Lower triangle of a dense symmetric matrix as (row, col, value) triplets (nonzeros),
/// the input format of SDP*.
pub fn dense_to_lower(m: &DMatrix<f64>) -> SymTriplets {
    let mut out = Vec::new();
    for c in 0..m.ncols() {
        for r in c..m.nrows() {
            let v = m[(r, c)];
            if v != 0.0 {
                out.push((r, c, v));
            }
        }
    }
    out
}
