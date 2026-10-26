//! SDP backend tests: tiny PSD problems driven through the SdpProblem
//! abstraction, checking MOSEK recovers the known optimum.

use pnp::sdp::{solve_sdp, SdpProblem};

// min X[1,1]  s.t.  X[0,0] = 1, X ⪰ 0  →  X = [[1,0],[0,0]], obj = 0.
fn toy_2x2() -> SdpProblem {
    SdpProblem {
        block_dims: vec![2],
        objective: vec![(0, vec![(1, 1, 1.0)])],
        eq_lhs: vec![vec![(0, vec![(0, 0, 1.0)])]],
        eq_rhs: vec![1.0],
    }
}

// min trace(X)  s.t.  X[0,2] = 1, X ⪰ 0  →  trace = 2, X[0,2] = 1, X[1,1] = 0.
fn toy_3x3() -> SdpProblem {
    SdpProblem {
        block_dims: vec![3],
        objective: vec![(0, vec![(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0)])],
        eq_lhs: vec![vec![(0, vec![(0, 2, 0.5), (2, 0, 0.5)])]],
        eq_rhs: vec![1.0],
    }
}

#[test]
fn psd_2x2() {
    let sol = solve_sdp(&toy_2x2());
    assert!((sol.objective_value - 0.0).abs() < 1e-7);
    assert!((sol.blocks[0][(0, 0)] - 1.0).abs() < 1e-7);
    assert!(sol.blocks[0][(1, 1)].abs() < 1e-7);
}

#[test]
fn psd_3x3() {
    let sol = solve_sdp(&toy_3x3());
    assert!((sol.objective_value - 2.0).abs() < 1e-6);
    assert!((sol.blocks[0][(0, 2)] - 1.0).abs() < 1e-6);
    assert!(sol.blocks[0][(1, 1)].abs() < 1e-6);
}
