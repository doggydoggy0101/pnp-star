//! MOSEK backend for the SDP abstraction, via the official `mosek` crate
//! (Optimizer API). Coefficient matrices are full-symmetric; we pass the lower
//! triangle to `append_sparse_sym_mat` and MOSEK doubles off-diagonals
//! internally, so the symmetric inner product <A, X> comes out right.

use std::collections::HashMap;

use mosek::{Boundkey, Solsta, Soltype, Task};
use nalgebra::DMatrix;

use crate::sdp::{BlockLinearForm, SdpProblem, SdpSolution};

/// Group a BlockLinearForm into per-block lower-triangular (i>=j) triplets.
fn lower_tri_by_block(form: &BlockLinearForm) -> HashMap<usize, (Vec<i32>, Vec<i32>, Vec<f64>)> {
    let mut acc: HashMap<usize, HashMap<(i32, i32), f64>> = HashMap::new();
    for (b, coeff) in form {
        let e = acc.entry(*b).or_default();
        for &(r, c, v) in coeff {
            if r >= c {
                *e.entry((r as i32, c as i32)).or_insert(0.0) += v;
            }
        }
    }
    acc.into_iter()
        .map(|(b, m)| {
            let (mut is, mut js, mut vs) = (Vec::new(), Vec::new(), Vec::new());
            for ((ri, ci), v) in m {
                is.push(ri);
                js.push(ci);
                vs.push(v);
            }
            (b, (is, js, vs))
        })
        .collect()
}

pub fn solve_mosek(prob: &SdpProblem) -> SdpSolution {
    let num_blocks = prob.block_dims.len();
    let dims: Vec<i32> = prob.block_dims.iter().map(|&d| d as i32).collect();
    let m_eq = prob.eq_lhs.len();

    let mut task = Task::new().expect("MOSEK: failed to create task");
    task.append_cons(m_eq as i32).unwrap();
    task.append_barvars(&dims).unwrap();

    // objective: Σ_i <C_i, X_i>
    for (b, (is, js, vs)) in lower_tri_by_block(&prob.objective) {
        let idx = task.append_sparse_sym_mat(dims[b], &is, &js, &vs).unwrap();
        task.put_barc_j(b as i32, &[idx], &[1.0]).unwrap();
    }

    // equality constraints: Σ_i <A_{j,i}, X_i> = b_j
    for (ci, form) in prob.eq_lhs.iter().enumerate() {
        for (b, (is, js, vs)) in lower_tri_by_block(form) {
            let idx = task.append_sparse_sym_mat(dims[b], &is, &js, &vs).unwrap();
            task.put_bara_ij(ci as i32, b as i32, &[idx], &[1.0])
                .unwrap();
        }
        let rhs = prob.eq_rhs[ci];
        task.put_con_bound(ci as i32, Boundkey::FX, rhs, rhs)
            .unwrap();
    }

    task.optimize().expect("MOSEK optimize failed");

    let solsta = task.get_sol_sta(Soltype::ITR).unwrap();
    assert!(
        matches!(solsta, Solsta::OPTIMAL),
        "MOSEK did not reach optimal (status {solsta})"
    );

    let mut blocks = Vec::with_capacity(num_blocks);
    for (i, &d) in prob.block_dims.iter().enumerate() {
        let mut barx = vec![0.0f64; d * (d + 1) / 2];
        task.get_barx_j(Soltype::ITR, i as i32, &mut barx).unwrap();
        // barx is column-major lower-triangular: entry (r,c) with r>=c at
        // index c*n + r - c*(c+1)/2.
        let mut x = DMatrix::zeros(d, d);
        for c in 0..d {
            for r in c..d {
                let val = barx[c * d + r - c * (c + 1) / 2];
                x[(r, c)] = val;
                x[(c, r)] = val;
            }
        }
        blocks.push(x);
    }

    SdpSolution {
        blocks,
        objective_value: task.get_primal_obj(Soltype::ITR).unwrap(),
    }
}
