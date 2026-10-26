//! Shor's relaxation with the alternative errors:
//!
//! - Weighted reprojection error (pinhole)
//!     Sun, L., & Deng, Z. (2020).
//!     Certifiably optimal and robust camera pose estimation from points and lines.
//!     IEEE Access, 8, 124032-124054.
//!
//! - Point-line error (spherical)
//!     Agostinho, S., Gomes, J., & Del Bue, A. (2023).
//!     CvxPnPL: A unified convex solution to the absolute pose estimation problem
//!     from point and line correspondences.
//!     Journal of Mathematical Imaging and Vision, 65(3), 492-512.

use std::time::Instant;

use nalgebra::{DMatrix, DVector, Matrix3, Vector3};

use crate::global::H_MIN;
use crate::linalg::{project_to_so3, rot2vec, row3, vec2rot};
use crate::sdp::{dense_to_coeff, solve_sdp, SdpProblem, RANK_TOL};
use crate::utils::{
    col_orthogonal_constraint_matrices, det_constraint_matrices, lift,
    marginalize as marginalize_cost, plerr_data_matrices, row_orthogonal_constraint_matrices,
    wrerr_data_matrices, DataMatrices,
};

pub struct AlternativeSolverResult {
    pub rotation: Matrix3<f64>,
    pub translation: Vector3<f64>,
    pub sdp_value: f64,
    pub qcqp_value: f64,
    pub optimality_gap: f64,
    pub rel_optimality_gap: f64,
    pub rank: usize,
    pub eigval_ratio: f64,
    /// False when `|h| < H_MIN`, so no pose could be dehomogenized; pose is NaN.
    pub pose_recovered: bool,
    pub runtime: f64,
}

pub struct AlternativeSolver;

impl AlternativeSolver {
    /// `intrinsic` is required for pinhole, ignored for spherical (pass `None`).
    pub fn solve<'a>(
        &self,
        world_points: &DMatrix<f64>,
        observations: &DMatrix<f64>,
        intrinsic: impl Into<Option<&'a Matrix3<f64>>>,
        marginalize: bool,
    ) -> AlternativeSolverResult {
        let intrinsic = intrinsic.into().copied().unwrap_or_else(Matrix3::identity);
        let t0 = Instant::now();

        // recentre on the centroid, then rescale by the radius. Change of world
        // frame, so R is unchanged and t = scale·t_raw − R·centroid on the way out.
        // The surrogate cost scales as 1/scale², undone on the reported values.
        let n_pts = world_points.nrows();
        let mut centroid = Vector3::zeros();
        for i in 0..n_pts {
            centroid += row3(world_points, i);
        }
        centroid /= n_pts as f64;
        let mut scale = 0.0f64;
        for i in 0..n_pts {
            scale = scale.max((row3(world_points, i) - centroid).norm());
        }
        // h = 0
        if scale < H_MIN {
            return AlternativeSolverResult {
                rotation: Matrix3::from_element(f64::NAN),
                translation: Vector3::from_element(f64::NAN),
                sdp_value: f64::NAN,
                qcqp_value: f64::NAN,
                optimality_gap: f64::NAN,
                rel_optimality_gap: f64::NAN,
                rank: 0,
                eigval_ratio: f64::NAN,
                pose_recovered: false,
                runtime: t0.elapsed().as_secs_f64(),
            };
        }
        let p = DMatrix::from_fn(n_pts, 3, |i, j| {
            (world_points[(i, j)] - centroid[j]) / scale
        });

        // detect model from observation shape: (N,3) spherical, (N,2) pinhole
        let dm: DataMatrices = if observations.ncols() == 3 {
            plerr_data_matrices(&p, observations)
        } else {
            wrerr_data_matrices(&p, observations, &intrinsic)
        };

        let (dim, mat_c) = if marginalize {
            (10usize, marginalize_cost(&dm))
        } else {
            (13usize, lift(&dm))
        };

        // min <C, X>  s.t.  X >= 0, X[0,0]=1, rotation constraints
        let mut prob = SdpProblem {
            block_dims: vec![dim],
            objective: vec![(0, dense_to_coeff(&mat_c))],
            eq_lhs: vec![vec![(0, vec![(0, 0, 1.0)])]],
            eq_rhs: vec![1.0],
        };
        for cons in [
            row_orthogonal_constraint_matrices(dim),
            col_orthogonal_constraint_matrices(dim),
            det_constraint_matrices(dim),
        ] {
            for c in cons {
                prob.eq_lhs.push(vec![(0, dense_to_coeff(&c))]);
                prob.eq_rhs.push(0.0);
            }
        }

        let sol = solve_sdp(&prob);
        let sdp_value = sol.objective_value;
        let mat_x = &sol.blocks[0];

        let se = mat_x.clone().symmetric_eigen();
        let evals = &se.eigenvalues;
        let mut order: Vec<usize> = (0..dim).collect();
        order.sort_by(|&a, &b| evals[a].partial_cmp(&evals[b]).unwrap()); // ascending
        let rank = evals.iter().filter(|&&e| e > RANK_TOL).count();
        let lambda_max = evals[order[dim - 1]];
        let lambda_2 = evals[order[dim - 2]];
        let eigval_ratio = if lambda_2 > 1e-10 {
            lambda_max / lambda_2
        } else {
            f64::INFINITY
        };

        let mut vec_temp = se.eigenvectors.column(order[dim - 1]).into_owned();
        if vec_temp[0].abs() < H_MIN {
            return AlternativeSolverResult {
                rotation: Matrix3::from_element(f64::NAN),
                translation: Vector3::from_element(f64::NAN),
                sdp_value: sdp_value * scale * scale,
                qcqp_value: f64::NAN,
                optimality_gap: f64::NAN,
                rel_optimality_gap: f64::NAN,
                rank,
                eigval_ratio,
                pose_recovered: false,
                runtime: t0.elapsed().as_secs_f64(),
            };
        }
        vec_temp /= vec_temp[0];

        let rotation = project_to_so3(&vec2rot(&vec_temp.rows(1, 9).into_owned()));
        let vec_r = rot2vec(&rotation);

        let t: DVector<f64> = if marginalize {
            -dm.ct
                .clone()
                .full_piv_lu()
                .solve(&(&dm.ctr * &vec_r))
                .expect("translation recovery singular")
        } else {
            vec_temp.rows(10, 3).into_owned()
        };

        let qcqp_value = (&dm.cr * &vec_r).dot(&vec_r)
            + 2.0 * (&dm.ctr * &vec_r).dot(&t)
            + (&dm.ct * &t).dot(&t)
            + dm.c;

        let optimality_gap = (qcqp_value - sdp_value).abs();
        let rel_optimality_gap = optimality_gap / qcqp_value.abs();

        // back to world frame
        let t_world = Vector3::new(t[0], t[1], t[2]) * scale - rotation * centroid;
        let s2 = scale * scale;
        AlternativeSolverResult {
            rotation,
            translation: t_world,
            sdp_value: sdp_value * s2,
            qcqp_value: qcqp_value * s2,
            optimality_gap: optimality_gap * s2,
            rel_optimality_gap,
            rank,
            eigval_ratio,
            pose_recovered: true,
            runtime: t0.elapsed().as_secs_f64(),
        }
    }
}
