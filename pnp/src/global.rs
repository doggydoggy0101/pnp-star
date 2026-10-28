use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use nalgebra::linalg::ColPivQR;
use nalgebra::{DMatrix, Dyn, Matrix2, Matrix3, Vector2, Vector3};

use crate::linalg::{project_to_so3, row3, skew, vec2rot};
use crate::nonlinear::{refine_pinhole, refine_spherical, LmOptions};
use crate::sdp::{
    dense_to_lower, solve_sdp, solve_star, Clique, Constraint, SdpProblem, StarSdp, SymTriplets,
    RANK_TOL,
};
use crate::utils::{
    col_orthogonal_constraint_matrices, det_constraint_matrices, normalize_pinhole,
    row_orthogonal_constraint_matrices,
};

/// Which SDP formulation to solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Formulation {
    /// (SDP): one PSD block of side 13 + 24N pinhole / 13 + 36N spherical.
    Dense,
    /// (Chordal-SDP): N clique blocks of side 37 / 49.
    Chordal,
    /// (Reduced-SDP): quotient-ring reduction, N blocks of side 34 / 46.
    #[default]
    Reduced,
}

/// Smallest `h` we will divide by; `vec_p` is unit, so `1/h` scales all of it.
pub(crate) const H_MIN: f64 = 1e-9;

/// Smallest camera-frame depth we will divide by when forming the objective.
const DEPTH_MIN: f64 = 1e-9;

pub struct GlobalSolverResult {
    pub rotation: Matrix3<f64>,
    pub translation: Vector3<f64>,
    pub sdp_value: f64,
    pub qcqp_value: f64,
    pub optimality_gap: f64,
    pub rel_optimality_gap: f64,
    pub rank: usize,
    pub eigval_ratio: f64,
    /// False when `|h| < H_MIN`, so no pose could be dehomogenized; pose is NaN.
    /// Only reachable at rank > 1, where the eigenvector is not a lifted pose.
    pub pose_recovered: bool,
    /// `rank == 1` and `e3^T P_i z != 0` for all i: certified global optimum of
    /// Problem 1. ! rank 1 on its own certifies the QCQP, not Problem 1.
    pub certified: bool,
    pub runtime: f64,
}

/// No usable pose: keep the rank and the bound, return NaN for the pose.
fn no_pose(sdp_value: f64, rank: usize, eigval_ratio: f64, t0: Instant) -> GlobalSolverResult {
    GlobalSolverResult {
        rotation: Matrix3::from_element(f64::NAN),
        translation: Vector3::from_element(f64::NAN),
        sdp_value,
        qcqp_value: f64::NAN,
        optimality_gap: f64::NAN,
        rel_optimality_gap: f64::NAN,
        rank,
        eigval_ratio,
        pose_recovered: false,
        certified: false,
        runtime: t0.elapsed().as_secs_f64(),
    }
}

fn m3_to_dm(m: &Matrix3<f64>) -> DMatrix<f64> {
    DMatrix::from_fn(3, 3, |i, j| m[(i, j)])
}

/// Quotient-ring reduction of one constraint (or objective) matrix: Ξ^T C Ξ, computed
/// from the nonzeros of C as Σ C[r,k] ξ_r ξ_k^T over the rows ξ_r of Ξ, given sparse. Ξ
/// is the identity on the hub and kept coordinates, so most rows are unit vectors: a
/// few entries per constraint instead of two dense products. Returns the lower triangle.
fn quotient_ring_reduce(xi_rows: &[Vec<(usize, f64)>], c: &DMatrix<f64>) -> SymTriplets {
    let mut acc: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    for k in 0..c.ncols() {
        for r in 0..c.nrows() {
            let v = c[(r, k)];
            if v == 0.0 {
                continue;
            }
            for &(a, ta) in &xi_rows[r] {
                for &(b, tb) in &xi_rows[k] {
                    *acc.entry((a, b)).or_insert(0.0) += v * ta * tb;
                }
            }
        }
    }
    acc.into_iter()
        .filter(|&((a, b), v)| a >= b && v != 0.0)
        .map(|((a, b), v)| (a, b, v))
        .collect()
}

/// Re-index a clique-local symmetric matrix into the single dense block.
/// `map[local] = dense`. Used only by `Formulation::Dense`.
fn coeff_at(m: &DMatrix<f64>, map: &[usize]) -> crate::sdp::SparseCoeff {
    let mut out = Vec::new();
    for c in 0..m.ncols() {
        for r in 0..m.nrows() {
            let v = m[(r, c)];
            if v != 0.0 {
                out.push((map[r], map[c], v));
            }
        }
    }
    out
}

/// Column permutation order from a ColPivQR: `perm[i]` = original column index
/// at pivot position `i` (strongest pivots first).
fn col_perm_indices(qr: &ColPivQR<f64, Dyn, Dyn>, ncols: usize) -> Vec<usize> {
    let mut labels = DMatrix::<f64>::from_fn(1, ncols, |_, j| j as f64);
    qr.p().permute_columns(&mut labels);
    (0..ncols)
        .map(|j| labels[(0, j)].round() as usize)
        .collect()
}

// ConstraintPool: assembles and prunes trace constraints of the form trace(C * X) = 0
struct ConstraintPool {
    dim_i: usize,
    pool: Vec<DMatrix<f64>>,
}

impl ConstraintPool {
    fn new(dim_i: usize) -> Self {
        ConstraintPool {
            dim_i,
            pool: Vec::new(),
        }
    }

    fn add_trace_constraint(&mut self, mat_c: DMatrix<f64>) {
        let sym = (&mat_c + mat_c.transpose()) * 0.5;
        self.pool.push(sym);
    }

    fn add_block_symmetry(&mut self, r_start: usize, r_end: usize, c_start: usize) {
        let dim_b = r_end - r_start;
        for br in 0..dim_b {
            for bc in (br + 1)..dim_b {
                let mut mat_c = DMatrix::zeros(self.dim_i, self.dim_i);
                mat_c[(r_start + br, c_start + bc)] = 1.0;
                mat_c[(r_start + bc, c_start + br)] = -1.0;
                self.add_trace_constraint(mat_c);
            }
        }
    }

    /// Only used when quotient-ring reduction is disabled. Lifting is mandatory
    /// (the unlifted relaxation is non-informative), so the model row is always
    /// multiplied by every basis index, dropping one dependent constraint.
    fn add_model_constraint(&mut self, mat_a: &DMatrix<f64>) {
        let (mut drop_r, mut drop_c) = (usize::MAX, usize::MAX);
        for c in (0..mat_a.ncols()).rev() {
            if mat_a[(0, c)] != 0.0 {
                drop_r = 1;
                drop_c = c;
                break;
            }
        }
        for r in 0..mat_a.nrows() {
            for c in 0..self.dim_i {
                if r == drop_r && c == drop_c {
                    continue;
                }
                let mut mat_c = DMatrix::zeros(self.dim_i, self.dim_i);
                for k in 0..mat_a.ncols() {
                    mat_c[(c, k)] = mat_a[(r, k)];
                }
                self.add_trace_constraint(mat_c);
            }
        }
    }

    /// Spherical unit-sphere constraint: ||y_i||^2 - 2 (f ⊗ z)^T y_i = 0, using
    /// ||v||^2 = ||y||^2 and f^T v = (f ⊗ z)^T y under ||z||^2 = 1.
    fn add_unit_sphere_constraint(
        &mut self,
        f: &Vector3<f64>,
        dim_hub: usize,
        dim_v: usize,
        dim_spoke: usize,
    ) {
        let mut mat_c = DMatrix::zeros(self.dim_i, self.dim_i);
        for j in 0..dim_v {
            let spoke_start = dim_hub + j * dim_spoke;
            for c in 0..dim_spoke {
                mat_c[(spoke_start + c, spoke_start + c)] += 1.0; // ||y||^2
                mat_c[(1 + c, spoke_start + c)] -= 2.0 * f[j]; // -2 (f ⊗ z)^T y
            }
        }
        self.add_trace_constraint(mat_c);
    }

    /// QR-based pruning: linearly independent subset (preserving order).
    fn prune(&self, tol: f64) -> Vec<DMatrix<f64>> {
        if self.pool.is_empty() {
            return vec![];
        }
        let d = self.dim_i;
        let n_con = self.pool.len();
        let n_triu = d * (d + 1) / 2;

        let mut mat_m = DMatrix::zeros(n_triu, n_con);
        let mut row = 0;
        for r in 0..d {
            for c in r..d {
                let s = if r == c { 1.0 } else { 2.0 };
                for (j, p) in self.pool.iter().enumerate() {
                    mat_m[(row, j)] = s * p[(r, c)];
                }
                row += 1;
            }
        }

        let qr = mat_m.col_piv_qr();
        let rmat = qr.r();
        let mn = rmat.nrows().min(rmat.ncols());
        let max_diag = (0..mn).map(|i| rmat[(i, i)].abs()).fold(0.0, f64::max);
        let thresh = tol * max_diag; // premultiplied threshold
        let rank = (0..mn).filter(|&i| rmat[(i, i)].abs() > thresh).count();

        let perm = col_perm_indices(&qr, n_con);
        let mut idx: Vec<usize> = perm[0..rank].to_vec();
        idx.sort_unstable();
        idx.into_iter().map(|k| self.pool[k].clone()).collect()
    }
}

// Quotient-ring reduction
/// Build the quotient-ring transform T (dim_full × dim_reduce) s.t. mat_A·T = 0.
pub fn build_quotient_ring_transform(mat_a: &DMatrix<f64>, dim_hub: usize) -> DMatrix<f64> {
    let num_rows = mat_a.nrows();
    let dim_full = mat_a.ncols();
    let dim_reduce = dim_full - num_rows;

    let mut mat_t = DMatrix::zeros(dim_full, dim_reduce);
    mat_t
        .view_mut((0, 0), (dim_hub, dim_hub))
        .copy_from(&DMatrix::identity(dim_hub, dim_hub));

    let col_hub = mat_a.columns(0, dim_hub).into_owned(); // (num_rows, dim_hub)
    let n_aux = dim_full - dim_hub;
    let col_aux = mat_a.columns(dim_hub, n_aux).into_owned(); // (num_rows, n_aux)

    let qr = col_aux.clone().col_piv_qr();
    let perm = col_perm_indices(&qr, n_aux);

    let drop_idx: Vec<usize> = (0..num_rows).map(|r| dim_hub + perm[r]).collect();

    let mut mat_b = DMatrix::zeros(num_rows, num_rows);
    for r in 0..num_rows {
        mat_b.set_column(r, &col_aux.column(perm[r]));
    }
    let mat_b_inv = mat_b.try_inverse().expect("quotient ring: B singular");

    let drop_set: BTreeSet<usize> = drop_idx.iter().copied().collect();
    let keep_idx: Vec<usize> = (dim_hub..dim_full)
        .filter(|k| !drop_set.contains(k))
        .collect();

    for (k, &kidx) in keep_idx.iter().enumerate() {
        mat_t[(kidx, dim_hub + k)] = 1.0;
    }

    let mut mat_c_keep = DMatrix::zeros(num_rows, keep_idx.len());
    for (k, &kidx) in keep_idx.iter().enumerate() {
        mat_c_keep.set_column(k, &mat_a.column(kidx));
    }

    let tmp1 = -&mat_b_inv * &col_hub; // (num_rows, dim_hub)
    let tmp2 = -&mat_b_inv * &mat_c_keep; // (num_rows, keep)
    for r in 0..num_rows {
        mat_t
            .view_mut((drop_idx[r], 0), (1, dim_hub))
            .copy_from(&tmp1.row(r));
        mat_t
            .view_mut((drop_idx[r], dim_hub), (1, keep_idx.len()))
            .copy_from(&tmp2.row(r));
    }

    mat_t
}

// GlobalSolver
pub struct GlobalSolver;

impl GlobalSolver {
    /// `intrinsic` is required for pinhole, ignored for spherical (pass `None`).
    /// `refine = false` returns the raw SDP rounding (skips the LM applied when the
    /// relaxation is not rank-1); identical to `true` when rank-1.
    pub fn solve<'a>(
        &self,
        world_points: &DMatrix<f64>,
        observations: &DMatrix<f64>,
        intrinsic: impl Into<Option<&'a Matrix3<f64>>>,
        formulation: Formulation,
        prune_constraint: bool,
        refine: bool,
    ) -> GlobalSolverResult {
        let intrinsic = intrinsic.into().copied().unwrap_or_else(Matrix3::identity);
        let t0 = Instant::now();

        let n = world_points.nrows();
        // recentre on the centroid, then rescale by the radius. Change of world
        // frame, so R is unchanged and t = scale·t_raw − R·centroid on the way out.
        let mut centroid = Vector3::zeros();
        for i in 0..n {
            centroid += row3(world_points, i);
        }
        centroid /= n as f64;
        let mut scale = 0.0f64;
        for i in 0..n {
            scale = scale.max((row3(world_points, i) - centroid).norm());
        }
        // h = 0
        if scale < H_MIN {
            return no_pose(f64::NAN, 0, f64::NAN, t0);
        }
        // centred + scaled world coords (N, 3)
        let p = DMatrix::from_fn(n, 3, |i, j| (world_points[(i, j)] - centroid[j]) / scale);

        let is_pinhole = observations.ncols() == 2;
        let dim_v = if is_pinhole { 2 } else { 3 };

        // normalized image points (N, 2) + intrinsic weight; W = I2 when fx = fy
        let (p_2d, w_int) = if is_pinhole {
            normalize_pinhole(observations, &intrinsic)
        } else {
            (DMatrix::zeros(n, 2), Matrix2::identity())
        };

        let dim_hub = 13usize;
        let dim_spoke = 12usize; // spoke y_j = v_j ⊗ z, no v_j*h column
        let spoke_total = dim_v * dim_spoke; // 24 pinhole / 36 spherical
        let dim_full = dim_hub + spoke_total; // clique side: 37 / 49
        let is_dense = formulation == Formulation::Dense;
        let is_reduced = formulation == Formulation::Reduced;
        let dim_i = if is_reduced { dim_full - 3 } else { dim_full };

        let e = [
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        ];

        // rotation constraints (row + col + det) on the 13×13 hub block
        let mut rot_con = row_orthogonal_constraint_matrices(dim_hub);
        rot_con.extend(col_orthogonal_constraint_matrices(dim_hub));
        rot_con.extend(det_constraint_matrices(dim_hub));

        // objective selector: 1 on every spoke diagonal entry (= ||y||^2)
        let mut diag_spoke = DMatrix::zeros(dim_full, dim_full);
        for k in dim_hub..dim_full {
            diag_spoke[(k, k)] = 1.0;
        }

        // hub constraints: trace(X_hub[1:,1:]) = 1 normalizes the direction, then rotation;
        // hub indices agree in all formulations
        let mut hub_constraints = vec![Constraint {
            matrix: (1..dim_hub).map(|k| (k, k, 1.0)).collect(),
            rhs: 1.0,
        }];
        for c_rot in &rot_con {
            hub_constraints.push(Constraint {
                matrix: dense_to_lower(c_rot),
                rhs: 0.0,
            });
        }

        // Dense: one MOSEK block holding every spoke, one objective over their diagonals,
        // trace(X_yy), and the hub constraints on its leading block
        let mut prob = SdpProblem::default();
        if is_dense {
            prob.block_dims = vec![dim_hub + n * spoke_total];
            let obj = (dim_hub..dim_hub + n * spoke_total)
                .map(|k| (k, k, 1.0))
                .collect();
            prob.objective.push((0, obj));
            for c in &hub_constraints {
                prob.eq_lhs.push(vec![(0, c.matrix.clone())]);
                prob.eq_rhs.push(c.rhs);
            }
        }
        // Chordal/Reduced: one clique per point, sharing the hub, solved by SDP*
        let mut cliques = Vec::with_capacity(n);

        for i in 0..n {
            // clique-local index -> dense-block index
            let clique_map: Vec<usize> = (0..dim_hub)
                .chain((0..spoke_total).map(|k| dim_hub + i * spoke_total + k))
                .collect();

            // mat_P (3, dim_hub): column-major kron(p[i], I3) + translation columns
            let mut mat_p = DMatrix::zeros(3, dim_hub);
            for j in 0..3 {
                for k in 0..3 {
                    mat_p[(k, 1 + j * 3 + k)] = p[(i, j)];
                }
            }
            for k in 0..3 {
                mat_p[(k, 10 + k)] = 1.0;
            }

            // mat_A (3, dim_full): model constraint; spoke columns index y_j (12) only
            let mut mat_a = DMatrix::zeros(3, dim_full);
            if is_pinhole {
                let (xi, yi) = (p_2d[(i, 0)], p_2d[(i, 1)]);
                let p2 = mat_p.row(2).into_owned();
                // Gamma rows, then weighted by W: the MLE is the pixel error (paper eq. 7)
                let g0 = xi * &p2 - mat_p.row(0); // 1×13, [0] = 0
                let g1 = yi * &p2 - mat_p.row(1);
                let row0_head = w_int[(0, 0)] * &g0 + w_int[(0, 1)] * &g1;
                let row1_head = w_int[(1, 0)] * &g0 + w_int[(1, 1)] * &g1;
                let (s1, s2) = (dim_hub, dim_hub + dim_spoke);
                // spoke coupling drops the (zero) h column: keep z-part [1..]
                let p2_z = p2.columns(1, dim_spoke).into_owned();
                let row0_z = row0_head.columns(1, dim_spoke).into_owned();
                let row1_z = row1_head.columns(1, dim_spoke).into_owned();
                mat_a.view_mut((0, 0), (1, dim_hub)).copy_from(&row0_head);
                mat_a.view_mut((1, 0), (1, dim_hub)).copy_from(&row1_head);
                mat_a.view_mut((0, s1), (1, dim_spoke)).copy_from(&(-&p2_z));
                mat_a.view_mut((1, s2), (1, dim_spoke)).copy_from(&(-&p2_z));
                // syzygy row 'v1*G2 - v2*G1 = 0' (drop zero h column)
                mat_a
                    .view_mut((2, s1), (1, dim_spoke))
                    .copy_from(&(-&row1_z));
                mat_a.view_mut((2, s2), (1, dim_spoke)).copy_from(&row0_z);
            } else {
                let fi = row3(observations, i);
                mat_a
                    .view_mut((0, 0), (3, dim_hub))
                    .copy_from(&(m3_to_dm(&skew(&fi)) * &mat_p));
                for (j, ej) in e.iter().enumerate() {
                    let block = m3_to_dm(&(-skew(ej))) * &mat_p; // (3, 13)
                    mat_a
                        .view_mut((0, dim_hub + j * dim_spoke), (3, dim_spoke))
                        .copy_from(&block.columns(1, dim_spoke));
                }
            }

            // objective (per clique; Dense pushed its single form above)
            // Reduced: the rows of Ξ_i, sparse
            let t_i: Option<Vec<Vec<(usize, f64)>>> = is_reduced.then(|| {
                let t = build_quotient_ring_transform(&mat_a, dim_hub);
                (0..t.nrows())
                    .map(|r| {
                        (0..t.ncols())
                            .filter(|&j| t[(r, j)] != 0.0)
                            .map(|j| (j, t[(r, j)]))
                            .collect()
                    })
                    .collect()
            });
            if !is_dense {
                cliques.push(Clique {
                    dim: dim_i,
                    objective: match &t_i {
                        Some(t) => quotient_ring_reduce(t, &diag_spoke),
                        None => dense_to_lower(&diag_spoke),
                    },
                    constraints: vec![],
                });
            }

            // constraint pool in full space, then reduce to dim_i
            let mut pool = ConstraintPool::new(dim_full);

            // Reduced folds the model rows into Ξ_i; Dense/Chordal impose them:
            // pinhole's 2 (row 2 is their syzygy), spherical's 3 (independent).
            if !is_reduced {
                let rows = if is_pinhole { 2 } else { 3 };
                pool.add_model_constraint(&mat_a.rows(0, rows).into_owned());
            }

            if !is_pinhole {
                let fi = row3(observations, i);
                pool.add_unit_sphere_constraint(&fi, dim_hub, dim_v, dim_spoke);
            }

            // block symmetry: z-rows (1..dim_hub) × spoke, then spoke × spoke (12×12)
            for j in 0..dim_v {
                pool.add_block_symmetry(1, dim_hub, dim_hub + j * dim_spoke);
            }
            for j in 0..dim_v {
                for k in 0..j {
                    pool.add_block_symmetry(
                        dim_hub + j * dim_spoke,
                        dim_hub + (j + 1) * dim_spoke,
                        dim_hub + k * dim_spoke,
                    );
                }
            }

            // Reduced's pool is independent by construction (disjoint entry pairs)
            let final_con = if prune_constraint && !is_reduced {
                pool.prune(1e-8)
            } else {
                pool.pool.clone()
            };

            if is_dense {
                // re-index into the single block
                for c in &final_con {
                    prob.eq_lhs.push(vec![(0, coeff_at(c, &clique_map))]);
                    prob.eq_rhs.push(0.0);
                }
            } else {
                for c in &final_con {
                    let matrix = match &t_i {
                        // Reduced: Ξ_i^T C Ξ_i; Chordal: C
                        Some(t) => quotient_ring_reduce(t, c),
                        None => dense_to_lower(c),
                    };
                    cliques[i].constraints.push(Constraint { matrix, rhs: 0.0 });
                }
            }
        }

        // MOSEK panics unless optimal, so a returned Dense solution has converged
        let (blocks, sdp_value, converged) = if is_dense {
            let sol = solve_sdp(&prob);
            (sol.blocks, sol.objective_value, true)
        } else {
            let sol = solve_star(&StarSdp {
                hub: dim_hub,
                cliques,
                hub_constraints,
            });
            (sol.blocks, sol.objective_value, sol.converged)
        };

        let x_hub = blocks[0].view((0, 0), (dim_hub, dim_hub)).into_owned();

        let se = x_hub.symmetric_eigen();
        let evals = &se.eigenvalues;
        let mut order: Vec<usize> = (0..dim_hub).collect();
        order.sort_by(|&a, &b| evals[a].partial_cmp(&evals[b]).unwrap()); // ascending
        let rank = evals.iter().filter(|&&v| v > RANK_TOL).count();
        let lambda_max = evals[order[dim_hub - 1]];
        let lambda_2 = evals[order[dim_hub - 2]];
        let eigval_ratio = if lambda_2 > 1e-10 {
            lambda_max / lambda_2
        } else {
            f64::INFINITY
        };

        let mut vec_p = se.eigenvectors.column(order[dim_hub - 1]).into_owned();
        if vec_p[0].abs() < H_MIN {
            // no pose to read off; report rank and the bound, not a garbage pose
            return no_pose(sdp_value, rank, eigval_ratio, t0);
        }
        vec_p /= vec_p[0];

        let mut rotation = project_to_so3(&vec2rot(&vec_p.rows(1, 9).into_owned()));
        let mut t_cw_raw = Vector3::new(vec_p[10], vec_p[11], vec_p[12]);
        let mut translation = t_cw_raw * scale - rotation * centroid;

        // camera-frame points at the recovered pose, reused for the objective below
        let mut p_cam: Vec<Vector3<f64>> =
            (0..n).map(|i| rotation * row3(&p, i) + t_cw_raw).collect();
        // for extraction, pinhole only
        let depth_ok = !is_pinhole || p_cam.iter().all(|q| q[2].abs() >= DEPTH_MIN);
        // ! rank 1 certifies only an optimal SDP solution: the solver must have converged
        let certified = converged && rank == 1 && depth_ok;

        if !certified && refine {
            let opts = LmOptions::default();
            if is_pinhole {
                refine_pinhole(
                    &mut rotation,
                    &mut translation,
                    world_points,
                    observations,
                    &intrinsic,
                    &opts,
                );
            } else {
                refine_spherical(
                    &mut rotation,
                    &mut translation,
                    world_points,
                    observations,
                    &opts,
                );
            }
            t_cw_raw = (translation + rotation * centroid) / scale;
            p_cam = (0..n).map(|i| rotation * row3(&p, i) + t_cw_raw).collect();
        }

        // qcqp value
        let mut qcqp_value = 0.0;
        for nn in 0..n {
            let p_c = p_cam[nn];
            if is_pinhole {
                let d = w_int
                    * Vector2::new(
                        p_c[0] / p_c[2] - p_2d[(nn, 0)],
                        p_c[1] / p_c[2] - p_2d[(nn, 1)],
                    );
                qcqp_value += d.norm_squared();
            } else {
                let u = p_c.normalize();
                qcqp_value += (row3(observations, nn) - u).norm_squared();
            }
        }

        let optimality_gap = (qcqp_value - sdp_value).abs();
        let rel_optimality_gap = optimality_gap / qcqp_value.abs();

        GlobalSolverResult {
            rotation,
            translation,
            sdp_value,
            qcqp_value,
            optimality_gap,
            rel_optimality_gap,
            rank,
            eigval_ratio,
            pose_recovered: true,
            certified,
            runtime: t0.elapsed().as_secs_f64(),
        }
    }
}
