//! Branch-and-bound over rotation space for the pinhole MLE, after C. Olsson,
//! F. Kahl, M. Oskarsson, "Optimal Estimation of Perspective Camera Pose",
//! ICPR 2006. Baseline for comparison, not a contribution of this crate.
//! Pinhole only: the bound needs the denominator affine in the lifted variable.
//!
//! Search is over (q, τ), q a non-unit quaternion: R = R(q)/‖q‖², t = τ/‖q‖²
//! (their eq. 2). (λq, λ²τ) is the same pose, so the paper fixes τ₃ = 1. With
//! the world frame translated to the reference point, τ₃ is its depth, (τ₁, τ₂)
//! its normalized image point (|·| ≤ γ_max), and ‖q‖² = 1/t₃, so
//! |q_j| ≤ 1/√(t₃^low).
//!
//! ! Both charts τ₃ = ±1 are searched. λ² > 0, so sign(τ₃) belongs to the ray:
//! τ₃ = +1 covers only poses with the reference point in front, and Problem 1
//! asks e₃ᵀPᵢz ≠ 0, not positivity. The paper uses +1 alone.
//!
//! ! Two errors in the paper: eqs. (5)-(8) print all four inequalities
//! inverted, and |q_i| ≤ 1/t₃^low should be 1/√(t₃^low).

use std::collections::BinaryHeap;
use std::time::Instant;

use mosek::{Boundkey, Objsense, Solsta, Soltype, Task};
use nalgebra::{DMatrix, Matrix3, Matrix4, Vector3};

use crate::linalg::row3;
use crate::utils::normalize_pinhole;

pub struct BnbSolverResult {
    pub rotation: Matrix3<f64>,
    pub translation: Vector3<f64>,
    /// Best objective found (upper bound on the global minimum).
    pub cost: f64,
    /// Weakest surviving bound over both charts; `cost - lower_bound` is the
    /// certified gap, normally ≈ `eps · cost`.
    pub lower_bound: f64,
    pub optimality_gap: f64,
    pub rel_optimality_gap: f64,
    pub nodes: usize,
    /// True if the ε-gap was met (rather than hitting `max_nodes`).
    pub converged: bool,
    pub runtime: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct BnbOptions {
    /// The paper's t₃^low, world units: lower bound on the reference point's
    /// depth. `None` uses 1 world unit. ! An assumption, not a derived bound:
    /// the certificate excludes optima with the reference point closer.
    /// Below ~3e-4 · max_i‖X_i‖ the q box (±1/√t₃^low) breaks MOSEK.
    pub min_depth: Option<f64>,
    /// The paper's γ_max, in normalized units: a cap on the reference point's
    /// reprojection error. `None` derives √U₀ / σ_min(W), U₀ the cost of
    /// R = I, t = (0, 0, 1 + max_i‖p_i‖): one residual ≤ f* ≤ U₀, so the box
    /// keeps the optimum.
    pub gamma: Option<f64>,
    /// Relative gap (cost - lower_bound) / cost at which to stop.
    pub eps: f64,
    /// Absolute gap floor. ! Needed as well as `eps`: at N = 3 the system is
    /// square so the optimum is 0 and a relative test can never be met.
    pub eps_abs: f64,
    pub max_nodes: usize,
}

impl Default for BnbOptions {
    fn default() -> Self {
        BnbOptions {
            min_depth: None,
            gamma: None,
            eps: 1e-4,
            eps_abs: 1e-9,
            max_nodes: 10_000,
        }
    }
}

/// The 9 symmetric 4×4 forms with qᵀ M_{kj} q = R(q)[k][j] for unit q
/// (scalar-first quaternion q = (q1, q2, q3, q4)).
fn rot_quadratic_forms() -> [[Matrix4<f64>; 3]; 3] {
    let diag = |a: f64, b: f64, c: f64, d: f64| Matrix4::from_diagonal(&[a, b, c, d].into());
    let pair = |i: usize, j: usize, v: f64| {
        let mut m = Matrix4::zeros();
        m[(i, j)] = v;
        m[(j, i)] = v;
        m
    };
    [
        [
            diag(1.0, 1.0, -1.0, -1.0),
            pair(1, 2, 1.0) + pair(0, 3, -1.0),
            pair(1, 3, 1.0) + pair(0, 2, 1.0),
        ],
        [
            pair(1, 2, 1.0) + pair(0, 3, 1.0),
            diag(1.0, -1.0, 1.0, -1.0),
            pair(2, 3, 1.0) + pair(0, 1, -1.0),
        ],
        [
            pair(1, 3, 1.0) + pair(0, 2, -1.0),
            pair(2, 3, 1.0) + pair(0, 1, 1.0),
            diag(1.0, -1.0, -1.0, 1.0),
        ],
    ]
}

/// Index of s_jk (j ≤ k) among the 10 upper-triangular entries.
fn sidx(j: usize, k: usize) -> usize {
    let (j, k) = if j <= k { (j, k) } else { (k, j) };
    // row-major upper triangle of a 4×4
    [[0, 1, 2, 3], [0, 4, 5, 6], [0, 0, 7, 8], [0, 0, 0, 9]][j][k]
}

/// One residual: numerator and denominator as linear forms over (t, s).
#[derive(Clone)]
struct Residual {
    num_t: [f64; 3],
    num_s: [f64; 10],
    den_t: [f64; 3],
    den_s: [f64; 10],
}

/// Spread a symmetric 4×4 quadratic form onto the 10 s-coefficients:
/// qᵀMq = Σ_{j≤k} c_jk M[j][k] s_jk, c_jk = 1 if j = k else 2.
fn form_to_s(m: &Matrix4<f64>) -> [f64; 10] {
    let mut out = [0.0; 10];
    for j in 0..4 {
        for k in j..4 {
            let c = if j == k { 1.0 } else { 2.0 };
            out[sidx(j, k)] += c * m[(j, k)];
        }
    }
    out
}

/// Interval of a linear form over (t, s) given boxes on t and s.
fn form_range(
    ct: &[f64; 3],
    cs: &[f64; 10],
    tb: &[(f64, f64); 3],
    sb: &[(f64, f64); 10],
) -> (f64, f64) {
    let mut lo = 0.0;
    let mut hi = 0.0;
    for (i, &c) in ct.iter().enumerate() {
        let (a, b) = (c * tb[i].0, c * tb[i].1);
        lo += a.min(b);
        hi += a.max(b);
    }
    for (i, &c) in cs.iter().enumerate() {
        let (a, b) = (c * sb[i].0, c * sb[i].1);
        lo += a.min(b);
        hi += a.max(b);
    }
    (lo, hi)
}

/// s_jk range induced by the q box.
fn s_box(qb: &[(f64, f64); 4]) -> [(f64, f64); 10] {
    let mut out = [(0.0, 0.0); 10];
    for j in 0..4 {
        for k in j..4 {
            let c = [
                qb[j].0 * qb[k].0,
                qb[j].0 * qb[k].1,
                qb[j].1 * qb[k].0,
                qb[j].1 * qb[k].1,
            ];
            let mut lo = c.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = c.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if j == k {
                lo = lo.max(0.0); // q_j² ≥ 0
            }
            out[sidx(j, k)] = (lo, hi);
        }
    }
    out
}

struct Node {
    lower: f64,
    qb: [(f64, f64); 4],
}
impl PartialEq for Node {
    fn eq(&self, o: &Self) -> bool {
        self.lower == o.lower
    }
}
impl Eq for Node {}
impl PartialOrd for Node {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Node {
    // BinaryHeap is a max-heap; invert so the smallest lower bound pops first.
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        o.lower.partial_cmp(&self.lower).unwrap()
    }
}

pub struct BnbSolver;

impl BnbSolver {
    /// Pinhole only, as in the paper: the bearing model's denominator is
    /// ‖Rp+t‖, a square root, and the eq. (9) construction does not apply.
    pub fn solve(
        &self,
        world_points: &DMatrix<f64>,
        image_points: &DMatrix<f64>,
        intrinsic: &Matrix3<f64>,
        opts: &BnbOptions,
    ) -> BnbSolverResult {
        let t0 = Instant::now();
        let n = world_points.nrows();

        // same scaling as GlobalSolver, so the two minimize the identical cost
        let mut scale = 0.0f64;
        for i in 0..n {
            scale = scale.max(row3(world_points, i).norm());
        }
        // reference point at the origin (paper: choose the world frame so X_i = 0)
        let p0 = row3(world_points, 0) / scale;
        let p: Vec<Vector3<f64>> = (0..n).map(|i| row3(world_points, i) / scale - p0).collect();

        // normalized image points + intrinsic weight W (W = I₂ when fx = fy, skew 0)
        let (p_2d, w_int) = normalize_pinhole(image_points, intrinsic);

        let rq = rot_quadratic_forms();
        let e = [
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        ];

        // M_k(X) with qᵀ M_k q = (R(q) X)_k
        let mk = |x: &Vector3<f64>, k: usize| -> Matrix4<f64> {
            (0..3).map(|j| rq[k][j] * x[j]).sum::<Matrix4<f64>>()
        };

        let mut res: Vec<Residual> = Vec::with_capacity(2 * n);
        for i in 0..n {
            let m3 = mk(&p[i], 2);
            let den_s = form_to_s(&m3);
            let den_t = [0.0, 0.0, 1.0]; // e₃ᵀt
            for k in 0..2 {
                // weighted numerator: Σ_j W[k,j] (x^j_i · depth - (RX+t)_j)
                let mut num_m = Matrix4::zeros();
                let mut num_t = Vector3::zeros();
                for j in 0..2 {
                    let wkj = w_int[(k, j)];
                    num_m += (m3 * p_2d[(i, j)] - mk(&p[i], j)) * wkj;
                    num_t += (e[2] * p_2d[(i, j)] - e[j]) * wkj;
                }
                res.push(Residual {
                    num_t: [num_t[0], num_t[1], num_t[2]],
                    num_s: form_to_s(&num_m),
                    den_t,
                    den_s,
                });
            }
        }

        // γ_max = √U₀ / σ_min(W): ‖W r₀‖² ≤ f* ≤ U₀ for the reference residual r₀.
        // U₀ at R = I, t = (0, 0, 1 + ρ), ρ = max_i‖p_i‖: every depth ≥ 1, so finite
        let gamma = opts.gamma.unwrap_or_else(|| {
            let t0 = Vector3::new(0.0, 0.0, 1.0 + scale);
            let mut u0 = 0.0;
            for i in 0..n {
                let pc = row3(world_points, i) + t0;
                let d = w_int
                    * nalgebra::Vector2::new(
                        pc[0] / pc[2] - p_2d[(i, 0)],
                        pc[1] / pc[2] - p_2d[(i, 1)],
                    );
                u0 += d.norm_squared();
            }
            // floor only widens the box
            (u0.sqrt() / w_int.singular_values().min()).max(1e-8)
        });

        // τ₃ = s fixes the scale, so (τ₁, τ₂) = s · (reference point's normalized
        // image point), boxed by ±γ_max. See the header for why both signs.
        let charts: [[(f64, f64); 3]; 2] = [1.0, -1.0].map(|s: f64| {
            [
                (s * p_2d[(0, 0)] - gamma, s * p_2d[(0, 0)] + gamma),
                (s * p_2d[(0, 1)] - gamma, s * p_2d[(0, 1)] + gamma),
                (s, s),
            ]
        });
        // t₃^low in scaled units; ‖q‖² = 1/t₃ ⇒ |q_j| ≤ 1/√(t₃^low)
        let d_lo = opts.min_depth.unwrap_or(1.0) / scale;
        let qmax = 1.0 / d_lo.sqrt();

        // true cost of a candidate (q, τ); R = R(q)/‖q‖², t = τ/‖q‖²
        let cost_at = |q: &[f64; 4], tau: &Vector3<f64>| -> f64 {
            let nq2 = q.iter().map(|v| v * v).sum::<f64>();
            if nq2 < 1e-12 {
                return f64::INFINITY;
            }
            let nq = nq2.sqrt();
            let t = &(tau / nq2);
            let qv = nalgebra::Vector4::new(q[0] / nq, q[1] / nq, q[2] / nq, q[3] / nq);
            let rot = Matrix3::from_fn(|k, j| (qv.transpose() * rq[k][j] * qv)[(0, 0)]);
            let mut f = 0.0;
            for i in 0..n {
                let pc = rot * p[i] + t;
                // Problem 1 asks only e₃ᵀPᵢz ≠ 0, so only zero depth is excluded
                if pc[2].abs() < 1e-12 {
                    return f64::INFINITY;
                }
                let d = w_int
                    * nalgebra::Vector2::new(
                        pc[0] / pc[2] - p_2d[(i, 0)],
                        pc[1] / pc[2] - p_2d[(i, 1)],
                    );
                f += d.norm_squared();
            }
            f
        };

        let root = [(-qmax, qmax); 4];
        let mut best_cost = f64::INFINITY;
        let mut best_q = [qmax, 0.0, 0.0, 0.0];
        let mut best_t = Vector3::new(0.0, 0.0, 1.0);
        let mut nodes = 0usize;
        // certified bound over the union of charts = the weakest of the two
        let mut lower_bound = f64::INFINITY;
        let mut converged = true;

        for tb in charts.iter() {
            let mut heap = BinaryHeap::new();
            heap.push(Node {
                lower: f64::NEG_INFINITY,
                qb: root,
            });
            let mut chart_lower = f64::NEG_INFINITY;
            let mut chart_conv = false;

            while let Some(node) = heap.pop() {
                // best-first, so the popped bound is this chart's lower bound
                chart_lower = node.lower.min(best_cost);
                if nodes >= opts.max_nodes {
                    break;
                }
                // every surviving box is worse than the incumbent ⇒ chart complete
                if node.lower >= best_cost {
                    chart_lower = best_cost;
                    chart_conv = true;
                    break;
                }
                // ! only meaningful once an incumbent exists: with best_cost = +inf
                // the gap is inf and any relative test would fire immediately
                if best_cost.is_finite() {
                    let gap = best_cost - node.lower;
                    if gap <= opts.eps * best_cost.abs() + opts.eps_abs {
                        chart_conv = true;
                        break;
                    }
                }
                nodes += 1;

                if let Some((lb, q_rel, t_rel)) = solve_node(&res, &node.qb, tb) {
                    if lb > best_cost {
                        continue; // pruned
                    }
                    let c = cost_at(&q_rel, &t_rel);
                    if c < best_cost {
                        best_cost = c;
                        best_q = q_rel;
                        best_t = t_rel;
                    }
                    // branch on the widest q dimension
                    let (wi, _) = node
                        .qb
                        .iter()
                        .enumerate()
                        .map(|(i, b)| (i, b.1 - b.0))
                        .fold((0, -1.0), |a, b| if b.1 > a.1 { b } else { a });
                    let mid = 0.5 * (node.qb[wi].0 + node.qb[wi].1);
                    for half in 0..2 {
                        let mut qb = node.qb;
                        if half == 0 {
                            qb[wi].1 = mid;
                        } else {
                            qb[wi].0 = mid;
                        }
                        heap.push(Node { lower: lb, qb });
                    }
                }
                // an unsolvable node is simply dropped: its box cannot be bounded,
                // which happens only where the depth envelope degenerates
            }
            if heap.is_empty() && best_cost.is_finite() {
                chart_lower = best_cost; // exhausted the tree
                chart_conv = true;
            }
            // the certified bound is the weaker of the two charts
            // union of the charts ⇒ the *weakest* of the two bounds
            lower_bound = lower_bound.min(chart_lower);
            converged &= chart_conv;
        }
        lower_bound = lower_bound.min(best_cost);

        let nq2 = best_q.iter().map(|v| v * v).sum::<f64>();
        let nq = nq2.sqrt();
        let qv = nalgebra::Vector4::new(
            best_q[0] / nq,
            best_q[1] / nq,
            best_q[2] / nq,
            best_q[3] / nq,
        );
        let rotation = Matrix3::from_fn(|k, j| (qv.transpose() * rq[k][j] * qv)[(0, 0)]);
        // t = τ/‖q‖², then undo the world translation and scaling
        let translation = (best_t / nq2 - rotation * p0) * scale;

        BnbSolverResult {
            rotation,
            translation,
            cost: best_cost,
            lower_bound,
            optimality_gap: (best_cost - lower_bound).abs(),
            rel_optimality_gap: (best_cost - lower_bound).abs()
                / best_cost.abs().max(f64::MIN_POSITIVE),
            nodes,
            converged,
            runtime: t0.elapsed().as_secs_f64(),
        }
    }
}

/// The convex relaxation on one box: eq. (9) subject to (5)-(8).
/// Returns (lower bound, relaxed q, relaxed t), or None if MOSEK fails.
fn solve_node(
    res: &[Residual],
    qb: &[(f64, f64); 4],
    tb: &[(f64, f64); 3],
) -> Option<(f64, [f64; 4], Vector3<f64>)> {
    let sb = s_box(qb);
    let m = res.len();
    // variables: q(4) | t(3) | s(10) | w(m)
    let (q0, t0v, s0, w0) = (0i32, 4i32, 7i32, 17i32);
    let nvar = 17 + m as i32;

    let mut task = Task::new()?;
    task.append_vars(nvar).ok()?;
    task.put_obj_sense(Objsense::MINIMIZE).ok()?;

    for (j, b) in qb.iter().enumerate() {
        task.put_var_bound(q0 + j as i32, Boundkey::RA, b.0, b.1)
            .ok()?;
    }
    for (j, b) in tb.iter().enumerate() {
        task.put_var_bound(t0v + j as i32, Boundkey::RA, b.0, b.1)
            .ok()?;
    }
    for (j, b) in sb.iter().enumerate() {
        task.put_var_bound(s0 + j as i32, Boundkey::RA, b.0, b.1)
            .ok()?;
    }
    for i in 0..m {
        task.put_var_bound(w0 + i as i32, Boundkey::LO, 0.0, 0.0)
            .ok()?;
        task.put_c_j(w0 + i as i32, 1.0).ok()?;
    }

    // linear constraints
    let mut rows: Vec<(Vec<i32>, Vec<f64>, i32, f64, f64)> = Vec::new();

    for j in 0..4 {
        let (lj, uj) = qb[j];
        // j = k: concave side is the secant s_jj ≤ (L+U) q_j - L·U
        rows.push((
            vec![s0 + sidx(j, j) as i32, q0 + j as i32],
            vec![1.0, -(lj + uj)],
            Boundkey::UP,
            0.0,
            -lj * uj,
        ));
        for k in (j + 1)..4 {
            let (lk, uk) = qb[k];
            let sjk = s0 + sidx(j, k) as i32;
            let (qj, qk) = (q0 + j as i32, q0 + k as i32);
            // eqs. (5)-(8): McCormick envelopes of s_jk = q_j q_k on the box
            // ! signs corrected, see the header
            rows.push((
                vec![sjk, qj, qk],
                vec![1.0, -lk, -lj],
                Boundkey::LO,
                -lj * lk,
                0.0,
            )); // s ≥ L_j q_k + L_k q_j - L_j L_k
            rows.push((
                vec![sjk, qj, qk],
                vec![1.0, -uk, -uj],
                Boundkey::LO,
                -uj * uk,
                0.0,
            )); // s ≥ U_j q_k + U_k q_j - U_j U_k
            rows.push((
                vec![sjk, qj, qk],
                vec![1.0, -lk, -uj],
                Boundkey::UP,
                0.0,
                -uj * lk,
            )); // s ≤ U_j q_k + L_k q_j - U_j L_k
            rows.push((
                vec![sjk, qj, qk],
                vec![1.0, -uk, -lj],
                Boundkey::UP,
                0.0,
                -lj * uk,
            )); // s ≤ L_j q_k + U_k q_j - L_j U_k
        }
    }

    task.append_cons(rows.len() as i32).ok()?;
    for (i, (idx, val, bk, lo, up)) in rows.iter().enumerate() {
        task.put_a_row(i as i32, idx, val).ok()?;
        task.put_con_bound(i as i32, *bk, *lo, *up).ok()?;
    }

    // conic constraints: 3 AFE rows per s_jj (s_jj ≥ q_j²) and per residual
    // (eq. 9), all RQUAD(3) = {(x₁,x₂,x₃) : 2x₁x₂ ≥ x₃², x₁,x₂ ≥ 0}
    let n_acc = 4 + m;
    task.append_afes(3 * n_acc as i64).ok()?;
    let rquad = task.append_r_quadratic_cone_domain(3).ok()?;
    let mut afe = 0i64;

    for j in 0..4 {
        // 2 · s_jj · 0.5 = s_jj ≥ q_j²
        task.put_afe_f_row(afe, &[s0 + sidx(j, j) as i32], &[1.0])
            .ok()?;
        task.put_afe_g(afe + 1, 0.5).ok()?;
        task.put_afe_f_row(afe + 2, &[q0 + j as i32], &[1.0]).ok()?;
        task.append_acc(rquad, &[afe, afe + 1, afe + 2], &[0.0; 3])
            .ok()?;
        afe += 3;
    }

    for (i, r) in res.iter().enumerate() {
        // concave envelope of the squared denominator: y² ≤ (L+U) y - L·U,
        // giving eq. (9)'s c·(B̂q̂) + d
        // ! holds for any sign of L,U; a positivity clamp would break τ₃ = -1
        let (dl, du) = form_range(&r.den_t, &r.den_s, tb, &sb);
        let (c, d) = (dl + du, -dl * du);

        // 2 · w_i · (½(c·den + d)) ≥ num²
        task.put_afe_f_row(afe, &[w0 + i as i32], &[1.0]).ok()?;
        let mut idx = Vec::new();
        let mut val = Vec::new();
        push_form(&mut idx, &mut val, &r.den_t, &r.den_s, t0v, s0);
        for v in val.iter_mut() {
            *v *= 0.5 * c;
        }
        task.put_afe_f_row(afe + 1, &idx, &val).ok()?;
        task.put_afe_g(afe + 1, 0.5 * d).ok()?;
        let mut idx = Vec::new();
        let mut val = Vec::new();
        push_form(&mut idx, &mut val, &r.num_t, &r.num_s, t0v, s0);
        task.put_afe_f_row(afe + 2, &idx, &val).ok()?;
        task.append_acc(rquad, &[afe, afe + 1, afe + 2], &[0.0; 3])
            .ok()?;
        afe += 3;
    }

    task.optimize().ok()?;
    match task.get_sol_sta(Soltype::ITR) {
        Ok(Solsta::OPTIMAL) => {}
        _ => return None,
    }
    let mut x = vec![0.0; nvar as usize];
    task.get_xx(Soltype::ITR, &mut x).ok()?;
    Some((
        task.get_primal_obj(Soltype::ITR).ok()?,
        [x[0], x[1], x[2], x[3]],
        Vector3::new(x[4], x[5], x[6]),
    ))
}

/// Append a linear form over (t, s) as MOSEK (index, value) pairs.
fn push_form(
    idx: &mut Vec<i32>,
    val: &mut Vec<f64>,
    ct: &[f64; 3],
    cs: &[f64; 10],
    t0v: i32,
    s0: i32,
) {
    for (i, &c) in ct.iter().enumerate() {
        if c != 0.0 {
            idx.push(t0v + i as i32);
            val.push(c);
        }
    }
    for (i, &c) in cs.iter().enumerate() {
        if c != 0.0 {
            idx.push(s0 + i as i32);
            val.push(c);
        }
    }
}
