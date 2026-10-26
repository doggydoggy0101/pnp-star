//! EPnP closed-form absolute pose estimation.
//!
//! Modified from:
//!   https://github.com/opencv/opencv/blob/4.x/modules/calib3d/src/epnp.cpp
//!   https://github.com/laurentkneip/opengv/blob/master/src/absolute_pose/modules/Epnp.cpp

use nalgebra::{DMatrix, DVector, Matrix3, SMatrix, SVector, Vector3};

use crate::linalg::row3;

type Matrix6x10 = SMatrix<f64, 6, 10>;
type Vector6 = SVector<f64, 6>;

/// Choose 4 control points: c0 = centroid; c1..c3 = c0 + sqrt(s_k^2 / N) * v_k.
fn choose_control_points(world_points: &DMatrix<f64>) -> [Vector3<f64>; 4] {
    let n = world_points.nrows();
    let mut centroid = Vector3::zeros();
    for i in 0..n {
        centroid += row3(world_points, i);
    }
    centroid /= n as f64;

    let mut centered = DMatrix::zeros(n, 3);
    for i in 0..n {
        let r = row3(world_points, i) - centroid;
        centered[(i, 0)] = r[0];
        centered[(i, 1)] = r[1];
        centered[(i, 2)] = r[2];
    }

    let svd = centered.svd(false, true);
    let s = &svd.singular_values;
    let v_t = svd.v_t.expect("SVD V^T"); // (3, 3), V.col(k) = v_t.row(k)

    let mut c = [centroid; 4];
    for k in 0..3 {
        let vk = Vector3::new(v_t[(k, 0)], v_t[(k, 1)], v_t[(k, 2)]);
        c[k + 1] = centroid + (s[k] * s[k] / n as f64).sqrt() * vk;
    }
    c
}

/// Barycentric coordinates alpha (N, 4): sum_j alpha_ij = 1, p_i = sum_j alpha_ij c_j.
fn compute_barycentric(world_points: &DMatrix<f64>, c: &[Vector3<f64>; 4]) -> DMatrix<f64> {
    let n = world_points.nrows();
    let mut b = Matrix3::zeros();
    for j in 0..3 {
        b.set_column(j, &(c[j + 1] - c[0]));
    }
    let lu = b.full_piv_lu();

    let mut alpha = DMatrix::zeros(n, 4);
    for i in 0..n {
        let d = row3(world_points, i) - c[0];
        let a = lu.solve(&d).expect("epnp: control-point basis singular");
        alpha[(i, 0)] = 1.0 - (a[0] + a[1] + a[2]);
        alpha[(i, 1)] = a[0];
        alpha[(i, 2)] = a[1];
        alpha[(i, 3)] = a[2];
    }
    alpha
}

/// Least-squares solve min ||A x - b|| via SVD.
fn lstsq(a: &DMatrix<f64>, b: &DVector<f64>) -> DVector<f64> {
    a.clone()
        .svd(true, true)
        .solve(b, 1e-12)
        .expect("epnp: lstsq failed")
}

/// L (6, 10): coefficients of the inter-control-point distance constraints, built
/// from the 4 smallest right-singular vectors ut.row(11-i).
fn compute_l_6x10(ut: &DMatrix<f64>) -> Matrix6x10 {
    let mut dv = [[[0.0f64; 3]; 6]; 4]; // dv[vec][pair][xyz]
    for i in 0..4 {
        let (mut a, mut b) = (0usize, 1usize);
        for j in 0..6 {
            for k in 0..3 {
                dv[i][j][k] = ut[(11 - i, 3 * a + k)] - ut[(11 - i, 3 * b + k)];
            }
            b += 1;
            if b > 3 {
                a += 1;
                b = a + 1;
            }
        }
    }
    let dot = |x: &[f64; 3], y: &[f64; 3]| x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
    let mut l = Matrix6x10::zeros();
    for i in 0..6 {
        l[(i, 0)] = dot(&dv[0][i], &dv[0][i]);
        l[(i, 1)] = 2.0 * dot(&dv[0][i], &dv[1][i]);
        l[(i, 2)] = dot(&dv[1][i], &dv[1][i]);
        l[(i, 3)] = 2.0 * dot(&dv[0][i], &dv[2][i]);
        l[(i, 4)] = 2.0 * dot(&dv[1][i], &dv[2][i]);
        l[(i, 5)] = dot(&dv[2][i], &dv[2][i]);
        l[(i, 6)] = 2.0 * dot(&dv[0][i], &dv[3][i]);
        l[(i, 7)] = 2.0 * dot(&dv[1][i], &dv[3][i]);
        l[(i, 8)] = 2.0 * dot(&dv[2][i], &dv[3][i]);
        l[(i, 9)] = dot(&dv[3][i], &dv[3][i]);
    }
    l
}

/// Rho (6): squared inter-control-point distances in the world frame.
fn compute_rho(c: &[Vector3<f64>; 4]) -> Vector6 {
    let d2 = |i: usize, j: usize| (c[i] - c[j]).norm_squared();
    Vector6::new(d2(0, 1), d2(0, 2), d2(0, 3), d2(1, 2), d2(1, 3), d2(2, 3))
}

// betas10 = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]

/// betas_approx_1 = [B11 B12 B13 B14] (columns 0,1,3,6 of L).
fn find_betas_1(l: &Matrix6x10, rho: &Vector6) -> [f64; 4] {
    let mut a = DMatrix::zeros(6, 4);
    for i in 0..6 {
        a[(i, 0)] = l[(i, 0)];
        a[(i, 1)] = l[(i, 1)];
        a[(i, 2)] = l[(i, 3)];
        a[(i, 3)] = l[(i, 6)];
    }
    let b4 = lstsq(&a, &DVector::from_column_slice(rho.as_slice()));
    let mut betas = [0.0; 4];
    if b4[0] < 0.0 {
        betas[0] = (-b4[0]).sqrt();
        betas[1] = -b4[1] / betas[0];
        betas[2] = -b4[2] / betas[0];
        betas[3] = -b4[3] / betas[0];
    } else {
        betas[0] = b4[0].sqrt();
        betas[1] = b4[1] / betas[0];
        betas[2] = b4[2] / betas[0];
        betas[3] = b4[3] / betas[0];
    }
    betas
}

/// betas_approx_2 = [B11 B12 B22] (columns 0,1,2 of L).
fn find_betas_2(l: &Matrix6x10, rho: &Vector6) -> [f64; 4] {
    let mut a = DMatrix::zeros(6, 3);
    for i in 0..6 {
        a[(i, 0)] = l[(i, 0)];
        a[(i, 1)] = l[(i, 1)];
        a[(i, 2)] = l[(i, 2)];
    }
    let b3 = lstsq(&a, &DVector::from_column_slice(rho.as_slice()));
    let mut betas = [0.0; 4];
    if b3[0] < 0.0 {
        betas[0] = (-b3[0]).sqrt();
        betas[1] = if b3[2] < 0.0 { (-b3[2]).sqrt() } else { 0.0 };
    } else {
        betas[0] = b3[0].sqrt();
        betas[1] = if b3[2] > 0.0 { b3[2].sqrt() } else { 0.0 };
    }
    if b3[1] < 0.0 {
        betas[0] = -betas[0];
    }
    betas
}

/// betas_approx_3 = [B11 B12 B22 B13 B23] (columns 0,1,2,3,4 of L).
fn find_betas_3(l: &Matrix6x10, rho: &Vector6) -> [f64; 4] {
    let mut a = DMatrix::zeros(6, 5);
    for i in 0..6 {
        for k in 0..5 {
            a[(i, k)] = l[(i, k)];
        }
    }
    let b5 = lstsq(&a, &DVector::from_column_slice(rho.as_slice()));
    let mut betas = [0.0; 4];
    if b5[0] < 0.0 {
        betas[0] = (-b5[0]).sqrt();
        betas[1] = if b5[2] < 0.0 { (-b5[2]).sqrt() } else { 0.0 };
    } else {
        betas[0] = b5[0].sqrt();
        betas[1] = if b5[2] > 0.0 { b5[2].sqrt() } else { 0.0 };
    }
    if b5[1] < 0.0 {
        betas[0] = -betas[0];
    }
    betas[2] = b5[3] / betas[0];
    betas
}

/// Gauss-Newton refinement of the betas against the rigidity constraints.
fn gauss_newton(l: &Matrix6x10, rho: &Vector6, betas: &mut [f64; 4]) {
    for _ in 0..5 {
        let mut a = DMatrix::zeros(6, 4);
        let mut b = DVector::zeros(6);
        for i in 0..6 {
            a[(i, 0)] = 2.0 * l[(i, 0)] * betas[0]
                + l[(i, 1)] * betas[1]
                + l[(i, 3)] * betas[2]
                + l[(i, 6)] * betas[3];
            a[(i, 1)] = l[(i, 1)] * betas[0]
                + 2.0 * l[(i, 2)] * betas[1]
                + l[(i, 4)] * betas[2]
                + l[(i, 7)] * betas[3];
            a[(i, 2)] = l[(i, 3)] * betas[0]
                + l[(i, 4)] * betas[1]
                + 2.0 * l[(i, 5)] * betas[2]
                + l[(i, 8)] * betas[3];
            a[(i, 3)] = l[(i, 6)] * betas[0]
                + l[(i, 7)] * betas[1]
                + l[(i, 8)] * betas[2]
                + 2.0 * l[(i, 9)] * betas[3];
            let pred = l[(i, 0)] * betas[0] * betas[0]
                + l[(i, 1)] * betas[0] * betas[1]
                + l[(i, 2)] * betas[1] * betas[1]
                + l[(i, 3)] * betas[0] * betas[2]
                + l[(i, 4)] * betas[1] * betas[2]
                + l[(i, 5)] * betas[2] * betas[2]
                + l[(i, 6)] * betas[0] * betas[3]
                + l[(i, 7)] * betas[1] * betas[3]
                + l[(i, 8)] * betas[2] * betas[3]
                + l[(i, 9)] * betas[3] * betas[3];
            b[i] = rho[i] - pred;
        }
        let dx = lstsq(&a, &b);
        for i in 0..4 {
            betas[i] += dx[i];
        }
    }
}

/// Camera-frame control points ccs = sum_i betas[i] * ut.row(11-i).
fn compute_ccs(betas: &[f64; 4], ut: &DMatrix<f64>) -> [Vector3<f64>; 4] {
    let mut ccs = [Vector3::zeros(); 4];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..3 {
                ccs[j][k] += betas[i] * ut[(11 - i, 3 * j + k)];
            }
        }
    }
    ccs
}

/// Camera-frame points pcs (N, 3) from barycentric coords and ccs.
fn compute_pcs(alpha: &DMatrix<f64>, ccs: &[Vector3<f64>; 4]) -> DMatrix<f64> {
    let n = alpha.nrows();
    let mut pcs = DMatrix::zeros(n, 3);
    for i in 0..n {
        let mut p = Vector3::zeros();
        for j in 0..4 {
            p += alpha[(i, j)] * ccs[j];
        }
        pcs[(i, 0)] = p[0];
        pcs[(i, 1)] = p[1];
        pcs[(i, 2)] = p[2];
    }
    pcs
}

/// Flip sign so the first point's depth matches its bearing hemisphere (OpenGV).
fn solve_for_sign(pcs: &mut DMatrix<f64>, ccs: &mut [Vector3<f64>; 4], sign0: f64) {
    let pz0 = pcs[(0, 2)];
    if (pz0 < 0.0 && sign0 > 0.0) || (pz0 > 0.0 && sign0 < 0.0) {
        for c in ccs.iter_mut() {
            *c = -*c;
        }
        for i in 0..pcs.nrows() {
            pcs[(i, 0)] = -pcs[(i, 0)];
            pcs[(i, 1)] = -pcs[(i, 1)];
            pcs[(i, 2)] = -pcs[(i, 2)];
        }
    }
}

/// Absolute orientation (Kabsch) aligning world points to camera points.
fn estimate_r_and_t(pcs: &DMatrix<f64>, world: &DMatrix<f64>) -> (Matrix3<f64>, Vector3<f64>) {
    let n = world.nrows();
    let mut pc0 = Vector3::zeros();
    let mut pw0 = Vector3::zeros();
    for i in 0..n {
        pc0 += row3(pcs, i);
        pw0 += row3(world, i);
    }
    pc0 /= n as f64;
    pw0 /= n as f64;

    let mut abt = Matrix3::zeros();
    for i in 0..n {
        abt += (row3(pcs, i) - pc0) * (row3(world, i) - pw0).transpose();
    }
    let svd = abt.svd(true, true);
    let u = svd.u.expect("SVD U");
    let v = svd.v_t.expect("SVD V^T").transpose();
    let mut r = u * v.transpose();
    if r.determinant() < 0.0 {
        let mut v_prime = v;
        v_prime.set_column(2, &(-v.column(2)));
        r = u * v_prime.transpose();
    }
    let t = pc0 - r * pw0;
    (r, t)
}

/// Reprojection error in normalized image coords (fu=fv=1, uc=vc=0).
fn reprojection_error(
    r: &Matrix3<f64>,
    t: &Vector3<f64>,
    world: &DMatrix<f64>,
    us: &DMatrix<f64>,
) -> f64 {
    let n = world.nrows();
    let mut sum = 0.0;
    for i in 0..n {
        let pc = r * row3(world, i) + t;
        let ue = pc[0] / pc[2];
        let ve = pc[1] / pc[2];
        let du = us[(i, 0)] - ue;
        let dv = us[(i, 1)] - ve;
        sum += (du * du + dv * dv).sqrt();
    }
    sum / n as f64
}

/// One betas hypothesis → (R, t, reprojection error).
#[allow(clippy::too_many_arguments)]
fn compute_r_and_t(
    ut: &DMatrix<f64>,
    betas: &[f64; 4],
    alpha: &DMatrix<f64>,
    world: &DMatrix<f64>,
    us: &DMatrix<f64>,
    sign0: f64,
) -> (Matrix3<f64>, Vector3<f64>, f64) {
    let mut ccs = compute_ccs(betas, ut);
    let mut pcs = compute_pcs(alpha, &ccs);
    solve_for_sign(&mut pcs, &mut ccs, sign0);
    let (r, t) = estimate_r_and_t(&pcs, world);
    let e = reprojection_error(&r, &t, world, us);
    (r, t, e)
}

/// EPnP closed-form pose estimation. `bearings` are (u_norm, v_norm, 1) for
/// pinhole or unit bearings for spherical.
pub fn epnp(world_points: &DMatrix<f64>, bearings: &DMatrix<f64>) -> (Matrix3<f64>, Vector3<f64>) {
    let n = world_points.nrows();
    let cws = choose_control_points(world_points);
    let alpha = compute_barycentric(world_points, &cws);

    // normalized image coords us = (x/z, y/z); sign of first point's hemisphere
    let mut us = DMatrix::zeros(n, 2);
    let mut sign0 = 1.0;
    for i in 0..n {
        let b = row3(bearings, i);
        us[(i, 0)] = b[0] / b[2];
        us[(i, 1)] = b[1] / b[2];
        if i == 0 {
            sign0 = if b[2] > 0.0 { 1.0 } else { -1.0 };
        }
    }

    // M (2N, 12), projection form with fu=fv=1, uc=vc=0
    let mut m = DMatrix::zeros(2 * n, 12);
    for i in 0..n {
        let (u, v) = (us[(i, 0)], us[(i, 1)]);
        for j in 0..4 {
            let a = alpha[(i, j)];
            m[(2 * i, 3 * j)] = a;
            m[(2 * i, 3 * j + 2)] = -a * u;
            m[(2 * i + 1, 3 * j + 1)] = a;
            m[(2 * i + 1, 3 * j + 2)] = -a * v;
        }
    }

    // right-singular vectors of M = eigenvectors of M^T M, ascending sing. value
    // toward row 11; ut.row(11-i) gives the 4 smallest.
    let mtm = m.tr_mul(&m); // (12, 12)
    let ut = mtm.svd(false, true).v_t.expect("SVD V^T"); // (12, 12)

    let l = compute_l_6x10(&ut);
    let rho = compute_rho(&cws);

    let mut best: Option<(f64, Matrix3<f64>, Vector3<f64>)> = None;
    for approx in 1..=3 {
        let mut betas = match approx {
            1 => find_betas_1(&l, &rho),
            2 => find_betas_2(&l, &rho),
            _ => find_betas_3(&l, &rho),
        };
        gauss_newton(&l, &rho, &mut betas);
        let (r, t, e) = compute_r_and_t(&ut, &betas, &alpha, world_points, &us, sign0);
        if best.as_ref().is_none_or(|(be, _, _)| e < *be) {
            best = Some((e, r, t));
        }
    }
    let (_, r, t) = best.expect("epnp: no hypothesis");
    (r, t)
}
