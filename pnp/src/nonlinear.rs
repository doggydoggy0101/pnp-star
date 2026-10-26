//! Levenberg-Marquardt refinement of (R, t) on SE(3) for absolute pose.
//!
//! Modified from:
//!   https://github.com/opencv/opencv/blob/4.x/modules/calib3d/src/solvepnp.cpp
//!   https://github.com/laurentkneip/opengv/blob/master/src/absolute_pose/methods.cpp

use nalgebra::{DMatrix, DVector, Matrix2, Matrix2x3, Matrix3, Vector2, Vector3};

use crate::linalg::{exp_so3, row3, skew};
use crate::utils::normalize_pinhole;

#[derive(Clone, Copy)]
pub struct LmOptions {
    pub max_iter: usize,
    pub ftol: f64, // relative cost change threshold
    pub xtol: f64, // step norm threshold
    pub lambda_init: f64,
    pub lambda_up: f64,
    pub lambda_down: f64,
}

impl Default for LmOptions {
    fn default() -> Self {
        LmOptions {
            max_iter: 50,
            ftol: 1e-12,
            xtol: 1e-12,
            lambda_init: 1e-3,
            lambda_up: 10.0,
            lambda_down: 0.1,
        }
    }
}

/// Pinhole residual + Jacobian. r_i = W ([x/z, y/z] - q_i), q = normalized image
/// points and W the intrinsic weight from `normalize_pinhole` (W = I2 when fx = fy).
fn pinhole_rj(
    r: &Matrix3<f64>,
    t: &Vector3<f64>,
    wp: &DMatrix<f64>,
    q: &DMatrix<f64>,
    w: &Matrix2<f64>,
) -> (DVector<f64>, DMatrix<f64>) {
    let n = wp.nrows();
    let mut res = DVector::zeros(2 * n);
    let mut jac = DMatrix::zeros(2 * n, 6);
    for i in 0..n {
        let pi = row3(wp, i);
        let p_cam = r * pi + t;
        let (x, y, z) = (p_cam[0], p_cam[1], p_cam[2]);
        let inv_z = 1.0 / z;
        let inv_z2 = inv_z * inv_z;
        let e = w * Vector2::new(x * inv_z - q[(i, 0)], y * inv_z - q[(i, 1)]);
        res[2 * i] = e[0];
        res[2 * i + 1] = e[1];

        let dproj = w * Matrix2x3::new(inv_z, 0.0, -x * inv_z2, 0.0, inv_z, -y * inv_z2);
        let dp_dphi = -r * skew(&pi);
        jac.view_mut((2 * i, 0), (2, 3))
            .copy_from(&(dproj * dp_dphi));
        jac.view_mut((2 * i, 3), (2, 3)).copy_from(&dproj);
    }
    (res, jac)
}

/// Spherical residual + Jacobian. r_i = p_cam/||p_cam|| - f_i.
fn spherical_rj(
    r: &Matrix3<f64>,
    t: &Vector3<f64>,
    wp: &DMatrix<f64>,
    f: &DMatrix<f64>,
) -> (DVector<f64>, DMatrix<f64>) {
    let n = wp.nrows();
    let mut res = DVector::zeros(3 * n);
    let mut jac = DMatrix::zeros(3 * n, 6);
    for i in 0..n {
        let pi = row3(wp, i);
        let fi = row3(f, i);
        let p_cam = r * pi + t;
        let nrm = p_cam.norm();
        let u = p_cam / nrm;
        let resid = u - fi;
        for k in 0..3 {
            res[3 * i + k] = resid[k];
        }
        let du = (Matrix3::identity() - u * u.transpose()) / nrm;
        let dp_dphi = -r * skew(&pi);
        jac.view_mut((3 * i, 0), (3, 3)).copy_from(&(du * dp_dphi));
        jac.view_mut((3 * i, 3), (3, 3)).copy_from(&du);
    }
    (res, jac)
}

/// LM loop: (J^T J + lambda * diag(J^T J)) dx = -J^T r, right-perturbation update.
fn lm_loop<F>(r: &mut Matrix3<f64>, t: &mut Vector3<f64>, mut compute: F, opts: &LmOptions)
where
    F: FnMut(&Matrix3<f64>, &Vector3<f64>) -> (DVector<f64>, DMatrix<f64>),
{
    let (mut res, mut jac) = compute(r, t);
    let mut cost = 0.5 * res.norm_squared();
    let mut lambda = opts.lambda_init;

    for _ in 0..opts.max_iter {
        let jtj = jac.transpose() * &jac; // (6, 6)
        let jtr = jac.transpose() * &res; // (6,)
        let mut lhs = jtj.clone();
        for k in 0..6 {
            lhs[(k, k)] += lambda * jtj[(k, k)];
        }
        let dx = match lhs.lu().solve(&(-&jtr)) {
            Some(v) => v,
            None => break,
        };

        let dphi = Vector3::new(dx[0], dx[1], dx[2]);
        let dt = Vector3::new(dx[3], dx[4], dx[5]);
        let r_try = *r * exp_so3(&dphi);
        let t_try = *t + dt;

        let (res_try, jac_try) = compute(&r_try, &t_try);
        let cost_try = 0.5 * res_try.norm_squared();

        if cost_try < cost {
            let rel = (cost - cost_try) / (cost + 1e-30);
            let step_norm = dx.norm();
            *r = r_try;
            *t = t_try;
            res = res_try;
            jac = jac_try;
            cost = cost_try;
            lambda *= opts.lambda_down;
            if rel < opts.ftol || step_norm < opts.xtol {
                break;
            }
        } else {
            lambda *= opts.lambda_up;
            if lambda > 1e16 {
                break;
            }
        }
    }
}

/// Refine pinhole pose. `image_points` in pixels; intrinsic normalizes internally.
///
/// Minimizes the pixel reprojection error of the MLE (paper eq. 7), evaluated in
/// normalized coordinates with the weight W from `normalize_pinhole`, so the cost is
/// the pixel error over det(Khat) -- same minimizer, better conditioned. GlobalSolver
/// applies the same W, so Local and Global optimize the identical objective for any
/// intrinsic. Reduces to the plain normalized error when fx = fy and skew is zero.
pub fn refine_pinhole(
    r: &mut Matrix3<f64>,
    t: &mut Vector3<f64>,
    world_points: &DMatrix<f64>,
    image_points: &DMatrix<f64>,
    intrinsic: &Matrix3<f64>,
    opts: &LmOptions,
) {
    let (q, w) = normalize_pinhole(image_points, intrinsic);
    lm_loop(
        r,
        t,
        |rin, tin| pinhole_rj(rin, tin, world_points, &q, &w),
        opts,
    );
}

/// Refine spherical pose. residual = (Rp+t)/||Rp+t|| - f (the GlobalSolver MLE).
pub fn refine_spherical(
    r: &mut Matrix3<f64>,
    t: &mut Vector3<f64>,
    world_points: &DMatrix<f64>,
    bearing_vectors: &DMatrix<f64>,
    opts: &LmOptions,
) {
    lm_loop(
        r,
        t,
        |rin, tin| spherical_rj(rin, tin, world_points, bearing_vectors),
        opts,
    );
}
