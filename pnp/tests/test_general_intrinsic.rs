//! General-intrinsic (fx != fy, nonzero skew) checks.
//!
//! The MLE is the pixel reprojection error (paper eq. 7). The solvers evaluate it in
//! normalized coordinates with the weight W from `normalize_pinhole`, so the cost is
//! the pixel error over det(Khat) -- same minimizer, better conditioned.

use nalgebra::{DMatrix, Matrix3, Vector3};
use pnp::data::PnpSyntheticDataset;
use pnp::global::{Formulation, GlobalSolver};
use pnp::local::LocalSolver;
use pnp::utils::normalize_pinhole;

fn row3(m: &DMatrix<f64>, i: usize) -> Vector3<f64> {
    Vector3::new(m[(i, 0)], m[(i, 1)], m[(i, 2)])
}

/// sum_i ||theta_i - Pi(R p_i + t)||^2 in pixels.
fn pixel_error(
    r: &Matrix3<f64>,
    t: &Vector3<f64>,
    world: &DMatrix<f64>,
    pixels: &DMatrix<f64>,
    k: &Matrix3<f64>,
) -> f64 {
    let mut acc = 0.0;
    for i in 0..world.nrows() {
        let h = k * (r * row3(world, i) + t);
        let du = h[0] / h[2] - pixels[(i, 0)];
        let dv = h[1] / h[2] - pixels[(i, 1)];
        acc += du * du + dv * dv;
    }
    acc
}

fn rot_err_deg(a: &Matrix3<f64>, b: &Matrix3<f64>) -> f64 {
    let c = ((a.transpose() * b).trace() - 1.0) / 2.0;
    c.clamp(-1.0, 1.0).acos().to_degrees()
}

#[test]
fn normalize_pinhole_roundtrip() {
    let k = Matrix3::new(900.0, 3.5, 310.0, 0.0, 640.0, 250.0, 0.0, 0.0, 1.0);
    let pix = DMatrix::from_row_slice(3, 2, &[100.0, 200.0, 320.0, 240.0, 615.0, 33.0]);
    let (q, w) = normalize_pinhole(&pix, &k);

    for i in 0..3 {
        // Khat q + chat == theta
        let u = k[(0, 0)] * q[(i, 0)] + k[(0, 1)] * q[(i, 1)] + k[(0, 2)];
        let v = k[(1, 1)] * q[(i, 1)] + k[(1, 2)];
        assert!((u - pix[(i, 0)]).abs() < 1e-9, "u roundtrip");
        assert!((v - pix[(i, 1)]).abs() < 1e-9, "v roundtrip");
    }
    // det W == 1, and W == I2 for the isotropic skew-free case
    assert!((w.determinant() - 1.0).abs() < 1e-12);
    let (q2, w2) = normalize_pinhole(&pix, &PnpSyntheticDataset::default_intrinsic());
    assert!((w2 - nalgebra::Matrix2::identity()).norm() < 1e-12);
    assert!(q2.nrows() == 3);
}

#[test]
fn zero_noise_anisotropic() {
    for k in [
        Matrix3::new(800.0, 0.0, 320.0, 0.0, 600.0, 240.0, 0.0, 0.0, 1.0), // fx != fy
        Matrix3::new(950.0, 4.0, 305.0, 0.0, 610.0, 255.0, 0.0, 0.0, 1.0), // + skew
    ] {
        let ds = PnpSyntheticDataset::new_with_intrinsic(
            10,
            (5.0, 35.0),
            180.0,
            5.0,
            0.0,
            "pinhole",
            7,
            k,
        );
        let local = LocalSolver.solve(&ds.world_points, &ds.image_points, &k, true);
        let global = GlobalSolver.solve(
            &ds.world_points,
            &ds.image_points,
            &k,
            Formulation::Reduced,
            true,
            true,
        );

        // LM converges tighter than the SDP; mirror the tolerances in test_solvers
        for (name, r, t, tol) in [
            ("local", &local.rotation, &local.translation, 1e-6),
            ("global", &global.rotation, &global.translation, 1e-3),
        ] {
            assert!(
                rot_err_deg(r, &ds.rotation) < tol,
                "{name}: rotation error {} deg",
                rot_err_deg(r, &ds.rotation)
            );
            assert!(
                (t - ds.translation).norm() < tol,
                "{name}: translation error {}",
                (t - ds.translation).norm()
            );
        }
        assert_eq!(global.rank, 1, "relaxation should be tight at zero noise");
    }
}

#[test]
fn global_objective_is_pixel_error() {
    let k = Matrix3::new(800.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0);
    let ds =
        PnpSyntheticDataset::new_with_intrinsic(12, (5.0, 35.0), 180.0, 5.0, 2.0, "pinhole", 3, k);
    let g = GlobalSolver.solve(
        &ds.world_points,
        &ds.image_points,
        &k,
        Formulation::Reduced,
        true,
        true,
    );

    // qcqp_value == pixel error / det(Khat)
    let det = k[(0, 0)] * k[(1, 1)];
    let px = pixel_error(
        &g.rotation,
        &g.translation,
        &ds.world_points,
        &ds.image_points,
        &k,
    );
    let rel = (g.qcqp_value - px / det).abs() / (px / det);
    assert!(
        rel < 1e-8,
        "qcqp_value {} vs pixel/det {}",
        g.qcqp_value,
        px / det
    );
}

#[test]
fn weighting_beats_unweighted_under_anisotropy() {
    // strongly anisotropic: the unweighted normalized cost is a different estimator
    let k = Matrix3::new(1200.0, 0.0, 320.0, 0.0, 400.0, 240.0, 0.0, 0.0, 1.0);
    let mut n_better = 0;
    let mut n_trials = 0;

    for seed in 0..12u64 {
        let ds = PnpSyntheticDataset::new_with_intrinsic(
            8,
            (5.0, 35.0),
            180.0,
            5.0,
            3.0,
            "pinhole",
            seed,
            k,
        );
        // ours: weighted == pixel MLE
        let ours = LocalSolver.solve(&ds.world_points, &ds.image_points, &k, true);
        // old behaviour: unweighted normalized cost (identity intrinsic on normalized pts)
        let (q, _) = normalize_pinhole(&ds.image_points, &k);
        let old = LocalSolver.solve(&ds.world_points, &q, &Matrix3::identity(), true);

        let e_ours = pixel_error(
            &ours.rotation,
            &ours.translation,
            &ds.world_points,
            &ds.image_points,
            &k,
        );
        let e_old = pixel_error(
            &old.rotation,
            &old.translation,
            &ds.world_points,
            &ds.image_points,
            &k,
        );

        assert!(
            e_ours <= e_old * (1.0 + 1e-9),
            "seed {seed}: weighted {e_ours} should not exceed unweighted {e_old}"
        );
        if e_ours < e_old * (1.0 - 1e-6) {
            n_better += 1;
        }
        n_trials += 1;
    }
    assert!(
        n_better * 2 > n_trials,
        "expected the weighting to strictly matter on most seeds, got {n_better}/{n_trials}"
    );
}
