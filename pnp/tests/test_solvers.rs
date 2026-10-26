//! Solver tests. Every solver must recover the pose on a zero-noise scene and
//! the relaxation must stay tight at low noise; both camera models are covered.

use nalgebra::{DMatrix, Matrix3};
use pnp::alternative::AlternativeSolver;
use pnp::data::PnpSyntheticDataset;
use pnp::global::{build_quotient_ring_transform, Formulation, GlobalSolver};
use pnp::local::LocalSolver;

const MODELS: [&str; 2] = ["pinhole", "spherical"];

fn scene(num_points: usize, noise: f64, model: &str, seed: u64) -> PnpSyntheticDataset {
    PnpSyntheticDataset::new(num_points, (5.0, 35.0), 30.0, 5.0, noise, model, seed)
}

/// Observations and intrinsic for the dataset's model: (N, 2) + K for pinhole,
/// (N, 3) + None for spherical.
fn inputs(d: &PnpSyntheticDataset) -> (&DMatrix<f64>, Option<&Matrix3<f64>>) {
    if d.camera_model == "pinhole" {
        (&d.image_points, Some(&d.intrinsic))
    } else {
        (&d.bearing_vectors, None)
    }
}

/// Thm 7/8: Dense ≡ Chordal, and Reduced is (weakly) tighter. N kept small —
/// the Dense block is 13 + 24N / 13 + 36N and does not scale.
fn formulations_agree(model: &str, noise: f64) {
    let d = scene(6, noise, model, 7);
    let (obs, k) = inputs(&d);
    let run = |f| GlobalSolver.solve(&d.world_points, obs, k, f, false, true);
    let dense = run(Formulation::Dense);
    let chordal = run(Formulation::Chordal);
    let reduced = run(Formulation::Reduced);

    // Grone: the chordal decomposition is exact, so the optima coincide
    assert!(
        (dense.sdp_value - chordal.sdp_value).abs() < 1e-6,
        "{model}: dense {} vs chordal {}",
        dense.sdp_value,
        chordal.sdp_value
    );
    // pinhole: the syzygy row is not implied by Chordal ⇒ Reduced can only be tighter
    assert!(
        reduced.sdp_value > chordal.sdp_value - 1e-6,
        "{model}: reduced {} < chordal {}",
        reduced.sdp_value,
        chordal.sdp_value
    );
    // the three must agree with each other, whatever the noise
    for (name, r) in [("dense", &dense), ("reduced", &reduced)] {
        let dr = (r.rotation - chordal.rotation).norm();
        let dt = (r.translation - chordal.translation).norm();
        assert!(
            dr < 1e-4 && dt < 1e-4,
            "{model}: {name} vs chordal dR {dr} dt {dt}"
        );
    }
    // at zero noise they must also hit ground truth and certify
    if noise == 0.0 {
        for r in [&dense, &chordal, &reduced] {
            assert_eq!(r.rank, 1, "{model}");
            assert!(d.compute_rotation_error(&r.rotation) < 1e-3, "{model}");
            assert!(
                d.compute_translation_error(&r.translation) < 1e-3,
                "{model}"
            );
        }
    }
}

#[test]
fn formulations_agree_pinhole() {
    formulations_agree("pinhole", 0.0);
    formulations_agree("pinhole", 1.0);
}

#[test]
fn formulations_agree_spherical() {
    formulations_agree("spherical", 0.0);
    formulations_agree("spherical", 0.5);
}

/// Spherical's three model rows are independent, so Chordal must lift all of
/// them to match Reduced. Lifting two (as for pinhole) left it strictly looser:
/// this scene gave rank 9 and a 10% lower bound against Reduced's rank 1.
#[test]
fn chordal_matches_reduced_spherical() {
    let d = PnpSyntheticDataset::new(10, (5.0, 35.0), 180.0, 5.0, 2.0, "spherical", 2);
    let run = |f| GlobalSolver.solve(&d.world_points, &d.bearing_vectors, None, f, false, false);
    let chordal = run(Formulation::Chordal);
    let reduced = run(Formulation::Reduced);
    let gap = (reduced.sdp_value - chordal.sdp_value).abs() / reduced.sdp_value;
    assert!(
        gap < 1e-5,
        "reduced {} vs chordal {}",
        reduced.sdp_value,
        chordal.sdp_value
    );
    assert_eq!((chordal.rank, reduced.rank), (1, 1));
}

#[test]
fn global_exact_at_zero_noise() {
    for model in MODELS {
        let d = scene(20, 0.0, model, 42);
        let (obs, k) = inputs(&d);
        let r = GlobalSolver.solve(&d.world_points, obs, k, Formulation::Reduced, false, true);
        assert_eq!(r.rank, 1, "{model}");
        assert!(r.pose_recovered, "{model}");
        assert!(r.certified, "{model}");
        assert!(r.optimality_gap < 1e-5, "{model}: gap {}", r.optimality_gap);
        assert!(d.compute_rotation_error(&r.rotation) < 1e-3, "{model}");
        assert!(
            d.compute_translation_error(&r.translation) < 1e-3,
            "{model}"
        );
    }
}

/// The relaxation must stay tight, and the certificate hold, under low noise.
#[test]
fn global_tight_at_low_noise() {
    for (model, noise) in [("pinhole", 1.0), ("spherical", 1.0)] {
        let d = scene(20, noise, model, 42);
        let (obs, k) = inputs(&d);
        let r = GlobalSolver.solve(&d.world_points, obs, k, Formulation::Reduced, false, true);
        assert_eq!(r.rank, 1, "{model}");
        assert!(r.certified, "{model}");
        assert!(
            r.rel_optimality_gap < 1e-3,
            "{model}: rel gap {}",
            r.rel_optimality_gap
        );
    }
}

/// Coincident world points leave `scale = 0`; extraction must decline rather
/// than divide by zero.
#[test]
fn degenerate_input_reports_no_pose() {
    for model in MODELS {
        let d = scene(6, 0.0, model, 42);
        let (obs, k) = inputs(&d);
        let p = DMatrix::from_fn(d.world_points.nrows(), 3, |_, j| d.world_points[(0, j)]);
        let r = GlobalSolver.solve(&p, obs, k, Formulation::Reduced, false, true);
        assert!(!r.pose_recovered, "{model}");
        assert!(!r.certified, "{model}");
        let a = AlternativeSolver.solve(&p, obs, k, true);
        assert!(!a.pose_recovered, "{model}");
    }
}

#[test]
fn alternative_exact_at_zero_noise() {
    for model in MODELS {
        for marginalize in [true, false] {
            let d = scene(20, 0.0, model, 42);
            let (obs, k) = inputs(&d);
            let r = AlternativeSolver.solve(&d.world_points, obs, k, marginalize);
            assert_eq!(r.rank, 1, "{model} marginalize={marginalize}");
            assert!(r.pose_recovered, "{model} marginalize={marginalize}");
            assert!(
                d.compute_rotation_error(&r.rotation) < 1e-3,
                "{model} marginalize={marginalize}"
            );
            assert!(
                d.compute_translation_error(&r.translation) < 1e-3,
                "{model} marginalize={marginalize}"
            );
        }
    }
}

#[test]
fn local_exact_at_zero_noise() {
    for model in MODELS {
        let d = scene(20, 0.0, model, 42);
        let (obs, k) = inputs(&d);
        let r = LocalSolver.solve(&d.world_points, obs, k, true);
        assert!(d.compute_rotation_error(&r.rotation) < 1e-6, "{model}");
        assert!(
            d.compute_translation_error(&r.translation) < 1e-6,
            "{model}"
        );
    }
}

#[test]
fn quotient_transform_annihilates_model() {
    // the transform T must satisfy mat_A · T = 0 for an arbitrary model matrix
    let dim_hub = 13;
    let dim_full = dim_hub + 2 * 12; // pinhole basis [h, z, y]
    let mut a = DMatrix::zeros(3, dim_full);
    for r in 0..3 {
        for c in 0..dim_full {
            a[(r, c)] = ((r * 7 + c * 3 + 1) % 11) as f64 - 5.0;
        }
    }
    let t = build_quotient_ring_transform(&a, dim_hub);
    assert_eq!(t.nrows(), dim_full);
    assert_eq!(t.ncols(), dim_full - 3);
    assert!((&a * &t).norm() < 1e-9);
}

// BnbSolver (Olsson et al. ICPR'06 baseline)

/// Two independent global solvers must agree on the MLE optimum. PnP* certifies
/// it via rank-1; BnB certifies it by exhausting its branch-and-bound tree.
#[test]
fn bnb_matches_global() {
    use pnp::bnb::{BnbOptions, BnbSolver};
    let opts = BnbOptions::default();
    for noise in [0.0, 2.0] {
        let d = PnpSyntheticDataset::new(6, (5.0, 35.0), 30.0, 5.0, noise, "pinhole", 7);
        let g = GlobalSolver.solve(
            &d.world_points,
            &d.image_points,
            &d.intrinsic,
            Formulation::Reduced,
            false,
            true,
        );
        let b = BnbSolver.solve(&d.world_points, &d.image_points, &d.intrinsic, &opts);
        assert!(b.converged, "bnb did not converge at noise {noise}");
        assert!(b.lower_bound <= b.cost + 1e-12, "invalid bound at {noise}");
        assert_eq!(g.rank, 1);
        let rel = (b.cost - g.qcqp_value).abs() / g.qcqp_value.max(1e-12);
        assert!(
            rel < 1e-3 || b.cost < 1e-8,
            "noise {noise}: bnb {} vs global {}",
            b.cost,
            g.qcqp_value
        );
        assert!((b.rotation - g.rotation).norm() < 1e-3);
        assert!((b.translation - g.translation).norm() < 1e-2);
    }
}
