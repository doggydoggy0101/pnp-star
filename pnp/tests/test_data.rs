//! Dataset tests (mirrors test/test_data.cpp on the C++ branch): generated
//! scenes must satisfy the ground-truth and structural invariants.

use pnp::data::PnpSyntheticDataset;
use pnp::linalg::row3;

#[test]
fn pinhole_zero_noise() {
    let d = PnpSyntheticDataset::new(20, (5.0, 15.0), 30.0, 5.0, 0.0, "pinhole", 7);
    // ground-truth pose must give zero error
    assert!(d.compute_reprojection_error(&d.rotation, &d.translation) < 1e-6);
    assert!(d.compute_rotation_error(&d.rotation) < 1e-7);
    assert!(d.compute_translation_error(&d.translation) < 1e-7);
}

#[test]
fn spherical_zero_noise() {
    let d = PnpSyntheticDataset::new(20, (5.0, 15.0), 30.0, 5.0, 0.0, "spherical", 11);
    assert!(d.compute_reprojection_error(&d.rotation, &d.translation) < 1e-10);
    assert!(d.compute_degree_error(&d.rotation, &d.translation) < 1e-6);
    assert!(d.compute_rotation_error(&d.rotation) < 1e-4); // degrees
    assert!(d.compute_translation_error(&d.translation) < 1e-7);
}

#[test]
fn bearing_vectors_unit_norm() {
    // bearing vectors must be unit vectors regardless of noise level
    let d = PnpSyntheticDataset::new(15, (5.0, 15.0), 30.0, 5.0, 1.0, "spherical", 3);
    for i in 0..d.num_points {
        let n = row3(&d.bearing_vectors, i).norm();
        assert!((n - 1.0).abs() < 1e-12, "bearing {i} norm {n}");
    }
}

#[test]
fn pixel_coordinates_in_image() {
    // all sampled pixels must lie within [0, W] x [0, H]
    let d = PnpSyntheticDataset::new(50, (5.0, 15.0), 30.0, 5.0, 0.0, "pinhole", 5);
    let w = 2.0 * d.intrinsic[(0, 2)];
    let h = 2.0 * d.intrinsic[(1, 2)];
    for i in 0..d.num_points {
        let (u, v) = (d.image_points[(i, 0)], d.image_points[(i, 1)]);
        assert!((0.0..=w).contains(&u), "u {u} out of [0,{w}]");
        assert!((0.0..=h).contains(&v), "v {v} out of [0,{h}]");
    }
}
