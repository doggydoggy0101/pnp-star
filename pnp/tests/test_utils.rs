//! Rotation-utility and data-matrix tests (mirrors test/test_utils.cpp).

use nalgebra::{DMatrix, DVector, Matrix3, Vector3};
use pnp::linalg::{exp_so3, project_to_so3, rot2vec, vec2rot};
use pnp::utils::{
    col_orthogonal_constraint_matrices, det_constraint_matrices, lift, marginalize,
    plerr_data_matrices, row_orthogonal_constraint_matrices, wrerr_data_matrices,
};

#[test]
fn rot_roundtrip() {
    let r = exp_so3(&Vector3::new(0.3, -0.7, 0.2));
    assert!((r - vec2rot(&rot2vec(&r))).norm() < 1e-13);
}

#[test]
fn project_identity() {
    assert!((project_to_so3(&Matrix3::identity()) - Matrix3::identity()).norm() < 1e-12);
}

#[test]
fn project_so3() {
    // project an arbitrary matrix onto SO(3): result has det 1 and R^T R = I
    let m = Matrix3::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    let r = project_to_so3(&m);
    assert!((r.determinant() - 1.0).abs() < 1e-10);
    assert!((r.transpose() * r - Matrix3::identity()).norm() < 1e-10);
}

#[test]
fn constraint_counts() {
    assert_eq!(row_orthogonal_constraint_matrices(10).len(), 5);
    assert_eq!(col_orthogonal_constraint_matrices(10).len(), 6);
    assert_eq!(det_constraint_matrices(10).len(), 9);
    for c in row_orthogonal_constraint_matrices(13) {
        assert_eq!(c.nrows(), 13);
    }
}

#[test]
fn constraints_satisfied_by_valid_rotation() {
    // for a valid rotation, trace(C_i * x x^T) = 0 for all constraint matrices
    let r = exp_so3(&Vector3::new(0.5, -0.3, 0.9));
    let mut x = DVector::zeros(10);
    x[0] = 1.0;
    x.rows_mut(1, 9).copy_from(&rot2vec(&r));
    let xxt = &x * x.transpose();

    let mut all = row_orthogonal_constraint_matrices(10);
    all.extend(col_orthogonal_constraint_matrices(10));
    all.extend(det_constraint_matrices(10));
    for (i, c) in all.iter().enumerate() {
        assert!((c * &xxt).trace().abs() < 1e-10, "constraint {i} residual");
    }
}

#[test]
fn data_matrix_shapes() {
    let n = 6;
    let mut world = DMatrix::zeros(n, 3);
    let mut pixel = DMatrix::zeros(n, 2);
    let k = Matrix3::new(580.0, 0.0, 320.0, 0.0, 580.0, 240.0, 0.0, 0.0, 1.0);
    for i in 0..n {
        let x = -1.0 + 2.0 * (i as f64) / (n as f64 - 1.0);
        world[(i, 0)] = x;
        world[(i, 1)] = x;
        world[(i, 2)] = 5.0;
        pixel[(i, 0)] = k[(0, 0)] * world[(i, 0)] / world[(i, 2)] + k[(0, 2)];
        pixel[(i, 1)] = k[(1, 1)] * world[(i, 1)] / world[(i, 2)] + k[(1, 2)];
    }
    // unit bearing vectors for the point-line data matrices
    let mut bearing = DMatrix::zeros(n, 3);
    for i in 0..n {
        let v = Vector3::new(
            (pixel[(i, 0)] - k[(0, 2)]) / k[(0, 0)],
            (pixel[(i, 1)] - k[(1, 2)]) / k[(1, 1)],
            1.0,
        )
        .normalize();
        bearing[(i, 0)] = v[0];
        bearing[(i, 1)] = v[1];
        bearing[(i, 2)] = v[2];
    }

    let dm_pl = plerr_data_matrices(&world, &bearing);
    assert_eq!((dm_pl.cr.nrows(), dm_pl.cr.ncols()), (9, 9));
    assert_eq!((dm_pl.ct.nrows(), dm_pl.ct.ncols()), (3, 3));
    assert_eq!((dm_pl.ctr.nrows(), dm_pl.ctr.ncols()), (3, 9));

    let dm_wr = wrerr_data_matrices(&world, &pixel, &k);
    assert_eq!(dm_wr.cr.nrows(), 9);

    let c10 = marginalize(&dm_pl);
    assert_eq!((c10.nrows(), c10.ncols()), (10, 10));

    let c13 = lift(&dm_pl);
    assert_eq!((c13.nrows(), c13.ncols()), (13, 13));
    assert!(
        (&c13 - c13.transpose()).norm() < 1e-12,
        "lift must be symmetric"
    );
}
