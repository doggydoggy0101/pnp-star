use nalgebra::{DMatrix, DVector, Matrix3, Vector3};

/// Read row `i` of an (N, 3) matrix as a `Vector3`.
pub fn row3(m: &DMatrix<f64>, i: usize) -> Vector3<f64> {
    Vector3::new(m[(i, 0)], m[(i, 1)], m[(i, 2)])
}

/// Write `v` into row `i` of an (N, 3) matrix.
pub fn set_row3(m: &mut DMatrix<f64>, i: usize, v: &Vector3<f64>) {
    m[(i, 0)] = v[0];
    m[(i, 1)] = v[1];
    m[(i, 2)] = v[2];
}

/// Skew-symmetric matrix of `v` (so that `skew(v) * w == v.cross(w)`).
pub fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(
        0.0, -v[2], v[1], //
        v[2], 0.0, -v[0], //
        -v[1], v[0], 0.0,
    )
}

/// Rodrigues exponential map so(3) -> SO(3).
pub fn exp_so3(phi: &Vector3<f64>) -> Matrix3<f64> {
    let theta = phi.norm();
    if theta < 1e-12 {
        return Matrix3::identity() + skew(phi);
    }
    let k = skew(&(phi / theta));
    let (s, c) = (theta.sin(), theta.cos());
    Matrix3::identity() + s * k + (1.0 - c) * (k * k)
}

/// vec[3*j + k] = R[k, j]  (column-major stacking).
pub fn rot2vec(rot: &Matrix3<f64>) -> DVector<f64> {
    let mut v = DVector::zeros(9);
    for j in 0..3 {
        for k in 0..3 {
            v[3 * j + k] = rot[(k, j)];
        }
    }
    v
}

/// Inverse of [`rot2vec`].
pub fn vec2rot(vec: &DVector<f64>) -> Matrix3<f64> {
    let mut r = Matrix3::zeros();
    for j in 0..3 {
        for k in 0..3 {
            r[(k, j)] = vec[3 * j + k];
        }
    }
    r
}

/// Project a 3x3 matrix onto SO(3) via SVD (with reflection handling).
pub fn project_to_so3(mat: &Matrix3<f64>) -> Matrix3<f64> {
    let svd = mat.svd(true, true);
    let u = svd.u.expect("SVD U");
    let v_t = svd.v_t.expect("SVD V^T");
    let rot = u * v_t;
    if rot.determinant() < 0.0 {
        let d = Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, -1.0));
        u * d * v_t
    } else {
        rot
    }
}
