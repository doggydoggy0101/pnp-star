use nalgebra::{DMatrix, DVector, Matrix2, Matrix3};

/// Quadratic data matrices for the Shor relaxation.
pub struct DataMatrices {
    pub cr: DMatrix<f64>,  // (9, 9)
    pub ct: DMatrix<f64>,  // (3, 3)
    pub ctr: DMatrix<f64>, // (3, 9)
    pub c: f64,
}

/// kron(p, I3) as a (3, 9) matrix: K[l, 3*j + l] = p[j].
fn kron_p_i3(p: &DVector<f64>) -> DMatrix<f64> {
    let mut k = DMatrix::zeros(3, 9);
    for j in 0..3 {
        for l in 0..3 {
            k[(l, 3 * j + l)] = p[j];
        }
    }
    k
}

fn row3(m: &DMatrix<f64>, i: usize) -> DVector<f64> {
    DVector::from_row_slice(&[m[(i, 0)], m[(i, 1)], m[(i, 2)]])
}

/// Pixel residuals scale with the focal length (~800 px), which conditions the SDP poorly,
/// so we rescale every residual by the constant `1/f`, `f = sqrt(fx fy)`.
///
/// `K = [Khat chat; 0 1]` maps a normalized point to the pixel `Khat (.) + chat`. Measured
/// `theta_i = Khat q_i + chat` and predicted `pi_i = Khat x_i + chat` (`x_i` the predicted
/// normalized point) give `theta_i - pi_i = Khat (q_i - x_i)`. Returns
/// `q_i = Khat^{-1} (theta_i - chat)` and `W = Khat / f`, so
/// `W (q_i - x_i) = (theta_i - pi_i) / f`.
pub fn normalize_pinhole(
    image_points: &DMatrix<f64>,
    intrinsic: &Matrix3<f64>,
) -> (DMatrix<f64>, Matrix2<f64>) {
    let (fx, fy) = (intrinsic[(0, 0)], intrinsic[(1, 1)]);
    let s = intrinsic[(0, 1)]; // skew
    let (cx, cy) = (intrinsic[(0, 2)], intrinsic[(1, 2)]);
    let det = fx * fy;
    assert!(det.abs() > 0.0, "normalize_pinhole: singular intrinsic");

    let n = image_points.nrows();
    let mut q = DMatrix::zeros(n, 2);
    for i in 0..n {
        let dv = (image_points[(i, 1)] - cy) / fy;
        q[(i, 0)] = (image_points[(i, 0)] - cx - s * dv) / fx;
        q[(i, 1)] = dv;
    }

    let f = det.abs().sqrt();
    let w = Matrix2::new(fx / f, s / f, 0.0, fy / f);
    (q, w)
}

/// Point-line error data matrices (spherical), matching `plerr_data_matrices`.
pub fn plerr_data_matrices(
    world_points: &DMatrix<f64>,
    bearing_vectors: &DMatrix<f64>,
) -> DataMatrices {
    let n = world_points.nrows();
    let mut cr = DMatrix::<f64>::zeros(9, 9);
    let mut ct = DMatrix::<f64>::zeros(3, 3);
    let mut ctr = DMatrix::<f64>::zeros(3, 9);
    let mut c = 0.0;

    let i3 = DMatrix::<f64>::identity(3, 3);
    for n_idx in 0..n {
        let p = row3(world_points, n_idx);
        let f = row3(bearing_vectors, n_idx);

        // temp_M[3i+j] = p[i] * f[j]
        let mut temp_m = DVector::<f64>::zeros(9);
        for i in 0..3 {
            for j in 0..3 {
                temp_m[3 * i + j] = p[i] * f[j];
            }
        }
        cr -= &temp_m * temp_m.transpose();

        // temp_N = I3 - f f^T
        let temp_n = &i3 - &f * f.transpose();
        ct += &temp_n;
        ctr += &temp_n * kron_p_i3(&p);
        c += p.norm_squared();
    }

    DataMatrices { cr, ct, ctr, c }
}

/// Weighted-reprojection error data matrices (pinhole), matching `wrerr_data_matrices`.
pub fn wrerr_data_matrices(
    world_points: &DMatrix<f64>,
    image_points: &DMatrix<f64>,
    intrinsic: &Matrix3<f64>,
) -> DataMatrices {
    let n = world_points.nrows();
    let i3 = DMatrix::<f64>::identity(3, 3);
    let e3 = DVector::from_row_slice(&[0.0, 0.0, 1.0]);

    let mut cr = DMatrix::<f64>::zeros(9, 9);
    let mut ct = DMatrix::<f64>::zeros(3, 3);
    let mut ctr = DMatrix::<f64>::zeros(3, 9);
    let c = 0.0; // wrerr leaves the constant term at 0

    // surrogate cost of [17, 18]: defined in normalized coords, left unweighted
    let (q, _) = normalize_pinhole(image_points, intrinsic);

    for n_idx in 0..n {
        let p = row3(world_points, n_idx);
        // direction vector (NOT normalized)
        let d = DVector::from_row_slice(&[q[(n_idx, 0)], q[(n_idx, 1)], 1.0]);

        let temp_n = &i3 - &d * e3.transpose();
        let ntn = temp_n.transpose() * &temp_n;
        let k = kron_p_i3(&p);

        ct += &ntn;
        ctr += &ntn * &k;
        cr += k.transpose() * &ntn * &k;
    }

    DataMatrices { cr, ct, ctr, c }
}

/// Marginalized cost matrix (10x10): Schur complement eliminating translation.
pub fn marginalize(dm: &DataMatrices) -> DMatrix<f64> {
    let mut cc = DMatrix::<f64>::identity(10, 10);
    // Ct^{-1} Ctr via full-pivot LU (Ct is symmetric, generically PD).
    let sol = dm
        .ct
        .clone()
        .full_piv_lu()
        .solve(&dm.ctr)
        .expect("marginalize: Ct singular");
    let schur = &dm.cr - dm.ctr.transpose() * sol;
    cc.view_mut((1, 1), (9, 9)).copy_from(&schur);
    cc[(0, 0)] = dm.c;
    cc
}

/// Lifted cost matrix (13x13) keeping translation explicit.
pub fn lift(dm: &DataMatrices) -> DMatrix<f64> {
    let mut cc = DMatrix::<f64>::identity(13, 13);
    cc.view_mut((1, 1), (9, 9)).copy_from(&dm.cr);
    cc.view_mut((1, 10), (9, 3)).copy_from(&dm.ctr.transpose());
    cc.view_mut((10, 1), (3, 9)).copy_from(&dm.ctr);
    cc.view_mut((10, 10), (3, 3)).copy_from(&dm.ct);
    cc[(0, 0)] = dm.c;
    cc
}

/// 5 row-orthogonality constraint matrices (R R^T = I).
pub fn row_orthogonal_constraint_matrices(dim: usize) -> Vec<DMatrix<f64>> {
    let mut list = Vec::new();
    for i in 0..3 {
        for j in i..3 {
            if i == 2 && j == 2 {
                continue; // linearly dependent
            }
            let mut mat = DMatrix::<f64>::zeros(dim, dim);
            for k in 0..3 {
                let idx1 = k * 3 + i;
                let idx2 = k * 3 + j;
                mat[(idx1 + 1, idx2 + 1)] += 0.5;
                mat[(idx2 + 1, idx1 + 1)] += 0.5;
            }
            if i == j {
                mat[(0, 0)] = -1.0;
            }
            list.push(mat);
        }
    }
    list
}

/// 6 column-orthogonality constraint matrices (R^T R = I).
pub fn col_orthogonal_constraint_matrices(dim: usize) -> Vec<DMatrix<f64>> {
    let mut list = Vec::new();
    for i in 0..3 {
        for j in i..3 {
            let mut mat = DMatrix::<f64>::zeros(dim, dim);
            for k in 0..3 {
                let idx1 = i * 3 + k;
                let idx2 = j * 3 + k;
                mat[(idx1 + 1, idx2 + 1)] += 0.5;
                mat[(idx2 + 1, idx1 + 1)] += 0.5;
            }
            if i == j {
                mat[(0, 0)] = -1.0;
            }
            list.push(mat);
        }
    }
    list
}

/// 9 determinant (cofactor) constraint matrices.
pub fn det_constraint_matrices(dim: usize) -> Vec<DMatrix<f64>> {
    let mut list = Vec::new();
    for i in 0..3 {
        for j in 0..3 {
            let mut mat = DMatrix::<f64>::zeros(dim, dim);

            let mut rows = [0usize; 2];
            let mut cols = [0usize; 2];
            let (mut ri, mut ci) = (0, 0);
            for r in 0..3 {
                if r != j {
                    rows[ri] = r;
                    ri += 1;
                }
            }
            for col in 0..3 {
                if col != i {
                    cols[ci] = col;
                    ci += 1;
                }
            }

            let idx1 = rows[0] * 3 + cols[0];
            let idx2 = rows[1] * 3 + cols[1];
            let sign = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
            mat[(idx1 + 1, idx2 + 1)] = sign * 0.5;
            mat[(idx2 + 1, idx1 + 1)] = sign * 0.5;

            let idx3 = rows[0] * 3 + cols[1];
            let idx4 = rows[1] * 3 + cols[0];
            mat[(idx3 + 1, idx4 + 1)] = -sign * 0.5;
            mat[(idx4 + 1, idx3 + 1)] = -sign * 0.5;

            let idx_r = j * 3 + i;
            mat[(idx_r + 1, 0)] = -0.5;
            mat[(0, idx_r + 1)] = -0.5;

            list.push(mat);
        }
    }
    list
}
