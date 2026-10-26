//! C ABI for the PnP core. All matrices are passed as row-major `double`
//! arrays. Rotations are returned row-major (R[i*3+j]).
//!
//! cbindgen generates `include/pnp_cxx.h` from this file; the
//! `include/pnp_cxx.hpp` shim wraps it in Eigen-typed classes.

use nalgebra::{DMatrix, Matrix3, Vector3};
use pnp::{alternative, bnb, data, global, local};

/// Which SDP formulation `pnp_global_solve` should solve (paper §6).
/// `Dense` is an ablation only — the block grows with N.
#[repr(C)]
pub enum PnpFormulation {
    Dense = 0,
    Chordal = 1,
    Reduced = 2,
}

impl From<PnpFormulation> for global::Formulation {
    fn from(f: PnpFormulation) -> Self {
        match f {
            PnpFormulation::Dense => global::Formulation::Dense,
            PnpFormulation::Chordal => global::Formulation::Chordal,
            PnpFormulation::Reduced => global::Formulation::Reduced,
        }
    }
}

#[repr(C)]
pub struct PnpPose {
    pub rotation: [f64; 9], // row-major
    pub translation: [f64; 3],
}

/// Result for the branch-and-bound baseline. `cost` is the best objective found
/// and `lower_bound` the certified bound; `converged` is false if the node
/// budget ran out.
#[repr(C)]
pub struct PnpBnbResult {
    pub pose: PnpPose,
    pub cost: f64,
    pub lower_bound: f64,
    pub optimality_gap: f64,
    pub rel_optimality_gap: f64,
    pub nodes: usize,
    pub converged: bool,
    pub runtime: f64,
}

#[repr(C)]
pub struct PnpLocalResult {
    pub pose: PnpPose,
    pub runtime: f64,
}

/// Shared result for the SDP solvers. `qcqp_value` is the rounded QCQP
/// value (upper bound); `sdp_value` is the SDP optimum (lower bound).
#[repr(C)]
pub struct PnpSdpResult {
    pub pose: PnpPose,
    pub sdp_value: f64,
    pub qcqp_value: f64,
    pub optimality_gap: f64,
    pub rel_optimality_gap: f64,
    pub rank: i32,
    pub eigval_ratio: f64,
    /// False when the solution could not be dehomogenized; the pose is NaN.
    pub pose_recovered: bool,
    /// Rank 1 and, for `pnp_global_solve`, `e3^T P_i z != 0` for all i.
    pub certified: bool,
    pub runtime: f64,
}

unsafe fn dmat(ptr: *const f64, rows: usize, cols: usize) -> DMatrix<f64> {
    let slice = std::slice::from_raw_parts(ptr, rows * cols);
    DMatrix::from_row_slice(rows, cols, slice)
}

unsafe fn mat3(ptr: *const f64) -> Matrix3<f64> {
    let slice = std::slice::from_raw_parts(ptr, 9);
    Matrix3::from_row_slice(slice)
}

unsafe fn vec3(ptr: *const f64) -> Vector3<f64> {
    Vector3::new(*ptr, *ptr.add(1), *ptr.add(2))
}

unsafe fn write_dmat(m: &DMatrix<f64>, out: *mut f64) {
    let (rows, cols) = m.shape();
    for i in 0..rows {
        for j in 0..cols {
            *out.add(i * cols + j) = m[(i, j)];
        }
    }
}

unsafe fn write_mat3(m: &Matrix3<f64>, out: *mut f64) {
    for i in 0..3 {
        for j in 0..3 {
            *out.add(i * 3 + j) = m[(i, j)];
        }
    }
}

unsafe fn write_vec3(v: &Vector3<f64>, out: *mut f64) {
    for i in 0..3 {
        *out.add(i) = v[i];
    }
}

fn pose_of(r: &Matrix3<f64>, t: &nalgebra::Vector3<f64>) -> PnpPose {
    let mut rotation = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            rotation[i * 3 + j] = r[(i, j)];
        }
    }
    PnpPose {
        rotation,
        translation: [t[0], t[1], t[2]],
    }
}

// ── PnpSyntheticDataset (opaque handle) ──────────────────────────────────────

/// Opaque handle wrapping the core dataset. Not `#[repr(C)]`, so cbindgen emits
/// it as an opaque struct; the C/C++ side only ever holds a pointer to it.
pub struct PnpSyntheticDataset {
    inner: data::PnpSyntheticDataset,
}

/// Create a synthetic dataset. `model_id`: 0 = pinhole, 1 = spherical. Returns
/// an owning handle; release it with [`pnp_dataset_free`].
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn pnp_dataset_create(
    num_points: usize,
    depth_min: f64,
    depth_max: f64,
    max_rotation_degree: f64,
    max_translation_norm: f64,
    noise: f64,
    model_id: u32,
    seed: u64,
) -> *mut PnpSyntheticDataset {
    let model = if model_id == 0 {
        "pinhole"
    } else {
        "spherical"
    };
    let ds = data::PnpSyntheticDataset::new(
        num_points,
        (depth_min, depth_max),
        max_rotation_degree,
        max_translation_norm,
        noise,
        model,
        seed,
    );
    Box::into_raw(Box::new(PnpSyntheticDataset { inner: ds }))
}

/// # Safety
/// `handle` must be a pointer returned by [`pnp_dataset_create`], or null.
#[no_mangle]
pub unsafe extern "C" fn pnp_dataset_free(handle: *mut PnpSyntheticDataset) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

/// # Safety
/// `handle` must be a live dataset pointer; `out` buffers must hold the stated
/// element counts (world/bearing: N*3, image: N*2, intrinsic/rotation: 9,
/// translation: 3), row-major.
#[no_mangle]
pub unsafe extern "C" fn pnp_dataset_num_points(handle: *const PnpSyntheticDataset) -> usize {
    (*handle).inner.num_points
}

/// # Safety
/// See [`pnp_dataset_num_points`].
#[no_mangle]
pub unsafe extern "C" fn pnp_dataset_world_points(
    handle: *const PnpSyntheticDataset,
    out: *mut f64,
) {
    write_dmat(&(*handle).inner.world_points, out);
}

/// # Safety
/// See [`pnp_dataset_num_points`]. Valid only for pinhole datasets (N*2).
#[no_mangle]
pub unsafe extern "C" fn pnp_dataset_image_points(
    handle: *const PnpSyntheticDataset,
    out: *mut f64,
) {
    write_dmat(&(*handle).inner.image_points, out);
}

/// # Safety
/// See [`pnp_dataset_num_points`]. Valid only for spherical datasets (N*3).
#[no_mangle]
pub unsafe extern "C" fn pnp_dataset_bearing_vectors(
    handle: *const PnpSyntheticDataset,
    out: *mut f64,
) {
    write_dmat(&(*handle).inner.bearing_vectors, out);
}

/// # Safety
/// See [`pnp_dataset_num_points`].
#[no_mangle]
pub unsafe extern "C" fn pnp_dataset_intrinsic(handle: *const PnpSyntheticDataset, out: *mut f64) {
    write_mat3(&(*handle).inner.intrinsic, out);
}

/// # Safety
/// See [`pnp_dataset_num_points`].
#[no_mangle]
pub unsafe extern "C" fn pnp_dataset_rotation(handle: *const PnpSyntheticDataset, out: *mut f64) {
    write_mat3(&(*handle).inner.rotation, out);
}

/// # Safety
/// See [`pnp_dataset_num_points`].
#[no_mangle]
pub unsafe extern "C" fn pnp_dataset_translation(
    handle: *const PnpSyntheticDataset,
    out: *mut f64,
) {
    write_vec3(&(*handle).inner.translation, out);
}

/// # Safety
/// `handle` must be a live dataset pointer.
#[no_mangle]
pub unsafe extern "C" fn pnp_dataset_reprojection_error(handle: *const PnpSyntheticDataset) -> f64 {
    (*handle).inner.reprojection_error
}

/// # Safety
/// `handle` must be a live dataset pointer.
#[no_mangle]
pub unsafe extern "C" fn pnp_dataset_degree_error(handle: *const PnpSyntheticDataset) -> f64 {
    (*handle).inner.degree_error
}

/// # Safety
/// `handle` live; `rot` is 9 row-major doubles, `t` is 3 doubles.
#[no_mangle]
pub unsafe extern "C" fn pnp_dataset_compute_reprojection_error(
    handle: *const PnpSyntheticDataset,
    rot: *const f64,
    t: *const f64,
) -> f64 {
    (*handle)
        .inner
        .compute_reprojection_error(&mat3(rot), &vec3(t))
}

/// # Safety
/// `handle` live; `rot` is 9 row-major doubles.
#[no_mangle]
pub unsafe extern "C" fn pnp_dataset_compute_rotation_error(
    handle: *const PnpSyntheticDataset,
    rot: *const f64,
) -> f64 {
    (*handle).inner.compute_rotation_error(&mat3(rot))
}

/// # Safety
/// `handle` live; `t` is 3 doubles.
#[no_mangle]
pub unsafe extern "C" fn pnp_dataset_compute_translation_error(
    handle: *const PnpSyntheticDataset,
    t: *const f64,
) -> f64 {
    (*handle).inner.compute_translation_error(&vec3(t))
}

/// # Safety
/// `handle` live; `rot` is 9 row-major doubles, `t` is 3 doubles.
#[no_mangle]
pub unsafe extern "C" fn pnp_dataset_compute_degree_error(
    handle: *const PnpSyntheticDataset,
    rot: *const f64,
    t: *const f64,
) -> f64 {
    (*handle).inner.compute_degree_error(&mat3(rot), &vec3(t))
}

/// LocalSolver (EPnP + LM). `obs_cols` is 2 (pinhole) or 3 (spherical).
///
/// # Safety
/// Pointers must reference valid arrays of the stated sizes; `out` must be valid.
#[no_mangle]
pub unsafe extern "C" fn pnp_local_solve(
    world: *const f64,
    n: usize,
    obs: *const f64,
    obs_cols: usize,
    intrinsic: *const f64,
    refine: bool,
    out: *mut PnpLocalResult,
) {
    let r = local::LocalSolver.solve(
        &dmat(world, n, 3),
        &dmat(obs, n, obs_cols),
        &mat3(intrinsic),
        refine,
    );
    *out = PnpLocalResult {
        pose: pose_of(&r.rotation, &r.translation),
        runtime: r.runtime,
    };
}

/// AlternativeSolver (Shor relaxation).
///
/// # Safety
/// See [`pnp_local_solve`].
#[no_mangle]
pub unsafe extern "C" fn pnp_alternative_solve(
    world: *const f64,
    n: usize,
    obs: *const f64,
    obs_cols: usize,
    intrinsic: *const f64,
    marginalize: bool,
    out: *mut PnpSdpResult,
) {
    let r = alternative::AlternativeSolver.solve(
        &dmat(world, n, 3),
        &dmat(obs, n, obs_cols),
        &mat3(intrinsic),
        marginalize,
    );
    *out = PnpSdpResult {
        pose: pose_of(&r.rotation, &r.translation),
        sdp_value: r.sdp_value,
        qcqp_value: r.qcqp_value,
        optimality_gap: r.optimality_gap,
        rel_optimality_gap: r.rel_optimality_gap,
        rank: r.rank as i32,
        eigval_ratio: r.eigval_ratio,
        pose_recovered: r.pose_recovered,
        certified: r.pose_recovered && r.rank == 1,
        runtime: r.runtime,
    };
}

/// GlobalSolver (multi-clique SDP, certifiable).
///
/// # Safety
/// See [`pnp_local_solve`].
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn pnp_global_solve(
    world: *const f64,
    n: usize,
    obs: *const f64,
    obs_cols: usize,
    intrinsic: *const f64,
    formulation: PnpFormulation,
    prune_constraint: bool,
    refine: bool,
    out: *mut PnpSdpResult,
) {
    let r = global::GlobalSolver.solve(
        &dmat(world, n, 3),
        &dmat(obs, n, obs_cols),
        &mat3(intrinsic),
        formulation.into(),
        prune_constraint,
        refine,
    );
    *out = PnpSdpResult {
        pose: pose_of(&r.rotation, &r.translation),
        sdp_value: r.sdp_value,
        qcqp_value: r.qcqp_value,
        optimality_gap: r.optimality_gap,
        rel_optimality_gap: r.rel_optimality_gap,
        rank: r.rank as i32,
        eigval_ratio: r.eigval_ratio,
        pose_recovered: r.pose_recovered,
        certified: r.certified,
        runtime: r.runtime,
    };
}

/// BnbSolver (Olsson et al. ICPR'06 branch-and-bound). Pinhole only.
/// `min_depth <= 0` uses 1 world unit; `gamma <= 0`
/// derives it from a trivial pose.
///
/// # Safety
/// See [`pnp_local_solve`].
#[no_mangle]
pub unsafe extern "C" fn pnp_bnb_solve(
    world: *const f64,
    n: usize,
    obs: *const f64,
    intrinsic: *const f64,
    min_depth: f64,
    gamma: f64,
    eps: f64,
    max_nodes: usize,
    out: *mut PnpBnbResult,
) {
    let d = bnb::BnbOptions::default();
    let opts = bnb::BnbOptions {
        min_depth: if min_depth > 0.0 {
            Some(min_depth)
        } else {
            None
        },
        gamma: (gamma > 0.0).then_some(gamma),
        eps: if eps > 0.0 { eps } else { d.eps },
        max_nodes: if max_nodes > 0 {
            max_nodes
        } else {
            d.max_nodes
        },
        ..d
    };
    let r = bnb::BnbSolver.solve(
        &dmat(world, n, 3),
        &dmat(obs, n, 2),
        &mat3(intrinsic),
        &opts,
    );
    *out = PnpBnbResult {
        pose: pose_of(&r.rotation, &r.translation),
        cost: r.cost,
        lower_bound: r.lower_bound,
        optimality_gap: r.optimality_gap,
        rel_optimality_gap: r.rel_optimality_gap,
        nodes: r.nodes,
        converged: r.converged,
        runtime: r.runtime,
    };
}
