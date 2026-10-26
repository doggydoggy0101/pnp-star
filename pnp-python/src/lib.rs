//! PyO3 bindings: the `pnp` Python module.

use nalgebra::{DMatrix, Matrix3, Vector3};
use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use pnp_core::{alternative, bnb, data, global, local};

fn dmat(a: &PyReadonlyArray2<f64>) -> DMatrix<f64> {
    let v = a.as_array();
    DMatrix::from_fn(v.nrows(), v.ncols(), |i, j| v[[i, j]])
}

fn mat3(a: Option<PyReadonlyArray2<f64>>) -> Matrix3<f64> {
    match a {
        Some(arr) => {
            let v = arr.as_array();
            Matrix3::from_fn(|i, j| v[[i, j]])
        }
        None => Matrix3::identity(),
    }
}

fn vec3(a: &PyReadonlyArray1<f64>) -> Vector3<f64> {
    let v = a.as_array();
    Vector3::new(v[0], v[1], v[2])
}

fn m3_py(py: Python<'_>, m: &Matrix3<f64>) -> Py<PyArray2<f64>> {
    Array2::from_shape_fn((3, 3), |(i, j)| m[(i, j)])
        .into_pyarray(py)
        .unbind()
}

fn v3_py(py: Python<'_>, v: &Vector3<f64>) -> Py<PyArray1<f64>> {
    Array1::from_vec(vec![v[0], v[1], v[2]])
        .into_pyarray(py)
        .unbind()
}

fn dm_py(py: Python<'_>, m: &DMatrix<f64>) -> Py<PyArray2<f64>> {
    Array2::from_shape_fn((m.nrows(), m.ncols()), |(i, j)| m[(i, j)])
        .into_pyarray(py)
        .unbind()
}

#[pyclass]
struct PnpSyntheticDataset {
    inner: data::PnpSyntheticDataset,
}

#[pymethods]
impl PnpSyntheticDataset {
    #[new]
    // ! text_signature repeats `signature` (keep in sync): PyO3 shows a tuple default as `...`
    #[pyo3(signature = (num_points, depth_range=(5.0, 35.0), max_rotation_degree=180.0,
        max_translation_norm=5.0, noise=None, camera_model="pinhole", seed=42),
        text_signature = "(num_points, depth_range=(5.0, 35.0), max_rotation_degree=180.0, \
        max_translation_norm=5.0, noise=None, camera_model=\"pinhole\", seed=42)")]
    #[allow(clippy::too_many_arguments)]
    fn new(
        num_points: usize,
        depth_range: (f64, f64),
        max_rotation_degree: f64,
        max_translation_norm: f64,
        noise: Option<f64>,
        camera_model: &str,
        seed: u64,
    ) -> Self {
        // None: 10 px pinhole / 1° spherical, the demos' values
        let noise = noise.unwrap_or(if camera_model == "spherical" {
            1.0
        } else {
            10.0
        });
        PnpSyntheticDataset {
            inner: data::PnpSyntheticDataset::new(
                num_points,
                depth_range,
                max_rotation_degree,
                max_translation_norm,
                noise,
                camera_model,
                seed,
            ),
        }
    }

    #[getter]
    fn num_points(&self) -> usize {
        self.inner.num_points
    }
    #[getter]
    fn camera_model(&self) -> String {
        self.inner.camera_model.clone()
    }
    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }
    #[getter]
    fn intrinsic(&self, py: Python<'_>) -> Py<PyArray2<f64>> {
        m3_py(py, &self.inner.intrinsic)
    }
    #[getter]
    fn rotation(&self, py: Python<'_>) -> Py<PyArray2<f64>> {
        m3_py(py, &self.inner.rotation)
    }
    #[getter]
    fn translation(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        v3_py(py, &self.inner.translation)
    }
    #[getter]
    fn world_points(&self, py: Python<'_>) -> Py<PyArray2<f64>> {
        dm_py(py, &self.inner.world_points)
    }
    #[getter]
    fn image_points(&self, py: Python<'_>) -> Py<PyArray2<f64>> {
        dm_py(py, &self.inner.image_points)
    }
    #[getter]
    fn bearing_vectors(&self, py: Python<'_>) -> Py<PyArray2<f64>> {
        dm_py(py, &self.inner.bearing_vectors)
    }
    #[getter]
    fn reprojection_error(&self) -> f64 {
        self.inner.reprojection_error
    }
    #[getter]
    fn degree_error(&self) -> f64 {
        self.inner.degree_error
    }

    fn compute_reprojection_error(
        &self,
        rot: PyReadonlyArray2<f64>,
        t: PyReadonlyArray1<f64>,
    ) -> f64 {
        self.inner
            .compute_reprojection_error(&mat3(Some(rot)), &vec3(&t))
    }
    fn compute_rotation_error(&self, rot: PyReadonlyArray2<f64>) -> f64 {
        self.inner.compute_rotation_error(&mat3(Some(rot)))
    }
    fn compute_translation_error(&self, t: PyReadonlyArray1<f64>) -> f64 {
        self.inner.compute_translation_error(&vec3(&t))
    }
    fn compute_degree_error(&self, rot: PyReadonlyArray2<f64>, t: PyReadonlyArray1<f64>) -> f64 {
        self.inner.compute_degree_error(&mat3(Some(rot)), &vec3(&t))
    }
}

#[pyclass]
struct LocalSolverResult {
    rotation: Matrix3<f64>,
    translation: Vector3<f64>,
    #[pyo3(get)]
    runtime: f64,
}

#[pymethods]
impl LocalSolverResult {
    #[getter]
    fn rotation(&self, py: Python<'_>) -> Py<PyArray2<f64>> {
        m3_py(py, &self.rotation)
    }
    #[getter]
    fn translation(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        v3_py(py, &self.translation)
    }
}

#[pyclass]
struct LocalSolver;

#[pymethods]
impl LocalSolver {
    #[new]
    fn new() -> Self {
        LocalSolver
    }
    #[pyo3(signature = (world_points, observations, intrinsic=None, refine=true))]
    fn solve(
        &self,
        world_points: PyReadonlyArray2<f64>,
        observations: PyReadonlyArray2<f64>,
        intrinsic: Option<PyReadonlyArray2<f64>>,
        refine: bool,
    ) -> LocalSolverResult {
        let r = local::LocalSolver.solve(
            &dmat(&world_points),
            &dmat(&observations),
            &mat3(intrinsic),
            refine,
        );
        LocalSolverResult {
            rotation: r.rotation,
            translation: r.translation,
            runtime: r.runtime,
        }
    }
}

#[pyclass]
struct AlternativeSolverResult {
    rotation: Matrix3<f64>,
    translation: Vector3<f64>,
    #[pyo3(get)]
    sdp_value: f64,
    #[pyo3(get)]
    qcqp_value: f64,
    #[pyo3(get)]
    optimality_gap: f64,
    #[pyo3(get)]
    rel_optimality_gap: f64,
    #[pyo3(get)]
    rank: usize,
    #[pyo3(get)]
    eigval_ratio: f64,
    /// False when the solution could not be dehomogenized; the pose is NaN.
    #[pyo3(get)]
    pose_recovered: bool,
    #[pyo3(get)]
    runtime: f64,
}

#[pymethods]
impl AlternativeSolverResult {
    #[getter]
    fn rotation(&self, py: Python<'_>) -> Py<PyArray2<f64>> {
        m3_py(py, &self.rotation)
    }
    #[getter]
    fn translation(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        v3_py(py, &self.translation)
    }
}

#[pyclass]
struct AlternativeSolver;

#[pymethods]
impl AlternativeSolver {
    #[new]
    fn new() -> Self {
        AlternativeSolver
    }
    #[pyo3(signature = (world_points, observations, intrinsic=None, marginalize=true))]
    fn solve(
        &self,
        world_points: PyReadonlyArray2<f64>,
        observations: PyReadonlyArray2<f64>,
        intrinsic: Option<PyReadonlyArray2<f64>>,
        marginalize: bool,
    ) -> AlternativeSolverResult {
        let r = alternative::AlternativeSolver.solve(
            &dmat(&world_points),
            &dmat(&observations),
            &mat3(intrinsic),
            marginalize,
        );
        AlternativeSolverResult {
            rotation: r.rotation,
            translation: r.translation,
            sdp_value: r.sdp_value,
            qcqp_value: r.qcqp_value,
            optimality_gap: r.optimality_gap,
            rel_optimality_gap: r.rel_optimality_gap,
            rank: r.rank,
            eigval_ratio: r.eigval_ratio,
            pose_recovered: r.pose_recovered,
            runtime: r.runtime,
        }
    }
}

#[pyclass]
struct BnbSolverResult {
    rotation: Matrix3<f64>,
    translation: Vector3<f64>,
    #[pyo3(get)]
    cost: f64,
    #[pyo3(get)]
    lower_bound: f64,
    #[pyo3(get)]
    optimality_gap: f64,
    #[pyo3(get)]
    rel_optimality_gap: f64,
    #[pyo3(get)]
    nodes: usize,
    #[pyo3(get)]
    converged: bool,
    #[pyo3(get)]
    runtime: f64,
}

#[pymethods]
impl BnbSolverResult {
    #[getter]
    fn rotation(&self, py: Python<'_>) -> Py<PyArray2<f64>> {
        m3_py(py, &self.rotation)
    }
    #[getter]
    fn translation(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        v3_py(py, &self.translation)
    }
}

/// Branch-and-bound baseline (Olsson et al., ICPR'06). Pinhole only.
#[pyclass]
struct BnbSolver;

#[pymethods]
impl BnbSolver {
    #[new]
    fn new() -> Self {
        BnbSolver
    }
    #[pyo3(signature = (world_points, image_points, intrinsic,
        min_depth=None, gamma=None, eps=None, max_nodes=None))]
    fn solve(
        &self,
        world_points: PyReadonlyArray2<f64>,
        image_points: PyReadonlyArray2<f64>,
        intrinsic: PyReadonlyArray2<f64>,
        min_depth: Option<f64>,
        gamma: Option<f64>,
        eps: Option<f64>,
        max_nodes: Option<usize>,
    ) -> BnbSolverResult {
        let d = bnb::BnbOptions::default();
        let opts = bnb::BnbOptions {
            min_depth,
            gamma,
            eps: eps.unwrap_or(d.eps),
            max_nodes: max_nodes.unwrap_or(d.max_nodes),
            ..d
        };
        let r = bnb::BnbSolver.solve(
            &dmat(&world_points),
            &dmat(&image_points),
            &mat3(Some(intrinsic)),
            &opts,
        );
        BnbSolverResult {
            rotation: r.rotation,
            translation: r.translation,
            cost: r.cost,
            lower_bound: r.lower_bound,
            optimality_gap: r.optimality_gap,
            rel_optimality_gap: r.rel_optimality_gap,
            nodes: r.nodes,
            converged: r.converged,
            runtime: r.runtime,
        }
    }
}

#[pyclass]
struct GlobalSolverResult {
    rotation: Matrix3<f64>,
    translation: Vector3<f64>,
    #[pyo3(get)]
    sdp_value: f64,
    #[pyo3(get)]
    qcqp_value: f64,
    #[pyo3(get)]
    optimality_gap: f64,
    #[pyo3(get)]
    rel_optimality_gap: f64,
    #[pyo3(get)]
    rank: usize,
    #[pyo3(get)]
    eigval_ratio: f64,
    /// False when the solution could not be dehomogenized; the pose is NaN.
    #[pyo3(get)]
    pose_recovered: bool,
    /// Rank 1 and `e3^T P_i z != 0` for all i: certified global optimum.
    #[pyo3(get)]
    certified: bool,
    #[pyo3(get)]
    runtime: f64,
}

#[pymethods]
impl GlobalSolverResult {
    #[getter]
    fn rotation(&self, py: Python<'_>) -> Py<PyArray2<f64>> {
        m3_py(py, &self.rotation)
    }
    #[getter]
    fn translation(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        v3_py(py, &self.translation)
    }
}

#[pyclass]
struct GlobalSolver;

#[pymethods]
impl GlobalSolver {
    #[new]
    fn new() -> Self {
        GlobalSolver
    }
    #[pyo3(signature = (world_points, observations, intrinsic=None,
        formulation="reduced", prune_constraint=false, refine=true))]
    fn solve(
        &self,
        world_points: PyReadonlyArray2<f64>,
        observations: PyReadonlyArray2<f64>,
        intrinsic: Option<PyReadonlyArray2<f64>>,
        formulation: &str,
        prune_constraint: bool,
        refine: bool,
    ) -> PyResult<GlobalSolverResult> {
        let formulation = match formulation {
            "dense" => global::Formulation::Dense,
            "chordal" => global::Formulation::Chordal,
            "reduced" => global::Formulation::Reduced,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "formulation must be 'dense', 'chordal' or 'reduced', got '{other}'"
                )))
            }
        };
        let r = global::GlobalSolver.solve(
            &dmat(&world_points),
            &dmat(&observations),
            &mat3(intrinsic),
            formulation,
            prune_constraint,
            refine,
        );
        Ok(GlobalSolverResult {
            rotation: r.rotation,
            translation: r.translation,
            sdp_value: r.sdp_value,
            qcqp_value: r.qcqp_value,
            optimality_gap: r.optimality_gap,
            rel_optimality_gap: r.rel_optimality_gap,
            rank: r.rank,
            eigval_ratio: r.eigval_ratio,
            pose_recovered: r.pose_recovered,
            certified: r.certified,
            runtime: r.runtime,
        })
    }
}

#[pymodule]
fn pnp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PnpSyntheticDataset>()?;
    m.add_class::<LocalSolver>()?;
    m.add_class::<LocalSolverResult>()?;
    m.add_class::<AlternativeSolver>()?;
    m.add_class::<AlternativeSolverResult>()?;
    m.add_class::<BnbSolver>()?;
    m.add_class::<BnbSolverResult>()?;
    m.add_class::<GlobalSolver>()?;
    m.add_class::<GlobalSolverResult>()?;
    Ok(())
}
