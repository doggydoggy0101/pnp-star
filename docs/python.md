# :snake: Python API

## Install

Needs a MOSEK [license](https://www.mosek.com/license/request/) at `~/mosek/mosek.lic`.

```bash
conda create -n pnp python=3.13 -y
conda activate pnp
pip install maturin numpy

maturin develop --release -m pnp-python/Cargo.toml
```

Re-run `maturin develop` after changing the Rust sources, moving the clone, or running `cargo clean` in `pnp-python/`.

## Usage

**Dataset.** Pinhole noise is Gaussian (pixels); spherical noise is Fisher (degrees). Only `num_points` is required; the values below are the defaults.

```python
import pnp

dataset = pnp.PnpSyntheticDataset(
    10,            # num_points
    (5.0, 35.0),   # depth_range (d_min, d_max) in metres
    180.0,         # max_rotation_degree
    5.0,           # max_translation_norm
    10.0,          # noise: pixels (pinhole) or degrees (spherical); default 10 / 1
    "pinhole",     # "pinhole" or "spherical"
    42,            # seed
)
dataset = pnp.PnpSyntheticDataset(10)                            # only num_points required
dataset = pnp.PnpSyntheticDataset(10, camera_model="spherical")  # override any default by name

# fields (ground truth)
dataset.world_points        # np.ndarray (N×3)
dataset.image_points        # np.ndarray (N×2), pinhole
dataset.bearing_vectors     # np.ndarray (N×3), spherical
dataset.intrinsic           # np.ndarray (3×3), camera intrinsic K
dataset.rotation            # np.ndarray (3×3)
dataset.translation         # np.ndarray (3,)
dataset.reprojection_error  # float
dataset.degree_error        # float, spherical

# errors vs. an estimate (R, t)
dataset.compute_reprojection_error(R, t)  # float
dataset.compute_rotation_error(R)         # float
dataset.compute_translation_error(t)      # float
dataset.compute_degree_error(R, t)        # float, spherical
```

**Local solver.** EPnP initialization and Levenberg–Marquardt refinement.

```python
import pnp

result = pnp.LocalSolver().solve(
    dataset.world_points,   # (N×3)
    dataset.image_points,   # (N×2) pinhole / dataset.bearing_vectors (N×3) spherical
    dataset.intrinsic,      # (3×3)         / None for spherical
    refine=True,            # EPnP+LM (default) / False: EPnP only
)
result.rotation     # np.ndarray (3×3)
result.translation  # np.ndarray (3,)
result.runtime      # float, seconds
```

**Alternative solver.** Certifiable solvers for alternative errors: weighted reprojection error for pinhole; point-line error for spherical. *These are not solving the MLE.*

```python
import pnp

result = pnp.AlternativeSolver().solve(
    dataset.world_points,    # (N×3)
    dataset.image_points,    # (N×2) pinhole / dataset.bearing_vectors spherical
    dataset.intrinsic,       # (3×3)         / None for spherical
    marginalize=True,        # default
)
result.rotation            # np.ndarray (3×3)
result.translation         # np.ndarray (3,)
result.sdp_value           # float, SDP optimum
result.qcqp_value          # float, rounded QCQP value
result.optimality_gap      # float
result.rel_optimality_gap  # float
result.rank                # int
result.pose_recovered      # bool, False => pose is NaN, see rank
result.eigval_ratio        # float
result.runtime             # float, seconds
```

**Global solver.** Certifiable solvers for MLE (PnP\*).

```python
import pnp

result = pnp.GlobalSolver().solve(
    dataset.world_points,           # (N×3)
    dataset.image_points,           # (N×2) pinhole / dataset.bearing_vectors spherical
    dataset.intrinsic,              # (3×3)         / None for spherical
    formulation="reduced",          # "reduced" (default) / "chordal" / "dense" (ablation, doesn't scale)
    prune_constraint=False,         # default
    refine=True,                    # LM when not rank-1 (default) / False: raw rounding
)
result.rotation            # np.ndarray (3×3)
result.translation         # np.ndarray (3,)
result.sdp_value           # float, SDP optimum
result.qcqp_value          # float, rounded QCQP value
result.optimality_gap      # float
result.rel_optimality_gap  # float
result.rank                # int
result.pose_recovered      # bool, False => pose is NaN, see rank
result.certified           # bool, rank 1 and e3^T P_i z != 0 for all i
result.eigval_ratio        # float
result.runtime             # float, seconds
```

**BnB solver.** BnB method for MLE.

```python
import pnp

result = pnp.BnbSolver().solve(
    dataset.world_points,           # (N×3)
    dataset.image_points,           # (N×2) pinhole only
    dataset.intrinsic,              # (3×3)
    min_depth=None,                 # None uses the paper's t3^low = 1 world unit
    gamma=None,                     # None derives the paper's gamma_max from a trivial pose
    eps=None, max_nodes=None,       # None uses the defaults
)
result.rotation            # np.ndarray (3×3)
result.translation         # np.ndarray (3,)
result.cost                # float, best objective found
result.lower_bound         # float, certified bound
result.optimality_gap      # float
result.rel_optimality_gap  # float
result.nodes               # int, branch-and-bound nodes explored
result.converged           # bool, False if the node budget ran out
result.runtime             # float, seconds
```
