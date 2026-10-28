# :crab: Rust API

## Install

Needs a MOSEK [license](https://www.mosek.com/license/request/) at `~/mosek/mosek.lic`.

```toml
[dependencies]
pnp = { git = "https://github.com/doggydoggy0101/pnp-star" }
nalgebra = "0.33"
```

```rust
// build.rs: bake the path to MOSEK into your binaries
fn main() {
    if let Some(dirs) = std::env::var_os("DEP_PNP_MOSEK_RPATH") {
        for dir in std::env::split_paths(&dirs) {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
        }
    }
}
```

MOSEK is linked dynamically; without the `build.rs`, a binary fails to load `libmosek64` at run time.

## Usage

**Dataset.** Pinhole noise is Gaussian (pixels); spherical noise is Fisher (degrees).

```rust
use pnp::data::PnpSyntheticDataset;

let dataset = PnpSyntheticDataset::new(
    10,            // num_points
    (5.0, 35.0),   // depth_range (d_min, d_max) in metres
    180.0,         // max_rotation_degree
    5.0,           // max_translation_norm
    10.0,          // noise: pixels (pinhole) or degrees (spherical)
    "pinhole",     // "pinhole" or "spherical"
    42,            // seed
);

// fields (ground truth)
dataset.world_points;        // DMatrix<f64> (N×3)
dataset.image_points;        // DMatrix<f64> (N×2), pinhole
dataset.bearing_vectors;     // DMatrix<f64> (N×3), spherical
dataset.intrinsic;           // Matrix3<f64>, camera intrinsic K
dataset.rotation;            // Matrix3<f64>
dataset.translation;         // Vector3<f64>
dataset.reprojection_error;  // f64
dataset.degree_error;        // f64, spherical

// errors vs. an estimate (R, t)
dataset.compute_reprojection_error(&R, &t); // f64
dataset.compute_rotation_error(&R);         // f64
dataset.compute_translation_error(&t);      // f64
dataset.compute_degree_error(&R, &t);       // f64, spherical
```

**Local solver.** EPnP initialization and Levenberg–Marquardt refinement.

```rust
use pnp::local::LocalSolver;

let result = LocalSolver.solve(
    &dataset.world_points,   // (N×3)
    &dataset.image_points,   // (N×2) pinhole / &dataset.bearing_vectors (N×3) spherical
    &dataset.intrinsic,      // Matrix3<f64>  / None for spherical
    true,                    // refine: EPnP+LM (default) / false: EPnP only
);
result.rotation;     // Matrix3<f64>
result.translation;  // Vector3<f64>
result.runtime;      // f64, seconds
```

**Alternative solver.** Certifiable solvers for alternative errors: weighted reprojection error for pinhole; point-line error for spherical. *These are not solving the MLE.*

```rust
use pnp::alternative::AlternativeSolver;

let result = AlternativeSolver.solve(
    &dataset.world_points,   // (N×3)
    &dataset.image_points,   // (N×2) pinhole / &dataset.bearing_vectors spherical
    &dataset.intrinsic,      // Matrix3<f64>  / None for spherical
    true,                    // marginalize (default)
);
result.rotation;            // Matrix3<f64>
result.translation;         // Vector3<f64>
result.sdp_value;           // f64, SDP optimum
result.qcqp_value;          // f64, rounded QCQP value
result.optimality_gap;      // f64
result.rel_optimality_gap;  // f64
result.rank;                // usize
result.pose_recovered;      // bool, false => pose is NaN, see rank
result.eigval_ratio;        // f64
result.runtime;             // f64, seconds
```

**Global solver.** Certifiable solvers for MLE (PnP\*).

```rust
use pnp::global::{Formulation, GlobalSolver};

let result = GlobalSolver.solve(
    &dataset.world_points,   // (N×3)
    &dataset.image_points,   // (N×2) pinhole / &dataset.bearing_vectors (N×3) spherical
    &dataset.intrinsic,      // Matrix3<f64>  / None for spherical
    Formulation::Reduced,    // Reduced (default) / Chordal / Dense (ablation, does not scale)
    false,                   // prune_constraint (default)
    true,                    // refine: LM when not rank-1 (default) / false: raw rounding
);
result.rotation;            // Matrix3<f64>
result.translation;         // Vector3<f64>
result.sdp_value;           // f64, SDP optimum
result.qcqp_value;          // f64, rounded QCQP value
result.optimality_gap;      // f64
result.rel_optimality_gap;  // f64
result.rank;                // usize
result.pose_recovered;      // bool, false => pose is NaN, see rank
result.certified;           // bool, SDP converged, rank 1, and e3^T P_i z != 0 for all i
result.eigval_ratio;        // f64
result.runtime;             // f64, seconds
```

**BnB solver.** BnB method for MLE.

```rust
use pnp::bnb::{BnbOptions, BnbSolver};

let result = BnbSolver.solve(
    &dataset.world_points,   // (N×3)
    &dataset.image_points,   // (N×2) pinhole only
    &dataset.intrinsic,      // Matrix3<f64>
    &BnbOptions::default(),  // gamma: None derives the paper's gamma_max from a trivial pose
);
result.rotation;            // Matrix3<f64>
result.translation;         // Vector3<f64>
result.cost;                // f64, best objective found
result.lower_bound;         // f64, certified bound
result.optimality_gap;      // f64
result.rel_optimality_gap;  // f64
result.nodes;               // usize, branch-and-bound nodes explored
result.converged;           // bool, false if the node budget ran out
result.runtime;             // f64, seconds
```
