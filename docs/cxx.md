# :croissant: C++ API

## Install

Needs a MOSEK [license](https://www.mosek.com/license/request/) at `~/mosek/mosek.lic`, Eigen, and C++17.

```cmake
cmake_minimum_required(VERSION 3.18)
project(project_name)
set(PNP_STAR /path/to/pnp-star)

find_package(Eigen3 REQUIRED NO_MODULE)

add_executable(${PROJECT_NAME} main.cpp)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_include_directories(${PROJECT_NAME} PRIVATE ${PNP_STAR}/pnp-cxx/include)
target_link_libraries(${PROJECT_NAME} PRIVATE
    Eigen3::Eigen
    ${PNP_STAR}/target/release/libpnp_cxx.dylib)   # .so on Linux
```

Build the library with `cargo build -p pnp-cxx --release`; re-run it and rebuild your project after changing the Rust sources, moving the clone, or running `cargo clean`.

## Usage

**Dataset.** Pinhole noise is Gaussian (pixels); spherical noise is Fisher (degrees).

```cpp
#include "pnp_cxx.hpp"

pnp::PnpSyntheticDataset dataset(
    10,            // num_points
    {5.0, 35.0},   // depth_range (d_min, d_max) in metres
    180.0,         // max_rotation_degree
    5.0,           // max_translation_norm
    10.0,          // noise: pixels (pinhole) or degrees (spherical)
    "pinhole",     // "pinhole" or "spherical"
    42);           // seed

// fields (ground truth)
dataset.world_points();        // Eigen::MatrixXd (N×3)
dataset.image_points();        // Eigen::MatrixXd (N×2), pinhole
dataset.bearing_vectors();     // Eigen::MatrixXd (N×3), spherical
dataset.intrinsic();           // Eigen::Matrix3d, camera intrinsic K
dataset.rotation();            // Eigen::Matrix3d
dataset.translation();         // Eigen::Vector3d
dataset.reprojection_error();  // double
dataset.degree_error();        // double, spherical

// errors vs. an estimate (R, t)
dataset.compute_reprojection_error(R, t);  // double
dataset.compute_rotation_error(R);         // double
dataset.compute_translation_error(t);      // double
dataset.compute_degree_error(R, t);        // double, spherical
```

**Local solver.** EPnP initialization and Levenberg–Marquardt refinement.

```cpp
#include "pnp_cxx.hpp"

auto result = pnp::LocalSolver().solve(
    dataset.world_points(),   // (N×3)
    dataset.image_points(),   // (N×2) pinhole / dataset.bearing_vectors() (N×3) spherical
    dataset.intrinsic(),      // Matrix3d      / Eigen::Matrix3d::Identity() spherical
    /*refine=*/true);         // EPnP+LM (default) / false: EPnP only
result.rotation;     // Eigen::Matrix3d
result.translation;  // Eigen::Vector3d
result.runtime;      // double, seconds
```

**Alternative solver.** Certifiable solvers for alternative errors: weighted reprojection error for pinhole; point-line error for spherical. *These are not solving the MLE.*

```cpp
#include "pnp_cxx.hpp"

auto result = pnp::AlternativeSolver().solve(
    dataset.world_points(),       // (N×3)
    dataset.image_points(),       // (N×2) pinhole / dataset.bearing_vectors() spherical
    dataset.intrinsic(),          // Matrix3d      / Eigen::Matrix3d::Identity() spherical
    /*marginalize=*/true);        // default
result.rotation;            // Eigen::Matrix3d
result.translation;         // Eigen::Vector3d
result.sdp_value;           // double, SDP optimum
result.qcqp_value;          // double, rounded QCQP value
result.optimality_gap;      // double
result.rel_optimality_gap;  // double
result.rank;                // int
result.pose_recovered;      // bool, false => pose is NaN, see rank
result.certified;           // bool, rank 1 only -- the surrogate cost is not the MLE
result.eigval_ratio;        // double
result.runtime;             // double, seconds
```

**Global solver.** Certifiable solvers for MLE (PnP\*).

```cpp
#include "pnp_cxx.hpp"

auto result = pnp::GlobalSolver().solve(
    dataset.world_points(),             // (N×3)
    dataset.image_points(),             // (N×2) pinhole / dataset.bearing_vectors() spherical
    dataset.intrinsic(),                // Matrix3d      / Eigen::Matrix3d::Identity() spherical
    pnp::Formulation::Reduced,          // Reduced (default) / Chordal / Dense (ablation, does not scale)
    /*prune_constraint=*/false,         // default
    /*refine=*/true);                   // LM when not rank-1 (default) / false: raw rounding
result.rotation;            // Eigen::Matrix3d
result.translation;         // Eigen::Vector3d
result.sdp_value;           // double, SDP optimum
result.qcqp_value;          // double, rounded QCQP value
result.optimality_gap;      // double
result.rel_optimality_gap;  // double
result.rank;                // int
result.pose_recovered;      // bool, false => pose is NaN, see rank
result.certified;           // bool, SDP converged, rank 1, and e3^T P_i z != 0 for all i
result.eigval_ratio;        // double
result.runtime;             // double, seconds
```

**BnB solver.** BnB method for MLE.

```cpp
#include "pnp_cxx.hpp"

auto result = pnp::BnbSolver().solve(
    dataset.world_points(),             // (N×3)
    dataset.image_points(),             // (N×2) pinhole only
    dataset.intrinsic(),                // Matrix3d
    /*min_depth=*/-1.0,                 // <= 0 uses the paper's t3^low = 1 world unit
    /*gamma=*/-1.0,                     // <= 0 derives the paper's gamma_max from a trivial pose
    /*eps=*/-1.0, /*max_nodes=*/0);     // <= 0 uses the defaults
result.rotation;            // Eigen::Matrix3d
result.translation;         // Eigen::Vector3d
result.cost;                // double, best objective found
result.lower_bound;         // double, certified bound
result.optimality_gap;      // double
result.rel_optimality_gap;  // double
result.nodes;               // std::size_t, branch-and-bound nodes explored
result.converged;           // bool, false if the node budget ran out
result.runtime;             // double, seconds
```
