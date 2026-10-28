# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Camera pose estimation (Perspective-n-Point) via SDP relaxation, written in Rust with C++ and Python
bindings. Two camera models — **pinhole** and **spherical** — handled through a unified bearing-vector
formulation. **MOSEK** is the SDP backend.

## Build

```bash
# Rust core:
cargo build --release
cargo test --release                            # ~6 s; ~35 s in debug
cargo run --release --example demo_pinhole      # / demo_spherical

# Python bindings (PyO3, built via maturin — not part of the cargo workspace):
maturin develop --release -m pnp-python/Cargo.toml
python3 example/python/demo_pinhole.py          # / demo_spherical.py

# C++ bindings (cdylib + header-only wrapper):
cargo build -p pnp-cxx --release
cmake -B example/cxx/build example/cxx && cmake --build example/cxx/build
./example/cxx/build/demo_pinhole                # / demo_spherical
```

MOSEK: the `mosek` crate downloads its own library at build time — the crate version *is* the
MOSEK version, so the dependency is pinned to one minor series rather than left to float. No MOSEK
installation is needed, only a license at `~/mosek/mosek.lic`. `build.rs` in `pnp/` and `pnp-cxx/`
bakes rpaths to it so binaries run without a loader-path variable. ! `mosek` is also a
build-dependency of `pnp`, only so Cargo runs its download before `pnp/build.rs` looks for it (a fresh
build otherwise bakes no rpath). `pnp` re-exports the directory as `DEP_PNP_MOSEK_RPATH`
(`links = "pnp"`) for downstream crates, whose own `build.rs` must emit it (see `docs/rust.md`).

## Release check

Run before publishing, ideally on a machine set up only from the README. Work in a scratch
directory, never in this checkout, and run every binary with no loader-path variable
(`env -u DYLD_LIBRARY_PATH` / `env -u LD_LIBRARY_PATH`). Report pass/fail per item with the exact
error; change docs or code only after the user agrees.

**1. Fresh clone, README only.** `git clone` into the scratch directory (to include uncommitted
changes instead: `git ls-files > files.txt && rsync -a --files-from=files.txt . <scratch>/`), so
`target/` starts empty. Follow the README's Set-up and Examples sections verbatim.
- All six demos (Rust, C++, Python × pinhole, spherical) print a `Global (MLE)` row with rank 1 and
  an opt gap near 6e-09 (pinhole) / 7e-09 (spherical), identical across the three languages up to
  MOSEK jitter. Pinhole also prints a `BnB (MLE)` row.
- `cargo test --release`: 26 passed.
- One MOSEK download: exactly one `target/release/build/mosek-*/out/mosek`.
- Binaries carry an rpath to that download: `otool -l <bin> | grep -A2 LC_RPATH` (macOS) /
  `readelf -d <bin> | grep -i runpath` (Linux). A missing one fails at start with
  `Library not loaded: @rpath/libmosek64…` / `libmosek64.so…: cannot open shared object file`.
- Known pitfalls: Python must be ≤ 3.13 (PyO3 0.23; the README pins 3.13 through conda); maturin
  refuses to run with both a venv and a conda env active; on Linux maturin needs `patchelf`.

**2. Docs usage, one consumer per language, outside the clone.**
- Python (`docs/python.md`): exec every `python` block in order, defining `R, t` for the error
  calls and re-creating a pinhole dataset before the solver blocks (the dataset block ends on a
  spherical one). Then, per solver: public attributes equal the documented list exactly, types match
  the comments, `__text_signature__` defaults match the documented ones, and spherical solves accept
  `None` for the intrinsic.
- Rust (`docs/rust.md`): a new crate depending on the clone's `pnp` by path, with the doc's
  `build.rs` snippet verbatim and a fresh `CARGO_TARGET_DIR`. `main.rs` makes every documented call
  and pins each documented field type (`let _: T = …`), plus spherical with `None`. Must build and
  run with no loader-path variable. It downloads MOSEK twice (no shared build-override), which is
  expected.
- C++ (`docs/cxx.md`): a CMake project whose `CMakeLists.txt` is the doc's snippet verbatim, with
  `PNP_STAR` set to the clone. `main.cpp` makes every documented call and pins each type with
  `static_assert(std::is_same_v<std::decay_t<decltype(expr)>, T>)`. Configures with no CMake
  warning and runs with no loader-path variable.

## Formatting

- **Rust**: `cargo fmt` after edits.
- `pnp-python/src/lib.rs` (PyO3) and `pnp-cxx/src/lib.rs` (FFI) are **Rust** — `cargo fmt`, not ruff.
- **Python** (`experiments/**/*.py`, `assets/*/plot_*.py`): `ruff format`.
- **C++** demos (`example/cxx/*.cpp`) and headers (`pnp-cxx/include/*.hpp`): `clang-format -i`.

## Architecture

Cargo workspace:
- `pnp/` — the Rust core library (all the math lives here).
- `pnp-cxx/` — C ABI (`src/lib.rs`) + header-only C++ wrapper (`include/pnp_cxx.hpp`). Both solvers
  share one `SdpSolverResult`, so `certified` exists on an `AlternativeSolver` result too, where it
  means rank 1 of that surrogate relaxation — not an MLE certificate.
- `pnp-python/` — PyO3 bindings (excluded from the workspace; built by maturin). Distribution name
  is `pnp-star`, import name is `pnp`.
- `example/{rust,python,cxx}/` — demos.
- `docs/{rust,cxx,python}.md` — the API reference; the README covers only set-up and the demos.
- `assets/{logo,star}/` — tracked figures and the scripts that generate them.

**Solvers** (`pnp/src/`):
- `LocalSolver` (`local.rs`) — EPnP initialization + LM refinement; solves the MLE *locally*.
  `refine=false` returns the EPnP estimate only. EPnP (`epnp.rs`) is a faithful OpenGV port
  (control points from the 4 smallest singular vectors, betas from the rigidity constraints, 3
  hypotheses chosen by reprojection error). LM (`nonlinear.rs`) is analytic-Jacobian on SE(3).
- `AlternativeSolver` (`alternative.rs`) — Shor SDP on a *surrogate* cost (weighted-reprojection for
  pinhole, point-line for spherical). **Not the MLE.** Its relaxation is a 10×10 SDP over SO(3)
  whose size is independent of N, which is why it is reliably tight — tightness bought by changing
  the objective.
- `GlobalSolver` (`global.rs`) — multi-clique SDP with quotient-ring reduction; the certified global
  MLE. `certified` (`rank == 1` **and** `e3^T P_i z != 0` for all i, pinhole) ⇒ the rounded solution
  is the global optimum of Problem 1. Otherwise it rounds and refines with LM (`refine=false`
  returns the raw rounding). ! Rank 1 alone is **not** the certificate: it certifies the QCQP, and
  the depth condition is what transfers that to Problem 1. `pose_recovered == false` is separate
  again — the hub block could not be dehomogenized (`|h| < H_MIN`) and the pose is NaN; it can only
  happen when rank > 1. The `Formulation` enum selects the paper's §6 ladder — `Reduced`
  (default, 34/46) / `Chordal` (37/49) / `Dense` (one block of 13+24N / 13+36N; ablation only,
  stalls past N ≈ 15).
- `BnbSolver` (`bnb.rs`) — **baseline, not ours**: branch-and-bound over the quaternion box after
  Olsson/Kahl/Oskarsson ICPR'06, the only prior method globally optimizing the same pixel
  reprojection error. Pinhole only. Its own MOSEK SOCP path (McCormick envelopes + rotated cones),
  independent of `SdpProblem`.

**Backend**: `sdp.rs`. Chordal and Reduced build a `StarSdp` (cliques sharing the 13×13 hub) and
solve it with SDP* (`sdp-star`, a git dependency); Dense and `AlternativeSolver` build an
`SdpProblem` and solve it with MOSEK (`mosek.rs`). SDP* reports its dual objective as `sdp_value`
(the lower bound) and a convergence flag, which `certified` also requires.

**Shared** (`pnp/src/`): `data.rs` (`PnpSyntheticDataset`), `utils.rs` (`DataMatrices`,
`marginalize`/`lift`, rotation constraint matrices), `linalg.rs` (`row3`, `skew`, `exp_so3`,
`project_to_so3`, rot↔vec).

## Key conventions

- **MLE objective is the pixel reprojection error** (paper eq. 7, `Pi = [I2 0] K P z / e3^T P z`).
  Solvers evaluate it in normalized coords with the weight `W = Khat/sqrt(det Khat)` from
  `utils::normalize_pinhole`, so the cost is the pixel error over `det(Khat)` — same minimizer,
  better conditioned. `Local` and `Global` minimize the *same* objective for **any** intrinsic,
  `fx != fy` and skew included; `W = I2` when `fx = fy` and skew is 0. Same objective as OpenCV's
  `solvePnPRefineLM`. Spherical is unweighted (bearing residual).
- Synthetic intrinsic defaults to the EPnP convention `fx=fy=800, cx=320, cy=240`
  (`PnpSyntheticDataset::default_intrinsic`); pass your own via `new_with_intrinsic`. Dataset noise
  is added in pixels, matching paper eq. (6).
- **Recentre and rescale** in `global.rs` and `alternative.rs`: world points are recentred on their
  centroid and rescaled by the cloud radius before the SDP, then `t` is mapped back (`R` is
  unchanged, so the relaxation and its optimal value are invariant). Numerically it matters a lot —
  a scene far from the world origin drives `h` toward zero, since a rank-1 solution has
  `h = 1/sqrt(3 + ||t||^2)` in the normalized frame. `scale == 0` (coincident points) returns the
  no-pose result rather than dividing by zero.
- Both SDP solvers count rank as eigenvalues `λ > RANK_TOL = 1e-5` (`sdp.rs`), in the recentred and
  rescaled frame. Rank-1 solutions sit at `λ₂ ≲ 3e-7` (MOSEK noise), so the gap is wide.
- `compute_rotation_error` returns **degrees**.
- Rotation vec is column-major: `vec[j*3+k] = R[k,j]`. `kron(p, I3)` is (3,9): `K[k, j*3+l] = p[j]·δ_{k,l}`.

## Comment style

Short, inline. Math notation inline (e.g. `// res = x/z - q`), `// !` for numerical warnings, `// TODO`.
No decorative separators.

## Research notes

`research/` (gitignored, local only) holds the paper draft (`pnp_ijcv.pdf`), the section notes, the
stability proof (`proof/`), and the experiment plan (`experiments.md`). `experiments/` (also
gitignored) holds the paper's experiment drivers, one folder per experiment plus `common.py`, and
the real-data sets. Neither is part of the shipped library; don't reference them from tracked code
or docs.
