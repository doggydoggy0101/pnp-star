use std::time::Instant;

use nalgebra::{DMatrix, Matrix3, Vector3};

use crate::epnp::epnp;
use crate::nonlinear::{refine_pinhole, refine_spherical, LmOptions};
use crate::utils::normalize_pinhole;

pub struct LocalSolverResult {
    pub rotation: Matrix3<f64>,
    pub translation: Vector3<f64>,
    pub runtime: f64, // seconds
}

/// EPnP initialization + Levenberg-Marquardt refinement.
#[derive(Default)]
pub struct LocalSolver;

impl LocalSolver {
    /// Solve absolute pose. `observations` is (N, 2) image points (pinhole, uses
    /// `intrinsic`) or (N, 3) unit bearing vectors (spherical; pass `None`).
    /// `refine = false` returns the EPnP estimate only (skips LM refinement).
    pub fn solve<'a>(
        &self,
        world_points: &DMatrix<f64>,
        observations: &DMatrix<f64>,
        intrinsic: impl Into<Option<&'a Matrix3<f64>>>,
        refine: bool,
    ) -> LocalSolverResult {
        let intrinsic = intrinsic.into().copied().unwrap_or_else(Matrix3::identity);
        let t0 = Instant::now();
        let n = world_points.nrows();
        let is_pinhole = observations.ncols() == 2;

        // bearings (N, 3): (u_norm, v_norm, 1) pinhole, or unit bearings spherical
        let mut bearings = DMatrix::zeros(n, 3);
        if is_pinhole {
            let (q, _) = normalize_pinhole(observations, &intrinsic);
            for i in 0..n {
                bearings[(i, 0)] = q[(i, 0)];
                bearings[(i, 1)] = q[(i, 1)];
                bearings[(i, 2)] = 1.0;
            }
        } else {
            bearings = observations.clone();
        }

        let (mut rotation, mut translation) = epnp(world_points, &bearings);

        if refine {
            let opts = LmOptions::default();
            if is_pinhole {
                refine_pinhole(
                    &mut rotation,
                    &mut translation,
                    world_points,
                    observations,
                    &intrinsic,
                    &opts,
                );
            } else {
                refine_spherical(
                    &mut rotation,
                    &mut translation,
                    world_points,
                    observations,
                    &opts,
                );
            }
        }

        LocalSolverResult {
            rotation,
            translation,
            runtime: t0.elapsed().as_secs_f64(),
        }
    }
}
