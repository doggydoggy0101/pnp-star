use std::f64::consts::PI;

use nalgebra::{DMatrix, Matrix3, Rotation3, Unit, Vector3};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

use crate::utils::normalize_pinhole;

pub struct PnpSyntheticDataset {
    pub num_points: usize,
    pub camera_model: String,
    pub seed: u64,

    pub intrinsic: Matrix3<f64>,
    pub rotation: Matrix3<f64>,
    pub translation: Vector3<f64>,
    pub world_points: DMatrix<f64>,    // (N, 3)
    pub image_points: DMatrix<f64>,    // (N, 2) pinhole
    pub bearing_vectors: DMatrix<f64>, // (N, 3) spherical
    pub reprojection_error: f64,
    pub degree_error: f64, // spherical only
}

fn set_row3(m: &mut DMatrix<f64>, i: usize, v: &Vector3<f64>) {
    m[(i, 0)] = v[0];
    m[(i, 1)] = v[1];
    m[(i, 2)] = v[2];
}

fn row3(m: &DMatrix<f64>, i: usize) -> Vector3<f64> {
    Vector3::new(m[(i, 0)], m[(i, 1)], m[(i, 2)])
}

impl PnpSyntheticDataset {
    /// Default synthetic intrinsic (EPnP convention): fx=fy=800, cx=320, cy=240.
    /// 7Scenes for reference: fx=fy=580, cx=320, cy=240.
    pub fn default_intrinsic() -> Matrix3<f64> {
        Matrix3::new(800.0, 0.0, 320.0, 0.0, 800.0, 240.0, 0.0, 0.0, 1.0)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        num_points: usize,
        depth_range: (f64, f64),
        max_rotation_degree: f64,
        max_translation_norm: f64,
        noise: f64,
        camera_model: &str,
        seed: u64,
    ) -> Self {
        Self::new_with_intrinsic(
            num_points,
            depth_range,
            max_rotation_degree,
            max_translation_norm,
            noise,
            camera_model,
            seed,
            Self::default_intrinsic(),
        )
    }

    /// Same as `new`, with a caller-supplied intrinsic. `intrinsic` must be upper
    /// triangular with third row e3^T; fx != fy and nonzero skew are supported.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_intrinsic(
        num_points: usize,
        depth_range: (f64, f64),
        max_rotation_degree: f64,
        max_translation_norm: f64,
        noise: f64,
        camera_model: &str,
        seed: u64,
        intrinsic: Matrix3<f64>,
    ) -> Self {
        assert!(
            camera_model == "pinhole" || camera_model == "spherical",
            "camera_model must be 'pinhole' or 'spherical'"
        );

        let (rotation, translation) =
            Self::generate_extrinsic(seed, max_rotation_degree, max_translation_norm);

        let mut ds = PnpSyntheticDataset {
            num_points,
            camera_model: camera_model.to_string(),
            seed,
            intrinsic,
            rotation,
            translation,
            world_points: DMatrix::zeros(num_points, 3),
            image_points: DMatrix::zeros(0, 0),
            bearing_vectors: DMatrix::zeros(0, 0),
            reprojection_error: 0.0,
            degree_error: 0.0,
        };

        let clean = if camera_model == "pinhole" {
            ds.generate_from_image(depth_range.0, depth_range.1)
        } else {
            ds.generate_from_sphere(depth_range.0, depth_range.1)
        };
        ds.init_projections(&clean, noise);
        ds
    }

    fn generate_extrinsic(
        seed: u64,
        max_rotation_degree: f64,
        max_translation_norm: f64,
    ) -> (Matrix3<f64>, Vector3<f64>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let randn = Normal::new(0.0, 1.0).unwrap();

        let mut axis = Vector3::new(
            randn.sample(&mut rng),
            randn.sample(&mut rng),
            randn.sample(&mut rng),
        );
        axis.normalize_mut();
        let angle = rng.gen_range(0.0..1.0) * max_rotation_degree * PI / 180.0;
        let rot = Rotation3::new(axis * angle).into_inner();

        let mut t_dir = Vector3::new(
            randn.sample(&mut rng),
            randn.sample(&mut rng),
            randn.sample(&mut rng),
        );
        t_dir.normalize_mut();
        let t_norm = rng.gen_range(0.0..1.0) * max_translation_norm;

        (rot, t_dir * t_norm)
    }

    fn generate_from_image(&mut self, d_min: f64, d_max: f64) -> DMatrix<f64> {
        let (cx, cy) = (self.intrinsic[(0, 2)], self.intrinsic[(1, 2)]);
        let (fx, fy) = (self.intrinsic[(0, 0)], self.intrinsic[(1, 1)]);
        let sk = self.intrinsic[(0, 1)]; // skew
        let (w, h) = (2.0 * cx, 2.0 * cy);

        let mut rng = StdRng::seed_from_u64(self.seed);
        let rt = self.rotation.transpose();

        let mut clean_pixels = DMatrix::zeros(self.num_points, 2);
        for i in 0..self.num_points {
            let u = rng.gen_range(0.0..w);
            let v = rng.gen_range(0.0..h);
            let d = rng.gen_range(d_min..d_max);

            // p_cam = d * K^{-1} (u, v, 1), K upper triangular
            let qy = (v - cy) / fy;
            let qx = (u - cx - sk * qy) / fx;
            let p_cam = Vector3::new(d * qx, d * qy, d);
            let p_world = rt * (p_cam - self.translation);
            set_row3(&mut self.world_points, i, &p_world);
            clean_pixels[(i, 0)] = u;
            clean_pixels[(i, 1)] = v;
        }
        clean_pixels
    }

    fn generate_from_sphere(&mut self, d_min: f64, d_max: f64) -> DMatrix<f64> {
        let mut rng = StdRng::seed_from_u64(self.seed);
        let rt = self.rotation.transpose();

        let mut clean_bv = DMatrix::zeros(self.num_points, 3);
        for i in 0..self.num_points {
            let z: f64 = rng.gen_range(-1.0..1.0);
            let phi: f64 = rng.gen_range(0.0..(2.0 * PI));
            let r = (1.0 - z * z).max(0.0).sqrt();
            let bv = Vector3::new(r * phi.cos(), r * phi.sin(), z);

            let d: f64 = rng.gen_range(d_min..d_max);
            let p_cam = d * bv;
            let p_world = rt * (p_cam - self.translation);
            set_row3(&mut self.world_points, i, &p_world);
            set_row3(&mut clean_bv, i, &bv);
        }
        clean_bv
    }

    fn init_projections(&mut self, clean: &DMatrix<f64>, noise: f64) {
        if self.camera_model == "pinhole" {
            let mut pix = clean.clone();
            if noise != 0.0 {
                let mut rng = StdRng::seed_from_u64(self.seed);
                let gauss = Normal::new(0.0, noise).unwrap();
                for n in 0..pix.nrows() {
                    for j in 0..2 {
                        pix[(n, j)] += gauss.sample(&mut rng);
                    }
                }
            }
            self.image_points = pix;
            self.bearing_vectors = self.pixel2bearing(&self.image_points);
            self.reprojection_error =
                self.compute_reprojection_error(&self.rotation, &self.translation);
        } else {
            let mut bv = clean.clone();
            if noise != 0.0 {
                let sigma_rad = noise * PI / 180.0;
                let kappa = 1.0 / (sigma_rad * sigma_rad);
                let mut rng = StdRng::seed_from_u64(self.seed);
                for n in 0..bv.nrows() {
                    let sampled = Self::sample_vmf(&row3(&bv, n), kappa, &mut rng);
                    set_row3(&mut bv, n, &sampled);
                }
            }
            self.bearing_vectors = bv;
            self.reprojection_error =
                self.compute_reprojection_error(&self.rotation, &self.translation);
            self.degree_error = self.compute_degree_error(&self.rotation, &self.translation);
        }
    }

    fn pixel2bearing(&self, pix: &DMatrix<f64>) -> DMatrix<f64> {
        let (q, _) = normalize_pinhole(pix, &self.intrinsic);
        let mut bv = DMatrix::zeros(pix.nrows(), 3);
        for i in 0..pix.nrows() {
            let v = Vector3::new(q[(i, 0)], q[(i, 1)], 1.0).normalize();
            set_row3(&mut bv, i, &v);
        }
        bv
    }

    fn sample_vmf(mean: &Vector3<f64>, kappa: f64, rng: &mut StdRng) -> Vector3<f64> {
        let b = -kappa + (kappa * kappa + 1.0).sqrt();
        let x0 = (1.0 - b) / (1.0 + b);
        let c = kappa * x0 + 2.0 * (1.0 - x0 * x0).ln();

        let w = loop {
            let z = rng.gen_range(0.0..1.0);
            let w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z);
            let u: f64 = rng.gen_range(0.0..1.0);
            let val = (1.0 - x0 * w).max(1e-15);
            if kappa * w + 2.0 * val.ln() - c >= u.ln() {
                break w;
            }
        };

        let phi = rng.gen_range(0.0..1.0) * 2.0 * PI;
        let sin_theta = (1.0 - w * w).max(0.0).sqrt();
        let v = Vector3::new(sin_theta * phi.cos(), sin_theta * phi.sin(), w);

        let ez = Vector3::new(0.0, 0.0, 1.0);
        let dot = ez.dot(mean).clamp(-1.0, 1.0);
        if dot > 1.0 - 1e-10 {
            return v;
        }
        if dot < -1.0 + 1e-10 {
            return Rotation3::from_axis_angle(&Vector3::x_axis(), PI) * v;
        }
        Rotation3::from_axis_angle(&Unit::new_normalize(ez.cross(mean)), dot.acos()) * v
    }

    fn world2camera(&self, pts: &DMatrix<f64>, r: &Matrix3<f64>, t: &Vector3<f64>) -> DMatrix<f64> {
        let mut cam = DMatrix::zeros(pts.nrows(), 3);
        for i in 0..pts.nrows() {
            set_row3(&mut cam, i, &(r * row3(pts, i) + t));
        }
        cam
    }

    fn camera2pixel(&self, cam: &DMatrix<f64>) -> DMatrix<f64> {
        let k = &self.intrinsic;
        let mut pix = DMatrix::zeros(cam.nrows(), 2);
        for i in 0..cam.nrows() {
            let hom = k * row3(cam, i);
            pix[(i, 0)] = hom[0] / hom[2];
            pix[(i, 1)] = hom[1] / hom[2];
        }
        pix
    }

    fn camera2bearing(&self, cam: &DMatrix<f64>) -> DMatrix<f64> {
        let mut bv = DMatrix::zeros(cam.nrows(), 3);
        for i in 0..cam.nrows() {
            set_row3(&mut bv, i, &row3(cam, i).normalize());
        }
        bv
    }

    pub fn compute_reprojection_error(&self, rot: &Matrix3<f64>, t: &Vector3<f64>) -> f64 {
        let cam = self.world2camera(&self.world_points, rot, t);
        let err = if self.camera_model == "pinhole" {
            self.camera2pixel(&cam) - &self.image_points
        } else {
            self.camera2bearing(&cam) - &self.bearing_vectors
        };
        err.iter().map(|e| e * e).sum()
    }

    pub fn compute_rotation_error(&self, rot: &Matrix3<f64>) -> f64 {
        let val = (rot.transpose() * self.rotation).trace().clamp(-1.0, 3.0);
        ((val - 1.0) / 2.0).clamp(-1.0, 1.0).acos().to_degrees() // degrees
    }

    pub fn compute_translation_error(&self, t: &Vector3<f64>) -> f64 {
        (t - self.translation).norm()
    }

    pub fn compute_degree_error(&self, rot: &Matrix3<f64>, t: &Vector3<f64>) -> f64 {
        let r_reform = rot.transpose();
        let t_reform = -r_reform * t;
        let mut error = 0.0;
        for i in 0..self.num_points {
            let u = row3(&self.world_points, i) - t_reform;
            let bv_i = row3(&self.bearing_vectors, i);
            let cos_i = (bv_i.dot(&(rot * u)) / u.norm()).clamp(-1.0, 1.0);
            error += cos_i.acos() * 180.0 / PI;
        }
        error / self.num_points as f64
    }
}
