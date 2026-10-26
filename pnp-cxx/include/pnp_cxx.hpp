// C++ shim over the PnP core's C ABI. Wraps the plain-C functions in
// `pnp_cxx.h` into Eigen-typed solver classes under namespace `pnp`.
#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "pnp_cxx.h"

namespace pnp {

struct LocalSolverResult {
    Eigen::Matrix3d rotation;
    Eigen::Vector3d translation;
    double runtime{0.0};
};

struct BnbSolverResult {
    Eigen::Matrix3d rotation;
    Eigen::Vector3d translation;
    double cost{0.0};         // best objective found
    double lower_bound{0.0};  // certified bound
    double optimality_gap{0.0};
    double rel_optimality_gap{0.0};
    std::size_t nodes{0};
    bool converged{false};
    double runtime{0.0};
};

struct SdpSolverResult {
    Eigen::Matrix3d rotation;
    Eigen::Vector3d translation;
    double sdp_value{0.0};
    double qcqp_value{0.0};  // rounded QCQP value (upper bound)
    double optimality_gap{0.0};
    double rel_optimality_gap{0.0};
    int rank{0};
    double eigval_ratio{0.0};
    bool pose_recovered{true};  // false => the pose is NaN, see rank
    // GlobalSolver: rank 1 and e3^T P_i z != 0 for all i, so the pose is the
    // certified global MLE. AlternativeSolver: rank 1 of its own relaxation
    // only -- that cost is a surrogate, so this is not an MLE certificate.
    bool certified{false};
    double runtime{0.0};
};

namespace detail {

inline std::vector<double> row_major(const Eigen::MatrixXd& m) {
    std::vector<double> v((size_t)m.rows() * m.cols());
    for (int i = 0; i < m.rows(); ++i)
        for (int j = 0; j < m.cols(); ++j) v[(size_t)i * m.cols() + j] = m(i, j);
    return v;
}

inline std::vector<double> row_major(const Eigen::Matrix3d& m) {
    std::vector<double> v(9);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) v[(size_t)i * 3 + j] = m(i, j);
    return v;
}

inline Eigen::Matrix3d to_rot(const double r[9]) {
    Eigen::Matrix3d R;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) R(i, j) = r[i * 3 + j];
    return R;
}

inline Eigen::Vector3d to_vec(const double t[3]) { return Eigen::Vector3d(t[0], t[1], t[2]); }

inline Eigen::MatrixXd to_mat(const std::vector<double>& buf, int rows, int cols) {
    Eigen::MatrixXd m(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) m(i, j) = buf[(size_t)i * cols + j];
    return m;
}

}  // namespace detail

// Synthetic dataset with known ground truth (RAII wrapper over the opaque C
// handle). Mirrors the Rust/Python PnpSyntheticDataset.
class PnpSyntheticDataset {
   public:
    PnpSyntheticDataset(int num_points, std::pair<double, double> depth_range,
                        double max_rotation_degree, double max_translation_norm, double noise,
                        const std::string& camera_model, uint64_t seed)
        : pinhole_(camera_model == "pinhole"), n_(num_points) {
        handle_ = pnp_dataset_create((uintptr_t)num_points, depth_range.first, depth_range.second,
                                     max_rotation_degree, max_translation_norm, noise,
                                     pinhole_ ? 0u : 1u, seed);
    }
    ~PnpSyntheticDataset() { pnp_dataset_free(handle_); }
    PnpSyntheticDataset(const PnpSyntheticDataset&) = delete;
    PnpSyntheticDataset& operator=(const PnpSyntheticDataset&) = delete;

    int num_points() const { return n_; }
    bool is_pinhole() const { return pinhole_; }

    Eigen::MatrixXd world_points() const { return fetch(&pnp_dataset_world_points, n_, 3); }
    Eigen::MatrixXd image_points() const { return fetch(&pnp_dataset_image_points, n_, 2); }
    Eigen::MatrixXd bearing_vectors() const { return fetch(&pnp_dataset_bearing_vectors, n_, 3); }

    Eigen::Matrix3d intrinsic() const {
        double b[9];
        pnp_dataset_intrinsic(handle_, b);
        return detail::to_rot(b);
    }
    Eigen::Matrix3d rotation() const {
        double b[9];
        pnp_dataset_rotation(handle_, b);
        return detail::to_rot(b);
    }
    Eigen::Vector3d translation() const {
        double b[3];
        pnp_dataset_translation(handle_, b);
        return detail::to_vec(b);
    }
    double reprojection_error() const { return pnp_dataset_reprojection_error(handle_); }
    double degree_error() const { return pnp_dataset_degree_error(handle_); }

    double compute_reprojection_error(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) const {
        auto r = detail::row_major(R);
        double tt[3] = {t[0], t[1], t[2]};
        return pnp_dataset_compute_reprojection_error(handle_, r.data(), tt);
    }
    double compute_rotation_error(const Eigen::Matrix3d& R) const {
        auto r = detail::row_major(R);
        return pnp_dataset_compute_rotation_error(handle_, r.data());
    }
    double compute_translation_error(const Eigen::Vector3d& t) const {
        double tt[3] = {t[0], t[1], t[2]};
        return pnp_dataset_compute_translation_error(handle_, tt);
    }
    double compute_degree_error(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) const {
        auto r = detail::row_major(R);
        double tt[3] = {t[0], t[1], t[2]};
        return pnp_dataset_compute_degree_error(handle_, r.data(), tt);
    }

   private:
    using Accessor = void (*)(const ::PnpSyntheticDataset*, double*);
    Eigen::MatrixXd fetch(Accessor f, int rows, int cols) const {
        std::vector<double> buf((size_t)rows * cols);
        f(handle_, buf.data());
        return detail::to_mat(buf, rows, cols);
    }

    bool pinhole_;
    int n_;
    ::PnpSyntheticDataset* handle_;
};

class LocalSolver {
   public:
    LocalSolverResult solve(const Eigen::MatrixXd& world_points,
                            const Eigen::MatrixXd& observations,
                            const Eigen::Matrix3d& intrinsic = Eigen::Matrix3d::Identity(),
                            bool refine = true) const {
        auto w = detail::row_major(world_points);
        auto o = detail::row_major(observations);
        auto k = detail::row_major(intrinsic);
        PnpLocalResult r;
        pnp_local_solve(w.data(), (uintptr_t)world_points.rows(), o.data(),
                        (uintptr_t)observations.cols(), k.data(), refine, &r);
        return {detail::to_rot(r.pose.rotation), detail::to_vec(r.pose.translation), r.runtime};
    }
};

class AlternativeSolver {
   public:
    SdpSolverResult solve(const Eigen::MatrixXd& world_points, const Eigen::MatrixXd& observations,
                          const Eigen::Matrix3d& intrinsic = Eigen::Matrix3d::Identity(),
                          bool marginalize = true) const {
        auto w = detail::row_major(world_points);
        auto o = detail::row_major(observations);
        auto k = detail::row_major(intrinsic);
        PnpSdpResult r;
        pnp_alternative_solve(w.data(), (uintptr_t)world_points.rows(), o.data(),
                              (uintptr_t)observations.cols(), k.data(), marginalize, &r);
        return unpack(r);
    }

   private:
    static SdpSolverResult unpack(const PnpSdpResult& r) {
        return {detail::to_rot(r.pose.rotation),
                detail::to_vec(r.pose.translation),
                r.sdp_value,
                r.qcqp_value,
                r.optimality_gap,
                r.rel_optimality_gap,
                r.rank,
                r.eigval_ratio,
                r.pose_recovered,
                r.certified,
                r.runtime};
    }
};

/// Which SDP formulation to solve (paper section 6).
/// ! Dense is an ablation only -- the block grows with N.
enum class Formulation {
    Dense = PnpFormulation_Dense,
    Chordal = PnpFormulation_Chordal,
    Reduced = PnpFormulation_Reduced,
};

/// Branch-and-bound baseline (Olsson et al., ICPR'06). Pinhole only: the bound
/// needs the perspective denominator to be affine in the lifted variable.
class BnbSolver {
   public:
    /// `min_depth <= 0` uses the paper's t3^low = 1 world unit; `gamma <= 0`
    /// derives gamma_max from a trivial pose.
    BnbSolverResult solve(const Eigen::MatrixXd& world_points, const Eigen::MatrixXd& image_points,
                          const Eigen::Matrix3d& intrinsic = Eigen::Matrix3d::Identity(),
                          double min_depth = -1.0, double gamma = -1.0, double eps = -1.0,
                          std::size_t max_nodes = 0) const {
        auto w = detail::row_major(world_points);
        auto o = detail::row_major(image_points);
        auto k = detail::row_major(intrinsic);
        PnpBnbResult r;
        pnp_bnb_solve(w.data(), (uintptr_t)world_points.rows(), o.data(), k.data(), min_depth,
                      gamma, eps, (uintptr_t)max_nodes, &r);
        return {detail::to_rot(r.pose.rotation),
                detail::to_vec(r.pose.translation),
                r.cost,
                r.lower_bound,
                r.optimality_gap,
                r.rel_optimality_gap,
                (std::size_t)r.nodes,
                r.converged,
                r.runtime};
    }
};

class GlobalSolver {
   public:
    SdpSolverResult solve(const Eigen::MatrixXd& world_points, const Eigen::MatrixXd& observations,
                          const Eigen::Matrix3d& intrinsic = Eigen::Matrix3d::Identity(),
                          Formulation formulation = Formulation::Reduced,
                          bool prune_constraint = false, bool refine = true) const {
        auto w = detail::row_major(world_points);
        auto o = detail::row_major(observations);
        auto k = detail::row_major(intrinsic);
        PnpSdpResult r;
        pnp_global_solve(w.data(), (uintptr_t)world_points.rows(), o.data(),
                         (uintptr_t)observations.cols(), k.data(), (PnpFormulation)formulation,
                         prune_constraint, refine, &r);
        return AlternativeSolver_unpack(r);
    }

   private:
    static SdpSolverResult AlternativeSolver_unpack(const PnpSdpResult& r) {
        return {detail::to_rot(r.pose.rotation),
                detail::to_vec(r.pose.translation),
                r.sdp_value,
                r.qcqp_value,
                r.optimality_gap,
                r.rel_optimality_gap,
                r.rank,
                r.eigval_ratio,
                r.pose_recovered,
                r.certified,
                r.runtime};
    }
};

}  // namespace pnp
