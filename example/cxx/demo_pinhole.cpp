// Pinhole demo: a 10 point scene solved by AlternativeSolver (weighted
// reprojection error), LocalSolver (MLE), BnbSolver (MLE), and GlobalSolver (MLE).

#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include "pnp_cxx.hpp"

struct Row {
    std::string name;
    std::string rank;
    std::string opt_gap;
    double reproj{0.0};
    double rot{0.0};
    double trans{0.0};
    std::string runtime;
};

static std::string fmt_sci(double v) {
    std::ostringstream ss;
    ss << std::scientific << std::setprecision(2) << v;
    return ss.str();
}

static std::string fmt_rt(double v) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(5) << v;
    return ss.str();
}

static void print_info(const pnp::PnpSyntheticDataset &data, const Eigen::Matrix3d &rotation,
                       const Eigen::Vector3d &translation, double runtime,
                       const std::string &name = "") {
    if (!name.empty()) std::cout << "\n" << name << ":\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  reprojection: " << data.compute_reprojection_error(rotation, translation)
              << "\n";
    std::cout << "  rotation:     " << data.compute_rotation_error(rotation) << "\n";
    std::cout << "  translation:  " << data.compute_translation_error(translation) << "\n";
    std::cout << "  runtime:      " << fmt_rt(runtime) << " s\n";
}

static void print_table(const std::vector<Row> &rows) {
    const int W = 95;
    std::cout << "\n" << std::string(W, '=') << "\n";
    std::cout << std::left << std::setw(20) << "Solver" << " | " << std::setw(5) << "Rank"
              << " | " << std::setw(10) << "Opt Gap" << " | " << std::setw(10) << "Reproj Err"
              << " | " << std::setw(10) << "Rot Err" << " | " << std::setw(10) << "Trans Err"
              << " | " << std::setw(10) << "Runtime(s)" << "\n";
    std::cout << std::string(W, '-') << "\n";
    for (const auto &r : rows) {
        std::ostringstream reproj, rot, trans;
        reproj << std::fixed << std::setprecision(3) << r.reproj;
        rot << std::fixed << std::setprecision(6) << r.rot;
        trans << std::fixed << std::setprecision(6) << r.trans;
        std::cout << std::left << std::setw(20) << r.name << " | " << std::setw(5) << r.rank
                  << " | " << std::setw(10) << r.opt_gap << " | " << std::setw(10) << reproj.str()
                  << " | " << std::setw(10) << rot.str() << " | " << std::setw(10) << trans.str()
                  << " | " << std::setw(10) << r.runtime << "\n";
    }
    std::cout << std::string(W, '=') << "\n";
}

int main() {
    pnp::PnpSyntheticDataset data(10, {5.0, 35.0}, 180.0, 5.0, 10.0, "pinhole", 42);
    const Eigen::MatrixXd wp = data.world_points();
    const Eigen::MatrixXd obs = data.image_points();
    const Eigen::Matrix3d K = data.intrinsic();

    std::cout << "Ground truth:\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  reprojection: " << data.reprojection_error() << "\n";

    std::vector<Row> summary;
    summary.push_back({"Ground Truth", "-", "-", data.reprojection_error(), 0.0, 0.0, "-"});

    auto wrep = pnp::AlternativeSolver().solve(wp, obs, K);
    std::cout << "\nGlobal (W-Reproj):\n";
    std::cout << "  rank:      " << wrep.rank << "\n";
    std::cout << "  rel gap:   " << fmt_sci(wrep.rel_optimality_gap) << "\n";
    print_info(data, wrep.rotation, wrep.translation, wrep.runtime);
    summary.push_back({"Global (W-Reproj)", std::to_string(wrep.rank),
                       fmt_sci(wrep.rel_optimality_gap),
                       data.compute_reprojection_error(wrep.rotation, wrep.translation),
                       data.compute_rotation_error(wrep.rotation),
                       data.compute_translation_error(wrep.translation), fmt_rt(wrep.runtime)});

    auto loc = pnp::LocalSolver().solve(wp, obs, K);
    print_info(data, loc.rotation, loc.translation, loc.runtime, "Local (MLE)");
    summary.push_back({"Local (MLE)", "-", "-",
                       data.compute_reprojection_error(loc.rotation, loc.translation),
                       data.compute_rotation_error(loc.rotation),
                       data.compute_translation_error(loc.translation), fmt_rt(loc.runtime)});

    auto bb = pnp::BnbSolver().solve(wp, obs, K);
    std::cout << "\nBnB (MLE):\n";
    std::cout << "  nodes:     " << bb.nodes << "\n";
    std::cout << "  rel gap:   " << fmt_sci(bb.rel_optimality_gap) << "\n";
    print_info(data, bb.rotation, bb.translation, bb.runtime);
    summary.push_back({"BnB (MLE)", "-", fmt_sci(bb.rel_optimality_gap),
                       data.compute_reprojection_error(bb.rotation, bb.translation),
                       data.compute_rotation_error(bb.rotation),
                       data.compute_translation_error(bb.translation), fmt_rt(bb.runtime)});

    auto ex = pnp::GlobalSolver().solve(wp, obs, K);
    std::cout << "\nGlobal (MLE):\n";
    std::cout << "  rank:      " << ex.rank << "\n";
    std::cout << "  rel gap:   " << fmt_sci(ex.rel_optimality_gap) << "\n";
    print_info(data, ex.rotation, ex.translation, ex.runtime);
    summary.push_back({"Global (MLE)", std::to_string(ex.rank), fmt_sci(ex.rel_optimality_gap),
                       data.compute_reprojection_error(ex.rotation, ex.translation),
                       data.compute_rotation_error(ex.rotation),
                       data.compute_translation_error(ex.translation), fmt_rt(ex.runtime)});

    print_table(summary);
    return 0;
}
