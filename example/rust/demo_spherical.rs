//! Spherical demo: a 10 point scene solved by AlternativeSolver (point-line
//! error), LocalSolver (MLE), and GlobalSolver (MLE).

use pnp::data::PnpSyntheticDataset;
use pnp::{alternative, global, local};

struct Row {
    name: String,
    rank: String,
    opt_gap: String,
    reproj: f64,
    deg: f64,
    rot: f64,
    trans: f64,
    runtime: String,
}

fn fmt_sci(v: f64) -> String {
    let s = format!("{v:.2e}");
    match s.find('e') {
        Some(p) => {
            let (m, e) = s.split_at(p);
            let exp = &e[1..];
            let (sign, digits) = match exp.strip_prefix('-') {
                Some(r) => ("-", r),
                None => ("+", exp),
            };
            format!("{m}e{sign}{digits:0>2}")
        }
        None => s,
    }
}

fn fmt_rt(v: f64) -> String {
    format!("{v:.5}")
}

fn print_info(
    data: &PnpSyntheticDataset,
    rot: &nalgebra::Matrix3<f64>,
    t: &nalgebra::Vector3<f64>,
    runtime: f64,
    name: &str,
) {
    if !name.is_empty() {
        println!("\n{name}:");
    }
    println!(
        "  reprojection: {:.6}",
        data.compute_reprojection_error(rot, t)
    );
    println!("  degree:       {:.6}", data.compute_degree_error(rot, t));
    println!("  rotation:     {:.6}", data.compute_rotation_error(rot));
    println!("  translation:  {:.6}", data.compute_translation_error(t));
    println!("  runtime:      {} s", fmt_rt(runtime));
}

fn print_table(rows: &[Row]) {
    const W: usize = 110;
    println!("\n{}", "=".repeat(W));
    println!(
        "{:<20} | {:<5} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10}",
        "Solver", "Rank", "Opt Gap", "Reproj Err", "Deg Err", "Rot Err", "Trans Err", "Runtime(s)"
    );
    println!("{}", "-".repeat(W));
    for r in rows {
        println!(
            "{:<20} | {:<5} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10}",
            r.name,
            r.rank,
            r.opt_gap,
            format!("{:.6}", r.reproj),
            format!("{:.6}", r.deg),
            format!("{:.6}", r.rot),
            format!("{:.6}", r.trans),
            r.runtime
        );
    }
    println!("{}", "=".repeat(W));
}

fn main() {
    let data = PnpSyntheticDataset::new(10, (5.0, 35.0), 180.0, 5.0, 1.0, "spherical", 42);
    let obs = &data.bearing_vectors;

    println!("Ground truth:");
    println!("  reprojection: {:.6}", data.reprojection_error);
    println!("  degree:       {:.6}", data.degree_error);

    let mut summary = vec![Row {
        name: "Ground Truth".into(),
        rank: "-".into(),
        opt_gap: "-".into(),
        reproj: data.reprojection_error,
        deg: data.degree_error,
        rot: 0.0,
        trans: 0.0,
        runtime: "-".into(),
    }];

    let ptl = alternative::AlternativeSolver.solve(&data.world_points, obs, None, true);
    println!("\nGlobal (Pt-Line):");
    println!("  rank:      {}", ptl.rank);
    println!("  rel gap:   {}", fmt_sci(ptl.rel_optimality_gap));
    print_info(&data, &ptl.rotation, &ptl.translation, ptl.runtime, "");
    summary.push(Row {
        name: "Global (Pt-Line)".into(),
        rank: ptl.rank.to_string(),
        opt_gap: fmt_sci(ptl.rel_optimality_gap),
        reproj: data.compute_reprojection_error(&ptl.rotation, &ptl.translation),
        deg: data.compute_degree_error(&ptl.rotation, &ptl.translation),
        rot: data.compute_rotation_error(&ptl.rotation),
        trans: data.compute_translation_error(&ptl.translation),
        runtime: fmt_rt(ptl.runtime),
    });

    let loc = local::LocalSolver.solve(&data.world_points, obs, None, true);
    print_info(
        &data,
        &loc.rotation,
        &loc.translation,
        loc.runtime,
        "Local (MLE)",
    );
    summary.push(Row {
        name: "Local (MLE)".into(),
        rank: "-".into(),
        opt_gap: "-".into(),
        reproj: data.compute_reprojection_error(&loc.rotation, &loc.translation),
        deg: data.compute_degree_error(&loc.rotation, &loc.translation),
        rot: data.compute_rotation_error(&loc.rotation),
        trans: data.compute_translation_error(&loc.translation),
        runtime: fmt_rt(loc.runtime),
    });

    let ex = global::GlobalSolver.solve(
        &data.world_points,
        obs,
        None,
        global::Formulation::Reduced,
        false,
        true,
    );
    println!("\nGlobal (MLE):");
    println!("  rank:      {}", ex.rank);
    println!("  rel gap:   {}", fmt_sci(ex.rel_optimality_gap));
    print_info(&data, &ex.rotation, &ex.translation, ex.runtime, "");
    summary.push(Row {
        name: "Global (MLE)".into(),
        rank: ex.rank.to_string(),
        opt_gap: fmt_sci(ex.rel_optimality_gap),
        reproj: data.compute_reprojection_error(&ex.rotation, &ex.translation),
        deg: data.compute_degree_error(&ex.rotation, &ex.translation),
        rot: data.compute_rotation_error(&ex.rotation),
        trans: data.compute_translation_error(&ex.translation),
        runtime: fmt_rt(ex.runtime),
    });

    print_table(&summary);
}
