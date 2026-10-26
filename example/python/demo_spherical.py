"""Spherical demo: a 10 point scene solved by AlternativeSolver (point-line
error), LocalSolver (MLE), and GlobalSolver (MLE).
"""

import pnp


def fmt(val, spec="<10.6f"):
    return f"{val:{spec}}" if isinstance(val, float) else f"{str(val):<10}"


def main():
    data = pnp.PnpSyntheticDataset(10, (5.0, 35.0), 180.0, 5.0, 1.0, "spherical", 42)
    wp, bv = data.world_points, data.bearing_vectors

    def print_info(rotation, translation, runtime, name=""):
        if name:
            print(f"\n{name}:")
        print(
            f"  reprojection: {data.compute_reprojection_error(rotation, translation):.6f}"
        )
        print(f"  degree:       {data.compute_degree_error(rotation, translation):.6f}")
        print(f"  rotation:     {data.compute_rotation_error(rotation):.6f}")
        print(f"  translation:  {data.compute_translation_error(translation):.6f}")
        print(f"  runtime:      {runtime:.5f} s")

    print("Ground truth:")
    print(f"  reprojection: {data.reprojection_error:.6f}")
    print(f"  degree:       {data.degree_error:.6f}")

    summary = [
        {
            "name": "Ground Truth",
            "rank": "-",
            "rel_gap": "-",
            "reproj": data.reprojection_error,
            "deg": data.degree_error,
            "rot": 0.0,
            "trans": 0.0,
            "runtime": "-",
        }
    ]

    ptl = pnp.AlternativeSolver().solve(wp, bv)
    print("\nGlobal (Pt-Line):")
    print(f"  rank:      {ptl.rank}")
    print(f"  rel gap:   {ptl.rel_optimality_gap:.2e}")
    print_info(ptl.rotation, ptl.translation, ptl.runtime)
    summary.append(
        {
            "name": "Global (Pt-Line)",
            "rank": ptl.rank,
            "rel_gap": ptl.rel_optimality_gap,
            "reproj": data.compute_reprojection_error(ptl.rotation, ptl.translation),
            "deg": data.compute_degree_error(ptl.rotation, ptl.translation),
            "rot": data.compute_rotation_error(ptl.rotation),
            "trans": data.compute_translation_error(ptl.translation),
            "runtime": ptl.runtime,
        }
    )

    loc = pnp.LocalSolver().solve(wp, bv)
    print_info(loc.rotation, loc.translation, loc.runtime, "Local (MLE)")
    summary.append(
        {
            "name": "Local (MLE)",
            "rank": "-",
            "rel_gap": "-",
            "reproj": data.compute_reprojection_error(loc.rotation, loc.translation),
            "deg": data.compute_degree_error(loc.rotation, loc.translation),
            "rot": data.compute_rotation_error(loc.rotation),
            "trans": data.compute_translation_error(loc.translation),
            "runtime": loc.runtime,
        }
    )

    ex = pnp.GlobalSolver().solve(wp, bv)
    print("\nGlobal (MLE):")
    print(f"  rank:      {ex.rank}")
    print(f"  rel gap:   {ex.rel_optimality_gap:.2e}")
    print_info(ex.rotation, ex.translation, ex.runtime)
    summary.append(
        {
            "name": "Global (MLE)",
            "rank": ex.rank,
            "rel_gap": ex.rel_optimality_gap,
            "reproj": data.compute_reprojection_error(ex.rotation, ex.translation),
            "deg": data.compute_degree_error(ex.rotation, ex.translation),
            "rot": data.compute_rotation_error(ex.rotation),
            "trans": data.compute_translation_error(ex.translation),
            "runtime": ex.runtime,
        }
    )

    print("\n" + "=" * 110)
    print(
        f"{'Solver':<20} | {'Rank':<5} | {'Opt Gap':<10} | {'Reproj Err':<10} | "
        f"{'Deg Err':<10} | {'Rot Err':<10} | {'Trans Err':<10} | {'Runtime(s)':<10}"
    )
    print("-" * 110)
    for r in summary:
        gap_fmt = "<10.2e" if isinstance(r["rel_gap"], float) else "<10"
        rt_fmt = "<10.5f" if isinstance(r["runtime"], float) else "<10"
        print(
            f"{r['name']:<20} | {str(r['rank']):<5} | {fmt(r['rel_gap'], gap_fmt)} | "
            f"{fmt(r['reproj'], '<10.6f')} | {fmt(r['deg'], '<10.6f')} | {fmt(r['rot'])} | "
            f"{fmt(r['trans'])} | {fmt(r['runtime'], rt_fmt)}"
        )
    print("=" * 110)


if __name__ == "__main__":
    main()
