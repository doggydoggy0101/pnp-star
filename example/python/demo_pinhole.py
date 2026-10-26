"""Pinhole demo: a 10 point scene solved by AlternativeSolver (weighted
reprojection error), LocalSolver (MLE), BnbSolver (MLE), and GlobalSolver (MLE).
"""

import pnp


def fmt(val, spec="<10.6f"):
    return f"{val:{spec}}" if isinstance(val, float) else f"{str(val):<10}"


def main():
    data = pnp.PnpSyntheticDataset(10, (5.0, 35.0), 180.0, 5.0, 10.0, "pinhole", 42)
    wp, obs, K = data.world_points, data.image_points, data.intrinsic

    def print_info(rotation, translation, runtime, name=""):
        if name:
            print(f"\n{name}:")
        print(
            f"  reprojection: {data.compute_reprojection_error(rotation, translation):.6f}"
        )
        print(f"  rotation:     {data.compute_rotation_error(rotation):.6f}")
        print(f"  translation:  {data.compute_translation_error(translation):.6f}")
        print(f"  runtime:      {runtime:.5f} s")

    print("Ground truth:")
    print(f"  reprojection: {data.reprojection_error:.6f}")

    summary = [
        {
            "name": "Ground Truth",
            "rank": "-",
            "rel_gap": "-",
            "reproj": data.reprojection_error,
            "rot": 0.0,
            "trans": 0.0,
            "runtime": "-",
        }
    ]

    wrep = pnp.AlternativeSolver().solve(wp, obs, K)
    print("\nGlobal (W-Reproj):")
    print(f"  rank:      {wrep.rank}")
    print(f"  rel gap:   {wrep.rel_optimality_gap:.2e}")
    print_info(wrep.rotation, wrep.translation, wrep.runtime)
    summary.append(
        {
            "name": "Global (W-Reproj)",
            "rank": wrep.rank,
            "rel_gap": wrep.rel_optimality_gap,
            "reproj": data.compute_reprojection_error(wrep.rotation, wrep.translation),
            "rot": data.compute_rotation_error(wrep.rotation),
            "trans": data.compute_translation_error(wrep.translation),
            "runtime": wrep.runtime,
        }
    )

    loc = pnp.LocalSolver().solve(wp, obs, K)
    print_info(loc.rotation, loc.translation, loc.runtime, "Local (MLE)")
    summary.append(
        {
            "name": "Local (MLE)",
            "rank": "-",
            "rel_gap": "-",
            "reproj": data.compute_reprojection_error(loc.rotation, loc.translation),
            "rot": data.compute_rotation_error(loc.rotation),
            "trans": data.compute_translation_error(loc.translation),
            "runtime": loc.runtime,
        }
    )

    bb = pnp.BnbSolver().solve(wp, obs, K)
    print("\nBnB (MLE):")
    print(f"  nodes:     {bb.nodes}")
    print(f"  rel gap:   {bb.rel_optimality_gap:.2e}")
    print_info(bb.rotation, bb.translation, bb.runtime)
    summary.append(
        {
            "name": "BnB (MLE)",
            "rank": "-",
            "rel_gap": bb.rel_optimality_gap,
            "reproj": data.compute_reprojection_error(bb.rotation, bb.translation),
            "rot": data.compute_rotation_error(bb.rotation),
            "trans": data.compute_translation_error(bb.translation),
            "runtime": bb.runtime,
        }
    )

    ex = pnp.GlobalSolver().solve(wp, obs, K)
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
            "rot": data.compute_rotation_error(ex.rotation),
            "trans": data.compute_translation_error(ex.translation),
            "runtime": ex.runtime,
        }
    )

    print("\n" + "=" * 95)
    print(
        f"{'Solver':<20} | {'Rank':<5} | {'Opt Gap':<10} | {'Reproj Err':<10} | "
        f"{'Rot Err':<10} | {'Trans Err':<10} | {'Runtime(s)':<10}"
    )
    print("-" * 95)
    for r in summary:
        gap_fmt = "<10.2e" if isinstance(r["rel_gap"], float) else "<10"
        rt_fmt = "<10.5f" if isinstance(r["runtime"], float) else "<10"
        print(
            f"{r['name']:<20} | {str(r['rank']):<5} | {fmt(r['rel_gap'], gap_fmt)} | "
            f"{fmt(r['reproj'], '<10.3f')} | {fmt(r['rot'])} | {fmt(r['trans'])} | "
            f"{fmt(r['runtime'], rt_fmt)}"
        )
    print("=" * 95)


if __name__ == "__main__":
    main()
