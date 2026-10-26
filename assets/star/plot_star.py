"""Chordal star graph illustration for Section 6.1.

Left: the variable dependency graph — hub h = (h, z) and N spoke variables y_i.
Right: the corresponding moment matrix. Each clique X^(i) is the hub together
with one spoke; the spoke-spoke blocks belong to no clique and are free, which
is exactly what Grone's theorem lets us complete.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, Rectangle

plt.rcParams.update(
    {
        "figure.dpi": 300,
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsfonts}",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)

INK = "#1f2937"  # outlines and text
HUB = "#4b5563"  # dark grey, easier alongside the saturated spokes
SPOKE = [  # palette shared with the experiment figures, reordered 1-5-6-2-4-3
    "#00659F",  # blue
    "#D62828",  # red
    "#2A9D8F",  # teal
    "#E8A33D",  # amber
    "#6A4C93",  # purple
    "#6A994E",  # olive
]
N = len(SPOKE)
DIM_HUB, DIM_SPOKE = 13, 24
SIDE = DIM_HUB + N * DIM_SPOKE

# axes windows, trimmed to the content; the panel widths are set from their
# aspect ratios so neither letterboxes and the gap between them stays small
STAR_XLIM, STAR_YLIM = (-1.44, 1.44), (-1.48, 1.55)
MAT_XLIM, MAT_YLIM = (-4, SIDE + 4), (SIDE + 20, -19)


def span(lim):
    return abs(lim[1] - lim[0])


OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)


def star(ax):
    r_hub, r_spoke, radius = 0.34, 0.17, 1.0
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False) + np.pi / 6

    # clique enclosure around the hub and the first spoke
    d = np.array([np.cos(ang[0]), np.sin(ang[0])])
    ax.add_patch(
        Ellipse(
            0.415 * radius * d,
            width=radius + r_hub + r_spoke + 0.16,
            height=2 * r_hub + 0.16,
            angle=np.degrees(ang[0]),
            facecolor="none",
            edgecolor=INK,
            lw=1.0,
            ls=(0, (4, 3)),
            zorder=1,
        )
    )
    ax.text(
        0.98 * radius,
        0.0,
        r"$x^{(1)}$",
        color=INK,
        fontsize=12,
        ha="left",
        va="center",
    )

    for i, th in enumerate(ang):
        d = np.array([np.cos(th), np.sin(th)])
        ax.plot(
            [r_hub * d[0], (radius - r_spoke) * d[0]],
            [r_hub * d[1], (radius - r_spoke) * d[1]],
            color=SPOKE[i],
            lw=3.0,
            solid_capstyle="round",
            zorder=2,
        )
        ax.add_patch(
            Circle(radius * d, r_spoke, facecolor=SPOKE[i], edgecolor="none", zorder=3)
        )
        label_r = 1.34 if np.isclose(th, 1.5 * np.pi) else 1.45
        ax.text(
            *(label_r * radius * d),
            rf"$y_{i + 1}$",
            color=SPOKE[i],
            fontsize=12,
            ha="center",
            va="center",
        )

    ax.add_patch(Circle((0, 0), r_hub, facecolor=HUB, edgecolor="none", zorder=4))
    ax.text(
        0,
        0,
        r"$h,z$",
        color="white",
        fontsize=11,
        ha="center",
        va="center",
        zorder=5,
    )
    ax.set_xlim(*STAR_XLIM)
    ax.set_ylim(*STAR_YLIM)
    ax.set_aspect("equal")
    ax.axis("off")


def matrix(ax):
    side = SIDE

    def block(x, y, w, h, color, alpha):
        ax.add_patch(
            Rectangle((x, y), w, h, facecolor=color, alpha=alpha, edgecolor="none")
        )

    for i, c in enumerate(SPOKE):
        lo = DIM_HUB + i * DIM_SPOKE
        block(lo, lo, DIM_SPOKE, DIM_SPOKE, c, 0.85)  # spoke diagonal
        block(0, lo, DIM_HUB, DIM_SPOKE, c, 0.45)  # hub-spoke coupling
        block(lo, 0, DIM_SPOKE, DIM_HUB, c, 0.45)
    block(0, 0, DIM_HUB, DIM_HUB, HUB, 1.0)  # shared hub

    # the first clique is the only contiguous one; the rest share its shape
    ax.add_patch(
        Rectangle(
            (0, 0),
            DIM_HUB + DIM_SPOKE,
            DIM_HUB + DIM_SPOKE,
            facecolor="none",
            edgecolor=INK,
            lw=1.1,
            ls=(0, (4, 3)),
        )
    )
    ax.text(
        0.5 * (DIM_HUB + DIM_SPOKE),
        -6,
        r"$X^{(1)}\!\in\mathbb{S}^{37}$",
        color=INK,
        fontsize=11,
        ha="center",
        va="bottom",
    )
    # sits in a spoke-spoke band, which belongs to no clique
    ax.text(
        DIM_HUB + 3.2 * DIM_SPOKE,
        DIM_HUB + 0.5 * DIM_SPOKE,
        "free (completable)",
        color="#8a94a6",
        fontsize=9,
        ha="center",
        va="center",
    )
    ax.add_patch(Rectangle((0, 0), side, side, facecolor="none", edgecolor=INK, lw=0.8))
    ax.text(
        side / 2,
        side + 8,
        r"$X\in\mathbb{S}^{13+24N}$",
        color=INK,
        fontsize=11,
        ha="center",
        va="top",
    )
    ax.set_xlim(*MAT_XLIM)
    ax.set_ylim(*MAT_YLIM)  # inverted: row 0 at the top
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    ratios = [
        span(STAR_XLIM) / span(STAR_YLIM),
        span(MAT_XLIM) / span(MAT_YLIM),
    ]
    fig, axes = plt.subplots(
        1, 2, figsize=(6.3, 3.3), gridspec_kw={"width_ratios": ratios, "wspace": 0.01}
    )
    star(axes[0])
    matrix(axes[1])
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"star.{ext}"))
    plt.close(fig)
    print("saved assets/star/star.{pdf,png}")


if __name__ == "__main__":
    main()
