"""pnp* logo — wordmark + star-graph glyph (the * = optimum / chordal star graph)."""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

plt.rcParams.update(
    {
        "figure.dpi": 600,
        "font.family": "serif",
        "text.usetex": True,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)

INK = "#1f2937"
ACCENT = "#d1495b"

OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)


def blank_axes(fig, bg, xlim, ylim):
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(bg)
    # bg rect so bbox=tight keeps the full canvas
    ax.add_patch(
        Rectangle(
            (xlim[0], ylim[0]),
            xlim[1] - xlim[0],
            ylim[1] - ylim[0],
            facecolor=bg,
            edgecolor="none",
            zorder=-10,
        )
    )
    return ax


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"saved {name}")


def draw_star_glyph(ax, cx, cy, s, lw, color):
    # star graph as an asterisk: 6 spokes with end dots + hub
    for th in np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6:
        d = np.array([np.cos(th), np.sin(th)])
        ax.plot(
            [cx + 0.40 * s * d[0], cx + 0.92 * s * d[0]],
            [cy + 0.40 * s * d[1], cy + 0.92 * s * d[1]],
            color=color,
            lw=lw,
            solid_capstyle="round",
        )
        ax.add_patch(
            Circle(
                (cx + s * d[0], cy + s * d[1]),
                0.14 * s,
                facecolor=color,
                edgecolor="none",
            )
        )
    ax.add_patch(Circle((cx, cy), 0.24 * s, facecolor=ACCENT, edgecolor="none"))


INK_LEFT = 0.4255  # data x of the wordmark's leftmost rendered ink
DROP = 0.08  # teaser line 1 below the pnp ink top, in em


def ink_left_x(fig, ax):
    """Data x of the leftmost dark pixel; the star is red, so it is excluded."""
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    dark = (buf[:, :, 0] < 120) & (buf[:, :, 1] < 120) & (buf[:, :, 2] < 120)
    cols = np.where(dark.any(axis=0))[0]
    return ax.transData.inverted().transform((cols[0], 0))[0]


def draw_lockup(ax, k=1.0):
    # layout in (0..2, 0..1) coordinates; k scales point-sized artists
    # (fontsize/lw) when the coordinate window is not 2 units wide
    t = ax.text(
        0.93,
        0.5,
        r"\textsf{\textbf{pnp}}",
        fontsize=130 * k,
        ha="center",
        va="center",
        color=INK,
    )
    # ! a usetex layout box carries side bearings that move with the TeX
    # install, so ha="center" does not pin the glyphs. Place by rendered ink,
    # the same reason ink_span() exists for the vertical alignment below.
    t.set_x(0.93 + (INK_LEFT - ink_left_x(ax.figure, ax)))
    draw_star_glyph(ax, 1.495, 0.695, 0.105, 2.6 * k, ACCENT)


def ink_span(fig, ax, x0, x1, y0, y1):
    """(ink_top, ink_bottom) in data coords of dark-ink pixels inside the window."""
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    h = buf.shape[0]
    dark = (buf[:, :, 0] < 120) & (buf[:, :, 1] < 120) & (buf[:, :, 2] < 120)
    (c0, r1), (c1, r0) = ax.transData.transform([(x0, y0), (x1, y1)])
    rows = np.where(dark[int(h - r0) : int(h - r1), int(c0) : int(c1)].any(axis=1))[0]
    top_disp, bot_disp = r0 - rows[0], r0 - rows[-1] - 1
    inv = ax.transData.inverted()
    return inv.transform((0, top_disp))[1], inv.transform((0, bot_disp))[1]


def place_tagline(ax, x, y, line1, line2, fontsize):
    kw = dict(
        ha="left",
        color=INK,
        usetex=False,
        fontfamily="Times New Roman",
        fontsize=fontsize,
    )
    return ax.text(x, y, line1, va="bottom", **kw), ax.text(x, y, line2, va="top", **kw)


def teaser():
    # lockup left + two-line tagline right; tagline ink spans exactly the pnp ink
    line1, line2 = "A certifiable solver for", "Perspective-n-Point"
    x_tag, x_max = 1.80, 3.24  # tagline left edge / line 2 right limit
    xlim, ylim = (0.30, 3.34), (0.16, 0.86)
    inch = 3.0  # figure inches per data unit: point sizes keep their data size

    # line 2 sets the font size (shrunk to end by x_max); the canvas then grows
    # right so a longer line 1 keeps the same margin
    fig = plt.figure(
        figsize=(
            round(inch * (xlim[1] - xlim[0]), 6),
            round(inch * (ylim[1] - ylim[0]), 6),
        )
    )
    ax = blank_axes(fig, "white", xlim, ylim)
    t1, t2 = place_tagline(ax, x_tag, 0.5, line1, line2, 40.0)
    fig.canvas.draw()
    to_data = ax.transData.inverted()
    r1, r2 = (t.get_window_extent().transformed(to_data).x1 for t in (t1, t2))
    plt.close(fig)
    fontsize = 40.0 * min(1.0, (x_max - x_tag) / (r2 - x_tag))
    r1 = x_tag + (r1 - x_tag) * fontsize / 40.0  # text width scales with size
    width = np.ceil(100 * inch * (max(xlim[1], r1 + xlim[1] - x_max) - xlim[0])) / 100
    xlim = (xlim[0], xlim[0] + width / inch)  # whole 0.01 in, so px sizes stay exact

    fig = plt.figure(
        figsize=(
            round(inch * (xlim[1] - xlim[0]), 6),
            round(inch * (ylim[1] - ylim[0]), 6),
        )
    )
    ax = blank_axes(fig, "white", xlim, ylim)
    draw_lockup(ax)
    pnp_top, pnp_bot = ink_span(fig, ax, 0.35, 1.40, *ylim)  # star is red -> excluded
    mid = 0.5 * (pnp_top + pnp_bot)
    t1, t2 = place_tagline(ax, x_tag, mid, line1, line2, fontsize)

    # shift each line so its ink touches the pnp ink top / bottom exactly, then
    # drop line 1 slightly (DROP em): flush with the pnp top, the lines look loose
    l1_top, _ = ink_span(fig, ax, x_tag, xlim[1], mid, ylim[1])
    _, l2_bot = ink_span(fig, ax, x_tag, xlim[1], ylim[0], mid)
    to_data = ax.transData.inverted()
    em = (
        to_data.transform((0, fontsize * fig.dpi / 72))[1]
        - to_data.transform((0, 0))[1]
    )
    t1.set_y(mid + (pnp_top - l1_top) - DROP * em)
    t2.set_y(mid + (pnp_bot - l2_bot))
    save(fig, "teaser")


def square():
    # same lockup, square window cropped tight to the content
    fig = plt.figure(figsize=(6, 6))
    ax = blank_axes(fig, "white", xlim=(0.36, 1.66), ylim=(-0.15, 1.15))
    draw_lockup(ax, k=2.0 / 1.3)
    save(fig, "logo")


if __name__ == "__main__":
    teaser()
    square()
