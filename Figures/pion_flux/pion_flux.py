#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pion flux f(x_L) for the five regulators with the H1 and ZEUS p_T cuts (Fig. 2).

JAM21 uses the JAM18 cutoffs, xFitter and GRV the cutoffs from ../lambda_fit.
"""

import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import AutoMinorLocator, FuncFormatter
from scipy.integrate import quad

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(HERE, "..", "lambda_fit", "lambda_fit_results.csv")
OUTBASE = os.path.join(HERE, "flux_allmodels_JAM21_xFitter_GRV")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "mathtext.default": "bfit",
    "font.weight": "bold",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 1.2,
    "axes.labelweight": "bold",
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.frameon": True,
    "legend.edgecolor": "black",
    "legend.fancybox": False,
    "legend.framealpha": 1.0,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 320,
})

# Physics
M_N = 0.9382720813
M_PI = 0.13957039
G_A = 1.267
F_PI = 0.093
ALPHA_PI_SLOPE = 1.0


def d_piN(k2, xL):
    xb = 1.0 - xL
    return -(k2 + xb * xb * M_N * M_N + xL * M_PI * M_PI) / xL


def t_piN(k2, xL):
    xb = 1.0 - xL
    return -(k2 + xb * xb * M_N * M_N) / xL


def s_piN(k2, xL):
    xb = 1.0 - xL
    return (k2 + M_N * M_N) / xL + (k2 + M_PI * M_PI) / xb


def alpha_pi(tval):
    return ALPHA_PI_SLOPE * tval


def regulator(model, k2, xL, lam):
    xb = 1.0 - xL
    dv = d_piN(k2, xL)
    tv = t_piN(k2, xL)
    sv = s_piN(k2, xL)
    if model == "s_exp":
        return np.exp((M_N * M_N - sv) / lam ** 2)
    if model == "t_exp":
        return np.exp(dv / lam ** 2)
    if model == "monopole":
        return (lam ** 2 - M_PI ** 2) / (lam ** 2 - tv)
    if model == "regge":
        return (xb ** (-alpha_pi(tv))) * np.exp(dv / lam ** 2)
    if model == "pauli_villars":
        denom = lam ** 2 - tv
        if denom == 0.0:
            return 0.0
        arg = 1.0 - (dv * dv) / (denom * denom)
        return np.sqrt(arg) if arg > 0.0 else 0.0
    raise ValueError(f"Unknown flux model: {model}")


def flux_piN(xL, model, lam, k2max):
    if xL <= 0.0 or xL >= 1.0 or k2max <= 0.0:
        return 0.0
    xb = 1.0 - xL
    pref = (G_A * G_A * M_N * M_N) / ((4.0 * np.pi * F_PI) ** 2)

    def integrand(k2):
        dv = d_piN(k2, xL)
        ff = regulator(model, k2, xL, lam)
        return pref * xb * (k2 + xb * xb * M_N * M_N) / (xL * xL * dv * dv) * ff * ff

    val, _ = quad(integrand, 0.0, k2max, limit=220)
    return val


# Models (key, legend name, colour, linestyle)
FLUX_MODELS = [
    ("s_exp",         "s-exp",         "navy",      "-"),
    ("t_exp",         "t-exp",         "crimson",   "--"),
    ("monopole",      "monopole",      "darkgreen", ":"),
    ("regge",         "Regge",         "goldenrod", "-."),
    ("pauli_villars", "Pauli-Villars", "teal",      (0, (3, 1, 1, 1))),
]
TABLE1 = {"s_exp": 1.31, "t_exp": 0.58, "monopole": 0.52, "regge": 0.78, "pauli_villars": 0.25}

PT_CUTS = [
    (r"H1:  $P_T \leq 0.2$ GeV",          lambda xL: 0.04,              "center left"),
    (r"ZEUS:  $P_T \leq 0.656\,x_L$ GeV",  lambda xL: (0.656 * xL) ** 2,  "upper left"),
]


def best_fit_lambdas():
    """{(pdf, flux): Lambda} from the common H1+ZEUS fit."""
    out = {}
    with open(RESULTS_CSV) as fh:
        for r in csv.DictReader(fh):
            if r["fit"] == "H1+ZEUS_common_Lambda":
                out[(r["pion_pdf"], r["flux"])] = float(r["Lambda_GeV"])
    return out


def columns():
    best = best_fit_lambdas()
    return [
        ("JAM21",               {m: TABLE1[m] for m, *_ in FLUX_MODELS},                     "{:.2f}"),
        ("xFitter",             {m: best[("xFitter", m)] for m, *_ in FLUX_MODELS},          "{:.2f}"),
        ("GRV",                 {m: best[("GRV", m)] for m, *_ in FLUX_MODELS},              "{:.2f}"),
    ]


def _num(v, pos):
    return f"{v:g}"


def main():
    cols = columns()
    xg = np.linspace(0.01, 0.99, 400)

    curves = {}
    for r, (_t, k2f, _loc) in enumerate(PT_CUTS):
        for c, (_name, lam, _f) in enumerate(cols):
            for key, *_ in FLUX_MODELS:
                curves[(r, c, key)] = np.array([flux_piN(x, key, lam[key], k2f(x)) for x in xg])

    PW, PH = 4.3, 3.5                                 # panel size (inches)
    ML, MR, MB, MT = 1.25, 0.15, 0.95, 0.15
    nr, nc = len(PT_CUTS), len(cols)
    fw, fh = ML + nc * PW + MR, MB + nr * PH + MT
    fig, axes = plt.subplots(nr, nc, figsize=(fw, fh), sharex=True, sharey="row",
                             gridspec_kw=dict(wspace=0.0, hspace=0.0))
    fig.subplots_adjust(left=ML / fw, right=1 - MR / fw, bottom=MB / fh, top=1 - MT / fh,
                        wspace=0.0, hspace=0.0)

    for r, (tag, _k2f, tag_loc) in enumerate(PT_CUTS):
        ymax = max(float(np.nanmax(curves[(r, c, key)])) for c in range(nc) for key, *_ in FLUX_MODELS)
        ytop = 1.30 * ymax                            # headroom for the legend / cut box
        step = 0.01 if ytop < 0.1 else 0.02
        yt = [v for v in np.arange(step, ytop, step) if v < ytop - 0.6 * step]
        for c, (name, lam, fmt) in enumerate(cols):
            ax = axes[r, c]
            for key, lbl, col, ls in FLUX_MODELS:
                ax.plot(xg, curves[(r, c, key)], color=col, ls=ls, lw=2.3, zorder=3,
                        label=rf"{lbl}  ($\Lambda={fmt.format(lam[key])}$)")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, ytop)
            ax.set_xticks([0.2, 0.4, 0.6, 0.8])      # 0 and 1 dropped (panels touch)
            ax.set_yticks(yt)                         # 0 and the top dropped (panels touch)
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.xaxis.set_major_formatter(FuncFormatter(_num))
            ax.tick_params(which="major", length=7, width=1.1)
            ax.tick_params(which="minor", length=3.5, width=0.9)
            ax.grid(True, which="major", ls="--", alpha=0.25)
            for t in ax.get_xticklabels() + ax.get_yticklabels():
                t.set_fontweight("bold")
            if c == 0:                                # H1 / ZEUS cut box once per row (first column)
                at = AnchoredText(tag, loc=tag_loc, frameon=True, pad=0.35, borderpad=0.5,
                                  prop=dict(fontsize=13, fontweight="bold"))
                at.patch.set_boxstyle("square")
                at.patch.set_linewidth(1.1)
                ax.add_artist(at)
            if r == 0:                                # PDF name inside the panel, top-right corner
                nm = AnchoredText(name, loc="upper right", frameon=False, pad=0.3, borderpad=0.4,
                                  prop=dict(fontsize=19, fontweight="bold"))
                ax.add_artist(nm)
                leg = ax.legend(loc="upper left", fontsize=10.5, handlelength=1.8,
                                labelspacing=0.3, borderpad=0.4, handletextpad=0.5)
                for t in leg.get_texts():
                    t.set_fontweight("bold")
            if c == 0:
                ax.set_ylabel(r"$\mathbf{f_{\pi N}(x_L)}$", fontsize=20, fontweight="bold")
            if r == nr - 1:
                ax.set_xlabel(r"$\mathbf{x_L}$", fontsize=20, fontweight="bold")

    # self-check: legend, cut box and curves must not overlap
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    problems = []
    for r in range(nr):
        for c in range(nc):
            ax = axes[r, c]
            boxes = [a.get_window_extent(rend) for a in ax.artists]
            if ax.get_legend() is not None:
                boxes.append(ax.get_legend().get_window_extent(rend))
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    if boxes[i].overlaps(boxes[j]):
                        problems.append(f"row {r} col {c}: legend and cut box overlap")
            for ln in ax.lines:
                pts = ax.transData.transform(ln.get_xydata())
                for b in boxes:
                    if np.any((pts[:, 0] > b.x0) & (pts[:, 0] < b.x1) & (pts[:, 1] > b.y0) & (pts[:, 1] < b.y1)):
                        problems.append(f"row {r} col {c}: a curve runs under a legend/box")
                        break
    for p in sorted(set(problems)):
        print("  WARNING:", p)
    if not problems:
        print("  layout check: no overlaps")

    fig.savefig(f"{OUTBASE}.pdf", dpi=340)
    fig.savefig(f"{OUTBASE}.png", dpi=300)
    plt.close(fig)
    print(f"Saved: {OUTBASE}.pdf / .png")


if __name__ == "__main__":
    main()
