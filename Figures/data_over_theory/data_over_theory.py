#!/usr/bin/env python3

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, MaxNLocator, NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
import lambda_fit as M
import ratio_tools as PF

LAMBDA = 1.31
N_XL = 2
BAND_ALPHA = 0.25
X_JITTER = 0.96
PAD_FRAC = 0.06
PDF_SPECS = [
    ("JAM21",   M.JAM21_SET,          "navy",        True),
    ("xFitter", "xFitterPI_NLO_VAR",  "crimson",     True),
    ("GRV",     "GRVPI1",             "forestgreen", False),
]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif", "mathtext.default": "bfit",
    "font.weight": "bold", "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": 1.0, "axes.labelweight": "bold",
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "legend.frameon": True, "legend.edgecolor": "black", "legend.fancybox": False,
    "legend.framealpha": 1.0, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def curves_for(expt, q2v):
    """{label: (central, lo, hi)} on PF.XPI_GRID, centrals as used in the fit."""
    out = {}
    for lab, sname, col, hb in PDF_SPECS:
        mem = PF.member_curves(sname, expt, q2v)
        if lab == "JAM21":
            c, lo, hi = PF.band(sname, True, expt, q2v)  # replica mean + 68 % band
        elif hb:
            c, lo, hi = mem[0], mem.min(axis=0), mem.max(axis=0)  # member 0 + envelope
        else:
            c, lo, hi = mem[0], None, None
        out[lab] = (c, lo, hi)
    return out


def _odd_log_fmt(v, pos):
    if v <= 0:
        return ""
    e = int(round(np.log10(v)))
    if abs(v - 10.0 ** e) > 1e-6 * 10.0 ** e or e % 2 == 0:
        return ""
    return rf"$\mathbf{{10^{{{e}}}}}$"


def _ratio_fmt(v, pos):
    if abs(v - round(v)) < 1e-9:
        return rf"$\mathbf{{{int(round(v))}}}$"
    if abs(v * 10.0 - round(v * 10.0)) < 1e-9:
        return rf"$\mathbf{{{v:.1f}}}$"
    return rf"$\mathbf{{{v:.2f}}}$"


FLUX_SPECS = [  # model, colour, legend label
    ("s_exp",         "navy",      r"$s$-exp"),
    ("t_exp",         "crimson",   r"$t$-exp"),
    ("monopole",      "darkgreen", "monopole"),
    ("regge",         "goldenrod", "Regge"),
    ("pauli_villars", "teal",      "Pauli-Villars"),
]


def _interior_ticks(lo, hi, frac=0.10):
    best = None
    for nb in (4, 5, 6, 7, 8):
        t = MaxNLocator(nbins=nb, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi)
        t = [v for v in t if lo + frac * (hi - lo) < v < hi - frac * (hi - lo)]
        if len(t) >= 3:
            return sorted(sorted(t, key=lambda v: abs(v - 1.0))[:3])  # the 3 around y = 1
        best = best or (t if len(t) >= 2 else None)
    return best


def _row_ticks(lo, hi, frac=0.12):
    found = {}
    for nb in range(2, 10):
        t = MaxNLocator(nbins=nb, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi)
        t = [float(v) for v in t if lo + frac * (hi - lo) < v < hi - frac * (hi - lo)]
        found.setdefault(len(t), t)
    for n in (3, 2):
        if n in found:
            return found[n]
    n = min((k for k in found if k > 3), default=None)
    return found[n][::2] if n else [1.0]


def make_figure(D, series, bands, outbase, jitter, ms, q2_label_above_line, split_ylim=False, per_row=False):
    expt = D["expt"]
    xls = sorted(set(D["x_l"]), reverse=True)[:N_XL]
    q2s = sorted(set(D["q2"]))
    xlim = (1e-4, 1.0) if expt == "H1" else (3e-4, 1.0)
    cols = [(xl, False) for xl in xls] + [(xl, True) for xl in xls]

    def theory(sr, q2v, xd, xl, shifted):
        return sr["fv"][xl] * np.interp(xd, PF.XPI_GRID, sr["curve"](q2v)) * (sr["sf"][xl] if shifted else 1.0)

    def y_range(group, rows=None):
        yv = [1.0]
        for xl, shifted in cols:
            if shifted not in group:
                continue
            yv += _y_values(xl, shifted, rows)
        y_lo, y_hi = min(yv), max(yv)
        if y_hi - y_lo < 0.3:  # (nearly) empty row: keep a sensible span
            c = 0.5 * (y_hi + y_lo)
            y_lo, y_hi = c - 0.15, c + 0.15
        pad = max(PAD_FRAC * (y_hi - y_lo), 0.02)
        return (max(0.0, y_lo - pad), y_hi + pad)

    def _y_values(xl, shifted, rows=None):
        yv = []
        for q2v in (q2s if rows is None else rows):
            m = np.isclose(D["x_l"], xl) & np.isclose(D["q2"], q2v)
            if not m.any():
                continue
            xd = D["x_pi"][m]
            for sr in series:
                th = theory(sr, q2v, xd, xl, shifted)
                g = th > 0
                yv += [float(np.min((D["d"][m][g] - D["err"][m][g]) / th[g])),
                       float(np.max((D["d"][m][g] + D["err"][m][g]) / th[g]))]
            mb = (PF.XPI_GRID >= 0.65 * xd.min()) & (PF.XPI_GRID <= 1.55 * xd.max())
            for col, alpha, bf in bands:
                c, lo, hi = bf(q2v)
                nz = c[mb] != 0
                yv += [float(np.min(lo[mb][nz] / c[mb][nz])), float(np.max(hi[mb][nz] / c[mb][nz]))]
        return yv

    split_ylim = split_ylim or per_row
    if per_row:
        row_ylim = {(i, sh): y_range({sh}, [q2v]) for i, q2v in enumerate(q2s) for sh in (False, True)}
    if split_ylim:
        ylims = {False: y_range({False, True}), True: y_range({True})}  # no-shift group
    else:
        ylims = dict.fromkeys((False, True), y_range({False, True}))

    PW, PH, GAP = 2.8, 0.95, (0.75 if split_ylim else 0.45)  # wider gap holds the shifted-group labels
    ML, MR, MB, MT = 0.95, 0.12, 0.75, 0.80
    nr = len(q2s)
    fw = ML + 4 * PW + GAP + MR
    fh = MB + nr * PH + MT
    fig = plt.figure(figsize=(fw, fh))
    axes = np.empty((nr, 4), dtype=object)
    for k in range(4):
        x0 = ML + k * PW + (GAP if k >= 2 else 0.0)
        for i in range(nr):
            y0 = MB + (nr - 1 - i) * PH
            axes[i, k] = fig.add_axes([x0 / fw, y0 / fh, PW / fw, PH / fh])

    offs = np.linspace(-1.0, 1.0, len(series))
    for k, (xl, shifted) in enumerate(cols):
        for i, q2v in enumerate(q2s):
            ax = axes[i, k]
            ax.axhline(1.0, color="0.55", lw=1.2, zorder=1)
            m = np.isclose(D["x_l"], xl) & np.isclose(D["q2"], q2v)
            if m.any():
                o = np.argsort(D["x_pi"][m])
                xd, yd, ed = D["x_pi"][m][o], D["d"][m][o], D["err"][m][o]
                mb = (PF.XPI_GRID >= 0.65 * xd.min()) & (PF.XPI_GRID <= 1.55 * xd.max())
                for col, alpha, bf in bands:
                    c, lo, hi = bf(q2v)
                    nz = c[mb] != 0
                    ax.fill_between(PF.XPI_GRID[mb][nz], lo[mb][nz] / c[mb][nz], hi[mb][nz] / c[mb][nz],
                                    color=col, alpha=alpha, lw=0, zorder=2)
                for mi, sr in enumerate(series):
                    th = theory(sr, q2v, xd, xl, shifted)
                    g = th > 0
                    ax.errorbar(xd[g] * jitter ** offs[mi], yd[g] / th[g], yerr=ed[g] / th[g], fmt="o",
                                ms=ms, mfc=sr["color"], mec=sr["color"], mew=0.6, ecolor=sr["color"],
                                elinewidth=0.9, capsize=0, color=sr["color"], zorder=5)
                hi_x = xd.max() > 5e-2
            else:
                hi_x = False
            ylim = row_ylim[(i, shifted)] if per_row else ylims[shifted]
            ax.set_xscale("log")
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            if per_row:
                ax.yaxis.set_major_locator(FixedLocator(_row_ticks(*ylim)))
            elif split_ylim and shifted:  # own scale: ticks clear of the edges
                ax.yaxis.set_major_locator(FixedLocator(_interior_ticks(*ylim)))
            else:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10], prune="both"))
            ax.yaxis.set_major_formatter(FuncFormatter(_ratio_fmt))
            ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
            ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.xaxis.set_major_formatter(FuncFormatter(_odd_log_fmt))  # 10^-3, 10^-1: never on a panel edge
            ax.tick_params(which="both", top=True, right=True, direction="in")
            ax.tick_params(which="major", length=5, width=1.0)
            ax.tick_params(which="minor", length=2.5, width=0.8)
            ax.tick_params(labelbottom=(i == nr - 1), labelleft=(k == 0))
            if split_ylim and k == 2:
                ax.tick_params(labelleft=True)
            ha, xp = ("left", 0.03) if hi_x else ("right", 0.97)
            if i == 0 and k == 0:
                ha, xp = "left", 0.03
            if q2_label_above_line:
                yp = min(0.97, (1.0 - ylim[0]) / (ylim[1] - ylim[0]) + 0.06)
                ax.text(xp, yp, rf"$Q^2={q2v:g}$", transform=ax.transAxes, ha=ha, va="bottom",
                        fontsize=8, fontweight="bold")
            else:
                ax.text(xp, 0.12, rf"$Q^2={q2v:g}$", transform=ax.transAxes, ha=ha, va="bottom",
                        fontsize=9, fontweight="bold")
            if i == 0:
                ax.set_title(rf"$x_L={xl:.2f}$", fontsize=12, fontweight="bold", pad=5)
            if i == nr - 1:
                ax.set_xlabel(r"$x_{\pi}$", fontsize=16, fontweight="bold", labelpad=2)

    top = (MB + nr * PH) / fh
    for g, txt in enumerate(("No shift", "Shifted theory")):
        xc = (ML + (2 * g + 1) * PW + (GAP if g else 0.0)) / fw
        fig.text(xc, top + 0.36 / fh, txt, ha="center", va="bottom", fontsize=14, fontweight="bold")

    hs = [Line2D([0], [0], marker="o", color=sr["color"], lw=0, ms=4, mfc=sr["color"], mec=sr["color"],
                 label=sr["label"]) for sr in series]
    leg = axes[0, 0].legend(handles=hs, loc="upper right", fontsize=7, handletextpad=0.3,
                            borderpad=0.3, labelspacing=0.2)
    for t in leg.get_texts():
        t.set_fontweight("bold")
    fig.text(0.25 / fw, (MB + 0.5 * nr * PH) / fh, "data/theory", rotation="vertical",
             ha="center", va="center", fontsize=16, fontweight="bold")
    axes[-1, 0].text(0.30, 0.62, r"$F_2^{LN}$   H1" if expt == "H1" else r"$\mathbf{r}$   ZEUS",
                     transform=axes[-1, 0].transAxes, ha="center", va="center", fontsize=11, fontweight="bold")

    for ext in ("png", "pdf"):
        fig.savefig(f"{outbase}.{ext}", dpi=340, facecolor="white")
    plt.close(fig)
    if per_row:
        for i, q2v in enumerate(q2s):
            print(f"    Q2={q2v:g}: " + "   ".join(
                f"{'shifted' if sh else 'no shift'} {row_ylim[(i, sh)][0]:.2f}-{row_ylim[(i, sh)][1]:.2f} "
                f"ticks {[round(t, 2) for t in _row_ticks(*row_ylim[(i, sh)])]}" for sh in (False, True)))
    print(f"Saved {os.path.basename(outbase)}.png/.pdf   (x_L = {[float(v) for v in xls]}, "
          "y range " + " | ".join(f"{a:.2f}-{b:.2f}" for a, b in dict.fromkeys(ylims.values())) + ")")


def three_pdf_figure(D, shifts, outbase):
    """JAM21, xFitter, GRV with s-exp (Lambda = 1.31)."""
    expt = D["expt"]
    xls = sorted(set(D["x_l"]), reverse=True)[:N_XL]
    cache = {}

    def cv(q2v):
        if q2v not in cache:
            cache[q2v] = curves_for(expt, q2v)
        return cache[q2v]

    fv = {xl: 2.0 * M.flux(float(xl), "s_exp", LAMBDA, expt) for xl in xls}
    series, bands = [], []
    for lab, sname, col, hb in PDF_SPECS:
        series.append(dict(label=lab, color=col, curve=(lambda q, lab=lab: cv(q)[lab][0]), fv=fv,
                           sf={xl: PF.shift_factor(shifts, lab, "s_exp", expt, xl) for xl in xls}))
        if hb:
            bands.append((col, BAND_ALPHA, (lambda q, lab=lab: cv(q)[lab])))
    make_figure(D, series, bands, outbase, jitter=X_JITTER, ms=2.6, q2_label_above_line=False, per_row=True)


def five_flux_figure(D, shifts, outbase):
    """JAM21 (replica mean) with the five flux models at their Table 1 Lambda."""
    expt = D["expt"]
    xls = sorted(set(D["x_l"]), reverse=True)[:N_XL]
    cache = {}

    def jam(q2v):
        if q2v not in cache:
            cache[q2v] = PF.band(M.JAM21_SET, True, expt, q2v)
        return cache[q2v]

    series = []
    for model, col, lab in FLUX_SPECS:
        lam = PF.PAPER_LAMBDA[model]
        series.append(dict(label=lab, color=col, curve=(lambda q: jam(q)[0]),
                           fv={xl: 2.0 * M.flux(float(xl), model, lam, expt) for xl in xls},
                           sf={xl: PF.shift_factor(shifts, "JAM21", model, expt, xl) for xl in xls}))
    bands = [("yellow", 0.85, jam)]  # JAM21 replica band (flux independent)
    make_figure(D, series, bands, outbase, jitter=0.94, ms=2.2, q2_label_above_line=(expt == "ZEUS"),
                split_ylim=True)


def main():
    shifts = PF.shift_table("paperLambda")  # Lambda fixed at the Table 1 values
    for D in (PF.zeus_all(), PF.h1_all()):
        three_pdf_figure(D, shifts, os.path.join(
            HERE, f"data_over_theory_3pdf_sexp_{D['expt']}_top2xL_noshift_vs_shift"))
        five_flux_figure(D, shifts, os.path.join(
            HERE, f"data_over_theory_JAM21_fiveflux_{D['expt']}_top2xL_noshift_vs_shift"))


if __name__ == "__main__":
    main()
