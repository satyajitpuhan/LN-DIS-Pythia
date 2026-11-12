#!/usr/bin/env python3
"""Helpers for data_over_theory.py: data loaders, F2^pi curves with PDF bands
and the shift factors from the Lambda fit. Run on its own it draws test versions of
Figs. 2 and 4-7 into ratio_tools/.
"""

import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter
from scipy.integrate import quad

import lhapdf
import lambda_fit as M

HERE = M.HERE
OUT_DIR = os.path.join(HERE, "..", "cache", "ratio_tools_test")
CURVE_CACHE = os.path.join(HERE, "..", "cache", "f2pi_member_curves.npz")

XPI_GRID = np.logspace(-4, np.log10(1.0 - 1e-6), 120)
PAPER_LAMBDA = {fm["model"]: fm["lam_jam"] for fm in M.FLUX_MODELS}

PDF_SPECS = [
    ("JAM21",   M.JAM21_SET,          "#0a7cff",     "-",  True),
    ("xFitter", "xFitterPI_NLO_VAR",  "crimson",     "--", True),
    ("GRV",     "GRVPI1",             "forestgreen", ":",  False),
]
GRV_EXTRA_LW = 1.4
FLUX_STYLE = {  # colour, linestyle, label
    "s_exp":         ("navy",      "-",               r"$s$-exp"),
    "t_exp":         ("crimson",   "--",              r"$t$-exp"),
    "monopole":      ("darkgreen", ":",               "monopole"),
    "regge":         ("goldenrod", "-.",              "Regge"),
    "pauli_villars": ("teal",      (0, (3, 1, 1, 1)), "Pauli-Villars"),
}

# grid-figure layout
PANEL_IN, MARG_L, MARG_R, MARG_B, MARG_T = 2.7, 1.95, 0.18, 1.15, 0.65
FS_AXLABEL, FS_TITLE, FS_ROWLAB, FS_LEGEND = 26, 23, 23, 12

ZEUS_REMOVE_XL = [0.24, 0.31, 0.43, 0.67, 0.79]  # rows of Fig. 6
ZEUS_DROP_Q2 = [1000.0]  # Fig. 6 columns

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif", "mathtext.default": "bfit",
    "font.weight": "bold", "axes.labelweight": "bold", "axes.linewidth": 1.4,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "xtick.labelsize": 17, "ytick.labelsize": 17,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "legend.frameon": True, "legend.edgecolor": "black", "legend.fancybox": False,
    "legend.framealpha": 1.0, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


# Lambda choices
def best_lambdas():
    path = M.OUT_CSV
    if not os.path.exists(path):
        raise SystemExit("run lambda_fit.py first (lambda_fit_results.csv missing)")
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["fit"] == "H1+ZEUS_common_Lambda":
                out[(r["pion_pdf"], r["flux"])] = float(r["Lambda_GeV"])
    return out


def shift_table(tag):
    """(pdf, flux) -> dict(nZ, nH, r={x_L: (r1, r2)}) of the H1+ZEUS fit for this version."""
    want = "H1+ZEUS_at_JAM_Lambda" if tag == "paperLambda" else "H1+ZEUS_common_Lambda"
    out = {}
    with open(M.OUT_CSV) as fh:
        for r in csv.DictReader(fh):
            if r["fit"] != want:
                continue
            rb = {}
            if r.get("r1") not in (None, ""):  # common r1, r2 -> every x_L bin
                rb["common"] = (float(r["r1"]), float(r["r2"]))
            for key, val in r.items():
                if key.startswith("r1_xL") and val not in ("", None):
                    xb = round(float(key[5:]), 2)
                    rb[xb] = (float(val), float(r[f"r2_xL{key[5:]}"]))
            out[(r["pion_pdf"], r["flux"])] = dict(nZ=float(r["n_ZEUS"]), nH=float(r["n_H1"]), r=rb)
    return out


def shift_factor(shifts, pdf, flux_model, expt, xl):
    """Shifted theory = raw theory x this:  1/(n_e s)."""
    sh = shifts[(pdf, flux_model)]
    if expt == "H1":
        return 1.0 / sh["nH"]
    xb = round(float(xl), 2)
    if "common" in sh["r"]:
        sc = M.zeus_scale(sh["r"]["common"], xb)
    else:
        sc = M.zeus_scale(sh["r"][xb], xb) if xb in sh["r"] else 1.0
    if sc <= 0.05:
        raise ValueError(f"ZEUS shift factor s={sc:.3f} for {pdf} {flux_model} x_L={xl}")
    return 1.0 / (sh["nZ"] * sc)


def lambda_table(tag):
    """(pdf_name, flux) -> Lambda for the chosen version."""
    if tag == "paperLambda":
        return {(p[0], m): PAPER_LAMBDA[m] for p in PDF_SPECS for m in PAPER_LAMBDA}
    return best_lambdas()


def h1_all():
    rows = []
    with open(M.H1_DAT) as fh:
        for line in fh:
            t = line.replace("D", "E").replace("d", "e").split()
            if len(t) < 7 or line.strip().startswith("#"):
                continue
            q2, xl, xpi, y = round(float(t[0]), 4), round(float(t[2]), 3), float(t[3]), float(t[4])
            if 0.0 < xpi < 1.0:
                rows.append((q2, xl, xpi, y, float(np.hypot(float(t[5]), float(t[6])))))
    a = np.array(rows)
    return dict(expt="H1", q2=a[:, 0], x_l=a[:, 1], x_pi=a[:, 2], d=a[:, 3], err=a[:, 4])


def zeus_all():
    z = np.loadtxt(M.ZEUS_DAT)
    f2p = np.loadtxt(M.F2P_TXT)[:, 2]
    x_l, q2, x, r, dr = np.round(z[:, 1], 2), z[:, 2], z[:, 3], z[:, 4], z[:, 5]
    dxl = np.where(np.isclose(x_l, 0.24), 0.08, np.where(np.isclose(x_l, 0.94), 0.12, 0.06))
    x_pi = x / (1.0 - x_l)
    m = (x_pi > 0) & (x_pi < 1)
    return dict(expt="ZEUS", q2=q2[m], x_l=x_l[m], x_pi=x_pi[m],
                d=((r / (100.0 * dxl)) * f2p)[m], err=((dr / (100.0 * dxl)) * f2p)[m])


_BDY = None
_CACHE = None


def _fast_grid(pdf, q2v, expt):
    """Same tabulation as M.GpuPdfGrid, one LHAPDF call per x for all flavours."""
    x_fine = M.Z_X_FINE if expt == "ZEUS" else M.H_X_FINE
    arr = np.empty((len(M.PIDS_NEEDED), len(x_fine)))
    qv = float(np.sqrt(q2v))
    for j, xv in enumerate(x_fine):
        d = pdf.xfxQ(float(xv), qv) if expt == "ZEUS" else pdf.xfxQ2(float(xv), float(q2v))
        for i, pid in enumerate(M.PIDS_NEEDED):
            arr[i, j] = d[pid]
    return M.GpuPdfGrid(None, q2v, x_fine, expt != "ZEUS",
                        arrays={pid: arr[i] for i, pid in enumerate(M.PIDS_NEEDED)})


def member_curves(set_name, expt, q2v):
    """(n_members, len(XPI_GRID)) F2^pi for every member of the set."""
    global _BDY, _CACHE
    if _CACHE is None:
        _CACHE = {}
        if os.path.exists(CURVE_CACHE):
            with np.load(CURVE_CACHE) as z:
                _CACHE = {k: z[k] for k in z.files}
    key = f"{set_name}|{expt}|{float(q2v):.6g}|{len(XPI_GRID)}"
    if key in _CACHE:
        return _CACHE[key]
    pset = lhapdf.getPDFSet(set_name)
    members = pset.mkPDFs()
    if _BDY is None:
        _BDY = np.array([M._bdy_int_cpu(v) for v in XPI_GRID])
    out = np.empty((len(members), len(XPI_GRID)))
    for k, m in enumerate(members):
        g = _fast_grid(m, q2v, expt)
        out[k] = (M.f2_nlo_vec_zeus(m, g, XPI_GRID, float(q2v)) if expt == "ZEUS"
                  else M.f2pi_minus_vec_h1(m, g, XPI_GRID, float(q2v), _BDY))
    _CACHE[key] = out
    np.savez(CURVE_CACHE, **_CACHE)
    print(f"  {set_name} {expt} Q2={q2v:g}: {len(members)} members done")
    return out


def band(set_name, has_band, expt, q2v):
    """(central, lo, hi), as in the original build_band()."""
    curves = member_curves(set_name, expt, q2v)
    if not has_band:
        return curves[0], None, None
    pset = lhapdf.getPDFSet(set_name)
    n = curves.shape[1]
    c, lo, hi = np.empty(n), np.empty(n), np.empty(n)
    if pset.errorType == "variations":
        lo, hi = curves.min(axis=0), curves.max(axis=0)
        c = 0.5 * (lo + hi)
    else:
        for j in range(n):
            u = pset.uncertainty(list(curves[:, j]), cl=68)
            c[j], lo[j], hi[j] = u.central, u.central - u.errminus, u.central + u.errplus
    return c, lo, hi


# tick formatters
def _log_fmt(odd):
    def f(v, pos):
        if v <= 0:
            return ""
        e = int(round(np.log10(v)))
        if abs(v - 10.0 ** e) > 1e-6 * 10.0 ** e or (e % 2 == 0) == odd:
            return ""
        return rf"$\mathbf{{10^{{{e}}}}}$"
    return f


def _lin_fmt(v, pos):
    return rf"$\mathbf{{{v:.2f}}}$"


def _ratio_fmt(v, pos):
    if abs(v - round(v)) < 1e-9:
        return rf"$\mathbf{{{int(round(v))}}}$"
    return rf"$\mathbf{{{v:.1f}}}$"


# Fig. 2
def fig2_flux(lam_of, tag):
    cuts = [(r"H1:  $P_T \leq 0.2$ GeV", lambda xL: 0.04),
            (r"ZEUS:  $P_T \leq 0.656\,x_L$ GeV", lambda xL: (0.656 * xL) ** 2)]
    pref = (M.G_A ** 2 * M.M_N ** 2) / ((4.0 * np.pi * M.F_PI) ** 2)

    def f_piN(xL, model, lam, k2max):
        xb = 1.0 - xL
        g = lambda k2: (pref * xb * (k2 + xb * xb * M.M_N ** 2) / (xL * xL * M.d_piN(k2, xL) ** 2)
                        * M.regulator(model, k2, xL, lam) ** 2)
        return quad(g, 0.0, k2max, limit=220)[0]

    xg = np.linspace(0.01, 0.99, 400)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8))
    for c, (title, k2f) in enumerate(cuts):
        ax = axes[c]
        ymax = 0.0
        for model, (col, ls, lab) in FLUX_STYLE.items():
            lam = lam_of(model)
            y = np.array([f_piN(x, model, lam, k2f(x)) for x in xg])
            ymax = max(ymax, y.max())
            ax.plot(xg, y, color=col, ls=ls, lw=2.3, label=rf"{lab}  ($\Lambda={lam:.2f}$)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.22 * ymax)
        ax.set_xlabel(r"$\mathbf{x_L}$", fontsize=20)
        ax.set_ylabel(r"$\mathbf{f_{\pi N}(x_L)}$", fontsize=20)
        ax.tick_params(labelsize=15, length=7, width=1.1)
        ax.grid(True, ls="--", alpha=0.25)
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            t.set_fontweight("bold")
        at = AnchoredText(title, loc=("upper right" if c == 0 else "upper left"), frameon=True,
                          pad=0.4, borderpad=0.6, prop=dict(fontsize=15, fontweight="bold"))
        at.patch.set_boxstyle("square")
        ax.add_artist(at)
    leg = axes[0].legend(loc="upper left", fontsize=11, handlelength=1.8, labelspacing=0.35)
    for t in leg.get_texts():
        t.set_fontweight("bold")
    fig.tight_layout()
    _save(fig, f"fig2_flux_{tag}")


# Figs. 4 / 6
def fig_grid(D, lam_of_pdf, shifts, outname):
    expt = D["expt"]
    if expt == "H1":
        xls = sorted(set(D["x_l"]), reverse=True)
        q2s = sorted(set(D["q2"]))
        ymax, yt, xlim = 0.14, [0.03, 0.06, 0.09, 0.12], (1e-4, 1.0)
    else:
        xls = [v for v in sorted(set(D["x_l"]), reverse=True)
               if not any(abs(v - r) < 5e-3 for r in ZEUS_REMOVE_XL)]
        q2s = [v for v in sorted(set(D["q2"])) if v not in ZEUS_DROP_Q2]
        ymax, yt, xlim = 0.25, [0.05, 0.10, 0.15, 0.20], (XPI_GRID[0], XPI_GRID[-1])
    nr, nc = len(xls), len(q2s)
    fw, fh = nc * PANEL_IN + MARG_L + MARG_R, nr * PANEL_IN + MARG_B + MARG_T
    fig, axes = plt.subplots(nr, nc, figsize=(fw, fh), sharex=True, sharey=True,
                             gridspec_kw=dict(wspace=0, hspace=0))
    fig.subplots_adjust(left=MARG_L / fw, right=1 - MARG_R / fw, bottom=MARG_B / fh,
                        top=1 - MARG_T / fh, wspace=0, hspace=0)
    bands = {(p[0], q): band(p[1], p[4], expt, q) for p in PDF_SPECS for q in q2s}
    for i, xl in enumerate(xls):
        for j, q2v in enumerate(q2s):
            ax = axes[i, j]
            for lab, sname, col, ls, hb in PDF_SPECS:
                fl = (2.0 * M.flux(float(xl), "s_exp", float(lam_of_pdf(lab)), expt)
                      * shift_factor(shifts, lab, "s_exp", expt, xl))
                c, lo, hi = bands[(lab, q2v)]
                if lo is not None:
                    ax.fill_between(XPI_GRID, fl * lo, fl * hi, color=col, alpha=0.25, lw=0, zorder=2)
                lw = 1.8 + (GRV_EXTRA_LW if lab == "GRV" else 0.0)
                ax.plot(XPI_GRID, fl * c, color=col, ls=ls, lw=lw, zorder=3)
            m = np.isclose(D["x_l"], xl) & np.isclose(D["q2"], q2v)
            if m.any():
                ax.errorbar(D["x_pi"][m], D["d"][m], yerr=D["err"][m], fmt="o", ms=3.4, color="black",
                            mfc="black", ecolor="black", elinewidth=0.9, capsize=2.0, zorder=5)
            ax.set_xscale("log")
            ax.set_xlim(*xlim)
            ax.set_ylim(0.0, ymax)
            ax.set_yticks(yt)
            ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
            ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.xaxis.set_major_formatter(FuncFormatter(_log_fmt(odd=True)))
            ax.yaxis.set_major_formatter(FuncFormatter(_lin_fmt))
            ax.tick_params(which="major", length=7, width=1.1)
            ax.tick_params(which="minor", length=3.5, width=0.9)
            if i == nr - 1:
                ax.tick_params(labelbottom=True)
            if i == 0:
                ax.set_title(rf"$Q^2={q2v:g}\ \mathrm{{GeV}}^2$", fontsize=FS_TITLE, fontweight="bold", pad=8)
            if j == 0:
                ax.text(-0.33, 0.5, rf"$x_L={xl:.2f}$", transform=ax.transAxes, rotation=90,
                        ha="center", va="center", fontsize=FS_ROWLAB, fontweight="bold")
    fig.text(0.18 / fw, MARG_B / fh + 0.5 * (1 - (MARG_B + MARG_T) / fh),
             r"$F_2^{LN}(x_\pi,Q^2,x_L)$", rotation="vertical", va="center",
             fontsize=FS_AXLABEL, fontweight="bold")
    fig.text(MARG_L / fw + 0.5 * (1 - (MARG_L + MARG_R) / fw), 0.16 / fh, r"$x_\pi$",
             ha="center", fontsize=FS_AXLABEL, fontweight="bold")
    hs = [Line2D([0], [0], marker="o", color="black", lw=0, ms=7, label=f"{expt} data")]
    for lab, sname, col, ls, hb in PDF_SPECS:
        hs.append(Line2D([0], [0], color=col, ls=ls, lw=2.2 + (GRV_EXTRA_LW if lab == "GRV" else 0),
                         label=rf"{lab} ($\Lambda$={lam_of_pdf(lab):.2f})"))
    leg = axes[0, 0].legend(handles=hs, loc="upper right", fontsize=FS_LEGEND - 1.5,
                            handlelength=2.2, borderpad=0.3, labelspacing=0.25)
    for t in leg.get_texts():
        t.set_fontweight("bold")
    _save(fig, outname, dpi=240)


# Figs. 5 / 7
def fig_ratio(D, lam_of_flux, shifts, outname):
    expt = D["expt"]
    jam = PDF_SPECS[0]
    xls = sorted(set(D["x_l"]), reverse=True)[:3]
    q2s = sorted(set(D["q2"]))
    xlim = (1e-4, 1.0) if expt == "H1" else (1e-3, 1.0)
    bands = {q: band(jam[1], True, expt, q) for q in q2s}
    models = list(FLUX_STYLE)
    offs = np.linspace(-1.0, 1.0, len(models))
    jit = 0.94

    # auto y-range
    yv = [1.0]
    for xl in xls:
        for q2v in q2s:
            m = np.isclose(D["x_l"], xl) & np.isclose(D["q2"], q2v)
            if not m.any():
                continue
            c, lo, hi = bands[q2v]
            for mdl in models:
                th = (2.0 * M.flux(float(xl), mdl, float(lam_of_flux(mdl)), expt)
                      * shift_factor(shifts, "JAM21", mdl, expt, xl) * np.interp(D["x_pi"][m], XPI_GRID, c))
                g = th > 0
                yv += [float(np.min((D["d"][m][g] - D["err"][m][g]) / th[g])),
                       float(np.max((D["d"][m][g] + D["err"][m][g]) / th[g]))]
    y_lo, y_hi = min(yv), max(yv)
    pad = max(0.06 * (y_hi - y_lo), 0.02)
    ylim = (max(0.0, y_lo - pad), y_hi + pad)

    nr, nc = len(q2s), len(xls)
    fig, axes = plt.subplots(nr, nc, figsize=(2.9 * nc + 0.9, 0.95 * nr + 1.9), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.0, wspace=0.06)
    for j, xl in enumerate(xls):
        for i, q2v in enumerate(q2s):
            ax = axes[i, j]
            ax.axhline(1.0, color="0.55", lw=1.2, zorder=1)
            m = np.isclose(D["x_l"], xl) & np.isclose(D["q2"], q2v)
            if m.any():
                xd, yd, ed = D["x_pi"][m], D["d"][m], D["err"][m]
                c, lo, hi = bands[q2v]
                mb = (XPI_GRID >= 0.65 * xd.min()) & (XPI_GRID <= 1.55 * xd.max())
                nz = c[mb] != 0
                ax.fill_between(XPI_GRID[mb][nz], lo[mb][nz] / c[mb][nz], hi[mb][nz] / c[mb][nz],
                                color="yellow", alpha=0.85, lw=0, zorder=2)
                for k, mdl in enumerate(models):
                    col = FLUX_STYLE[mdl][0]
                    th = (2.0 * M.flux(float(xl), mdl, float(lam_of_flux(mdl)), expt)
                          * shift_factor(shifts, "JAM21", mdl, expt, xl) * np.interp(xd, XPI_GRID, c))
                    g = th > 0
                    ax.errorbar(xd[g] * jit ** offs[k], yd[g] / th[g], yerr=ed[g] / th[g], fmt="o", ms=2.6,
                                color=col, mfc=col, mec=col, elinewidth=0.9, capsize=0, zorder=5)
                hi_x = xd.max() > 5e-2
            else:
                hi_x = False
            ax.set_xscale("log")
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10], prune="both"))
            ax.yaxis.set_major_formatter(FuncFormatter(_ratio_fmt))
            ax.xaxis.set_major_formatter(FuncFormatter(_log_fmt(odd=False)))
            ax.tick_params(which="both", top=True, right=True, direction="in", labelsize=10)
            ax.tick_params(which="major", length=5, width=1.0)
            ax.tick_params(which="minor", length=2.5, width=0.8)
            ha, xp = ("left", 0.03) if hi_x else ("right", 0.97)
            ax.text(xp, 0.12, rf"$Q^2={q2v:g}$", transform=ax.transAxes, ha=ha, va="bottom",
                    fontsize=9, fontweight="bold")
            if i == 0:
                ax.set_title(rf"$x_L={xl:.2f}$", fontsize=12, fontweight="bold", pad=5)
    hs = [Line2D([0], [0], marker="o", lw=0, ms=4, color=FLUX_STYLE[m][0],
                 label=rf"{FLUX_STYLE[m][2]} ($\Lambda$={lam_of_flux(m):.2f})") for m in models]
    leg = axes[0, 0].legend(handles=hs, loc="upper right", fontsize=6.5, handletextpad=0.3,
                            borderpad=0.3, labelspacing=0.2)
    for t in leg.get_texts():
        t.set_fontweight("bold")
    for j in range(nc):
        axes[-1, j].set_xlabel(r"$x_{\pi}$", fontsize=16, fontweight="bold", labelpad=2)
    fig.text(0.035, 0.5, "data/theory", va="center", rotation="vertical", fontsize=16, fontweight="bold")
    axes[-1, 0].text(0.30, 0.62, r"$F_2^{LN}$   H1" if expt == "H1" else r"$r$   ZEUS",
                     transform=axes[-1, 0].transAxes, ha="center", va="center", fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0.05, 0.0, 1.0, 1.0])
    fig.subplots_adjust(hspace=0.0)
    _save(fig, outname, dpi=340)


def _save(fig, name, dpi=300):
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"wrote ratio_tools/{name}.png/.pdf")


def main():
    print(f"[ratio_tools] backend: {'CuPy (GPU)' if M.GPU else 'NumPy (CPU)'};  JAM21 = {M.JAM21_SET}")
    H, Z = h1_all(), zeus_all()
    for tag in ("paperLambda", "bestLambda"):
        lam, sh = lambda_table(tag), shift_table(tag)
        fig2_flux(lambda m: lam[("JAM21", m)], tag)
        fig_grid(H, lambda p: lam[(p, "s_exp")], sh, f"fig4_H1_F2LN_sexp_{tag}")
        fig_ratio(H, lambda m: lam[("JAM21", m)], sh, f"fig5_H1_ratio_fiveflux_{tag}")
        fig_grid(Z, lambda p: lam[(p, "s_exp")], sh, f"fig6_ZEUS_F2LN_sexp_{tag}")
        fig_ratio(Z, lambda m: lam[("JAM21", m)], sh, f"fig7_ZEUS_ratio_fiveflux_{tag}")

if __name__ == "__main__":
    main()
