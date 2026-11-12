#!/usr/bin/env python3
"""H1 F2^LN(3) against x_pi in 7x7 panels with the JAM21, xFitter and GRV Sullivan terms and
their PDF uncertainties (Fig. 4). Per-member curves are cached in f2pi_ln3_curves_3sets.npz.
"""

import os
import warnings
from functools import lru_cache

import numpy as np
import lhapdf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import quad

warnings.filterwarnings('ignore')

# GPU backend (CuPy) with automatic CPU fallback.
try:
    import cupy as xp
    GPU = True
    print("[h1_gpu] CuPy found -- running on GPU.")
except Exception:
    import numpy as xp
    GPU = False
    print("[h1_gpu] CuPy not available -- falling back to NumPy (CPU).")


def to_gpu(arr):
    return xp.asarray(arr)


def to_cpu(arr):
    return xp.asnumpy(arr) if GPU else np.asarray(arr)


# Global plot style (white background, bold ticks)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "mathtext.default": "bfit",   # bold *italic* math (variables stay italic)
    "font.weight": "bold",        # bold for any plain (non-math) text too
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 1.4,
    "axes.labelweight": "bold",
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
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
    "figure.dpi": 150,
})

# label, LHAPDF set name, color, linestyle, has_band
PDF_SPECS = [
    ("JAM21",   "JAM21PionPDFnlo", "#0a7cff",     "-",  True),
    ("xFitter", "xFitterPI_NLO_VAR",  "crimson",     "--", True),
    ("GRV",     "GRVPI1",             "forestgreen", ":",  False),
]

# Extra line width added to the GRV curve (it has no band, so it needs to be
# thicker to stand out). Raise this number to make GRV even bolder.
GRV_EXTRA_LW = 1.4

# STEP 1: NLO F2_{pi-} machinery -- GPU, vectorized over an x array
# for one PDF member at one Q^2 at a time.
CF = 4.0 / 3.0
TR = 0.5

E2_u = (2.0 / 3.0) ** 2
E2_d = (1.0 / 3.0) ** 2
E2_s = (1.0 / 3.0) ** 2
E2_c = (2.0 / 3.0) ** 2
E2_b = (1.0 / 3.0) ** 2
CHARGE_SQ_SUM = E2_u + E2_d + E2_s + E2_c + E2_b

# pi- flavor mapping
FLAVOR_MAP_PIMINUS = [
    (1, -1, E2_u),
    (2, -2, E2_d),
    (3, -3, E2_s),
    (4, -4, E2_c),
    (5, -5, E2_b),
]

Cq2_DELTA = CF * (np.pi**2 / 3.0 - 9.0 / 2.0)

PIDS_NEEDED = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 21]
X_FINE = np.logspace(-6, np.log10(1.0 - 1e-9), 5000)

# Fixed-order Gauss-Legendre quadrature nodes/weights on [-1, 1] used for the
# [x, 1-1e-6] integrals. Generous headroom since GPU makes a high order
# essentially free; the z -> 1 endpoint for the "plus" term is analytically
# regulated ((fz - fx) -> 0 there), the regular term is smooth, and the
# gluon term's log((1-z)/z) is integrable and cut off at 1-1e-6 exactly as
# in the original.
N_QUAD = 200
_GL_NODES_CPU, _GL_WEIGHTS_CPU = np.polynomial.legendre.leggauss(N_QUAD)
GL_NODES = to_gpu(_GL_NODES_CPU)
GL_WEIGHTS = to_gpu(_GL_WEIGHTS_CPU)


def _bdy_int_cpu(x):
    """quad(Cq2_singular(z)/(1-z), 0, x) -- pure kernel, no PDF/Q2
    dependence. Has a genuine (integrable) log singularity at z=0, so this
    stays on CPU with adaptive scipy.quad; it's only evaluated once per x
    point (120 total), not once per member/Q2 as in the original.
    """
    def integ_bdy(z):
        if z <= 0.0 or z >= 1.0:
            return 0.0
        cq2_sing = CF * (1.0 + z**2) * (np.log((1.0 - z) / z) - 0.75)
        return cq2_sing / (1.0 - z)
    val, _ = quad(integ_bdy, 0.0, float(x), limit=60)
    return val


def _build_pdf_grid_cpu(pdf, q2val):
    """The one unavoidably-CPU step: query LHAPDF on the fixed X_FINE grid.
    Uses xfxQ2 (Q^2 directly), matching the original's xfq().
    """
    return {
        pid: np.array([pdf.xfxQ2(pid, float(xx), float(q2val)) for xx in X_FINE])
        for pid in PIDS_NEEDED
    }


class GpuPdfGrid:
    """Tabulates one member's PDF grid at one Q^2 (CPU, via LHAPDF) and
    interpolates xfQ2(pid, x) for arbitrary (batched) x arrays on the GPU.
    """

    def __init__(self, pdf, q2val):
        grids_cpu = _build_pdf_grid_cpu(pdf, q2val)
        self.x_fine = to_gpu(X_FINE)
        self.grids = {pid: to_gpu(g) for pid, g in grids_cpu.items()}

    def xfq(self, pid, xval):
        shape = xval.shape
        flat = xval.reshape(-1)
        out = xp.interp(flat, self.x_fine, self.grids[pid], left=0.0, right=0.0)
        valid = (flat > 0.0) & (flat < 1.0)
        out = xp.where(valid, out, 0.0)
        return out.reshape(shape)


def f2pi_minus_vec(pdf, grid, x_arr_cpu, q2val, bdy_int_arr_cpu):
    """Vectorized NLO F2_pi^- across all x in x_arr_cpu (numpy 1D) for one
    PDF member at one Q^2. bdy_int_arr_cpu must align 1:1 with x_arr_cpu.
    Returns a numpy array of the same length.
    """
    x_arr = to_gpu(np.asarray(x_arr_cpu, dtype=np.float64))
    bdy_int = to_gpu(np.asarray(bdy_int_arr_cpu, dtype=np.float64))
    n_x = x_arr.shape[0]

    alpha_s = float(pdf.alphasQ(np.sqrt(float(q2val))))
    pref = alpha_s / (2.0 * np.pi)
    zmax = 1.0 - 1e-6

    half_width = (zmax - x_arr) / 2.0
    mid = (zmax + x_arr) / 2.0
    z = mid[:, None] + half_width[:, None] * GL_NODES[None, :]   # (n_x, N)
    w = GL_WEIGHTS[None, :]
    jac = half_width[:, None]

    xo = x_arr[:, None] / z
    valid_xo = xo < 1.0

    one_minus_z = 1.0 - z
    log_term = xp.log(xp.clip(one_minus_z, 1e-300, None) / xp.clip(z, 1e-300, None))

    cq2_singular = CF * (1.0 + z ** 2) * (log_term - 0.75)
    cq2_regular = CF * (1.5 + 2.0 * z)
    cg2_kernel = 2.0 * TR * ((z ** 2 + one_minus_z ** 2) * log_term - 1.0 + 8.0 * z * one_minus_z)

    f2_lo_val = xp.zeros(n_x, dtype=xp.float64)
    c2q_total = xp.zeros(n_x, dtype=xp.float64)

    for pid_q, pid_qb, e2 in FLAVOR_MAP_PIMINUS:
        xqq = grid.xfq(pid_q, x_arr) + grid.xfq(pid_qb, x_arr)   # fx, shape (n_x,)
        f2_lo_val = f2_lo_val + e2 * xqq

        fz = grid.xfq(pid_q, xo) + grid.xfq(pid_qb, xo)
        fz = xp.where(valid_xo, fz, 0.0)

        integrand_plus = xp.where(
            valid_xo, cq2_singular * (fz - xqq[:, None]) / one_minus_z, 0.0
        )
        plus_c = xp.sum(jac * w * integrand_plus, axis=1)
        i_plus = plus_c - xqq * bdy_int

        integrand_reg = xp.where(valid_xo, cq2_regular * fz, 0.0)
        reg_c = xp.sum(jac * w * integrand_reg, axis=1)

        i_delta = Cq2_DELTA * xqq

        c2q_total = c2q_total + e2 * pref * (i_plus + reg_c + i_delta)

    fg = grid.xfq(21, xo)
    fg = xp.where(valid_xo, fg, 0.0)
    integrand_g = xp.where(valid_xo, cg2_kernel * fg, 0.0)
    gluon_c = xp.sum(jac * w * integrand_g, axis=1)
    c2g = pref * CHARGE_SQ_SUM * gluon_c

    f2 = f2_lo_val + c2q_total + c2g
    out_of_range = (x_arr <= 0.0) | (x_arr >= 1.0)
    f2 = xp.where(out_of_range, 0.0, f2)
    return to_cpu(f2)


XPI_GRID = np.logspace(-4, np.log10(1.0 - 1e-6), 120)
Q2_bins  = [7.3, 11.0, 16.0, 24.0, 37.0, 55.0, 82.0]

# Pure-kernel boundary integral, independent of PDF/member/Q2: compute once.
BDY_INT_GRID = np.array([_bdy_int_cpu(xv) for xv in XPI_GRID])

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "f2ln_h1_curves.npz")

# STEP 2: Compute (or load) central + band per set, per Q2
def _f2_curve(pdf, Q2):
    grid = GpuPdfGrid(pdf, float(Q2))
    return f2pi_minus_vec(pdf, grid, XPI_GRID, float(Q2), BDY_INT_GRID)


def all_members(pset):
    if pset.errorType == "replicas":
        return [pset.mkPDF(i) for i in range(pset.size)]
    return pset.mkPDFs()


def build_band(pset, curves):
    """(central, lo, hi) per x, dispatching on error type.
    replicas/(sym)hessian -> LHAPDF's prescription; variations -> envelope.
    """
    n = curves.shape[1]
    central = np.empty(n); lo = np.empty(n); hi = np.empty(n)
    if pset.errorType == "variations":
        for j in range(n):
            col = curves[:, j]
            lo[j] = col.min(); hi[j] = col.max()
            # Use the band midpoint (average of the min/max envelope) as the
            # plotted central instead of member 0. For one-sided systematic
            # variations the nominal member (col[0]) sits at the EDGE of the
            # envelope, which makes the dashed line hug one side of the band;
            # the midpoint keeps it centered.
            central[j] = 0.5 * (lo[j] + hi[j])
    else:
        for j in range(n):
            unc = pset.uncertainty(list(curves[:, j]), cl=68)
            central[j] = unc.central
            lo[j] = unc.central - unc.errminus
            hi[j] = unc.central + unc.errplus
    return central, lo, hi


def compute_set(label, set_name, has_band):
    central = {}; lo = {}; hi = {}
    if has_band:
        pset = lhapdf.getPDFSet(set_name)
        members = all_members(pset)
        print(f"{label} ({set_name}): {len(members)} members ({pset.errorType}).")
        for Q2 in Q2_bins:
            curves = np.empty((len(members), len(XPI_GRID)))
            for mi, m in enumerate(members):
                curves[mi] = _f2_curve(m, Q2)
                if (mi + 1) % 25 == 0 or mi == len(members) - 1:
                    print(f"  {label} Q2={Q2:g}: member {mi + 1}/{len(members)} done.")
            c, l, h = build_band(pset, curves)
            central[Q2], lo[Q2], hi[Q2] = c, l, h
    else:
        pdf = lhapdf.mkPDF(set_name, 0)
        print(f"{label} ({set_name}): central member only, no error set.")
        for Q2 in Q2_bins:
            c = _f2_curve(pdf, Q2)
            central[Q2], lo[Q2], hi[Q2] = c, None, None
    return central, lo, hi


def save_cache(CENTRAL, LO, HI):
    stack = lambda d: np.array([d[Q2] for Q2 in Q2_bins])
    flat = dict(
        xpi=XPI_GRID, q2=np.array(Q2_bins),
        labels=np.array([s[0] for s in PDF_SPECS]),
    )
    for label, set_name, color, ls, has_band in PDF_SPECS:
        flat[f"{label}__central"] = stack(CENTRAL[label])
        if has_band:
            flat[f"{label}__lo"] = stack(LO[label])
            flat[f"{label}__hi"] = stack(HI[label])
    np.savez(CACHE_FILE, **flat)
    print(f"Cached curves -> {CACHE_FILE}")


def load_cache_if_valid():
    if not os.path.exists(CACHE_FILE):
        return None
    data = np.load(CACHE_FILE, allow_pickle=False)
    if data["xpi"].shape != XPI_GRID.shape or not np.allclose(data["xpi"], XPI_GRID):
        print("Cached x-grid differs; recomputing."); return None
    if not np.allclose(data["q2"], np.array(Q2_bins)):
        print("Cached Q2 bins differ; recomputing."); return None
    if [str(s) for s in data["labels"]] != [s[0] for s in PDF_SPECS]:
        print("Cached set list differs; recomputing."); return None

    unstack = lambda arr: {Q2: arr[k] for k, Q2 in enumerate(Q2_bins)}
    CENTRAL = {}; LO = {}; HI = {}
    for label, set_name, color, ls, has_band in PDF_SPECS:
        if f"{label}__central" not in data.files:
            print(f"No cached data for {label}; recomputing."); return None
        CENTRAL[label] = unstack(data[f"{label}__central"])
        if f"{label}__lo" in data.files:
            LO[label] = unstack(data[f"{label}__lo"])
            HI[label] = unstack(data[f"{label}__hi"])
        else:
            LO[label] = {Q2: None for Q2 in Q2_bins}
            HI[label] = {Q2: None for Q2 in Q2_bins}
    print(f"Loaded cached curves from {CACHE_FILE} (delete it to recompute).")
    return CENTRAL, LO, HI


# STEP 3: Pion flux model (i)  -- physics flux, NOT a normalization
M_N        = 0.9382720813
M_PI       = 0.13957039
G_A        = 1.267
F_PI       = 0.093
LAMBDA_CUT = 1.31
K2_MAX     = 0.04


def d_piN(k2, xL):
    xbar = 1.0 - xL
    return -(k2 + xbar*xbar*M_N*M_N + xL*M_PI*M_PI) / xL


@lru_cache(maxsize=128)
def f_piN_model_i(xL):
    if xL <= 0.0 or xL >= 1.0:
        return 0.0
    xbar = 1.0 - xL
    pref = (G_A * G_A * M_N * M_N) / ((4.0 * np.pi * F_PI) ** 2)

    def integrand(k2):
        dval = d_piN(k2, xL)
        sval = (k2 + M_N*M_N) / xL + (k2 + M_PI*M_PI) / xbar
        ff   = np.exp((M_N*M_N - sval) / (LAMBDA_CUT * LAMBDA_CUT))
        numer = xbar * (k2 + xbar*xbar*M_N*M_N)
        denom = xL*xL * dval*dval
        return (numer / denom) * (ff * ff)

    val, _ = quad(integrand, 0.0, K2_MAX, limit=220)
    return pref * val


# STEP 4: Read H1 data
def read_h1_data(path):
    grouped = {}
    q2_vals = set()
    xl_vals = set()

    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            toks = s.replace("D", "E").replace("d", "e").split()
            if len(toks) < 7:
                continue
            try:
                q2   = round(float(toks[0]), 4)
                xl   = round(float(toks[2]), 3)
                xpi  = float(toks[3])
                y    = float(toks[4])
                dstat = abs(float(toks[5]))
                dsys  = abs(float(toks[6]))
            except ValueError:
                continue
            if xpi <= 0.0 or xpi >= 1.0:
                continue
            err = np.hypot(dstat, dsys)
            grouped.setdefault((q2, xl), []).append((xpi, y, err))
            q2_vals.add(q2)
            xl_vals.add(xl)

    panel_data = {}
    for key, rows in grouped.items():
        arr = np.array(rows, dtype=float)
        arr = arr[np.argsort(arr[:, 0])]
        panel_data[key] = {"xpi": arr[:, 0], "y": arr[:, 1], "err": arr[:, 2]}

    return sorted(q2_vals), sorted(xl_vals), panel_data


# STEP 5: 7x7 panel plot builder
YMAX = 0.14


from matplotlib.ticker import (FuncFormatter, AutoMinorLocator,
                               LogLocator, NullFormatter)


def _bold_log_fmt(v, pos):
    r"""Bold power-of-ten x labels, every 2 decades. Odd exponents are used
    (10^-3, 10^-1) so no label sits on the panel edge -- with wspace=0 an
    edge label would collide with the neighbouring panel's.
    Tick labels are mathtext, so bold has to come from \mathbf, not from
    set_fontweight.
    """
    if v <= 0.0:
        return ""
    exp = int(round(np.log10(v)))
    if abs(v - 10.0 ** exp) > 1e-6 * 10.0 ** exp:
        return ""            # minor tick between decades -> no label
    if exp % 2 == 0:
        return ""            # label every other decade, odd exponents only
    return rf"$\mathbf{{10^{{{exp}}}}}$"


def _bold_lin_fmt(v, pos):
    """Bold linear y labels (0.00, 0.05, 0.10 ...)."""
    return rf"$\mathbf{{{v:.2f}}}$"


def _bold_ticklabels(ax):
    ax.xaxis.set_major_formatter(FuncFormatter(_bold_log_fmt))
    ax.yaxis.set_major_formatter(FuncFormatter(_bold_lin_fmt))


# Layout constants for the gapless panel grid (inches)
PANEL_IN = 2.7      # side of each square panel
MARG_L   = 1.95     # room for y-axis label, y tick labels, x_L row labels
MARG_R   = 0.18
MARG_B   = 1.15     # room for x tick labels + x-axis label
MARG_T   = 0.65     # room for the Q^2 column titles

# Font sizes. Everything is bold (tick labels via the \mathbf formatters).
FS_AXLABEL = 26     # F_2^LN and x_pi axis labels
FS_TITLE   = 23     # Q^2 column titles
FS_ROWLAB  = 23     # x_L row labels
FS_LEGEND  = 12


def make_figure(mode, outbase, q2_list, xl_list, panel_data, split_factor,
                CENTRAL, LO, HI):
    """mode = 'central' or 'band'. Overlays all sets in PDF_SPECS.
    Rows run from the HIGHEST x_L (top) to the lowest (bottom).
    """
    xl_list = sorted([float(v) for v in xl_list], reverse=True)
    nrows = len(xl_list)
    ncols = len(q2_list)
    fig_w = ncols * PANEL_IN + MARG_L + MARG_R
    fig_h = nrows * PANEL_IN + MARG_B + MARG_T
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(fig_w, fig_h),
        sharex=True, sharey=True,
        facecolor="white",
        gridspec_kw=dict(wspace=0.0, hspace=0.0),
    )
    # Panels butt directly against each other; margins are absolute inches so
    # every axes box comes out exactly PANEL_IN x PANEL_IN (square, no gaps).
    fig.subplots_adjust(
        left=MARG_L / fig_w, right=1.0 - MARG_R / fig_w,
        bottom=MARG_B / fig_h, top=1.0 - MARG_T / fig_h,
        wspace=0.0, hspace=0.0,
    )
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    for i, xl in enumerate(xl_list):
        flux = split_factor[xl]
        for j, q2 in enumerate(q2_list):
            ax = axes[i, j]
            ax.set_facecolor("white")

            for label, set_name, color, ls, has_band in PDF_SPECS:
                cen = CENTRAL[label][q2]
                if mode == "band" and LO[label][q2] is not None:
                    ax.fill_between(XPI_GRID, flux*LO[label][q2],
                                    flux*HI[label][q2], color=color,
                                    alpha=0.25, linewidth=0, zorder=2)
                # GRV has no uncertainty band, so it is a bare line competing
                # with two filled bands -- and it is dotted, which reads
                # thinner than a solid stroke. Draw it noticeably thicker.
                base_lw = 2.4 if mode == "central" else 2.1
                lw = base_lw + GRV_EXTRA_LW if label == "GRV" else base_lw
                ax.plot(XPI_GRID, flux*cen, color=color, ls=ls, lw=lw, zorder=3)

            # H1 data
            d = panel_data.get((q2, xl))
            if d is not None and d["xpi"].size > 0:
                ax.errorbar(
                    d["xpi"], d["y"], yerr=d["err"],
                    fmt='o', ms=6.0, mfc='black', mec='black', mew=1.2,
                    ecolor='black', elinewidth=1.4, capsize=3.2,
                    color='black', zorder=5
                )

            ax.set_xscale("log")
            ax.set_xlim(1e-4, 1.0)
            ax.set_ylim(0.0, YMAX)
            # Drop the 0.00 / YMAX ticks: with hspace=0 an end label would run
            # into the panel above/below.
            ax.set_yticks([0.03, 0.06, 0.09, 0.12])
            ax.yaxis.set_minor_locator(AutoMinorLocator())

            # Be explicit about the log-x ticking: a major tick on every
            # decade (the formatter then labels only 10^-3 and 10^-1) and
            # 2..9 sub-decade minors, unlabelled. Relying on the default
            # LogLocator here can silently yield no labelled major at all.
            ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
            ax.xaxis.set_minor_locator(
                LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12)
            )
            ax.xaxis.set_minor_formatter(NullFormatter())

            ax.tick_params(which='both', top=True, right=True)
            ax.tick_params(which='major', length=7, width=1.1)
            ax.tick_params(which='minor', length=3.5, width=0.9)
            ax.tick_params(axis='both', which='major', pad=4)
            _bold_ticklabels(ax)

            # Bottom row must show its x tick numbers (sharex hides them on
            # the other rows, which is what we want).
            if i == nrows - 1:
                ax.tick_params(axis='x', labelbottom=True)

            if i == 0:
                ax.set_title(rf"$Q^2={q2:g}\ \mathrm{{GeV}}^2$",
                             fontsize=FS_TITLE, fontweight='bold', pad=8)
            if j == 0:
                # Rotated to run vertically alongside the row, like a second
                # y-axis label -- keeps the left margin narrow.
                ax.text(-0.33, 0.5, rf"$x_L={xl:.2f}$",
                        transform=ax.transAxes, va='center', ha='center',
                        rotation=90, fontsize=FS_ROWLAB, fontweight='bold')

    # ---- axis labels (no top header) -- placed in absolute figure fractions
    # so they track the margins above.
    fig.text(0.18 / fig_w, MARG_B / fig_h + 0.5 * (1.0 - (MARG_B + MARG_T) / fig_h),
             r"$F_2^{LN}(x_\pi,Q^2,x_L)$", va='center', ha='left',
             rotation='vertical', fontsize=FS_AXLABEL, fontweight='bold')
    fig.text(MARG_L / fig_w + 0.5 * (1.0 - (MARG_L + MARG_R) / fig_w), 0.16 / fig_h,
             r"$x_{\pi}$", ha='center', va='bottom',
             fontsize=FS_AXLABEL, fontweight='bold')

    # legend in the FIRST panel
    handles = [Line2D([0], [0], marker='o', color='black', lw=0, ms=7,
                      mfc='black', label='H1 data')]
    for label, set_name, color, ls, has_band in PDF_SPECS:
        leg_lw = 2.2 + GRV_EXTRA_LW if label == "GRV" else 2.2
        handles.append(Line2D([0], [0], color=color, ls=ls, lw=leg_lw, label=label))
    leg = axes[0, 0].legend(handles=handles, loc='upper right',
                            fontsize=FS_LEGEND, handlelength=2.0,
                            borderpad=0.4, labelspacing=0.3)
    for t in leg.get_texts():
        t.set_fontweight('bold')

    outpdf = f"{outbase}.pdf"
    outpng = f"{outbase}.png"
    fig.savefig(outpdf, format='pdf', bbox_inches='tight', dpi=340)
    fig.savefig(outpng, format='png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved: {outpdf}")
    print(f"Saved: {outpng}")


# MAIN
def main():
    loaded = load_cache_if_valid()
    if loaded is None:
        print("Computing NLO F2_pi^- for all sets/members (JAM21 pass is the slow one) ...")
        CENTRAL = {}; LO = {}; HI = {}
        for label, set_name, color, ls, has_band in PDF_SPECS:
            c, l, h = compute_set(label, set_name, has_band)
            CENTRAL[label], LO[label], HI[label] = c, l, h
        save_cache(CENTRAL, LO, HI)
    else:
        CENTRAL, LO, HI = loaded

    # Ensure the dashed central line sits in the MIDDLE of the band for
    # 'variations'-type sets (e.g. xFitter). This is derived purely from the
    # cached LO/HI envelope, so it also corrects any older cache that still
    # stored member 0 as the central -- without forcing a recompute.
    for label, set_name, color, ls, has_band in PDF_SPECS:
        if not has_band:
            continue
        try:
            if lhapdf.getPDFSet(set_name).errorType == "variations":
                for Q2 in Q2_bins:
                    CENTRAL[label][Q2] = 0.5 * (LO[label][Q2] + HI[label][Q2])
        except Exception:
            pass

    base_dir = os.path.dirname(os.path.abspath(__file__))
    h1_path = os.path.join(base_dir, "..", "data", "h1.dat")
    q2_list, xl_list, panel_data = read_h1_data(h1_path)
    print(f"Q2 bins: {q2_list}")
    print(f"xL bins: {xl_list}")

    split_factor = {xl: 2.0 * f_piN_model_i(float(xl)) for xl in xl_list}

    make_figure("central", "FLN3_piminus_49panels_JAM21_xFitter_GRV_central",
                q2_list, xl_list, panel_data, split_factor, CENTRAL, LO, HI)
    make_figure("band", "FLN3_piminus_49panels_JAM21_xFitter_GRV_band",
                q2_list, xl_list, panel_data, split_factor, CENTRAL, LO, HI)
    print("Done!")


if __name__ == "__main__":
    main()
