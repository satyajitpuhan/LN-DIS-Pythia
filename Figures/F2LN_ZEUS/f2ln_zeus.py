#!/usr/bin/env python3
"""ZEUS F2^LN against x_pi with the JAM21, xFitter and GRV Sullivan terms and their PDF
uncertainties (Fig. 6). The data points are r / Delta x_L times F2^p from CT18NLO.
"""

import ctypes
import os
import sys
import warnings
from functools import lru_cache

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import quad, IntegrationWarning

warnings.filterwarnings("ignore", message=".*Unable to import Axes3D.*")
warnings.filterwarnings("ignore", category=IntegrationWarning)
warnings.filterwarnings("ignore")

LOCAL_LHAPDF_ROOT = "/home/aries/local"
LOCAL_LHAPDF_SITE = f"{LOCAL_LHAPDF_ROOT}/lib/python3.10/dist-packages"
LOCAL_LHAPDF_LIB = f"{LOCAL_LHAPDF_ROOT}/lib/libLHAPDF.so"
LOCAL_LHAPDF_DATA = f"{LOCAL_LHAPDF_ROOT}/share/LHAPDF"

if os.path.isdir(LOCAL_LHAPDF_SITE) and LOCAL_LHAPDF_SITE not in sys.path:
    sys.path.insert(0, LOCAL_LHAPDF_SITE)

if os.path.exists(LOCAL_LHAPDF_LIB):
    ctypes.CDLL(LOCAL_LHAPDF_LIB, mode=ctypes.RTLD_GLOBAL)

os.environ.setdefault("LHAPDF_DATA_PATH", LOCAL_LHAPDF_DATA)
os.environ.setdefault("LHAPATH", LOCAL_LHAPDF_DATA)

import lhapdf

# GPU backend (CuPy) with automatic CPU fallback
try:
    import cupy as xp
    GPU = True
    print("[zeus_gpu] CuPy found -- running on GPU.")
except Exception:
    import numpy as xp
    GPU = False
    print("[zeus_gpu] CuPy not available -- falling back to NumPy (CPU).")


def to_gpu(arr):
    return xp.asarray(arr)


def to_cpu(arr):
    return xp.asnumpy(arr) if GPU else np.asarray(arr)


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.weight": "bold",
        "mathtext.fontset": "dejavuserif",
        "mathtext.default": "bfit",
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
    }
)

CF = 4.0 / 3.0
TR = 0.5

FLAVOR_INFO = [
    (2, -2, (2.0 / 3.0) ** 2),
    (1, -1, (1.0 / 3.0) ** 2),
    (3, -3, (1.0 / 3.0) ** 2),
    (4, -4, (2.0 / 3.0) ** 2),
    (5, -5, (1.0 / 3.0) ** 2),
]
CHARGE_SQ_SUM = sum(e2 for _, _, e2 in FLAVOR_INFO)

M_N = 0.9382720813
M_PI = 0.13957039
G_A = 1.267
F_PI = 0.093
LAMBDA = 1.31

X_FINE = np.logspace(-6, np.log10(1.0 - 1e-8), 5000)
PIDS_NEEDED = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 21]

N_QUAD = 200
_GL_NODES_CPU, _GL_WEIGHTS_CPU = np.polynomial.legendre.leggauss(N_QUAD)
GL_NODES = to_gpu(_GL_NODES_CPU)
GL_WEIGHTS = to_gpu(_GL_WEIGHTS_CPU)

# label, LHAPDF set name, color, linestyle, has_band
PDF_SPECS = [
    ("JAM21",   "JAM21PionPDFnlo", "#0a7cff",     "-",  True),
    ("xFitter", "xFitterPI_NLO_VAR",  "crimson",     "--", True),
    ("GRV",     "GRVPI1",             "forestgreen", ":",  False),
]

GRV_EXTRA_LW = 1.4

# Proton PDF used to compute F2^p at the ZEUS (x, Q2) points
PROTON_PDF_NAME = "CT18NLO"
PROTON_PDF = lhapdf.mkPDF(PROTON_PDF_NAME, 0)

F2P_TABLE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "F2p_CT18NLO_values.txt")
CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "f2ln_zeus_curves.npz")


# NLO structure-function machinery
def _build_pdf_grid_cpu(pdf, q2val):
    """Query LHAPDF on the fixed X_FINE grid (CPU)."""
    qval = np.sqrt(q2val)
    return {
        pid: np.array([pdf.xfxQ(pid, float(xx), qval) for xx in X_FINE])
        for pid in PIDS_NEEDED
    }


class GpuPdfGrid:

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


def f2_nlo_vec(pdf, grid, x_arr_cpu, q2val):
    """Vectorized NLO F2 across all x in x_arr_cpu (numpy 1D) for one PDF member at one Q^2."""
    x_arr = to_gpu(np.asarray(x_arr_cpu, dtype=np.float64))
    n_x = x_arr.shape[0]

    qval = float(np.sqrt(q2val))
    pref = float(pdf.alphasQ(qval)) / (2.0 * np.pi)
    zmax = 1.0 - 1e-6

    half_width = (zmax - x_arr) / 2.0
    mid = (zmax + x_arr) / 2.0
    z = mid[:, None] + half_width[:, None] * GL_NODES[None, :]  # (n_x, N)
    w = GL_WEIGHTS[None, :]
    jac = half_width[:, None]

    xo = x_arr[:, None] / z
    valid_xo = xo < 1.0

    one_minus_z = 1.0 - z
    log_one_minus_z = xp.log(xp.clip(one_minus_z, 1e-300, None))
    log_z = xp.log(xp.clip(z, 1e-300, None))
    log_one_minus_x = xp.log(xp.clip(1.0 - x_arr, 1e-300, None))

    f2_lo_val = xp.zeros(n_x, dtype=xp.float64)
    c2q_total = xp.zeros(n_x, dtype=xp.float64)

    for pid_q, pid_qb, e2 in FLAVOR_INFO:
        fx = grid.xfq(pid_q, x_arr) + grid.xfq(pid_qb, x_arr)
        f2_lo_val = f2_lo_val + e2 * fx

        fz = grid.xfq(pid_q, xo) + grid.xfq(pid_qb, xo)
        fz = xp.where(valid_xo, fz, 0.0)

        integrand1 = xp.where(
            valid_xo, log_one_minus_z / one_minus_z * (fz - fx[:, None]), 0.0
        )
        i1_val = xp.sum(jac * w * integrand1, axis=1)

        integrand2 = xp.where(valid_xo, (fz - fx[:, None]) / one_minus_z, 0.0)
        i2_val = xp.sum(jac * w * integrand2, axis=1)

        kernel3 = (
            3.0 + 2.0 * z
            - (1.0 + z) * log_one_minus_z
            - ((1.0 + z * z) / one_minus_z) * log_z
        )
        integrand3 = xp.where(valid_xo, kernel3 * fz, 0.0)
        i3_val = xp.sum(jac * w * integrand3, axis=1)

        cq = CF * (
            2.0 * i1_val
            - (log_one_minus_x ** 2) * fx
            - 1.5 * i2_val
            - 1.5 * fx * log_one_minus_x
            + i3_val
            - (np.pi ** 2 / 3.0 + 9.0 / 2.0) * fx
        )
        c2q_total = c2q_total + e2 * pref * cq

    fg = grid.xfq(21, xo)
    fg = xp.where(valid_xo, fg, 0.0)
    kernel_g = (
        (z ** 2 + one_minus_z ** 2) * (log_one_minus_z - log_z)
        + 8.0 * z * one_minus_z
        - 1.0
    )
    integrand_g = xp.where(valid_xo, 2.0 * TR * kernel_g * fg, 0.0)
    ig = xp.sum(jac * w * integrand_g, axis=1)
    c2g = pref * CHARGE_SQ_SUM * ig

    f2 = f2_lo_val + c2q_total + c2g
    out_of_range = (x_arr <= 0.0) | (x_arr >= 0.999999)
    f2 = xp.where(out_of_range, 0.0, f2)
    return to_cpu(f2)


def f2p_ct18_batch(x_arr, q2_arr):
    x_arr = np.asarray(x_arr, dtype=np.float64)
    q2_arr = np.asarray(q2_arr, dtype=np.float64)
    out = np.empty_like(x_arr)
    for q2v in np.unique(q2_arr):
        mask = q2_arr == q2v
        grid = GpuPdfGrid(PROTON_PDF, float(q2v))
        out[mask] = f2_nlo_vec(PROTON_PDF, grid, x_arr[mask], float(q2v))
    return out


@lru_cache(maxsize=None)
def flux(x_l_val):
    def integrand(k2):
        xb = 1.0 - x_l_val
        if xb <= 1e-10:
            return 0.0
        d_val = -(k2 + xb * xb * M_N * M_N + x_l_val * M_PI * M_PI) / x_l_val
        s_val = (k2 + M_N * M_N) / x_l_val + (k2 + M_PI * M_PI) / xb
        ff = np.exp((M_N * M_N - s_val) / (LAMBDA**2))
        pref = (G_A * G_A * M_N * M_N) / ((4.0 * np.pi * F_PI) ** 2)
        return pref * xb * (k2 + xb * xb * M_N * M_N) / (x_l_val * x_l_val * d_val * d_val) * (ff * ff)

    kmax = 0.65 * x_l_val
    val, _ = quad(integrand, 0.0, kmax * kmax, limit=200)
    return val


# ZEUS data (F2^p from CT18NLO)
def get_f2p_values(x, q2, filename=F2P_TABLE_FILE):
    need_compute = True
    if os.path.exists(filename):
        try:
            saved = np.loadtxt(filename)
            if (
                saved.ndim == 2
                and saved.shape[0] == len(x)
                and np.allclose(saved[:, 0], x, rtol=1e-8, atol=1e-12)
                and np.allclose(saved[:, 1], q2, rtol=1e-8, atol=1e-12)
            ):
                need_compute = False
        except Exception:
            need_compute = True

    if need_compute:
        f2p_computed = f2p_ct18_batch(x, q2)
        np.savetxt(
            filename,
            np.column_stack([x, q2, f2p_computed]),
            fmt="%.8e",
            header=f"x Q2 F2p_{PROTON_PDF_NAME}(x,Q2)  [NLO]",
        )
        print(f"Computed and saved F2p values to {filename} ({len(f2p_computed)} rows)")
    else:
        print(f"{filename} already matches this zeus.dat; skipping recompute")

    table = np.loadtxt(filename)
    return table[:, 2]


def load_zeus():
    data = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "zeus.dat"))
    x_l = data[:, 1]
    q2 = data[:, 2]
    x = data[:, 3]
    r = data[:, 4]
    dr = data[:, 5]

    f2p_ref = get_f2p_values(x, q2)

    delta_x_l = np.where(
        np.isclose(x_l, 0.24),
        0.08,
        np.where(np.isclose(x_l, 0.94), 0.12, 0.06),
    )
    x_pi = x / (1.0 - x_l)
    f2ln_data = (r / (100.0 * delta_x_l)) * f2p_ref
    f2ln_err = (dr / (100.0 * delta_x_l)) * f2p_ref
    return {
        "x_l": x_l, "q2": q2, "x": x, "x_pi": x_pi,
        "r": r, "dr": dr, "f2p_ref": f2p_ref,
        "delta_x_l": delta_x_l, "f2ln_data": f2ln_data, "f2ln_err": f2ln_err,
    }


def build_xpi_grid(zeus):
    xmax = min(0.95, max(0.60, 1.05 * float(np.max(zeus["x_pi"]))))
    return np.logspace(-4, np.log10(xmax), 110)


# Per-set central + band (member 0 and full replica band)
def all_members(pset):
    if pset.errorType == "replicas":
        return [pset.mkPDF(i) for i in range(pset.size)]
    return pset.mkPDFs()


def build_band(pset, curves):
    """(central, lo, hi) per x, dispatching on error type."""
    n = curves.shape[1]
    central = np.empty(n); lo = np.empty(n); hi = np.empty(n)
    if pset.errorType == "variations":
        for j in range(n):
            col = curves[:, j]
            lo[j] = col.min(); hi[j] = col.max()
            central[j] = 0.5 * (lo[j] + hi[j])
    else:
        for j in range(n):
            unc = pset.uncertainty(list(curves[:, j]), cl=68)
            central[j] = unc.central
            lo[j] = unc.central - unc.errminus
            hi[j] = unc.central + unc.errplus
    return central, lo, hi


def _f2pi_curve(pdf, q2v, xpi_grid):
    grid = GpuPdfGrid(pdf, float(q2v))
    return f2_nlo_vec(pdf, grid, xpi_grid, float(q2v))


def compute_set(label, set_name, has_band, q2_vals, xpi_grid):
    member0 = {}; central = {}; lo = {}; hi = {}
    if has_band:
        pset = lhapdf.getPDFSet(set_name)
        members = all_members(pset)
        print(f"{label} ({set_name}): {len(members)} members ({pset.errorType}).")
        for q2v in q2_vals:
            q2v = float(q2v)
            curves = np.empty((len(members), len(xpi_grid)))
            for mi, m in enumerate(members):
                curves[mi] = _f2pi_curve(m, q2v, xpi_grid)
                if (mi + 1) % 25 == 0 or mi == len(members) - 1:
                    print(f"  {label} Q2={q2v:g}: member {mi + 1}/{len(members)} done.")
            member0[q2v] = curves[0].copy()
            c, l, h = build_band(pset, curves)
            central[q2v], lo[q2v], hi[q2v] = c, l, h
    else:
        pdf = lhapdf.mkPDF(set_name, 0)
        print(f"{label} ({set_name}): single member, no error set.")
        for q2v in q2_vals:
            q2v = float(q2v)
            cv = _f2pi_curve(pdf, q2v, xpi_grid)
            member0[q2v] = cv; central[q2v] = cv; lo[q2v] = None; hi[q2v] = None
    return member0, central, lo, hi


def save_cache(MEMBER0, CENTRAL, LO, HI, q2_vals, xpi_grid):
    stack = lambda d: np.array([d[float(q)] for q in q2_vals])
    flat = dict(xpi=xpi_grid, q2=np.array([float(q) for q in q2_vals]),
                labels=np.array([s[0] for s in PDF_SPECS]))
    for label, set_name, color, ls, has_band in PDF_SPECS:
        flat[f"{label}__member0"] = stack(MEMBER0[label])
        flat[f"{label}__central"] = stack(CENTRAL[label])
        if has_band:
            flat[f"{label}__lo"] = stack(LO[label])
            flat[f"{label}__hi"] = stack(HI[label])
    np.savez(CACHE_FILE, **flat)
    print(f"Cached curves -> {CACHE_FILE}")


def load_cache_if_valid(q2_vals, xpi_grid):
    if not os.path.exists(CACHE_FILE):
        return None
    data = np.load(CACHE_FILE, allow_pickle=False)
    if data["xpi"].shape != xpi_grid.shape or not np.allclose(data["xpi"], xpi_grid):
        print("Cached x_pi grid differs; recomputing."); return None
    if data["q2"].shape != np.shape(q2_vals) or not np.allclose(data["q2"], np.array([float(q) for q in q2_vals])):
        print("Cached Q2 list differs; recomputing."); return None
    if [str(s) for s in data["labels"]] != [s[0] for s in PDF_SPECS]:
        print("Cached set list differs; recomputing."); return None

    unstack = lambda arr: {float(q): arr[k] for k, q in enumerate(q2_vals)}
    MEMBER0 = {}; CENTRAL = {}; LO = {}; HI = {}
    for label, set_name, color, ls, has_band in PDF_SPECS:
        if f"{label}__central" not in data.files:
            print(f"No cached data for {label}; recomputing."); return None
        MEMBER0[label] = unstack(data[f"{label}__member0"])
        CENTRAL[label] = unstack(data[f"{label}__central"])
        if f"{label}__lo" in data.files:
            LO[label] = unstack(data[f"{label}__lo"])
            HI[label] = unstack(data[f"{label}__hi"])
        else:
            LO[label] = {float(q): None for q in q2_vals}
            HI[label] = {float(q): None for q in q2_vals}
    print(f"Loaded cached curves from {CACHE_FILE} (delete it to recompute).")
    return MEMBER0, CENTRAL, LO, HI


# 12x8 panel figure (no header)
BAND_ALPHA = 0.28


from matplotlib.ticker import (FuncFormatter, AutoMinorLocator,
                               LogLocator, NullFormatter)


def _bold_log_fmt(v, pos):
    """Bold power-of-ten x labels, every 2 decades."""
    if v <= 0.0:
        return ""
    exp = int(round(np.log10(v)))
    if abs(v - 10.0 ** exp) > 1e-6 * 10.0 ** exp:
        return ""  # minor tick between decades -> no label
    if exp % 2 == 0:
        return ""
    return rf"$\mathbf{{10^{{{exp}}}}}$"


def _bold_lin_fmt(v, pos):
    """Bold linear y labels (0.05, 0.10, ...)."""
    return rf"$\mathbf{{{v:.2f}}}$"


def _bold_ticklabels(ax):
    ax.xaxis.set_major_formatter(FuncFormatter(_bold_log_fmt))
    ax.yaxis.set_major_formatter(FuncFormatter(_bold_lin_fmt))


# Layout constants for the gapless panel grid (inches)
PANEL_IN = 2.7  # side of each square panel
MARG_L   = 1.95
MARG_R   = 0.18
MARG_B   = 1.15  # room for x tick labels + x-axis label
MARG_T   = 0.65  # room for the Q^2 column titles

FS_AXLABEL = 26  # F_2^LN and x_pi axis labels
FS_TITLE   = 23  # Q^2 column titles
FS_ROWLAB  = 23  # x_L row labels
FS_LEGEND  = 12
FS_ACC     = 16  # the "NLO" accuracy tag


def make_figure(mode, zeus, q2_vals, xl_vals, xpi_grid,
                MEMBER0, CENTRAL, LO, HI, outbase):
    """mode = 'member0' or 'band'."""
    xl_vals = sorted([float(x) for x in xl_vals], reverse=True)
    nrows = len(xl_vals)
    ncols = len(q2_vals)
    fig_w = ncols * PANEL_IN + MARG_L + MARG_R
    fig_h = nrows * PANEL_IN + MARG_B + MARG_T
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(fig_w, fig_h),
        sharex=True, sharey=True, facecolor="white",
        gridspec_kw=dict(wspace=0.0, hspace=0.0),
    )
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

    for i, xl in enumerate(xl_vals):
        flux_val = 2.0 * flux(float(xl))
        for j, q2v in enumerate(q2_vals):
            q2v = float(q2v)
            ax = axes[i, j]
            ax.set_facecolor("white")

            for label, set_name, color, ls, has_band in PDF_SPECS:
                # GRV has no band and is dotted, so draw it noticeably thicker
                extra = GRV_EXTRA_LW if label == "GRV" else 0.0
                if mode == "member0":
                    ax.plot(xpi_grid, flux_val * MEMBER0[label][q2v],
                            color=color, ls=ls, lw=2.4 + extra, zorder=3)
                else:
                    if LO[label][q2v] is not None:
                        ax.fill_between(xpi_grid, flux_val * LO[label][q2v],
                                        flux_val * HI[label][q2v], color=color,
                                        alpha=BAND_ALPHA, linewidth=0, zorder=2)
                    ax.plot(xpi_grid, flux_val * CENTRAL[label][q2v],
                            color=color, ls=ls, lw=2.1 + extra, zorder=3)

            sel = np.isclose(zeus["x_l"], xl) & np.isclose(zeus["q2"], q2v)
            if np.any(sel):
                ord_idx = np.argsort(zeus["x_pi"][sel])
                ax.errorbar(
                    zeus["x_pi"][sel][ord_idx],
                    zeus["f2ln_data"][sel][ord_idx],
                    yerr=zeus["f2ln_err"][sel][ord_idx],
                    fmt="o", ms=6.0, mfc="black", mec="black", mew=1.2,
                    ecolor="black", elinewidth=1.4, capsize=3.2,
                    color="black", zorder=5,
                )

            ax.set_xscale("log")
            ax.set_xlim(float(xpi_grid[0]), float(xpi_grid[-1]))
            ax.set_ylim(0.0, 0.25)
            ax.set_yticks([0.05, 0.10, 0.15, 0.20])
            ax.yaxis.set_minor_locator(AutoMinorLocator())

            ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
            ax.xaxis.set_minor_locator(
                LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12)
            )
            ax.xaxis.set_minor_formatter(NullFormatter())

            ax.tick_params(which="both", top=True, right=True)
            ax.tick_params(which="major", length=7, width=1.1)
            ax.tick_params(which="minor", length=3.5, width=0.9)
            ax.tick_params(axis="both", which="major", pad=4)
            _bold_ticklabels(ax)

            if i == 0:
                ax.set_title(rf"$Q^2={q2v:g}\ \mathrm{{GeV}}^2$",
                             fontsize=FS_TITLE, fontweight="bold", pad=8)
            if j == 0:
                ax.text(-0.33, 0.5, rf"$x_L={xl:.2f}$", transform=ax.transAxes,
                        va="center", ha="center", rotation=90,
                        fontsize=FS_ROWLAB, fontweight="bold")

    fig.text(0.18 / fig_w, MARG_B / fig_h + 0.5 * (1.0 - (MARG_B + MARG_T) / fig_h),
             r"$F_2^{LN}(x_\pi,Q^2,x_L)$", va="center", ha="left",
             rotation="vertical", fontsize=FS_AXLABEL, fontweight="bold")
    fig.text(MARG_L / fig_w + 0.5 * (1.0 - (MARG_L + MARG_R) / fig_w), 0.16 / fig_h,
             r"$x_{\pi}$", ha="center", va="bottom",
             fontsize=FS_AXLABEL, fontweight="bold")

    handles = [Line2D([0], [0], marker="o", color="black", lw=0, ms=7,
                      mfc="black", label="ZEUS data")]
    for label, set_name, color, ls, has_band in PDF_SPECS:
        leg_lw = 2.2 + (GRV_EXTRA_LW if label == "GRV" else 0.0)
        handles.append(Line2D([0], [0], color=color, ls=ls, lw=leg_lw, label=label))
    leg = axes[0, 0].legend(handles=handles, loc="upper right", fontsize=FS_LEGEND,
                            handlelength=2.0, borderpad=0.4, labelspacing=0.3)
    for t in leg.get_texts():
        t.set_fontweight("bold")

    fig.savefig(f"{outbase}.png", bbox_inches="tight", dpi=380, facecolor="white")
    fig.savefig(f"{outbase}.pdf", bbox_inches="tight", dpi=340, facecolor="white")
    plt.close(fig)
    print(f"Saved {outbase}.png/.pdf")


def main():
    zeus = load_zeus()
    q2_vals = np.unique(zeus["q2"])
    xl_vals = np.unique(zeus["x_l"])
    xpi_grid = build_xpi_grid(zeus)

    loaded = load_cache_if_valid(q2_vals, xpi_grid)
    if loaded is None:
        print("Computing pion F2 for all sets/members (JAM21 replica pass is the slow one) ...")
        MEMBER0 = {}; CENTRAL = {}; LO = {}; HI = {}
        for label, set_name, color, ls, has_band in PDF_SPECS:
            m0, c, l, h = compute_set(label, set_name, has_band, q2_vals, xpi_grid)
            MEMBER0[label], CENTRAL[label], LO[label], HI[label] = m0, c, l, h
        save_cache(MEMBER0, CENTRAL, LO, HI, q2_vals, xpi_grid)
    else:
        MEMBER0, CENTRAL, LO, HI = loaded

    for label, set_name, color, ls, has_band in PDF_SPECS:
        if not has_band:
            continue
        try:
            if lhapdf.getPDFSet(set_name).errorType == "variations":
                for q2v in q2_vals:
                    q2v = float(q2v)
                    CENTRAL[label][q2v] = 0.5 * (LO[label][q2v] + HI[label][q2v])
        except Exception:
            pass

    REMOVE_XL = [0.24, 0.31, 0.43, 0.67, 0.79]

    def _removed_xl(x):
        return any(abs(float(x) - r) < 5e-3 for r in REMOVE_XL)

    q2_plot = [q for q in q2_vals if not np.isclose(float(q), 1000.0)]
    xl_plot = [x for x in xl_vals if not _removed_xl(x)]
    print(f"Plotting {len(xl_plot)} x_L rows x {len(q2_plot)} Q^2 columns "
          f"(removed x_L={REMOVE_XL} and Q^2=1000).")

    make_figure("member0", zeus, q2_plot, xl_plot, xpi_grid,
                MEMBER0, CENTRAL, LO, HI,
                "F2LN_vs_xpi_grid_JAM21_xFitter_GRV_member0")
    make_figure("band", zeus, q2_plot, xl_plot, xpi_grid,
                MEMBER0, CENTRAL, LO, HI,
                "F2LN_vs_xpi_grid_JAM21_xFitter_GRV_band")
    print("Done!")


if __name__ == "__main__":
    main()
