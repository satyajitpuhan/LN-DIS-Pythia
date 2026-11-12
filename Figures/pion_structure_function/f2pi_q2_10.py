#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F2^pi at Q^2 = 10 GeV^2 for JAM21, xFitter and GRV with PDF uncertainty bands (Fig. 3).
Uses the GPU through cupy if it is installed.
"""

import warnings
import numpy as np
import lhapdf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# GPU backend (CuPy) with automatic CPU fallback.
try:
    import cupy as xp
    GPU = True
    print("[band_gpu] CuPy found -- running on GPU.")
except Exception:
    import numpy as xp
    GPU = False
    print("[band_gpu] CuPy not available -- falling back to NumPy (CPU).")


def to_gpu(arr):
    return xp.asarray(arr)


def to_cpu(arr):
    return xp.asnumpy(arr) if GPU else np.asarray(arr)


def _bold_ticklabels(ax):
    for lab in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lab.set_fontweight("bold")
    for lab in list(ax.get_xticklabels(minor=True)) + list(ax.get_yticklabels(minor=True)):
        lab.set_fontweight("bold")


plt.rcParams.update({
    "font.family": "serif",
    "axes.linewidth": 1.6,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.pad": 8,
    "ytick.major.pad": 8,
})

# 1. NLO DIS COEFFICIENT-FUNCTION MACHINERY (GPU, vectorized over x_grid)
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

X_FINE = np.logspace(-5, np.log10(1.0 - 1e-8), 5000)

# Fixed-order Gauss-Legendre quadrature nodes/weights on [-1, 1].
# N_QUAD controls accuracy vs. speed; raise it if you need tighter agreement
# with the original adaptive-quad results near the z->1 log singularity
# (which is analytically regulated by the (fz - fx) "+"-distribution terms,
# so it is finite, just slowly resolved). 200 is generous headroom for a
# smooth band plot; the whole point of moving to GPU is that a high order
# here is essentially free.
N_QUAD = 200
_GL_NODES_CPU, _GL_WEIGHTS_CPU = np.polynomial.legendre.leggauss(N_QUAD)
GL_NODES = to_gpu(_GL_NODES_CPU)      # (N_QUAD,)
GL_WEIGHTS = to_gpu(_GL_WEIGHTS_CPU)  # (N_QUAD,)


def _build_pdf_grid_cpu(pdf, q2val):
    """The one unavoidably-CPU step: query LHAPDF on the fixed X_FINE grid."""
    qval = np.sqrt(q2val)
    pids_needed = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 21]
    return {
        pid: np.array([pdf.xfxQ(pid, float(xx), qval) for xx in X_FINE])
        for pid in pids_needed
    }


class GpuPdfGrid:
    """Holds one member's tabulated PDF grid on the GPU and interpolates
    xfQ(pid, x) for arbitrary (batched) x arrays, entirely on-device.
    """

    def __init__(self, pdf, q2val):
        grids_cpu = _build_pdf_grid_cpu(pdf, q2val)
        self.x_fine = to_gpu(X_FINE)
        self.grids = {pid: to_gpu(g) for pid, g in grids_cpu.items()}

    def xfq(self, pid, xval):
        """xval: GPU array of any shape. Returns array of same shape.
        Matches original semantics: 0 outside (0, 1).
        """
        shape = xval.shape
        flat = xval.reshape(-1)
        out = xp.interp(flat, self.x_fine, self.grids[pid], left=0.0, right=0.0)
        valid = (flat > 0.0) & (flat < 1.0)
        out = xp.where(valid, out, 0.0)
        return out.reshape(shape)


def f2_lo_vec(grid, x_arr):
    """x_arr: GPU array (n_x,). Returns GPU array (n_x,)."""
    total = xp.zeros_like(x_arr)
    for pid_q, pid_qb, e2 in FLAVOR_INFO:
        total = total + e2 * (grid.xfq(pid_q, x_arr) + grid.xfq(pid_qb, x_arr))
    return total


def f2_nlo_vec(pdf, grid, x_arr_cpu, q2val):
    """Vectorized NLO F2^pi across all x in x_arr_cpu (numpy 1D) for one
    PDF member. Returns a numpy array of the same length.
    """
    x_arr = to_gpu(x_arr_cpu.astype(np.float64))
    n_x = x_arr.shape[0]

    qval = float(np.sqrt(q2val))
    pref = float(pdf.alphasQ(qval)) / (2.0 * np.pi)
    zmax = 1.0 - 1e-6

    # Map GL nodes on [-1,1] -> z in [x, zmax], per x point.
    # z[j, k], jac[j] with shape (n_x, N_QUAD) / (n_x,)
    half_width = (zmax - x_arr) / 2.0                      # (n_x,)
    mid = (zmax + x_arr) / 2.0                              # (n_x,)
    z = mid[:, None] + half_width[:, None] * GL_NODES[None, :]   # (n_x, N)
    w = GL_WEIGHTS[None, :]                                       # (1, N)
    jac = half_width[:, None]                                     # (n_x, 1)

    xo = x_arr[:, None] / z                                       # (n_x, N)
    valid_xo = xo < 1.0

    one_minus_z = 1.0 - z
    log_one_minus_z = xp.log(xp.clip(one_minus_z, 1e-300, None))
    log_z = xp.log(xp.clip(z, 1e-300, None))
    log_one_minus_x = xp.log(xp.clip(1.0 - x_arr, 1e-300, None))

    f2_lo_val = xp.zeros(n_x, dtype=xp.float64)
    c2q_total = xp.zeros(n_x, dtype=xp.float64)

    for pid_q, pid_qb, e2 in FLAVOR_INFO:
        fx = grid.xfq(pid_q, x_arr) + grid.xfq(pid_qb, x_arr)     # (n_x,)
        f2_lo_val = f2_lo_val + e2 * fx

        fz = grid.xfq(pid_q, xo) + grid.xfq(pid_qb, xo)           # (n_x, N)
        fz = xp.where(valid_xo, fz, 0.0)

        integrand1 = xp.where(
            valid_xo,
            log_one_minus_z / one_minus_z * (fz - fx[:, None]),
            0.0,
        )
        i1_val = xp.sum(jac * w * integrand1, axis=1)

        integrand2 = xp.where(
            valid_xo,
            (fz - fx[:, None]) / one_minus_z,
            0.0,
        )
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
            - (np.pi ** 2 / 3.0 + 4.5) * fx
        )
        c2q_total = c2q_total + e2 * pref * cq

    fg = grid.xfq(21, xo)                                          # (n_x, N)
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


# 2. PDF SETS  (label, LHAPDF set name, color, linestyle, has_band)
PDF_SPECS = [
    ("JAM21",   "JAM21PionPDFnlo", "#0a7cff", "-",  True),   # bright blue
    ("xFitter", "xFitterPI_NLO_VAR",  "#ff1744", "--", True),   # bright red
    ("GRV",     "GRVPI1",             "#00c853", ":",  False),  # bright green
]

Q2_TARGET = 10.0
x_grid = np.logspace(-3, np.log10(0.999), 60)


def all_members(pset):
    if pset.errorType == "replicas":
        return [pset.mkPDF(i) for i in range(pset.size)]
    return pset.mkPDFs()


def build_band(pset, curves):
    """(central, lo, hi) per x, dispatching on error type.
    replicas/(sym)hessian -> LHAPDF's prescription; variations -> envelope.
    """
    n = curves.shape[1]
    central = np.empty(n)
    lo = np.empty(n)
    hi = np.empty(n)

    if pset.errorType == "variations":
        for j in range(n):
            col = curves[:, j]
            central[j] = col[0]          # member 0 is the central value
            lo[j] = col.min()
            hi[j] = col.max()
    else:
        for j in range(n):
            unc = pset.uncertainty(list(curves[:, j]), cl=68)
            central[j] = unc.central
            lo[j] = unc.central - unc.errminus
            hi[j] = unc.central + unc.errplus

    return central, lo, hi


def compute_f2_with_band(set_name, has_band):
    if has_band:
        pset = lhapdf.getPDFSet(set_name)
        members = all_members(pset)
        print(f"{set_name}: {len(members)} members ({pset.errorType}), GPU batch per member.")

        curves = np.empty((len(members), len(x_grid)))
        for i, mem in enumerate(members):
            grid = GpuPdfGrid(mem, Q2_TARGET)          # CPU: LHAPDF grid build
            curves[i] = f2_nlo_vec(mem, grid, x_grid, Q2_TARGET)  # GPU: batched
            print(f"  {set_name} member {i + 1}/{len(members)} done.")

        return build_band(pset, curves)
    else:
        pdf = lhapdf.mkPDF(set_name, 0)
        print(f"{set_name}: central member only, no error set available.")
        grid = GpuPdfGrid(pdf, Q2_TARGET)
        central = f2_nlo_vec(pdf, grid, x_grid, Q2_TARGET)
        return central, None, None


def main():
    # 3. COMPUTE
    results = {}
    for label, set_name, color, ls, has_band in PDF_SPECS:
        cen, lo, hi = compute_f2_with_band(set_name, has_band)
        results[label] = dict(central=cen, lo=lo, hi=hi, color=color, ls=ls)

    # 4. PLOT  (filled bands only -- no central line, no boundary edges)
    BAND_ALPHA = 0.50   # the band fill itself is the bright element

    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    legend_handles = []
    for label, set_name, color, ls, has_band in PDF_SPECS:
        r = results[label]
        if r["lo"] is not None:
            ax.fill_between(x_grid, r["lo"], r["hi"], color=color,
                            alpha=BAND_ALPHA, linewidth=0, zorder=2)
            legend_handles.append(
                Patch(facecolor=color, alpha=BAND_ALPHA, edgecolor="none", label=label))
        else:
            ax.plot(x_grid, r["central"], color=color, lw=2.6, ls=ls, zorder=3)
            legend_handles.append(
                Line2D([0], [0], color=color, lw=2.6, ls=ls, label=label))

    ax.set_xscale("log")
    ax.set_xlim(1e-3, 1.0)
    ax.set_ylim(0, 0.7)
    ax.set_xlabel(r"$\mathbf{x_{\pi}}$", fontsize=20, fontweight="bold")
    ax.set_ylabel(r"$\mathbf{F_2(x_{\pi},Q^2)}$", fontsize=20, fontweight="bold")

    ax.grid(True, which="both", ls="--", alpha=0.3)
    yticks = [t for t in ax.get_yticks() if t > 0]
    ax.set_yticks(yticks)
    _bold_ticklabels(ax)

    ax.text(
        0.05, 0.06,
        rf"$\mathbf{{Q^2 = {Q2_TARGET:g}\,GeV^2}}$",
        transform=ax.transAxes,
        fontsize=15,
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="black", linewidth=1.2),
    )

    leg = ax.legend(handles=legend_handles, loc="upper right", frameon=True,
                    edgecolor="black", fancybox=False, fontsize=11,
                    handlelength=1.8, labelspacing=0.35, borderpad=0.4)
    leg.get_frame().set_linewidth(1.2)
    for text in leg.get_texts():
        text.set_fontweight("bold")

    plt.tight_layout()
    fig.savefig("F2pi_JAM21_xFitter_GRV_Q2_10.pdf", bbox_inches="tight",
                facecolor="white")
    fig.savefig("F2pi_JAM21_xFitter_GRV_Q2_10.png", dpi=300, bbox_inches="tight",
                facecolor="white")

    print("Saved: F2pi_JAM21_xFitter_GRV_Q2_10.pdf / .png")


if __name__ == "__main__":
    main()
