#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H1 part of Fig. 9: three pion PDFs times five Pythia settings.

The physics and plotting code is taken from h1_five_settings.py; the Pythia part
comes from hists/H1_*.npz. --mode previous uses the inputs of the submitted version,
--mode new the JAM21 replica mean and the refitted Lambda for xFitter and GRV.
"""
import os, sys


# from h1_five_settings.py: imports, style, constants, binning, helpers

import argparse, os, warnings, gc
PYTHIA_OUT = os.environ.get("PYTHIA_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Pythia", "output"))
from functools import lru_cache

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib.ticker import AutoMinorLocator, FuncFormatter
import numpy as np
import uproot
import lhapdf
from scipy.integrate import quad

warnings.filterwarnings('ignore')

# FONT SIZES / WEIGHTS -- house style.
# Scaled up for the large 4.8 x 4.0 inch staircase panels.
FS_AXLABEL = 42     # F_2^LN(...) and x_L axis labels
FS_TITLE   = 28     # Q^2 column titles
FS_ROWLAB  = 26     # x_B row labels
FS_TICK    = 24     # x and y tick numbers
FS_LEGEND  = 24     # the one shared corner legend

SPINE_LW   = 3.2    # panel box -- deliberately heavy
LW_PYTHIA  = 2.6    # raw Pythia (DIS background) histograms
LW_JAM     = 3.2    # raw JAM21 pion histogram
LW_SUM     = 3.2    # Pythia + JAM21 sums
DATA_MS    = 7.0    # H1 marker size
DATA_MEW   = 1.5
DATA_ELW   = 2.0    # error-bar line width
DATA_CAPS  = 3.5

matplotlib.rcParams.update({
    "font.family"         : "serif",
    "font.serif"          : ["Times New Roman", "DejaVu Serif"],
    "font.weight"         : "bold",
    "mathtext.fontset"    : "dejavuserif",
    "mathtext.default"    : "bfit",      # bold *italic* math (H1 style)
    "axes.labelsize"      : FS_ROWLAB,
    "axes.labelweight"    : "bold",
    "axes.titlesize"      : FS_TITLE,
    "xtick.labelsize"     : FS_TICK,
    "ytick.labelsize"     : FS_TICK,
    "xtick.direction"     : "in",
    "ytick.direction"     : "in",
    "xtick.top"           : True,
    "ytick.right"         : True,
    "xtick.major.size"    : 7,
    "ytick.major.size"    : 7,
    "xtick.minor.size"    : 3.5,
    "ytick.minor.size"    : 3.5,
    "xtick.minor.visible" : True,
    "ytick.minor.visible" : True,
    "legend.framealpha"   : 0.92,
    "legend.edgecolor"    : "0.45",
    "legend.fontsize"     : FS_LEGEND,
    "figure.dpi"          : 150,
    "axes.linewidth"      : 3.2,     # bold panel boundaries
})

# CONSTANTS
N_TOTAL = 500_000_000

C_DEFAULT = "#1f77b4"    # blue   - Pythia default PDF
C_CT18    = "#d62728"    # red    - Pythia CT18 NLO
C_CT18_REMNANT = "#9467bd" # purple - Pythia CT18 NLO + Remnant
C_CT18_PTUNE = "#ff7f0e" # orange - Pythia CT18NLO+pythia Tune
C_CT18_GLUON = "#17becf" # cyan  - Pythia CT18NLO+gluon
C_DATA    = "black"

# Extra stroke width applied to the x / y axis labels so they render heavier
# WITHOUT changing their font size. Raise for even bolder, 0.0 to disable.
AXIS_LABEL_STROKE = 1.1
C_JAM     = "navy"       # JAM21 raw histogram (no normalisation) - shared

# Ordered list of (key, display label, color) used throughout
PDF_SETS = [
    ("default",     "Pythia default",           C_DEFAULT),
    ("ct18",        "CT18 NLO",                  C_CT18),
    ("ct18remnant", "CT18 NLO + Remnant",        C_CT18_REMNANT),
    ("ct18ptune",   "CT18 NLO + Tuning",         C_CT18_PTUNE),
    ("ct18gluon",   "CT18 NLO + Gluon",          C_CT18_GLUON),
]

# Ordered list of (key, LHAPDF-set-name, display label, color) for the
# pion PDF used to build the JAM21-style pi-N flux x F2^pi curve.
# Only JAM21 is used - all other pion PDFs have been removed.
PION_PDF_SETS = [
    ("jam21", "JAM21PionPDFnlo_pT", "JAM21", "navy"),
]

# The publication overview grid needs all THREE pion PDFs.
# Kept separate so the JAM21-only grids above are unchanged.
PION_PDF_SETS_OVERVIEW = [
    ("jam21",   "JAM21PionPDFnlo",   "JAM21",   "#1f77b4"),  # blue
    ("xfitter", "xFitterPI_NLO_VAR", "xFitter", "#ff7f0e"),  # orange
    ("grv",     "GRVPI1",            "GRV",     "#9467bd"),  # purple
]

ALPHA      = 1.0 / 137.036
GEV2_TO_NB = 0.3893793656e6
E_E        = 27.5
E_P        = 920.0
S          = 4.0 * E_E * E_P

Q2_MIN, Q2_MAX = 6.0,    100.0
X_MIN,  X_MAX  = 1.5e-4, 3.0e-2
Y_MIN,  Y_MAX  = 0.02,   0.6
W2_MIN         = 4.0
EePrime_MIN    = 11.0
THETA_E_MIN    = 156.0 * np.pi / 180.0
THETA_E_MAX    = 175.0 * np.pi / 180.0
THETA_N_MAX    = 0.75e-3
PT_MAX         = 0.2

Q2_BINS = [
    ( 6.00,   9.00),
    ( 9.00,  13.50),
    (13.50,  20.00),
    (20.00,  30.00),
    (30.00,  45.00),
    (45.00,  67.00),
    (67.00, 100.00),
]
X_BINS = [
    (1.50e-4, 3.20e-4),
    (3.20e-4, 6.82e-4),
    (6.82e-4, 1.45e-3),
    (1.45e-3, 3.10e-3),
    (3.10e-3, 6.60e-3),
    (6.60e-3, 1.41e-2),
    (1.41e-2, 3.00e-2),
]

Q2_MEANS = [7.3, 11.0, 16.0, 24.0, 37.0, 55.0, 82.0]
X_MEANS  = [2.24e-4, 4.78e-4, 1.02e-3, 2.17e-3, 4.63e-3, 9.87e-3, 2.10e-2]

NQ = len(Q2_BINS)
NX = len(X_BINS)

XL_EDGES = np.array([0.32, 0.41, 0.50, 0.59, 0.68, 0.77, 0.86, 0.95])
XL_CTRS  = 0.5 * (XL_EDGES[:-1] + XL_EDGES[1:])
XL_XERR  = 0.5 * (XL_EDGES[1:]  - XL_EDGES[:-1])
XL_BWS   = XL_EDGES[1:] - XL_EDGES[:-1]
N_XL     = len(XL_CTRS)   # 7


# HELPERS
def pick(av, candidates):
    for c in candidates:
        if c in av: return c
    return None

def sci(v):
    m, e = f"{v:.2e}".split("e")
    return rf"{float(m):.2f}\!\times\!10^{{{int(e)}}}"

def q2_label(lo, hi):
    return rf"$Q^2\!\in\![{lo:.4g},\,{hi:.4g}]\ \mathrm{{GeV}}^2$"

def xb_label(lo, hi):
    return rf"$x_B\!\in\![{sci(lo)},\,{sci(hi)}]$"

def compute_y(Q2, x):
    return Q2 / (S * x)

def dis_prefactor(x, Q2):
    return (4.0 * np.pi * ALPHA**2 / (x * Q2**2)) * GEV2_TO_NB

def dsig_to_f2ln(dsig_per_xl, x, Q2, dx, dQ2):
    y   = compute_y(Q2, x)
    kin = 1.0 - y + 0.5 * y**2
    P   = dis_prefactor(x, Q2)
    return np.asarray(dsig_per_xl, dtype=float) / (P * kin * dx * dQ2)


# from h1_five_settings.py: H1 data

# H1 DATA
def load_hdat_f2ln(path):
    raw  = np.loadtxt(path)
    Q2m  = raw[:, 0];  xBm = raw[:, 1];  xLc = raw[:, 2]
    F2   = raw[:, 4];  stat = raw[:, 5]; syst = raw[:, 6]
    tot  = np.hypot(stat, syst)

    def q2_to_iq(q2):
        for i, (lo, hi) in enumerate(Q2_BINS):
            if lo <= q2 <= hi: return i
        for i, qm in enumerate(Q2_MEANS):
            if abs(q2 - qm) / qm < 0.05: return i
        return None

    def xb_to_ix(xb):
        for i, (lo, hi) in enumerate(X_BINS):
            if lo <= xb <= hi: return i
        for i, xm in enumerate(X_MEANS):
            if abs(xb - xm) / xm < 0.05: return i
        return None

    data = {}
    for k in range(len(raw)):
        iq = q2_to_iq(Q2m[k]);  ix = xb_to_ix(xBm[k])
        if iq is None or ix is None:
            print(f"  WARNING: unrecognised Q2={Q2m[k]} or xB={xBm[k]} — skipped")
            continue
        if (iq, ix) not in data:
            data[(iq, ix)] = [[], [], [], []]
        data[(iq, ix)][0].append(xLc[k])
        data[(iq, ix)][1].append(F2[k])
        data[(iq, ix)][2].append(stat[k])
        data[(iq, ix)][3].append(tot[k])

    return {k: tuple(np.array(v) for v in vv) for k, vv in data.items()}


# from h1_five_settings.py: NLO F2^pi-, s-exp flux (Lambda = 1.31)

_CF = 4.0 / 3.0;  _TR = 0.5

def _Cq2_singular(z):
    if z <= 0.0 or z >= 1.0: return 0.0
    return _CF * (1.0 + z**2) * (np.log((1.0 - z) / z) - 0.75)

def _Cq2_regular(z):
    return _CF * (1.5 + 2.0 * z)

_Cq2_DELTA = _CF * (np.pi**2 / 3.0 - 9.0 / 2.0)

def _Cg2_kernel(z):
    if z <= 0.0 or z >= 1.0: return 0.0
    split = (z**2 + (1.0 - z)**2) * np.log((1.0 - z) / z)
    poly  = -1.0 + 8.0 * z * (1.0 - z)
    return 2.0 * _TR * (split + poly)

_E2_u = (2/3)**2;  _E2_d = (1/3)**2;  _E2_s = (1/3)**2
_E2_c = (2/3)**2;  _E2_b = (1/3)**2
_CHARGE_SQ_SUM = _E2_u + _E2_d + _E2_s + _E2_c + _E2_b

_FLAVOR_MAP_PIMINUS = [(1,-1,_E2_u),(2,-2,_E2_d),(3,-3,_E2_s),(4,-4,_E2_c),(5,-5,_E2_b)]

def _xfq(pdf, pid, x, Q2):
    if x <= 0.0 or x >= 1.0: return 0.0
    return pdf.xfxQ2(pid, float(x), float(Q2))

def compute_f2pi_minus_allflav_nlo(x, pdf, Q2):
    if x <= 0.0 or x >= 1.0: return 0.0
    alpha_s = pdf.alphasQ(np.sqrt(Q2))
    pref = alpha_s / (2.0 * np.pi)
    F2   = 0.0
    def integ_bdy(z): return _Cq2_singular(z) / (1.0 - z)
    bdy_int, _ = quad(integ_bdy, 0.0, x, limit=60)
    for pid_q, pid_qb, e2 in _FLAVOR_MAP_PIMINUS:
        xqq = _xfq(pdf, pid_q, x, Q2) + _xfq(pdf, pid_qb, x, Q2)
        F2 += e2 * xqq
        def integ_plus(z, _xqq=xqq, _pq=pid_q, _pqb=pid_qb):
            xz = x / z
            fv = (_xfq(pdf, _pq, xz, Q2) + _xfq(pdf, _pqb, xz, Q2)) if xz < 1.0 else 0.0
            return _Cq2_singular(z) * (fv - _xqq) / (1.0 - z)
        plus_c, _ = quad(integ_plus, x, 1.0 - 1e-6, limit=60)
        I_plus = plus_c - xqq * bdy_int
        def integ_reg(z, _pq=pid_q, _pqb=pid_qb):
            xz = x / z
            if xz >= 1.0: return 0.0
            return _Cq2_regular(z) * (_xfq(pdf, _pq, xz, Q2) + _xfq(pdf, _pqb, xz, Q2))
        reg_c, _  = quad(integ_reg, x, 1.0 - 1e-6, limit=60)
        I_delta   = _Cq2_DELTA * xqq
        F2 += e2 * pref * (I_plus + reg_c + I_delta)
    def integ_gluon(z):
        xz = x / z
        if xz >= 1.0: return 0.0
        return _Cg2_kernel(z) * _xfq(pdf, 21, xz, Q2)
    gluon_c, _ = quad(integ_gluon, x, 1.0 - 1e-6, limit=100)
    F2 += pref * _CHARGE_SQ_SUM * gluon_c
    return F2

_M_N=0.9382720813; _M_PI=0.13957039; _G_A=1.267; _F_PI=0.093
_LAMBDA_CUT=1.31;  _K2_MAX=0.04

def _d_piN(k2, xL):
    xbar = 1.0 - xL
    return -(k2 + xbar*xbar*_M_N*_M_N + xL*_M_PI*_M_PI) / xL

@lru_cache(maxsize=512)
def f_piN_model_i(xL):
    if xL <= 0.0 or xL >= 1.0: return 0.0
    xbar = 1.0 - xL
    pref = (_G_A*_G_A*_M_N*_M_N) / ((4.0*np.pi*_F_PI)**2)
    def integrand(k2):
        dval  = _d_piN(k2, xL)
        sval  = (k2+_M_N*_M_N)/xL + (k2+_M_PI*_M_PI)/xbar
        ff    = np.exp((_M_N*_M_N - sval)/(_LAMBDA_CUT*_LAMBDA_CUT))
        numer = xbar*(k2+xbar*xbar*_M_N*_M_N)
        denom = xL*xL*dval*dval
        return (numer/denom)*(ff*ff)
    val, _ = quad(integrand, 0.0, _K2_MAX, limit=220)
    return pref * val


# from h1_five_settings.py: pion F2^LN histograms (f2pi_nlo)

# GENERIC PION-PDF LN FLUX x F2^pi  (works for any LHAPDF pion set)
# compute_f2pi_minus_allflav_nlo() and f_piN_model_i() above already take a
# generic `pdf` object / are PDF-independent, so the same physics is reused
# here for MAP23Pion_PRVnlo, FANTO10_n15, xFitterPI_NLO_VAR, GRVPI1, etc.
# Only the rebinned histogram (not the raw fine grid) is needed downstream
# for plotting/fitting, so we skip the raw-grid step and rebin directly.
def _pion_f2ln_pointwise(pdf_obj, xL, xbj, q2):
    x_pi = xbj / (1.0 - xL)
    if x_pi <= 0.0 or x_pi >= 1.0:
        return 0.0
    return 2.0 * f_piN_model_i(float(xL)) * compute_f2pi_minus_allflav_nlo(float(x_pi), pdf_obj, q2)


try:
    import f2pi_nlo
    _HAVE_GPU_PATH = True
except Exception as _e:                                  # pragma: no cover
    print(f"  NOTE: f2pi_nlo unavailable ({_e}); using the slow CPU path.")
    _HAVE_GPU_PATH = False

# Set to False to force the original scipy.quad implementation.
USE_FAST_F2PI = True


def rebin_pion_to_xl_edges(pdf_obj, n_sub=12, label="pion"):
    """Batched (GPU/vectorised) rebin, falling back to the original
    scipy.quad loop if f2pi_nlo can't be imported or USE_FAST_F2PI is off.
    """
    if USE_FAST_F2PI and _HAVE_GPU_PATH:
        print(f"    [{label}] backend: {f2pi_nlo.backend_name()}", flush=True)
        return f2pi_nlo.rebin_pion_to_xl_edges_fast(
            pdf_obj, X_MEANS, Q2_MEANS, XL_EDGES, N_XL,
            f_piN_model_i, n_sub=n_sub, label=label)
    return rebin_pion_to_xl_edges_cpu(pdf_obj, n_sub=n_sub)


def rebin_pion_to_xl_edges_cpu(pdf_obj, n_sub=12):
    """Original per-point scipy.quad implementation. Kept as the reference
    the batched path is verified against.
    """
    jam_hist = {}
    for ix, xbj in enumerate(X_MEANS):
        for iq, q2 in enumerate(Q2_MEANS):
            vals = np.zeros(N_XL)
            for i in range(N_XL):
                lo, hi = XL_EDGES[i], XL_EDGES[i+1]
                sub    = lo + (np.arange(n_sub)+0.5)*(hi-lo)/n_sub
                yv     = np.array([_pion_f2ln_pointwise(pdf_obj, float(xL), xbj, q2) for xL in sub])
                vals[i] = yv.mean()
            jam_hist[(iq, ix)] = vals
    return jam_hist


def load_pion_pdf_hists(pion_pdf_sets=PION_PDF_SETS, skip_keys=()):
    """Loads every pion PDF in `pion_pdf_sets` via LHAPDF (member 0) and builds
    its rebinned F2LN histogram per (iq, ix) cell.
    """
    hists = {}
    for key, pdfname, label, _ in pion_pdf_sets:
        if key in skip_keys:
            continue
        print(f"\nLoading pion PDF '{label}' ({pdfname}) ...")
        try:
            pdf_obj = lhapdf.mkPDF(pdfname, 0)
        except Exception as e:
            print(f"  WARNING: could not load {pdfname}: {e}  — skipped")
            continue
        hists[key] = rebin_pion_to_xl_edges(pdf_obj, label=label)
        print(f"  → rebinned onto {N_XL}-bin x_L histogram for {NQ*NX} cells")
    return hists


# from h1_five_settings.py: cell averages

# COMBINED OVERVIEW  -  every (pion PDF x Pythia PDF) combination
def average_over_cells(model_dict, h1_data):
    """Simple unweighted average of a per-cell model dict over every cell
    that also has H1 data, giving one representative x_L curve.
    """
    total = np.zeros(N_XL); n = 0
    for cell, model in model_dict.items():
        if cell in h1_data:
            total += model; n += 1
    return total / n if n else np.full(N_XL, np.nan)


def average_h1(h1_data):
    """Inverse-variance-weighted average of all H1 points, rebinned onto
    XL_EDGES, collapsing every (Q2, x_B) cell onto one representative curve.
    """
    sum_wf = np.zeros(N_XL); sum_w = np.zeros(N_XL)
    for xL_h1, F2_h1, _stat, tot_h1 in h1_data.values():
        idx = np.clip(np.searchsorted(XL_EDGES[1:], xL_h1, side='left'), 0, N_XL - 1)
        w = np.where(tot_h1 > 0, 1.0 / tot_h1**2, 0.0)
        for j, i in zip(range(len(idx)), idx):
            sum_wf[i] += w[j] * F2_h1[j]
            sum_w[i]  += w[j]
    avg = np.divide(sum_wf, sum_w, out=np.full(N_XL, np.nan), where=sum_w > 0)
    err = np.divide(1.0, np.sqrt(sum_w, out=np.full(N_XL, np.nan), where=sum_w > 0),
                     out=np.full(N_XL, np.nan), where=sum_w > 0)
    return avg, err


# from h1_five_settings.py: publication overview grid

# ONE colour scheme, reused identically in every panel (deliberately NOT
# coloured per Pythia set -- the columns are told apart by their headings,
# not by hue). Line styles differ too, so it survives greyscale printing.
C_DIS_OV   = "#1f77b4"   # DIS fragmentation  (Pythia proton PDF)
C_PDFAV_OV = "#2ca02c"   # PDF average        (pion PDF)
C_TOTAL_OV = "#d62728"   # Total = DIS + PDF

LS_DIS_OV, LS_PDFAV_OV, LS_TOTAL_OV = "--", "-.", "-"

# The 7x7 grids can afford FS_TITLE=28, but five long column headings
# ("CT18 NLO + Remnant" ...) collide at that size, so this grid uses its own.
FS_TITLE_OV = 22

# The chi^2/N box is an annotation, not a heading -- at FS_LEGEND (24) it
# dominates the panel and crowds the curves, so it gets its own size.
FS_CHI2_OV = 15

# H1 F_2^LN reaches ~0.12; 0.14 ceiling.
OVERVIEW_YLIM   = (0.0, 0.14)
OVERVIEW_XTICKS = [0.4, 0.6, 0.8]          # interior ticks only: with
OVERVIEW_YTICKS = [0.03, 0.06, 0.09, 0.12]  # wspace=hspace=0 the panels
                                            # touch, so an edge tick would
                                            # collide with the neighbour.


def _ov_tick_fmt(v, pos):
    """Bold tick labels; whole numbers drop their trailing decimals."""
    if abs(v - round(v)) < 1e-9:
        return rf"$\mathbf{{{int(round(v))}}}$"
    return rf"$\mathbf{{{v:.2f}}}$"


def _average_pair_over_cells(py_dict, pion_dict, h1_data):
    """Cell-average the Pythia (DIS fragmentation) and pion (PDF average)
    F_2^LN histograms over every (Q^2, x_B) cell that carries an H1
    measurement, counting -- in each x_L bin independently -- only the
    cells where BOTH curves are finite.
    """
    s_dis = np.zeros(N_XL)
    s_pdf = np.zeros(N_XL)
    n_cnt = np.zeros(N_XL)

    for cell in h1_data:
        py = py_dict.get(cell) if py_dict else None
        h  = pion_dict.get(cell) if pion_dict else None
        if py is None or h is None:
            continue
        py = np.asarray(py, dtype=float)
        h  = np.asarray(h,  dtype=float)
        good = np.isfinite(py) & np.isfinite(h)
        if not good.any():
            continue
        s_dis[good] += py[good]
        s_pdf[good] += h[good]
        n_cnt[good] += 1.0

    with np.errstate(invalid="ignore", divide="ignore"):
        safe = np.where(n_cnt > 0, n_cnt, 1.0)
        dis_avg = np.where(n_cnt > 0, s_dis / safe, np.nan)
        pdf_avg = np.where(n_cnt > 0, s_pdf / safe, np.nan)
    return dis_avg, pdf_avg, dis_avg + pdf_avg


def _chi2_total_over_cells(py_dict, pion_dict, h1_data):
    """Overall chi^2 and N of the H1 points against the per-cell Total
    (Pythia + pion), accumulated over the whole grid, using the TOTAL
    (stat (+) syst) error as the weight.
    """
    chi2 = 0.0
    ndof = 0
    for cell, h1_cell in h1_data.items():
        if h1_cell is None:
            continue
        xL_h1, F2_h1, _stat_h1, tot_h1 = h1_cell
        py = py_dict.get(cell) if py_dict else None
        h  = pion_dict.get(cell) if pion_dict else None
        if py is None or h is None:
            continue

        F2_h1  = np.asarray(F2_h1,  dtype=float)
        tot_h1 = np.asarray(tot_h1, dtype=float)
        combined = np.asarray(py, dtype=float) + np.where(np.isfinite(h), h, 0.0)

        bin_idx = np.clip(np.digitize(xL_h1, XL_EDGES) - 1, 0, N_XL - 1)
        model = combined[bin_idx]

        valid = np.isfinite(model) & np.isfinite(F2_h1) & (tot_h1 > 0)
        if not valid.any():
            continue
        chi2 += float(np.sum(((F2_h1[valid] - model[valid]) / tot_h1[valid]) ** 2))
        ndof += int(valid.sum())
    return chi2, ndof


def make_publication_overview_grid(pythia_dict, pion_hists, h1_data, out,
                                   pion_pdf_sets=None):
    """Publication grid: pion PDFs (rows) x Pythia proton PDFs (columns), vs H1."""
    from matplotlib.transforms import Bbox

    if pion_pdf_sets is None:
        pion_pdf_sets = PION_PDF_SETS_OVERVIEW

    pythia_labels = list(pythia_dict.keys())
    n_py   = len(pythia_labels)
    n_pion = len(pion_pdf_sets)
    if n_py == 0 or n_pion == 0:
        print("  nothing to draw -- overview grid skipped")
        return

    h1_avg, h1_err = average_h1(h1_data)
    hvalid = np.isfinite(h1_avg) & np.isfinite(h1_err)

    # Panels touch edge-to-edge (H1 staircase convention). Panel size matches
    # the 7x7 grids, so the shared FS_* sizes stay consistent between figures.
    fig, axes = plt.subplots(
        n_pion, n_py,
        figsize=(n_py * 4.8, n_pion * 4.0),
        gridspec_kw=dict(wspace=0.0, hspace=0.0),
        sharex=True, sharey=True, squeeze=False)
    fig.patch.set_facecolor("white")
    # Default subplot margins leave ~12% blank above the titles, which would
    # open a dead gap under the shared legend.
    fig.subplots_adjust(left=0.085, right=0.985, top=0.935, bottom=0.075,
                        wspace=0.0, hspace=0.0)

    legend_src = None
    row_label_artists = []

    for ip, (pion_key, _pdfname, pion_label, _pion_color) in enumerate(pion_pdf_sets):
        pion_dict = pion_hists.get(pion_key)

        for jp, py_label in enumerate(pythia_labels):
            ax = axes[ip, jp]
            ax.set_facecolor("white")
            py_dict = pythia_dict[py_label]

            chi2, ndof = 0.0, 0

            if pion_dict is not None:
                dis_avg, pdf_avg, tot_avg = _average_pair_over_cells(
                    py_dict, pion_dict, h1_data)

                ax.stairs(dis_avg, XL_EDGES, color=C_DIS_OV, lw=LW_PYTHIA,
                          ls=LS_DIS_OV, label="DIS fragmentation")
                ax.stairs(pdf_avg, XL_EDGES, color=C_PDFAV_OV, lw=LW_JAM,
                          ls=LS_PDFAV_OV, label="Sullivan Process")
                ax.stairs(tot_avg, XL_EDGES, color=C_TOTAL_OV, lw=LW_SUM,
                          ls=LS_TOTAL_OV, label="Total")

                chi2, ndof = _chi2_total_over_cells(py_dict, pion_dict, h1_data)

            if hvalid.any():
                ax.errorbar(XL_CTRS[hvalid], h1_avg[hvalid],
                            xerr=XL_XERR[hvalid], yerr=h1_err[hvalid],
                            fmt="o", ms=DATA_MS, mfc="black", mec="black",
                            mew=DATA_MEW, ecolor="black", elinewidth=DATA_ELW,
                            capsize=DATA_CAPS, capthick=DATA_ELW,
                            color="black", zorder=10, label="H1 (avg)")

            if legend_src is None and pion_dict is not None:
                legend_src = ax

            # the ONLY per-panel annotation: chi^2/N
            if ndof > 0:
                ax.text(0.965, 0.955, rf"$\chi^2/N = {chi2 / ndof:.2f}$",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=FS_CHI2_OV, fontweight="bold", zorder=12,
                        bbox=dict(boxstyle="round,pad=0.24", facecolor="white",
                                  edgecolor="0.45", alpha=0.92, linewidth=1.0))

            # axes, ticks, spines
            ax.set_xlim(XL_EDGES[0], XL_EDGES[-1])
            ax.set_ylim(*OVERVIEW_YLIM)
            ax.set_xticks(OVERVIEW_XTICKS)
            ax.set_yticks(OVERVIEW_YTICKS)
            ax.xaxis.set_major_formatter(FuncFormatter(_ov_tick_fmt))
            ax.yaxis.set_major_formatter(FuncFormatter(_ov_tick_fmt))
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.tick_params(which="both", direction="in", top=True, right=True,
                           labelsize=FS_TICK)
            ax.tick_params(which="major", length=10, width=2.0, pad=6)
            ax.tick_params(which="minor", length=5, width=1.4)
            ax.tick_params(labelbottom=(ip == n_pion - 1), labelleft=(jp == 0))
            plt.setp(ax.get_xticklabels(), fontweight="bold")
            plt.setp(ax.get_yticklabels(), fontweight="bold")
            for sp in ax.spines.values():
                sp.set_linewidth(SPINE_LW)

            # column heading: Pythia proton PDF set (top row only)
            if ip == 0:
                ax.set_title(py_label, fontsize=FS_TITLE_OV,
                             fontweight="bold", pad=10)

            # row heading: pion PDF name (left column only)
            if jp == 0:
                row_label_artists.append(
                    ax.text(-0.30, 0.5, pion_label, transform=ax.transAxes,
                            ha="center", va="center", rotation=90,
                            fontsize=FS_ROWLAB, fontweight="bold"))

    # ---- ONE shared legend, above the grid (never repeated per panel) --
    if legend_src is not None:
        handles, labels = legend_src.get_legend_handles_labels()
        seen, h2, l2 = set(), [], []
        for hh, ll in zip(handles, labels):
            if ll not in seen:
                seen.add(ll)
                h2.append(hh)
                l2.append(ll)
        if h2:
            leg = fig.legend(h2, l2, loc="lower center",
                             bbox_to_anchor=(0.5, 1.004), ncol=len(l2),
                             handlelength=2.4, handletextpad=0.7,
                             borderpad=0.8, columnspacing=2.2,
                             framealpha=0.96, edgecolor="0.35",
                             prop={"weight": "bold", "size": FS_LEGEND})
            leg.get_frame().set_facecolor("white")
            leg.get_frame().set_linewidth(1.2)

    # universal axis labels, hugging the panel block
    all_axes = list(axes.flat)
    pos = [ax.get_position() for ax in all_axes]
    cx = 0.5 * (min(p.x0 for p in pos) + max(p.x1 for p in pos))
    cy = 0.5 * (min(p.y0 for p in pos) + max(p.y1 for p in pos))
    renderer = None
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()
        union = Bbox.union([ax.get_tightbbox(renderer) for ax in all_axes])
        ux0, uy0 = inv.transform((union.x0, union.y0))
    except Exception:
        ux0 = min(p.x0 for p in pos) - 0.03
        uy0 = min(p.y0 for p in pos) - 0.03

    pad = 0.012
    # Same font size, stroked so the glyphs render heavier (mathtext has no
    # weight above bold) -- identical trick to the 7x7 grids.
    _lbl_stroke = [patheffects.withStroke(linewidth=AXIS_LABEL_STROKE,
                                          foreground="black")]

    t_x = fig.text(cx, uy0 - pad, r"$x_L$", ha="center", va="top",
                   fontsize=FS_AXLABEL, fontweight="bold")
    t_x.set_path_effects(_lbl_stroke)

    # The row headings (JAM21 / xFitter / GRV) live in the left margin, so
    # anchor the universal y label outside THEM, not merely outside the axes.
    y_anchor = ux0
    if renderer is not None:
        try:
            inv = fig.transFigure.inverted()
            for t in row_label_artists:
                bb = t.get_window_extent(renderer).transformed(inv)
                y_anchor = min(y_anchor, bb.x0)
        except Exception:
            pass

    t_y = fig.text(y_anchor - pad - 0.008, cy,
                   r"$F_{2}^{LN}(x_B,\,Q^{2},\,x_L)$",
                   ha="center", va="center", rotation=90,
                   fontsize=FS_AXLABEL, fontweight="bold")
    t_y.set_path_effects(_lbl_stroke)

    for ext in ("png", "pdf"):
        fname = f"{out}_all_combinations_overview.{ext}"
        fig.savefig(fname, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  → {fname}")
    plt.close(fig)


# Revised analysis: Pythia from the count histograms, new pion inputs
import csv as _csv

HERE_ = os.path.dirname(os.path.abspath(__file__))
import lambda_fit as _ZLF                      # noqa: E402

HIST_DIR = os.path.join(HERE_, "..", "inputs", "pythia_histograms")
H1_DAT_PATH = os.path.join(HERE_, "..", "data", "h1.dat")
FIT_CSV = os.path.join(HERE_, "..", "lambda_fit", "lambda_fit_results.csv")
H1_HISTS = [("default", "Pythia default"), ("ct18", "CT18 NLO"), ("ct18remnant", "CT18 NLO + Remnant"),
            ("ct18ptune", "CT18 NLO + Tuning"), ("ct18gluon", "CT18 NLO + Gluon")]
JAM21_TABLE1_LAMBDA = 1.31
# chi^2/N printed in the overview figure of the submitted version
ORIGINAL_CHI2N = {"jam21": [5.81, 5.99, 5.95, 17.25, 5.35],
                  "xfitter": [1.96, 2.01, 5.05, 6.96, 1.92],
                  "grv": [8.04, 8.28, 7.21, 19.49, 7.52]}


def pythia_f2ln_from_hist(npz_path):
    """The tail of h1_five_settings.py's process_pythia(), applied to the stored counts."""
    z = np.load(npz_path)
    assert np.allclose(z["xl_edges"], XL_EDGES), "x_L binning differs from h1_five_settings.py"
    counts, sigma_dis = z["counts"], float(z["sigma_dis"])
    f2ln_py = {}
    for iq, (q2lo, q2hi) in enumerate(Q2_BINS):
        q2_mean = Q2_MEANS[iq]
        dQ2 = q2hi - q2lo
        for ix, (xlo, xhi) in enumerate(X_BINS):
            x_mean = X_MEANS[ix]
            dx = xhi - xlo
            c = counts[iq, ix]
            dsig = sigma_dis * c.astype(float) / N_TOTAL / XL_BWS
            f2ln_py[(iq, ix)] = dsig_to_f2ln(dsig, x_mean, q2_mean, dx, dQ2)
    print(f"  {os.path.basename(npz_path)}: sigma = {sigma_dis:.4f} nb, binned events = {int(counts.sum()):,}")
    return f2ln_py


def make_flux(lam):
    """h1_five_settings.py's f_piN_model_i with the cutoff Lambda as a parameter (identical integrand, K2_MAX = 0.04)."""
    @lru_cache(maxsize=512)
    def f(xL):
        if xL <= 0.0 or xL >= 1.0: return 0.0
        xbar = 1.0 - xL
        pref = (_G_A*_G_A*_M_N*_M_N) / ((4.0*np.pi*_F_PI)**2)
        def integrand(k2):
            dval  = _d_piN(k2, xL)
            sval  = (k2+_M_N*_M_N)/xL + (k2+_M_PI*_M_PI)/xbar
            ff    = np.exp((_M_N*_M_N - sval)/(lam*lam))
            numer = xbar*(k2+xbar*xbar*_M_N*_M_N)
            denom = xL*xL*dval*dval
            return (numer/denom)*(ff*ff)
        val, _ = quad(integrand, 0.0, _K2_MAX, limit=220)
        return pref * val
    return f


class MeanPionPdfGrid(f2pi_nlo.PionPdfGrid):
    """f2pi_nlo.PionPdfGrid holding the replica-MEAN x f(x) (same X_FINE, same log-x interpolation)."""
    def __init__(self, arrays, q2val):
        self.q2 = float(q2val)
        self.log_x = f2pi_nlo.to_gpu(f2pi_nlo._LOG_X_FINE)
        self.grids = {pid: f2pi_nlo.to_gpu(np.asarray(arrays[pid])) for pid in f2pi_nlo.PIDS_NEEDED}


def rebin_fast_grid(pdf_obj, grid_fn, flux_fn, n_sub=12, label=""):
    """f2pi_nlo.rebin_pion_to_xl_edges_fast, except the PDF grid comes from grid_fn(Q2)."""
    sub_pts = []
    for i in range(N_XL):
        lo, hi = XL_EDGES[i], XL_EDGES[i + 1]
        sub_pts.append(lo + (np.arange(n_sub) + 0.5) * (hi - lo) / n_sub)
    sub_pts = np.concatenate(sub_pts)
    flux_grid = np.array([2.0 * float(flux_fn(float(xl))) for xl in sub_pts]).reshape(N_XL, n_sub)
    out = {}
    for iq, q2 in enumerate(Q2_MEANS):
        grid = grid_fn(float(q2))
        x_pi_all = np.concatenate([xbj / (1.0 - sub_pts) for xbj in X_MEANS])
        vals = f2pi_nlo.f2pi_minus_nlo_vec(pdf_obj, grid, x_pi_all, float(q2))
        vals = vals.reshape(len(X_MEANS), N_XL, n_sub) * flux_grid[None, :, :]
        for ix in range(len(X_MEANS)):
            out[(iq, ix)] = vals[ix].mean(axis=1)
        print(f"    [{label}] Q2={q2:g} done", flush=True)
    return out


def best_lambdas():
    """New s-exp Lambda of xFitter and GRV: common H1+ZEUS best fit."""
    out = {}
    with open(FIT_CSV) as fh:
        for r in _csv.DictReader(fh):
            if r["fit"] == "H1+ZEUS_common_Lambda" and r["flux"] == "s_exp":
                out[r["pion_pdf"]] = float(r["Lambda_GeV"])
    return out


def pion_hists_new():
    lam = best_lambdas()
    hists = {}
    jam = "JAM21PionPDFnlo"
    pdf0 = lhapdf.mkPDF(jam, 0)                                   # alpha_s only (same for every replica)
    print(f"\nJAM21 = mean of the {jam} replicas, Lambda = {JAM21_TABLE1_LAMBDA}")
    hists["jam21"] = rebin_fast_grid(
        pdf0, lambda q2: MeanPionPdfGrid(_ZLF._mean_arrays(jam, q2, f2pi_nlo.X_FINE, True), q2),
        make_flux(JAM21_TABLE1_LAMBDA), label="JAM21 mean")
    for key, setn, lab in (("xfitter", "xFitterPI_NLO_VAR", "xFitter"), ("grv", "GRVPI1", "GRV")):
        p0 = lhapdf.mkPDF(setn, 0)
        print(f"\n{lab} = member 0 of {setn}, Lambda = {lam[lab]:.4f}")
        hists[key] = rebin_fast_grid(p0, lambda q2, p=p0: f2pi_nlo.PionPdfGrid(p, q2),
                                     make_flux(lam[lab]), label=lab)
    used = {"JAM21": JAM21_TABLE1_LAMBDA, "xFitter": lam["xFitter"], "GRV": lam["GRV"]}
    return hists, used


def write_chi2_table(pythia_dict, pion_hists, h1_data, path, title, compare):
    py_labels = list(pythia_dict.keys())
    L = [title, "=" * len(title), "",
         f"{'pion PDF':10s} " + " ".join(f"{p:>20s}" for p in py_labels)]
    for key, _pdfname, label, _c in PION_PDF_SETS_OVERVIEW:
        cells = []
        for j, pl in enumerate(py_labels):
            chi2, n = _chi2_total_over_cells(pythia_dict[pl], pion_hists[key], h1_data)
            txt = f"{chi2 / n:.2f}"
            if compare:
                txt += f" (fig {ORIGINAL_CHI2N[key][j]:.2f})"
            cells.append(txt)
        L.append(f"{label:10s} " + " ".join(f"{c:>20s}" for c in cells))
    L.append("")
    L.append("chi^2/N of the H1 points vs Total = Pythia + Sullivan (total error), N = "
             f"{_chi2_total_over_cells(pythia_dict[py_labels[0]], pion_hists['jam21'], h1_data)[1]}")
    txt = "\n".join(L)
    open(path, "w").write(txt + "\n")
    print("\n" + txt)


def run(mode):
    print(f"Loading H1 data from {H1_DAT_PATH} ...")
    h1_data = load_hdat_f2ln(H1_DAT_PATH)
    pythia_dict = {label: pythia_f2ln_from_hist(os.path.join(HIST_DIR, f"H1_{key}.npz"))
                   for key, label in H1_HISTS}
    if mode == "previous":
        print("\nPion PDFs exactly as in h1_five_settings.py: member 0, Lambda = 1.31 for all three")
        pion_hists = load_pion_pdf_hists(PION_PDF_SETS_OVERVIEW)
        title = "H1 overview, PREVIOUS settings (member 0, Lambda = 1.31): chi^2/N   [original figure value]"
    else:
        pion_hists, used = pion_hists_new()
        title = ("H1 overview, NEW settings: JAM21 replica mean (Lambda = 1.31), "
                 f"xFitter Lambda = {used['xFitter']:.3f}, GRV Lambda = {used['GRV']:.3f}: chi^2/N")
    out = os.path.join(HERE_, "..", "cache", f"H1_{mode}")
    make_publication_overview_grid(pythia_dict, pion_hists, h1_data, out)
    write_chi2_table(pythia_dict, pion_hists, h1_data, f"{out}_chi2.txt", title, compare=(mode == "previous"))


if __name__ == "__main__":
    import argparse as _ap
    _p = _ap.ArgumentParser()
    _p.add_argument("--mode", choices=["previous", "new"], required=True)
    run(_p.parse_args().mode)

