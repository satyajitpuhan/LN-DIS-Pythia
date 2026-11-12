#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ZEUS part of Fig. 9: three pion PDFs times five Pythia settings.

The physics and plotting code is taken from zeus_three_pions.py; the Pythia part
comes from hists/ZEUS_*.npz. --mode previous / new as in h1_sullivan.py.
"""


# from zeus_three_pions.py: imports, style, constants, cuts, binning

import argparse, os, sys, ctypes, warnings
PYTHIA_OUT = os.environ.get("PYTHIA_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Pythia", "output"))
from functools import lru_cache

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib.ticker import AutoMinorLocator, FuncFormatter
from matplotlib.transforms import Bbox
from scipy.integrate import quad, IntegrationWarning

warnings.filterwarnings("ignore", category=IntegrationWarning)

# Standalone: ZEUS binning, pion physics, Pythia/ZEUS loaders, rcParams and
# grid helpers are all defined below, as in zeus_five_settings.py.

# FONT SIZES / WEIGHTS -- same house style as zeus_five_settings.py, so
# the ZEUS three-pion-PDF grids come out visually identical to the
# five-PDF-set "DIS background" grids (bold serif, heavy panel boxes,
# large tick/axis labels, thick data markers).
FS_AXLABEL = 42     # F_2^LN(...) and x_L axis labels
FS_TITLE   = 28     # Q^2 column titles
FS_ROWLAB  = 26     # x_B row labels
FS_TICK    = 24     # x and y tick numbers
FS_LEGEND  = 24     # the one shared corner legend

SPINE_LW  = 3.2     # panel box -- deliberately heavy
LW_BKG    = 2.6     # CT18NLO+Remnant DIS-background histogram
LW_PION   = 3.2     # raw pion-PDF histograms (JAM21 / xFitter / GRV)
LW_SUM    = 3.2     # CT18NLO+Remnant + pion sums
DATA_MS   = 7.0     # ZEUS marker size
DATA_MEW  = 1.5
DATA_ELW  = 2.0     # error-bar line width
DATA_CAPS = 3.5

matplotlib.rcParams.update({
    "font.family"        : "serif",
    "font.serif"         : ["Times New Roman", "DejaVu Serif"],
    "font.weight"        : "bold",
    "mathtext.fontset"   : "dejavuserif",
    "mathtext.default"   : "bfit",     # bold *italic* math (H1 style)
    "axes.labelsize"     : FS_ROWLAB,
    "axes.labelweight"   : "bold",
    "axes.titlesize"     : FS_TITLE,
    "axes.titleweight"   : "bold",
    "xtick.labelsize"    : FS_TICK,
    "ytick.labelsize"    : FS_TICK,
    "xtick.direction"    : "in",
    "ytick.direction"    : "in",
    "xtick.top"          : True,
    "ytick.right"        : True,
    "xtick.major.size"   : 7,
    "ytick.major.size"   : 7,
    "xtick.minor.size"   : 3.5,
    "ytick.minor.size"   : 3.5,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.framealpha"  : 0.92,
    "legend.edgecolor"   : "0.45",
    "legend.fontsize"    : FS_LEGEND,
    "figure.dpi"         : 150,
    "axes.linewidth"     : SPINE_LW,     # bold panel boundaries
})

# CONSTANTS
N_TOTAL    = 500_000_000       # N_generated: 5x10^8 events
CHUNK_SIZE = 2_000_000         # tree entries streamed per chunk

C_MC   = "#1f77b4"
C_DATA = "black"

AXIS_LABEL_STROKE = 1.1
TICK_LABEL_STROKE = 0.6


def _stroke(*artists, width=TICK_LABEL_STROKE):
    """Make text noticeably heavier at the SAME font size by stroking the
    glyph outlines.
    """
    if width <= 0:
        return
    fx = [patheffects.withStroke(linewidth=width, foreground="black")]
    for a in artists:
        if a is not None:
            a.set_path_effects(fx)


def bold_ticklabels(ax, width=TICK_LABEL_STROKE):
    """Bold + stroked numeric tick labels on both axes."""
    for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        t.set_fontweight("bold")
        _stroke(t, width=width)


# ---- Pythia ROOT files (used only to look up the CT18NLO+Remnant default
# path in main()) --------------------------------------------------------
PYTHIA_FILES = [
    dict(path=os.path.join(PYTHIA_OUT, "zeus_default.root"),
         label="Pythia default", color="#1f77b4"),
    dict(path=os.path.join(PYTHIA_OUT, "zeus_ct18nlo_tuning.root"),
         label="CT18NLO + Tuning", color="#ff7f0e"),
    dict(path=os.path.join(PYTHIA_OUT, "zeus_ct18nlo.root"),
         label="CT18NLO", color="#d62728"),
    dict(path=os.path.join(PYTHIA_OUT, "zeus_ct18nlo_remnant.root"),
         label="CT18NLO + Remnant", color="#9467bd"),
    dict(path=os.path.join(PYTHIA_OUT, "zeus_ct18nlo_gluon.root"),
         label="CT18NLO + Gluon", color="#17becf"),
]

ALPHA      = 1.0 / 137.036
GEV2_TO_NB = 0.3893793656e6
E_E        = 27.5
E_P        = 820.0             # ZEUS 1995-97 running, NOT HERA-II 920
S          = 4.0 * E_E * E_P   # 90 200 GeV^2

# ZEUS cuts
EePrime_MIN  = 10.0
THETA_N_MAX  = 0.8e-3
PT_COEFF     = 0.656
# NOTE: explicitly NO y cut, NO W^2 cut, NO electron-angle cut.

# x_L binning (ZEUS, 12 bins)
XL_EDGES = np.array([0.20, 0.28, 0.34, 0.40, 0.46, 0.52, 0.58,
                      0.64, 0.70, 0.76, 0.82, 0.88, 1.00])
XL_CTRS  = 0.5 * (XL_EDGES[:-1] + XL_EDGES[1:])
XL_XERR  = 0.5 * (XL_EDGES[1:]  - XL_EDGES[:-1])
XL_BWS   = XL_EDGES[1:] - XL_EDGES[:-1]
N_XL     = len(XL_CTRS)        # 12

# ZEUS Q^2 bins (7 columns)
# The former 8th column, Q^2 in [640, 1280] (nominal 960, tabulated in
# zeus.dat under Q2=1000), has been dropped on request - the grid is now
# 7 x 7. Per-panel ("subplot") size is unchanged: the figure width scales
# with NQ_ZEUS, so each panel keeps the same dimensions as before.
Q2_BINS_ZEUS = [
    (7.0,    4.0,    10.0),
    (15.0,   10.0,   20.0),
    (30.0,   20.0,   40.0),
    (60.0,   40.0,   80.0),
    (120.0,  80.0,   160.0),
    (240.0,  160.0,  320.0),
    (480.0,  320.0,  640.0),
]

# No Q^2 column needs a non-nominal zeus.dat key any more (the [640,1280]
# column that was tabulated under Q2=1000 has been removed).
Q2_ZEUS_DAT_KEY = {}

# ZEUS x_BJ bins (7 rows)
X_BINS_ZEUS = [
    (1.1e-4, 8.0e-5, 1.5e-4),
    (2.1e-4, 1.5e-4, 3.0e-4),
    (4.2e-4, 3.0e-4, 6.0e-4),
    (8.5e-4, 6.0e-4, 1.2e-3),
    (17.e-4, 1.2e-3, 2.4e-3),
    (49.e-4, 2.4e-3, 1.0e-2),
    (32.e-3, 1.0e-2, 1.0e-1),
]

NQ_ZEUS = len(Q2_BINS_ZEUS)     # 7
NX_ZEUS = len(X_BINS_ZEUS)      # 7


# from zeus_three_pions.py: NLO F2^pi (quad), s-exp flux, pion grid


# JAM21-style pion-PDF NLO F2^pi + leading-neutron flux
# (works for any LHAPDF pion set; load_pion_hists_zeus() calls this for
# JAM21, xFitter and GRV)
LOCAL_LHAPDF_ROOT = "/home/aries/local"
LOCAL_LHAPDF_SITE = f"{LOCAL_LHAPDF_ROOT}/lib/python3.10/dist-packages"
LOCAL_LHAPDF_LIB  = f"{LOCAL_LHAPDF_ROOT}/lib/libLHAPDF.so"
LOCAL_LHAPDF_DATA = f"{LOCAL_LHAPDF_ROOT}/share/LHAPDF"

if os.path.isdir(LOCAL_LHAPDF_SITE) and LOCAL_LHAPDF_SITE not in sys.path:
    sys.path.insert(0, LOCAL_LHAPDF_SITE)
if os.path.exists(LOCAL_LHAPDF_LIB):
    ctypes.CDLL(LOCAL_LHAPDF_LIB, mode=ctypes.RTLD_GLOBAL)
os.environ.setdefault("LHAPDF_DATA_PATH", LOCAL_LHAPDF_DATA)
os.environ.setdefault("LHAPATH", LOCAL_LHAPDF_DATA)

CF, TR = 4.0 / 3.0, 0.5
QUAD_OPTS_Q = dict(limit=300, epsabs=1e-10, epsrel=1e-4)
QUAD_OPTS_G = dict(limit=400, epsabs=1e-10, epsrel=1e-4)

FLAVOR_INFO = [
    (2, -2, (2.0 / 3.0) ** 2),
    (1, -1, (1.0 / 3.0) ** 2),
    (3, -3, (1.0 / 3.0) ** 2),
    (4, -4, (2.0 / 3.0) ** 2),
    (5, -5, (1.0 / 3.0) ** 2),
]
CHARGE_SQ_SUM = sum(e2 for _, _, e2 in FLAVOR_INFO)

M_N, M_PI = 0.9382720813, 0.13957039
G_A, F_PI, LAMBDA = 1.267, 0.093, 1.31

X_FINE      = np.logspace(-6, np.log10(1.0 - 1e-8), 5000)
PIDS_NEEDED = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 21]
PDF_CACHE   = {}

F2P_TABLE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "F2p_CT18NLO_values.txt")


def _build_pdf_grid(pdf, q2val):
    key = (id(pdf), float(q2val))
    if key not in PDF_CACHE:
        qval = np.sqrt(q2val)
        PDF_CACHE[key] = {
            pid: np.array([pdf.xfxQ(pid, float(xx), qval) for xx in X_FINE])
            for pid in PIDS_NEEDED
        }
    return PDF_CACHE[key]


def xfq(pdf, pid, xval, q2val):
    if xval <= 0.0 or xval >= 1.0:
        return 0.0
    grids = _build_pdf_grid(pdf, q2val)
    return float(np.interp(xval, X_FINE, grids[pid], left=0.0, right=0.0))


def f2_lo(pdf, xval, q2val):
    if xval <= 0.0 or xval >= 1.0:
        return 0.0
    return sum(e2 * (xfq(pdf, q, xval, q2val) + xfq(pdf, qb, xval, q2val))
               for q, qb, e2 in FLAVOR_INFO)


def f2_nlo(pdf, xval, q2val):
    """NLO F2 (DIS scheme) for the pion, evaluated from any LHAPDF pion PDF."""
    if xval <= 0.0 or xval >= 0.999999:
        return 0.0

    pref = pdf.alphasQ(np.sqrt(q2val)) / (2.0 * np.pi)
    zmax = 1.0 - 1e-6

    f2_lo_val = 0.0
    c2q_total = 0.0
    for pid_q, pid_qb, e2 in FLAVOR_INFO:
        fx = xfq(pdf, pid_q, xval, q2val) + xfq(pdf, pid_qb, xval, q2val)
        f2_lo_val += e2 * fx

        def i1(z, _fx=fx, _pq=pid_q, _pqb=pid_qb):
            xo = xval / z
            if xo >= 1.0:
                return 0.0
            fz = xfq(pdf, _pq, xo, q2val) + xfq(pdf, _pqb, xo, q2val)
            return np.log(1.0 - z) / (1.0 - z) * (fz - _fx)

        def i2(z, _fx=fx, _pq=pid_q, _pqb=pid_qb):
            xo = xval / z
            if xo >= 1.0:
                return 0.0
            fz = xfq(pdf, _pq, xo, q2val) + xfq(pdf, _pqb, xo, q2val)
            return (fz - _fx) / (1.0 - z)

        def i3(z, _pq=pid_q, _pqb=pid_qb):
            xo = xval / z
            if xo >= 1.0:
                return 0.0
            fz = xfq(pdf, _pq, xo, q2val) + xfq(pdf, _pqb, xo, q2val)
            kernel = (3.0 + 2.0 * z - (1.0 + z) * np.log(1.0 - z)
                      - ((1.0 + z * z) / (1.0 - z)) * np.log(z))
            return kernel * fz

        i1_val, _ = quad(i1, xval, zmax, **QUAD_OPTS_Q)
        i2_val, _ = quad(i2, xval, zmax, **QUAD_OPTS_Q)
        i3_val, _ = quad(i3, xval, zmax, **QUAD_OPTS_Q)

        cq = CF * (
            2.0 * i1_val
            - (np.log(1.0 - xval) ** 2) * fx
            - 1.5 * i2_val
            - 1.5 * fx * np.log(1.0 - xval)
            + i3_val
            - (np.pi**2 / 3.0 + 9.0 / 2.0) * fx
        )
        c2q_total += e2 * pref * cq

    def ig(z):
        xo = xval / z
        if xo >= 1.0:
            return 0.0
        fg = xfq(pdf, 21, xo, q2val)
        kernel = (z**2 + (1.0 - z) ** 2) * np.log((1.0 - z) / z) + 8.0 * z * (1.0 - z) - 1.0
        return 2.0 * TR * kernel * fg

    ig_val, _ = quad(ig, xval, zmax, **QUAD_OPTS_G)
    c2g = pref * CHARGE_SQ_SUM * ig_val
    return f2_lo_val + c2q_total + c2g


@lru_cache(maxsize=None)
def flux(x_l_val):
    """pi+N leading-neutron flux f_{pi/N}(x_L) (JAM-style, non-standard kmax)."""
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


def compute_all_jam21_f2ln_zeus(pdf):
    """Full 7(x) x 8(Q2) rectangular grid. F_2LN(x_L) = 2*f_{pi/N}(x_L)*F_2^pi(x_pi,Q2),
    with x_pi = x_BJ / (1 - x_L). Pion-PDF-agnostic: called for each of
    JAM21 / xFitter / GRV in load_pion_hists_zeus().
    Returns dict (iq, ix) -> f2ln array[N_XL] (NaN where x_pi >= 1).
    """
    f2ln_jam = {}
    for iq, (q2_nom, q2lo, q2hi) in enumerate(Q2_BINS_ZEUS):
        for ix, (x_nom, xlo, xhi) in enumerate(X_BINS_ZEUS):
            curve = np.full(N_XL, np.nan)
            for i, xl_c in enumerate(XL_CTRS):
                xpi = x_nom / (1.0 - xl_c)
                if xpi >= 1.0:
                    continue
                f2pi = f2_nlo(pdf, float(xpi), float(q2_nom))
                curve[i] = 2.0 * flux(float(xl_c)) * f2pi
            f2ln_jam[(iq, ix)] = curve
        print(f"  F_2LN computed for Q2 column {iq+1}/{NQ_ZEUS} (Q2={q2_nom:g})")
    return f2ln_jam


# from zeus_three_pions.py: F2p table, helpers, F_2^LN conversion

def load_f2p_table(filename=F2P_TABLE_FILE):
    table = np.loadtxt(filename)
    return table[:, 0], table[:, 1], table[:, 2]   # x, Q2, F2p(CT18NLO)


def match_f2p(x_query, q2_query, x_table, q2_table, f2p_table):
    """Row-order fast path first, per-row (x,Q2) lookup fallback."""
    x_query, q2_query = np.asarray(x_query), np.asarray(q2_query)
    if (len(x_query) == len(x_table)
            and np.allclose(x_query, x_table, rtol=1e-6)
            and np.allclose(q2_query, q2_table, rtol=1e-6)):
        return np.asarray(f2p_table, dtype=float).copy()

    out = np.full(len(x_query), np.nan)
    for i, (xv, q2v) in enumerate(zip(x_query, q2_query)):
        mask = (np.isclose(x_table, xv, rtol=1e-4)
                & np.isclose(q2_table, q2v, rtol=1e-4))
        idx = np.where(mask)[0]
        if idx.size > 0:
            out[i] = f2p_table[idx[0]]
        else:
            print(f"  WARNING: no F2p_CT18NLO match for x={xv:.4e}, Q2={q2v:.4g}")
    return out


def pick(av, candidates):
    for c in candidates:
        if c in av:
            return c
    return None


def sci(v):
    m, e = f"{v:.2e}".split("e")
    return rf"{float(m):.2f}\!\times\!10^{{{int(e)}}}"


def q2_label(lo, hi):
    return rf"$Q^2\!\in\![{lo:.4g},\,{hi:.4g}]\ \mathrm{{GeV}}^2$"


def xb_label(lo, hi):
    return rf"$x_{{BJ}}\!\in\![{sci(lo)},\,{sci(hi)}]$"


def compute_y(Q2, x):
    return Q2 / (S * x)


def dis_prefactor(x, Q2):
    """P(x,Q^2) [nb/GeV^2] such that dsigma/dx_L = P x kin(y) x F_2LN x dx x dQ^2."""
    return (4.0 * np.pi * ALPHA**2 / (x * Q2**2)) * GEV2_TO_NB


def dsig_to_f2ln(dsig_per_xl, x, Q2, dx, dQ2):
    y   = compute_y(Q2, x)
    kin = 1.0 - y + 0.5 * y**2
    P   = dis_prefactor(x, Q2)
    denom = P * kin * dx * dQ2
    return np.asarray(dsig_per_xl, dtype=float) / denom


# from zeus_three_pions.py: Pythia F_2^LN, ZEUS data

def compute_all_pythia_f2ln_zeus(data):
    """For every cell of the FULL 7(x) x 8(Q2) grid, compute F_2LN from
    Pythia dsigma/dx_L. Returns dict (iq, ix) -> (f2ln array[N_XL], n_events).
    """
    sigma_dis = data["sigma_dis"]
    counts    = data["counts"]

    f2ln_py = {}
    nevts   = {}

    for iq, (q2_nom, q2lo, q2hi) in enumerate(Q2_BINS_ZEUS):
        dQ2 = q2hi - q2lo
        for ix, (x_nom, xlo, xhi) in enumerate(X_BINS_ZEUS):
            dx = xhi - xlo

            hist = counts[iq, ix]
            n = int(hist.sum())
            nevts[(iq, ix)] = n

            dsig = sigma_dis * hist / N_TOTAL / XL_BWS
            f2ln = dsig_to_f2ln(dsig, x_nom, q2_nom, dx, dQ2)
            f2ln_py[(iq, ix)] = f2ln

    return f2ln_py, nevts


# ZEUS DATA (zeus.dat)
def load_zeus_f2ln(path, f2p_table_file=F2P_TABLE_FILE):
    """zeus.dat columns:
    0: xL bin index   1: xL centre   2: Q2   3: x_BJ
    4: R_LN (%)       5: dR_LN (%)   6: F2p_ref (model A)  7: F2p_ref (model B)
    """
    raw = np.loadtxt(path)
    x_l  = raw[:, 1]
    q2   = raw[:, 2]
    xbj  = raw[:, 3]
    r    = raw[:, 4]
    dr   = raw[:, 5]

    x_table, q2_table, f2p_table = load_f2p_table(f2p_table_file)
    f2p = match_f2p(xbj, q2, x_table, q2_table, f2p_table)

    delta_xl = np.where(np.isclose(x_l, 0.24), 0.08,
                np.where(np.isclose(x_l, 0.94), 0.12, 0.06))

    f2ln = (r / (100.0 * delta_xl)) * f2p
    f2ln_err = (dr / (100.0 * delta_xl)) * f2p

    lut = {}
    for iq, (q2_nom, _, _) in enumerate(Q2_BINS_ZEUS):
        q2_key = Q2_ZEUS_DAT_KEY.get(iq, q2_nom)
        for ix, (x_nom, _, _) in enumerate(X_BINS_ZEUS):
            lut[(round(q2_key, 6), round(x_nom, 8))] = (iq, ix)

    # The Q^2 in [640, 1280] column (tabulated in zeus.dat under Q2=1000)
    # has been dropped from the grid - skip its rows quietly rather than
    # emitting an "unmatched point" warning for every one of them.
    _DROPPED_Q2_DAT_KEYS = {1000.0}

    data = {}
    for k in range(len(raw)):
        if any(abs(float(q2[k]) - dq) / dq < 1e-3 for dq in _DROPPED_Q2_DAT_KEYS):
            continue
        key = (round(float(q2[k]), 6), round(float(xbj[k]), 8))
        cell = lut.get(key)
        if cell is None:
            for (qq, xx), c in lut.items():
                if abs(qq - q2[k]) / qq < 1e-3 and abs(xx - xbj[k]) / xx < 1e-3:
                    cell = c
                    break
        if cell is None:
            print(f"  WARNING: unmatched ZEUS point Q2={q2[k]}, x={xbj[k]}")
            continue
        data.setdefault(cell, [[], [], []])
        data[cell][0].append(x_l[k])
        data[cell][1].append(f2ln[k])
        data[cell][2].append(f2ln_err[k])

    out = {}
    for cell, (xl_l, f2_l, e_l) in data.items():
        order = np.argsort(xl_l)
        out[cell] = (np.array(xl_l)[order], np.array(f2_l)[order], np.array(e_l)[order])
    return out


# from zeus_three_pions.py: chi^2 table, pion PDF sets, pion histograms

def save_chi2_table(results, out_path, title=""):
    """Write an overall chi^2 summary (one row per PDF-set curve) to a
    plain-text file.
    """
    with open(out_path, "w") as f:
        if title:
            f.write(f"{title}\n")
            f.write("=" * len(title) + "\n\n")
        header = f"{'PDF set':40s} {'chi2':>14s} {'N':>8s} {'chi2/N':>12s}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for label, r in results.items():
            f.write(f"{label:40s} {r['chi2']:14.4f} {r['ndof']:8d} "
                     f"{r['chi2_per_dof']:12.4f}\n")
    print(f"  -> {out_path}")


# -- proton PDF used here: CT18NLO + Remnant (grey dashed background) --
CT18RE_LABEL = "CT18NLO + Remnant"
C_CT18RE_BKG = "purple"


def _pythia_label(label):
    """'CT18NLO+Remnant' -> 'Pythia (CT18NLO+Remnant)', but leave names
    that already say Pythia alone -- identical convention to zeus_five_settings.py's
    _pythia_label(), so the two scripts' legends read the same way.
    """
    return label if label.strip().lower().startswith("pythia") else f"Pythia ({label})"

# -- the three pion PDFs (key, LHAPDF-set-name, display label, colour) --
PION_PDF_SETS = [
    ("jam21",   "JAM21PionPDFnlo", "JAM21",   "#0a7cff"),
    ("xfitter", "xFitterPI_NLO_VAR",  "xFitter", "#d62728"),  # red (no canonical analogue)
    ("grv",     "GRVPI1",             "GRV",     "#2ca02c"),  # green (no canonical analogue)
]


def _xl_tick_fmt(v, pos):
    """Bold x_L tick labels: 0.50, 0.75, 1 -- whole numbers lose the
    trailing '.00' (H1/ZEUS style). Matches zeus_five_settings.py's tick pattern.
    """
    if abs(v - round(v)) < 1e-9:
        return rf"$\mathbf{{{int(round(v))}}}$"
    return rf"$\mathbf{{{v:.2f}}}$"


# Load the three pion PDFs and compute each F_2^{LN} on the ZEUS grid
def load_pion_hists_zeus(pion_pdf_sets):
    import lhapdf
    hists = {}
    for key, pdfname, label, _ in pion_pdf_sets:
        print(f"\nLoading pion PDF '{label}' ({pdfname}) ...")
        try:
            pdf_obj = lhapdf.mkPDF(pdfname, 0)
        except Exception as e:
            print(f"  WARNING: could not load {pdfname}: {e}  — skipped")
            continue
        # compute_all_jam21_f2ln_zeus is pion-PDF-agnostic: it takes any pion
        # PDF object and returns the (iq, ix) -> F_2^{LN} array dict.
        hists[key] = compute_all_jam21_f2ln_zeus(pdf_obj)
        print(f"  → F_2LN computed on the ZEUS grid for '{label}'")
    return hists


# from zeus_three_pions.py: cell averages, chi^2, publication overview grid

# ONE colour scheme, reused identically in every panel (deliberately NOT
# coloured per Pythia set -- the columns are distinguished by their
# headings, not by hue). Line styles differ too, so the figure survives
# greyscale printing.
C_DIS   = "#1f77b4"   # DIS fragmentation  (Pythia proton PDF)
C_PDFAV = "#2ca02c"   # PDF average        (pion PDF)
C_TOTAL = "#d62728"   # Total = DIS + PDF

LS_DIS, LS_PDFAV, LS_TOTAL = "--", "-.", "-"

# The 7x7 grids can afford FS_TITLE=28, but five long column headings
# ("CT18NLO + Remnant" ...) collide at that size, so this grid uses its own.
FS_TITLE_OV = 22

# The chi^2/N box is an annotation, not a heading -- at FS_LEGEND (24) it
# dominates the panel and crowds the curves, so it gets its own size.
FS_CHI2_OV = 15

OVERVIEW_YLIM   = (0.0, 0.25)
OVERVIEW_XTICKS = [0.4, 0.6, 0.8]        # interior ticks only: with
OVERVIEW_YTICKS = [0.05, 0.10, 0.15, 0.20]  # wspace=hspace=0 the panels
                                            # touch, so edge ticks would
                                            # collide with the neighbour.


def _ov_tick_fmt(v, pos):
    """Bold tick labels; whole numbers drop their trailing decimals."""
    if abs(v - round(v)) < 1e-9:
        return rf"$\mathbf{{{int(round(v))}}}$"
    return rf"$\mathbf{{{v:.2f}}}$"


def _average_pair_over_cells(py_dict, pion_dict, zeus_data):
    """Cell-average the Pythia (DIS fragmentation) and pion (PDF average)
    F_2^LN histograms over every (Q^2, x_BJ) cell that carries a ZEUS
    measurement, counting -- in each x_L bin independently -- only the
    cells where BOTH curves are finite.
    """
    s_dis = np.zeros(N_XL)
    s_pdf = np.zeros(N_XL)
    n_cnt = np.zeros(N_XL)

    for cell in zeus_data:
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
        dis_avg = np.where(n_cnt > 0, s_dis / np.where(n_cnt > 0, n_cnt, 1.0), np.nan)
        pdf_avg = np.where(n_cnt > 0, s_pdf / np.where(n_cnt > 0, n_cnt, 1.0), np.nan)
    return dis_avg, pdf_avg, dis_avg + pdf_avg


def _average_zeus_overview(zeus_data):
    """Inverse-variance-weighted average of every ZEUS point, rebinned
    onto XL_EDGES, collapsing the whole (Q^2, x_BJ) grid onto a single
    representative x_L curve.
    """
    sum_wf = np.zeros(N_XL)
    sum_w  = np.zeros(N_XL)
    for xl_d, f2_d, e_d in zeus_data.values():
        e_d = np.asarray(e_d, dtype=float)
        f2_d = np.asarray(f2_d, dtype=float)
        idx = np.clip(np.searchsorted(XL_EDGES[1:], xl_d, side="left"), 0, N_XL - 1)
        w = np.where(e_d > 0, 1.0 / np.where(e_d > 0, e_d, 1.0) ** 2, 0.0)
        for j, i in enumerate(idx):
            sum_wf[i] += w[j] * f2_d[j]
            sum_w[i]  += w[j]

    avg = np.divide(sum_wf, sum_w, out=np.full(N_XL, np.nan), where=sum_w > 0)
    err = np.divide(1.0, np.sqrt(np.where(sum_w > 0, sum_w, 1.0)),
                    out=np.full(N_XL, np.nan), where=sum_w > 0)
    return avg, err


def _chi2_total_over_cells(py_dict, pion_dict, zeus_data):
    """Overall chi^2 and N of the ZEUS points against the per-cell Total
    (Pythia + pion), accumulated over the whole grid.
    """
    chi2 = 0.0
    ndof = 0
    for cell, (xl_d, f2_d, e_d) in zeus_data.items():
        py = py_dict.get(cell) if py_dict else None
        h  = pion_dict.get(cell) if pion_dict else None
        if py is None or h is None:
            continue
        f2_d = np.asarray(f2_d, dtype=float)
        e_d  = np.asarray(e_d,  dtype=float)
        combined = np.asarray(py, dtype=float) + np.where(np.isfinite(h), h, 0.0)

        bin_idx = np.clip(np.digitize(xl_d, XL_EDGES) - 1, 0, N_XL - 1)
        model = combined[bin_idx]

        valid = np.isfinite(model) & np.isfinite(f2_d) & (e_d > 0)
        if not valid.any():
            continue
        chi2 += float(np.sum(((f2_d[valid] - model[valid]) / e_d[valid]) ** 2))
        ndof += int(valid.sum())
    return chi2, ndof


def make_publication_overview_grid(pythia_dict, pion_hists, zeus_data, out,
                                   pion_pdf_sets=PION_PDF_SETS):
    """Publication grid: pion PDFs (rows) x Pythia proton PDFs (columns)."""
    # Enforce H1-matching column order
    desired_order = [
        "Pythia default",
        "CT18NLO",
        "CT18NLO + Remnant",
        "CT18NLO + Tuning",
        "CT18NLO + Gluon"
    ]
    pythia_labels = [label for label in desired_order if label in pythia_dict]
    n_py   = len(pythia_labels)
    n_pion = len(pion_pdf_sets)
    if n_py == 0 or n_pion == 0:
        print("  nothing to draw -- overview grid skipped")
        return

    zeus_avg, zeus_err = _average_zeus_overview(zeus_data)
    zvalid = np.isfinite(zeus_avg) & np.isfinite(zeus_err)

    # Panels touch edge-to-edge (H1/ZEUS staircase convention). Panel size
    # matches the 7x7 grids, so the shared FS_* sizes stay consistent.
    fig, axes = plt.subplots(
        n_pion, n_py,
        figsize=(n_py * 4.8, n_pion * 4.0),
        gridspec_kw=dict(wspace=0.0, hspace=0.0),
        sharex=True, sharey=True, squeeze=False)
    fig.patch.set_facecolor("white")
    # Default subplot margins leave ~12% blank above the titles, which
    # would open a dead gap under the shared legend.
    fig.subplots_adjust(left=0.085, right=0.985, top=0.935, bottom=0.075,
                        wspace=0.0, hspace=0.0)

    legend_src = None
    row_label_artists = []

    for ip, (pion_key, _pdfname, pion_label, _pion_color) in enumerate(pion_pdf_sets):
        pion_dict = pion_hists.get(pion_key)

        for jp, py_label in enumerate(pythia_labels):
            ax = axes[ip, jp]
            ax.set_facecolor("white")
            py_dict = pythia_dict[py_label][0]

            chi2, ndof = 0.0, 0

            if pion_dict is not None:
                dis_avg, pdf_avg, tot_avg = _average_pair_over_cells(
                    py_dict, pion_dict, zeus_data)

                ax.stairs(dis_avg, XL_EDGES, color=C_DIS, lw=LW_BKG,
                          ls=LS_DIS, label="DIS fragmentation")
                ax.stairs(pdf_avg, XL_EDGES, color=C_PDFAV, lw=LW_PION,
                          ls=LS_PDFAV, label="Sullivan Process")
                ax.stairs(tot_avg, XL_EDGES, color=C_TOTAL, lw=LW_SUM,
                          ls=LS_TOTAL, label="Total")

                chi2, ndof = _chi2_total_over_cells(py_dict, pion_dict, zeus_data)

            if zvalid.any():
                ax.errorbar(XL_CTRS[zvalid], zeus_avg[zvalid],
                            xerr=XL_XERR[zvalid], yerr=zeus_err[zvalid],
                            fmt="o", ms=DATA_MS, mfc="black", mec="black",
                            mew=DATA_MEW, ecolor="black", elinewidth=DATA_ELW,
                            capsize=DATA_CAPS, capthick=DATA_ELW,
                            color="black", zorder=10, label="ZEUS (avg)")

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
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l)
                h2.append(h)
                l2.append(l)
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
    # Same font size, stroked so the glyphs render heavier (mathtext has
    # no weight above bold) -- identical trick to the 7x7 grids.
    _lbl_stroke = [patheffects.withStroke(linewidth=AXIS_LABEL_STROKE,
                                          foreground="black")]

    t_x = fig.text(cx, uy0 - pad, r"$x_L$", ha="center", va="top",
                   fontsize=FS_AXLABEL, fontweight="bold")
    t_x.set_path_effects(_lbl_stroke)

    # The row headings (JAM21 / xFitter / GRV) live in the left margin, so
    # anchor the universal y label outside THEM, not merely outside the axes.
    y_anchor = ux0
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
        print(f"  -> {fname}")
    plt.close(fig)


# Revised analysis: Pythia from the count histograms, new pion inputs
import csv as _csv

HERE_ = os.path.dirname(os.path.abspath(__file__))
import lambda_fit as _ZLF                      # noqa: E402

HIST_DIR = os.path.join(HERE_, "..", "inputs", "pythia_histograms")
ZEUS_DAT_PATH = os.path.join(HERE_, "..", "data", "zeus.dat")
F2P_PATH = os.path.join(HERE_, "..", "data", "F2p_CT18NLO_values.txt")
FIT_CSV = os.path.join(HERE_, "..", "lambda_fit", "lambda_fit_results.csv")
ZEUS_HISTS = [("default", "Pythia default"), ("ct18", "CT18NLO"), ("ct18re", "CT18NLO + Remnant"),
              ("ct18tune", "CT18NLO + Tuning"), ("ct18gluon", "CT18NLO + Gluon")]
JAM21_TABLE1_LAMBDA = 1.31
# chi^2/N printed in the original figure ZEUS/compare_f2ln_zeus_ctre_three_all_combinations_overview.png
ORIGINAL_CHI2N = {"jam21": [13.01, 15.73, 8.57, 31.78, 16.40],
                  "xfitter": [15.44, 17.18, 17.01, 18.86, 17.50],
                  "grv": [16.64, 19.75, 10.46, 35.89, 20.50]}


def pythia_data_from_hist(npz_path):
    """dict(sigma_dis, counts) exactly as zeus_three_pions.py's load_pythia() returns it."""
    z = np.load(npz_path)
    assert np.allclose(z["xl_edges"], XL_EDGES), "x_L binning differs from zeus_three_pions.py"
    print(f"  {os.path.basename(npz_path)}: sigma = {float(z['sigma_dis']):.4f} nb, "
          f"binned events = {int(z['counts'].sum()):,}")
    return dict(sigma_dis=float(z["sigma_dis"]), counts=z["counts"])


def make_flux(lam):
    """zeus_three_pions.py's flux() with the cutoff Lambda as a parameter (identical integrand, kmax = 0.65 x_L)."""
    @lru_cache(maxsize=None)
    def f(x_l_val):
        def integrand(k2):
            xb = 1.0 - x_l_val
            if xb <= 1e-10:
                return 0.0
            d_val = -(k2 + xb * xb * M_N * M_N + x_l_val * M_PI * M_PI) / x_l_val
            s_val = (k2 + M_N * M_N) / x_l_val + (k2 + M_PI * M_PI) / xb
            ff = np.exp((M_N * M_N - s_val) / (lam**2))
            pref = (G_A * G_A * M_N * M_N) / ((4.0 * np.pi * F_PI) ** 2)
            return pref * xb * (k2 + xb * xb * M_N * M_N) / (x_l_val * x_l_val * d_val * d_val) * (ff * ff)

        kmax = 0.65 * x_l_val
        val, _ = quad(integrand, 0.0, kmax * kmax, limit=200)
        return val
    return f


class MeanPDF:
    """Stand-in pion PDF for zeus_three_pions.py's f2_nlo: alpha_s from member 0 (identical in every
    JAM21 replica); x f(x) = replica MEAN, put straight into the PDF_CACHE of zeus_three_pions.py on its X_FINE.
    """
    def __init__(self, set_name):
        self.set_name = set_name
        self._m0 = lhapdf.mkPDF(set_name, 0)
        for q2_nom, _lo, _hi in Q2_BINS_ZEUS:
            PDF_CACHE[(id(self), float(q2_nom))] = _ZLF._mean_arrays(set_name, q2_nom, X_FINE, False)

    def alphasQ(self, q):
        return self._m0.alphasQ(q)

    def xfxQ(self, *a):
        raise RuntimeError("MeanPDF: every x f(x) must come from PDF_CACHE")


def compute_f2ln_zeus_flux(pdf, flux_fn):
    """zeus_three_pions.py's compute_all_jam21_f2ln_zeus, with the flux passed in."""
    f2ln_jam = {}
    for iq, (q2_nom, q2lo, q2hi) in enumerate(Q2_BINS_ZEUS):
        for ix, (x_nom, xlo, xhi) in enumerate(X_BINS_ZEUS):
            curve = np.full(N_XL, np.nan)
            for i, xl_c in enumerate(XL_CTRS):
                xpi = x_nom / (1.0 - xl_c)
                if xpi >= 1.0:
                    continue
                f2pi = f2_nlo(pdf, float(xpi), float(q2_nom))
                curve[i] = 2.0 * flux_fn(float(xl_c)) * f2pi
            f2ln_jam[(iq, ix)] = curve
        print(f"  F_2LN computed for Q2 column {iq+1}/{NQ_ZEUS} (Q2={q2_nom:g})", flush=True)
    return f2ln_jam


def best_lambdas():
    """New s-exp Lambda of xFitter and GRV: common H1+ZEUS best fit."""
    out = {}
    with open(FIT_CSV) as fh:
        for r in _csv.DictReader(fh):
            if r["fit"] == "H1+ZEUS_common_Lambda" and r["flux"] == "s_exp":
                out[r["pion_pdf"]] = float(r["Lambda_GeV"])
    return out


# zeus_three_pions.py keys PDF_CACHE by id(pdf). Once a PDF object is freed, Python may give the
# next PDF the same id and the cached grid of the OLD PDF would be reused. So the
# cache is emptied before every pion PDF and every PDF object is kept alive.
_ALIVE = []


def pion_hists_previous():
    """zeus_three_pions.py's load_pion_hists_zeus, one pion PDF at a time on an empty PDF_CACHE."""
    hists = {}
    for s in PION_PDF_SETS:
        PDF_CACHE.clear()
        hists.update(load_pion_hists_zeus([s]))
    return hists


def pion_hists_new():
    lam = best_lambdas()
    hists = {}
    print(f"\nJAM21 = mean of the JAM21PionPDFnlo replicas, Lambda = {JAM21_TABLE1_LAMBDA}")
    PDF_CACHE.clear()
    _ALIVE.append(MeanPDF("JAM21PionPDFnlo"))
    hists["jam21"] = compute_f2ln_zeus_flux(_ALIVE[-1], make_flux(JAM21_TABLE1_LAMBDA))
    for key, setn, lab in (("xfitter", "xFitterPI_NLO_VAR", "xFitter"), ("grv", "GRVPI1", "GRV")):
        print(f"\n{lab} = member 0 of {setn}, Lambda = {lam[lab]:.4f}")
        PDF_CACHE.clear()
        _ALIVE.append(lhapdf.mkPDF(setn, 0))
        hists[key] = compute_f2ln_zeus_flux(_ALIVE[-1], make_flux(lam[lab]))
    used = {"JAM21": JAM21_TABLE1_LAMBDA, "xFitter": lam["xFitter"], "GRV": lam["GRV"]}
    return hists, used


def write_chi2_table(pythia_dict, pion_hists, zeus_data, path, title, compare):
    py_labels = [lab for _k, lab in ZEUS_HISTS]
    L = [title, "=" * len(title), "",
         f"{'pion PDF':10s} " + " ".join(f"{p:>20s}" for p in py_labels)]
    for key, _pdfname, label, _c in PION_PDF_SETS:
        cells = []
        for j, pl in enumerate(py_labels):
            chi2, n = _chi2_total_over_cells(pythia_dict[pl][0], pion_hists[key], zeus_data)
            txt = f"{chi2 / n:.2f}"
            if compare:
                txt += f" (fig {ORIGINAL_CHI2N[key][j]:.2f})"
            cells.append(txt)
        L.append(f"{label:10s} " + " ".join(f"{c:>20s}" for c in cells))
    L.append("")
    L.append("chi^2/N of the ZEUS points vs Total = Pythia + Sullivan, N = "
             f"{_chi2_total_over_cells(pythia_dict[py_labels[0]][0], pion_hists['jam21'], zeus_data)[1]}")
    txt = "\n".join(L)
    open(path, "w").write(txt + "\n")
    print("\n" + txt)


def run(mode):
    import lhapdf as _lh
    globals()["lhapdf"] = _lh
    print(f"Loading ZEUS F_2LN data from {ZEUS_DAT_PATH} ...")
    zeus_data = load_zeus_f2ln(ZEUS_DAT_PATH, f2p_table_file=F2P_PATH)
    colors = {s["label"]: s["color"] for s in PYTHIA_FILES}
    pythia_dict = {}
    for key, label in ZEUS_HISTS:
        f2ln, _ = compute_all_pythia_f2ln_zeus(pythia_data_from_hist(os.path.join(HIST_DIR, f"ZEUS_{key}.npz")))
        pythia_dict[label] = (f2ln, label, colors[label])
    if mode == "previous":
        print("\nPion PDFs exactly as in zeus_three_pions.py: member 0, Lambda = 1.31 for all three")
        pion_hists = pion_hists_previous()
        title = "ZEUS overview, PREVIOUS settings (member 0, Lambda = 1.31): chi^2/N   [original figure value]"
    else:
        pion_hists, used = pion_hists_new()
        title = ("ZEUS overview, NEW settings: JAM21 replica mean (Lambda = 1.31), "
                 f"xFitter Lambda = {used['xFitter']:.3f}, GRV Lambda = {used['GRV']:.3f}: chi^2/N")
    out = os.path.join(HERE_, "..", "cache", f"ZEUS_{mode}")
    make_publication_overview_grid(pythia_dict, pion_hists, zeus_data, out)
    write_chi2_table(pythia_dict, pion_hists, zeus_data, f"{out}_chi2.txt", title, compare=(mode == "previous"))


if __name__ == "__main__":
    import argparse as _ap
    _p = _ap.ArgumentParser()
    _p.add_argument("--mode", choices=["previous", "new"], required=True)
    run(_p.parse_args().mode)

