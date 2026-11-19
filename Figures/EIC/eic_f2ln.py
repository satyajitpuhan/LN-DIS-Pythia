#!/usr/bin/env python3
"""EIC F2^LN grids for 5x41, 10x100 and 18x275 GeV: Pythia (CT18NLO) plus the Sullivan term
with JAM21, xFitter and GRV (Figs. 14-16). Each beam setting has its own (x_B, Q^2)
binning, see CFG_BINNING.
"""

import argparse, os, sys, time, warnings
PYTHIA_OUT = os.environ.get("PYTHIA_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Pythia", "output"))
from functools import lru_cache

import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib.ticker import (AutoMinorLocator, FuncFormatter, MaxNLocator,
                               LogLocator, LogFormatterMathtext)
from matplotlib.transforms import Bbox
import lhapdf
from scipy.integrate import quad

warnings.filterwarnings("ignore")

FS_AXLABEL = 42
FS_TITLE   = 28  # Q^2 column titles   (bin range only)
FS_ROWLAB  = FS_TITLE  # x_B row labels
FS_TICK    = 24  # x and y tick numbers
FS_LEGEND  = 24  # the one shared corner legend

SPINE_LW   = 3.2  # panel box
LW_BKG     = 2.6  # Pythia DIS-background histogram
LW_SIG     = 3.2  # pion-PDF (JAM21) histogram
LW_TOTAL   = 3.6

AXIS_LABEL_STROKE = 1.1
TICK_LABEL_STROKE = 0.6

matplotlib.rcParams.update({
    "font.family"         : "serif",
    "font.serif"          : ["Times New Roman", "DejaVu Serif"],
    "font.weight"         : "bold",
    "mathtext.fontset"    : "dejavuserif",
    "mathtext.default"    : "bfit",  # bold *italic* math (H1 style)
    "axes.labelsize"      : FS_ROWLAB,
    "axes.labelweight"    : "bold",
    "axes.titlesize"      : FS_TITLE,
    "axes.titleweight"    : "bold",
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
    "axes.linewidth"      : SPINE_LW,  # bold panel boundaries
})

FS_ROWLBL = FS_ROWLAB
FS_AXIS   = FS_AXLABEL

HIDE_EMPTY_PANELS = True

REF_HLINES   = []
REF_HL_COLOR = "0.35"
REF_HL_LW    = 2.0
REF_HL_LS    = (0, (7, 5))

PAIR_BOTTOM = {}

LEGEND_MARGIN_X = 0.010
LEGEND_MARGIN_Y = 0.020

PANEL_SIZE = 4.0

YLOG = True

NQ = 6
NX = 6

# x_L binning
XL_LO, XL_HI = 0.30, 1.00

XL_WIDTHS = [0.02]

XL_DW_BASE    = 0.005
XL_EDGES_BASE = np.round(np.arange(XL_LO, XL_HI + 0.5 * XL_DW_BASE,
                                   XL_DW_BASE), 6)
N_XL_BASE     = len(XL_EDGES_BASE) - 1


def set_xl_binning(dw):
    """Switch the module-level x_L binning to uniform width `dw`."""
    global XL_EDGES, XL_CTRS, XL_BWS, N_XL, XL_DW
    n = int(round((XL_HI - XL_LO) / dw))
    if abs(n * dw - (XL_HI - XL_LO)) > 1e-9:
        raise ValueError(f"x_L width {dw} does not divide "
                         f"[{XL_LO}, {XL_HI}] a whole number of times")
    k = dw / XL_DW_BASE
    if abs(k - round(k)) > 1e-9:
        raise ValueError(f"x_L width {dw} is not a multiple of the base "
                         f"width {XL_DW_BASE}")
    XL_DW    = float(dw)
    XL_EDGES = np.round(np.linspace(XL_LO, XL_HI, n + 1), 6)
    XL_CTRS  = 0.5 * (XL_EDGES[:-1] + XL_EDGES[1:])
    XL_BWS   = XL_EDGES[1:] - XL_EDGES[:-1]
    N_XL     = n
    return XL_EDGES


def merge_base_counts(counts_base):
    k = int(round(XL_DW / XL_DW_BASE))
    return np.asarray(counts_base, dtype=np.int64).reshape(-1, k).sum(axis=1)


def xl_tag(dw):
    """Filename fragment for a bin width: 0.025 -> 'dxL0p025'."""
    return "dxL" + f"{dw:.3f}".replace(".", "p")


XL_EDGES = XL_CTRS = XL_BWS = None
N_XL = 0
XL_DW = XL_WIDTHS[0]
set_xl_binning(XL_WIDTHS[0])

ALPHA      = 1.0 / 137.036
GEV2_TO_NB = 0.3893793656e6

THETA_E_MIN_DEG  = 135.0  # deg
THETA_E_MAX_DEG  = 178.0  # deg
EEPRIME_MIN      = 1.0  # GeV
THETA_N_MAX_MRAD = 5.0  # mrad

CFG_THETA_N_MRAD = {
    "5x41":   5.0,
    "10x100": 2.5,
    "18x275": 1.0,
}

TAN_THETA_N_MAX = float(np.tan(THETA_N_MAX_MRAD * 1e-3))

# Optional hard ceiling on the flux k_perp^2 integration
K2_CAP = None

DROP_PARTIAL_BINS = True
PARTIAL_MIN_FRAC  = 0.999

DROP_RADIATIVE = True
Q2_GEN_MIN  = 1.0
XB_MAX_PHYS = 1.0

N_GEN_FIXED = 5.0e8

def cut_text():
    return (r"$135^\circ<\theta_{e'}<178^\circ$, $E_{e'}>1$ GeV, "
            rf"$\theta_n<{THETA_N_MAX_MRAD:g}$ mrad")


CUT_TEXT = cut_text()

CHUNK_SIZE = 2_000_000

# output location
OUT_DIR  = "plots_cfgbins"
NAME_TAG = "cfgbins"

# legend labels
LBL_BKG   = "Pythia (CT18 NLO)"
LBL_TOTAL = "Total"

C_BKG   = "black"
C_JAM   = "navy"  # JAM21
C_TOTAL = "#d62728"

PION_PDF_SETS = [
    ("jam21",   "JAM21PionPDFnlo",   "JAM21",   "#0a7cff"),  # bright blue
    ("xfitter", "xFitterPI_NLO_VAR", "xFitter", "#d62728"),  # red
    ("grv",     "GRVPI1",            "GRV",     "#2ca02c"),  # green
]
MAIN_PION_KEY = "jam21"

# Configs: (name, root_path, E_e, E_p)
CONFIGS = [
    ("5x41",   os.path.join(PYTHIA_OUT, "eic_5x41.root"),     5.0,  41.0),
    ("10x100", os.path.join(PYTHIA_OUT, "eic_10x100.root"), 10.0, 100.0),
    ("18x275", os.path.join(PYTHIA_OUT, "eic_18x275.root"), 18.0, 275.0),
]
CFG_COLORS = {"5x41": "#1f77b4", "10x100": "#2ca02c", "18x275": "#d62728"}

CFG_RANGES = {
    # Q2_lo  Q2_hi    xB_lo    xB_hi
    "5x41":   (   1.0,   20.0,  1.0e-3,   0.5),
    "10x100": (   1.0,   80.0,  2.0e-4,   0.5),
    "18x275": (   1.0,  250.0,  5.0e-5,   0.5),
}

CFG_Q2_EDGES = {
    "5x41":   [1, 1.5, 2.5, 4,  7,  12,  20],
    "10x100": [1, 2,   4,   9,  18, 40,  80],
    "18x275": [1, 2.5, 6,   15, 40, 100, 250],
}


def _round_sig(v, n=2):
    """Round to n significant figures (1.42e-2 -> 1.4e-2)."""
    if v == 0.0:
        return 0.0
    from math import floor, log10
    return round(v, -int(floor(log10(abs(v)))) + (n - 1))


def _nice_x_edges(x_lo, x_hi, nx, n_sig=2):
    e = np.geomspace(x_lo, x_hi, nx + 1)
    r = [float(x_lo)] + [_round_sig(v, n_sig) for v in e[1:-1]] + [float(x_hi)]
    for i in range(1, len(r)):
        if r[i] <= r[i - 1]:
            r[i] = r[i - 1] * 1.001
    return r


CFG_X_EDGES = {name: _nice_x_edges(CFG_RANGES[name][2],
                                   CFG_RANGES[name][3], NX)
               for name in CFG_RANGES}


class Binning:

    def __init__(self, name, q2_lo, q2_hi, x_lo, x_hi, nq=NQ, nx=NX,
                 q2_edges=None, x_edges=None):
        self.name = name
        if q2_edges is not None:
            # rounded bin edges
            self.q2_edges = np.asarray(q2_edges, dtype=float)
            nq = len(self.q2_edges) - 1
        else:
            self.q2_edges = np.geomspace(q2_lo, q2_hi, nq + 1)
        if x_edges is not None:
            self.x_edges = np.asarray(x_edges, dtype=float)
            nx = len(self.x_edges) - 1
        else:
            self.x_edges = np.geomspace(x_lo, x_hi, nx + 1)
        self.nq, self.nx = nq, nx
        self.q2_bins  = [(self.q2_edges[i], self.q2_edges[i+1]) for i in range(nq)]
        self.x_bins   = [(self.x_edges[i],  self.x_edges[i+1])  for i in range(nx)]
        # geometric-mean fallbacks, per cell
        gq = np.sqrt(self.q2_edges[:-1] * self.q2_edges[1:])
        gx = np.sqrt(self.x_edges[:-1]  * self.x_edges[1:])
        self.q2_rep = {(iq, ix): float(gq[iq]) for iq in range(nq) for ix in range(nx)}
        self.x_rep  = {(iq, ix): float(gx[ix]) for iq in range(nq) for ix in range(nx)}
        self.measured = {(iq, ix): False for iq in range(nq) for ix in range(nx)}
        self.dropped  = set()

    def set_measured_means(self, q2_sum, x_sum, n_cell):
        n_meas = 0
        for cell, n in n_cell.items():
            if n > 0:
                self.q2_rep[cell] = q2_sum[cell] / n
                self.x_rep[cell]  = x_sum[cell]  / n
                self.measured[cell] = True
                n_meas += 1
        return n_meas

    def describe(self):
        q = ", ".join(f"{e:.3g}" for e in self.q2_edges)
        x = ", ".join(f"{e:.3g}" for e in self.x_edges)
        return (f"  Q2 edges : {q}\n"
                f"  x_B edges: {x}")


CFG_BINNING = {name: Binning(name, *CFG_RANGES[name],
                             q2_edges=CFG_Q2_EDGES.get(name),
                             x_edges=CFG_X_EDGES.get(name))
               for name, _, _, _ in CONFIGS}


def sci(v):
    m, e = f"{v:.2e}".split("e")
    return rf"{float(m):.2f}\!\times\!10^{{{int(e)}}}"

def q2_label(lo, hi):
    return rf"$[{lo:g},\,{hi:g}]$"

def _mant(v):
    return f"{v:.1f}" if v < 10.0 else f"{v:.0f}"


def xb_label(lo, hi):
    e = int(np.floor(np.log10(lo)))
    return (rf"$[{_mant(lo / 10.0**e)},\,{_mant(hi / 10.0**e)}]$" + "\n"
            + rf"$\times 10^{{{e}}}$")

def beam_tex(name):
    return name.replace("x", r"\times")

def beam_legend(name):
    return rf"${beam_tex(name)}$"

def compute_y(Q2, x, S):
    return Q2 / (S * x)

def dis_prefactor(x, Q2):
    return (4.0 * np.pi * ALPHA**2 / (x * Q2**2)) * GEV2_TO_NB

def cell_phase_space_area(x_lo, x_hi, q_lo, q_hi, S):
    """Area of the part of the (x_B, Q^2) cell that satisfies y <= 1."""
    xa = q_lo / S  # S x = q_lo
    xb = q_hi / S  # S x = q_hi
    area = 0.0
    lo = max(x_lo, xa); hi = min(x_hi, xb)
    if hi > lo:
        area += 0.5 * S * (hi * hi - lo * lo) - q_lo * (hi - lo)
    lo = max(x_lo, xb); hi = x_hi
    if hi > lo:
        area += (q_hi - q_lo) * (hi - lo)
    return max(area, 0.0)


def effective_q2_hi(x_hi, q_lo, q_hi, S):
    """Largest Q^2 actually reachable in this cell: min(q_hi, S x_hi)."""
    return min(q_hi, S * x_hi)


def dsig_to_f2ln(dsig_per_xl, x, Q2, dx, dQ2, S, area=None):
    y   = compute_y(Q2, x, S)
    kin = 1.0 - y + 0.5 * y**2
    P   = dis_prefactor(x, Q2)
    A   = (dx * dQ2) if area is None else area
    return np.asarray(dsig_per_xl, dtype=float) / (P * kin * A)


def _stroke(*artists, width=TICK_LABEL_STROKE):
    if width <= 0:
        return
    fx = [patheffects.withStroke(linewidth=width, foreground="black")]
    for a in artists:
        if a is not None:
            a.set_path_effects(fx)


def _bold_ticks(ax, size=FS_TICK):
    for lbl in list(ax.get_xticklabels(which="both")) + \
               list(ax.get_yticklabels(which="both")):
        lbl.set_fontweight("bold")
        lbl.set_fontsize(size)
        _stroke(lbl)
    for off in (ax.xaxis.get_offset_text(), ax.yaxis.get_offset_text()):
        off.set_fontweight("bold")
        off.set_fontsize(size)

def _bold_legend(leg):
    if leg is None:
        return leg
    for txt in leg.get_texts():
        txt.set_fontweight("bold")
    return leg

def _legend(ax, ncol=1, fontsize=FS_LEGEND):
    leg = ax.legend(loc="upper right", fontsize=fontsize, ncol=ncol,
                    handlelength=1.4, handletextpad=0.4, borderpad=0.35,
                    framealpha=0.85, labelspacing=0.25, columnspacing=0.8)
    return _bold_legend(leg)


# NLO F2 of the pi-, vectorised: on the GPU with cupy if it is installed, else numpy
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import f2pi_nlo as F2V  # noqa: E402

# JAM21 is the mean of its replicas, as everywhere else in the paper (member 0 of
# JAM21PionPDFnlo is not the replica mean). --jam21-member0 gives member 0 (arXiv v1).
JAM21_REPLICA_MEAN = True
MEAN_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache",
                          "jam21_replica_mean_eic.npz")
REPLICA_CHUNK = 50  # replicas held in memory at a time


def _accumulate(pdfs, q2_list, acc):
    """acc[iq, ip, :] += x f(x, Q^2) on F2V.X_FINE for every PDF in pdfs.

    As in the scalar code of arXiv v1, x f = 0 outside the x range of the set; Q^2 is
    already clamped to the Q^2 range of the set by the caller.
    """
    m0 = pdfs[0]
    inside = (F2V.X_FINE >= m0.xMin) & (F2V.X_FINE <= m0.xMax)
    xs = F2V.X_FINE[inside].tolist()
    for m in pdfs:
        for iq, q2 in enumerate(q2_list):
            qs = [q2] * len(xs)
            for ip, pid in enumerate(F2V.PIDS_NEEDED):
                acc[iq, ip, inside] += m.xfxQ2(pid, xs, qs)


def replica_mean_tables(setname, q2_list):
    """x f(x, Q^2) averaged over replicas 1..N at every Q^2 in q2_list, on F2V.X_FINE.

    The replicas are loaded REPLICA_CHUNK at a time, so only a few are in memory at once.
    The result is cached on disk per Q^2.
    """
    cache = {}
    if os.path.exists(MEAN_CACHE):
        with np.load(MEAN_CACHE) as z:
            cache = {k: z[k] for k in z.files}
    key = lambda q2: f"{setname}|{q2:.9g}|{len(F2V.X_FINE)}"
    todo = [q2 for q2 in q2_list if key(q2) not in cache]
    if todo:
        n = lhapdf.getPDFSet(setname).size
        acc = np.zeros((len(todo), len(F2V.PIDS_NEEDED), len(F2V.X_FINE)))
        t0 = time.time()
        verb = lhapdf.verbosity()
        lhapdf.setVerbosity(0)
        try:
            for start in range(1, n, REPLICA_CHUNK):
                stop = min(start + REPLICA_CHUNK, n)
                chunk = [lhapdf.mkPDF(setname, i) for i in range(start, stop)]
                _accumulate(chunk, todo, acc)
                del chunk
                print(f"    {setname}: replicas {start}-{stop - 1} of {n - 1} done "
                      f"({len(todo)} Q2 values, {time.time() - t0:5.1f} s)", flush=True)
        finally:
            lhapdf.setVerbosity(verb)
        acc /= (n - 1)
        for q2, a in zip(todo, acc):
            cache[key(q2)] = a
        os.makedirs(os.path.dirname(MEAN_CACHE), exist_ok=True)
        np.savez(MEAN_CACHE, **cache)
    return [cache[key(q2)] for q2 in q2_list]


class _Table(F2V.PionPdfGrid):
    """x f(x, Q^2) at one Q^2 on F2V.X_FINE, from an array (n_pids, n_x)."""

    def __init__(self, arr, q2):
        self.q2 = q2
        self.log_x = F2V.to_gpu(F2V._LOG_X_FINE)
        self.grids = {pid: F2V.to_gpu(arr[i]) for i, pid in enumerate(F2V.PIDS_NEEDED)}


class PionPDF:
    """A pion PDF set: member 0, or the mean of the replicas (JAM21).

    x f(x, Q^2) is tabulated at every Q^2 that is needed and F2 is computed from the
    tables with f2pi_nlo.f2pi_minus_nlo_vec. Q^2 is clamped to the Q^2 range of the set,
    as in the scalar code of arXiv v1.
    """

    def __init__(self, key, setname, label, color, replica_mean=False):
        self.key, self.setname = key, setname
        self.label, self.color = label, color
        self.replica_mean = replica_mean
        self.pdf = lhapdf.mkPDF(setname, 0)  # alpha_s (same for all replicas), ranges
        self._tabs = {}

    def __repr__(self):
        return f"<PionPDF {self.key}: {self.setname}>"

    def clamp_q2(self, q2):
        return min(max(float(q2), self.pdf.q2Min), self.pdf.q2Max)

    def prepare(self, q2_list):
        todo = sorted({self.clamp_q2(q) for q in q2_list} - set(self._tabs))
        if not todo:
            return
        if self.replica_mean:
            arrs = replica_mean_tables(self.setname, todo)
        else:
            acc = np.zeros((len(todo), len(F2V.PIDS_NEEDED), len(F2V.X_FINE)))
            _accumulate([self.pdf], todo, acc)
            arrs = list(acc)
        for q2, a in zip(todo, arrs):
            self._tabs[q2] = _Table(a, q2)

    def release(self):
        self._tabs.clear()  # free the GPU tables

    def f2pi(self, x, q2):
        """NLO F2 of the pi- at all x (1-D numpy) and one Q^2; 0 for x outside (0, 1)."""
        q2c = self.clamp_q2(q2)
        return F2V.f2pi_minus_nlo_vec(self.pdf, self._tabs[q2c], x, q2c)


def load_pion_pdfs(sets=PION_PDF_SETS):
    loaded = []
    for key, setname, label, color in sets:
        mean = key == "jam21" and JAM21_REPLICA_MEAN
        print(f"Loading pion PDF '{label}'  ({setname}, "
              f"{'mean of the replicas' if mean else 'member 0'}) ...")
        try:
            pp = PionPDF(key, setname, label, color, replica_mean=mean)
        except Exception as e:
            print(f"  !!! could not load {setname!r}: {e}")
            continue
        pdf = pp.pdf
        print(f"  x=[{pdf.xMin:.2e}, {pdf.xMax:.2e}]  "
              f"Q2=[{pdf.q2Min:.2e}, {pdf.q2Max:.2e}]")
        loaded.append(pp)
    if not loaded:
        raise RuntimeError("No pion PDF set could be loaded -- nothing to plot.")
    print(f"  F2 of the pion computed on: {F2V.backend_name()}")
    return loaded


# pion flux (Model I)
_M_N=0.9382720813; _M_PI=0.13957039; _G_A=1.267; _F_PI=0.093
_LAMBDA_CUT=1.31
_K2_MAX=0.04  # GeV^2

def _d_piN(k2, xL):
    xbar = 1.0 - xL
    return -(k2 + xbar*xbar*_M_N*_M_N + xL*_M_PI*_M_PI) / xL

@lru_cache(maxsize=65536)
def f_piN_model_i(xL, k2max=_K2_MAX):
    """Pion flux integrated over k_perp^2 from 0 to k2max."""
    if xL <= 0.0 or xL >= 1.0: return 0.0
    if k2max <= 0.0: return 0.0
    xbar = 1.0 - xL
    pref = (_G_A*_G_A*_M_N*_M_N) / ((4.0*np.pi*_F_PI)**2)
    def integrand(k2):
        dval  = _d_piN(k2, xL)
        sval  = (k2+_M_N*_M_N)/xL + (k2+_M_PI*_M_PI)/xbar
        ff    = np.exp((_M_N*_M_N - sval)/(_LAMBDA_CUT*_LAMBDA_CUT))
        numer = xbar*(k2+xbar*xbar*_M_N*_M_N)
        denom = xL*xL*dval*dval
        return (numer/denom)*(ff*ff)
    val, _ = quad(integrand, 0.0, k2max, limit=220)
    return pref * val


def _set_theta_n_max(mrad):
    global THETA_N_MAX_MRAD, TAN_THETA_N_MAX
    THETA_N_MAX_MRAD = float(mrad)
    TAN_THETA_N_MAX  = float(np.tan(THETA_N_MAX_MRAD * 1e-3))
    f_piN_model_i.cache_clear()


def k2max_for_config(E_p):
    def f(xL):
        pt = TAN_THETA_N_MAX * float(xL) * E_p
        k2 = pt * pt
        if K2_CAP is not None:
            k2 = min(k2, float(K2_CAP))
        return k2
    return f


def rebin_pdf_to_xl_edges(ppdf, k2max_fn, binning, n_sub=12, verbose=True,
                          tag=""):
    """Sullivan F2LN per (Q2, x_B) cell, averaged over n_sub points in each x_L bin."""
    cells = [(iq, ix) for ix in range(binning.nx) for iq in range(binning.nq)]
    ppdf.prepare([binning.q2_rep[c] for c in cells])
    sub = np.array([XL_EDGES[i] + (np.arange(n_sub) + 0.5) * (XL_EDGES[i+1] - XL_EDGES[i]) / n_sub
                    for i in range(N_XL)])  # (N_XL, n_sub)
    flux = np.array([[2.0 * f_piN_model_i(float(xL), float(k2max_fn(float(xL))))
                      for xL in row] for row in sub])
    hist = {}
    for cell in cells:
        xbj, q2 = binning.x_rep[cell], binning.q2_rep[cell]
        x_pi = xbj / (1.0 - sub)
        f2 = ppdf.f2pi(x_pi.ravel(), q2).reshape(sub.shape)  # 0 for x_pi >= 1
        hist[cell] = (flux * f2).mean(axis=1)
    ppdf.release()
    if verbose:
        print(f"  [{ppdf.label}{tag}] done: {len(cells)} (Q2, xB) cells x {N_XL} "
              f"x_L bins")
    return hist


def _resolve_paths(path):
    import glob as _glob
    if os.path.isdir(path):
        files = sorted(_glob.glob(os.path.join(path, "*.root")))
    elif any(ch in path for ch in "*?[]"):
        files = sorted(_glob.glob(path))
    else:
        files = [path]
    if not files:
        raise FileNotFoundError(f"No ROOT file(s) found for: {path!r}")
    missing = [p for p in files if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"ROOT file(s) not found: {missing}")
    return files


def process_root(path, E_e, E_p, label, binning, treename="tree",
                 chunk_size=CHUNK_SIZE):
    S = 4.0 * E_e * E_p
    NQc, NXc = binning.nq, binning.nx

    files = _resolve_paths(path)
    if len(files) > 1:
        print(f"[{label}] {len(files)} input files matched:")
        for p in files:
            print(f"    {p}")

    def meta(f, key):
        try:
            return f[key].member("fVal")
        except Exception:
            return None

    sigma_dis = None
    sigma_all = []
    n_tot = 0
    trees = []
    open_files = []
    for p in files:
        try:
            f = uproot.open(p, object_cache=None, array_cache=None)
        except Exception as e:
            raise RuntimeError(f"[{label}] could not open {p!r}: {e}") from e
        open_files.append(f)
        s = meta(f, "sigmaGen_nb")
        if s is not None:
            sigma_all.append((os.path.basename(p), float(s)))
            sigma_dis = float(s)  # final file wins
        if treename not in f:
            available = [k.split(";")[0] for k in f.keys()]
            raise RuntimeError(
                f"[{label}] tree {treename!r} not found in {p!r}. "
                f"Objects available: {available}")
        t = f[treename]
        n_tot += t.num_entries
        trees.append((p, t))

    if sigma_dis is None:
        raise RuntimeError(f"[{label}] sigmaGen_nb not found in any of {files}.")

    if len(sigma_all) > 1:
        print(f"[{label}] sigmaGen_nb per file: "
              + ", ".join(f"{b}={v:.6g}" for b, v in sigma_all))
        print(f"[{label}] using FINAL sigma_gen = {sigma_dis:.6g} nb")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        N_gen = N_GEN_FIXED
        print(f"[{label}] using fixed N_total = {N_gen:.3g} "
              f"(tree entries = {n_tot:,} across {len(files)} file(s))")

        want = ["Q2", "xB", "xL", "EePrime", "theta_e", "theta_n"]

        cells  = [(iq, ix) for iq in range(NQc) for ix in range(NXc)]
        counts = {c: np.zeros(N_XL_BASE, dtype=np.int64) for c in cells}
        # accumulators for the event-weighted mean kinematics per cell
        q2_sum = {c: 0.0 for c in cells}
        x_sum  = {c: 0.0 for c in cells}
        n_cell = {c: 0   for c in cells}

        n_seen = 0
        n_pass = 0
        cf = dict(theta_e=0, eeprime=0, theta_n=0, rad=0, nm1=0)
        xl_sum = 0.0
        n_xl   = 0
        pt_sum = 0.0
        pt_max = 0.0

        print(f"[{label}] streaming {len(files)} file(s), {n_tot:,} entries ...")
        for fpath, tree in trees:
            missing_branches = [w for w in want if w not in tree.keys()]
            if missing_branches:
                raise RuntimeError(
                    f"[{label}] branch(es) {missing_branches} not in tree "
                    f"{treename!r} of {fpath!r}. Available: {tree.keys()}")
            for chunk in tree.iterate(want, step_size=chunk_size, library="np"):
                n_chunk = len(chunk["Q2"])
                n_seen += n_chunk

                theta_e_deg  = np.degrees(chunk["theta_e"])
                theta_n_mrad = 1000.0 * chunk["theta_n"]
                pt_n = (chunk["xL"] * E_p) * np.tan(np.abs(chunk["theta_n"]))

                m_theta_e = ((theta_e_deg > THETA_E_MIN_DEG)
                             & (theta_e_deg < THETA_E_MAX_DEG))
                m_eeprime = chunk["EePrime"] > EEPRIME_MIN
                m_theta_n = np.abs(theta_n_mrad) < THETA_N_MAX_MRAD
                rad_ok = (((chunk["Q2"] >= Q2_GEN_MIN) & (chunk["xB"] <= XB_MAX_PHYS))
                          if DROP_RADIATIVE else np.ones(n_chunk, dtype=bool))

                cf["theta_e"] += int(m_theta_e.sum())
                cf["eeprime"] += int(m_eeprime.sum())
                cf["theta_n"] += int(m_theta_n.sum())
                cf["rad"]     += int(rad_ok.sum())

                nm1 = m_theta_e & m_eeprime & rad_ok
                cf["nm1"] += int(nm1.sum())
                if nm1.any():
                    xl_sum += float(chunk["xL"][nm1].sum())
                    n_xl   += int(nm1.sum())

                sel = nm1 & m_theta_n
                n_pass += int(sel.sum())
                if not sel.any():
                    print(f"    ... {n_seen:,}/{n_tot:,} entries "
                          f"({100*n_seen/n_tot:5.1f} %), passing: {n_pass:,}",
                          end="\r")
                    continue

                _pt = pt_n[sel]
                pt_sum += float(_pt.sum())
                pt_max  = max(pt_max, float(_pt.max()))

                q2_sel = chunk["Q2"][sel]
                xb_sel = chunk["xB"][sel]
                xl_sel = chunk["xL"][sel]

                for iq in range(NQc):
                    q2lo, q2hi = binning.q2_bins[iq]
                    q2_mask = (q2_sel >= q2lo) & (q2_sel < q2hi)
                    if not q2_mask.any():
                        continue
                    for ix in range(NXc):
                        xlo, xhi = binning.x_bins[ix]
                        m = q2_mask & (xb_sel >= xlo) & (xb_sel < xhi)
                        if not m.any():
                            continue
                        cell = (iq, ix)
                        # event-weighted mean kinematics for this cell
                        q2_sum[cell] += float(q2_sel[m].sum())
                        x_sum[cell]  += float(xb_sel[m].sum())
                        n_cell[cell] += int(m.sum())
                        hist, _ = np.histogram(xl_sel[m], bins=XL_EDGES_BASE)
                        counts[cell] += hist

                print(f"    ... {n_seen:,}/{n_tot:,} entries "
                      f"({100*n_seen/n_tot:5.1f} %), passing: {n_pass:,}",
                      end="\r")
        print()

    for f in open_files:
        try:
            f.close()
        except Exception:
            pass

    print(f"[{label}] sigma_gen = {sigma_dis:.4f} nb | N_total = {N_gen:.0f} | "
          f"E_e={E_e:g} x E_p={E_p:g} GeV (sqrt(s) = {np.sqrt(S):.2f} GeV) | "
          f"pass: {n_pass:,}/{n_tot:,}")

    # representative kinematics
    n_meas = binning.set_measured_means(q2_sum, x_sum, n_cell)
    print(f"[{label}] event-weighted <Q2>,<x_B> measured in {n_meas}/"
          f"{NQc*NXc} cells (the rest are empty -> geometric-mean fallback)")

    # cut-flow / acceptance diagnostics
    def _pct(a, b):
        return f"{100.0 * a / b:6.2f} %" if b else "   n/a"
    xl_bar    = (xl_sum / n_xl) if n_xl else 0.7
    pt_of_ang = TAN_THETA_N_MAX * xl_bar * E_p
    print(f"[{label}] cut-flow (marginal, of {n_tot:,} stored entries):")
    print(f"           theta_e in [{THETA_E_MIN_DEG:g},{THETA_E_MAX_DEG:g}] deg : "
          f"{cf['theta_e']:>12,}  {_pct(cf['theta_e'], n_tot)}")
    print(f"           E_e' > {EEPRIME_MIN:g} GeV                    : "
          f"{cf['eeprime']:>12,}  {_pct(cf['eeprime'], n_tot)}")
    print(f"           theta_n < {THETA_N_MAX_MRAD:g} mrad   [APPLIED] : "
          f"{cf['theta_n']:>12,}  {_pct(cf['theta_n'], n_tot)}")
    print(f"           non-radiative                    : "
          f"{cf['rad']:>12,}  {_pct(cf['rad'], n_tot)}")
    print(f"[{label}] neutron-cut efficiency after all other cuts (N-1): "
          f"{n_pass:,}/{cf['nm1']:,} = {_pct(n_pass, cf['nm1'])}")

    pt_lo = TAN_THETA_N_MAX * XL_EDGES[0]  * E_p
    pt_hi = TAN_THETA_N_MAX * XL_EDGES[-1] * E_p
    pt_bar = (pt_sum / n_pass) if n_pass else 0.0
    print(f"[{label}] p_T = x_L * E_p * tan(theta_n) is DERIVED, not cut on.")
    print(f"[{label}]   implied window p_T^max = {pt_lo:.3f} -> {pt_hi:.3f} GeV "
          f"over x_L = {XL_EDGES[0]:.2f} .. {XL_EDGES[-1]:.2f}")
    print(f"[{label}]   selected neutrons: <p_T> = {pt_bar:.3f} GeV, "
          f"max p_T = {pt_max:.3f} GeV   (<x_L> = {xl_bar:.3f})")
    print(f"[{label}]   JAM21 flux integrated over the same window: "
          f"k_perp^2 max = {pt_lo**2:.4f} -> {pt_hi**2:.4f} GeV^2"
          + (f", capped at {K2_CAP:g}" if K2_CAP is not None else " (uncapped)"))

    partial = []
    for iq in range(NQc):
        q2lo, q2hi = binning.q2_bins[iq]
        dQ2 = q2hi - q2lo
        for ix in range(NXc):
            xlo, xhi = binning.x_bins[ix]
            dx = xhi - xlo
            cell = (iq, ix)
            c = counts[cell]

            area  = cell_phase_space_area(xlo, xhi, q2lo, q2hi, S)
            afull = dx * dQ2
            frac  = area / afull if afull > 0 else 0.0

            if DROP_PARTIAL_BINS and frac < PARTIAL_MIN_FRAC:
                if c.sum() > 0:
                    partial.append((iq, ix, frac,
                                    effective_q2_hi(xhi, q2lo, q2hi, S), q2hi))
                binning.dropped.add(cell)
                counts[cell][:] = 0
                continue

            if c.sum() > 0 and frac < 0.999:
                partial.append((iq, ix, frac,
                                effective_q2_hi(xhi, q2lo, q2hi, S), q2hi))

    n_forbidden = sum(1 for iq in range(NQc) for ix in range(NXc)
                      if cell_phase_space_area(*binning.x_bins[ix],
                                               *binning.q2_bins[iq], S) <= 0.0)
    if DROP_PARTIAL_BINS:
        print(f"[{label}] dropped {len(binning.dropped)} cell(s) that are not "
              f"fully inside y <= 1 (of which {n_forbidden} are entirely "
              f"forbidden); only bins covering their full quoted "
              f"(x_B, Q^2) range are plotted.")
    if partial:
        print(f"[{label}] {len(partial)} of the dropped cell(s) had events:")
        for iq, ix, frac, q_eff, q_nom in sorted(partial, key=lambda t: t[2]):
            print(f"           cell (iq={iq}, ix={ix}): only "
                  f"{100*frac:5.1f} % accessible, effective Q2 upper edge "
                  f"{q_eff:.3g} vs nominal {q_nom:.3g} GeV^2")

    return counts, sigma_dis, N_gen


def f2ln_bkg_for_binning(counts_base, binning, S, sigma_dis, N_gen):
    f2ln_bkg = {}
    for iq in range(binning.nq):
        q2lo, q2hi = binning.q2_bins[iq]
        dQ2 = q2hi - q2lo
        for ix in range(binning.nx):
            xlo, xhi = binning.x_bins[ix]
            dx = xhi - xlo
            cell = (iq, ix)
            if cell in binning.dropped:
                f2ln_bkg[cell] = np.zeros(N_XL)
                continue
            c = merge_base_counts(counts_base[cell])
            area = cell_phase_space_area(xlo, xhi, q2lo, q2hi, S)
            if area <= 0.0:
                area = dx * dQ2  # forbidden -> empty anyway
            dsig = sigma_dis * c.astype(float) / N_gen / XL_BWS
            f2ln_bkg[cell] = dsig_to_f2ln(dsig, binning.x_rep[cell],
                                          binning.q2_rep[cell], dx, dQ2, S,
                                          area=area)
    return f2ln_bkg


XL_TICKS = [0.50, 0.75, 1.00]


def _xl_tick_fmt(v, pos):
    """Bold x_L tick labels: 0.40, 0.60, ..."""
    if abs(v - round(v)) < 1e-9:
        return rf"$\mathbf{{{int(round(v))}}}$"
    return rf"$\mathbf{{{v:.2f}}}$"


def _setup_axes(ax):
    ax.set_xlim(XL_EDGES[0] - 0.01, XL_EDGES[-1] + 0.01)
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_linewidth(SPINE_LW)
    ax.set_xticks(XL_TICKS)
    ax.xaxis.set_major_formatter(FuncFormatter(_xl_tick_fmt))
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True,
                   labelsize=FS_TICK)
    ax.tick_params(which="major", length=10, width=2.0, pad=6)
    ax.tick_params(which="minor", length=5, width=1.4)
    ax.minorticks_on()
    # horizontal reference lines, under the data
    for yv in REF_HLINES:
        ax.axhline(yv, color=REF_HL_COLOR, lw=REF_HL_LW, ls=REF_HL_LS,
                   zorder=0.5)


# per-cell curve recipes
def _curves_bkg(f2ln_bkg, cell):
    """Background-only panel: [] when the cell is empty (panel gets hidden)."""
    bkg = f2ln_bkg.get(cell)
    if bkg is None or not np.any(bkg != 0):
        return []
    return [(bkg, C_BKG, LW_BKG, ':', LBL_BKG)]


def _curves_combo(f2ln_bkg, sig_hist, ppdf, cell):
    """Background + pion PDF + Total."""
    bkg = f2ln_bkg.get(cell, np.zeros(N_XL))
    sig = sig_hist.get(cell, np.zeros(N_XL))
    if not np.any(bkg != 0):
        return []
    return [(bkg + sig, C_TOTAL,     LW_TOTAL, '-',  LBL_TOTAL),
            (bkg,       C_BKG,       LW_BKG,   ':',  LBL_BKG),
            (sig,       ppdf.color,  LW_SIG,   '--', ppdf.label)]


def _curves_totals(f2ln_bkg, sig_hists, ppdfs, cell):
    """One Total curve per pion PDF set: Total_i = Pythia background + pion_i."""
    bkg = f2ln_bkg.get(cell, np.zeros(N_XL))
    if not np.any(bkg != 0):
        return []
    out = []
    for pp in ppdfs:
        sig = sig_hists.get(pp.key, {}).get(cell)
        if sig is None:
            continue
        out.append((bkg + sig, pp.color, LW_TOTAL, '-',
                    f"{pp.label} + {LBL_BKG}"))
    out.append((bkg, C_BKG, LW_BKG, ':', LBL_BKG))
    return out


def _apply_common_ylim(axes, arrays, log=None):
    log = YLOG if log is None else log
    vals = np.concatenate([np.asarray(a, float).ravel() for a in arrays]) \
           if arrays else np.array([0.0])
    if REF_HLINES:
        vals = np.concatenate([vals, np.asarray(REF_HLINES, float)])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return
    vmax = float(vals.max())
    if vmax <= 0.0:
        return
    if log:
        pos = vals[vals > 0.0]
        vmin = float(pos.min()) if pos.size else vmax * 1e-6
        vmin = max(vmin, vmax * 1e-6)
        lim = (vmin / 1.5, vmax * 1.5)
    else:
        lim = (0.0, vmax * 1.08)
    for ax in np.ravel(axes):
        if log:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
            ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
        else:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
            ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_ylim(*lim)


# empty-corner geometry (legend + F_2^LN label)
def _largest_empty_block_at(cell_has_data, nrow, ncol, corner="top-left"):
    def _all_empty(cells):
        return all(not cell_has_data.get(c, False) for c in cells)

    best = None  # (area, block)
    if corner == "top-left":
        for r1 in range(nrow):
            for q1 in range(ncol):
                block = [(r, q) for r in range(0, r1 + 1) for q in range(0, q1 + 1)]
                if block and _all_empty(block):
                    if best is None or len(block) > best[0]:
                        best = (len(block), block)
    else:  # bottom-right
        for r0 in range(nrow):
            for q0 in range(ncol):
                block = [(r, q) for r in range(r0, nrow) for q in range(q0, ncol)]
                if block and _all_empty(block):
                    if best is None or len(block) > best[0]:
                        best = (len(block), block)
    return best[1] if best is not None else None


def _largest_corner_empty_block(cell_has_data, nrow, ncol):
    a = _largest_empty_block_at(cell_has_data, nrow, ncol, "bottom-right")
    b = _largest_empty_block_at(cell_has_data, nrow, ncol, "top-left")
    cands = [c for c in (a, b) if c]
    if not cands:
        return None
    return max(cands, key=len)


def _block_centre(axes, block):
    positions = [axes[r][q].get_position() for r, q in block]
    x0 = min(p.x0 for p in positions); x1 = max(p.x1 for p in positions)
    y0 = min(p.y0 for p in positions); y1 = max(p.y1 for p in positions)
    return 0.5 * (x0 + x1), 0.5 * (y0 + y1)


def _place_legend_in_empty_space(fig, axes, cell_has_data, source_ax, nrow, ncol,
                                 extra_label=None, fontsize=None, title=None):
    handles, labels = source_ax.get_legend_handles_labels()
    if not handles:
        return False

    def _rank(lbl):
        if lbl == LBL_BKG:   return 0
        if lbl == LBL_TOTAL: return 2
        return 1
    pairs = sorted(zip(handles, labels), key=lambda hl: _rank(hl[1]))
    handles = [h for h, _ in pairs]; labels = [l for _, l in pairs]

    title_kw = {}
    if title:
        title_kw = dict(title=title,
                        title_fontproperties={"weight": "bold",
                                              "size": fontsize or FS_LEGEND})

    block = _largest_corner_empty_block(cell_has_data, nrow, ncol)
    if not block:
        vis = [a for a in np.ravel(axes) if a.get_visible()]
        pos = [a.get_position() for a in vis]
        gx1 = max(p.x1 for p in pos)
        try:
            fig.canvas.draw()
            rend  = fig.canvas.get_renderer()
            inv   = fig.transFigure.inverted()
            union = Bbox.union([a.get_tightbbox(rend) for a in vis])
            _, uy1 = inv.transform((union.x1, union.y1))
        except Exception:
            uy1 = max(p.y1 for p in pos) + 0.02
        leg = fig.legend(handles, labels, loc="lower right",
                         bbox_to_anchor=(gx1, uy1 + 0.055),
                         ncol=len(handles), handlelength=2.0,
                         handletextpad=0.7, borderpad=0.8, columnspacing=1.6,
                         framealpha=1.0, edgecolor="0.35",
                         prop={"weight": "bold", "size": FS_LEGEND},
                         **title_kw)
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_linewidth(1.6)
        leg.set_zorder(1000)
        _bold_legend(leg)
        return False

    positions = [axes[r][q].get_position() for r, q in block]
    bx0 = min(p.x0 for p in positions); bx1 = max(p.x1 for p in positions)
    by0 = min(p.y0 for p in positions); by1 = max(p.y1 for p in positions)
    cx, cy = 0.5 * (bx0 + bx1), 0.5 * (by0 + by1)
    avail_w = 0.92 * (bx1 - bx0)
    avail_h = 0.92 * (by1 - by0)
    max_w = avail_w
    max_h = avail_h if not extra_label else 0.55 * avail_h

    def _draw(fs, ycen=None):
        leg = fig.legend(handles, labels, loc="center",
                         bbox_to_anchor=(cx, cy if ycen is None else ycen),
                         ncol=1, handlelength=2.0, handletextpad=0.7,
                         borderpad=0.9, labelspacing=0.55, framealpha=1.0,
                         edgecolor="0.35",
                         prop={"weight": "bold", "size": fs},
                         **({} if not title else
                            dict(title=title,
                                 title_fontproperties={"weight": "bold",
                                                       "size": fs})))
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_alpha(1.0)
        leg.get_frame().set_linewidth(1.6)
        leg.set_zorder(1000)
        _bold_legend(leg)
        try:
            fig.canvas.draw()
            bb = leg.get_window_extent().transformed(fig.transFigure.inverted())
        except Exception:
            bb = None
        return leg, bb

    fs = float(fontsize if fontsize else FS_LEGEND)
    leg, bb = _draw(fs)
    for _ in range(12):
        if bb is None or (bb.width <= max_w and bb.height <= max_h) or fs <= 8.0:
            break
        fs *= max(0.55, min(0.92, min(max_w / bb.width, max_h / bb.height)))
        leg.remove(); leg, bb = _draw(fs)

    if not extra_label:
        return False
    if bb is None:
        try:
            fig.canvas.draw()
            bb = leg.get_window_extent().transformed(fig.transFigure.inverted())
        except Exception:
            return False

    gap   = 0.010
    box_w = 0.98 * (bx1 - bx0)
    fs2   = float(FS_AXLABEL)
    t2, bb2 = None, None
    for _ in range(14):
        if t2 is not None:
            t2.remove()
        t2 = fig.text(cx, cy, extra_label, ha="center", va="center",
                      fontsize=fs2, fontweight="bold", zorder=1000,
                      bbox=dict(boxstyle="square,pad=0.55", facecolor="white",
                                edgecolor="0.35", linewidth=1.6))
        try:
            fig.canvas.draw()
            bb2 = t2.get_window_extent(fig.canvas.get_renderer()) \
                    .transformed(fig.transFigure.inverted())
        except Exception:
            break
        if bb2.width <= box_w or fs2 <= fs:
            break
        fs2 = max(fs, fs2 * max(0.7, min(0.94, box_w / max(bb2.width, 1e-9))))

    if bb2 is None:
        return True

    def _redraw_pair(f1, f2):
        nonlocal leg, bb, t2, bb2
        leg.remove(); leg, bb = _draw(f1)
        t2.remove()
        t2 = fig.text(cx, cy, extra_label, ha="center", va="center",
                      fontsize=f2, fontweight="bold", zorder=1000,
                      bbox=dict(boxstyle="square,pad=0.55", facecolor="white",
                                edgecolor="0.35", linewidth=1.6))
        try:
            fig.canvas.draw()
            bb2 = t2.get_window_extent(fig.canvas.get_renderer()) \
                    .transformed(fig.transFigure.inverted())
        except Exception:
            pass
        return bb, bb2

    for _ in range(14):
        if bb is None or bb2 is None:
            break
        h_pair = bb.height + gap + bb2.height
        w_pair = max(bb.width, bb2.width)
        if (h_pair <= avail_h and w_pair <= avail_w) or min(fs, fs2) <= 8.0:
            break
        k = max(0.6, min(0.94, min(avail_h / h_pair, avail_w / w_pair)))
        fs, fs2 = fs * k, fs2 * k
        bb, bb2 = _redraw_pair(fs, fs2)

    if bb is None or bb2 is None:
        return True

    h_pair = bb.height + gap + bb2.height
    w_pair = max(bb.width, bb2.width)
    cx_p = bx1 - 0.5 * w_pair - LEGEND_MARGIN_X
    cy_p = by1 - 0.5 * h_pair - LEGEND_MARGIN_Y
    # never push it out of the block on the other side
    if bx1 - bx0 >= w_pair:
        cx_p = max(cx_p, bx0 + 0.5 * w_pair)
    else:
        cx_p = cx  # block narrower than the pair
    if by1 - by0 >= h_pair:
        cy_p = max(cy_p, by0 + 0.5 * h_pair)
    else:
        cy_p = cy

    y_leg = cy_p + 0.5 * h_pair - 0.5 * bb.height
    y_txt = cy_p - 0.5 * h_pair + 0.5 * bb2.height
    leg.remove()
    leg = fig.legend(handles, labels, loc="center",
                     bbox_to_anchor=(cx_p, y_leg),
                     ncol=1, handlelength=2.0, handletextpad=0.7,
                     borderpad=0.9, labelspacing=0.55, framealpha=1.0,
                     edgecolor="0.35",
                     prop={"weight": "bold", "size": fs},
                     **({} if not title else
                        dict(title=title,
                             title_fontproperties={"weight": "bold",
                                                   "size": fs})))
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(1.0)
    leg.get_frame().set_linewidth(1.6)
    leg.set_zorder(1000)
    _bold_legend(leg)
    t2.set_position((cx_p, y_txt))
    PAIR_BOTTOM["y"] = y_txt - 0.5 * bb2.height
    PAIR_BOTTOM["x0"] = cx_p - 0.5 * w_pair
    PAIR_BOTTOM["x1"] = cx_p + 0.5 * w_pair
    return True


def _place_f2_label_top_left(fig, axes, cell_has_data, nrow, ncol, text,
                             fontsize, stroke=None, dx=0.0,
                             row_label_texts=None):
    block = _largest_empty_block_at(cell_has_data, nrow, ncol, "top-left")
    if not block:
        return False

    rows = sorted({r for r, _ in block})
    cols = sorted({q for _, q in block})
    positions = [axes[r][q].get_position() for r, q in block]
    x0 = min(p.x0 for p in positions); x1 = max(p.x1 for p in positions)
    y0 = min(p.y0 for p in positions); y1 = max(p.y1 for p in positions)
    try:
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        inv  = fig.transFigure.inverted()
        for r in rows:
            t = (row_label_texts or {}).get(r)
            if t is None:
                continue
            bb = t.get_window_extent(rend).transformed(inv)
            x1 = min(x1, bb.x0 - 0.004)
    except Exception:
        if len(cols) > 1:
            x1 = min(axes[r][cols[-1]].get_position().x0 for r in rows)
    if x1 <= x0:
        return False

    cx, cy = 0.5 * (x0 + x1) + dx, 0.5 * (y0 + y1)
    max_w, max_h = 0.94 * (x1 - x0), 0.90 * (y1 - y0)

    fs = float(fontsize)
    t = None
    for _ in range(10):
        if t is not None:
            t.remove()
        t = fig.text(cx, cy, text, ha="center", va="center",
                     fontsize=fs, fontweight="bold", linespacing=1.25)
        if stroke:
            t.set_path_effects(stroke)
        try:
            fig.canvas.draw()
            bb = t.get_window_extent(fig.canvas.get_renderer()) \
                  .transformed(fig.transFigure.inverted())
        except Exception:
            break
        if (bb.width <= max_w and bb.height <= max_h) or fs <= 18.0:
            break
        fs *= max(0.6, min(0.92, min(max_w / bb.width, max_h / bb.height)))
    return True


# the grid


def _clear_row_labels(fig, axes, row_label_texts, pad=0.06):
    """Push each x_B row label left until it clears that panel's y tick numbers."""
    try:
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
    except Exception:
        return
    for row, txt in row_label_texts.items():
        ax = txt.axes
        x_left = 0.0
        for tl in ax.get_yticklabels():
            if not tl.get_visible() or not tl.get_text():
                continue
            bb = tl.get_window_extent(rend).transformed(ax.transAxes.inverted())
            x_left = min(x_left, bb.x0)  # leftmost edge of the tick numbers
        txt.set_position((x_left - pad, 0.5))


def _fit_column_titles(fig, axes, row_label_texts=None,
                       fontsize=FS_TITLE, frac=0.86):
    titled = [ax for ax in np.ravel(axes)
              if ax.get_visible() and ax.get_title()]
    if not titled:
        return float(fontsize)
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    except Exception:
        return float(fontsize)
    scale = 1.0
    for ax in titled:
        bb = ax.title.get_window_extent(renderer)
        panel_w = ax.get_window_extent(renderer).width
        if bb.width > 0:
            scale = min(scale, frac * panel_w / bb.width)
    size = max(10.0, fontsize * scale) if scale < 1.0 else float(fontsize)
    for ax in titled:
        ax.title.set_fontsize(size)
    for t in (row_label_texts or {}).values():
        t.set_fontsize(size)
    return size


def _place_direction_arrows(fig, axes, vis, fontsize=FS_AXLABEL):
    """The two axis labels of the grid."""
    pos = [ax.get_position() for ax in vis]
    gx0 = min(p.x0 for p in pos); gx1 = max(p.x1 for p in pos)
    gy0 = min(p.y0 for p in pos); gy1 = max(p.y1 for p in pos)

    try:
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        inv  = fig.transFigure.inverted()
        union = Bbox.union([ax.get_tightbbox(rend) for ax in vis])
        ux0, uy0 = inv.transform((union.x0, union.y0))
        ux1, uy1 = inv.transform((union.x1, union.y1))
    except Exception:
        ux0, uy0, uy1 = gx0 - 0.05, gy0 - 0.03, gy1 + 0.03

    arrow = dict(arrowstyle="-|>", mutation_scale=75,
                 color="black", lw=6.0, joinstyle="miter",
                 shrinkA=0.0, shrinkB=0.0)

    ay = uy1 + 0.012
    ax0 = gx0 + 0.30 * (gx1 - gx0)
    ax1 = gx0 + 0.85 * (gx1 - gx0)
    fig.text(ax0 - 0.012, ay, r"$Q^{2}\ [\mathrm{GeV}^{2}]$",
             ha="right", va="center",
             fontsize=fontsize, fontweight="bold")
    fig.add_artist(plt.matplotlib.patches.FancyArrowPatch(
        (ax0, ay), (ax1, ay), transform=fig.transFigure, **arrow))

    axp = ux0 - 0.018
    ay1 = gy0 + 0.78 * (gy1 - gy0)
    ay0 = gy0 + 0.22 * (gy1 - gy0)
    fig.text(axp, ay1 + 0.012, r"$x_{B}$", ha="center", va="bottom",
             fontsize=fontsize, fontweight="bold")
    fig.add_artist(plt.matplotlib.patches.FancyArrowPatch(
        (axp, ay1), (axp, ay0), transform=fig.transFigure, **arrow))


def _make_grid(binning, out, suffix, curves_fn, config_label=None,
               ylog=None, legend_title=None):
    """Generic NX x NQ grid, square panels, zero inter-panel spacing."""
    nx, nq = binning.nx, binning.nq

    fig_w, fig_h = nq * PANEL_SIZE, nx * PANEL_SIZE
    fig, axes = plt.subplots(nx, nq, figsize=(fig_w, fig_h),
                             gridspec_kw=dict(wspace=0.0, hspace=0.0),
                             sharex=False, sharey=True)
    fig.patch.set_facecolor("white")

    left, bottom = 0.10, 0.06
    span_w = 0.98 - left  # fraction of the figure width
    panel_in = span_w * fig_w / nq  # panel side, inches
    span_h = panel_in * nx / fig_h
    fig.subplots_adjust(left=left, right=left + span_w,
                        bottom=bottom, top=bottom + span_h,
                        wspace=0.0, hspace=0.0)

    curves, has_curve = {}, {}
    for ix in range(nx):
        row = nx - 1 - ix
        for iq in range(nq):
            c = curves_fn((iq, ix))
            curves[(row, iq)] = c
            has_curve[(row, iq)] = bool(c)

    visible = dict(has_curve) if HIDE_EMPTY_PANELS \
              else {k: True for k in has_curve}

    first_visible_row, last_visible_row = {}, {}
    for iq in range(nq):
        rr = [r for r in range(nx) if visible[(r, iq)]]
        if rr:
            first_visible_row[iq] = min(rr); last_visible_row[iq] = max(rr)
    first_visible_col = {}
    for row in range(nx):
        cc = [q for q in range(nq) if visible[(row, q)]]
        if cc:
            first_visible_col[row] = min(cc)

    drawn = []
    row_label_texts = {}
    legend_source_ax = None
    for ix in range(nx):
        row = nx - 1 - ix
        xlo, xhi = binning.x_bins[ix]
        for iq in range(nq):
            ax = axes[row, iq]

            if not visible[(row, iq)]:
                ax.set_visible(False)
                continue

            _setup_axes(ax)
            for vals, color, lw, ls, lbl in curves[(row, iq)]:
                ax.stairs(vals, XL_EDGES, color=color, lw=lw, ls=ls, label=lbl)
                if np.any(np.asarray(vals) != 0):
                    drawn.append(vals)

            if legend_source_ax is None and curves[(row, iq)]:
                legend_source_ax = ax

            if row == first_visible_row.get(iq):
                q2lo, q2hi = binning.q2_bins[iq]
                ax.set_title(q2_label(q2lo, q2hi), fontsize=FS_TITLE, pad=9,
                             fontweight="bold")

            if iq == first_visible_col.get(row):
                row_label_texts[row] = ax.text(
                    -0.06, 0.5, xb_label(xlo, xhi),
                    transform=ax.transAxes, ha="right", va="center",
                    rotation=0, linespacing=1.4,
                    fontsize=FS_ROWLAB, fontweight="bold")
                ax.tick_params(labelleft=True)
            else:
                ax.tick_params(labelleft=False)

            ax.tick_params(labelbottom=(row == last_visible_row.get(iq)))

    _apply_common_ylim(axes, drawn, log=ylog)
    for ax in np.ravel(axes):
        if ax.get_visible():
            _bold_ticks(ax)
    label_size = _fit_column_titles(fig, axes, row_label_texts)
    _clear_row_labels(fig, axes, row_label_texts)

    _f2_corner = r"$F_{2}^{LN}(x_B,\,Q^{2},\,x_L)$"
    _f2_flat   = r"$F_{2}^{LN}(x_B,\,Q^{2},\,x_L)$"
    if config_label is not None:
        _f2_corner += "\n" + rf"$[{config_label}]$"
        _f2_flat   += rf"$\ [{config_label}]$"

    PAIR_BOTTOM.clear()
    f2_in_legend = False
    if legend_source_ax is not None:
        f2_in_legend = _place_legend_in_empty_space(
            fig, axes, has_curve, legend_source_ax, nx, nq,
            extra_label=_f2_flat, fontsize=label_size, title=legend_title)

    # universal axis labels, anchored to the visible-panel block
    vis = [ax for ax in axes.flat if ax.get_visible()]
    if not vis:
        plt.close(fig)
        print(f"  !!! [{out}] {suffix}: every panel empty -- figure skipped")
        return
    pos = [ax.get_position() for ax in vis]
    cx = 0.5 * (min(p.x0 for p in pos) + max(p.x1 for p in pos))
    cy = 0.5 * (min(p.y0 for p in pos) + max(p.y1 for p in pos))
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()
        union = Bbox.union([ax.get_tightbbox(renderer) for ax in vis])
        ux0, uy0 = inv.transform((union.x0, union.y0))
    except Exception:
        ux0 = min(p.x0 for p in pos) - 0.03
        uy0 = min(p.y0 for p in pos) - 0.03

    pad = 0.010
    _lbl_stroke = [patheffects.withStroke(linewidth=AXIS_LABEL_STROKE,
                                          foreground="black")]
    y_xl = uy0 - pad
    pb = PAIR_BOTTOM.get("y")
    if (pb is not None and pb < uy0
            and PAIR_BOTTOM.get("x0", 1) <= cx <= PAIR_BOTTOM.get("x1", -1)):
        y_xl = pb - pad
    t_x = fig.text(cx, y_xl, r"$x_L$", ha="center", va="top",
                   fontsize=FS_AXLABEL, fontweight="bold")
    t_x.set_path_effects(_lbl_stroke)

    placed = f2_in_legend
    if not placed and HIDE_EMPTY_PANELS:
        placed = _place_f2_label_top_left(fig, axes, has_curve, nx, nq,
                                          _f2_corner, FS_AXLABEL,
                                          stroke=_lbl_stroke, dx=-0.018,
                                          row_label_texts=row_label_texts)
    if not placed:
        t_y = fig.text(ux0 - pad - 0.02, cy, _f2_flat,
                       ha="center", va="center", rotation=90,
                       fontsize=FS_AXLABEL, fontweight="bold")
        t_y.set_path_effects(_lbl_stroke)

    _place_direction_arrows(fig, axes, vis)

    os.makedirs(OUT_DIR, exist_ok=True)
    tag = f"{NAME_TAG}_" if NAME_TAG else ""
    for ext in ("png", "pdf"):
        fname = os.path.join(OUT_DIR, f"{out}_{tag}{suffix}.{ext}")
        fig.savefig(fname, dpi=180, bbox_inches="tight", facecolor="white")
        print(f"  -> {fname}")
    plt.close(fig)


def make_grid_background(f2ln_bkg, binning, out):
    _make_grid(binning, out, f"background_grid{binning.nq}x{binning.nx}",
               lambda cell: _curves_bkg(f2ln_bkg, cell),
               config_label=beam_tex(out))


def make_grid_combo(f2ln_bkg, sig_hist, ppdf, binning, out):
    _make_grid(binning, out, f"f2ln_grid{binning.nq}x{binning.nx}",
               lambda cell: _curves_combo(f2ln_bkg, sig_hist, ppdf, cell),
               config_label=beam_tex(out))


def make_grid_totals(f2ln_bkg, sig_hists, ppdfs, binning, out, dw=None):
    suffix = f"totals_grid{binning.nq}x{binning.nx}"
    if dw is not None:
        suffix += "_" + xl_tag(dw)
    _make_grid(binning, out, suffix,
               lambda cell: _curves_totals(f2ln_bkg, sig_hists, ppdfs, cell),
               config_label=beam_tex(out), ylog=False)


def dump_binning_table(name, binning, E_e, E_p):
    S = 4.0 * E_e * E_p
    print(f"\n[{name}] (Q2, x_B) grid and representative kinematics "
          f"(sqrt(s) = {np.sqrt(S):.1f} GeV):")
    print(binning.describe())
    print(f"  {'cell':>8}  {'<Q2>':>9}  {'<x_B>':>11}  {'y':>7}  "
          f"{'Q2_hi eff':>10}  {'acc.':>6}  src")
    for iq in range(binning.nq):
        q2lo, q2hi = binning.q2_bins[iq]
        for ix in range(binning.nx):
            xlo, xhi = binning.x_bins[ix]
            cell = (iq, ix)
            q2 = binning.q2_rep[cell]
            xb = binning.x_rep[cell]
            y  = compute_y(q2, xb, S)
            # what the events can actually reach inside this cell
            q_eff = effective_q2_hi(xhi, q2lo, q2hi, S)
            afull = (xhi - xlo) * (q2hi - q2lo)
            frac  = cell_phase_space_area(xlo, xhi, q2lo, q2hi, S) / afull
            src = "measured" if binning.measured[cell] else "geom (empty)"
            flag = ""
            if cell in binning.dropped:
                flag = ("  <- DROPPED, forbidden" if frac <= 0.0
                        else "  <- DROPPED, partly beyond y=1")
            elif y > 1.0:
                flag = "  <- y>1, forbidden"
            elif frac < 0.999:
                flag = "  <- partly beyond y=1"
            print(f"  ({iq},{ix})  {q2:9.3g}  {xb:11.4g}  {y:7.3f}  "
                  f"{q_eff:10.3g}  {100*frac:5.1f}%  {src}{flag}")


def main():
    global K2_CAP, OUT_DIR, NAME_TAG, YLOG, HIDE_EMPTY_PANELS, REF_HLINES
    global DROP_PARTIAL_BINS, JAM21_REPLICA_MEAN

    parser = argparse.ArgumentParser()
    parser.add_argument("--root-5x41",   default=CONFIGS[0][1])
    parser.add_argument("--root-10x100", default=CONFIGS[1][1])
    parser.add_argument("--root-18x275", default=CONFIGS[2][1])
    parser.add_argument("--tree", default="tree")
    parser.add_argument("--n-sub", type=int, default=12,
                        help="sub-samples per x_L bin when averaging the pion "
                             "PDF (lower = faster, default 12)")
    parser.add_argument("--theta-n-max", type=float, default=THETA_N_MAX_MRAD,
                        help=f"neutron angular cut in mrad "
                             f"(default {THETA_N_MAX_MRAD:g}); the p_T window "
                             f"and the flux k_perp^2 ceiling both follow from it")
    parser.add_argument("--k2-cap", type=float, default=None,
                        help="optional hard ceiling (GeV^2) on the flux "
                             "k_perp^2 integration; omit for no cap, pass "
                             "0.04 to restore the model's original truncation")
    parser.add_argument("--outdir", default=OUT_DIR,
                        help=f"folder for all figures (default {OUT_DIR!r})")
    parser.add_argument("--tag", default=NAME_TAG,
                        help=f"name tag inserted into every filename "
                             f"(default {NAME_TAG!r})")
    parser.add_argument("--keep-empty", action="store_true",
                        help="draw the full square 6x6 block of panels; by "
                             "default panels with no curve are hidden so the "
                             "grid collapses into the H1/ZEUS staircase shape")
    parser.add_argument("--reflines", default="",
                        help="optional comma-separated y values for horizontal "
                             "reference lines drawn across every panel "
                             "(default: none, e.g. --reflines 1,0.01)")
    parser.add_argument("--keep-partial", action="store_true",
                        help="keep bins that the y = 1 line cuts, normalised "
                             "to their accessible area; by default such bins "
                             "are dropped along with the fully forbidden ones")
    parser.add_argument("--no-totals", action="store_true",
                        help="skip the totals figure (one Total curve per pion "
                             "PDF set); it costs one pion rebin per extra set")
    parser.add_argument("--ylinear", action="store_true",
                        help="use a shared LINEAR y-axis; the default is a "
                             "shared LOG axis, because the grid spans several "
                             "decades and on a linear scale the low-rate "
                             "panels sit flat against the axis")
    parser.add_argument("--jam21-member0", action="store_true",
                        help="use member 0 of JAM21PionPDFnlo instead of the mean "
                             "of the replicas (as in arXiv v1)")
    args = parser.parse_args()

    JAM21_REPLICA_MEAN = not args.jam21_member0
    YLOG     = not args.ylinear
    DROP_PARTIAL_BINS = not args.keep_partial
    HIDE_EMPTY_PANELS = not args.keep_empty
    REF_HLINES = [float(v) for v in args.reflines.split(',') if v.strip()]
    K2_CAP   = args.k2_cap
    OUT_DIR  = args.outdir
    NAME_TAG = args.tag
    os.makedirs(OUT_DIR, exist_ok=True)
    theta_override = (args.theta_n_max
                      if args.theta_n_max != THETA_N_MAX_MRAD else None)

    paths = {"5x41":   args.root_5x41,
             "10x100": args.root_10x100,
             "18x275": args.root_18x275}

    print()
    pion_pdfs = load_pion_pdfs()
    ppdf = next((p for p in pion_pdfs if p.key == MAIN_PION_KEY), pion_pdfs[0])
    print(f"\nPion PDF sets loaded: "
          f"{', '.join(p.label for p in pion_pdfs)}  (JAM21: "
          f"{'mean of the replicas' if JAM21_REPLICA_MEAN else 'member 0'})")
    print(f"  background+pion figure uses: {ppdf.label} ({ppdf.setname})")
    if not args.no_totals and len(pion_pdfs) > 1:
        print(f"  totals figure compares    : "
              f"{', '.join(p.label for p in pion_pdfs)}")

    print("\nNeutron selection: an angular cut theta_n < theta_max ONLY, "
          "per beam config:")
    for name, _, E_e, E_p in CONFIGS:
        th = theta_override if theta_override is not None \
             else CFG_THETA_N_MRAD.get(name, THETA_N_MAX_MRAD)
        tn = float(np.tan(th * 1e-3))
        print(f"    {name:>7} : theta_n < {th:g} mrad  "
              f"(tan = {tn:.6g}; p_T^max = {tn * E_p:.3f} GeV at x_L = 1)")
    if theta_override is not None:
        print(f"    [--theta-n-max {theta_override:g} overrides the table above]")
    print("p_T is derived, not cut on:  p_T = x_L * E_p * tan(theta_n)")
    print(f"Flux k_perp^2 ceiling = p_T^max^2"
          + (f", capped at {K2_CAP:g} GeV^2" if K2_CAP is not None
             else " (uncapped)"))
    print("\nPER-CONFIG (Q2, x_B) grids -- 6x6 log-spaced over each beam's own "
          "accessible region:")
    for name, _, E_e, E_p in CONFIGS:
        q2lo, q2hi, xlo, xhi = CFG_RANGES[name]
        S_cfg   = 4.0 * E_e * E_p
        y_worst = q2hi / (S_cfg * xlo)
        print(f"    {name:>7} : Q2 [{q2lo:g}, {q2hi:g}] GeV^2, "
              f"x_B [{xlo:g}, {xhi:g}],  S = {S_cfg:g},  "
              f"worst-corner y = {y_worst:.3g}")
    if DROP_PARTIAL_BINS:
        print("  Grids extend past y = 1 (full low-x_B reach kept), but only "
              "cells lying ENTIRELY inside y <= 1 are plotted: forbidden and "
              "partially-cut bins are both dropped.")
    else:
        print("  Grids extend past y = 1; forbidden cells stay empty and "
              "cells the line cuts are kept, normalised to their accessible "
              "area (--keep-partial).")
    print(f"\nFigures -> {os.path.abspath(OUT_DIR)}/<cfg>_{NAME_TAG}_*.png|pdf")

    failed = []
    for name, _, E_e, E_p in CONFIGS:
        root_path = paths[name]
        binning   = CFG_BINNING[name]
        theta_n = (theta_override if theta_override is not None
                   else CFG_THETA_N_MRAD.get(name, THETA_N_MAX_MRAD))
        _set_theta_n_max(theta_n)
        print(f"\n{'='*74}\nLoading Pythia ({name}): {root_path}\n"
              f"  neutron cut: theta_n < {THETA_N_MAX_MRAD:g} mrad\n{'='*74}")
        try:
            counts_base, sigma_dis, N_gen = process_root(
                root_path, E_e, E_p, name, binning, args.tree)
        except Exception as e:
            import traceback
            print(f"\n!!! [{name}] FAILED -- skipping, continuing with the rest.")
            traceback.print_exc()
            failed.append((name, str(e)))
            continue

        S_cfg = 4.0 * E_e * E_p
        dump_binning_table(name, binning, E_e, E_p)

        k2fn = k2max_for_config(E_p)
        print(f"\n[{name}] pion PDFs: k_perp^2 max "
              f"{k2fn(XL_EDGES[0]):.4f} -> {k2fn(XL_EDGES[-1]):.4f} GeV^2, "
              f"evaluated at the measured <Q2>,<x_B> of each cell")
        wanted = pion_pdfs if not args.no_totals else [ppdf]

        for iw, dw in enumerate(XL_WIDTHS):
            set_xl_binning(dw)
            f2ln_bkg = f2ln_bkg_for_binning(counts_base, binning, S_cfg,
                                            sigma_dis, N_gen)
            n_sub = max(3, int(round(args.n_sub * dw / XL_WIDTHS[0])))
            print(f"\n[{name}] x_L binning: {N_XL} bins of "
                  f"Delta x_L = {dw:g}   (pion n_sub = {n_sub})")
            sig_hists = {}
            for pp in wanted:
                sig_hists[pp.key] = rebin_pdf_to_xl_edges(
                    pp, k2fn, binning, n_sub=n_sub,
                    tag=f" {name} dxL={dw:g}")

            print(f"Saving grids for {name}, Delta x_L = {dw:g} ...")
            if iw == 0:
                make_grid_background(f2ln_bkg, binning, out=name)
                make_grid_combo(f2ln_bkg, sig_hists[ppdf.key], ppdf,
                                binning, out=name)
            if not args.no_totals and len(wanted) > 1:
                make_grid_totals(f2ln_bkg, sig_hists, wanted, binning,
                                 out=name, dw=dw)
        set_xl_binning(XL_WIDTHS[0])

    print("\nNote: no combined overlay grid -- each config now has its own "
          "(Q2, x_B) binning, so the panels are not comparable cell by cell.")

    if failed:
        print("\n=== Failed configs ===")
        for name, msg in failed:
            print(f"  {name}: {msg}")

    print("\nDone.")


if __name__ == "__main__":
    main()
