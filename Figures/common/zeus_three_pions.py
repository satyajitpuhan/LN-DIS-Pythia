"""ZEUS: Pythia "CT18NLO + Remnant" with the JAM21, xFitter and GRV Sullivan terms
(F2^LN grids, chi^2) and the overview of three pion PDFs times five Pythia settings.
"""

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


# PYTHIA (ROOT file) -- ZEUS cuts, ZEUS binning
def load_pythia(path):
    """Streams the tree in CHUNK_SIZE-entry batches to keep peak RAM bounded."""
    import uproot

    with uproot.open(path, object_cache=None, array_cache=None) as f:
        try:
            sigma_dis = float(f["sigmaGen_nb"].members["fVal"])
        except Exception:
            raise RuntimeError("sigmaGen_nb not found in ROOT file")

        tree = None
        for key in f.keys():
            try:
                obj = f[key]
                if hasattr(obj, "keys") and "xL" in obj.keys():
                    tree = obj
                    break
            except Exception:
                pass
        if tree is None:
            raise RuntimeError("No tree with xL found")

        av    = set(tree.keys())
        q2_b  = pick(av, ["Q2", "q2"])
        xb_b  = pick(av, ["xB", "xb"])
        eep_b = pick(av, ["EePrime", "Ee_prime", "ePrime"])
        thN_b = pick(av, ["theta_n", "thetaN"])
        pt_b  = pick(av, ["pT", "pt", "PT"])

        want = list(dict.fromkeys(
            ["xL"] + [b for b in [q2_b, xb_b, eep_b, thN_b, pt_b] if b]))

        total_entries = tree.num_entries
        print(f"  Tree has {total_entries:,} entries -- "
              f"streaming in chunks of {CHUNK_SIZE:,} ...")

        counts       = np.zeros((NQ_ZEUS, NX_ZEUS, N_XL), dtype=np.int64)
        n_seen       = 0
        n_after_cuts = 0
        n_after_pt   = 0

        for chunk in tree.iterate(expressions=want, library="np",
                                   step_size=CHUNK_SIZE):
            xl = chunk["xL"]
            n_chunk = len(xl)
            n_seen += n_chunk

            sel = np.ones(n_chunk, dtype=bool)
            if eep_b and eep_b in chunk:
                sel &= chunk[eep_b] > EePrime_MIN
            if thN_b and thN_b in chunk:
                sel &= chunk[thN_b] <= THETA_N_MAX
            n_after_cuts += int(sel.sum())

            pt = chunk[pt_b] if (pt_b and pt_b in chunk) else np.full(n_chunk, np.inf)
            sel_base = sel & (pt < PT_COEFF * xl)
            n_after_pt += int(sel_base.sum())

            if sel_base.any():
                xl_s = xl[sel_base]
                q2_s = chunk[q2_b][sel_base] if (q2_b and q2_b in chunk) else None
                xb_s = chunk[xb_b][sel_base] if (xb_b and xb_b in chunk) else None

                for iq, (_, q2lo, q2hi) in enumerate(Q2_BINS_ZEUS):
                    q2_mask = ((q2_s >= q2lo) & (q2_s < q2hi)
                               if q2_s is not None
                               else np.ones(len(xl_s), dtype=bool))
                    if not q2_mask.any():
                        continue
                    for ix, (_, xlo, xhi) in enumerate(X_BINS_ZEUS):
                        xb_mask = ((xb_s >= xlo) & (xb_s < xhi)
                                   if xb_s is not None
                                   else np.ones(len(xl_s), dtype=bool))
                        cell_mask = q2_mask & xb_mask
                        if not cell_mask.any():
                            continue
                        hist, _ = np.histogram(xl_s[cell_mask], bins=XL_EDGES)
                        counts[iq, ix] += hist

            print(f"    ... {n_seen:,}/{total_entries:,} entries read", end="\r")

        print()

    print(f"  sigma_DIS = {sigma_dis:.4f} nb")
    print(f"  N_total (generated)                  = {N_TOTAL:,}")
    print(f"  Events after E_e', theta_n cuts       = {n_after_cuts:,}")
    print(f"  Events + pT < {PT_COEFF}*xL            = {n_after_pt:,}")

    return dict(sigma_dis=sigma_dis, counts=counts)


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


# BOLD ("H1-style") Q^2 / x_B labels, and figure-level legend placement
# in the largest empty corner of the ZEUS staircase grid.
def _q2_label_bold(lo, hi):
    return rf"$Q^2 \in [{lo:.4g},\,{hi:.4g}]$"


def _xb_label_bold(lo, hi):
    # Two-line form, identical to zeus_five_settings.py's _xb_label_bold -- keeps the row
    # label compact enough to sit in the staircase margin at FS_ROWLAB.
    return rf"$x_{{B}} \in [{sci(lo)},$" + "\n" + rf"${sci(hi)}]$"


def _largest_corner_empty_block(cell_has_data, nrow, ncol):
    """Largest fully-empty rectangle anchored at the bottom-right or
    top-left corner (whichever is bigger).
    """
    def _all_empty(cells):
        return all(not cell_has_data.get(c, False) for c in cells)

    best = None
    for r0 in range(nrow):               # bottom-right anchored
        for q0 in range(ncol):
            block = [(r, q) for r in range(r0, nrow) for q in range(q0, ncol)]
            if block and _all_empty(block):
                if best is None or len(block) > best[0]:
                    best = (len(block), block)
    for r1 in range(nrow):               # top-left anchored
        for q1 in range(ncol):
            block = [(r, q) for r in range(0, r1 + 1) for q in range(0, q1 + 1)]
            if block and _all_empty(block):
                if best is None or len(block) > best[0]:
                    best = (len(block), block)
    return best[1] if best is not None else None


def _place_legend_in_empty_space(fig, axes, cell_has_data, source_ax, nrow, ncol):
    """Grab the curve handles from a populated panel and draw one big,
    bold legend centred in the largest empty corner block, shrinking the
    font until it fits inside that block -- identical algorithm to
    zeus_five_settings.py's _place_legend_in_empty_space().
    """
    handles, labels = source_ax.get_legend_handles_labels()
    if not handles:
        return

    block = _largest_corner_empty_block(cell_has_data, nrow, ncol)
    if not block:
        fig.legend(handles, labels, loc="upper right",
                   prop={"weight": "bold", "size": FS_LEGEND}, framealpha=0.96)
        return

    positions = [axes[r][q].get_position() for r, q in block]
    bx0 = min(p.x0 for p in positions); bx1 = max(p.x1 for p in positions)
    by0 = min(p.y0 for p in positions); by1 = max(p.y1 for p in positions)
    cx, cy = 0.5 * (bx0 + bx1), 0.5 * (by0 + by1)
    max_w, max_h = 0.96 * (bx1 - bx0), 0.96 * (by1 - by0)

    fs = float(FS_LEGEND)
    leg = None
    for _ in range(12):
        if leg is not None:
            leg.remove()
        leg = fig.legend(handles, labels, loc="center",
                         bbox_to_anchor=(cx, cy),
                         ncol=1, handlelength=2.0, handletextpad=0.7,
                         borderpad=0.9, labelspacing=0.55, framealpha=0.96,
                         edgecolor="0.35",
                         prop={"weight": "bold", "size": fs})
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_linewidth(1.2)
        try:
            fig.canvas.draw()
            bb = leg.get_window_extent().transformed(fig.transFigure.inverted())
        except Exception:
            break
        if (bb.width <= max_w and bb.height <= max_h) or fs <= 8.0:
            break
        fs *= max(0.55, min(0.92, min(max_w / bb.width, max_h / bb.height)))


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


# CELL DRAWING
#   mode='pions': CT18NLO+Remnant background (grey, dashed) + three pion PDF
#                 histograms (JAM21 / xFitter / GRV) + ZEUS data
#   mode='sums' : three summed curves  CT18NLO+Remnant + pion  + ZEUS data
def _draw_cell_three_zeus(ax, ct_f2ln, pion_hists, cell, zeus_cell, mode,
                          show_legend=False, legend_fontsize=6):
    """mode='pions' : CT18NLO+Remnant bkg (dashed) + 3 raw pion histograms
    mode='sums'  : 3 summed curves CT18NLO+Remnant+pion (solid)
    mode='both'  : bkg (dashed) + 3 raw pion histograms (thin, dotted)
                   + 3 summed curves (thick, solid) - everything in one
                   panel, same content used for the standalone individual
                   per-cell plots.
    show_legend  : draw a per-axes legend (used for standalone individual
                   plots; the 7x7 grids instead use one shared figure-level
                   legend, so they leave this False).
    """
    py = ct_f2ln.get(cell)

    if mode in ("pions", "both"):
        if py is not None:
            # No "(bkg)" suffix -- zeus_five_settings.py's raw-Pythia legend entries are
            # just _pythia_label(label), e.g. "Pythia (CT18NLO+Remnant)".
            ax.stairs(py, XL_EDGES, color=C_CT18RE_BKG, lw=LW_BKG, ls='--',
                      label=_pythia_label(CT18RE_LABEL))
        for key, _pn, label, color in PION_PDF_SETS:
            h = pion_hists.get(key, {}).get(cell)
            if h is not None:
                lw  = LW_PION * 0.75 if mode == "both" else LW_PION
                ls  = ':' if mode == "both" else '-'
                # Plain pion-PDF name, e.g. "JAM21" -- matches zeus_five_settings.py's raw
                # "JAM21" legend entry (no "(raw)" qualifier there either).
                ax.stairs(h, XL_EDGES, color=color, lw=lw, ls=ls, label=label)

    if mode in ("sums", "both"):
        for key, _pn, label, color in PION_PDF_SETS:
            h = pion_hists.get(key, {}).get(cell)
            if py is not None and h is not None:
                combined = py + np.where(np.isfinite(h), h, 0.0)
                ax.stairs(combined, XL_EDGES, color=color, lw=LW_SUM,
                          label=f"{_pythia_label(CT18RE_LABEL)} + {label}")

    if zeus_cell is not None:
        xl_d, f2_d, e_d = zeus_cell
        # EXACTLY the H1/ZEUS data-point style used in zeus_five_settings.py: filled black
        # circles with a single VERTICAL error bar -- no horizontal x
        # error bars, heavy marker/line weights.
        ax.errorbar(xl_d, f2_d, yerr=e_d,
                    fmt="o", ms=DATA_MS, mfc="black", mec="black",
                    mew=DATA_MEW, ecolor="black", elinewidth=DATA_ELW,
                    capsize=DATA_CAPS, capthick=DATA_ELW,
                    color="black", zorder=10, label="ZEUS data")

    ax.set_ylim(0., 0.35)
    ax.set_xlim(XL_EDGES[0] - 0.02, 1.0)
    ax.set_facecolor("white")

    for sp in ax.spines.values():
        sp.set_linewidth(SPINE_LW)

    # zeus_five_settings.py-style bold ticks: explicit x_L ticks with the 0.25 tick dropped,
    # bold FuncFormatter labels, matching major/minor tick_params, and the
    # same explicit y-ticks (0.10/0.20/0.30) as zeus_five_settings.py.
    ax.set_xticks([0.50, 0.75, 1.00])
    ax.xaxis.set_major_formatter(FuncFormatter(_xl_tick_fmt))
    ax.set_yticks([0.10, 0.20, 0.30])
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True,
                   labelsize=FS_TICK)
    ax.tick_params(which="major", length=10, width=2.0, pad=6)
    ax.tick_params(which="minor", length=5, width=1.4)
    plt.setp(ax.get_xticklabels(), fontweight="bold")
    plt.setp(ax.get_yticklabels(), fontweight="bold")
    ax.minorticks_on()

    if show_legend:
        ax.legend(loc="upper right", fontsize=legend_fontsize,
                  handlelength=1.2, handletextpad=0.3,
                  borderpad=0.3, framealpha=0.85, labelspacing=0.25)


# Staircase-corner block placement for the F_2^LN label -- ported
# directly from zeus_five_settings.py so this label sits INSIDE the empty top-left
# corner (horizontal), matching the published H1/ZEUS style, instead
# of a plain rotated label fixed outside the grid.
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


def _block_centre(axes, block):
    positions = [axes[r][q].get_position() for r, q in block]
    x0 = min(p.x0 for p in positions); x1 = max(p.x1 for p in positions)
    y0 = min(p.y0 for p in positions); y1 = max(p.y1 for p in positions)
    return 0.5 * (x0 + x1), 0.5 * (y0 + y1)


def _place_f2_label_top_left(fig, axes, cell_has_data, nrow, ncol, text,
                             fontsize, stroke=None, dx=0.0):
    """Put the F_2^LN axis label INSIDE the empty top-left staircase
    corner, horizontal -- the way the published H1/ZEUS figure does it.
    Also places "ZEUS" label above it.
    `dx` (figure fraction) shifts the label horizontally, e.g. negative
    to move it further left and clear of the staircase edge.
    Returns True if placed, False if there is no empty top-left block.
    """
    block = _largest_empty_block_at(cell_has_data, nrow, ncol, "top-left")
    if not block:
        return False
    cx, cy = _block_centre(axes, block)
    cx += dx
    
    # Place "ZEUS" label above F_2LN label
    t_zeus = fig.text(cx, cy + 0.03, "ZEUS", ha="center", va="center",
                      fontsize=fontsize, fontweight="bold")
    if stroke:
        t_zeus.set_path_effects(stroke)
    
    # Place F_2LN label below ZEUS
    t = fig.text(cx, cy, text, ha="center", va="center",
                 fontsize=fontsize, fontweight="bold")
    if stroke:
        t.set_path_effects(stroke)
    return True


# ZEUS 7x7 GRID  (H1 staircase style)
def _make_grid_three_zeus(ct_f2ln, pion_hists, zeus_data, out, suffix, mode):
    # figsize / zero gridspec spacing taken from zeus_five_settings.py's
    # _make_grid_bg_generic -- panels touch edge-to-edge, "H1 staircase" style.
    fig, axes = plt.subplots(NX_ZEUS, NQ_ZEUS,
                             figsize=(NQ_ZEUS * 4.8, NX_ZEUS * 4.0),
                             gridspec_kw=dict(wspace=0.0, hspace=0.0),
                             sharex=False, sharey=False)
    fig.patch.set_facecolor("white")

    # which (row, iq) panels carry a ZEUS measurement
    cell_has_data = {}
    for ix in range(NX_ZEUS):
        row = NX_ZEUS - 1 - ix
        for iq in range(NQ_ZEUS):
            cell_has_data[(row, iq)] = ((iq, ix) in zeus_data)

    legend_cell = None
    for row in range(NX_ZEUS):
        for iq in range(NQ_ZEUS):
            if cell_has_data[(row, iq)]:
                legend_cell = (row, iq); break
        if legend_cell is not None:
            break

    first_visible_row, last_visible_row = {}, {}
    for iq in range(NQ_ZEUS):
        rr = [row for row in range(NX_ZEUS) if cell_has_data[(row, iq)]]
        if rr:
            first_visible_row[iq] = min(rr); last_visible_row[iq] = max(rr)
    first_visible_col = {}
    for row in range(NX_ZEUS):
        cc = [iq for iq in range(NQ_ZEUS) if cell_has_data[(row, iq)]]
        if cc:
            first_visible_col[row] = min(cc)

    legend_source_ax = None
    for ix in range(NX_ZEUS):
        row = NX_ZEUS - 1 - ix
        _, xlo, xhi = X_BINS_ZEUS[ix]
        for iq in range(NQ_ZEUS):
            ax   = axes[row, iq]
            cell = (iq, ix)

            if not cell_has_data[(row, iq)]:
                ax.set_visible(False)
                continue

            ax.set_facecolor("white")

            _draw_cell_three_zeus(ax, ct_f2ln, pion_hists, cell,
                                  zeus_data.get(cell), mode)

            if (row, iq) == legend_cell:
                legend_source_ax = ax

            if row == first_visible_row.get(iq):
                _, q2lo, q2hi = Q2_BINS_ZEUS[iq]
                ax.set_title(_q2_label_bold(q2lo, q2hi), fontsize=FS_TITLE,
                             pad=9, fontweight="bold")

            if iq == first_visible_col.get(row):
                # zeus_five_settings.py places the (two-line) x_B row label as free-standing
                # text just outside the panel's left edge, rather than a
                # matplotlib ylabel, so long labels don't get clipped.
                ax.text(-0.26, 0.5, _xb_label_bold(xlo, xhi),
                        transform=ax.transAxes, ha="right", va="center",
                        rotation=0, linespacing=1.4,
                        fontsize=FS_ROWLAB, fontweight="bold")
                ax.tick_params(labelleft=True)
            else:
                ax.tick_params(labelleft=False)

            ax.tick_params(labelbottom=(row == last_visible_row.get(iq)))

    # one big, bold legend in the largest empty corner
    if legend_source_ax is not None:
        _place_legend_in_empty_space(fig, axes, cell_has_data,
                                       legend_source_ax, NX_ZEUS, NQ_ZEUS)

    # universal axis labels anchored just outside the visible-panel block
    vis = [ax for ax in axes.flat if ax.get_visible()]
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
    # Same font size, stroked so the glyphs render heavier (mathtext has no
    # weight above bold).
    _lbl_stroke = [patheffects.withStroke(linewidth=AXIS_LABEL_STROKE,
                                          foreground="black")]
    t_x = fig.text(cx, uy0 - pad, r"$x_L$",
                   ha="center", va="top", fontsize=FS_AXLABEL, fontweight="bold")
    t_x.set_path_effects(_lbl_stroke)

    _f2_text = r"$F_{2}^{LN}(x_B,\,Q^{2},\,x_L)$"
    if not _place_f2_label_top_left(fig, axes, cell_has_data, NX_ZEUS, NQ_ZEUS,
                                    _f2_text, FS_AXLABEL, stroke=_lbl_stroke,
                                    dx=-0.045):
        t_y = fig.text(ux0 - pad - 0.02, cy, _f2_text,
                 ha="center", va="center", rotation=90,
                 fontsize=FS_AXLABEL, fontweight="bold")
        t_y.set_path_effects(_lbl_stroke)

    for ext in ("png", "pdf"):
        fname = f"{out}_f2ln_grid_zeus{suffix}.{ext}"
        fig.savefig(fname, dpi=180, bbox_inches="tight", facecolor="white")
        print(f"  -> {fname}")
    plt.close(fig)


def make_grid_three_pions_zeus(ct_f2ln, pion_hists, zeus_data, out):
    """Grid 1: CT18NLO+Remnant background + three raw pion-PDF histograms."""
    _make_grid_three_zeus(ct_f2ln, pion_hists, zeus_data, out,
                          "_three_pion_pdfs", "pions")


def make_grid_ctre_plus_pions_zeus(ct_f2ln, pion_hists, zeus_data, out):
    """Grid 2: three summed curves  CT18NLO+Remnant + pion  (added)."""
    _make_grid_three_zeus(ct_f2ln, pion_hists, zeus_data, out,
                          "_ctre_plus_three_pions", "sums")


# INDIVIDUAL PER-CELL PLOTS  (make_individual(): one standalone PNG/PDF per (Q^2, x_B) cell across the
# full NQ_ZEUS x NX_ZEUS grid, each showing CT18NLO+Remnant bkg + all 3
# raw pion PDFs + all 3 CT18NLO+Remnant+pion sums + ZEUS data (where
# present), with its own legend - no staircase hiding.)
def make_individual_zeus(ct_f2ln, pion_hists, zeus_data, out):
    ind_dir = f"{out}_f2ln_zeus_individual"
    os.makedirs(ind_dir, exist_ok=True)

    for ix, (x_nom, xlo, xhi) in enumerate(X_BINS_ZEUS):
        for iq, (q2_nom, q2lo, q2hi) in enumerate(Q2_BINS_ZEUS):
            fig, ax = plt.subplots(figsize=(5, 4))
            cell = (iq, ix)

            _draw_cell_three_zeus(ax, ct_f2ln, pion_hists, cell,
                                  zeus_data.get(cell), mode="both",
                                  show_legend=True, legend_fontsize=6)

            ax.set_xlabel(r"$x_L$", fontsize=12, fontweight="bold")
            ax.set_ylabel(r"$F_{2}^{LN}$", fontsize=12, fontweight="bold")
            ax.set_title(q2_label(q2lo, q2hi) + "   " + xb_label(xlo, xhi),
                        fontsize=9)

            fig.tight_layout()
            base = f"{ind_dir}/f2ln_q{iq}_x{ix}"
            for ext in ("png", "pdf"):
                fig.savefig(f"{base}.{ext}", dpi=150, bbox_inches="tight")
            plt.close(fig)

    print(f"  → {ind_dir}/  ({NQ_ZEUS*NX_ZEUS} plots)")


# CHI^2  (ZEUS data vs. each of the three CT18NLO+Remnant + pion sums)
def compute_chi2_ctre_plus_pions(ct_f2ln, pion_hists, zeus_data):
    """For each pion PDF (JAM21 / xFitter / GRV), compute the overall chi^2 of
    ZEUS data vs. the summed curve  CT18NLO+Remnant + pion  - exactly the
    curve drawn in make_grid_ctre_plus_pions_zeus() (mode='sums').
    """
    results = {}
    for key, _pn, label, _color in PION_PDF_SETS:
        h_dict = pion_hists.get(key)
        if h_dict is None:
            continue

        curve_label = f"{_pythia_label(CT18RE_LABEL)} + {label}"
        chi2 = 0.0
        ndof = 0
        for cell, (xl_d, f2_d, e_d) in zeus_data.items():
            py = ct_f2ln.get(cell)
            h  = h_dict.get(cell)
            if py is None or h is None:
                continue

            combined = py + np.where(np.isfinite(h), h, 0.0)

            # which x_L bin (stairs step) each data point falls into
            bin_idx = np.clip(np.digitize(xl_d, XL_EDGES) - 1, 0, N_XL - 1)
            model_vals = combined[bin_idx]

            valid = np.isfinite(model_vals) & np.isfinite(f2_d) & (e_d > 0)
            if not valid.any():
                continue
            chi2 += float(np.sum(((f2_d[valid] - model_vals[valid]) / e_d[valid]) ** 2))
            ndof += int(valid.sum())

        results[curve_label] = dict(
            chi2=chi2,
            ndof=ndof,
            chi2_per_dof=(chi2 / ndof if ndof > 0 else float("nan")),
        )
    return results


# PUBLICATION-STYLE OVERVIEW PLOT (All combinations)

# PUBLICATION-STYLE OVERVIEW GRID  (all combinations)
#   3 rows  = pion PDFs      (JAM21 / xFitter / GRV)
#   5 cols  = Pythia proton PDF sets
#   Every panel shows exactly three model curves plus the data:
#       * DIS fragmentation  -- cell-averaged Pythia   F_2^LN
#       * PDF average        -- cell-averaged pion-PDF F_2^LN
#       * Total              -- their sum
#       * ZEUS (avg)         -- inverse-variance cell-average of the data
#   House style: bold serif, heavy panel boxes,
#   panels touching edge-to-edge, universal x_L / F_2^LN(x_B,Q^2,x_L)
#   axis labels, one shared legend, and per-panel chi^2/N only.

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


# MAIN
def main():
    default_ctre = next((s["path"] for s in PYTHIA_FILES
                         if s["label"] == "CT18NLO + Remnant"), None)

    parser = argparse.ArgumentParser()
    parser.add_argument("--zdat",     default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "zeus.dat"))
    parser.add_argument("--f2ptable", default=F2P_TABLE_FILE,
                        help="F2p_CT18NLO_values.txt (x, Q2, F2^p NLO)")
    parser.add_argument("--ctre",     default=default_ctre,
                        help="CT18NLO+Remnant Pythia ROOT file")
    parser.add_argument("--out",      default="compare_f2ln_zeus_ctre_three")
    args = parser.parse_args()

    # -- ZEUS data --------------------------------------------------
    print(f"Loading ZEUS F_2LN data from {args.zdat} "
          f"(F2p sourced from {args.f2ptable}) ...")
    zeus_data = load_zeus_f2ln(args.zdat, f2p_table_file=args.f2ptable)
    print(f"  Loaded {len(zeus_data)} (Q2,x) cells "
          f"({sum(len(v[0]) for v in zeus_data.values())} data points)")

    # -- three pion PDFs (JAM21 / xFitter / GRV) on the ZEUS grid ---
    print("\nComputing three pion-PDF F_2LN histograms "
          "(JAM21 / xFitter / GRV) on the ZEUS grid ...")
    pion_hists = load_pion_hists_zeus(PION_PDF_SETS)
    if not pion_hists:
        raise SystemExit("No pion PDFs could be loaded — nothing to plot.")
    print(f"  Loaded pion PDFs: {', '.join(pion_hists.keys())}")

    # -- CT18NLO+Remnant Pythia DIS background ----------------------
    print(f"\nLoading Pythia (CT18NLO+Remnant): {args.ctre}")
    if not args.ctre or not os.path.exists(args.ctre):
        raise SystemExit(f"CT18NLO+Remnant ROOT file not found: {args.ctre}")
    py_data = load_pythia(args.ctre)
    print("Computing F_2LN from Pythia dσ/dx_L (CT18NLO+Remnant, ZEUS cuts/bins) ...")
    ct_f2ln, _nevts = compute_all_pythia_f2ln_zeus(py_data)

    # -- Grid 1: three pion PDFs (+ CT18NLO+Remnant background) -----
    print("\nSaving ZEUS three-pion-PDF grid ...")
    make_grid_three_pions_zeus(ct_f2ln, pion_hists, zeus_data, args.out)

    # -- Grid 2: CT18NLO+Remnant + pion sums (added curves) ---------
    print("\nSaving ZEUS CT18NLO+Remnant + three-pion sum grid ...")
    make_grid_ctre_plus_pions_zeus(ct_f2ln, pion_hists, zeus_data, args.out)

    # -- overall chi^2: ZEUS data vs. each CT18NLO+Remnant + pion sum --
    print("\nComputing overall chi^2 for CT18NLO+Remnant + each pion PDF ...")
    chi2_ctre_plus_pions = compute_chi2_ctre_plus_pions(ct_f2ln, pion_hists, zeus_data)
    save_chi2_table(
        chi2_ctre_plus_pions,
        f"{args.out}_chi2_ctre_plus_three_pions.txt",
        title="Overall chi^2: ZEUS data vs. CT18NLO+Remnant + pion PDF curves",
    )

    # -- Individual per-cell plots ------
    print("\nSaving individual per-cell plots ...")
    make_individual_zeus(ct_f2ln, pion_hists, zeus_data, args.out)

    # NOW DO THE SAME WITH CT18NLO (WITHOUT REMNANT)
    global CT18RE_LABEL  # Temporarily change for CT18NLO plots
    ct18nlo_path = os.path.join(PYTHIA_OUT, "zeus_ct18nlo.root")
    original_label = CT18RE_LABEL  # Save original
    CT18RE_LABEL = "CT18NLO"  # Change label for CT18NLO plots
    
    print(f"\n{'='*70}")
    print(f"Loading Pythia (CT18NLO): {ct18nlo_path}")
    if not os.path.exists(ct18nlo_path):
        raise SystemExit(f"CT18NLO ROOT file not found: {ct18nlo_path}")
    
    py_data_ct18nlo = load_pythia(ct18nlo_path)
    print("Computing F_2LN from Pythia dσ/dx_L (CT18NLO, ZEUS cuts/bins) ...")
    ct_f2ln_ct18nlo, _nevts_ct18nlo = compute_all_pythia_f2ln_zeus(py_data_ct18nlo)

    # -- Grid 1: three pion PDFs (+ CT18NLO background) -----
    print("\nSaving ZEUS three-pion-PDF grid (CT18NLO) ...")
    make_grid_three_pions_zeus(ct_f2ln_ct18nlo, pion_hists, zeus_data, "compare_f2ln_zeus_ct18nlo_three")

    # -- Grid 2: CT18NLO + pion sums (added curves) ---------
    print("\nSaving ZEUS CT18NLO + three-pion sum grid ...")
    make_grid_ctre_plus_pions_zeus(ct_f2ln_ct18nlo, pion_hists, zeus_data, "compare_f2ln_zeus_ct18nlo_three")

    # -- overall chi^2: ZEUS data vs. each CT18NLO + pion sum --
    print("\nComputing overall chi^2 for CT18NLO + each pion PDF ...")
    chi2_ct18nlo_plus_pions = compute_chi2_ctre_plus_pions(ct_f2ln_ct18nlo, pion_hists, zeus_data)
    save_chi2_table(
        chi2_ct18nlo_plus_pions,
        "compare_f2ln_zeus_ct18nlo_three_chi2_ct18nlo_plus_three_pions.txt",
        title="Overall chi^2: ZEUS data vs. CT18NLO + pion PDF curves",
    )

    # -- Individual per-cell plots for CT18NLO ------
    print("\nSaving individual per-cell plots (CT18NLO) ...")
    make_individual_zeus(ct_f2ln_ct18nlo, pion_hists, zeus_data, "compare_f2ln_zeus_ct18nlo_three")
    
    CT18RE_LABEL = original_label  # Restore original label

    # PUBLICATION-STYLE OVERVIEW: All Pythia x All Pion combinations
    print(f"\n{'='*70}")
    print("Generating publication-style overview grid...")
    print(f"{'='*70}")
    
    # Load all Pythia PDFs and compute F_2LN
    pythia_dict = {}  # {label: (ct_f2ln, label, color)}
    
    for pythia_spec in PYTHIA_FILES:
        pythia_path = pythia_spec["path"]
        pythia_label = pythia_spec["label"]
        pythia_color = pythia_spec["color"]
        
        if not os.path.exists(pythia_path):
            print(f"  ⚠ Skipping {pythia_label}: file not found at {pythia_path}")
            continue
        
        print(f"\n  Loading {pythia_label}...")
        try:
            py_data_all = load_pythia(pythia_path)
            ct_f2ln_all, _ = compute_all_pythia_f2ln_zeus(py_data_all)
            pythia_dict[pythia_label] = (ct_f2ln_all, pythia_label, pythia_color)
            print(f"    ✓ Loaded successfully")
        except Exception as e:
            print(f"    ✗ Error: {e}")
            continue
    
    if len(pythia_dict) > 0:
        print(f"\nCreating publication grid with {len(pythia_dict)} Pythia PDFs...")
        make_publication_overview_grid(pythia_dict, pion_hists, zeus_data, args.out)
    else:
        print("  ⚠ No Pythia PDFs could be loaded for overview grid")

    print("\nDone.")


if __name__ == "__main__":
    main()
