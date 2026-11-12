"""H1: Pythia "CT18 NLO" with the JAM21, xFitter and GRV Sullivan terms (F2^LN grids and
chi^2 table).
"""

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

# FONT SIZES
FS_AXLABEL = 42  # F_2^LN(...) and x_L axis labels
FS_TITLE   = 28  # Q^2 column titles
FS_ROWLAB  = 26  # x_B row labels
FS_TICK    = 24  # x and y tick numbers
FS_LEGEND  = 24  # the one shared corner legend

matplotlib.rcParams.update({
    "font.family"         : "serif",
    "font.serif"          : ["Times New Roman", "DejaVu Serif"],
    "font.weight"         : "bold",
    "mathtext.fontset"    : "dejavuserif",
    "mathtext.default"    : "bfit",  # bold *italic* math (H1 style)
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
    "axes.linewidth"      : 3.2,  # bold panel boundaries
})

SPINE_LW   = 3.2  # panel box
LW_BKG     = 2.6  # CT18 NLO background (dashed grey)
LW_PION    = 3.2  # raw pion-PDF histograms
LW_SUM     = 3.2  # CT18 + pion sums
DATA_MS    = 7.0  # H1 marker size
DATA_MEW   = 1.5
DATA_ELW   = 2.0  # error-bar line width
DATA_CAPS  = 3.5

N_TOTAL = 500_000_000

C_DATA     = "black"

AXIS_LABEL_STROKE = 1.1
C_CT18_BKG = "#555555"  # grey

PDF_SETS = [
    ("ct18", "CT18 NLO", "#d62728"),
]

PYTHIA_FILES = [
    dict(path=os.path.join(PYTHIA_OUT, "h1_default.root"),
         label="Pythia default", color="#1f77b4"),
    dict(path=os.path.join(PYTHIA_OUT, "h1_ct18nlo_tuning.root"),
         label="CT18NLO + Tuning", color="#ff7f0e"),
    dict(path=os.path.join(PYTHIA_OUT, "h1_ct18nlo.root"),
         label="CT18NLO", color="#d62728"),
    dict(path=os.path.join(PYTHIA_OUT, "h1_ct18nlo_remnant.root"),
         label="CT18NLO + Remnant", color="#9467bd"),
    dict(path=os.path.join(PYTHIA_OUT, "h1_ct18nlo_gluon.root"),
         label="CT18NLO + Gluon", color="#17becf"),
]

CT18_LABEL = "CT18 NLO"

PION_PDF_SETS = [
    ("jam21",   "JAM21PionPDFnlo", "JAM21",   "#0a7cff"),  # deep blue
    ("xfitter", "xFitterPI_NLO_VAR",  "xFitter", "#ff7f0e"),  # orange
    ("grv",     "GRVPI1",             "GRV",     "#9467bd"),  # purple
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
N_XL     = len(XL_CTRS)


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

CHUNK_STEP_SIZE = 200_000

def _peak_rss_mb():
    import resource
    # ru_maxrss is KB on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def process_pythia(path, label, chunk_step_size=CHUNK_STEP_SIZE):
    with uproot.open(path) as f:
        try:
            sigma_dis = float(f["sigmaGen_nb"].members["fVal"])
        except Exception:
            raise RuntimeError(f"sigmaGen_nb not found in {path}")

        tree = None
        for key in f.keys():
            try:
                obj = f[key]
                if hasattr(obj, "keys") and "xL" in obj.keys():
                    tree = obj; break
            except Exception:
                pass
        if tree is None:
            raise RuntimeError(f"No tree with xL found in {path}")

        av    = set(tree.keys())
        q2_b  = pick(av, ["Q2",  "q2"])
        xb_b  = pick(av, ["xB",  "xb"])
        y_b   = pick(av, ["y",   "yBj"])
        w2_b  = pick(av, ["W2",  "w2"])
        eep_b = pick(av, ["EePrime", "Ee_prime", "ePrime"])
        thE_b = pick(av, ["theta_e", "thetaE"])
        thN_b = pick(av, ["theta_n", "thetaN"])
        pt_b  = pick(av, ["pT",  "pt", "PT"])

        want = list(dict.fromkeys(
            ["xL"] + [b for b in [q2_b, xb_b, y_b, w2_b, eep_b, thE_b, thN_b, pt_b] if b]))

        # raw event counts per (iq, ix) cell, per xL bin
        counts = {(iq, ix): np.zeros(N_XL, dtype=np.int64)
                  for iq in range(NQ) for ix in range(NX)}
        n_after_kin = 0
        n_after_pt  = 0

        print(f"  [{label}] streaming {path} ...")
        n_chunks = 0
        for chunk in tree.iterate(want, step_size=chunk_step_size, library="np",
                                   decompression_executor=uproot.source.futures.TrivialExecutor(),
                                   interpretation_executor=uproot.source.futures.TrivialExecutor()):
            n_chunks += 1
            xl = chunk["xL"].astype(np.float32, copy=False)
            pt = chunk[pt_b].astype(np.float32, copy=False) if pt_b else np.zeros(len(xl), dtype=np.float32)

            sel = np.ones(len(xl), dtype=bool)

            def cut(b, fn):
                nonlocal sel
                if b and b in chunk: sel &= fn(chunk[b])

            cut(q2_b,  lambda a: (a >= Q2_MIN) & (a <= Q2_MAX))
            cut(xb_b,  lambda a: (a >= X_MIN)  & (a <= X_MAX))
            cut(y_b,   lambda a: (a >= Y_MIN)  & (a <= Y_MAX))
            cut(w2_b,  lambda a:  a >  W2_MIN)
            cut(eep_b, lambda a:  a >  EePrime_MIN)
            cut(thE_b, lambda a: (a >= THETA_E_MIN) & (a <= THETA_E_MAX))
            cut(thN_b, lambda a:  a <= THETA_N_MAX)

            n_after_kin += int(sel.sum())
            sel &= (pt < PT_MAX)
            n_after_pt  += int(sel.sum())

            if sel.any():
                xl_sel = xl[sel]
                q2_sel = chunk[q2_b][sel] if q2_b else None
                xb_sel = chunk[xb_b][sel] if xb_b else None

                for iq, (q2lo, q2hi) in enumerate(Q2_BINS):
                    q2_mask = ((q2_sel >= q2lo) & (q2_sel < q2hi)
                               if q2_sel is not None else np.ones(len(xl_sel), dtype=bool))
                    if not q2_mask.any():
                        continue
                    for ix, (xlo, xhi) in enumerate(X_BINS):
                        xb_mask = ((xb_sel >= xlo) & (xb_sel < xhi)
                                   if xb_sel is not None else np.ones(len(xl_sel), dtype=bool))
                        m = q2_mask & xb_mask
                        if not m.any():
                            continue
                        xl_cell = xl_sel[m]
                        for i in range(N_XL):
                            counts[(iq, ix)][i] += int(
                                ((xl_cell >= XL_EDGES[i]) & (xl_cell < XL_EDGES[i + 1])).sum())

                del xl_sel, q2_sel, xb_sel

            del chunk, xl, pt, sel

            if n_chunks % 20 == 0:
                print(f"    [{label}] ... {n_chunks} chunks read, "
                      f"{n_after_kin:,} pass kin cuts so far, "
                      f"peak RSS so far: {_peak_rss_mb():,.0f} MB")

        print(f"  [{label}] σ_DIS = {sigma_dis:.4f} nb   |  after kin cuts: {n_after_kin:,}"
              f"  |  + pT<{PT_MAX}: {n_after_pt:,}   |  chunks read: {n_chunks}"
              f"  |  peak RSS: {_peak_rss_mb():,.0f} MB")

    f2ln_py = {}
    nevts   = {}
    for iq, (q2lo, q2hi) in enumerate(Q2_BINS):
        q2_mean = Q2_MEANS[iq]
        dQ2 = q2hi - q2lo
        for ix, (xlo, xhi) in enumerate(X_BINS):
            x_mean = X_MEANS[ix]
            dx = xhi - xlo
            c = counts[(iq, ix)]
            n = int(c.sum())
            nevts[(iq, ix)] = n
            dsig = sigma_dis * c.astype(float) / N_TOTAL / XL_BWS
            f2ln = dsig_to_f2ln(dsig, x_mean, q2_mean, dx, dQ2)
            f2ln_py[(iq, ix)] = f2ln

    return f2ln_py, nevts, sigma_dis


def process_all_pythia(paths):
    """Loads and processes every requested PDF-set ROOT file."""
    results = {}
    for key, label, _ in PDF_SETS:
        path = paths.get(key)
        if not path:
            continue
        print(f"\nLoading Pythia ({label}): {path}")
        f2ln, nevts, sigma_dis = process_pythia(path, label)
        gc.collect()
        results[key] = dict(f2ln=f2ln, nevts=nevts, sigma_dis=sigma_dis)
    return results
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
def _pion_f2ln_pointwise(pdf_obj, xL, xbj, q2):
    x_pi = xbj / (1.0 - xL)
    if x_pi <= 0.0 or x_pi >= 1.0:
        return 0.0
    return 2.0 * f_piN_model_i(float(xL)) * compute_f2pi_minus_allflav_nlo(float(x_pi), pdf_obj, q2)


try:
    import f2pi_nlo
    _HAVE_GPU_PATH = True
except Exception as _e:
    print(f"  NOTE: f2pi_nlo unavailable ({_e}); using the slow CPU path.")
    _HAVE_GPU_PATH = False

# Set to False to force the original scipy.quad implementation
USE_FAST_F2PI = True


def rebin_pion_to_xl_edges(pdf_obj, n_sub=12, label=""):
    if USE_FAST_F2PI and _HAVE_GPU_PATH:
        print(f"    [{label}] backend: {f2pi_nlo.backend_name()}", flush=True)
        return f2pi_nlo.rebin_pion_to_xl_edges_fast(
            pdf_obj, X_MEANS, Q2_MEANS, XL_EDGES, N_XL,
            f_piN_model_i, n_sub=n_sub, label=label)
    return rebin_pion_to_xl_edges_cpu(pdf_obj, n_sub=n_sub, label=label)


def rebin_pion_to_xl_edges_cpu(pdf_obj, n_sub=12, label=""):
    """Original per-point scipy.quad implementation."""
    import time
    jam_hist = {}
    n_cells  = len(X_MEANS) * len(Q2_MEANS)
    done     = 0
    t0       = time.time()
    for ix, xbj in enumerate(X_MEANS):
        for iq, q2 in enumerate(Q2_MEANS):
            vals = np.zeros(N_XL)
            for i in range(N_XL):
                lo, hi = XL_EDGES[i], XL_EDGES[i+1]
                sub    = lo + (np.arange(n_sub)+0.5)*(hi-lo)/n_sub
                yv     = np.array([_pion_f2ln_pointwise(pdf_obj, float(xL), xbj, q2) for xL in sub])
                vals[i] = yv.mean()
            jam_hist[(iq, ix)] = vals
            done += 1
            el   = time.time() - t0
            eta  = el / done * (n_cells - done)
            print(f"    [{label}] cell {done}/{n_cells} "
                  f"(x_B={xbj:.2e}, Q2={q2:g})  "
                  f"elapsed {el:6.1f}s  ETA {eta:6.1f}s", flush=True)
    return jam_hist


def load_pion_pdf_hists(pion_pdf_sets=PION_PDF_SETS, skip_keys=()):
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


def _largest_empty_block_at(cell_has_data, corner="top-left"):
    """Largest fully-empty rectangle of (row, iq) cells anchored at one corner of the grid."""
    def _all_empty(cells):
        return all(not cell_has_data.get(c, False) for c in cells)

    best = None  # (area, block)

    if corner == "top-left":
        for r1 in range(NX):
            for q1 in range(NQ):
                block = [(r, q) for r in range(0, r1 + 1) for q in range(0, q1 + 1)]
                if block and _all_empty(block):
                    if best is None or len(block) > best[0]:
                        best = (len(block), block)
    else:  # bottom-right
        for r0 in range(NX):
            for q0 in range(NQ):
                block = [(r, q) for r in range(r0, NX) for q in range(q0, NQ)]
                if block and _all_empty(block):
                    if best is None or len(block) > best[0]:
                        best = (len(block), block)

    return best[1] if best is not None else None


def _largest_corner_empty_block(cell_has_data):
    blocks = [b for b in (_largest_empty_block_at(cell_has_data, "bottom-right"),
                          _largest_empty_block_at(cell_has_data, "top-left"))
              if b]
    if not blocks:
        return None
    return max(blocks, key=len)


def _block_centre(axes, block):
    """Figure-fraction centre of a rectangular block of (row, col) cells."""
    positions = [axes[r][q].get_position() for r, q in block]
    x0 = min(p.x0 for p in positions)
    x1 = max(p.x1 for p in positions)
    y0 = min(p.y0 for p in positions)
    y1 = max(p.y1 for p in positions)
    return 0.5 * (x0 + x1), 0.5 * (y0 + y1)


def _place_f2_label_top_left(fig, axes, cell_has_data, text, fontsize,
                             stroke=None):
    block = _largest_empty_block_at(cell_has_data, "top-left")
    if not block:
        return False

    cx, cy = _block_centre(axes, block)

    # Place "H1" label above F_2LN label
    t_h1 = fig.text(cx, cy + 0.08, "H1", ha="center", va="center",
                    fontsize=fontsize, fontweight="bold")
    if stroke:
        t_h1.set_path_effects(stroke)

    # Place F_2LN label below H1
    t = fig.text(cx, cy, text, ha="center", va="center",
                 fontsize=fontsize, fontweight="bold")
    if stroke:
        t.set_path_effects(stroke)
    return True


def _place_legend_in_empty_space(fig, axes, cell_has_data, source_ax):
    handles, labels = source_ax.get_legend_handles_labels()
    if not handles:
        return

    block = _largest_corner_empty_block(cell_has_data)
    if not block:
        return

    positions = [axes[r][q].get_position() for r, q in block]
    bx0 = min(p.x0 for p in positions)
    bx1 = max(p.x1 for p in positions)
    by0 = min(p.y0 for p in positions)
    by1 = max(p.y1 for p in positions)
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


def _xl_tick_fmt(v, pos):
    if abs(v - round(v)) < 1e-9:
        return rf"$\mathbf{{{int(round(v))}}}$"
    return rf"$\mathbf{{{v:.2f}}}$"


def _q2_label_bold(lo, hi):
    # No "GeV^2"
    return rf"$Q^2 \in [{lo:.4g},\,{hi:.4g}]$"


def _xb_label_bold(lo, hi):
    return rf"$x_B \in [{sci(lo)},$" + "\n" + rf"${sci(hi)}]$"


def _draw_cell_three(ax, ct18_f2ln, pion_hists, cell, h1_cell, mode,
                     show_legend=False, legend_fontsize=6):
    py = ct18_f2ln.get(cell)

    if mode in ("pions", "both"):
        # CT18 NLO Pythia DIS background
        if py is not None:
            ax.stairs(py, XL_EDGES, color=C_CT18_BKG, lw=LW_BKG, ls='--',
                      label=f"Pythia ({CT18_LABEL})")
        # three raw pion-PDF histograms
        for key, _pdfname, label, color in PION_PDF_SETS:
            h = pion_hists.get(key, {}).get(cell)
            if h is not None:
                lw  = LW_PION * 0.72 if mode == "both" else LW_PION
                ls  = ':' if mode == "both" else '-'
                lbl = f"{label} (raw)" if mode == "both" else label
                ax.stairs(h, XL_EDGES, color=color, lw=lw, ls=ls, label=lbl)

    if mode in ("sums", "both"):
        # three plain sums  CT18 NLO + pion  (no normalisation)
        for key, _pdfname, label, color in PION_PDF_SETS:
            h = pion_hists.get(key, {}).get(cell)
            if py is not None and h is not None:
                ax.stairs(py + h, XL_EDGES, color=color, lw=LW_SUM,
                          label=f"Pythia ({CT18_LABEL}) + {label}")

    # H1 data (always)
    if h1_cell is not None:
        xL_h1, F2_h1, stat_h1, tot_h1 = h1_cell
        ax.errorbar(xL_h1, F2_h1, yerr=tot_h1,
                    fmt="o", ms=DATA_MS, mfc="black", mec="black",
                    mew=DATA_MEW, ecolor="black", elinewidth=DATA_ELW,
                    capsize=DATA_CAPS, capthick=DATA_ELW,
                    color="black", zorder=10, label="H1 data")

    ax.set_xlim(XL_EDGES[0] - 0.02, 1.0)
    ax.set_ylim(0., 0.14)
    ax.set_facecolor("white")

    # Bold panel boundary
    for sp in ax.spines.values():
        sp.set_linewidth(SPINE_LW)

    ax.set_xticks([0.50, 0.75, 1.00])
    ax.xaxis.set_major_formatter(FuncFormatter(_xl_tick_fmt))
    ax.set_yticks([0.03, 0.06, 0.09, 0.12])
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.tick_params(which="both", direction="in", top=True, right=True,
                   labelsize=FS_TICK)
    ax.tick_params(which="major", length=10, width=2.0, pad=6)
    ax.tick_params(which="minor", length=5, width=1.4)
    plt.setp(ax.get_xticklabels(), fontweight="bold")
    plt.setp(ax.get_yticklabels(), fontweight="bold")

    if show_legend:
        ax.legend(loc="upper right", fontsize=legend_fontsize,
                  handlelength=1.2, handletextpad=0.3,
                  borderpad=0.3, framealpha=0.85, labelspacing=0.25)


def _make_grid_three_generic(ct18_f2ln, pion_hists, h1_data, out, suffix, mode):
    fig, axes = plt.subplots(NX, NQ,
                             figsize=(NQ * 4.8, NX * 4.0),
                             gridspec_kw=dict(wspace=0.0, hspace=0.0),
                             sharex=False, sharey=False)
    fig.patch.set_facecolor("white")

    cell_has_data = {}
    for ix in range(NX):
        row = NX - 1 - ix
        for iq in range(NQ):
            cell_has_data[(row, iq)] = h1_data.get((iq, ix)) is not None

    # first data-bearing panel (reading order)
    legend_cell = None
    for row in range(NX):
        for iq in range(NQ):
            if cell_has_data[(row, iq)]:
                legend_cell = (row, iq); break
        if legend_cell is not None:
            break

    last_visible_row  = {}
    first_visible_row = {}
    for iq in range(NQ):
        rows_with_data = [row for row in range(NX) if cell_has_data[(row, iq)]]
        if rows_with_data:
            first_visible_row[iq] = min(rows_with_data)
            last_visible_row[iq]  = max(rows_with_data)

    first_visible_col = {}
    for row in range(NX):
        cols_with_data = [iq for iq in range(NQ) if cell_has_data[(row, iq)]]
        if cols_with_data:
            first_visible_col[row] = min(cols_with_data)

    legend_source_ax = None
    for ix in range(NX):
        row = NX - 1 - ix
        for iq in range(NQ):
            ax   = axes[row, iq]
            cell = (iq, ix)

            if not cell_has_data[(row, iq)]:
                ax.set_visible(False)
                continue

            ax.set_facecolor("white")
            _draw_cell_three(ax, ct18_f2ln, pion_hists, cell,
                             h1_data.get(cell), mode)

            if (row, iq) == legend_cell:
                legend_source_ax = ax

            if row == first_visible_row.get(iq):
                q2lo, q2hi = Q2_BINS[iq]
                ax.set_title(_q2_label_bold(q2lo, q2hi), fontsize=FS_TITLE,
                            pad=9, fontweight="bold")

            if iq == first_visible_col.get(row):
                xlo, xhi = X_BINS[ix]
                ax.text(-0.26, 0.5, _xb_label_bold(xlo, xhi),
                        transform=ax.transAxes, ha="right", va="center",
                        rotation=0, linespacing=1.4,
                        fontsize=FS_ROWLAB, fontweight="bold")
                ax.tick_params(labelleft=True)
            else:
                ax.tick_params(labelleft=False)

            ax.tick_params(labelbottom=(row == last_visible_row.get(iq)))

    if legend_source_ax is not None:
        _place_legend_in_empty_space(fig, axes, cell_has_data, legend_source_ax)

    from matplotlib.transforms import Bbox
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
        # fallback: use bare axes edges if no renderer is available
        ux0 = min(p.x0 for p in pos) - 0.03
        uy0 = min(p.y0 for p in pos) - 0.03

    pad = 0.010
    _lbl_stroke = [patheffects.withStroke(linewidth=AXIS_LABEL_STROKE,
                                          foreground="black")]

    t_x = fig.text(cx, uy0 - pad, r"$x_L$",
                   ha="center", va="top", fontsize=FS_AXLABEL,
                   fontweight="bold")
    t_x.set_path_effects(_lbl_stroke)

    _f2_text = r"$F_{2}^{LN}(x_B,\,Q^{2},\,x_L)$"
    if not _place_f2_label_top_left(fig, axes, cell_has_data, _f2_text,
                                    FS_AXLABEL, stroke=_lbl_stroke):
        t_y = fig.text(ux0 - pad, cy, _f2_text,
                       ha="center", va="center", rotation=90,
                       fontsize=FS_AXLABEL, fontweight="bold")
        t_y.set_path_effects(_lbl_stroke)

    for ext in ("png", "pdf"):
        fname = f"{out}_f2ln_grid7x7{suffix}.{ext}"
        fig.savefig(fname, dpi=180, bbox_inches="tight", facecolor="white")
        print(f"  → {fname}")
    plt.close(fig)


def make_grid_three_pions(ct18_f2ln, pion_hists, h1_data, out):
    """Grid 1: CT18 NLO background + three raw pion-PDF histograms vs H1."""
    _make_grid_three_generic(ct18_f2ln, pion_hists, h1_data, out,
                             suffix="_three_pion_pdfs", mode="pions")


def make_grid_ct18_plus_pions(ct18_f2ln, pion_hists, h1_data, out):
    """Grid 2: three summed curves  CT18 NLO + pion  (added) vs H1."""
    _make_grid_three_generic(ct18_f2ln, pion_hists, h1_data, out,
                             suffix="_ct18_plus_three_pions", mode="sums")


def make_individual(ct18_f2ln, pion_hists, h1_data, out):
    ind_dir = f"{out}_f2ln_individual"
    os.makedirs(ind_dir, exist_ok=True)

    for ix, (xlo, xhi) in enumerate(X_BINS):
        for iq, (q2lo, q2hi) in enumerate(Q2_BINS):
            fig, ax = plt.subplots(figsize=(5, 4))
            cell = (iq, ix)

            _draw_cell_three(ax, ct18_f2ln, pion_hists, cell,
                             h1_data.get(cell), mode="both",
                             show_legend=True, legend_fontsize=6)

            ax.set_xlabel(r"$x_L$", fontsize=12, fontweight="bold")
            ax.set_ylabel(r"$F_{2LN}$", fontsize=12, fontweight="bold")
            ax.set_title(q2_label(q2lo, q2hi) + "   " + xb_label(xlo, xhi), fontsize=9)

            fig.tight_layout()
            base = f"{ind_dir}/f2ln_q{iq}_x{ix}"
            for ext in ("png", "pdf"):
                fig.savefig(f"{base}.{ext}", dpi=150, bbox_inches="tight")
            plt.close(fig)

    print(f"  → {ind_dir}/  ({NQ*NX} plots)")


def compute_chi2_ct18_plus_pions(ct18_f2ln, pion_hists, h1_data):
    """For each pion PDF (JAM21 / xFitter / GRV), compute the overall chi^2 of H1 data vs."""
    results = {}
    for key, _pdfname, label, _color in PION_PDF_SETS:
        h_dict = pion_hists.get(key)
        if h_dict is None:
            continue

        curve_label = f"Pythia ({CT18_LABEL}) + {label}"
        chi2 = 0.0
        ndof = 0
        for cell, h1_cell in h1_data.items():
            if h1_cell is None:
                continue
            xL_h1, F2_h1, _stat_h1, tot_h1 = h1_cell

            py = ct18_f2ln.get(cell)
            h  = h_dict.get(cell)
            if py is None or h is None:
                continue

            combined = py + h

            # which x_L bin (stairs step) each data point falls into
            bin_idx = np.clip(np.digitize(xL_h1, XL_EDGES) - 1, 0, N_XL - 1)
            model_vals = combined[bin_idx]

            valid = np.isfinite(model_vals) & np.isfinite(F2_h1) & (tot_h1 > 0)
            if not valid.any():
                continue
            chi2 += float(np.sum(((F2_h1[valid] - model_vals[valid]) / tot_h1[valid]) ** 2))
            ndof += int(valid.sum())

        results[curve_label] = dict(
            chi2=chi2,
            ndof=ndof,
            chi2_per_dof=(chi2 / ndof if ndof > 0 else float("nan")),
        )
    return results


def save_chi2_table(results, out_path, title=""):
    """Write an overall chi^2 summary (one row per PDF-set curve) to a plain-text file."""
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
    print(f"  → {out_path}")


def run_one_proton_pdf(root_path, label, out, pion_hists, h1_data,
                       make_individual_plots=True):
    global CT18_LABEL
    previous_label = CT18_LABEL
    CT18_LABEL = label  # every legend / chi^2 row picks this up

    try:
        print(f"\n{'=' * 70}")
        print(f"Proton PDF: {label}")
        print(f"  ROOT file : {root_path}")
        print(f"  out prefix: {out}")
        print(f"{'=' * 70}")

        if not root_path or not os.path.exists(root_path):
            print(f"  ⚠ ROOT file not found — skipping {label}")
            return None

        f2ln, _nevts, _sigma = process_pythia(root_path, label)

        print(f"\nSaving three-pion-PDF 7×7 grid ({label}) ...")
        make_grid_three_pions(f2ln, pion_hists, h1_data, out)

        print(f"\nSaving {label} + three-pion sum 7×7 grid ...")
        make_grid_ct18_plus_pions(f2ln, pion_hists, h1_data, out)

        print(f"\nComputing overall chi^2 for {label} + each pion PDF ...")
        chi2_tab = compute_chi2_ct18_plus_pions(f2ln, pion_hists, h1_data)
        save_chi2_table(
            chi2_tab,
            f"{out}_chi2_{_slug(label)}_plus_three_pions.txt",
            title=f"Overall chi^2: H1 data vs. {label} + pion PDF curves",
        )

        if make_individual_plots:
            print(f"\nSaving individual per-cell plots ({label}) ...")
            make_individual(f2ln, pion_hists, h1_data, out)

        return f2ln

    finally:
        CT18_LABEL = previous_label


def _slug(label):
    """'CT18 NLO + Gluon' -> 'ct18_nlo_gluon'  (safe for filenames)."""
    keep = [c.lower() if c.isalnum() else "_" for c in label]
    out = "".join(keep)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ct18",
        default=os.path.join(PYTHIA_OUT, "h1_ct18nlo.root"),
        help="Pythia ROOT file (CT18 NLO)")
    parser.add_argument(
        "--ct18gluon",
        default=os.path.join(PYTHIA_OUT, "h1_ct18nlo_gluon.root"),
        help="Pythia ROOT file (CT18 NLO + Gluon)")
    parser.add_argument("--hdat", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "h1.dat"))
    parser.add_argument("--out", default="ct18_three_pion",
                        help="Output filename prefix for the CT18 NLO pass")
    parser.add_argument("--out-gluon", dest="out_gluon",
                        default="ct18gluon_three_pion",
                        help="Output filename prefix for the CT18 NLO + Gluon pass")
    parser.add_argument("--skip-individual", action="store_true",
                        help="Skip the 49 per-cell plots for each proton PDF")
    parser.add_argument("--only", choices=["ct18", "gluon", "both"],
                        default="both",
                        help="Run just one proton PDF instead of both")
    args = parser.parse_args()

    # H1 data
    print(f"Loading H1 data from {args.hdat} ...")
    h1_data = load_hdat_f2ln(args.hdat)
    print(f"  Loaded {len(h1_data)} cells")

    print("\nComputing three pion-PDF F_2LN histograms (JAM21 / xFitter / GRV) ...")
    pion_hists = load_pion_pdf_hists(PION_PDF_SETS)
    if not pion_hists:
        raise SystemExit("No pion PDFs could be loaded — nothing to plot.")
    print(f"  Loaded pion PDFs: {', '.join(pion_hists.keys())}")

    # one full output set per proton PDF, under its own prefix
    passes = []
    if args.only in ("ct18", "both"):
        passes.append((args.ct18,      "CT18 NLO",          args.out))
    if args.only in ("gluon", "both"):
        passes.append((args.ct18gluon, "CT18 NLO + Gluon",  args.out_gluon))

    produced = {}
    for root_path, label, out_prefix in passes:
        f2ln = run_one_proton_pdf(
            root_path, label, out_prefix, pion_hists, h1_data,
            make_individual_plots=not args.skip_individual)
        if f2ln is not None:
            produced[label] = out_prefix

    print(f"\n{'=' * 70}")
    if produced:
        print("Wrote output sets for:")
        for label, out_prefix in produced.items():
            print(f"  {label:22s} -> {out_prefix}_*")
    else:
        print("No proton PDF produced output — check the ROOT file paths.")
    print(f"{'=' * 70}")

    print("\nDone.")


if __name__ == "__main__":
    main()
