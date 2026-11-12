"""H1: the five Pythia settings with the JAM21 Sullivan term (F2^LN grids, chi^2) and the
overview of three pion PDFs times five Pythia settings.
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

# FONT SIZES / WEIGHTS
FS_AXLABEL = 42  # F_2^LN(...) and x_L axis labels
FS_TITLE   = 28  # Q^2 column titles
FS_ROWLAB  = 26  # x_B row labels
FS_TICK    = 24  # x and y tick numbers
FS_LEGEND  = 24  # the one shared corner legend

SPINE_LW   = 3.2  # panel box
LW_PYTHIA  = 2.6  # raw Pythia (DIS background) histograms
LW_JAM     = 3.2  # raw JAM21 pion histogram
LW_SUM     = 3.2  # Pythia + JAM21 sums
DATA_MS    = 7.0  # H1 marker size
DATA_MEW   = 1.5
DATA_ELW   = 2.0  # error-bar line width
DATA_CAPS  = 3.5

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

N_TOTAL = 500_000_000

C_DEFAULT = "#1f77b4"  # blue
C_CT18    = "#d62728"  # red
C_CT18_REMNANT = "#9467bd"  # purple
C_CT18_PTUNE = "#ff7f0e"  # orange
C_CT18_GLUON = "#17becf"  # cyan
C_DATA    = "black"

AXIS_LABEL_STROKE = 1.1
C_JAM     = "navy"  # JAM21 raw histogram (no normalisation)

# Ordered list of (key, display label, color) used throughout
PDF_SETS = [
    ("default",     "Pythia default",           C_DEFAULT),
    ("ct18",        "CT18 NLO",                  C_CT18),
    ("ct18remnant", "CT18 NLO + Remnant",        C_CT18_REMNANT),
    ("ct18ptune",   "CT18 NLO + Tuning",         C_CT18_PTUNE),
    ("ct18gluon",   "CT18 NLO + Gluon",          C_CT18_GLUON),
]

PION_PDF_SETS = [
    ("jam21", "JAM21PionPDFnlo_pT", "JAM21", "navy"),
]

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


print("Loading JAM21PionPDFnlo (member 0) ...")
_pdf_jam = lhapdf.mkPDF("JAM21PionPDFnlo", 0)
print(f"  x=[{_pdf_jam.xMin:.2e}, {_pdf_jam.xMax:.2e}]  "
      f"Q2=[{_pdf_jam.q2Min:.2e}, {_pdf_jam.q2Max:.2e}]")

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

JAM_XL_GRID = np.linspace(0.05, 0.98, 60)

def load_jam21_f2ln():
    jam = {}
    for ix, xbj in enumerate(X_MEANS):
        for iq, q2 in enumerate(Q2_MEANS):
            xl_vals, f2ln_vals = [], []
            for xL in JAM_XL_GRID:
                x_pi = xbj / (1.0 - xL)
                if x_pi <= 0.0 or x_pi >= 1.0: continue
                flux  = 2.0 * f_piN_model_i(float(xL))
                f2pi  = compute_f2pi_minus_allflav_nlo(float(x_pi), _pdf_jam, q2)
                xl_vals.append(xL)
                f2ln_vals.append(flux * f2pi)
            jam[(iq, ix)] = (np.asarray(xl_vals), np.asarray(f2ln_vals))
            print(f"  JAM21 NLO: xBj={xbj:.3e}, Q2={q2:g} ({len(xl_vals)} pts)")
    return jam

def _jam_f2ln_pointwise(xL, xbj, q2):
    x_pi = xbj / (1.0 - xL)
    if x_pi <= 0.0 or x_pi >= 1.0: return 0.0
    return 2.0 * f_piN_model_i(float(xL)) * compute_f2pi_minus_allflav_nlo(float(x_pi), _pdf_jam, q2)

def rebin_jam21_to_xl_edges(jam_data, n_sub=12):
    """Average JAM21 over each XL bin to get a histogram-compatible array."""
    if USE_FAST_F2PI and _HAVE_GPU_PATH:
        return rebin_pion_to_xl_edges(_pdf_jam, n_sub=n_sub, label="JAM21")

    jam_hist = {}
    for (iq, ix) in jam_data.keys():
        xbj = X_MEANS[ix];  q2 = Q2_MEANS[iq]
        vals = np.zeros(N_XL)
        for i in range(N_XL):
            lo, hi = XL_EDGES[i], XL_EDGES[i+1]
            sub    = lo + (np.arange(n_sub)+0.5)*(hi-lo)/n_sub
            yv     = np.array([_jam_f2ln_pointwise(float(xL), xbj, q2) for xL in sub])
            vals[i] = yv.mean()
        jam_hist[(iq, ix)] = vals
    return jam_hist


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


def rebin_pion_to_xl_edges(pdf_obj, n_sub=12, label="pion"):
    if USE_FAST_F2PI and _HAVE_GPU_PATH:
        print(f"    [{label}] backend: {f2pi_nlo.backend_name()}", flush=True)
        return f2pi_nlo.rebin_pion_to_xl_edges_fast(
            pdf_obj, X_MEANS, Q2_MEANS, XL_EDGES, N_XL,
            f_piN_model_i, n_sub=n_sub, label=label)
    return rebin_pion_to_xl_edges_cpu(pdf_obj, n_sub=n_sub)


def rebin_pion_to_xl_edges_cpu(pdf_obj, n_sub=12):
    """Original per-point scipy.quad implementation."""
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


def compute_sum_model(f2ln_py, jam_hist):
    """Returns sum_model[(iq,ix)] = py + jam_hist, for cells with both."""
    sum_model = {}
    for cell, py in f2ln_py.items():
        jh = jam_hist.get(cell)
        if jh is None:
            continue
        sum_model[cell] = py + jh
    return sum_model


# chi^2/n  for sum model vs H1 data  (for annotation only)
def compute_chi2(sum_model, h1_data):
    """Returns chi2[(iq,ix)] = chi2/n."""
    chi2_table = {}
    for cell, model in sum_model.items():
        h1 = h1_data.get(cell)
        if h1 is None:
            continue
        xL_h1, F2_h1, _, tot_h1 = h1
        idx = np.searchsorted(XL_EDGES[1:], xL_h1, side='left')
        idx = np.clip(idx, 0, N_XL - 1)
        model_at = model[idx]
        denom    = np.where(tot_h1 > 0, tot_h1, 1e-30)
        c2n = np.sum(((F2_h1 - model_at) / denom)**2) / max(len(F2_h1), 1)
        chi2_table[cell] = c2n
    return chi2_table


def finalize_pdf_set(results, jam_hist, h1_data):
    for key, label, _ in PDF_SETS:
        if key not in results:
            continue
        r = results[key]
        sum_model = compute_sum_model(r["f2ln"], jam_hist)
        chi2_table = compute_chi2(sum_model, h1_data)
        vals = list(chi2_table.values())
        mean = np.mean(vals) if vals else float('nan')
        print(f"  [{label}]  Pythia+JAM21 sum (no normalisation)   "
              f"mean χ²/n = {mean:.4f}  ({len(vals)} cells)")
        r["sum_model"] = sum_model
        r["chi2"] = chi2_table


def compute_overall_chi2(results, h1_data):
    out = {}
    for key, label, _ in PDF_SETS:
        if key not in results:
            continue
        r = results[key]
        sum_model = r.get("sum_model", {})

        chi2 = 0.0
        ndof = 0
        for cell, model in sum_model.items():
            h1 = h1_data.get(cell)
            if h1 is None:
                continue
            xL_h1, F2_h1, _stat_h1, tot_h1 = h1

            idx = np.clip(np.searchsorted(XL_EDGES[1:], xL_h1, side='left'),
                          0, N_XL - 1)
            model_at = model[idx]

            valid = np.isfinite(model_at) & np.isfinite(F2_h1) & (tot_h1 > 0)
            if not valid.any():
                continue
            chi2 += float(np.sum(((F2_h1[valid] - model_at[valid]) / tot_h1[valid]) ** 2))
            ndof += int(valid.sum())

        out[f"{label} + JAM21"] = dict(
            chi2=chi2,
            ndof=ndof,
            chi2_per_dof=(chi2 / ndof if ndof > 0 else float("nan")),
        )
    return out


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


def _draw_cell(ax, results, cell, h1_cell, jam_hist_cell,
               compact=True, sum_only=False):
    """Draw one (Q^2, x_B) cell onto ax."""
    fs = 5 if compact else 7

    for key, label, color in PDF_SETS:
        if key not in results:
            continue
        r    = results[key]
        py   = r["f2ln"].get(cell)
        n    = r["nevts"].get(cell, 0)
        sig  = r["sigma_dis"]
        sum_cell  = r.get("sum_model", {}).get(cell)
        chi2_cell = r.get("chi2", {}).get(cell)

        # raw Pythia step-histogram
        if not sum_only and py is not None:
            lbl = (rf"{label} $N={n:,}$" if compact
                   else rf"{label}  $N={n:,}$,  $\sigma={sig:.3f}\ \mathrm{{nb}}$")
            ax.stairs(py, XL_EDGES, color=color, lw=1.3, label=lbl)

        if sum_cell is not None:
            c2n = chi2_cell if chi2_cell is not None else float('nan')
            lbl = (rf"{label}+JAM21 $\chi^2/n={c2n:.2f}$" if compact
                   else rf"{label}+JAM21 (no norm.)  $\chi^2/n={c2n:.2f}$")
            ax.stairs(sum_cell, XL_EDGES, color=color, lw=2.0, ls='--',
                      label=lbl, alpha=0.9)

    # JAM21 histogram (raw, no normalisation)
    if not sum_only and jam_hist_cell is not None:
        ax.stairs(jam_hist_cell, XL_EDGES, color=C_JAM, lw=1.3, ls='-',
                  label="JAM21 (hist, no norm.)")

    # H1 data
    if h1_cell is not None:
        xL_h1, F2_h1, stat_h1, tot_h1 = h1_cell
        ax.errorbar(xL_h1, F2_h1, xerr=XL_XERR[:len(xL_h1)], yerr=tot_h1,
                    fmt="none", color=C_DATA, lw=0.7, capsize=0, zorder=9)
        ax.errorbar(xL_h1, F2_h1, xerr=XL_XERR[:len(xL_h1)], yerr=stat_h1,
                    fmt="o", color=C_DATA, ms=2.5, lw=0.9, capsize=1.5, zorder=10,
                    markerfacecolor="white", label="H1 data")

    ax.set_xlim(XL_EDGES[0] - 0.01, XL_EDGES[-1] + 0.01)
    ax.set_ylim(0., 0.14)
    ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=6)
    ax.minorticks_on()
    ax.legend(loc="upper right", fontsize=fs,
              handlelength=1.2, handletextpad=0.3,
              borderpad=0.3, framealpha=0.85, labelspacing=0.2)


def _make_grid_generic(results, h1_data, jam_hist, out, suffix, title, sum_only):
    fig, axes = plt.subplots(NX, NQ,
                             figsize=(NQ * 3.0, NX * 2.6),
                             sharex=False, sharey=False)
    fig.suptitle(title, fontsize=9, y=1.002)

    for iq, (q2lo, q2hi) in enumerate(Q2_BINS):
        axes[0, iq].set_title(q2_label(q2lo, q2hi), fontsize=7, pad=4)

    for ix, (xlo, xhi) in enumerate(X_BINS):
        row = NX - 1 - ix
        axes[row, 0].set_ylabel(
            xb_label(xlo, xhi) + "\n" + r"$F_{2LN}$", fontsize=6.5)

    for ix in range(NX):
        row = NX - 1 - ix
        for iq in range(NQ):
            ax   = axes[row, iq]
            cell = (iq, ix)

            _draw_cell(ax, results, cell,
                       h1_data.get(cell), jam_hist.get(cell),
                       compact=True, sum_only=sum_only)

            if row == NX - 1:
                ax.set_xlabel(r"$x_L$", fontsize=7)
            else:
                ax.tick_params(labelbottom=False)
            if iq != 0:
                ax.tick_params(labelleft=False)

    fig.tight_layout(rect=[0, 0, 1, 1])
    for ext in ("png", "pdf"):
        fname = f"{out}_f2ln_grid7x7{suffix}.{ext}"
        fig.savefig(fname, dpi=180, bbox_inches="tight")
        print(f"  → {fname}")
    plt.close(fig)


def make_grid(results, h1_data, jam_hist, out):
    """Main 7x7 grid: raw Pythia (xN sets) + JAM21 (raw) + sum (xN sets) + H1."""
    title = (rf"$F_{{2LN}}(x_L)$ — Pythia (default/CT18 NLO/"
              rf"CT18 NLO+Remnant/CT18NLO+pythia Tune/CT18NLO+gluon)"
              rf" + JAM21 (hist) vs H1 data"
              rf"$\quad p_T<{PT_MAX}\ \mathrm{{GeV}}$")
    _make_grid_generic(results, h1_data, jam_hist, out,
                       suffix="", title=title, sum_only=False)


def make_grid_sum_only(results, h1_data, jam_hist, out):
    title = (rf"$F_{{2LN}}(x_L)$ — raw sum $\mathrm{{Py}}_k+\mathrm{{JAM21}}$ (no normalisation)"
              rf" (default/CT18 NLO/CT18 NLO+Remnant/CT18NLO+pythia Tune/CT18NLO+gluon) vs H1 data"
              rf"$\quad p_T<{PT_MAX}\ \mathrm{{GeV}}$")
    _make_grid_generic(results, h1_data, jam_hist, out,
                       suffix="_sumonly", title=title, sum_only=True)


def _draw_cell_bg(ax, results, cell, h1_cell, jam_hist_cell,
                   include_pythia, include_jam, include_sum, compact=True):
    """Draw one (Q^2, x_B) cell."""
    # raw Pythia step-histograms (DIS background)
    if include_pythia:
        for key, label, color in PDF_SETS:
            if key not in results:
                continue
            py = results[key]["f2ln"].get(cell)
            if py is not None:
                ax.stairs(py, XL_EDGES, color=color, lw=LW_PYTHIA,
                          label=_pythia_label(label))

    # raw JAM21 pion histogram (shared, no normalisation)
    if include_jam and jam_hist_cell is not None:
        ax.stairs(jam_hist_cell, XL_EDGES, color=C_JAM, lw=LW_JAM, ls='-',
                  label="JAM21")

    # summed model  Pythia_k + JAM21  (the "added" curves)
    if include_sum:
        for key, label, color in PDF_SETS:
            if key not in results:
                continue
            sum_cell = results[key].get("sum_model", {}).get(cell)
            if sum_cell is not None:
                ax.stairs(sum_cell, XL_EDGES, color=color, lw=LW_SUM, ls='-',
                          label=f"{_pythia_label(label)} + JAM21")

    # H1 data
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


def _largest_corner_empty_block(cell_has_data):
    def _all_empty(cells):
        return all(not cell_has_data.get(c, False) for c in cells)

    blocks = [b for b in (_largest_empty_block_at(cell_has_data, "bottom-right"),
                          _largest_empty_block_at(cell_has_data, "top-left"))
              if b]
    if not blocks:
        return None
    return max(blocks, key=len)


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
        # shrink by whichever dimension overflows more
        fs *= max(0.55, min(0.92, min(max_w / bb.width, max_h / bb.height)))


def _pythia_label(label):
    return label if label.strip().lower().startswith("pythia") else f"Pythia ({label})"


def _xl_tick_fmt(v, pos):
    if abs(v - round(v)) < 1e-9:
        return rf"$\mathbf{{{int(round(v))}}}$"
    return rf"$\mathbf{{{v:.2f}}}$"


def _q2_label_bold(lo, hi):
    # No "GeV^2"
    return rf"$Q^2 \in [{lo:.4g},\,{hi:.4g}]$"


def _xb_label_bold(lo, hi):
    return rf"$x_B \in [{sci(lo)},$" + "\n" + rf"${sci(hi)}]$"


def _make_grid_bg_generic(results, h1_data, jam_hist, out, suffix,
                          include_pythia, include_jam, include_sum=False):
    fig, axes = plt.subplots(NX, NQ,
                             figsize=(NQ * 4.8, NX * 4.0),
                             gridspec_kw=dict(wspace=0.0, hspace=0.0),
                             sharex=False, sharey=False)
    fig.patch.set_facecolor("white")
    # No fig.suptitle(...)

    cell_has_data = {}
    for ix in range(NX):
        row = NX - 1 - ix
        for iq in range(NQ):
            cell_has_data[(row, iq)] = h1_data.get((iq, ix)) is not None

    legend_cell = None
    for row in range(NX):
        for iq in range(NQ):
            if cell_has_data[(row, iq)]:
                legend_cell = (row, iq)
                break
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

            # drop empty panels entirely
            if not cell_has_data[(row, iq)]:
                ax.set_visible(False)
                continue

            ax.set_facecolor("white")

            _draw_cell_bg(ax, results, cell,
                         h1_data.get(cell), jam_hist.get(cell),
                         include_pythia=include_pythia,
                         include_jam=include_jam,
                         include_sum=include_sum,
                         compact=True)

            if (row, iq) == legend_cell:
                legend_source_ax = ax

            # Q^2 title on the top-most visible panel of this column
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

    # one big, bold legend in the largest empty corner of the grid
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


def make_grid_bg_only(results, h1_data, jam_hist, out):
    _make_grid_bg_generic(results, h1_data, jam_hist, out,
                          suffix="_dis_background_only",
                          include_pythia=True, include_jam=False,
                          include_sum=False)


def make_grid_bg_jam(results, h1_data, jam_hist, out):
    _make_grid_bg_generic(results, h1_data, jam_hist, out,
                          suffix="_dis_background_jam21",
                          include_pythia=False, include_jam=False,
                          include_sum=True)


def make_individual(results, h1_data, jam_hist, out):
    ind_dir = f"{out}_f2ln_individual"
    os.makedirs(ind_dir, exist_ok=True)

    for ix, (xlo, xhi) in enumerate(X_BINS):
        for iq, (q2lo, q2hi) in enumerate(Q2_BINS):
            fig, ax = plt.subplots(figsize=(5, 4))
            cell = (iq, ix)

            _draw_cell(ax, results, cell,
                       h1_data.get(cell), jam_hist.get(cell),
                       compact=False, sum_only=False)

            ax.set_xlabel(r"$x_L$", fontsize=12)
            ax.set_ylabel(r"$F_{2LN}$", fontsize=12)
            ax.set_title(q2_label(q2lo, q2hi) + "   " + xb_label(xlo, xhi), fontsize=9)
            ax.tick_params(which="both", direction="in", top=True, right=True)
            ax.minorticks_on()

            fig.tight_layout()
            base = f"{ind_dir}/f2ln_q{iq}_x{ix}"
            for ext in ("png", "pdf"):
                fig.savefig(f"{base}.{ext}", dpi=150, bbox_inches="tight")
            plt.close(fig)

    print(f"  → {ind_dir}/  ({NQ*NX} plots)")


def average_over_cells(model_dict, h1_data):
    total = np.zeros(N_XL); n = 0
    for cell, model in model_dict.items():
        if cell in h1_data:
            total += model; n += 1
    return total / n if n else np.full(N_XL, np.nan)


def average_h1(h1_data):
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


def make_master_overview(combo_results, h1_data, out,
                          pion_pdf_sets=PION_PDF_SETS, pdf_sets=PDF_SETS):
    h1_avg, h1_err = average_h1(h1_data)
    valid = ~np.isnan(h1_avg)

    n_pion = len(pion_pdf_sets); n_py = len(pdf_sets)
    fig, axes = plt.subplots(n_pion, n_py,
                              figsize=(n_py * 3.0, n_pion * 2.6),
                              sharex=True, squeeze=False)
    fig.suptitle(r"All combinations: pion PDF $\times$ Pythia PDF "
                 r"(cell-averaged raw sum $\mathrm{Py}+\mathrm{pion}$, no normalisation, vs H1)",
                 fontsize=10, y=1.01)

    for ip, (pion_key, _pdfname, pion_label, pion_color) in enumerate(pion_pdf_sets):
        for jp, (py_key, py_label, py_color) in enumerate(pdf_sets):
            ax = axes[ip, jp]
            r = combo_results.get(pion_key, {}).get(py_key)
            if r is not None:
                model_avg = average_over_cells(r["sum_model"], h1_data)
                c2n = np.nanmean(list(r["chi2"].values())) if r.get("chi2") else float('nan')
                ax.stairs(model_avg, XL_EDGES, color=py_color, lw=1.8, ls='--',
                          label=rf"raw sum, $\langle\chi^2/n\rangle={c2n:.2f}$")
            ax.errorbar(XL_CTRS[valid], h1_avg[valid], xerr=XL_XERR[valid],
                        yerr=h1_err[valid], fmt="o", color=C_DATA, ms=2.5,
                        lw=0.8, capsize=1.2, markerfacecolor="white",
                        label="H1 (avg)")

            ax.set_xlim(XL_EDGES[0] - 0.01, XL_EDGES[-1] + 0.01)
            ax.set_ylim(0., 0.10)
            ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=6)
            ax.minorticks_on()
            ax.legend(loc="upper right", fontsize=5, handlelength=1.1,
                      handletextpad=0.3, borderpad=0.3, framealpha=0.85)

            if ip == 0:
                ax.set_title(py_label, fontsize=8, color=py_color)
            if jp == 0:
                ax.set_ylabel(pion_label + "\n" + r"$F_{2LN}$", fontsize=7, color=pion_color)
            if ip == n_pion - 1:
                ax.set_xlabel(r"$x_L$", fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 1])
    for ext in ("png", "pdf"):
        fname = f"{out}_all_combinations_overview.{ext}"
        fig.savefig(fname, dpi=180, bbox_inches="tight")
        print(f"  → {fname}")
    plt.close(fig)


C_DIS_OV   = "#1f77b4"  # DIS fragmentation  (Pythia proton PDF)
C_PDFAV_OV = "#2ca02c"  # PDF average        (pion PDF)
C_TOTAL_OV = "#d62728"

LS_DIS_OV, LS_PDFAV_OV, LS_TOTAL_OV = "--", "-.", "-"

FS_TITLE_OV = 22

FS_CHI2_OV = 15

# H1 F_2^LN reaches ~0.12; 0.14 ceiling
OVERVIEW_YLIM   = (0.0, 0.14)
OVERVIEW_XTICKS = [0.4, 0.6, 0.8]
OVERVIEW_YTICKS = [0.03, 0.06, 0.09, 0.12]


def _ov_tick_fmt(v, pos):
    """Bold tick labels; whole numbers drop their trailing decimals."""
    if abs(v - round(v)) < 1e-9:
        return rf"$\mathbf{{{int(round(v))}}}$"
    return rf"$\mathbf{{{v:.2f}}}$"


def _average_pair_over_cells(py_dict, pion_dict, h1_data):
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

    fig, axes = plt.subplots(
        n_pion, n_py,
        figsize=(n_py * 4.8, n_pion * 4.0),
        gridspec_kw=dict(wspace=0.0, hspace=0.0),
        sharex=True, sharey=True, squeeze=False)
    fig.patch.set_facecolor("white")
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
    _lbl_stroke = [patheffects.withStroke(linewidth=AXIS_LABEL_STROKE,
                                          foreground="black")]

    t_x = fig.text(cx, uy0 - pad, r"$x_L$", ha="center", va="top",
                   fontsize=FS_AXLABEL, fontweight="bold")
    t_x.set_path_effects(_lbl_stroke)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--default",
        default=os.path.join(PYTHIA_OUT, "h1_default.root"),
        help="Pythia ROOT file (default PDF)")
    parser.add_argument(
        "--ct18",
        default=os.path.join(PYTHIA_OUT, "h1_ct18nlo.root"),
        help="Pythia ROOT file (CT18 NLO)")
    parser.add_argument(
        "--ct18remnant",
        default=os.path.join(PYTHIA_OUT, "h1_ct18nlo_remnant.root"),
        help="Pythia ROOT file (CT18 NLO + Remnant)")
    parser.add_argument(
        "--ct18ptune",
        default=os.path.join(PYTHIA_OUT, "h1_ct18nlo_tuning.root"),
        help="Pythia ROOT file (CT18NLO+pythia Tune)")
    parser.add_argument(
        "--ct18gluon",
        default=os.path.join(PYTHIA_OUT, "h1_ct18nlo_gluon.root"),
        help="Pythia ROOT file (CT18NLO+gluon)")
    parser.add_argument("--hdat", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "h1.dat"))
    parser.add_argument("--out", default="sum_compare",
                        help="Output filename prefix (no extension)")
    args = parser.parse_args()

    paths = {"default": args.default, "ct18": args.ct18,
              "ct18remnant": args.ct18remnant, "ct18ptune": args.ct18ptune,
              "ct18gluon": args.ct18gluon}

    # H1 data
    print(f"Loading H1 data from {args.hdat} ...")
    h1_data = load_hdat_f2ln(args.hdat)
    print(f"  Loaded {len(h1_data)} cells")

    # JAM21 NLO (shared across all PDF sets)
    print("\nComputing JAM21 NLO F_2LN curves ...")
    jam_data = load_jam21_f2ln()

    print("\nRebinning JAM21 onto XL_EDGES ...")
    jam_hist = rebin_jam21_to_xl_edges(jam_data)

    results = process_all_pythia(paths)
    if not results:
        raise SystemExit("No input ROOT files were loaded — nothing to plot.")

    print("\nBuilding summed model  Pythia + JAM21  (no normalisation) ...")
    finalize_pdf_set(results, jam_hist, h1_data)

    # DIS background only (raw Pythia, all six sets) vs H1
    print("\nSaving DIS background-only 7×7 grid ...")
    make_grid_bg_only(results, h1_data, jam_hist, args.out)

    # Summed  Pythia + JAM21  (added curves, all six sets) vs H1
    print("\nSaving summed  Pythia + JAM21  7×7 grid ...")
    make_grid_bg_jam(results, h1_data, jam_hist, args.out)

    # overall chi^2: H1 data vs. each Pythia_k + JAM21 sum
    print("\nComputing overall chi^2 for each Pythia PDF set + JAM21 ...")
    overall_chi2 = compute_overall_chi2(results, h1_data)
    save_chi2_table(
        overall_chi2,
        f"{args.out}_chi2_pythia_plus_jam21.txt",
        title="Overall chi^2: H1 data vs. Pythia PDF-set + JAM21 curves",
    )

    # publication overview grid: 3 pion PDFs x 5 Pythia PDF sets
    print("\n" + "=" * 70)
    print("Generating publication-style overview grid ...")
    print("=" * 70)

    print("\nComputing three pion-PDF F_2LN histograms "
          "(JAM21 / xFitter / GRV) ...")
    pion_hists_ov = load_pion_pdf_hists(PION_PDF_SETS_OVERVIEW)
    if not pion_hists_ov:
        print("  ⚠ No pion PDFs could be loaded — overview grid skipped")
    else:
        # label -> f2ln dict, in the canonical PDF_SETS column order
        pythia_dict = {label: results[key]["f2ln"]
                       for key, label, _c in PDF_SETS if key in results}
        if pythia_dict:
            print(f"\nCreating publication grid with {len(pythia_dict)} "
                  f"Pythia PDF sets x {len(pion_hists_ov)} pion PDFs ...")
            make_publication_overview_grid(pythia_dict, pion_hists_ov,
                                           h1_data, args.out)
        else:
            print("  ⚠ No Pythia PDF sets available — overview grid skipped")

    print("\nDone.")

if __name__ == "__main__":
    main()
