"""ZEUS: the five Pythia settings, alone and with the JAM21 Sullivan term."""

import argparse, os, sys, ctypes, warnings, gc
PYTHIA_OUT = os.environ.get("PYTHIA_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Pythia", "output"))
from functools import lru_cache

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib.ticker import AutoMinorLocator, FuncFormatter
from scipy.integrate import quad, IntegrationWarning

warnings.filterwarnings("ignore", category=IntegrationWarning)

# FONT SIZES / WEIGHTS
FS_AXLABEL = 42  # F_2^LN(...) and x_L axis labels
FS_TITLE   = 28  # Q^2 column titles
FS_ROWLAB  = 26  # x_BJ row labels
FS_TICK    = 24  # x and y tick numbers
FS_LEGEND  = 24  # the one shared corner legend

SPINE_LW   = 3.2  # panel box
LW_PYTHIA  = 2.6  # raw Pythia (DIS background) histograms
LW_JAM     = 3.2  # raw JAM21 pion histogram
LW_SUM     = 3.2  # Pythia + JAM21 sums
DATA_MS    = 7.0  # ZEUS marker size
DATA_MEW   = 1.5
DATA_ELW   = 2.0  # error-bar line width
DATA_CAPS  = 3.5

matplotlib.rcParams.update({
    "font.family"         : "serif",
    "font.serif"          : ["Times New Roman", "DejaVu Serif"],
    "font.weight"         : "bold",
    "mathtext.fontset"    : "dejavuserif",
    "mathtext.default"    : "bfit",  # bold *italic* math (H1/ZEUS style)
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

CHUNK_SIZE = 2_000_000

C_DATA = "black"

AXIS_LABEL_STROKE = 1.1

C_JAM21 = "navy"  # JAM21 raw histogram (no normalisation)

PDF_SETS = [
    ("default",     "Pythia default",      "#1f77b4"),
    ("ct18ptune",   "CT18NLO + Tuning",    "#ff7f0e"),
    ("ct18",        "CT18NLO",             "#d62728"),
    ("ct18remnant", "CT18NLO+Remnant",     "#9467bd"),
    ("ct18gluon",   "CT18NLO + Gluon",     "#17becf"),
]

PDF_SET_DEFAULT_PATHS = {
    "default":     os.path.join(PYTHIA_OUT, "zeus_default.root"),
    "ct18ptune":   os.path.join(PYTHIA_OUT, "zeus_ct18nlo_tuning.root"),
    "ct18":        os.path.join(PYTHIA_OUT, "zeus_ct18nlo.root"),
    "ct18remnant": os.path.join(PYTHIA_OUT, "zeus_ct18nlo_remnant.root"),
    "ct18gluon":   os.path.join(PYTHIA_OUT, "zeus_ct18nlo_gluon.root"),
}

JAM21_PDF_NAME = "JAM21PionPDFnlo"
PION_PDF_SETS = [
    ("jam21", JAM21_PDF_NAME, "JAM21", C_JAM21),
]

ALPHA      = 1.0 / 137.036
GEV2_TO_NB = 0.3893793656e6
E_E        = 27.5
E_P        = 820.0
S          = 4.0 * E_E * E_P  # 90 200 GeV^2

# ZEUS cuts
EePrime_MIN  = 10.0  # E_e' > 10 GeV
THETA_N_MAX  = 0.8e-3  # theta_n  < 0.8 mrad
PT_COEFF     = 0.656

F2P_TABLE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "F2p_CT18NLO_values.txt")

XL_EDGES = np.array([0.20, 0.28, 0.34, 0.40, 0.46, 0.52, 0.58,
                      0.64, 0.70, 0.76, 0.82, 0.88, 1.00])
XL_CTRS  = 0.5 * (XL_EDGES[:-1] + XL_EDGES[1:])
XL_XERR  = 0.5 * (XL_EDGES[1:]  - XL_EDGES[:-1])
XL_BWS   = XL_EDGES[1:] - XL_EDGES[:-1]
N_XL     = len(XL_CTRS)

# ZEUS Q^2 bins (7 columns)
Q2_BINS_ZEUS = [
    (7.0,    4.0,    10.0),
    (15.0,   10.0,   20.0),
    (30.0,   20.0,   40.0),
    (60.0,   40.0,   80.0),
    (120.0,  80.0,   160.0),
    (240.0,  160.0,  320.0),
    (480.0,  320.0,  640.0),
]

Q2_ZEUS_DAT_KEY = {}

X_BINS_ZEUS = [
    (1.1e-4, 8.0e-5, 1.5e-4),
    (2.1e-4, 1.5e-4, 3.0e-4),
    (4.2e-4, 3.0e-4, 6.0e-4),
    (8.5e-4, 6.0e-4, 1.2e-3),
    (17.e-4, 1.2e-3, 2.4e-3),
    (49.e-4, 2.4e-3, 1.0e-2),
    (32.e-3, 1.0e-2, 1.0e-1),
]

NQ_ZEUS = len(Q2_BINS_ZEUS)
NX_ZEUS = len(X_BINS_ZEUS)


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
    return rf"$x_{{B}}\!\in\![{sci(lo)},\,{sci(hi)}]$"

def compute_y(Q2, x):
    return Q2 / (S * x)

def dis_prefactor(x, Q2):
    """P(x,Q^2) [nb/GeV^2] such that dsigma/dx_L = P x kin(y) x F_2LN x dx x dQ^2."""
    return (4.0 * np.pi * ALPHA**2 / (x * Q2**2)) * GEV2_TO_NB

def dsig_to_f2ln(dsig_per_xl, x, Q2, dx, dQ2):
    y   = compute_y(Q2, x)
    kin = 1.0 - y + 0.5 * y**2
    P   = dis_prefactor(x, Q2)
    return np.asarray(dsig_per_xl, dtype=float) / (P * kin * dx * dQ2)


# JAM21 pion PDF (NLO F2^pi) + leading-neutron flux
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

X_FINE       = np.logspace(-6, np.log10(1.0 - 1e-8), 5000)
PIDS_NEEDED  = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 21]
PDF_CACHE    = {}


def load_jam21_pdf():
    """Try to load the JAM21 pion PDF via LHAPDF; return None on failure."""
    try:
        import lhapdf
        pdf = lhapdf.mkPDF(JAM21_PDF_NAME, 0)
        print(f"  Loaded pion PDF: {JAM21_PDF_NAME}")
        return pdf
    except Exception as e:
        print(f"  WARNING: could not load JAM21 pion PDF ({e}); "
              f"JAM21 curves will be omitted.")
        return None


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
    """NLO F2 (DIS scheme) for the pion, evaluated from the JAM21 pion PDF."""
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
    """Full 7(x) x 8(Q2) rectangular grid."""
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
        print(f"  JAM21 F_2LN computed for Q2 column {iq+1}/{NQ_ZEUS} "
              f"(Q2={q2_nom:g})")
    return f2ln_jam


def load_f2p_table(filename=F2P_TABLE_FILE):
    table = np.loadtxt(filename)
    return table[:, 0], table[:, 1], table[:, 2]  # x, Q2, F2p(CT18NLO)


def match_f2p(x_query, q2_query, x_table, q2_table, f2p_table):
    """Map each ZEUS-row (x,Q2) onto the corresponding F2p_CT18NLO value."""
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


def load_pythia(path):
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

        print()  # newline after the progress line

    print(f"  sigma_DIS = {sigma_dis:.4f} nb")
    print(f"  N_total (generated)                  = {N_TOTAL:,}")
    print(f"  Events after E_e', theta_n cuts       = {n_after_cuts:,}")
    print(f"  Events + pT < {PT_COEFF}*xL            = {n_after_pt:,}")

    return dict(sigma_dis=sigma_dis, counts=counts)


def compute_all_pythia_f2ln_zeus(data):
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


def process_pythia(path, label):
    print(f"  Loading Pythia ROOT file ({label}) from {path} ...")
    py_data = load_pythia(path)
    sigma_dis = py_data["sigma_dis"]
    print(f"  Computing F_2LN from Pythia dσ/dx_L ({label}, ZEUS cuts/bins) ...")
    f2ln_py, nevts = compute_all_pythia_f2ln_zeus(py_data)
    return f2ln_py, nevts, sigma_dis


def process_all_pythia(paths):
    """Loads and processes every requested PDF-set ROOT file."""
    results = {}
    for key, label, _ in PDF_SETS:
        path = paths.get(key)
        if not path or not os.path.exists(path):
            print(f"\nWARNING: ROOT file '{path}' ({label}) not found -- "
                  f"this PDF set will be omitted.")
            continue
        print(f"\nLoading Pythia ({label}): {path}")
        f2ln, nevts, sigma_dis = process_pythia(path, label)
        gc.collect()
        results[key] = dict(f2ln=f2ln, nevts=nevts, sigma_dis=sigma_dis)
    return results


def load_zeus_f2ln(path, f2p_table_file=F2P_TABLE_FILE):
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


def compute_sum_model(f2ln_py, jam_hist):
    """Returns sum_model[(iq,ix)] = py + jam_hist, for cells with both."""
    sum_model = {}
    for cell, py in f2ln_py.items():
        jh = jam_hist.get(cell)
        if jh is None:
            continue
        sum_model[cell] = py + jh
    return sum_model


def compute_chi2(sum_model, zeus_data):
    """Returns chi2[(iq,ix)] = chi2/n."""
    chi2_table = {}
    for cell, model in sum_model.items():
        zd = zeus_data.get(cell)
        if zd is None:
            continue
        xl_d, f2_d, e_d = zd
        idx = np.searchsorted(XL_EDGES[1:], xl_d, side='left')
        idx = np.clip(idx, 0, N_XL - 1)
        model_at = model[idx]
        denom    = np.where(e_d > 0, e_d, 1e-30)
        c2n = np.sum(((f2_d - model_at) / denom) ** 2) / max(len(f2_d), 1)
        chi2_table[cell] = c2n
    return chi2_table


def finalize_pdf_set(results, jam_hist, zeus_data):
    for key, label, _ in PDF_SETS:
        if key not in results:
            continue
        r = results[key]
        sum_model = compute_sum_model(r["f2ln"], jam_hist)
        chi2_table = compute_chi2(sum_model, zeus_data)
        vals = list(chi2_table.values())
        mean = np.mean(vals) if vals else float('nan')
        print(f"  [{label}]  Pythia+JAM21 sum (no normalisation)   "
              f"mean χ²/n = {mean:.4f}  ({len(vals)} cells)")
        r["sum_model"] = sum_model
        r["chi2"] = chi2_table


def compute_overall_chi2(results, zeus_data):
    out = {}
    for key, label, _ in PDF_SETS:
        if key not in results:
            continue
        r = results[key]
        sum_model = r.get("sum_model", {})

        chi2 = 0.0
        ndof = 0
        for cell, model in sum_model.items():
            zd = zeus_data.get(cell)
            if zd is None:
                continue
            xl_d, f2_d, e_d = zd

            idx = np.clip(np.searchsorted(XL_EDGES[1:], xl_d, side='left'),
                          0, N_XL - 1)
            model_at = model[idx]

            valid = np.isfinite(model_at) & np.isfinite(f2_d) & (e_d > 0)
            if not valid.any():
                continue
            chi2 += float(np.sum(((f2_d[valid] - model_at[valid]) / e_d[valid]) ** 2))
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
    print(f"  -> {out_path}")


def _draw_cell(ax, results, cell, zeus_cell, jam_hist_cell,
               compact=True, sum_only=False):
    """Draw one (Q^2, x_BJ) cell onto ax."""
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
        ax.stairs(jam_hist_cell, XL_EDGES, color=C_JAM21, lw=1.3, ls='-',
                  label="JAM21 (raw)")

    # ZEUS data
    if zeus_cell is not None:
        xl_d, f2_d, e_d = zeus_cell
        ax.errorbar(xl_d, f2_d, yerr=e_d,
                    fmt="o", ms=2.8, mfc="black", mec="black", mew=0.7,
                    ecolor="black", elinewidth=0.8, capsize=1.8,
                    color="black", zorder=10, label="ZEUS data")

    ax.set_xlim(XL_EDGES[0] - 0.01, XL_EDGES[-1] + 0.01)
    ax.set_ylim(0., 0.35)
    ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=6)
    ax.minorticks_on()
    ax.legend(loc="upper right", fontsize=fs,
              handlelength=1.2, handletextpad=0.3,
              borderpad=0.3, framealpha=0.85, labelspacing=0.2)


def _make_grid_generic(results, zeus_data, jam_hist, out, suffix, title, sum_only):
    fig, axes = plt.subplots(NX_ZEUS, NQ_ZEUS,
                             figsize=(NQ_ZEUS * 3.2, NX_ZEUS * 2.8),
                             sharex=False, sharey=False)
    fig.suptitle(title, fontsize=9, y=1.002)

    cell_has_data = {}
    for ix in range(NX_ZEUS):
        row = NX_ZEUS - 1 - ix
        for iq in range(NQ_ZEUS):
            cell_has_data[(row, iq)] = ((iq, ix) in zeus_data)

    first_visible_row, last_visible_row = {}, {}
    for iq in range(NQ_ZEUS):
        rr = [row for row in range(NX_ZEUS) if cell_has_data[(row, iq)]]
        if rr:
            first_visible_row[iq] = min(rr)
            last_visible_row[iq]  = max(rr)
    first_visible_col = {}
    for row in range(NX_ZEUS):
        cc = [iq for iq in range(NQ_ZEUS) if cell_has_data[(row, iq)]]
        if cc:
            first_visible_col[row] = min(cc)

    for ix in range(NX_ZEUS):
        row = NX_ZEUS - 1 - ix
        x_nom, xlo, xhi = X_BINS_ZEUS[ix]
        for iq in range(NQ_ZEUS):
            ax   = axes[row, iq]
            cell = (iq, ix)

            if not cell_has_data[(row, iq)]:
                ax.set_visible(False)
                continue

            _draw_cell(ax, results, cell,
                       zeus_data.get(cell), jam_hist.get(cell),
                       compact=True, sum_only=sum_only)

            if row == first_visible_row.get(iq):
                q2_nom, q2lo, q2hi = Q2_BINS_ZEUS[iq]
                ax.set_title(q2_label(q2lo, q2hi), fontsize=7, pad=4)

            if iq == first_visible_col.get(row):
                ax.set_ylabel(xb_label(xlo, xhi) + "\n" + r"$F_{2LN}$", fontsize=6.5)
            else:
                ax.tick_params(labelleft=False)

            if row == last_visible_row.get(iq):
                ax.set_xlabel(r"$x_L$", fontsize=7)
            else:
                ax.tick_params(labelbottom=False)

    fig.tight_layout(rect=[0, 0, 1, 1])
    for ext in ("png", "pdf"):
        fname = f"{out}_f2ln_grid_zeus{suffix}.{ext}"
        fig.savefig(fname, dpi=180, bbox_inches="tight")
        print(f"  -> {fname}")
    plt.close(fig)


def make_grid(results, zeus_data, jam_hist, out):
    """Main small-panel grid: raw Pythia (x5 sets) + JAM21 (raw) + sum (x5 sets) + ZEUS data."""
    title = (rf"$F_{{2LN}}(x_L)$ -- Pythia (baseline/CT18NLO Tuning/CT18NLO/"
              rf"CT18NLO+Remnant/CT18NLO+gluon) + JAM21 (hist) vs ZEUS data")
    _make_grid_generic(results, zeus_data, jam_hist, out,
                       suffix="", title=title, sum_only=False)


def make_grid_sum_only(results, zeus_data, jam_hist, out):
    title = (rf"$F_{{2LN}}(x_L)$ -- raw sum $\mathrm{{Py}}_k+\mathrm{{JAM21}}$ "
              rf"(no normalisation) (baseline/CT18NLO Tuning/CT18NLO/"
              rf"CT18NLO+Remnant/CT18NLO+gluon) vs ZEUS data")
    _make_grid_generic(results, zeus_data, jam_hist, out,
                       suffix="_sumonly", title=title, sum_only=True)


def _pythia_label(label):
    """'CT18NLO' -> 'Pythia (CT18NLO)', but leave names that already say Pythia alone."""
    return label if label.strip().lower().startswith("pythia") else f"Pythia ({label})"


def _xl_tick_fmt(v, pos):
    if abs(v - round(v)) < 1e-9:
        return rf"$\mathbf{{{int(round(v))}}}$"
    return rf"$\mathbf{{{v:.2f}}}$"


def _q2_label_bold(lo, hi):
    return rf"$Q^2 \in [{lo:.4g},\,{hi:.4g}]$"


def _xb_label_bold(lo, hi):
    return rf"$x_{{B}} \in [{sci(lo)},$" + "\n" + rf"${sci(hi)}]$"


def _draw_cell_bg(ax, results, cell, zeus_cell, jam_hist_cell,
                   include_pythia, include_jam, include_sum, compact=True):
    """Draw one (Q^2, x_B) cell."""
    if include_pythia:
        for key, label, color in PDF_SETS:
            if key not in results:
                continue
            py = results[key]["f2ln"].get(cell)
            if py is not None:
                ax.stairs(py, XL_EDGES, color=color, lw=LW_PYTHIA,
                          label=_pythia_label(label))

    if include_jam and jam_hist_cell is not None:
        ax.stairs(jam_hist_cell, XL_EDGES, color=C_JAM21, lw=LW_JAM, ls='-',
                  label="JAM21")

    if include_sum:
        for key, label, color in PDF_SETS:
            if key not in results:
                continue
            sum_cell = results[key].get("sum_model", {}).get(cell)
            if sum_cell is not None:
                ax.stairs(sum_cell, XL_EDGES, color=color, lw=LW_SUM, ls='-',
                          label=f"{_pythia_label(label)} + JAM21")

    if zeus_cell is not None:
        xl_d, f2_d, e_d = zeus_cell
        ax.errorbar(xl_d, f2_d, yerr=e_d,
                    fmt="o", ms=DATA_MS, mfc="black", mec="black",
                    mew=DATA_MEW, ecolor="black", elinewidth=DATA_ELW,
                    capsize=DATA_CAPS, capthick=DATA_ELW,
                    color="black", zorder=10, label="ZEUS data")

    ax.set_xlim(XL_EDGES[0] - 0.02, 1.0)
    ax.set_ylim(0., 0.35)
    ax.set_facecolor("white")

    for sp in ax.spines.values():
        sp.set_linewidth(SPINE_LW)

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
    blocks = [b for b in (_largest_empty_block_at(cell_has_data, nrow, ncol, "bottom-right"),
                          _largest_empty_block_at(cell_has_data, nrow, ncol, "top-left"))
              if b]
    if not blocks:
        return None
    return max(blocks, key=len)


def _block_centre(axes, block):
    positions = [axes[r][q].get_position() for r, q in block]
    x0 = min(p.x0 for p in positions); x1 = max(p.x1 for p in positions)
    y0 = min(p.y0 for p in positions); y1 = max(p.y1 for p in positions)
    return 0.5 * (x0 + x1), 0.5 * (y0 + y1)


def _place_f2_label_top_left(fig, axes, cell_has_data, nrow, ncol, text,
                             fontsize, stroke=None, dx=0.0):
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


def _place_legend_in_empty_space(fig, axes, cell_has_data, source_ax, nrow, ncol):
    handles, labels = source_ax.get_legend_handles_labels()
    if not handles:
        return

    block = _largest_corner_empty_block(cell_has_data, nrow, ncol)
    if not block:
        fig.legend(handles, labels, loc="upper right",
                   prop={"weight": "bold", "size": 22}, framealpha=0.96)
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


def _make_grid_bg_generic(results, zeus_data, jam_hist, out, suffix,
                          include_pythia, include_jam, include_sum=False):
    fig, axes = plt.subplots(NX_ZEUS, NQ_ZEUS,
                             figsize=(NQ_ZEUS * 4.8, NX_ZEUS * 4.0),
                             gridspec_kw=dict(wspace=0.0, hspace=0.0),
                             sharex=False, sharey=False)
    fig.patch.set_facecolor("white")
    # No fig.suptitle(...)

    cell_has_data = {}
    for ix in range(NX_ZEUS):
        row = NX_ZEUS - 1 - ix
        for iq in range(NQ_ZEUS):
            cell_has_data[(row, iq)] = zeus_data.get((iq, ix)) is not None

    legend_cell = None
    for row in range(NX_ZEUS):
        for iq in range(NQ_ZEUS):
            if cell_has_data[(row, iq)]:
                legend_cell = (row, iq)
                break
        if legend_cell is not None:
            break

    last_visible_row  = {}
    first_visible_row = {}
    for iq in range(NQ_ZEUS):
        rows_with_data = [row for row in range(NX_ZEUS) if cell_has_data[(row, iq)]]
        if rows_with_data:
            first_visible_row[iq] = min(rows_with_data)
            last_visible_row[iq]  = max(rows_with_data)

    first_visible_col = {}
    for row in range(NX_ZEUS):
        cols_with_data = [iq for iq in range(NQ_ZEUS) if cell_has_data[(row, iq)]]
        if cols_with_data:
            first_visible_col[row] = min(cols_with_data)

    legend_source_ax = None
    for ix in range(NX_ZEUS):
        row = NX_ZEUS - 1 - ix
        for iq in range(NQ_ZEUS):
            ax   = axes[row, iq]
            cell = (iq, ix)

            if not cell_has_data[(row, iq)]:
                ax.set_visible(False)
                continue

            ax.set_facecolor("white")

            _draw_cell_bg(ax, results, cell,
                         zeus_data.get(cell), jam_hist.get(cell),
                         include_pythia=include_pythia,
                         include_jam=include_jam,
                         include_sum=include_sum,
                         compact=True)

            if (row, iq) == legend_cell:
                legend_source_ax = ax

            if row == first_visible_row.get(iq):
                q2_nom, q2lo, q2hi = Q2_BINS_ZEUS[iq]
                ax.set_title(_q2_label_bold(q2lo, q2hi), fontsize=FS_TITLE,
                            pad=9, fontweight="bold")

            if iq == first_visible_col.get(row):
                x_nom, xlo, xhi = X_BINS_ZEUS[ix]
                ax.text(-0.26, 0.5, _xb_label_bold(xlo, xhi),
                        transform=ax.transAxes, ha="right", va="center",
                        rotation=0, linespacing=1.4,
                        fontsize=FS_ROWLAB, fontweight="bold")
                ax.tick_params(labelleft=True)
            else:
                ax.tick_params(labelleft=False)

            ax.tick_params(labelbottom=(row == last_visible_row.get(iq)))

    if legend_source_ax is not None:
        _place_legend_in_empty_space(fig, axes, cell_has_data, legend_source_ax,
                                     NX_ZEUS, NQ_ZEUS)

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
        ux0 = min(p.x0 for p in pos) - 0.03
        uy0 = min(p.y0 for p in pos) - 0.03

    pad = 0.010
    _lbl_stroke = [patheffects.withStroke(linewidth=AXIS_LABEL_STROKE,
                                          foreground="black")]
    t_x = fig.text(cx, uy0 - pad, r"$x_L$",
                   ha="center", va="top", fontsize=FS_AXLABEL, fontweight="bold")
    t_x.set_path_effects(_lbl_stroke)

    _f2_text = r"$F_{2}^{LN}(x_{B},\,Q^{2},\,x_L)$"
    if not _place_f2_label_top_left(fig, axes, cell_has_data, NX_ZEUS, NQ_ZEUS,
                                    _f2_text, FS_AXLABEL, stroke=_lbl_stroke,
                                    dx=-0.045):
        t_y = fig.text(ux0 - pad, cy, _f2_text,
                       ha="center", va="center", rotation=90,
                       fontsize=FS_AXLABEL, fontweight="bold")
        t_y.set_path_effects(_lbl_stroke)

    for ext in ("png", "pdf"):
        fname = f"{out}_f2ln_grid_zeus{suffix}.{ext}"
        fig.savefig(fname, dpi=180, bbox_inches="tight", facecolor="white")
        print(f"  -> {fname}")
    plt.close(fig)


def make_grid_bg_only(results, zeus_data, jam_hist, out):
    _make_grid_bg_generic(results, zeus_data, jam_hist, out,
                          suffix="_dis_background_only",
                          include_pythia=True, include_jam=False,
                          include_sum=False)


def make_grid_bg_jam(results, zeus_data, jam_hist, out):
    _make_grid_bg_generic(results, zeus_data, jam_hist, out,
                          suffix="_dis_background_jam21",
                          include_pythia=False, include_jam=False,
                          include_sum=True)


def make_individual(results, zeus_data, jam_hist, out):
    ind_dir = f"{out}_f2ln_individual"
    os.makedirs(ind_dir, exist_ok=True)

    n_saved = 0
    for ix, (x_nom, xlo, xhi) in enumerate(X_BINS_ZEUS):
        for iq, (q2_nom, q2lo, q2hi) in enumerate(Q2_BINS_ZEUS):
            cell = (iq, ix)
            if cell not in zeus_data:
                continue

            fig, ax = plt.subplots(figsize=(5, 4))

            _draw_cell(ax, results, cell,
                       zeus_data.get(cell), jam_hist.get(cell),
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
            n_saved += 1

    print(f"  -> {ind_dir}/  ({n_saved} plots)")


def average_over_cells(model_dict, zeus_data):
    total = np.zeros(N_XL); n = 0
    for cell, model in model_dict.items():
        if cell in zeus_data:
            total += model; n += 1
    return total / n if n else np.full(N_XL, np.nan)


def average_zeus(zeus_data):
    sum_wf = np.zeros(N_XL); sum_w = np.zeros(N_XL)
    for xl_d, f2_d, e_d in zeus_data.values():
        idx = np.clip(np.searchsorted(XL_EDGES[1:], xl_d, side='left'), 0, N_XL - 1)
        w = np.where(e_d > 0, 1.0 / e_d**2, 0.0)
        for j, i in zip(range(len(idx)), idx):
            sum_wf[i] += w[j] * f2_d[j]
            sum_w[i]  += w[j]
    avg = np.divide(sum_wf, sum_w, out=np.full(N_XL, np.nan), where=sum_w > 0)
    err = np.divide(1.0, np.sqrt(sum_w, out=np.full(N_XL, np.nan), where=sum_w > 0),
                     out=np.full(N_XL, np.nan), where=sum_w > 0)
    return avg, err


def make_master_overview(combo_results, zeus_data, out,
                          pion_pdf_sets=PION_PDF_SETS, pdf_sets=PDF_SETS):
    zeus_avg, zeus_err = average_zeus(zeus_data)
    valid = ~np.isnan(zeus_avg)

    n_pion = len(pion_pdf_sets); n_py = len(pdf_sets)
    fig, axes = plt.subplots(n_pion, n_py,
                              figsize=(n_py * 3.0, n_pion * 2.6),
                              sharex=True, squeeze=False)
    fig.suptitle(r"All combinations: pion PDF $\times$ Pythia PDF "
                 r"(cell-averaged raw sum $\mathrm{Py}+\mathrm{pion}$, no normalisation, vs ZEUS)",
                 fontsize=10, y=1.01)

    for ip, (pion_key, _pdfname, pion_label, pion_color) in enumerate(pion_pdf_sets):
        for jp, (py_key, py_label, py_color) in enumerate(pdf_sets):
            ax = axes[ip, jp]
            r = combo_results.get(pion_key, {}).get(py_key)
            if r is not None:
                model_avg = average_over_cells(r["sum_model"], zeus_data)
                c2n = np.nanmean(list(r["chi2"].values())) if r.get("chi2") else float('nan')
                ax.stairs(model_avg, XL_EDGES, color=py_color, lw=1.8, ls='--',
                          label=rf"raw sum, $\langle\chi^2/n\rangle={c2n:.2f}$")
            ax.errorbar(XL_CTRS[valid], zeus_avg[valid], xerr=XL_XERR[valid],
                        yerr=zeus_err[valid], fmt="o", color=C_DATA, ms=2.5,
                        lw=0.8, capsize=1.2, markerfacecolor="white",
                        label="ZEUS (avg)")

            ax.set_xlim(XL_EDGES[0] - 0.01, XL_EDGES[-1] + 0.01)
            ax.set_ylim(0., 0.35)
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
        print(f"  -> {fname}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--default",
        default=PDF_SET_DEFAULT_PATHS["default"],
        help="Pythia ROOT file (baseline PDF)")
    parser.add_argument(
        "--ct18ptune",
        default=PDF_SET_DEFAULT_PATHS["ct18ptune"],
        help="Pythia ROOT file (CT18NLO + Tuning)")
    parser.add_argument(
        "--ct18",
        default=PDF_SET_DEFAULT_PATHS["ct18"],
        help="Pythia ROOT file (CT18NLO)")
    parser.add_argument(
        "--ct18remnant",
        default=PDF_SET_DEFAULT_PATHS["ct18remnant"],
        help="Pythia ROOT file (CT18NLO + Remnant)")
    parser.add_argument(
        "--ct18gluon",
        default=PDF_SET_DEFAULT_PATHS["ct18gluon"],
        help="Pythia ROOT file (CT18NLO + Gluon)")
    parser.add_argument("--zdat", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "zeus.dat"))
    parser.add_argument("--f2ptable", default=F2P_TABLE_FILE,
                         help="F2p_CT18NLO_values.txt (x, Q2, F2^p NLO)")
    parser.add_argument("--out", default="compare_f2ln_zeus",
                        help="Output filename prefix (no extension)")
    parser.add_argument("--no-jam21", action="store_true",
                         help="skip the JAM21 pion-PDF theory curve")
    args = parser.parse_args()

    paths = {"default": args.default, "ct18ptune": args.ct18ptune,
             "ct18": args.ct18, "ct18remnant": args.ct18remnant,
             "ct18gluon": args.ct18gluon}

    # ZEUS data
    print(f"Loading ZEUS F_2LN data from {args.zdat} "
          f"(F2p sourced from {args.f2ptable}) ...")
    zeus_data = load_zeus_f2ln(args.zdat, f2p_table_file=args.f2ptable)
    print(f"  Loaded {len(zeus_data)} (Q2,x) cells "
          f"({sum(len(v[0]) for v in zeus_data.values())} data points)")

    # JAM21 NLO (shared across all PDF sets)
    have_jam21 = False
    jam_hist = {}
    if not args.no_jam21:
        print(f"\nLoading JAM21 pion PDF ({JAM21_PDF_NAME}) ...")
        jam_pdf = load_jam21_pdf()
        if jam_pdf is not None:
            have_jam21 = True
            print("Computing F_2LN = 2*flux(x_L)*F_2^pi_NLO(x_pi,Q2) "
                  "over the full ZEUS grid ...")
            jam_hist = compute_all_jam21_f2ln_zeus(jam_pdf)
    if not have_jam21:
        print("  WARNING: proceeding without JAM21 -- sum curves will "
              "equal raw Pythia.")

    results = process_all_pythia(paths)
    if not results:
        raise SystemExit("No input ROOT files were loaded -- nothing to plot.")

    # build the summed model  Pythia_k + JAM21  (no normalisation)
    print("\nBuilding summed model  Pythia + JAM21  (no normalisation) ...")
    finalize_pdf_set(results, jam_hist, zeus_data)

    # DIS background only (raw Pythia, all five sets) vs ZEUS
    print("\nSaving DIS background-only grid ...")
    make_grid_bg_only(results, zeus_data, jam_hist, args.out)

    print("\nSaving summed  Pythia + JAM21  grid ...")
    make_grid_bg_jam(results, zeus_data, jam_hist, args.out)

    # Main small-panel grid: raw Pythia + JAM21 + sum + ZEUS data
    print("\nSaving main F_2LN comparison grid ...")
    make_grid(results, zeus_data, jam_hist, args.out)

    # Sum-only small-panel grid
    print("\nSaving sum-only grid (Pythia_k + JAM21, no normalisation) ...")
    make_grid_sum_only(results, zeus_data, jam_hist, args.out)

    # individual cell plots
    print("\nSaving individual cell plots ...")
    make_individual(results, zeus_data, jam_hist, args.out)

    # overall chi^2: ZEUS data vs. each Pythia_k + JAM21 sum
    print("\nComputing overall chi^2 for each Pythia PDF set + JAM21 ...")
    overall_chi2 = compute_overall_chi2(results, zeus_data)
    save_chi2_table(
        overall_chi2,
        f"{args.out}_chi2_pythia_plus_jam21.txt",
        title="Overall chi^2: ZEUS data vs. Pythia PDF-set + JAM21 curves",
    )

    # master overview (single pion PDF: JAM21)
    print("\nSaving combined overview (pion PDF x Pythia PDF) ...")
    combo_results = {"jam21": results}
    make_master_overview(combo_results, zeus_data, args.out)

    print("\nDone.")


if __name__ == "__main__":
    main()
