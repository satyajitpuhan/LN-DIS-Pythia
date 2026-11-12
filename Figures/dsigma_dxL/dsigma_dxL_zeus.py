"""ZEUS dsigma/dx_L of the five Pythia samples compared with the data (Fig. 8, right),
and F2^LN from Pythia on the ZEUS grid.
"""

import argparse, os
PYTHIA_OUT = os.environ.get("PYTHIA_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Pythia", "output"))
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib.ticker import MultipleLocator

matplotlib.rcParams.update({
    "font.family"        : "serif",
    "font.serif"         : ["Times New Roman", "DejaVu Serif"],
    "font.weight"        : "bold",
    "mathtext.fontset"   : "dejavuserif",
    "mathtext.default"   : "bfit",  # bold *italic* math (H1 style)
    "axes.labelsize"     : 10,
    "axes.labelweight"   : "bold",
    "axes.titlesize"     : 8,
    "axes.titleweight"   : "bold",
    "xtick.labelsize"    : 10,
    "ytick.labelsize"    : 10,
    "xtick.direction"    : "in",
    "ytick.direction"    : "in",
    "xtick.top"          : True,
    "ytick.right"        : True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.framealpha"  : 0.92,
    "legend.edgecolor"   : "0.45",
    "legend.fontsize"    : 6,
    "figure.dpi"         : 150,
    "axes.linewidth"     : 1.4,
})

N_TOTAL = 500_000_000

CHUNK_SIZE = 2_000_000

C_DEFAULT      = "#1f77b4"  # blue
C_CT18         = "#d62728"  # red
C_CT18_REMNANT = "#9467bd"  # purple
C_CT18_PTUNE   = "#ff7f0e"  # orange
C_CT18_GLUON   = "#17becf"  # cyan
C_DATA         = "black"

FS_AXLABEL = 26  # x_L and dsigma/dx_L axis labels
FS_TICK    = 17  # x and y tick numbers
FS_LEGEND  = 12  # curve legend

TICK_LABEL_STROKE = 0.6
AXIS_LABEL_STROKE = 1.1  # the big x_L / F_2^{LN} axis labels


def _stroke(*artists, width=TICK_LABEL_STROKE):
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

C_MC   = C_DEFAULT
C_SUM  = C_CT18

# Multiple Pythia ROOT files (5 total)
PYTHIA_FILES = [
    dict(path=os.path.join(PYTHIA_OUT, "zeus_default.root"),
         label="Pythia default",
         color=C_DEFAULT, ls="solid", lw=2.4),
    dict(path=os.path.join(PYTHIA_OUT, "zeus_ct18nlo.root"),
         label="CT18NLO",
         color=C_CT18, ls="solid", lw=3.4),
    dict(path=os.path.join(PYTHIA_OUT, "zeus_ct18nlo_remnant.root"),
         label="CT18NLO+Remnant",
         color=C_CT18_REMNANT, ls="dotted", lw=2.4),
    dict(path=os.path.join(PYTHIA_OUT, "zeus_ct18nlo_tuning.root"),
         label="CT18NLO Tuning",
         color=C_CT18_PTUNE, ls="dashed", lw=2.4),
    dict(path=os.path.join(PYTHIA_OUT, "zeus_ct18nlo_gluon.root"),
         label="CT18NLO+gluon",
         color=C_CT18_GLUON, ls="dashdot", lw=2.4),
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

XL_EDGES = np.array([0.20, 0.28, 0.34, 0.40, 0.46, 0.52, 0.58,
                      0.64, 0.70, 0.76, 0.82, 0.88, 1.00])
XL_CTRS  = 0.5 * (XL_EDGES[:-1] + XL_EDGES[1:])
XL_XERR  = 0.5 * (XL_EDGES[1:]  - XL_EDGES[:-1])
XL_BWS   = XL_EDGES[1:] - XL_EDGES[:-1]
N_XL     = len(XL_CTRS)

# ZEUS Q^2 bins (8 columns)
Q2_BINS_ZEUS = [
    (7.0,    4.0,    10.0),
    (15.0,   10.0,   20.0),
    (30.0,   20.0,   40.0),
    (60.0,   40.0,   80.0),
    (120.0,  80.0,   160.0),
    (240.0,  160.0,  320.0),
    (480.0,  320.0,  640.0),
    (1000.0, 640.0,  10000.0),
]

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
NX_ZEUS = len(X_BINS_ZEUS)  # 7   -> full 7x8 = 56-panel grid


# JAM21 pion PDF (NLO F2^pi) + leading-neutron flux
import ctypes, sys, warnings
from functools import lru_cache
from scipy.integrate import quad, IntegrationWarning

warnings.filterwarnings("ignore", category=IntegrationWarning)

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

JAM21_PDF_NAME  = "JAM21PionPDFnlo_pT"
C_JAM21         = "navy"
F2P_TABLE_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "F2p_CT18NLO_values.txt")


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
        print(f"  Tree has {total_entries:,} entries — "
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


def zeus_cell_mask(zeus_data):
    """Boolean (NQ_ZEUS, NX_ZEUS) mask, True for cells with ZEUS data."""
    mask = np.zeros((NQ_ZEUS, NX_ZEUS), dtype=bool)
    for (iq, ix) in zeus_data.keys():
        mask[iq, ix] = True
    return mask


def dsigma_zeus_total_from_counts(counts, sigma_dis, cell_mask):
    """counts: (NQ_ZEUS, NX_ZEUS, N_XL) histogram from load_pythia()."""
    total_hist = counts[cell_mask].sum(axis=0)  # -> shape (N_XL,)
    return sigma_dis * total_hist / N_TOTAL / XL_BWS


def compute_all_pythia_dsigma_zeus_total(pythia_runs, cell_mask):
    runs = []
    for run in pythia_runs:
        dsigma = np.full(N_XL, np.nan)
        if run["have_pythia"]:
            dsigma = dsigma_zeus_total_from_counts(
                run["counts"], run["sigma_dis"], cell_mask)
        runs.append(dict(label=run["label"], color=run["color"],
                          ls=run.get("ls", "solid"), lw=run.get("lw", 2.4),
                          have_pythia=run["have_pythia"], dsigma=dsigma))
    return runs


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
        for ix, (x_nom, _, _) in enumerate(X_BINS_ZEUS):
            lut[(round(q2_nom, 6), round(x_nom, 8))] = (iq, ix)

    data = {}
    for k in range(len(raw)):
        key = (round(float(q2[k]), 6), round(float(xbj[k]), 8))
        # allow small relative mismatch just in case of rounding
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


def _q2_label_bold(lo, hi):
    return rf"$Q^2\!\in\![{lo:.4g},\,{hi:.4g}]\ \mathbf{{GeV}}^2$"


def _xb_label_bold(lo, hi):
    return rf"$x_B\!\in\![{sci(lo)},\,{sci(hi)}]$"


def _largest_corner_empty_block(cell_has_data, nrow, ncol):
    def _all_empty(cells):
        return all(not cell_has_data.get(c, False) for c in cells)

    best = None
    for r0 in range(nrow):  # bottom-right anchored
        for q0 in range(ncol):
            block = [(r, q) for r in range(r0, nrow) for q in range(q0, ncol)]
            if block and _all_empty(block):
                if best is None or len(block) > best[0]:
                    best = (len(block), block)
    for r1 in range(nrow):  # top-left anchored
        for q1 in range(ncol):
            block = [(r, q) for r in range(0, r1 + 1) for q in range(0, q1 + 1)]
            if block and _all_empty(block):
                if best is None or len(block) > best[0]:
                    best = (len(block), block)
    return best[1] if best is not None else None


def _place_legend_in_empty_space(fig, axes, cell_has_data, source_ax, nrow, ncol):
    handles, labels = source_ax.get_legend_handles_labels()
    if not handles:
        return
    block = _largest_corner_empty_block(cell_has_data, nrow, ncol)
    if not block:  # no empty corner -> top-right fallback
        fig.legend(handles, labels, loc="upper right",
                   prop={"weight": "bold", "size": 22}, framealpha=0.96)
        return
    positions = [axes[r][q].get_position() for r, q in block]
    x0 = min(p.x0 for p in positions); x1 = max(p.x1 for p in positions)
    y0 = min(p.y0 for p in positions); y1 = max(p.y1 for p in positions)
    leg = fig.legend(handles, labels, loc="center",
                     bbox_to_anchor=(0.5 * (x0 + x1), 0.5 * (y0 + y1)),
                     ncol=1, handlelength=2.2, handletextpad=0.7,
                     borderpad=0.9, labelspacing=0.55, framealpha=0.96,
                     edgecolor="0.35", prop={"weight": "bold", "size": 22})
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_linewidth(1.4)


def make_grid(pythia_runs, zeus_data, out,
              f2ln_jam=None, have_jam21=False, include_jam21=True):
    fig, axes = plt.subplots(NX_ZEUS, NQ_ZEUS,
                              figsize=(NQ_ZEUS * 4.4, NX_ZEUS * 4.0),
                              gridspec_kw=dict(wspace=0.05, hspace=0.08),
                              sharex=False, sharey=False)
    fig.patch.set_facecolor("white")

    jam_available = (include_jam21 and have_jam21 and f2ln_jam is not None)

    cell_has_data = {}
    for ix in range(NX_ZEUS):
        row = NX_ZEUS - 1 - ix
        for iq in range(NQ_ZEUS):
            cell_has_data[(row, iq)] = ((iq, ix) in zeus_data)

    # first data-bearing panel (reading order)
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
        x_nom, xlo, xhi = X_BINS_ZEUS[ix]
        for iq in range(NQ_ZEUS):
            ax   = axes[row, iq]
            cell = (iq, ix)

            if not cell_has_data[(row, iq)]:
                ax.set_visible(False)
                continue

            ax.set_facecolor("white")
            ax.set_ylim(0., 0.35)
            ax.set_xlim(XL_EDGES[0] - 0.01, XL_EDGES[-1] + 0.01)

            jam_arr = f2ln_jam[(iq, ix)] if jam_available else None
            for run in pythia_runs:
                if not run["have_pythia"]:
                    continue
                py_arr = run["f2ln_py"][(iq, ix)]
                if jam_arr is not None:
                    combined = py_arr + np.where(np.isfinite(jam_arr), jam_arr, 0.0)
                    curve_label = f"{run['label']} + JAM21"
                else:
                    combined = py_arr
                    curve_label = run["label"]
                ax.stairs(combined, XL_EDGES, color=run["color"], lw=1.7,
                          label=curve_label)

            xl_d, f2_d, e_d = zeus_data[(iq, ix)]
            ax.errorbar(xl_d, f2_d, yerr=e_d, fmt="o", color=C_DATA,
                        ms=4, lw=0.9, capsize=1.8, zorder=10,
                        markerfacecolor="white", label="ZEUS data")

            ax.tick_params(which="both", direction="in", top=True, right=True,
                           labelsize=10)
            bold_ticklabels(ax)  # bold + stroked x/y tick numbers
            ax.minorticks_on()

            if (row, iq) == legend_cell:
                legend_source_ax = ax

            # Q^2 title on the top-most visible panel of this column
            if row == first_visible_row.get(iq):
                _, q2lo, q2hi = Q2_BINS_ZEUS[iq]
                ax.set_title(_q2_label_bold(q2lo, q2hi), fontsize=13, pad=8,
                             fontweight="bold")
                _stroke(ax.title)  # extra-bold Q^2 label

            if iq == first_visible_col.get(row):
                ax.set_ylabel(_xb_label_bold(xlo, xhi), fontsize=12,
                              fontweight="bold")
                _stroke(ax.yaxis.label)  # extra-bold x_B label
                ax.tick_params(labelleft=True)
            else:
                ax.tick_params(labelleft=False)

            # x ticks only on the bottom-most visible panel of this column
            ax.tick_params(labelbottom=(row == last_visible_row.get(iq)))

    # one big, bold legend in the largest empty corner
    if legend_source_ax is not None:
        _place_legend_in_empty_space(fig, axes, cell_has_data,
                                     legend_source_ax, NX_ZEUS, NQ_ZEUS)

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
    fig.text(cx, uy0 - pad, r"$x_L$",
             ha="center", va="top", fontsize=24, fontweight="bold")
    fig.text(ux0 - pad, cy, r"$F_{2}^{LN}(x_B,\,Q^{2},\,x_L)$",
             ha="center", va="center", rotation=90,
             fontsize=24, fontweight="bold")

    for ext in ("png", "pdf"):
        fname = f"{out}_f2ln_grid_zeus.{ext}"
        fig.savefig(fname, dpi=180, bbox_inches="tight", facecolor="white")
        print(f"  -> {fname}")
    plt.close(fig)


def zeus_data_to_total_dsigma(zeus_data):
    dsig_sum = np.zeros(N_XL)
    err2_sum = np.zeros(N_XL)
    has_data = np.zeros(N_XL, dtype=bool)

    for (iq, ix), (xl_arr, f2_arr, err_arr) in zeus_data.items():
        q2_nom, q2lo, q2hi = Q2_BINS_ZEUS[iq]
        x_nom, xlo, xhi    = X_BINS_ZEUS[ix]
        dx, dQ2 = xhi - xlo, q2hi - q2lo

        y   = compute_y(q2_nom, x_nom)
        kin = 1.0 - y + 0.5 * y**2
        conv = dis_prefactor(x_nom, q2_nom) * kin * dx * dQ2

        for xl_v, f2_v, err_v in zip(xl_arr, f2_arr, err_arr):
            i = int(np.argmin(np.abs(XL_CTRS - xl_v)))
            dsig_sum[i] += f2_v * conv
            err2_sum[i] += (err_v * conv) ** 2
            has_data[i]  = True

    return has_data, dsig_sum, np.sqrt(err2_sum)


def make_zeus_total_plot(pythia_runs, zeus_data, out, legend_mode="full",
                         suffix="_dsigma_zeus_total"):
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for run in pythia_runs:
        if not run["have_pythia"]:
            continue
        lbl = run["label"] if legend_mode == "full" else "_nolegend_"
        ax.stairs(run["dsigma"], XL_EDGES, color=run["color"],
                   lw=run.get("lw", 2.4), ls=run.get("ls", "solid"),
                   label=lbl)

    has_data, dsig_tot, dsig_err = zeus_data_to_total_dsigma(zeus_data)
    if has_data.any():
        ax.errorbar(XL_CTRS[has_data], dsig_tot[has_data],
                     xerr=XL_XERR[has_data], yerr=dsig_err[has_data],
                     fmt="o", color="black", ms=6, lw=1.3, capsize=3,
                     zorder=10, label="ZEUS")

    y_candidates = [0.0]
    for run in pythia_runs:
        if run["have_pythia"]:
            y_candidates.append(np.nanmax(run["dsigma"]))
    if has_data.any():
        y_candidates.append(np.nanmax(dsig_tot[has_data] + dsig_err[has_data]))
    ymax = max(y_candidates)
    ax.set_ylim(0.0, ymax * 1.15)
    ax.set_xlim(XL_EDGES[0], XL_EDGES[-1])
    ax.set_xlabel(r"$\mathbf{x_L}$", fontsize=FS_AXLABEL, labelpad=6)
    ax.set_ylabel(r"$\mathbf{d\sigma/dx_L\ [nb]}$", fontsize=FS_AXLABEL, labelpad=6)
    ax.minorticks_on()
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.tick_params(which="major", direction="in", length=7, width=1.4,
                   top=True, right=True, pad=6, labelsize=FS_TICK)
    ax.tick_params(which="minor", direction="in", length=3.5, width=1.0,
                   top=True, right=True)
    bold_ticklabels(ax)  # bold + stroked x/y tick numbers

    leg = ax.legend(loc="upper right", prop={"weight": "bold", "size": FS_LEGEND},
                     framealpha=0.92, edgecolor="0.45")
    leg.get_frame().set_facecolor("white")

    for ext in ("png", "pdf"):
        fname = f"{out}{suffix}.{ext}"
        fig.savefig(fname, dpi=220, bbox_inches="tight", facecolor="white")
        print(f"  -> {fname}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zdat",     default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "zeus.dat"))
    parser.add_argument("--f2ptable", default=F2P_TABLE_FILE,
                         help="F2p_CT18NLO_values.txt (x, Q2, F2^p NLO)")
    parser.add_argument("--out",      default="compare_f2ln_zeus")
    parser.add_argument("--no-jam21", action="store_true",
                         help="skip the JAM21 pion-PDF theory curve")
    args = parser.parse_args()

    print(f"Loading ZEUS F_2LN data from {args.zdat} "
          f"(F2p sourced from {args.f2ptable}) ...")
    zeus_data = load_zeus_f2ln(args.zdat, f2p_table_file=args.f2ptable)
    print(f"  Loaded {len(zeus_data)} (Q2,x) cells "
          f"({sum(len(v[0]) for v in zeus_data.values())} data points)")

    # load and process the Pythia ROOT files
    pythia_runs = []
    for spec in PYTHIA_FILES:
        path, label, color = spec["path"], spec["label"], spec["color"]
        ls, lw = spec.get("ls", "solid"), spec.get("lw", 2.4)
        have_pythia = os.path.exists(path)
        f2ln_py, nevts, sigma_dis = {}, {}, 0.0
        counts = None
        if have_pythia:
            print(f"\nLoading Pythia ROOT file ({label}) from {path} ...")
            py_data = load_pythia(path)
            sigma_dis = py_data["sigma_dis"]
            counts = py_data["counts"]
            print(f"Computing F_2LN from Pythia dσ/dx_L ({label}, ZEUS cuts/bins) ...")
            f2ln_py, nevts = compute_all_pythia_f2ln_zeus(py_data)
        else:
            print(f"\nWARNING: ROOT file '{path}' ({label}) not found — "
                  f"this Pythia curve will be omitted.")
        pythia_runs.append(dict(label=label, color=color, ls=ls, lw=lw,
                                 have_pythia=have_pythia,
                                 f2ln_py=f2ln_py, nevts=nevts,
                                 sigma_dis=sigma_dis, counts=counts))

    have_jam21 = False
    f2ln_jam = {}
    if not args.no_jam21:
        print(f"\nLoading JAM21 pion PDF ({JAM21_PDF_NAME}) ...")
        jam_pdf = load_jam21_pdf()
        if jam_pdf is not None:
            have_jam21 = True
            print("Computing F_2LN = 2*flux(x_L)*F_2^pi_NLO(x_pi,Q2) "
                  "over the full ZEUS grid ...")
            f2ln_jam = compute_all_jam21_f2ln_zeus(jam_pdf)

    n_sets = len(PYTHIA_FILES)

    print(f"\nSaving ZEUS F_2LN comparison grid #1 "
          f"(all {n_sets} ROOT files, Pythia only, no JAM21) ...")
    make_grid(pythia_runs, zeus_data, f"{args.out}_root_only",
              f2ln_jam=f2ln_jam, have_jam21=have_jam21, include_jam21=False)

    print(f"\nSaving ZEUS F_2LN comparison grid #2 "
          f"(all {n_sets} ROOT files + JAM21 pion PDF, combined) ...")
    make_grid(pythia_runs, zeus_data, f"{args.out}_root_plus_jam21",
              f2ln_jam=f2ln_jam, have_jam21=have_jam21, include_jam21=True)

    print(f"\nComputing ZEUS-cuts total dsigma/dx_L for all {n_sets} "
          f"ROOT files, restricted to the {len(zeus_data)} data-bearing "
          f"(Q2,x_B) cells (no pion PDF) ...")
    cell_mask = zeus_cell_mask(zeus_data)
    zeus_total_runs = compute_all_pythia_dsigma_zeus_total(pythia_runs, cell_mask)
    print(f"\nSaving ZEUS-cuts total dsigma/dx_L comparison plot "
          f"(+ZEUS data) ...")
    make_zeus_total_plot(zeus_total_runs, zeus_data, args.out,
                         legend_mode="full", suffix="_dsigma_zeus_total")

    print(f"\nSaving ZEUS-cuts total dsigma/dx_L plot with a ZEUS-only "
          f"legend ...")
    make_zeus_total_plot(zeus_total_runs, zeus_data, args.out,
                         legend_mode="zeus_only",
                         suffix="_dsigma_zeus_total_zeusonly")

    print("\nDone.")


if __name__ == "__main__":
    main()
