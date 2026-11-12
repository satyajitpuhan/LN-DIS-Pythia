#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fit of the pion-flux cutoff Lambda to the two highest x_L bins of H1 and ZEUS.

Uses the chi^2 of Cao et al., PRD 103, 114014 (Eq. 17): the normalizations and the two
correlated ZEUS systematics are nuisance parameters. Writes lambda_fit_results.csv/.txt,
lambda_fit_table.tex and chi2_vs_lambda.pdf (Fig. 17).
"""

import csv
import math
import os
import warnings
from functools import lru_cache

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar

warnings.filterwarnings("ignore")

import lhapdf

lhapdf.setVerbosity(0)

try:
    import cupy as xp
    xp.zeros(1)
    GPU = True
except Exception:
    import numpy as xp
    GPU = False

# SETTINGS
HERE = os.path.dirname(os.path.abspath(__file__))
ZEUS_DAT = os.path.join(HERE, "..", "data", "zeus.dat")
F2P_TXT = os.path.join(HERE, "..", "data", "F2p_CT18NLO_values.txt")
H1_DAT = os.path.join(HERE, "..", "data", "h1.dat")
OUT_TXT = os.path.join(HERE, "..", "lambda_fit", "lambda_fit_results.txt")
OUT_CSV = os.path.join(HERE, "..", "lambda_fit", "lambda_fit_results.csv")
OUT_FIG = os.path.join(HERE, "..", "lambda_fit", "chi2_vs_lambda")

N_XL = 2                                                    # two highest x_L bins, each experiment
# True: separate nuisance shifts r1, r2 for EACH ZEUS x_L bin (4 parameters: r1, r2 at x_L = 0.85 and 0.94),
# each acting only on the points of its own bin.  False: one common r1, r2 for both bins.
ZEUS_R_PER_XL = False
DNORM_ZEUS = float(os.environ.get("DNORM_LN_ZEUS", 0.088))  # delta n, ZEUS
DNORM_H1 = float(os.environ.get("DNORM_LN_H1", 0.10))       # delta n, H1

# ZEUS NPB 637, 3, Table 1: x_L -> (acceptance unc. [%], energy-scale shift [%]
# for +2% scale, symmetrised (up-down)/2 with sign).  Same table as
# PionPDFextraction/extraction_corrected/extract_ln.py::ZEUS_TAB1.
ZEUS_TAB1 = {0.24: (2.0, -4.5), 0.31: (2.0, -4.5), 0.37: (2.5, -4.5),
             0.43: (3.0, -4.5), 0.49: (3.0, -4.5), 0.55: (3.5, -4.5),
             0.61: (4.0, -4.0), 0.67: (5.0, -4.0), 0.73: (5.0, -2.0),
             0.79: (5.5, 1.5), 0.85: (6.0, 6.5), 0.94: (6.0, 25.5)}

# Lambda values used for JAM21 (JAM18, Table I of the paper)
# and the Lambda window scanned in the fits (lower edges stay above m_pi where
# the monopole / Pauli-Villars forms need it).
FLUX_MODELS = [
    dict(model="s_exp",         lam_jam=1.31, rng=(0.50, 4.00), label=r"$s$-exp"),
    dict(model="t_exp",         lam_jam=0.58, rng=(0.20, 2.50), label=r"$t$-exp"),
    dict(model="monopole",      lam_jam=0.52, rng=(0.20, 2.50), label="monopole"),
    dict(model="regge",         lam_jam=0.78, rng=(0.20, 2.50), label="Regge"),
    dict(model="pauli_villars", lam_jam=0.25, rng=(0.15, 1.50), label="Pauli-Villars"),
]

# JAM21 LHAPDF set used in the paper (arXiv:2609.03368); this set, not
# JAM21PionPDFnlo_pT, reproduces the data/theory ratios of Figs. 5 and 7.
JAM21_SET = "JAM21PionPDFnlo"

PION_SETS = [
    dict(name="JAM21",   lhapdf=JAM21_SET,            color="navy",      ls="-"),
    dict(name="xFitter", lhapdf="xFitterPI_NLO_VAR",  color="crimson",   ls="--"),
    dict(name="GRV",     lhapdf="GRVPI1",             color="darkgreen", ls="-."),
]
# How the central PDF of each set is taken.  JAM21: pset.uncertainty(...).central
# over all replicas = the MEAN of replicas 1..N.  NOTE member 0 of the JAM21 sets is NOT that mean
# (differs by 2-20 %), so member 0 must not be used for JAM21.  F2 is linear in
# the PDFs and every JAM21 replica has the same alpha_s, so F2(mean PDF) is the
# replica mean of F2.  xFitter / GRV: member 0 (their central member).
CENTRAL = {"JAM21PionPDFnlo": "replica_mean", "JAM21PionPDFnlo_pT": "replica_mean"}
# xFitterPI_NLO_VAR: member 0 (its central member).  To use the mean of all its members instead,
# add  "xFitterPI_NLO_VAR": "replica_mean"  above.
# replica_mean: replicas sets -> mean of replicas 1..N (member 0 excluded, as LHAPDF's central);
#               variations sets (xFitter) -> mean of ALL members 0..N.
MEAN_CACHE = os.path.join(HERE, "..", "cache", "jam21_replica_mean_grids.npz")

N_SCAN = 161          # Lambda grid points per scan

# Fit plots (F2^LN vs x_pi, H1 and ZEUS points, one figure per flux model):
#   "JAM"    -> the H1+ZEUS fit at the JAM Lambda values used in the paper
#   "common" -> the H1+ZEUS fit with the common best-fit Lambda
FIT_PLOT_DIR = os.path.join(HERE, "..", "lambda_fit", "fit_plots")
PLOT_FITS = ("JAM",)              # add "common" for the common best-fit Lambda too
FIT_PLOT_FLUXES = ("s_exp",)      # which flux models get fit plots
THEORY_STYLE = dict(color="dimgray", ls="--", lw=2.0)   # unshifted theory curve
ZEUS_MARK = dict(marker="o", color="black", mfc="black", ms=5.0, label_name="ZEUS")
H1_MARK = dict(marker="s", color="darkorange", mfc="darkorange", ms=5.0, label_name="H1")
XPI_PLOT = np.logspace(-4.5, np.log10(0.98), 260)
X_PAD = 3.0          # fit plots: x range = data range padded by this factor
X_MIN_DECADES = 1.5  # ... but at least this many decades wide


def to_gpu(a):
    return xp.asarray(a)


def to_cpu(a):
    return xp.asnumpy(a) if GPU else np.asarray(a)


CF = 4.0 / 3.0
TR = 0.5
M_N = 0.9382720813
M_PI = 0.13957039
G_A = 1.267
F_PI = 0.093
PIDS_NEEDED = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 21]
N_QUAD = 200
_GL_N, _GL_W = np.polynomial.legendre.leggauss(N_QUAD)
GL_NODES = to_gpu(_GL_N)
GL_WEIGHTS = to_gpu(_GL_W)


class GpuPdfGrid:
    """Tabulate x f(x, Q^2) once on a fixed x grid (CPU, LHAPDF), interpolate on GPU.
    use_q2=True -> xfxQ2 (H1 machinery), False -> xfxQ (ZEUS machinery).
    """

    def __init__(self, pdf, q2val, x_fine, use_q2, arrays=None):
        self.x_fine = to_gpu(x_fine)
        if arrays is not None:
            self.grids = {pid: to_gpu(arrays[pid]) for pid in PIDS_NEEDED}
            return
        if use_q2:
            g = {pid: np.array([pdf.xfxQ2(pid, float(x), float(q2val)) for x in x_fine]) for pid in PIDS_NEEDED}
        else:
            qv = np.sqrt(q2val)
            g = {pid: np.array([pdf.xfxQ(pid, float(x), qv) for x in x_fine]) for pid in PIDS_NEEDED}
        self.grids = {pid: to_gpu(v) for pid, v in g.items()}

    def xfq(self, pid, xval):
        shape = xval.shape
        flat = xval.reshape(-1)
        out = xp.interp(flat, self.x_fine, self.grids[pid], left=0.0, right=0.0)
        out = xp.where((flat > 0.0) & (flat < 1.0), out, 0.0)
        return out.reshape(shape)


# ZEUS NLO F2
Z_FLAVOR_INFO = [(2, -2, (2.0 / 3.0) ** 2), (1, -1, (1.0 / 3.0) ** 2), (3, -3, (1.0 / 3.0) ** 2),
                 (4, -4, (2.0 / 3.0) ** 2), (5, -5, (1.0 / 3.0) ** 2)]
Z_CHARGE_SQ_SUM = sum(e2 for _, _, e2 in Z_FLAVOR_INFO)
Z_X_FINE = np.logspace(-6, np.log10(1.0 - 1e-8), 5000)


def f2_nlo_vec_zeus(pdf, grid, x_arr_cpu, q2val):
    x_arr = to_gpu(np.asarray(x_arr_cpu, dtype=np.float64))
    n_x = x_arr.shape[0]
    pref = float(pdf.alphasQ(float(np.sqrt(q2val)))) / (2.0 * np.pi)
    zmax = 1.0 - 1e-6
    half_width = (zmax - x_arr) / 2.0
    mid = (zmax + x_arr) / 2.0
    z = mid[:, None] + half_width[:, None] * GL_NODES[None, :]
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
    for pid_q, pid_qb, e2 in Z_FLAVOR_INFO:
        fx = grid.xfq(pid_q, x_arr) + grid.xfq(pid_qb, x_arr)
        f2_lo_val = f2_lo_val + e2 * fx
        fz = xp.where(valid_xo, grid.xfq(pid_q, xo) + grid.xfq(pid_qb, xo), 0.0)
        i1 = xp.sum(jac * w * xp.where(valid_xo, log_one_minus_z / one_minus_z * (fz - fx[:, None]), 0.0), axis=1)
        i2 = xp.sum(jac * w * xp.where(valid_xo, (fz - fx[:, None]) / one_minus_z, 0.0), axis=1)
        kernel3 = (3.0 + 2.0 * z - (1.0 + z) * log_one_minus_z - ((1.0 + z * z) / one_minus_z) * log_z)
        i3 = xp.sum(jac * w * xp.where(valid_xo, kernel3 * fz, 0.0), axis=1)
        cq = CF * (2.0 * i1 - (log_one_minus_x ** 2) * fx - 1.5 * i2 - 1.5 * fx * log_one_minus_x
                   + i3 - (np.pi ** 2 / 3.0 + 9.0 / 2.0) * fx)
        c2q_total = c2q_total + e2 * pref * cq

    fg = xp.where(valid_xo, grid.xfq(21, xo), 0.0)
    kernel_g = (z ** 2 + one_minus_z ** 2) * (log_one_minus_z - log_z) + 8.0 * z * one_minus_z - 1.0
    ig = xp.sum(jac * w * xp.where(valid_xo, 2.0 * TR * kernel_g * fg, 0.0), axis=1)
    f2 = f2_lo_val + c2q_total + pref * Z_CHARGE_SQ_SUM * ig
    f2 = xp.where((x_arr <= 0.0) | (x_arr >= 0.999999), 0.0, f2)
    return to_cpu(f2)


# H1 NLO F2_{pi-}
E2_u, E2_d, E2_s, E2_c, E2_b = (2 / 3) ** 2, (1 / 3) ** 2, (1 / 3) ** 2, (2 / 3) ** 2, (1 / 3) ** 2
H_CHARGE_SQ_SUM = E2_u + E2_d + E2_s + E2_c + E2_b
H_FLAVOR_MAP_PIMINUS = [(1, -1, E2_u), (2, -2, E2_d), (3, -3, E2_s), (4, -4, E2_c), (5, -5, E2_b)]
Cq2_DELTA = CF * (np.pi ** 2 / 3.0 - 9.0 / 2.0)
H_X_FINE = np.logspace(-6, np.log10(1.0 - 1e-9), 5000)


def _bdy_int_cpu(x):
    def integ_bdy(z):
        if z <= 0.0 or z >= 1.0:
            return 0.0
        return CF * (1.0 + z ** 2) * (np.log((1.0 - z) / z) - 0.75) / (1.0 - z)
    val, _ = quad(integ_bdy, 0.0, float(x), limit=60)
    return val


def f2pi_minus_vec_h1(pdf, grid, x_arr_cpu, q2val, bdy_int_arr_cpu):
    x_arr = to_gpu(np.asarray(x_arr_cpu, dtype=np.float64))
    bdy_int = to_gpu(np.asarray(bdy_int_arr_cpu, dtype=np.float64))
    n_x = x_arr.shape[0]
    pref = float(pdf.alphasQ(np.sqrt(float(q2val)))) / (2.0 * np.pi)
    zmax = 1.0 - 1e-6
    half_width = (zmax - x_arr) / 2.0
    mid = (zmax + x_arr) / 2.0
    z = mid[:, None] + half_width[:, None] * GL_NODES[None, :]
    w = GL_WEIGHTS[None, :]
    jac = half_width[:, None]
    xo = x_arr[:, None] / z
    valid_xo = xo < 1.0
    one_minus_z = 1.0 - z
    log_term = xp.log(xp.clip(one_minus_z, 1e-300, None) / xp.clip(z, 1e-300, None))
    cq2_singular = CF * (1.0 + z ** 2) * (log_term - 0.75)
    cq2_regular = CF * (1.5 + 2.0 * z)
    cg2_kernel = 2.0 * TR * ((z ** 2 + one_minus_z ** 2) * log_term - 1.0 + 8.0 * z * one_minus_z)

    f2_lo_val = xp.zeros(n_x, dtype=xp.float64)
    c2q_total = xp.zeros(n_x, dtype=xp.float64)
    for pid_q, pid_qb, e2 in H_FLAVOR_MAP_PIMINUS:
        xqq = grid.xfq(pid_q, x_arr) + grid.xfq(pid_qb, x_arr)
        f2_lo_val = f2_lo_val + e2 * xqq
        fz = xp.where(valid_xo, grid.xfq(pid_q, xo) + grid.xfq(pid_qb, xo), 0.0)
        plus_c = xp.sum(jac * w * xp.where(valid_xo, cq2_singular * (fz - xqq[:, None]) / one_minus_z, 0.0), axis=1)
        reg_c = xp.sum(jac * w * xp.where(valid_xo, cq2_regular * fz, 0.0), axis=1)
        c2q_total = c2q_total + e2 * pref * ((plus_c - xqq * bdy_int) + reg_c + Cq2_DELTA * xqq)

    fg = xp.where(valid_xo, grid.xfq(21, xo), 0.0)
    gluon_c = xp.sum(jac * w * xp.where(valid_xo, cg2_kernel * fg, 0.0), axis=1)
    f2 = f2_lo_val + c2q_total + pref * H_CHARGE_SQ_SUM * gluon_c
    f2 = xp.where((x_arr <= 0.0) | (x_arr >= 1.0), 0.0, f2)
    return to_cpu(f2)


# Sullivan flux, five regulators;
# only the k_perp^2 ceiling differs: ZEUS (0.65 x_L)^2, H1 0.04 GeV^2
ALPHA_PI_SLOPE = 1.0
H1_K2_MAX = 0.04


def d_piN(k2, xL):
    xb = 1.0 - xL
    return -(k2 + xb * xb * M_N * M_N + xL * M_PI * M_PI) / xL


def t_piN(k2, xL):
    xb = 1.0 - xL
    return -(k2 + xb * xb * M_N * M_N) / xL


def s_piN(k2, xL):
    xb = 1.0 - xL
    return (k2 + M_N * M_N) / xL + (k2 + M_PI * M_PI) / xb


def regulator(model, k2, xL, lam):
    xb = 1.0 - xL
    dv, tv, sv = d_piN(k2, xL), t_piN(k2, xL), s_piN(k2, xL)
    if model == "s_exp":
        return np.exp((M_N * M_N - sv) / lam ** 2)
    if model == "t_exp":
        return np.exp(dv / lam ** 2)
    if model == "monopole":
        return (lam ** 2 - M_PI ** 2) / (lam ** 2 - tv)
    if model == "regge":
        return (xb ** (-ALPHA_PI_SLOPE * tv)) * np.exp(dv / lam ** 2)
    if model == "pauli_villars":
        denom = lam ** 2 - tv
        if denom == 0.0:
            return 0.0
        arg = 1.0 - (dv * dv) / (denom * denom)
        return np.sqrt(arg) if arg > 0.0 else 0.0
    raise ValueError(model)


@lru_cache(maxsize=None)
def flux(x_l_val, model, lam, expt):
    if x_l_val <= 0.0 or x_l_val >= 1.0:
        return 0.0
    xb = 1.0 - x_l_val
    pref = (G_A * G_A * M_N * M_N) / ((4.0 * np.pi * F_PI) ** 2)

    def integrand(k2):
        dv = d_piN(k2, x_l_val)
        ff = regulator(model, k2, x_l_val, lam)
        return pref * xb * (k2 + xb * xb * M_N * M_N) / (x_l_val * x_l_val * dv * dv) * ff * ff

    k2max = (0.65 * x_l_val) ** 2 if expt == "ZEUS" else H1_K2_MAX
    val, _ = quad(integrand, 0.0, k2max, limit=220)
    return val


# data
def load_zeus_top(n_xl=N_XL):
    z = np.loadtxt(ZEUS_DAT)
    f2p_tab = np.loadtxt(F2P_TXT)
    if not (f2p_tab.shape[0] == z.shape[0]
            and np.allclose(f2p_tab[:, 0], z[:, 3], rtol=1e-6)
            and np.allclose(f2p_tab[:, 1], z[:, 2], rtol=1e-6)):
        raise RuntimeError("F2p_CT18NLO_values.txt rows do not match zeus.dat (x, Q2)")
    x_l, q2, x, r, dr = z[:, 1], z[:, 2], z[:, 3], z[:, 4], z[:, 5]
    f2p = f2p_tab[:, 2]
    keep = sorted(set(np.round(x_l, 2)))[-n_xl:]
    m = np.isin(np.round(x_l, 2), keep)
    x_l, q2, x, r, dr, f2p = np.round(x_l[m], 2), q2[m], x[m], r[m], dr[m], f2p[m]
    dxl = np.where(np.isclose(x_l, 0.24), 0.08, np.where(np.isclose(x_l, 0.94), 0.12, 0.06))
    d = (r / (100.0 * dxl)) * f2p
    alpha = (dr / (100.0 * dxl)) * f2p
    frac = np.array([ZEUS_TAB1[float(v)] for v in x_l]) / 100.0
    if ZEUS_R_PER_XL:                       # one (r1, r2) pair per x_L bin
        rows, labels = [], []
        for xb in keep:
            mb = np.isclose(x_l, xb)
            for k in range(2):
                b = np.zeros(len(d))
                b[mb] = frac[mb, k] * d[mb]
                rows.append(b)
                labels.append((float(xb), k))
        beta = np.array(rows)                                     # (4, N)
    else:                                   # one common (r1, r2)
        beta = (frac * d[:, None]).T                              # (2, N)
        labels = [(None, 0), (None, 1)]
    return dict(expt="ZEUS", x_l=x_l, q2=q2, x_pi=x / (1.0 - x_l), d=d, alpha=alpha,
                beta=beta, r_labels=labels, xl_bins=keep, dnorm=DNORM_ZEUS)


def zeus_r_by_bin(ev, Z):
    """{x_L: (r1, r2)} for every fitted ZEUS bin (the same pair in every bin if ZEUS_R_PER_XL is False)."""
    out = {}
    for xb in Z["xl_bins"]:
        r1 = r2 = 0.0
        for val, (bx, k) in zip(ev["r"], Z["r_labels"]):
            if bx is None or np.isclose(bx, float(xb)):
                if k == 0:
                    r1 = float(val)
                else:
                    r2 = float(val)
        out[round(float(xb), 2)] = (r1, r2)
    return out


def load_h1_top(n_xl=N_XL):
    rows = []
    with open(H1_DAT) as fh:
        for line in fh:
            t = line.replace("D", "E").replace("d", "e").split()
            if len(t) < 7 or line.strip().startswith("#"):
                continue
            q2, xl, xpi, y, st, sy = (float(t[0]), round(float(t[2]), 3), float(t[3]),
                                      float(t[4]), abs(float(t[5])), abs(float(t[6])))
            if 0.0 < xpi < 1.0:
                rows.append((q2, xl, xpi, y, st, sy))
    a = np.array(rows)
    keep = sorted(set(a[:, 1]))[-n_xl:]
    a = a[np.isin(a[:, 1], keep)]
    return dict(expt="H1", x_l=a[:, 1], q2=a[:, 0], x_pi=a[:, 2], d=a[:, 3],
                alpha=np.hypot(a[:, 4], a[:, 5]), beta=None, xl_bins=keep, dnorm=DNORM_H1)


_PDF_CACHE = {}
_BDY_PLOT = None
_MEAN = None          # replica-mean grids, key "set|q2|xQ2" -> (11, n_x) array
_MEMBERS = {}


def get_pdf(set_name):
    """Member 0 -- used for alpha_s(Q) always, and as the PDF for member-0 sets."""
    if set_name not in _PDF_CACHE:
        _PDF_CACHE[set_name] = lhapdf.mkPDF(set_name, 0)
    return _PDF_CACHE[set_name]


def _mean_arrays(set_name, q2v, x_fine, use_q2):
    """x f(x,Q^2) averaged over replicas 1..N on x_fine (disk-cached)."""
    global _MEAN
    if _MEAN is None:
        _MEAN = {}
        if os.path.exists(MEAN_CACHE):
            with np.load(MEAN_CACHE) as z:
                _MEAN = {k: z[k] for k in z.files}
    key = f"{set_name}|{float(q2v):.6g}|{'Q2' if use_q2 else 'Q'}|{len(x_fine)}|{x_fine[-1]:.12g}"
    if key not in _MEAN:
        if set_name not in _MEMBERS:
            _MEMBERS[set_name] = lhapdf.getPDFSet(set_name).mkPDFs()
        etype = lhapdf.getPDFSet(set_name).errorType
        mem = _MEMBERS[set_name][1:] if etype == "replicas" else _MEMBERS[set_name]
        acc = np.zeros((len(PIDS_NEEDED), len(x_fine)))
        qv = float(np.sqrt(q2v))
        for m in mem:
            for j, xv in enumerate(x_fine):
                d = m.xfxQ2(float(xv), float(q2v)) if use_q2 else m.xfxQ(float(xv), qv)
                for i, pid in enumerate(PIDS_NEEDED):
                    acc[i, j] += d[pid]
        _MEAN[key] = acc / len(mem)
        os.makedirs(os.path.dirname(MEAN_CACHE), exist_ok=True)
        np.savez(MEAN_CACHE, **_MEAN)
        print(f"  {set_name}: replica mean of {len(mem)} replicas at Q2={q2v:g} cached")
    arr = _MEAN[key]
    return {pid: arr[i] for i, pid in enumerate(PIDS_NEEDED)}


def make_grid(set_name, q2v, x_fine, use_q2):
    if CENTRAL.get(set_name) == "replica_mean":
        return GpuPdfGrid(None, q2v, x_fine, use_q2, arrays=_mean_arrays(set_name, q2v, x_fine, use_q2))
    return GpuPdfGrid(get_pdf(set_name), float(q2v), x_fine, use_q2)


def f2pi_curve(set_name, expt, q2v):
    """F2^pi on XPI_PLOT at one Q^2, with the experiment's own machinery."""
    global _BDY_PLOT
    pdf = get_pdf(set_name)
    if expt == "ZEUS":
        grid = make_grid(set_name, q2v, Z_X_FINE, use_q2=False)
        return f2_nlo_vec_zeus(pdf, grid, XPI_PLOT, float(q2v))
    if _BDY_PLOT is None:
        _BDY_PLOT = np.array([_bdy_int_cpu(v) for v in XPI_PLOT])
    grid = make_grid(set_name, q2v, H_X_FINE, use_q2=True)
    return f2pi_minus_vec_h1(pdf, grid, XPI_PLOT, float(q2v), _BDY_PLOT)


def f2pi_at_points(set_name, data):
    pdf = get_pdf(set_name)
    out = np.empty_like(data["x_pi"])
    for q2v in np.unique(data["q2"]):
        msk = data["q2"] == q2v
        xpi = data["x_pi"][msk]
        if data["expt"] == "ZEUS":
            grid = make_grid(set_name, q2v, Z_X_FINE, use_q2=False)
            out[msk] = f2_nlo_vec_zeus(pdf, grid, xpi, float(q2v))
        else:
            grid = make_grid(set_name, q2v, H_X_FINE, use_q2=True)
            bdy = np.array([_bdy_int_cpu(v) for v in xpi])
            out[msk] = f2pi_minus_vec_h1(pdf, grid, xpi, float(q2v), bdy)
    return out


# JAM Eq. (17) chi^2, r_k and n profiled
class JamChi2:
    def __init__(self, data):
        self.d = data["d"]
        self.alpha = data["alpha"]
        self.beta = data["beta"] if data["beta"] is not None else np.zeros((0, len(self.d)))
        self.dn = data["dnorm"]
        C = np.diag(self.alpha ** 2) + self.beta.T @ self.beta
        L = np.linalg.cholesky(C)
        self.W = np.linalg.solve(L, np.eye(len(L)))       # W^T W = C^-1
        self.Wd = self.W @ self.d
        self.A = self.beta / self.alpha
        self.K = self.beta.shape[0]

    def _tot(self, ln_n, Wt):
        n = math.exp(ln_n)
        res = self.Wd - Wt / n
        return float(res @ res) + ((1.0 - n) / self.dn) ** 2

    def profile_n(self, t):
        Wt = self.W @ t
        grid = np.linspace(math.log(0.05), math.log(20.0), 241)
        k = int(np.argmin([self._tot(g, Wt) for g in grid]))
        lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
        res = minimize_scalar(self._tot, bounds=(lo, hi), args=(Wt,), method="bounded",
                              options={"xatol": 1e-12})
        return math.exp(res.x), float(res.fun)

    def evaluate(self, t):
        n, chi2_tot = self.profile_n(t)
        u = (self.d - t / n) / self.alpha
        if self.K:
            r = np.linalg.solve(np.eye(self.K) + self.A @ self.A.T, self.A @ u)
            resid = (self.d - r @ self.beta - t / n) / self.alpha
        else:
            r = np.zeros(0)
            resid = u
        return dict(n=n, r=r, chi2=chi2_tot, chi2_pts=float(resid @ resid),
                    sum_r2=float(r @ r), pen_n=((1.0 - n) / self.dn) ** 2, resid=resid)


def theory(f2pi, data, model, lam):
    fl = np.array([flux(float(v), model, float(lam), data["expt"]) for v in data["x_l"]])
    return 2.0 * fl * f2pi


def fit_lambda(fun, rng):
    """Minimise a profiled chi^2(Lambda) on [rng]; Delta chi^2 = 1 interval."""
    lams = np.linspace(rng[0], rng[1], N_SCAN)
    c2 = np.array([fun(L) for L in lams])
    k = int(np.argmin(c2))
    a, b = lams[max(k - 1, 0)], lams[min(k + 1, N_SCAN - 1)]
    res = minimize_scalar(fun, bounds=(a, b), method="bounded", options={"xatol": 1e-6})
    lam_best, c2_min = float(res.x), float(res.fun)
    g = lambda L: fun(L) - (c2_min + 1.0)
    err_lo = err_hi = None
    j = np.where((lams < lam_best) & (c2 > c2_min + 1.0))[0]
    if len(j):
        err_lo = lam_best - brentq(g, lams[j[-1]], lam_best, xtol=1e-6)
    j = np.where((lams > lam_best) & (c2 > c2_min + 1.0))[0]
    if len(j):
        err_hi = brentq(g, lam_best, lams[j[0]], xtol=1e-6) - lam_best
    return dict(lam=lam_best, chi2=c2_min, err_lo=err_lo, err_hi=err_hi,
                at_edge=(k == 0 or k == N_SCAN - 1), scan_lam=lams, scan_chi2=c2)


def fmt_err(v):
    return "   open" if v is None else f"{v:7.3f}"


def per_bin(ev, data):
    return {xl: float(np.sum(ev["resid"][data["x_l"] == xl] ** 2)) for xl in data["xl_bins"]}


# Fit plot: F2^LN vs x_pi, ZEUS block (top) and H1 block (bottom)
def _odd_decade_fmt(v, pos):
    if v <= 0:
        return ""
    e = int(round(np.log10(v)))
    if not np.isclose(v, 10.0 ** e) or e % 2 == 0:
        return ""
    return rf"$\mathbf{{10^{{{e}}}}}$"


def column_xlim(D, q2v):
    """x_pi range of one Q^2 column (all x_L rows of that experiment), padded."""
    xs = D["x_pi"][np.isclose(D["q2"], q2v)]
    lo, hi = xs.min() / X_PAD, min(xs.max() * X_PAD, 1.0)
    if np.log10(hi / lo) < X_MIN_DECADES:
        lo = hi / 10.0 ** X_MIN_DECADES
    return lo, hi


def decade_fmt_inside(lo, hi, edge=0.12):
    """Label 10^n only if it sits away from the panel edges (panels touch)."""
    a, b = np.log10(lo), np.log10(hi)
    span = b - a

    def f(v, pos):
        if v <= 0:
            return ""
        e = np.log10(v)
        if abs(e - round(e)) > 1e-6 or e < a + edge * span or e > b - edge * span:
            return ""
        return rf"$\mathbf{{10^{{{int(round(e))}}}}}$"
    return f


def zeus_scale(r, xl):
    """s = 1 - r1 f1(x_L) - r2 f2(x_L): the factor the ZEUS correlated shift
    puts on the data in x_L bin xl (beta_k,i = f_k(x_L) d_i).
    """
    f1, f2 = np.array(ZEUS_TAB1[float(xl)]) / 100.0
    return 1.0 - r[0] * f1 - r[1] * f2


def make_fit_figure(path_base, fm, Z, H, curves, entries, header):
    """entries: list of dict(name, color, ls, lam, evz, evh) -- one per pion PDF.
    curves[(pdf_name, expt, q2)] = F2^pi on XPI_PLOT.
    Solid curve  = 2 f F2^pi / (n_e s_xL)  -- the theory the data are fitted to
                   (s_xL = ZEUS correlated shift, 1 for H1).
    Dotted curve = 2 f F2^pi                -- raw, no n_e, no shift.
    """
    from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter
    from matplotlib.lines import Line2D

    blocks = []
    for D, mk in ((Z, ZEUS_MARK), (H, H1_MARK)):
        q2s = sorted(set(np.round(D["q2"], 4)))
        for xl in sorted(D["xl_bins"], reverse=True):
            blocks.append(dict(D=D, mk=mk, xl=float(xl), q2s=q2s))
    ncol = max(len(b["q2s"]) for b in blocks)

    P, ML, MR, MT, MB, GAP = 2.2, 1.75, 0.15, 2.05, 0.75, 0.62
    W = ML + ncol * P + MR
    Hgt = MT + 4 * P + GAP + MB
    fig = plt.figure(figsize=(W, Hgt))

    for bi, b in enumerate(blocks):
        D, expt, xl = b["D"], b["D"]["expt"], b["xl"]
        top_in = MT + bi * P + (GAP if bi >= 2 else 0.0)
        axes_row = []
        ymax = 0.0
        for D2 in (D,):
            rowmask = np.isclose(D2["x_l"], xl)
            ymax = float(np.max(D2["d"][rowmask] + D2["alpha"][rowmask]))
        for ci, q2v in enumerate(b["q2s"]):
            xlo, xhi = column_xlim(D, q2v)
            ax = fig.add_axes([(ML + ci * P) / W, 1.0 - (top_in + P) / Hgt, P / W, P / Hgt])
            axes_row.append(ax)
            m = (np.isclose(D["x_l"], xl)) & (np.isclose(D["q2"], q2v))
            xp_, d_, a_ = D["x_pi"][m], D["d"][m], D["alpha"][m]
            for e in entries:
                ev = e["evz"] if expt == "ZEUS" else e["evh"]
                s = zeus_scale(zeus_r_by_bin(ev, Z)[round(xl, 2)], xl) if expt == "ZEUS" else 1.0
                f2 = curves[(e["name"], expt, float(q2v))]
                raw = 2.0 * flux(xl, fm["model"], float(e["lam"]), expt) * f2
                fit = raw / (ev["n"] * s) if s > 0.05 else np.full_like(raw, np.nan)
                if len(entries) == 1:
                    ax.plot(XPI_PLOT, raw, THEORY_STYLE["ls"], color=THEORY_STYLE["color"],
                            lw=THEORY_STYLE["lw"])
                    ax.plot(XPI_PLOT, fit, "-", color=e["color"], lw=2.3)
                else:
                    ax.plot(XPI_PLOT, fit, e["ls"], color=e["color"], lw=1.8)
                    ax.plot(XPI_PLOT, raw, ":", color=e["color"], lw=1.1, alpha=0.9)
            if len(xp_):
                mk = b["mk"]
                ax.errorbar(xp_, d_, yerr=a_, fmt=mk["marker"], color=mk["color"], mfc=mk["mfc"],
                            ms=mk["ms"], capsize=2, elinewidth=1.1, zorder=5)
            ax.set_xscale("log")
            ax.set_xlim(xlo, xhi)
            ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
            ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.xaxis.set_major_formatter(FuncFormatter(decade_fmt_inside(xlo, xhi)))
            ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=12)
            ax.text(0.05, 0.94, rf"$\mathbf{{Q^2={q2v:g}}}$", transform=ax.transAxes,
                    ha="left", va="top", fontsize=11, fontweight="bold", zorder=10,
                    bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85))
            last_in_block = bi in (1, 3)
            ax.tick_params(labelbottom=last_in_block)
            if ci != 0:
                ax.tick_params(labelleft=False)
        for ax in axes_row:
            ax.set_ylim(0.0, 1.30 * ymax if ymax > 0 else 1.0)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="upper"))
            for t in ax.get_yticklabels() + ax.get_xticklabels():
                t.set_fontweight("bold")
        axes_row[0].text(-0.36, 0.5, rf"$\mathbf{{{expt}}}$" + "\n" + rf"$\mathbf{{x_L={xl:g}}}$",
                         transform=axes_row[0].transAxes, rotation=90, ha="center", va="center",
                         fontsize=15, fontweight="bold")

    fig.text((ML + ncol * P / 2) / W, 0.28 / Hgt, r"$\mathbf{x_\pi}$", ha="center", va="center", fontsize=22)
    fig.text(0.32 / W, 1.0 - (MT + 2 * P + GAP / 2) / Hgt, r"$\mathbf{F_2^{LN(3)}}$",
             rotation=90, ha="center", va="center", fontsize=22)

    handles = []
    if len(entries) == 1:
        e = entries[0]
        ez, eh = e["evz"], e["evh"]
        ntot = len(Z["d"]) + len(H["d"])
        handles.append(Line2D([], [], color=THEORY_STYLE["color"], ls=THEORY_STYLE["ls"], lw=2.2,
                              label=rf"{e['name']} Theory:  $2f_{{\pi N}}F_2^\pi$  (no $n_e$, no shift)"))
        handles.append(Line2D([], [], color=e["color"], ls="-", lw=2.6,
                              label=rf"{e['name']} Shifted:  $2f_{{\pi N}}F_2^\pi/(n_e\,s)$,  "
                                    rf"ZEUS $s=1-r_1\beta_1-r_2\beta_2$, H1 $s=1$"))
        rb = zeus_r_by_bin(ez, Z)
        rtxt = (",  ".join(rf"$r_{{1,2}}({xb:g})$={rb[xb][0]:+.2f}, {rb[xb][1]:+.2f}" for xb in rb)
                if ZEUS_R_PER_XL else
                rf"$r_1$={next(iter(rb.values()))[0]:+.2f},  $r_2$={next(iter(rb.values()))[1]:+.2f}")
        handles.append(Line2D([], [], ls="none", label=(
            rf"$\Lambda$={e['lam']:.2f} GeV,  $n_Z$={ez['n']:.3f},  {rtxt},  $n_{{H1}}$={eh['n']:.3f}")))
        handles.append(Line2D([], [], ls="none", label=(
            rf"$\chi^2_Z$={ez['chi2']:.1f} (N={len(Z['d'])}),  $\chi^2_{{H1}}$={eh['chi2']:.1f} (N={len(H['d'])}),  "
            rf"$\chi^2$/N={(ez['chi2'] + eh['chi2']) / ntot:.2f}")))
    for e in (entries if len(entries) > 1 else []):
        ez, eh = e["evz"], e["evh"]
        tot = ez["chi2"] + eh["chi2"]
        rb = zeus_r_by_bin(ez, Z)
        rtxt = (",  ".join(rf"$r_{{1,2}}({xb:g})$={rb[xb][0]:+.2f}, {rb[xb][1]:+.2f}" for xb in rb)
                if ZEUS_R_PER_XL else
                rf"$r_1$={next(iter(rb.values()))[0]:+.2f},  $r_2$={next(iter(rb.values()))[1]:+.2f}")
        handles.append(Line2D([], [], color=e["color"], ls=e["ls"], lw=2.2, label=(
            rf"{e['name']}: $\Lambda$={e['lam']:.2f},  $n_Z$={ez['n']:.3f},  {rtxt},  $n_{{H1}}$={eh['n']:.3f},  "
            rf"$\chi^2$/N={tot / (len(Z['d']) + len(H['d'])):.2f}")))
    handles.append(Line2D([], [], color="black", marker="o", ls="none", ms=6,
                          label=rf"ZEUS data, $x_L$ = {', '.join(f'{v:g}' for v in Z['xl_bins'])} (stat)"))
    handles.append(Line2D([], [], color="darkorange", marker="s", ls="none", ms=6,
                          label=rf"H1 data, $x_L$ = {', '.join(f'{v:g}' for v in H['xl_bins'])} (stat$\oplus$sys)"))
    if len(entries) > 1:
        handles.append(Line2D([], [], color="grey", ls="-", lw=2.0,
                              label=r"solid: fitted theory $2fF_2^\pi/(n_e\,s)$,  ZEUS $s = 1-r_1\beta_1-r_2\beta_2$"))
        handles.append(Line2D([], [], color="grey", ls=":", lw=1.6,
                              label=r"dotted: raw $2fF_2^\pi$ (no $n_e$, no shift)"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0 - 0.08 / Hgt), ncol=2,
               handlelength=3.0,
               frameon=True, edgecolor="black", fancybox=False, prop={"weight": "bold", "size": 11.5})
    fig.text(ML / W, 1.0 - (MT - 0.12) / Hgt, header, ha="left", va="bottom", fontsize=13, fontweight="bold")

    for ext in ("png", "pdf"):
        fig.savefig(f"{path_base}.{ext}", dpi=220)
    plt.close(fig)


# chi^2(Lambda) figure: H1+ZEUS only, paper (Ivan) panel style + header box
CHI2_PDF_STYLE = {"JAM21": ("#0a7cff", "-", 0.0), "xFitter": ("crimson", "--", 0.0),
                  "GRV": ("forestgreen", ":", 1.4)}          # colour, ls, extra lw (per.py)
CHI2_FLUX_TITLE = {"s_exp": r"$s$-exp", "t_exp": r"$t$-exp", "monopole": "monopole",
                   "regge": "Regge", "pauli_villars": "Pauli-Villars"}


def make_chi2_figure(fitC, fixedC, Z, H):
    """chi^2/N (H1+ZEUS) vs Lambda, one panel per flux, no gaps, no header.
    JAM21: profiled chi^2/N curve, star = its Table 1 Lambda (JAM's own value).
    xFitter, GRV: profiled chi^2/N curves, best fit marked.  Lambda values written in each panel.
    Uniform x ticks (4-step frame, ticks at 1/4, 1/2, 3/4); the tightest y range (from 6 up) on which
    every Lambda label can be placed without touching a curve, marker, the legend or the frame.
    """
    from matplotlib.ticker import FuncFormatter, AutoMinorLocator
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif", "mathtext.default": "bfit",
        "font.weight": "bold", "axes.labelweight": "bold", "axes.linewidth": 1.4,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "xtick.labelsize": 17, "ytick.labelsize": 17,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "legend.frameon": True, "legend.edgecolor": "black", "legend.fancybox": False,
        "legend.framealpha": 1.0, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })
    PANEL_IN, MARG_L, MARG_R, MARG_B, MARG_T = 2.7, 1.25, 0.18, 1.15, 0.65
    FS_AXLABEL, FS_TITLE, FS_LEGEND = 26, 23, 12
    FITTED = [p for p in PION_SETS if p["name"] != "JAM21"]
    NTOT = len(Z["d"]) + len(H["d"])
    nM = len(FLUX_MODELS)
    fw = nM * PANEL_IN + MARG_L + MARG_R
    fh = PANEL_IN + MARG_B + MARG_T

    def jam_point(fm):
        e = fixedC[("JAM21", fm["model"])]
        return (e["evz"]["chi2"] + e["evh"]["chi2"]) / NTOT

    def bold_num(v, pos):
        return rf"$\mathbf{{{v:g}}}$"

    def bold_2f(v, pos):
        return rf"$\mathbf{{{v:.2f}}}$"

    def uniform_frame(smin, smax, lo_bound):
        """[xlo, xlo + 4 step] covering [smin, smax]; never starts below the scan start."""
        smin = max(smin, lo_bound)
        k = int(np.floor(np.log10((smax - smin) / 4.0)))
        for _ in range(6):
            for m in (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0):
                step = m * 10.0 ** k
                xlo = np.floor(smin / step + 1e-9) * step
                if xlo < lo_bound - 1e-12:
                    alt = np.ceil(lo_bound / step - 1e-9) * step
                    if alt <= smin + 1e-9:
                        xlo = alt
                if xlo + 4.0 * step >= smax - 1e-12:
                    return xlo, step
            k += 1
        raise RuntimeError("no tick frame found")

    def windows():
        out = []
        for c, fm in enumerate(FLUX_MODELS):
            keys = [fm["lam_jam"], fitC[("JAM21", fm["model"])]["lam"]] + \
                   [fitC[(p["name"], fm["model"])]["lam"] for p in FITTED]
            kmin, kmax = min(keys), max(keys)
            m = max(0.35 * (kmax - kmin), 0.25 * kmax)
            smin, smax = kmin - m, kmax + m
            out.append(uniform_frame(smin, smax, fm["rng"][0]))
        return out

    FRAMES = windows()

    YLO = -1.0                                   # y axis starts a little below zero (room for the Lambda row)

    def build(YMAX):
        fig = plt.figure(figsize=(fw, fh))
        axes = []
        for c, fm in enumerate(FLUX_MODELS):
            ax = fig.add_axes([(MARG_L + c * PANEL_IN) / fw, MARG_B / fh, PANEL_IN / fw, PANEL_IN / fh])
            axes.append(ax)
            for ps in PION_SETS:
                col, ls, extra = CHI2_PDF_STYLE[ps["name"]]
                ft = fitC[(ps["name"], fm["model"])]
                ax.plot(ft["scan_lam"], ft["scan_chi2"] / NTOT, ls=ls, color=col, lw=2.4 + extra, zorder=3)
                if ps["name"] != "JAM21":
                    ax.plot([ft["lam"]], [ft["chi2"] / NTOT], "o", color=col, ms=8, mec="black", mew=0.8,
                            zorder=5)
            ax.plot([fm["lam_jam"]], [jam_point(fm)], marker="*", ls="none", color=CHI2_PDF_STYLE["JAM21"][0],
                    ms=20, mec="black", mew=0.9, zorder=6)
            xlo, step = FRAMES[c]
            ax.set_xlim(xlo, xlo + 4.0 * step)
            ax.set_xticks([xlo + step, xlo + 2.0 * step, xlo + 3.0 * step])
            ax.set_ylim(YLO, YMAX)
            ax.set_yticks(list(np.arange(0.0, YMAX, 1.0)))   # ends dropped (panels touch)
            ax.xaxis.set_minor_locator(AutoMinorLocator(5))
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.xaxis.set_major_formatter(FuncFormatter(bold_2f))
            ax.yaxis.set_major_formatter(FuncFormatter(bold_num))
            ax.tick_params(which="both", top=True, right=True)
            ax.tick_params(which="major", length=7, width=1.1, pad=5)
            ax.tick_params(which="minor", length=3.5, width=0.9)
            if c != 0:
                ax.tick_params(labelleft=False)
            ax.set_title(CHI2_FLUX_TITLE[fm["model"]], fontsize=FS_TITLE, fontweight="bold", pad=8)

        axes[0].set_ylabel(r"$\chi^2/N$ (H1+ZEUS)", fontsize=FS_AXLABEL - 5, fontweight="bold", labelpad=6)
        fig.text((MARG_L + 0.5 * nM * PANEL_IN) / fw, 0.16 / fh, r"$\Lambda$ [GeV]",
                 ha="center", va="bottom", fontsize=FS_AXLABEL, fontweight="bold")

        hs = [Line2D([0], [0], color=CHI2_PDF_STYLE["JAM21"][0], ls="-", lw=2.4, label="JAM21")]
        hs += [Line2D([0], [0], color=CHI2_PDF_STYLE[p["name"]][0], ls=CHI2_PDF_STYLE[p["name"]][1],
                      lw=2.2 + CHI2_PDF_STYLE[p["name"]][2], marker="o", ms=7, mec="black", mew=0.8,
                      label=p["name"]) for p in FITTED]
        hs.append(Line2D([0], [0], marker="*", ls="none", color=CHI2_PDF_STYLE["JAM21"][0], ms=15,
                         mec="black", mew=0.8, label="Table 1"))
        leg = axes[0].legend(handles=hs, loc="upper left", ncol=2, fontsize=FS_LEGEND - 1, handlelength=1.9,
                             borderpad=0.35, labelspacing=0.3, columnspacing=0.8, handletextpad=0.5)
        for t in leg.get_texts():
            t.set_fontweight("bold")

        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        lb = leg.get_window_extent(rend)

        def collides(ax, texts):
            ab = ax.get_window_extent(rend)
            for t in texts:
                tb = t.get_window_extent(rend).expanded(1.04, 1.10)
                if tb.x0 < ab.x0 or tb.x1 > ab.x1 or tb.y1 > ab.y1 or tb.y0 < ab.y0:
                    return "leaves the panel"
                if ax is axes[0] and tb.overlaps(lb):
                    return "touches the legend"
                for ln in ax.lines:
                    xy = ln.get_xydata()
                    if len(xy) == 1:                          # marker
                        px = ax.transData.transform(xy)[0]
                        r = ln.get_markersize() * fig.dpi / 72.0 * 0.6
                        if tb.x0 - r < px[0] < tb.x1 + r and tb.y0 - r < px[1] < tb.y1 + r:
                            return "touches a marker"
                        continue
                    n = 20 * len(xy)
                    xs = np.interp(np.linspace(0, len(xy) - 1, n), np.arange(len(xy)), xy[:, 0])
                    ys = np.interp(np.linspace(0, len(xy) - 1, n), np.arange(len(xy)), xy[:, 1])
                    pts = ax.transData.transform(np.column_stack([xs, ys]))
                    if np.any((pts[:, 0] > tb.x0) & (pts[:, 0] < tb.x1) & (pts[:, 1] > tb.y0)
                              & (pts[:, 1] < tb.y1)):
                        return "touches a curve"
            return None

        from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea

        def lam_row(ax, fm):
            """One row at the bottom middle:  Lambda = a, b, c  (JAM21 Table 1, xFitter, GRV), values in colour."""
            vals = [("JAM21", f"{fm['lam_jam']:.2f}")] + \
                   [(p["name"], f"{fitC[(p['name'], fm['model'])]['lam']:.2f}") for p in FITTED]
            prop = dict(fontsize=12.5, fontweight="bold")
            nums = []
            for k, (nm, v) in enumerate(vals):
                nums.append(TextArea(v, textprops=dict(color=CHI2_PDF_STYLE[nm][0], **prop)))
                if k < len(vals) - 1:
                    nums.append(TextArea(",", textprops=dict(color="black", **prop)))
            # comma glued to the number before it, one space after it
            groups, k = [], 0
            while k < len(nums):
                grp = [nums[k]] + ([nums[k + 1]] if k + 1 < len(nums) else [])
                groups.append(HPacker(children=grp, align="baseline", pad=0, sep=0))
                k += 2
            row = HPacker(children=[TextArea(r"$\Lambda$ =", textprops=dict(color="black", **prop))] + groups,
                          align="baseline", pad=0, sep=5)
            box = AnchoredOffsetbox(loc="lower center", child=row, frameon=False, bbox_to_anchor=(0.5, 0.035),
                                    bbox_transform=ax.transAxes, borderpad=0.0, pad=0.0)
            box.set_zorder(10)
            ax.add_artist(box)
            return box

        bad = []
        # the legend must not hide any curve or marker of the first panel
        for ln in axes[0].lines:
            xy = ln.get_xydata()
            n = max(1, 20 * len(xy))
            xs = np.interp(np.linspace(0, len(xy) - 1, n), np.arange(len(xy)), xy[:, 0])
            ys = np.interp(np.linspace(0, len(xy) - 1, n), np.arange(len(xy)), xy[:, 1])
            pts = axes[0].transData.transform(np.column_stack([xs, ys]))
            if np.any((pts[:, 0] > lb.x0) & (pts[:, 0] < lb.x1) & (pts[:, 1] > lb.y0) & (pts[:, 1] < lb.y1)):
                bad.append("legend hides a curve in the first panel")
                break
        for c, fm in enumerate(FLUX_MODELS):
            box = lam_row(axes[c], fm)
            fig.canvas.draw()
            why = collides(axes[c], [box])
            if why is not None:
                bad.append(f"{fm['model']} ({why})")
        return fig, bad

    ymin_try = float(np.ceil(max(fitC[(p["name"], fm["model"])]["chi2"] / NTOT
                                 for fm in FLUX_MODELS for p in FITTED) + 0.5))
    for YMAX in (ymin_try, ymin_try + 1.0, ymin_try + 2.0):
        fig, bad = build(YMAX)
        if not bad:
            break
        plt.close(fig)
        print(f"  [chi2 plot] y up to {YMAX:g}: no clean label spot in {bad}; trying a larger y range")
    if bad:
        print(f"  [chi2 plot] WARNING: labels could not be placed cleanly in {bad}")
    else:
        print(f"  [chi2 plot] y range {YLO:g} to {YMAX:g}; all Lambda rows placed cleanly")
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT_FIG}.{ext}", dpi=300, facecolor="white")
    plt.close(fig)


def make_tex_table(fitC, fixedC, Z, H):
    """Small LaTeX table (and its PDF, cropped to the table) with the values of the chi^2 figure."""
    import subprocess
    import tempfile
    nZ, nH = len(Z["d"]), len(H["d"])
    zb = Z["xl_bins"]

    def sh(r, xl):
        return r[0] * ZEUS_TAB1[float(xl)][0] + r[1] * ZEUS_TAB1[float(xl)][1]

    def row(name, lam_txt, ez, eh):
        tot = ez["chi2"] + eh["chi2"]
        rb = zeus_r_by_bin(ez, Z)
        r0, r1_ = rb[round(float(zb[0]), 2)], rb[round(float(zb[1]), 2)]
        rcells = (f"${r0[0]:+.2f}$ & ${r0[1]:+.2f}$ & ${r1_[0]:+.2f}$ & ${r1_[1]:+.2f}$" if ZEUS_R_PER_XL
                  else f"${r0[0]:+.2f}$ & ${r0[1]:+.2f}$")
        return (f" & {name} & {lam_txt} & {rcells} & "
                f"${sh(r0, zb[0]):+.1f}$ & ${sh(r1_, zb[1]):+.1f}$ & "
                f"{ez['n']:.3f} & {eh['n']:.3f} & {tot:.1f} & {tot / (nZ + nH):.2f} \\\\")

    NC = 13 if ZEUS_R_PER_XL else 11
    flux_tex = {"s_exp": "$s$-exp", "t_exp": "$t$-exp", "monopole": "Monopole",
                "regge": "Regge", "pauli_villars": "PV"}
    L = [r"\documentclass[11pt]{article}",
         r"\usepackage[paperwidth=30cm,paperheight=20cm,margin=1cm]{geometry}",
         r"\usepackage{booktabs,multirow,amsmath}",
         r"\pagestyle{empty}",
         r"\begin{document}",
         (r"\begin{tabular}{llccccccccccc}" if ZEUS_R_PER_XL else r"\begin{tabular}{llccccccccc}"),
         r"\toprule",
         (rf"\multicolumn{{{NC}}}{{l}}{{H1+ZEUS, JAM Eq.~(17) $\chi^2$, $N = {nZ + nH}$.\quad "
          rf"$\delta n$: H1 = {DNORM_H1:.2f}, ZEUS = {DNORM_ZEUS:.3f}.}} \\"),
         (rf"\multicolumn{{{NC}}}{{l}}{{$x_L$: H1 = {H['xl_bins'][0]:g}, {H['xl_bins'][1]:g} "
          rf"({nH} pts, $\alpha$ = stat$\oplus$sys);\quad ZEUS = {zb[0]:g}, {zb[1]:g} ({nZ} pts, $\alpha$ = stat).}} \\"),
         (rf"\multicolumn{{{NC}}}{{l}}{{ZEUS $\beta_k$ (fraction of $d_i$):\ $x_L={zb[0]:g}$: "
          rf"$\beta_1={ZEUS_TAB1[float(zb[0])][0]:.1f}\%$, $\beta_2={ZEUS_TAB1[float(zb[0])][1]:+.1f}\%$;\quad "
          rf"$x_L={zb[1]:g}$: $\beta_1={ZEUS_TAB1[float(zb[1])][0]:.1f}\%$, "
          rf"$\beta_2={ZEUS_TAB1[float(zb[1])][1]:+.1f}\%$.}} \\"),
         ((r"\multicolumn{13}{l}{$r_1$, $r_2$ fitted separately in each ZEUS $x_L$ bin;\quad "
           r"shift$(x_L)=r_1(x_L)\beta_1+r_2(x_L)\beta_2$ \ (theory shifted by $+$shift$\times d_i$).} \\")
          if ZEUS_R_PER_XL else
          (r"\multicolumn{11}{l}{$r_1$, $r_2$ common to both ZEUS $x_L$ bins;\quad "
           r"shift$(x_L)=r_1\beta_1+r_2\beta_2$ \ (theory shifted by $+$shift$\times d_i$).} \\")),
         r"\midrule"]
    if ZEUS_R_PER_XL:
        L += [(rf" & & & \multicolumn{{2}}{{c}}{{$x_L={zb[0]:g}$}} & \multicolumn{{2}}{{c}}{{$x_L={zb[1]:g}$}} "
               r"& & & & & & \\"),
              r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
              (rf"Flux & PDF & $\Lambda$ [GeV] & $r_1$ & $r_2$ & $r_1$ & $r_2$ & shift({zb[0]:g}) [\%] & "
               rf"shift({zb[1]:g}) [\%] & $n_{{\rm ZEUS}}$ & $n_{{\rm H1}}$ & $\chi^2$ & $\chi^2/N$ \\")]
    else:
        L += [(rf"Flux & PDF & $\Lambda$ [GeV] & $r_1$ & $r_2$ & shift({zb[0]:g}) [\%] & "
               rf"shift({zb[1]:g}) [\%] & $n_{{\rm ZEUS}}$ & $n_{{\rm H1}}$ & $\chi^2$ & $\chi^2/N$ \\")]
    L += [r"\midrule"]
    for i, fm in enumerate(FLUX_MODELS):
        mdl = fm["model"]
        fj = fixedC[("JAM21", mdl)]
        rows = [row("JAM21", f"{fm['lam_jam']:.2f} (Table 1)", fj["evz"], fj["evh"])]
        for nm in ("xFitter", "GRV"):
            ft = fitC[(nm, mdl)]
            lam_txt = (f"${ft['lam']:.3f}^{{+{ft['err_hi']:.3f}}}_{{-{ft['err_lo']:.3f}}}$"
                       if ft["err_hi"] is not None and ft["err_lo"] is not None else f"{ft['lam']:.3f}")
            rows.append(row(nm, lam_txt, ft["evz"], ft["evh"]))
        rows[0] = rf"\multirow{{3}}{{*}}{{{flux_tex[mdl]}}}" + rows[0]
        L += rows
        L.append(r"\midrule" if i < len(FLUX_MODELS) - 1 else r"\bottomrule")
    L += [r"\end{tabular}", r"\end{document}"]
    tex = "\n".join(L) + "\n"
    tex_path = os.path.join(HERE, "..", "lambda_fit", "lambda_fit_table.tex")
    pdf_path = os.path.join(HERE, "..", "lambda_fit", "lambda_fit_table.pdf")
    with open(tex_path, "w") as fh:
        fh.write(tex)
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                            f"-output-directory={tmp}", tex_path], capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:])
            raise RuntimeError("pdflatex failed")
        try:                                                # crop the page to the table
            import fitz
            doc = fitz.open(os.path.join(tmp, "lambda_fit_table.pdf"))
            pg = doc[0]
            bb = None
            for b in pg.get_text("blocks"):
                rb = fitz.Rect(b[:4])
                bb = rb if bb is None else bb | rb
            for d in pg.get_drawings():
                bb = d["rect"] if bb is None else bb | d["rect"]
            pg.set_cropbox((bb + (-8, -8, 8, 8)) & pg.rect)
            doc.save(pdf_path)
        except Exception:
            import shutil
            shutil.copy(os.path.join(tmp, "lambda_fit_table.pdf"), pdf_path)
    print(f"wrote {tex_path}\nwrote {pdf_path}")


def main():
    print(f"[lambda_fit] backend: {'CuPy (GPU)' if GPU else 'NumPy (CPU)'}")
    Z = load_zeus_top()
    H = load_h1_top()
    chiZ, chiH = JamChi2(Z), JamChi2(H)
    nZ, nH = len(Z["d"]), len(H["d"])
    print(f"ZEUS: {nZ} points, x_L {Z['xl_bins']}, dn = {DNORM_ZEUS}")
    print(f"H1  : {nH} points, x_L {H['xl_bins']}, dn = {DNORM_H1}")

    fZ, fH = {}, {}
    for ps in PION_SETS:
        how = "replica mean" if CENTRAL.get(ps["lhapdf"]) == "replica_mean" else "member 0"
        print(f"NLO F2^pi at the data points: {ps['name']} ({ps['lhapdf']}, {how})")
        fZ[ps["name"]] = f2pi_at_points(ps["lhapdf"], Z)
        fH[ps["name"]] = f2pi_at_points(ps["lhapdf"], H)

    fixedZ, fitZ, fitH, fitC, fixedC = {}, {}, {}, {}, {}
    rows = []
    for fm in FLUX_MODELS:
        mdl = fm["model"]
        for ps in PION_SETS:
            nm = ps["name"]
            chi_z = lambda L, nm=nm: chiZ.profile_n(theory(fZ[nm], Z, mdl, L))[1]
            chi_h = lambda L, nm=nm: chiH.profile_n(theory(fH[nm], H, mdl, L))[1]

            evz = chiZ.evaluate(theory(fZ[nm], Z, mdl, fm["lam_jam"]))
            fixedZ[(nm, mdl)] = evz
            evh_jam = chiH.evaluate(theory(fH[nm], H, mdl, fm["lam_jam"]))
            fixedC[(nm, mdl)] = dict(evz=evz, evh=evh_jam)

            fz = fit_lambda(chi_z, fm["rng"])
            fz["evz"] = chiZ.evaluate(theory(fZ[nm], Z, mdl, fz["lam"]))
            fitZ[(nm, mdl)] = fz

            fh = fit_lambda(chi_h, fm["rng"])
            fh["evh"] = chiH.evaluate(theory(fH[nm], H, mdl, fh["lam"]))
            fitH[(nm, mdl)] = fh

            fc = fit_lambda(lambda L: chi_z(L) + chi_h(L), fm["rng"])
            fc["evz"] = chiZ.evaluate(theory(fZ[nm], Z, mdl, fc["lam"]))
            fc["evh"] = chiH.evaluate(theory(fH[nm], H, mdl, fc["lam"]))
            fitC[(nm, mdl)] = fc

            def add(fit_kind, lam, ft, evz=None, evh=None):
                r = dict(fit=fit_kind, pion_pdf=nm, flux=mdl, Lambda_GeV=lam,
                         Lambda_err_minus=ft.get("err_lo") if ft else None,
                         Lambda_err_plus=ft.get("err_hi") if ft else None,
                         Lambda_at_scan_edge=ft.get("at_edge") if ft else None)
                if evz is not None:
                    pb = per_bin(evz, Z)
                    rb = zeus_r_by_bin(evz, Z)
                    if ZEUS_R_PER_XL:
                        for xb, (a1, a2) in rb.items():
                            r[f"r1_xL{xb:g}"] = a1
                            r[f"r2_xL{xb:g}"] = a2
                    else:
                        r["r1"], r["r2"] = next(iter(rb.values()))
                    r.update(n_ZEUS=evz["n"],
                             chi2_ZEUS=evz["chi2"], chi2_ZEUS_points=evz["chi2_pts"],
                             sum_r2=evz["sum_r2"], pen_n_ZEUS=evz["pen_n"], N_ZEUS=nZ,
                             chi2_ZEUS_xL085=pb[0.85], chi2_ZEUS_xL094=pb[0.94])
                if evh is not None:
                    r.update(n_H1=evh["n"], chi2_H1=evh["chi2"], pen_n_H1=evh["pen_n"], N_H1=nH)
                tot = (evz["chi2"] if evz is not None else 0.0) + (evh["chi2"] if evh is not None else 0.0)
                npt = (nZ if evz is not None else 0) + (nH if evh is not None else 0)
                r.update(chi2_total=tot, N_total=npt, chi2_per_pt=tot / npt)
                rows.append(r)

            add("ZEUS_at_JAM_Lambda", fm["lam_jam"], None, evz=evz)
            add("H1+ZEUS_at_JAM_Lambda", fm["lam_jam"], None, evz=evz, evh=evh_jam)
            add("ZEUS_best_fit", fz["lam"], fz, evz=fz["evz"])
            add("H1_best_fit", fh["lam"], fh, evh=fh["evh"])
            add("H1+ZEUS_common_Lambda", fc["lam"], fc, evz=fc["evz"], evh=fc["evh"])
        print(f"  done: {mdl}")

    # text report
    L = []
    L.append("Leading-neutron F2^LN -- JAM Eq.(17) chi^2, two highest x_L bins per experiment")
    L.append("=" * 92)
    L.append("chi^2_e = sum_i [(d_i - sum_k r_k b_k,i - t_i/n_e)/alpha_i]^2 + sum_k r_k^2 + ((1-n_e)/dn_e)^2")
    L.append("theory  : t = 2 f_piN(x_L; Lambda) F2^pi_NLO(x_pi,Q^2)")
    L.append(f"pion PDF: JAM21 = mean of the replicas of {JAM21_SET} (as in the paper / FinalDis);"
             " xFitter = member 0 of xFitterPI_NLO_VAR; GRV (GRVPI1) single member")
    L.append(f"ZEUS    : x_L = {Z['xl_bins']}, N = {nZ}; alpha = stat; dn = {DNORM_ZEUS}; "
             "k_perp^2 <= (0.65 x_L)^2")
    for xl in Z["xl_bins"]:
        a, e = ZEUS_TAB1[xl]
        L.append(f"          x_L = {xl:.2f}:  b1 (acceptance) = {a:4.1f} %,  b2 (+-2% energy scale) = {e:+5.1f} %  (x d_i)")
    L.append(f"H1      : x_L = {H['xl_bins']}, N = {nH}; alpha = stat (+) sys; no beta; dn = {DNORM_H1}; "
             f"k_perp^2 <= {H1_K2_MAX}")
    L.append("r1, r2, n profiled at every Lambda; theory shifted by +r_k b_k (data by -r_k b_k).")
    L.append("Lambda errors: Delta chi^2 = 1 on the profiled curve; 'open' = no crossing inside the scan window.")
    L.append("")

    if ZEUS_R_PER_XL:
        RCOLS = " ".join(f"{f'r{k}({xb:g})':>9s}" for xb in Z["xl_bins"] for k in (1, 2))
    else:
        RCOLS = f"{'r1':>9s} {'r2':>9s}"

    def rvals(e):
        rb = zeus_r_by_bin(e, Z)
        if not ZEUS_R_PER_XL:
            a1, a2 = rb[round(float(Z["xl_bins"][0]), 2)]
            return f"{a1:+9.3f} {a2:+9.3f}"
        return " ".join(f"{rb[round(float(xb), 2)][k]:+9.3f}" for xb in Z["xl_bins"] for k in (0, 1))

    hdrZ = (f"{'flux':14s} {'Lambda':>7s} {'n_ZEUS':>7s} {RCOLS} "
            f"{'chi2':>8s} {'pts':>8s} {'sum r^2':>7s} {'pen n':>6s} {'chi2/N':>7s}")

    def lineZ(model, lam, e):
        return (f"{model:14s} {lam:7.3f} {e['n']:7.4f} {rvals(e)} "
                f"{e['chi2']:8.2f} {e['chi2_pts']:8.2f} {e['sum_r2']:7.3f} {e['pen_n']:6.3f} "
                f"{e['chi2'] / nZ:7.3f}")

    def lam_line(model, ft, rng):
        return (f"{model:14s} {ft['lam']:7.3f} {fmt_err(ft['err_lo'])} {fmt_err(ft['err_hi'])}  "
                f"[{rng[0]:.2f},{rng[1]:.2f}]  {'AT EDGE' if ft['at_edge'] else ''}")

    lam_hdr = f"{'flux':14s} {'Lambda':>7s} {'-err':>7s} {'+err':>7s}  {'window':>11s}  edge?"

    L.append("A) ZEUS -- JAM21 at the Lambda values used for JAM in FinalDis  ->  r1, r2")
    L.append("-" * 92)
    L.append(hdrZ)
    for fm in FLUX_MODELS:
        L.append(lineZ(fm["model"], fm["lam_jam"], fixedZ[("JAM21", fm["model"])]))
    L.append("")

    L.append("B) ZEUS -- best-fit Lambda for xFitter and GRV")
    L.append("-" * 92)
    for nm in ("xFitter", "GRV"):
        L.append(f"[{nm}]")
        L.append(lam_hdr)
        for fm in FLUX_MODELS:
            L.append(lam_line(fm["model"], fitZ[(nm, fm["model"])], fm["rng"]))
        L.append(hdrZ)
        for fm in FLUX_MODELS:
            ft = fitZ[(nm, fm["model"])]
            L.append(lineZ(fm["model"], ft["lam"], ft["evz"]))
        L.append("")

    L.append("C) ZEUS reference")
    L.append("-" * 92)
    L.append("[JAM21, best-fit Lambda]")
    L.append(lam_hdr)
    for fm in FLUX_MODELS:
        L.append(lam_line(fm["model"], fitZ[("JAM21", fm["model"])], fm["rng"]))
    L.append(hdrZ)
    for fm in FLUX_MODELS:
        ft = fitZ[("JAM21", fm["model"])]
        L.append(lineZ(fm["model"], ft["lam"], ft["evz"]))
    for nm in ("xFitter", "GRV"):
        L.append(f"[{nm} at the JAM Lambda values]")
        L.append(hdrZ)
        for fm in FLUX_MODELS:
            L.append(lineZ(fm["model"], fm["lam_jam"], fixedZ[(nm, fm["model"])]))
    L.append("")

    L.append("D) H1 + ZEUS combined -- ONE common Lambda (separate n_ZEUS, n_H1; r1, r2 for ZEUS)")
    L.append("-" * 92)
    hdrC = (f"{'flux':14s} {'Lambda':>7s} {'-err':>7s} {'+err':>7s} {'n_ZEUS':>7s} {RCOLS} "
            f"{'n_H1':>7s} {'chi2_Z':>8s} {'chi2_H1':>8s} {'chi2':>8s} {'chi2/N':>7s}  edge?")
    for ps in PION_SETS:
        nm = ps["name"]
        L.append(f"[{nm}]   N = {nZ} + {nH} = {nZ + nH}")
        L.append(hdrC)
        for fm in FLUX_MODELS:
            fc = fitC[(nm, fm["model"])]
            ez, eh = fc["evz"], fc["evh"]
            L.append(f"{fm['model']:14s} {fc['lam']:7.3f} {fmt_err(fc['err_lo'])} {fmt_err(fc['err_hi'])} "
                     f"{ez['n']:7.4f} {rvals(ez)} {eh['n']:7.4f} "
                     f"{ez['chi2']:8.2f} {eh['chi2']:8.2f} {fc['chi2']:8.2f} {fc['chi2'] / (nZ + nH):7.3f}  "
                     f"{'AT EDGE' if fc['at_edge'] else ''}")
        L.append("")
    L.append("E) H1 + ZEUS at the FinalDis JAM Lambda values (Lambda FIXED; n_ZEUS, r1, r2, n_H1 profiled)")
    L.append("-" * 92)
    L.append("   With Lambda fixed the H1 and ZEUS chi^2 share no parameter, so r1, r2, n_ZEUS here are the")
    L.append("   same numbers as the ZEUS-only fit at that Lambda (A, C); H1 adds n_H1 and chi2_H1.")
    hdrE = (f"{'flux':14s} {'Lambda':>7s} {'n_ZEUS':>7s} {RCOLS} {'n_H1':>7s} "
            f"{'chi2_Z':>8s} {'chi2_H1':>8s} {'chi2':>8s} {'chi2/N':>7s}")
    for ps in PION_SETS:
        nm = ps["name"]
        L.append(f"[{nm}]   N = {nZ} + {nH} = {nZ + nH}")
        L.append(hdrE)
        for fm in FLUX_MODELS:
            ez, eh = fixedC[(nm, fm["model"])]["evz"], fixedC[(nm, fm["model"])]["evh"]
            tot = ez["chi2"] + eh["chi2"]
            L.append(f"{fm['model']:14s} {fm['lam_jam']:7.3f} {ez['n']:7.4f} {rvals(ez)} "
                     f"{eh['n']:7.4f} {ez['chi2']:8.2f} {eh['chi2']:8.2f} {tot:8.2f} {tot / (nZ + nH):7.3f}")
        L.append("")
    L.append("   for comparison, the separate best fits (Lambda_ZEUS-only | Lambda_H1-only):")
    L.append(f"   {'flux':14s} " + "  ".join(f"{p['name']:>17s}" for p in PION_SETS))
    for fm in FLUX_MODELS:
        cells = []
        for ps in PION_SETS:
            a = fitZ[(ps["name"], fm["model"])]["lam"]
            b = fitH[(ps["name"], fm["model"])]["lam"]
            cells.append(f"{a:7.3f} | {b:7.3f}")
        L.append(f"   {fm['model']:14s} " + "  ".join(f"{c:>17s}" for c in cells))
    L.append("")
    L.append("per-x_L chi^2, H1-only fit details: lambda_fit_results.csv")

    txt = "\n".join(L)
    with open(OUT_TXT, "w") as fh:
        fh.write(txt + "\n")
    print("\n" + txt)

    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})

    make_chi2_figure(fitC, fixedC, Z, H)
    make_tex_table(fitC, fixedC, Z, H)
    print(f"\nwrote {OUT_TXT}\nwrote {OUT_CSV}\nwrote {OUT_FIG}.png/.pdf")

    # fit plots: F2^LN vs x_pi, H1 and ZEUS points
    if PLOT_FITS:
        os.makedirs(FIT_PLOT_DIR, exist_ok=True)
        print("theory curves for the fit plots ...")
        curves = {}
        for ps in PION_SETS:
            for D in (Z, H):
                for q2v in sorted(set(np.round(D["q2"], 4))):
                    curves[(ps["name"], D["expt"], float(q2v))] = f2pi_curve(ps["lhapdf"], D["expt"], q2v)
        for kind in PLOT_FITS:
            for fm in FLUX_MODELS:
                mdl = fm["model"]
                if mdl not in FIT_PLOT_FLUXES:
                    continue
                for ps in PION_SETS:
                    nm = ps["name"]
                    if kind == "JAM":
                        src, lam = fixedC[(nm, mdl)], fm["lam_jam"]
                        what, tag = rf"FinalDis JAM $\Lambda$ = {lam:g} GeV", "JAMLambda"
                    else:
                        src, lam = fitC[(nm, mdl)], fitC[(nm, mdl)]["lam"]
                        what, tag = rf"common best-fit $\Lambda$ = {lam:.3f} GeV", "commonLambda"
                    entry = dict(name=nm, color=ps["color"], ls=ps["ls"], lam=lam,
                                 evz=src["evz"], evh=src["evh"])
                    head = (f"{nm}:  H1 + ZEUS, {fm['label']} flux, {what}"
                            r"  (JAM Eq. 17 $\chi^2$;  $Q^2$ in GeV$^2$)")
                    base = os.path.join(FIT_PLOT_DIR, f"fit_F2LN_H1ZEUS_{mdl}_{tag}_{nm}")
                    make_fit_figure(base, fm, Z, H, curves, [entry], head)
                    print(f"wrote {base}.png/.pdf")


if __name__ == "__main__":
    main()
