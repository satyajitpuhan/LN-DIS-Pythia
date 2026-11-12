#!/usr/bin/env python3
"""d2N/dx_B dQ^2 for the three EIC beam settings, before and after the fiducial cuts (Fig. 12)."""

import os
PYTHIA_OUT = os.environ.get("PYTHIA_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Pythia", "output"))
import sys

import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm

# Configuration

FILES = sys.argv[1:] or [
    os.path.join(PYTHIA_OUT, "eic_5x41.root"),
    os.path.join(PYTHIA_OUT, "eic_10x100.root"),
    os.path.join(PYTHIA_OUT, "eic_18x275.root"),
]
TREENAME = "tree"
OUTDIR = "plots"

CHUNK_SIZE = 2_000_000  # rows per uproot.iterate() chunk
RESERVOIR_CAP = 300_000  # only used when MAKE_COMPARISONS is True

# Acceptance cuts
THETA_E_MIN_DEG = 135.0  # deg
THETA_E_MAX_DEG = 178.0  # deg
EEPRIME_MIN = 1.0  # GeV
THETA_N_MAX_MRAD = 5.0  # mrad

APPLY_THETA_E_CUT = True

USE_WEIGHTS = False

# radiative tail removal
DROP_RADIATIVE = True
Q2_GEN_MIN = 1.0  # GeV^2, effective generated floor
XB_MAX_PHYS = 1.0

# density normalisation
LUMI_FB = 10.0  # fb^-1

COMPARE_MODE = "counts"

# x_L binning
XL_DELTA = 0.02
XL_EDGES = np.round(np.arange(0.0, 1.0 + 0.5 * XL_DELTA, XL_DELTA), 6)

DRAW_Y_LINES = True
Y_LINES = [1.0, 0.5, 0.1, 0.01]

XB_LIM = (1e-5, 1.0)
Q2_LIM = (0.8, 1e4)
X_EDGES = np.logspace(np.log10(XB_LIM[0]), np.log10(XB_LIM[1]), 61)
Q2_EDGES = np.logspace(np.log10(max(Q2_GEN_MIN, Q2_LIM[0])),
                       np.log10(Q2_LIM[1]), 41)

# figure options
UNIVERSAL_AXIS_LABELS = True  # one x_B / Q^2 label for the whole figure
COMMON_COLOR_SCALE = False
CUTS_ON_EVERY_CUT_PANEL = True
MAKE_PER_ENERGY_ROW = False  # the old 1x4 per-file figures
MAKE_COMPARISONS = False

COLORS = ["black", "#d62728", "#0096FF", "#2ca02c", "#9467bd", "#ff7f0e"]

# Style

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 14,
    "font.weight": "bold",
    "axes.labelsize": 16,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.linewidth": 1.6,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "xtick.major.width": 1.4,
    "ytick.major.width": 1.4,
    "xtick.minor.width": 1.0,
    "ytick.minor.width": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
})

try:
    plt.rcParams["mathtext.default"] = "bf"  # bold upright
except ValueError:
    pass


def bold_ticks(ax, labelsize=10):
    """Bold tick labels on ax."""
    ax.tick_params(which="both", labelsize=labelsize)
    for lab in list(ax.get_xticklabels(which="both")) + \
               list(ax.get_yticklabels(which="both")):
        lab.set_fontweight("bold")


os.makedirs(OUTDIR, exist_ok=True)

if Q2_GEN_MIN < Q2_LIM[0]:
    print(f"[warn] Q2_GEN_MIN = {Q2_GEN_MIN} is below the axis floor "
          f"{Q2_LIM[0]}: bins between them are not drawn.")

BRANCHES = ["Q2", "xB", "y", "W2", "s", "xL", "xpi", "t", "MX2", "pT",
            "En", "Pn", "pz_n", "theta_n", "eta_n", "rapidity_n", "phi_n",
            "EePrime", "PePrime", "theta_e", "eta_e", "rapidity_e", "phi_e",
            "weight", "nNeutrons"]

OBS = {
    "Q2":           (r"$Q^2$ [GeV$^2$]",        True),
    "xB":           (r"$x_B$",                  True),
    "y":            (r"$y$",                    True),
    "W2":           (r"$W^2$ [GeV$^2$]",        True),
    "EePrime":      (r"$E_{e'}$ [GeV]",         False),
    "PePrime":      (r"$|\vec{p}_{e'}|$ [GeV]", False),
    "theta_e_deg":  (r"$\theta_{e'}$ [deg]",    False),
    "eta_e":        (r"$\eta_{e'}$",            False),
    "rapidity_e":   (r"$y_{e'}$",               False),
    "phi_e":        (r"$\phi_{e'}$ [rad]",      False),
    "En":           (r"$E_n$ [GeV]",            False),
    "Pn":           (r"$|\vec{p}_n|$ [GeV]",    False),
    "pT":           (r"$p_T^{\,n}$ [GeV]",      False),
    "pz_n":         (r"$p_z^{\,n}$ [GeV]",      False),
    "xL":           (r"$x_L$",                  False),
    "minus_t":      (r"$-t$ [GeV$^2$]",         False),
    "MX2":          (r"$M_X^2$ [GeV$^2$]",      False),
    "theta_n_mrad": (r"$\theta_n$ [mrad]",      True),
    "eta_n":        (r"$\eta_n$",               False),
    "rapidity_n":   (r"$y_n$",                  False),
    "phi_n":        (r"$\phi_n$ [rad]",         False),
    "nNeutrons":    (r"$N_n$",                  False),
    "xpi":          (r"$x_\pi$",                True),
}

EXTRA_SCALE_VARIANTS = {
    "Q2": [("linear", "log",    "Q2_linx_logy.png"),
           ("linear", "linear", "Q2_linx_liny.png")],
}

YLAB = {"counts": "Events",
        "lumi": f"Events / {LUMI_FB:g} fb$^{{-1}}$",
        "shape": "normalised to unit area"}[COMPARE_MODE]

# Shared per-chunk helpers


def guess_beams(s):
    prod = s / 4.0
    for ee, ep in [(18, 275), (10, 100), (5, 100), (5, 41), (18, 100),
                   (10, 275), (5, 275), (10, 41), (18, 41)]:
        if abs(ee * ep - prod) < 1e-6:
            return ee, ep
    return None, None


CFG_THETA_N_MRAD = {
    "5x41":   5.0,
    "10x100": 2.5,
    "18x275": 1.0,
}


def theta_n_max_for(s):
    """theta_n cut (mrad) for the beam config with this s."""
    ee, ep = guess_beams(s)
    return CFG_THETA_N_MRAD.get(f"{ee}x{ep}", THETA_N_MAX_MRAD)


def chunk_derive(raw, n_chunk, theta_n_max=None):
    theta_e_deg = np.degrees(raw["theta_e"])
    theta_n_mrad = 1000.0 * raw["theta_n"]

    th_n = THETA_N_MAX_MRAD if theta_n_max is None else theta_n_max
    CUT = ((raw["EePrime"] > EEPRIME_MIN)
           & (np.abs(theta_n_mrad) < th_n))
    if APPLY_THETA_E_CUT:
        CUT &= ((theta_e_deg > THETA_E_MIN_DEG)
                & (theta_e_deg < THETA_E_MAX_DEG))
    RAD_OK = (((raw["Q2"] >= Q2_GEN_MIN) & (raw["xB"] <= XB_MAX_PHYS))
              if DROP_RADIATIVE else np.ones(n_chunk, dtype=bool))
    weights_true = raw["weight"].astype(float) if USE_WEIGHTS else np.ones(n_chunk)

    d = {
        "Q2":           raw["Q2"],
        "xB":           raw["xB"],
        "y":            raw["y"],
        "W2":           raw["W2"],
        "EePrime":      raw["EePrime"],
        "PePrime":      raw["PePrime"],
        "theta_e_deg":  theta_e_deg,
        "eta_e":        raw["eta_e"],
        "rapidity_e":   raw["rapidity_e"],
        "phi_e":        raw["phi_e"],
        "En":           raw["En"],
        "Pn":           raw["Pn"],
        "pT":           raw["pT"],
        "pz_n":         raw["pz_n"],
        "xL":           raw["xL"],
        "minus_t":      -raw["t"],
        "MX2":          raw["MX2"],
        "theta_n_mrad": theta_n_mrad,
        "eta_n":        raw["eta_n"],
        "rapidity_n":   raw["rapidity_n"],
        "phi_n":        raw["phi_n"],
        "nNeutrons":    raw["nNeutrons"].astype(float),
        "xpi":          raw["xpi"],
    }
    for k in ("rapidity_e", "rapidity_n"):  # sentinel values written by the generator
        d[k] = np.where(d[k] < -900, np.nan, d[k])
    d["xpi"] = np.where(d["xpi"] <= 0, np.nan, d["xpi"])
    d = {k: np.asarray(v, dtype=float) for k, v in d.items()}
    return d, CUT, RAD_OK, weights_true


def reservoir_update(buf, new_vals):
    if new_vals.size == 0:
        return buf
    combined = np.concatenate([buf, new_vals]) if buf.size else new_vals
    if combined.size <= RESERVOIR_CAP:
        return combined
    idx = np.random.choice(combined.size, size=RESERVOIR_CAP, replace=False)
    return combined[idx]


# Pass 1

def load_pass1(path, color):
    print(f"[info] opening {path}:{TREENAME}")
    rfile = uproot.open(path, object_cache=None, array_cache=None)
    tree = rfile[TREENAME]
    n = tree.num_entries

    def meta(key):
        try:
            return rfile[key].member("fVal")
        except Exception:
            return None

    r = {"path": path, "color": color, "n_tot": n}
    r["sigma"] = meta("sigmaGen_nb")
    r["N_gen"] = meta("nGenerated") or float(n)
    r["N_dis"] = meta("nDIS")
    r["N_neu"] = meta("nNeutronEvt")
    r["w_sum"] = meta("weightSum")
    try:
        r["pdf"] = rfile["pdfSet"].member("fTitle")
    except Exception:
        r["pdf"] = "unknown"

    r["H2D"] = {False: np.zeros((len(X_EDGES) - 1, len(Q2_EDGES) - 1)),
                True:  np.zeros((len(X_EDGES) - 1, len(Q2_EDGES) - 1))}
    r["n2D"] = {False: 0, True: 0}
    r["reservoir"] = {}
    r["n_finite"] = {}

    n_cut_total = 0
    n_radok_dropped = 0
    n_seen = 0
    s_val = None

    for chunk in tree.iterate(BRANCHES, step_size=CHUNK_SIZE, library="np"):
        n_chunk = len(chunk["Q2"])
        n_seen += n_chunk
        if s_val is None:
            s_val = float(chunk["s"][0])
            theta_n_max = theta_n_max_for(s_val)
            r["theta_n_max"] = theta_n_max
            print(f"       neutron cut for this config: "
                  f"theta_n < {theta_n_max:g} mrad")

        d, CUT, RAD_OK, weights_true = chunk_derive(chunk, n_chunk,
                                                    theta_n_max)
        NOCUT = np.ones(n_chunk, dtype=bool)
        n_cut_total += int((CUT & RAD_OK).sum())
        n_radok_dropped += int((~RAD_OK).sum())

        for cutstate, mask_src in ((False, NOCUT), (True, CUT)):
            m = mask_src & RAD_OK
            x, q, w = d["xB"][m], d["Q2"][m], weights_true[m]
            g = np.isfinite(x) & np.isfinite(q) & (x > 0) & (q > 0)
            H, _, _ = np.histogram2d(x[g], q[g], bins=[X_EDGES, Q2_EDGES],
                                     weights=w[g])
            r["H2D"][cutstate] += H
            r["n2D"][cutstate] += int(g.sum())

            if MAKE_COMPARISONS:  # reservoirs only needed for the overlays
                for key in OBS:
                    v = d[key][m]
                    gv = np.isfinite(v)
                    if OBS[key][1]:
                        gv &= v > 0
                    v = v[gv]
                    key_c = (key, cutstate)
                    r["n_finite"][key_c] = r["n_finite"].get(key_c, 0) + v.size
                    r["reservoir"][key_c] = reservoir_update(
                        r["reservoir"].get(key_c, np.array([])), v)

        print(f"    ... {n_seen:,}/{n:,} entries "
              f"({100 * n_seen / max(n, 1):5.1f} %)", end="\r")
    print()

    r["s"] = s_val
    ee, ep = guess_beams(r["s"])
    r["Ee"], r["Ep"] = ee, ep
    r["tag"] = f"{ee}x{ep}" if ee else f"s{r['s']:.0f}"
    r["label"] = (rf"$e+p$ {ee}$\times${ep} GeV" if ee
                  else rf"$\sqrt{{s}}$ = {np.sqrt(r['s']):.1f} GeV")
    r["col_title"] = (rf"{ee} $\times$ {ep} GeV" if ee
                      else rf"$\sqrt{{s}}$ = {np.sqrt(r['s']):.1f} GeV")
    r["beam_label"] = (f"$E_e$ = {ee} GeV\n$E_p$ = {ep} GeV" if ee
                       else f"$\\sqrt{{s}}$ = {np.sqrt(r['s']):.1f} GeV")

    denom = r["w_sum"] if (USE_WEIGHTS and r["w_sum"]) else r["N_gen"]
    r["lumi_w"] = (r["sigma"] * 1e6 * LUMI_FB / denom) if r["sigma"] else 1.0

    r["n_cut"] = n_cut_total
    r["n_radok_dropped"] = n_radok_dropped

    print(f"       {r['tag']}   s = {r['s']:g} GeV^2   PDF = {r['pdf']}")
    print(f"       entries = {n}   N_gen = {r['N_gen']:.0f}   "
          f"sigma_gen = {r['sigma']:.4g} nb")
    print(f"       weightSum = {r['w_sum']:.6g}"
          + ("  (== N_gen: unweighted)"
             if r["w_sum"] and abs(r["w_sum"] - r["N_gen"]) < 1
             else "  (!= N_gen: WEIGHTED run -> USE_WEIGHTS = True)"))
    print(f"       after cuts = {r['n_cut']} ({100*r['n_cut']/n:.2f} %)"
          f"   radiative dropped = {r['n_radok_dropped']}"
          f" ({100*r['n_radok_dropped']/n:.2f} %)")
    print(f"       weight for L = {LUMI_FB:g} fb^-1 : {r['lumi_w']:.4g}/event")
    return r


def common_bins(key, cut, nbins=100, logx=None):
    if key == "xL":
        return XL_EDGES  # fixed uniform Delta x_L, not percentiles
    if logx is None:
        logx = OBS[key][1]
    lo, hi = np.inf, -np.inf
    for run in RUNS:
        v = run["reservoir"].get((key, cut))
        if v is None or v.size < 2:
            continue
        a, b = np.percentile(v, [0.05, 99.95])
        lo, hi = min(lo, a), max(hi, b)
    if not np.isfinite(lo) or hi <= lo:
        return None
    if logx:
        return np.logspace(np.log10(lo), np.log10(hi), nbins + 1)
    pad = 0.02 * (hi - lo)
    return np.linspace(lo - pad, hi + pad, nbins + 1)


def build_edge_jobs():
    jobs = {}
    for key in OBS:
        for cutstate in (False, True):
            logx = OBS[key][1]
            edges = common_bins(key, cutstate, logx=logx)
            if edges is not None:
                jobs[(key, cutstate, logx)] = edges
    for key, variants in EXTRA_SCALE_VARIANTS.items():
        for xscale, _, _ in variants:
            logx = (xscale == "log")
            for cutstate in (False, True):
                if (key, cutstate, logx) not in jobs:
                    edges = common_bins(key, cutstate, logx=logx)
                    if edges is not None:
                        jobs[(key, cutstate, logx)] = edges
    return jobs


# Pass 2

def load_pass2(run, edge_jobs):
    path = run["path"]
    print(f"[info] re-streaming {path} for 1D histograms ...")
    rfile = uproot.open(path, object_cache=None, array_cache=None)
    tree = rfile[TREENAME]
    n = tree.num_entries

    hist = {job: np.zeros(len(edges) - 1) for job, edges in edge_jobs.items()}
    n_seen = 0
    for chunk in tree.iterate(BRANCHES, step_size=CHUNK_SIZE, library="np"):
        n_chunk = len(chunk["Q2"])
        n_seen += n_chunk
        d, CUT, RAD_OK, weights_true = chunk_derive(
            chunk, n_chunk, run.get("theta_n_max"))
        NOCUT = np.ones(n_chunk, dtype=bool)

        for (key, cutstate, logx), edges in edge_jobs.items():
            m = (CUT if cutstate else NOCUT) & RAD_OK
            v = d[key][m]
            w = weights_true[m]
            g = np.isfinite(v)
            if logx:
                g &= v > 0
            h, _ = np.histogram(v[g], bins=edges, weights=w[g])
            hist[(key, cutstate, logx)] += h

        print(f"    ... {n_seen:,}/{n:,} entries "
              f"({100 * n_seen / max(n, 1):5.1f} %)", end="\r")
    print()
    run["hist1d"] = hist
    return run


# Load pass 1 for every file

RUNS = [load_pass1(f, COLORS[i % len(COLORS)]) for i, f in enumerate(FILES)]

RUNS.sort(key=lambda r: r["s"])

_seen = {}
for r in RUNS:
    _seen[r["tag"]] = _seen.get(r["tag"], 0) + 1
for r in RUNS:
    if _seen[r["tag"]] > 1:
        r["tag"] += "_" + os.path.splitext(os.path.basename(r["path"]))[0]

_THETA_E_LINE = (rf"${THETA_E_MIN_DEG:g}^\circ < \theta_{{e'}} < "
                 rf"{THETA_E_MAX_DEG:g}^\circ$" "\n") if APPLY_THETA_E_CUT else ""
def cut_text(theta_n=None):
    """Cut list for the annotation box."""
    if theta_n is None:
        vals = " / ".join(f"{CFG_THETA_N_MRAD[k]:g}"
                          for k in ("5x41", "10x100", "18x275")
                          if k in CFG_THETA_N_MRAD)
        th = rf"$\theta_n <$ {vals} mrad"
    else:
        th = rf"$\theta_n < {theta_n:g}$ mrad"
    return (_THETA_E_LINE +
            rf"$E_{{e'}} > {EEPRIME_MIN:g}$ GeV" "\n"
            + th + "\n"
            rf"$Q^2 \geq {Q2_GEN_MIN:g}$ GeV$^2$" "\n"
            rf"$x_B \leq {XB_MAX_PHYS:g}$")


CUT_TEXT = cut_text()  # generic, for multi-config figures

# phase-space lines only (shown on the no-cut row)
PHASE_TEXT = (rf"$Q^2 \geq {Q2_GEN_MIN:g}$ GeV$^2$" "\n"
              rf"$x_B \leq {XB_MAX_PHYS:g}$")

LUMI_TEXT = rf"$\int L\, \mathrm{{d}}t = {LUMI_FB:g}$ fb$^{{-1}}$"

CBAR_LABEL = (r"$\mathrm{d}^2N/\mathrm{d}x_B\,\mathrm{d}Q^2$ "
              r"[GeV$^{-2}$]")
CBAR_UNITS = r"[GeV$^{-2}$]"

CBAR_LABEL_MODE = "none"


def cbar_label_for(j, ncol):
    if CBAR_LABEL_MODE == "full":
        return CBAR_LABEL
    if CBAR_LABEL_MODE == "last":
        return CBAR_LABEL if j == ncol - 1 else CBAR_UNITS
    if CBAR_LABEL_MODE == "none":
        return ""
    return CBAR_UNITS

# Q2 vs xB density panels


def density_H(run, cut):
    """d2N/dxB dQ2 for LUMI_FB, on the fixed (X_EDGES, Q2_EDGES) grid."""
    H = run["H2D"][cut] * run["lumi_w"]
    return H / np.outer(np.diff(X_EDGES), np.diff(Q2_EDGES))


def draw_density(ax, H, vmin=None, vmax=None, dyn_range=1e8):
    if vmax is None:
        vmax = max(H.max(), 1.0)
    if vmin is None:
        vmin = H[H > 0].min() if np.any(H > 0) else vmax / dyn_range
    vmin = max(vmin, vmax / dyn_range)
    Hm = np.ma.masked_where(H <= 0, H)
    return ax.pcolormesh(X_EDGES, Q2_EDGES, Hm.T, cmap="rainbow",
                         norm=LogNorm(vmin=vmin, vmax=vmax))


def draw_y_lines(ax, s_cm, ylines=Y_LINES, fs=8, frac=0.55, lw=1.2):
    """Q2 = xB*y*s iso-lines with inline labels (no legend box)."""
    x0, x1 = ax.get_xlim()
    q0, q1 = ax.get_ylim()
    xx = np.logspace(np.log10(x0), np.log10(x1), 200)
    ax.figure.canvas.draw()
    for yv in ylines:
        ax.plot(xx, xx * yv * s_cm, color="red", lw=lw, zorder=5,
                solid_capstyle="butt")
        xa = max(x0, q0 / (yv * s_cm))
        xb = min(x1, q1 / (yv * s_cm))
        if xb <= xa:
            continue
        xlab = 10 ** (np.log10(xa) + frac * (np.log10(xb) - np.log10(xa)))
        qlab = xlab * yv * s_cm
        p1 = ax.transData.transform((xlab, qlab))
        p2 = ax.transData.transform((xlab * 1.02, qlab * 1.02))
        ang = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
        ax.text(xlab, qlab, f"$y$ = {yv:g}", color="red", fontsize=fs,
                rotation=ang, rotation_mode="anchor", fontweight="bold",
                ha="center", va="bottom", zorder=6,
                path_effects=[pe.withStroke(linewidth=2.0,
                                            foreground="white")])


def panel(ax, run, cut, vmin=None, vmax=None, row_label=None,
          show_lumi=False, show_cuts=False, cbar=True, axis_labels=None,
          cbar_label=None):
    if axis_labels is None:
        axis_labels = not UNIVERSAL_AXIS_LABELS
    if cbar_label is None:
        cbar_label = cbar_label_for(0, 1)
    H = density_H(run, cut)
    pcm = draw_density(ax, H, vmin=vmin, vmax=vmax)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*XB_LIM)
    ax.set_ylim(*Q2_LIM)
    if axis_labels:
        ax.set_xlabel(r"$x_B$", fontsize=19, fontweight="bold")
        ax.set_ylabel(r"$Q^2$ [GeV$^2$]", fontsize=19, fontweight="bold")
    bold_ticks(ax, labelsize=14)
    ax.tick_params(which="major", length=8, width=1.6)
    ax.tick_params(which="minor", length=4, width=1.2)

    if cbar:
        cb = ax.figure.colorbar(pcm, ax=ax, fraction=0.046, pad=0.02)
        if cbar_label:
            cb.set_label(cbar_label, fontsize=14, fontweight="bold")
        cb.ax.tick_params(labelsize=12)
        for lab in cb.ax.get_yticklabels():
            lab.set_fontweight("bold")

    if DRAW_Y_LINES:
        draw_y_lines(ax, run["s"])

    if row_label:
        ax.text(0.97, 0.96, row_label, transform=ax.transAxes,
                ha="right", va="top", fontsize=15, fontweight="bold",
                color="black", zorder=9,
                bbox=dict(boxstyle="round", fc="white", ec="black",
                          lw=1.2, alpha=0.92))

    if show_lumi:
        ax.text(0.04, 0.96, LUMI_TEXT, transform=ax.transAxes,
                ha="left", va="top", fontsize=14, fontweight="bold",
                color="black", zorder=9,
                bbox=dict(boxstyle="round", fc="white", ec="black",
                          lw=1.2, alpha=0.92))

    if show_cuts:
        ax.text(0.04, 0.96, cut_text(run.get("theta_n_max")),
                transform=ax.transAxes,
                ha="left", va="top", fontsize=13, fontweight="bold",
                color="black", zorder=9, linespacing=1.4,
                bbox=dict(boxstyle="round", fc="white", ec="black",
                          lw=1.2, alpha=0.92))


ncol = len(RUNS)
ROWS = [(False, "No cuts"), (True, "Cuts")]

vmin_g = vmax_g = None
if COMMON_COLOR_SCALE:
    all_H = [density_H(r, c) for r in RUNS for c, _ in ROWS]
    vmax_g = max(H.max() for H in all_H)
    pos = [H[H > 0].min() for H in all_H if np.any(H > 0)]
    vmin_g = min(pos) if pos else None

fig, axes = plt.subplots(2, ncol, figsize=(6.6 * ncol, 11.0), squeeze=False)

for i, (cut, row_name) in enumerate(ROWS):
    for j, run in enumerate(RUNS):
        panel(axes[i][j], run, cut, vmin=vmin_g, vmax=vmax_g,
              row_label=row_name, cbar_label=cbar_label_for(j, ncol),
              show_lumi=(i == 0 and j == 0),
              show_cuts=(cut and (CUTS_ON_EVERY_CUT_PANEL or j == 0)))
    # bold column headers with the beam energies (top row only)
    if i == 0:
        for j, run in enumerate(RUNS):
            axes[0][j].set_title(run["col_title"], fontsize=20,
                                 fontweight="bold", pad=14)

SUPTITLE_UNITS = True
SUPTITLE = (r"$\mathrm{d}^2N/\mathrm{d}x_B\,\mathrm{d}Q^2$"
            + (r" [GeV$^{-2}$]" if SUPTITLE_UNITS else ""))

fig.suptitle(SUPTITLE, fontsize=22, fontweight="bold")
if UNIVERSAL_AXIS_LABELS:
    fig.tight_layout(rect=[0.032, 0.042, 1, 0.965])
    fig.canvas.draw()
    pl, pr = axes[-1][0].get_position(), axes[-1][-1].get_position()
    ptop, pbot = axes[0][0].get_position(), axes[-1][0].get_position()
    fig.text(0.5 * (pl.x0 + pr.x1), 0.010, r"$x_B$",
             ha="center", va="bottom", fontsize=25, fontweight="bold")
    fig.text(0.008, 0.5 * (pbot.y0 + ptop.y1), r"$Q^2$ [GeV$^2$]",
             rotation=90, ha="left", va="center",
             fontsize=25, fontweight="bold")
else:
    fig.tight_layout(rect=[0, 0, 1, 0.965])

out = f"{OUTDIR}/d2N_dxB_dQ2_2x{ncol}.png"
fig.savefig(out)
fig.savefig(out.replace(".png", ".pdf"))
plt.close(fig)
print(f"\n[out] {out}")


if MAKE_PER_ENERGY_ROW:
    for run in RUNS:
        fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.5), squeeze=False)
        for k, (cut, row_name) in enumerate(ROWS):
            panel(axes[0][k], run, cut, row_label=row_name,
                  show_lumi=(k == 0), show_cuts=cut, axis_labels=True)
        fig.suptitle(rf"{SUPTITLE}   —   {run['col_title']}",
                     fontsize=17, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        out = f"{OUTDIR}/d2N_dxB_dQ2_{run['tag']}.png"
        fig.savefig(out)
        fig.savefig(out.replace(".png", ".pdf"))
        plt.close(fig)
        print(f"[out] {out}")

# 1D comparison overlays (off by default)

EDGE_JOBS = {}


def compare(key, cut, outdir, xscale=None, yscale=None, filename=None):
    logx = OBS[key][1] if xscale is None else (xscale == "log")
    logy = True if yscale is None else (yscale == "log")
    edges = EDGE_JOBS.get((key, cut, logx))
    if edges is None:
        return
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    any_data = False
    for run in RUNS:
        h = run.get("hist1d", {}).get((key, cut, logx))
        if h is None or h.sum() <= 0:
            continue
        hh = h.copy()
        if COMPARE_MODE == "lumi":
            hh = hh * run["lumi_w"]
        elif COMPARE_MODE == "shape":
            hh = hh / max(hh.sum(), 1e-30)
        _th = run.get("theta_n_max", THETA_N_MAX_MRAD)
        _lbl = (rf"{run['label']}  ($\theta_n<{_th:g}$ mrad)" if cut
                else run["label"])
        ax.stairs(hh, edges, color=run["color"], lw=1.8, label=_lbl)
        any_data = True
    if not any_data:
        plt.close(fig)
        return
    ax.set_xlabel(OBS[key][0], fontweight="bold")
    ax.set_ylabel(YLAB, fontweight="bold")
    ax.set_xscale("log" if logx else "linear")
    ax.set_yscale("log" if logy else "linear")
    bold_ticks(ax, labelsize=11)
    leg = ax.legend(fontsize=10, loc="best")
    for t in leg.get_texts():
        t.set_fontweight("bold")
    if cut:
        ax.text(0.02, 0.02, CUT_TEXT, transform=ax.transAxes, ha="left",
                va="bottom", fontsize=9, fontweight="bold", color="firebrick")
    ax.set_title("with cuts" if cut else "no cuts", fontsize=12,
                 fontweight="bold")
    fig.tight_layout()
    png_name = filename or f"{key}.png"
    fig.savefig(os.path.join(outdir, png_name))
    fig.savefig(os.path.join(outdir, os.path.splitext(png_name)[0] + ".pdf"))
    plt.close(fig)


if MAKE_COMPARISONS:
    EDGE_JOBS = build_edge_jobs()
    RUNS = [load_pass2(r, EDGE_JOBS) for r in RUNS]

    for cut, sub in [(False, "compare_nocuts"), (True, "compare_cuts")]:
        d = os.path.join(OUTDIR, sub)
        os.makedirs(d, exist_ok=True)
        for key in OBS:
            compare(key, cut, d)
        print(f"[out] {d}/  ({len(OBS)} observables, {len(RUNS)} files each)")

    for cut, sub in [(False, "compare_nocuts"), (True, "compare_cuts")]:
        d = os.path.join(OUTDIR, sub)
        for key, variants in EXTRA_SCALE_VARIANTS.items():
            for xscale, yscale, fname in variants:
                compare(key, cut, d, xscale=xscale, yscale=yscale,
                        filename=fname)
                print(f"[out] {d}/{fname}")

print(f"\nDone -> ./{OUTDIR}/")
