"""H1 dsigma/dx_L of the five Pythia samples compared with the H1 data (Fig. 8, left)."""

import argparse
import os
PYTHIA_OUT = os.environ.get("PYTHIA_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Pythia", "output"))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import uproot

# FONT SIZES
FS_AXLABEL = 26  # x_L and dsigma/dx_L axis labels
FS_TICK    = 17  # x and y tick numbers
FS_LEGEND  = 12  # curve legend
FS_ANNOT   = 13  # kinematic-cut box
FS_PTLABEL = 16  # the pT-condition box

TICK_LABEL_STROKE = 0.6
AXIS_LABEL_STROKE = 0.8  # the big x_L / F_2^{LN} axis labels

matplotlib.rcParams.update({
    "font.family"        : "serif",
    "font.serif"         : ["Times New Roman", "DejaVu Serif"],
    "font.weight"        : "bold",
    "mathtext.fontset"   : "dejavuserif",
    "mathtext.default"   : "bfit",  # bold *italic* math (zeus style)
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
    "xtick.major.size"   : 7,
    "ytick.major.size"   : 7,
    "xtick.minor.size"   : 3.5,
    "ytick.minor.size"   : 3.5,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.framealpha"  : 0.92,
    "legend.edgecolor"   : "0.45",
    "legend.fontsize"    : 6,
    "figure.dpi"         : 150,
    "axes.linewidth"     : 1.4,
})


def _stroke(*artists, width=TICK_LABEL_STROKE):
    if width <= 0:
        return
    import matplotlib.patheffects as patheffects
    fx = [patheffects.withStroke(linewidth=width, foreground="black")]
    for a in artists:
        if a is not None:
            a.set_path_effects(fx)


def bold_ticklabels(ax, width=TICK_LABEL_STROKE):
    """Bold + stroked numeric tick labels on both axes."""
    for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        t.set_fontweight("bold")
        _stroke(t, width=width)

# Colours
C_DEFAULT      = "#1f77b4"  # blue
C_CT18         = "#d62728"  # red
C_CT18_REMNANT = "#9467bd"  # purple
C_CT18_PTUNE   = "#ff7f0e"  # orange
C_CT18_GLUON   = "#17becf"  # cyan
C_DATA         = "black"

PDF_SETS = [
    ("default",     "Pythia default",     C_DEFAULT,      "solid",   2.4),
    ("ct18",        "CT18 NLO",           C_CT18,          "solid",   3.4),
    ("ct18remnant", "CT18 NLO + Remnant", C_CT18_REMNANT,  "dotted",  2.4),
    ("ct18ptune",   "CT18 NLO + Tuning",  C_CT18_PTUNE,    "dashed",  2.4),
    ("ct18gluon",   "CT18 NLO + Gluon",   C_CT18_GLUON,    "dashdot", 2.4),
]

# aliases
ANNOT_FONTSIZE = FS_ANNOT
PT_LABEL_FONTSIZE = FS_PTLABEL

Q2_MIN=6.0;   Q2_MAX=100.0
X_MIN=1.5e-4; X_MAX=3.0e-2
Y_MIN=0.02;   Y_MAX=0.6
W2_MIN=4.0
EePrime_MIN=11.0
THETA_E_MIN=156.0*np.pi/180.0
THETA_E_MAX=175.0*np.pi/180.0
THETA_N_MAX=0.75e-3

# Binning
XMIN=0.0; XMAX=1.0; NBINS=20
BW=(XMAX-XMIN)/NBINS
x_edges=np.linspace(XMIN, XMAX, NBINS+1)
x_centers=0.5*(x_edges[:-1]+x_edges[1:])

h1_edges  = np.array([0.30,0.37,0.44,0.51,0.58,0.65,0.72,0.79,0.86,0.93,1.00])
xd_centers= 0.5*(h1_edges[:-1]+h1_edges[1:])
xd_err    = 0.5*(h1_edges[1:]-h1_edges[:-1])

dsig_xl=np.array([10.6,12.0,14.0,15.6,16.5,16.7,16.0,12.4, 7.3, 2.3])
err_xl =np.sqrt(np.array([0.11,0.11,0.11,0.11,0.11,0.10,0.10,0.08,0.06,0.03])**2+
                np.array([ 1.3, 1.1, 1.3, 1.4, 1.4, 1.4, 1.3, 1.2, 1.2, 0.9])**2)
dsig_02=np.array([8.21,6.75,6.26,5.92,5.81,5.71,5.57,4.54,2.57,0.63])
err_02 =np.sqrt(np.array([0.09,0.07,0.06,0.05,0.05,0.05,0.05,0.04,0.03,0.01])**2+
                np.array([0.99,0.53,0.45,0.46,0.42,0.40,0.31,0.25,0.38,0.26])**2)

N_DIS_FIXED = 500_000_000


def pick(av, candidates):
    for c in candidates:
        if c in av: return c
    return None


def load_alllf_generic(path, label, pt_mode):
    with uproot.open(path) as f:
        # find the tree
        tree = None
        for key in f.keys():
            try:
                obj = f[key]
                if hasattr(obj, "keys") and "xL" in obj.keys():
                    tree = obj; break
            except Exception:
                pass
        if tree is None:
            raise RuntimeError(f"No tree with 'xL' in {path}")

        av    = set(tree.keys())
        q2_b  = pick(av, ["Q2","q2","Q2hat"])
        xb_b  = pick(av, ["xB","xb","x_B"])
        y_b   = pick(av, ["y","yBj","inelasticity"])
        w2_b  = pick(av, ["W2","w2","Wsq"])
        eep_b = pick(av, ["EePrime","Ee_prime","ePrime","E_e_prime"])
        thE_b = pick(av, ["theta_e","thetaE","ThetaE","theta_el"])
        thN_b = pick(av, ["theta_n","thetaN","ThetaN","theta_neu"])
        pt_b  = pick(av, ["pT","pt","PT","pT_n"])

        want = list(dict.fromkeys(
            ["xL"]+[b for b in [q2_b,xb_b,y_b,w2_b,eep_b,thE_b,thN_b,pt_b] if b]))
        arrs = {b: tree[b].array(library="np") for b in want}

        sigmaGen  = float(f["sigmaGen_nb"].value)
        n_dis     = float(f["nDIS"].value)
        n_neutron = float(f["nNeutronEvt"].value)

    n_total = float(N_DIS_FIXED)
    xl      = arrs["xL"]
    sel     = np.ones(len(xl), dtype=bool)

    cutflow = {}

    def cut(b, fn, lbl):
        nonlocal sel
        if b and b in arrs:
            before = int(sel.sum())
            sel &= fn(arrs[b])
            after  = int(sel.sum())
            cutflow[lbl] = (before, after)

    cut(q2_b,  lambda a:(a>=Q2_MIN)&(a<=Q2_MAX),            "Q2  6–100 GeV²")
    cut(xb_b,  lambda a:(a>=X_MIN) &(a<=X_MAX),             "xB  1.5e-4–3e-2")
    cut(y_b,   lambda a:(a>=Y_MIN) &(a<=Y_MAX),             "y   0.02–0.6")
    cut(w2_b,  lambda a: a > W2_MIN,                         "W2  >4 GeV²")
    cut(eep_b, lambda a: a > EePrime_MIN,                    "E'e >11 GeV")
    cut(thE_b, lambda a:(a>=THETA_E_MIN)&(a<=THETA_E_MAX),   "θe  156°–175°")
    cut(thN_b, lambda a: a <= THETA_N_MAX,                   "θn  <0.75 mrad")

    if pt_b and pt_b in arrs:
        before = int(sel.sum())
        if pt_mode == "xl069":
            sel &= arrs[pt_b] < xl * 0.69
        else:
            sel &= arrs[pt_b] < 0.2
        cutflow[f"pT ({pt_mode})"] = (before, int(sel.sum()))

    h_n, _ = np.histogram(xl[sel], bins=x_edges)
    dsig    = sigmaGen * h_n / n_total / BW

    # terminal summary
    print(f"\n  [{label} | {pt_mode}]")
    print(f"    sigmaGen                 = {sigmaGen:.6f} nb")
    print(f"    N_total (N_DIS_FIXED)    = {int(n_total):>10,}  ← denominator")
    print(f"    nDIS    (stored)         = {int(n_dis):>10,}")
    print(f"    nNeutronEvt (stored)     = {int(n_neutron):>10,}")
    print(f"    tree entries             = {len(xl):>10,}")
    print(f"    Cut-flow:")
    for lbl,(bf,af) in cutflow.items():
        print(f"      {lbl:35s} {bf:>8,} → {af:>8,}  (removed {bf-af:,})")
    print(f"    ∫dσ/dxL dxL (sum)        = {dsig.sum()*BW:.4f} nb")

    return dsig, sigmaGen, cutflow


def style_ax(ax, ylabel=r"$\mathbf{d\sigma/dx_L\ [nb]}$"):
    ax.set_xlabel(r"$\mathbf{x_L}$", fontsize=FS_AXLABEL, labelpad=6)
    ax.set_ylabel(ylabel, fontsize=FS_AXLABEL, labelpad=6)
    _stroke(ax.xaxis.label, width=AXIS_LABEL_STROKE)
    _stroke(ax.yaxis.label, width=AXIS_LABEL_STROKE)
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(bottom=0)
    ax.tick_params(which="major", direction="in", length=7, width=1.4,
                   top=True, right=True, pad=6, labelsize=FS_TICK)
    ax.tick_params(which="minor", direction="in", length=3.5, width=1.0,
                   top=True, right=True)
    ax.minorticks_on()
    bold_ticklabels(ax)


def make_comparison(curves_xl, curves_02, out):
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(17, 7.2))

    def _panel(ax, curves, data_y, data_err, tag, pt_lbl, show_leg):
        for key, lbl, color, ls, lw in PDF_SETS:
            if key not in curves:
                continue
            y, sig = curves[key]
            ax.stairs(y, x_edges, color=color, lw=lw, ls=ls, label=lbl)
        ax.errorbar(xd_centers, data_y, yerr=data_err, xerr=xd_err,
                    fmt="o", color=C_DATA, ms=7, lw=1.8, capsize=4,
                    zorder=5, label="H1")
        style_ax(ax)
        leg = ax.legend(frameon=True, loc="upper right",
                        fontsize=FS_LEGEND, handlelength=2.0)

    _panel(ax_a, curves_xl, dsig_xl, err_xl,
           "a", r"$p_T \leq x_L\times0.69\ \mathrm{GeV}$", True)
    _panel(ax_b, curves_02, dsig_02, err_02,
           "b", r"$p_T \leq 0.2\ \mathrm{GeV}$", False)

    fig.tight_layout()
    fname = f"{out}_comparison.png"
    fig.savefig(fname, dpi=220, bbox_inches="tight")
    print(f"\n  → {fname}")
    fname_pdf = f"{out}_comparison.pdf"
    fig.savefig(fname_pdf, bbox_inches="tight")
    print(f"  → {fname_pdf}")
    plt.close(fig)


def make_detail(curves, data_y, data_err, pt_lbl, pt_tag, out):
    fig, ax_main = plt.subplots(figsize=(9.5, 7.5))

    for key, lbl, color, ls, lw in PDF_SETS:
        if key not in curves:
            continue
        y, sig = curves[key]
        ax_main.stairs(y, x_edges, color=color, lw=lw, ls=ls, label=lbl)

    ax_main.errorbar(xd_centers, data_y, yerr=data_err, xerr=xd_err,
                     fmt="o", color=C_DATA, ms=7, lw=1.8, capsize=4,
                     zorder=5, label="H1")
    ax_main.set_xlabel(r"$\mathbf{x_L}$", fontsize=FS_AXLABEL, labelpad=6)
    ax_main.set_ylabel(r"$\mathbf{d\sigma/dx_L\ [nb]}$", fontsize=FS_AXLABEL,
                       labelpad=6)
    _stroke(ax_main.xaxis.label, width=AXIS_LABEL_STROKE)
    _stroke(ax_main.yaxis.label, width=AXIS_LABEL_STROKE)
    ax_main.set_xlim(XMIN, XMAX)
    ax_main.set_ylim(bottom=0)
    leg = ax_main.legend(frameon=True, loc="upper right", fontsize=FS_LEGEND)
    ax_main.tick_params(which="both", direction="in", top=True, right=True)
    ax_main.tick_params(which="major", labelsize=FS_TICK, pad=6)
    ax_main.minorticks_on()
    bold_ticklabels(ax_main)

    fname = f"{out}_detail_{pt_tag}.png"
    fig.savefig(fname, dpi=220, bbox_inches="tight")
    print(f"  → {fname}")
    fname_pdf = f"{out}_detail_{pt_tag}.pdf"
    fig.savefig(fname_pdf, bbox_inches="tight")
    print(f"  → {fname_pdf}")
    plt.close(fig)


def print_summary(curves_xl, curves_02):
    col_w = 16
    keys_present = [k for k,*_ in PDF_SETS if k in curves_xl]
    hdr = f"{'':33s}" + "".join(f"{lbl:>{col_w}s}" for k,lbl,*_ in PDF_SETS if k in curves_xl)
    sep = "─" * len(hdr)

    print("\n" + sep)
    print("SUMMARY TABLE")
    print(sep)
    print(hdr)
    print(sep)

    row = "  " + f"{'σ_DIS [nb]':31s}"
    for k in keys_present:
        row += f"{curves_xl[k][1]:>{col_w}.4f}"
    print(row)

    row = "  " + f"{'∫dσ/dxL dxL (xl069) nb':31s}"
    for k in keys_present:
        y,_ = curves_xl[k]
        row += f"{y.sum()*BW:>{col_w}.4f}"
    print(row)

    row = "  " + f"{'∫dσ/dxL dxL (pt02)  nb':31s}"
    for k in keys_present:
        y,_ = curves_02[k]
        row += f"{y.sum()*BW:>{col_w}.4f}"
    print(row)

    print(sep)
    print(f"  N_total (denominator) = {N_DIS_FIXED:,}")
    print(sep)


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
        help="Pythia ROOT file (CT18+gluon)")
    parser.add_argument("--out", default="dsigma_pdf_compare",
                        help="Output filename prefix (no extension)")
    args = parser.parse_args()

    paths = {"default": args.default,
              "ct18": args.ct18, "ct18remnant": args.ct18remnant,
              "ct18ptune": args.ct18ptune, "ct18gluon": args.ct18gluon}

    hdr = "="*64
    print(f"\n{hdr}")
    print("Formula: dσ/dxL = σ_gen × N_bin / N_total / ΔxL")
    print(f"N_total = N_DIS_FIXED = {N_DIS_FIXED:,}")
    print(hdr)

    curves_xl = {}
    curves_02 = {}

    for key, lbl, *_ in PDF_SETS:
        path = paths.get(key)
        if not path:
            continue
        print(f"\n══ Loading: {lbl}  ══")
        dsig_a, sig_a, _ = load_alllf_generic(path, lbl, "xl069")
        dsig_b, sig_b, _ = load_alllf_generic(path, lbl, "pt02")
        curves_xl[key] = (dsig_a, sig_a)
        curves_02[key] = (dsig_b, sig_b)

    if not curves_xl:
        raise SystemExit("No input ROOT files were loaded — nothing to plot.")

    # Summary table
    print_summary(curves_xl, curves_02)

    # Figure 1: side-by-side comparison
    print("\n── Saving figures ──")
    make_comparison(curves_xl, curves_02, args.out)

    # Figure 2a/2b: detail plots
    make_detail(curves_xl, dsig_xl, err_xl,
                r"$p_T \leq x_L\times0.69\ \mathrm{GeV}$", "xl069", args.out)
    make_detail(curves_02, dsig_02, err_02,
                r"$p_T \leq 0.2\ \mathrm{GeV}$", "pt02", args.out)

    print("\nDone.")


if __name__ == "__main__":
    main()
