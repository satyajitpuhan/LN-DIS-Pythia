"""Helpers shared by the H1 and ZEUS grid figures: Pythia F2LN(3) from the stored count
histograms, the bin-averaged Sullivan term, and the plotting functions of the original scripts."""
import glob
import importlib.util
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.dirname(HERE)
HIST = os.path.join(FIGURES, "inputs", "pythia_histograms")
SULLIVAN = os.path.join(FIGURES, "inputs", "sullivan_binavg")
H1_DAT = os.path.join(FIGURES, "data", "h1.dat")
ZEUS_DAT = os.path.join(FIGURES, "data", "zeus.dat")
F2P = os.path.join(FIGURES, "data", "F2p_CT18NLO_values.txt")
PAPER_COLORS = {"jam21": "#0a7cff", "xfitter": "#d62728", "grv": "#2ca02c"}
PIONS = ("jam21", "xfitter", "grv")
N_SUB = 12

# zeus_five_settings.py PDF_SETS key -> histogram file name
ZEUS_KEY = {"default": "default", "ct18ptune": "ct18tune", "ct18": "ct18",
            "ct18remnant": "ct18re", "ct18gluon": "ct18gluon"}


def load_original(name):
    path = os.path.join(HERE, f"{name}.py")
    sys.path.insert(0, HERE)
    spec = importlib.util.spec_from_file_location(f"orig_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def sullivan_h1(cells):
    """Bin-averaged Sullivan term for H1, {pion: {(iq, ix): F2LN[x_L bins]}}."""
    z = np.load(os.path.join(SULLIVAN, f"sullivan_h1_n{N_SUB}.npz"))
    out = {p: {c: z[f"{p}|{c[0]}|{c[1]}"] for c in cells} for p in PIONS}
    for p in PIONS:
        for c, v in out[p].items():
            assert np.all(np.isfinite(v)), f"{p} {c}: no bin average in a data cell"
    return out


def sullivan_zeus():
    """Bin-averaged Sullivan term for ZEUS, {pion: {(iq, ix): F2LN[x_L bins]}}."""
    out = {}
    for p in PIONS:
        z = np.load(os.path.join(SULLIVAN, f"sullivan_zeus_{p}_n{N_SUB}.npz"))
        out[p] = {tuple(map(int, k.split("|"))): z[k] for k in z.files if not k.startswith("npts")}
    return out


def check_h1_cuts(m):
    """The chunked histograms were made with these cuts/bins (make_pythia_histograms.py)."""
    assert (m.Q2_MIN, m.Q2_MAX, m.X_MIN, m.X_MAX, m.Y_MIN, m.Y_MAX, m.W2_MIN, m.EePrime_MIN,
            m.THETA_N_MAX, m.PT_MAX, m.N_TOTAL) == (6.0, 100.0, 1.5e-4, 3.0e-2, 0.02, 0.6, 4.0, 11.0,
                                                     0.75e-3, 0.2, 500_000_000)
    assert np.isclose(m.THETA_E_MIN, 156 * np.pi / 180) and np.isclose(m.THETA_E_MAX, 175 * np.pi / 180)


def h1_pythia(m, key):
    """Tail of the original process_pythia(), applied to the stored counts."""
    z = np.load(os.path.join(HIST, f"H1_{key}.npz"))
    assert np.allclose(z["xl_edges"], m.XL_EDGES)
    assert np.allclose(z["q2_edges"], [b[0] for b in m.Q2_BINS] + [m.Q2_BINS[-1][1]])
    assert np.allclose(z["x_edges"], [b[0] for b in m.X_BINS] + [m.X_BINS[-1][1]])
    counts, sigma_dis = z["counts"], float(z["sigma_dis"])
    f2ln, nevts = {}, {}
    for iq, (q2lo, q2hi) in enumerate(m.Q2_BINS):
        for ix, (xlo, xhi) in enumerate(m.X_BINS):
            c = counts[iq, ix]
            nevts[(iq, ix)] = int(c.sum())
            dsig = sigma_dis * c.astype(float) / m.N_TOTAL / m.XL_BWS
            f2ln[(iq, ix)] = m.dsig_to_f2ln(dsig, m.X_MEANS[ix], m.Q2_MEANS[iq], xhi - xlo, q2hi - q2lo)
    return dict(f2ln=f2ln, nevts=nevts, sigma_dis=sigma_dis)


def zeus_pythia(m, key):
    assert (m.EePrime_MIN, m.THETA_N_MAX, m.PT_COEFF, m.N_TOTAL) == (10.0, 0.8e-3, 0.656, 500_000_000)
    z = np.load(os.path.join(HIST, f"ZEUS_{key}.npz"))
    assert np.allclose(z["xl_edges"], m.XL_EDGES)
    assert np.allclose(z["q2_edges"], [b[1] for b in m.Q2_BINS_ZEUS] + [m.Q2_BINS_ZEUS[-1][2]])
    assert np.allclose(z["x_edges"], [b[1] for b in m.X_BINS_ZEUS] + [m.X_BINS_ZEUS[-1][2]])
    sigma_dis = float(z["sigma_dis"])
    f2ln, nevts = m.compute_all_pythia_f2ln_zeus(dict(sigma_dis=sigma_dis, counts=z["counts"]))
    return dict(f2ln=f2ln, nevts=nevts, sigma_dis=sigma_dis)


def save_as(draw, final):
    """Run an original plotting function into a scratch prefix, rename its png/pdf."""
    tmp = os.path.join(FIGURES, "cache", "_tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    draw(os.path.join(tmp, "x"))
    made = sorted(glob.glob(os.path.join(tmp, "x*")))
    assert sorted(os.path.splitext(f)[1] for f in made) == [".pdf", ".png"], made
    os.makedirs(os.path.dirname(final), exist_ok=True)
    for f in made:
        shutil.move(f, final + os.path.splitext(f)[1])
    shutil.rmtree(tmp)
    print(f"  saved {os.path.relpath(final, FIGURES)}.pdf", flush=True)


def set_colors(m, colors):
    m.PION_PDF_SETS[:] = [(k, n, lab, colors[k]) for k, n, lab, _c in m.PION_PDF_SETS]
