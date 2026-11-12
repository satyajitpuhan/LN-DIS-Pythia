#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read the Pythia ROOT trees in chunks, apply the H1 or ZEUS cuts and count the events in
the (Q^2, x_B, x_L) bins. Output: hists/<EXPT>_<config>.npz.

Usage: python3 make_pythia_histograms.py [H1|ZEUS]
"""

import os
PYTHIA_OUT = os.environ.get("PYTHIA_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "Pythia", "output"))
import sys
import time
import resource

import numpy as np
import uproot

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = HERE
STEP = 1_000_000                       # entries per chunk

DEG = np.pi / 180.0
CONFIG = {
    "H1": dict(
        files=[("default",     "Pythia default",     os.path.join(PYTHIA_OUT, "h1_default.root")),
               ("ct18",        "CT18 NLO",           os.path.join(PYTHIA_OUT, "h1_ct18nlo.root")),
               ("ct18remnant", "CT18 NLO + Remnant", os.path.join(PYTHIA_OUT, "h1_ct18nlo_remnant.root")),
               ("ct18ptune",   "CT18 NLO + Tuning",  os.path.join(PYTHIA_OUT, "h1_ct18nlo_tuning.root")),
               ("ct18gluon",   "CT18 NLO + Gluon",   os.path.join(PYTHIA_OUT, "h1_ct18nlo_gluon.root"))],
        branches=["xL", "Q2", "xB", "y", "W2", "EePrime", "theta_e", "theta_n", "pT"],
        q2_edges=np.array([6.0, 9.0, 13.5, 20.0, 30.0, 45.0, 67.0, 100.0]),
        x_edges=np.array([1.5e-4, 3.2e-4, 6.82e-4, 1.45e-3, 3.10e-3, 6.60e-3, 1.41e-2, 3.0e-2]),
        xl_edges=np.array([0.32, 0.41, 0.50, 0.59, 0.68, 0.77, 0.86, 0.95]),
    ),
    "ZEUS": dict(
        files=[("default",   "Pythia default",    os.path.join(PYTHIA_OUT, "zeus_default.root")),
               ("ct18",      "CT18NLO",           os.path.join(PYTHIA_OUT, "zeus_ct18nlo.root")),
               ("ct18re",    "CT18NLO + Remnant", os.path.join(PYTHIA_OUT, "zeus_ct18nlo_remnant.root")),
               ("ct18tune",  "CT18NLO + Tuning",  os.path.join(PYTHIA_OUT, "zeus_ct18nlo_tuning.root")),
               ("ct18gluon", "CT18NLO + Gluon",   os.path.join(PYTHIA_OUT, "zeus_ct18nlo_gluon.root"))],
        branches=["xL", "Q2", "xB", "EePrime", "theta_n", "pT"],
        q2_edges=np.array([4.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0]),
        x_edges=np.array([8.0e-5, 1.5e-4, 3.0e-4, 6.0e-4, 1.2e-3, 2.4e-3, 1.0e-2, 1.0e-1]),
        xl_edges=np.array([0.20, 0.28, 0.34, 0.40, 0.46, 0.52, 0.58, 0.64, 0.70, 0.76, 0.82, 0.88, 1.00]),
    ),
}


def cuts(expt, c):
    """(kinematic selection, + P_T selection) -- h1_five_settings.py and zeus_three_pions.py."""
    if expt == "H1":
        sel = ((c["Q2"] >= 6.0) & (c["Q2"] <= 100.0)
               & (c["xB"] >= 1.5e-4) & (c["xB"] <= 3.0e-2)
               & (c["y"] >= 0.02) & (c["y"] <= 0.6)
               & (c["W2"] > 4.0) & (c["EePrime"] > 11.0)
               & (c["theta_e"] >= 156.0 * DEG) & (c["theta_e"] <= 175.0 * DEG)
               & (c["theta_n"] <= 0.75e-3))
        return sel, sel & (c["pT"] < 0.2)
    sel = (c["EePrime"] > 10.0) & (c["theta_n"] <= 0.8e-3)
    return sel, sel & (c["pT"] < 0.656 * c["xL"])


def bin_index(v, edges):
    """Right-open bin index, -1 outside."""
    i = np.searchsorted(edges, v, side="right") - 1
    return np.where((i >= 0) & (i < len(edges) - 1), i, -1)


def process(expt, key, label, path):
    cfg = CONFIG[expt]
    out = os.path.join(OUT_DIR, f"{expt}_{key}.npz")
    if os.path.exists(out):
        print(f"[{expt}/{key}] already done -> {out}", flush=True)
        return
    nq, nx, nxl = len(cfg["q2_edges"]) - 1, len(cfg["x_edges"]) - 1, len(cfg["xl_edges"]) - 1
    counts = np.zeros(nq * nx * nxl, dtype=np.int64)
    n_read = n_kin = n_pt = 0
    t0 = time.time()
    with uproot.open(path) as f:
        sigma = float(f["sigmaGen_nb"].members["fVal"])
        ngen = float(f["nGenerated"].members["fVal"]) if "nGenerated" in f else float("nan")
        tree = f["tree"]                                   # highest cycle (as the original scripts)
        n_tot = tree.num_entries
        print(f"[{expt}/{key}] {label}: {path}\n    tree;{tree.name} {n_tot:,} entries, "
              f"sigmaGen = {sigma:.4f} nb, nGenerated = {ngen:,.0f}", flush=True)
        for k, c in enumerate(tree.iterate(cfg["branches"], step_size=STEP, library="np")):
            n_read += len(c["xL"])
            kin, sel = cuts(expt, c)
            n_kin += int(kin.sum())
            n_pt += int(sel.sum())
            if sel.any():
                iq = bin_index(c["Q2"][sel], cfg["q2_edges"])
                ix = bin_index(c["xB"][sel], cfg["x_edges"])
                il = bin_index(c["xL"][sel], cfg["xl_edges"])
                ok = (iq >= 0) & (ix >= 0) & (il >= 0)
                flat = (iq[ok] * nx + ix[ok]) * nxl + il[ok]
                counts += np.bincount(flat, minlength=counts.size)
            del c, kin, sel
            if k % 25 == 0:
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                print(f"    {n_read/n_tot*100:5.1f} %  ({n_read:,}/{n_tot:,})  after cuts {n_pt:,}  "
                      f"{time.time()-t0:6.0f} s  peak RSS {rss:.0f} MB", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(out, counts=counts.reshape(nq, nx, nxl), sigma_dis=sigma, n_generated=ngen,
             n_read=n_read, n_entries=n_tot, n_after_kin=n_kin, n_after_pt=n_pt,
             q2_edges=cfg["q2_edges"], x_edges=cfg["x_edges"], xl_edges=cfg["xl_edges"],
             path=path, label=label)
    print(f"[{expt}/{key}] done in {time.time()-t0:.0f} s: read {n_read:,}, kin cuts {n_kin:,}, "
          f"+P_T {n_pt:,}, binned {int(counts.sum()):,}  -> {out}", flush=True)


def main():
    expts = sys.argv[1:] or ["H1", "ZEUS"]
    for expt in expts:
        for key, label, path in CONFIG[expt]["files"]:
            process(expt, key, label, path)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
