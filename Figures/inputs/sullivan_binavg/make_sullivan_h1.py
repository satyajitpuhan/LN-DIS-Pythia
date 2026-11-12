#!/usr/bin/env python3
"""H1 Sullivan term averaged over each (x_B, Q^2) bin: 12 x 12 points equally spaced in
ln x_B and ln Q^2, points with y > 1 left out, on top of the usual average over 12 points
in each x_L bin.
"""
import argparse
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))

import numpy as np
import lhapdf
import h1_sullivan as OH

G = OH.f2pi_nlo
JAM = "JAM21PionPDFnlo"
N_DEFAULT = 12
YCUTS = {"phys": (0.0, 1.0), "h1": (0.02, 0.6)}


def log_mid(lo, hi, n):
    return np.exp(np.log(lo) + (np.arange(n) + 0.5) / n * np.log(hi / lo))


def q2_samples(n):
    return {iq: log_mid(lo, hi, n) for iq, (lo, hi) in enumerate(OH.Q2_BINS)}


def mean_file(n):
    return os.path.join(HERE, "..", "..", "cache", f"jam21_replica_mean_h1_binavg_n{n}.npz")


def _prep_worker(args):
    k, q2_list = args
    OH._ZLF.MEAN_CACHE = os.path.join(HERE, "..", "..", "cache", f"_mean_part{k}.npz")
    OH._ZLF._MEAN = {}
    for q2 in q2_list:
        OH._ZLF._mean_arrays(JAM, float(q2), G.X_FINE, True)
    return OH._ZLF.MEAN_CACHE


def prep(n, workers=6):
    from multiprocessing import Pool
    os.makedirs(os.path.join(HERE, "..", "..", "cache"), exist_ok=True)
    q2_all = np.concatenate(list(q2_samples(n).values()))
    with Pool(workers) as pool:
        parts = pool.map(_prep_worker, [(k, q2_all[k::workers]) for k in range(workers)])
    merged = {}
    for f in parts:
        with np.load(f) as z:
            merged.update({key: z[key] for key in z.files})
        os.remove(f)
    np.savez(mean_file(n), **merged)
    print(f"prep done: {len(merged)} Q^2 values -> {mean_file(n)}")


def binavg(pdf_obj, grid_fn, flux_fn, samples, ylim, label=""):
    n_sub = 12
    sub = np.concatenate([OH.XL_EDGES[i] + (np.arange(n_sub) + 0.5) * (OH.XL_EDGES[i + 1] - OH.XL_EDGES[i]) / n_sub
                          for i in range(OH.N_XL)])
    flux_grid = np.array([2.0 * float(flux_fn(float(xl))) for xl in sub]).reshape(OH.N_XL, n_sub)
    acc = {c: np.zeros(OH.N_XL) for c in samples}
    cnt = {c: 0 for c in samples}
    q2_all = sorted({float(q) for (_x, qs) in samples.values() for q in qs})
    t0 = time.time()
    for q2 in q2_all:
        grid = grid_fn(q2)
        cells, xs_list = [], []
        for c, (xs, qs) in samples.items():
            if not np.any(np.isclose(qs, q2, rtol=0, atol=0)):
                continue
            y = q2 / (OH.S * xs)
            xok = xs[(y >= ylim[0]) & (y <= ylim[1])]
            if len(xok):
                cells.append((c, len(xok)))
                xs_list.append(xok)
        if not cells:
            continue
        xall = np.concatenate(xs_list)
        x_pi = (xall[:, None] / (1.0 - sub[None, :])).ravel()
        vals = G.f2pi_minus_nlo_vec(pdf_obj, grid, x_pi, q2).reshape(len(xall), OH.N_XL, n_sub)
        vals = (vals * flux_grid[None, :, :]).mean(axis=2)  # x_L sub-point average
        k = 0
        for c, m in cells:
            acc[c] += vals[k:k + m].sum(axis=0)
            cnt[c] += m
            k += m
    print(f"  [{label}] {len(q2_all)} Q^2 values, {time.time() - t0:.0f} s", flush=True)
    out = {c: (acc[c] / cnt[c] if cnt[c] else np.full(OH.N_XL, np.nan)) for c in samples}
    return out, cnt


def grid_samples(n):
    qs = q2_samples(n)
    return {(iq, ix): (log_mid(xlo, xhi, n), qs[iq])
            for iq in range(len(OH.Q2_BINS)) for ix, (xlo, xhi) in enumerate(OH.X_BINS)}


def nominal_samples():
    return {(iq, ix): (np.array([OH.X_MEANS[ix]]), np.array([float(OH.Q2_MEANS[iq])]))
            for iq in range(len(OH.Q2_BINS)) for ix in range(len(OH.X_BINS))}


def pdf_setups(mode, n):
    p0 = {s: lhapdf.mkPDF(s, 0) for s in (JAM, "xFitterPI_NLO_VAR", "GRVPI1")}
    std = lambda p: (lambda q2: G.PionPdfGrid(p, q2))
    if mode == "previous":
        return [("jam21", p0[JAM], std(p0[JAM]), OH.f_piN_model_i),
                ("xfitter", p0["xFitterPI_NLO_VAR"], std(p0["xFitterPI_NLO_VAR"]), OH.f_piN_model_i),
                ("grv", p0["GRVPI1"], std(p0["GRVPI1"]), OH.f_piN_model_i)]
    lam = OH.best_lambdas()
    OH._ZLF.MEAN_CACHE = mean_file(n)
    OH._ZLF._MEAN = None
    mean_grid = lambda q2: OH.MeanPionPdfGrid(OH._ZLF._mean_arrays(JAM, q2, G.X_FINE, True), q2)
    return [("jam21", p0[JAM], mean_grid, OH.make_flux(OH.JAM21_TABLE1_LAMBDA)),
            ("xfitter", p0["xFitterPI_NLO_VAR"], std(p0["xFitterPI_NLO_VAR"]), OH.make_flux(lam["xFitter"])),
            ("grv", p0["GRVPI1"], std(p0["GRVPI1"]), OH.make_flux(lam["GRV"]))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--nsub", type=int, default=N_DEFAULT)
    ap.add_argument("--ycut", choices=list(YCUTS), default="phys")
    ap.add_argument("--modes", default="new")
    a = ap.parse_args()
    if a.prep:
        prep(a.nsub)
        return
    for mode in a.modes.split(","):
        out = {}
        for key, pdf, gfn, ffn in pdf_setups(mode, a.nsub):
            h, cnt = binavg(pdf, gfn, ffn, grid_samples(a.nsub), YCUTS[a.ycut], label=f"{key} {mode}")
            out.update({f"{key}|{iq}|{ix}": v for (iq, ix), v in h.items()})
            out.update({f"npts|{iq}|{ix}": np.array(m) for (iq, ix), m in cnt.items()})
        tag = "" if a.ycut == "phys" else "_ywindow"
        name = f"sullivan_h1{tag}_n{a.nsub}.npz" if mode == "new" else f"sullivan_h1{tag}_{mode}_n{a.nsub}.npz"
        path = os.path.join(HERE, name)
        np.savez(path, **out)
        print(f"-> {path}", flush=True)


if __name__ == "__main__":
    main()
