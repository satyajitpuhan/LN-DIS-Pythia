#!/usr/bin/env python3
"""ZEUS Sullivan term averaged over each (x_B, Q^2) bin (12 x 12 points equally spaced in
ln x_B and ln Q^2, y <= 1); x_L stays at the bin centre.
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
import zeus_sullivan as OZ

OZ.lhapdf = lhapdf
SETS = {"jam21": ("JAM21PionPDFnlo", "JAM21"), "xfitter": ("xFitterPI_NLO_VAR", "xFitter"),
        "grv": ("GRVPI1", "GRV")}
N_DEFAULT = 12


def log_mid(lo, hi, n):
    return np.exp(np.log(lo) + (np.arange(n) + 0.5) / n * np.log(hi / lo))


def q2_samples(n):
    return {iq: log_mid(lo, hi, n) for iq, (_nom, lo, hi) in enumerate(OZ.Q2_BINS_ZEUS)}


def data_cells():
    zd = OZ.load_zeus_f2ln(OZ.ZEUS_DAT_PATH, f2p_table_file=OZ.F2P_PATH)
    return sorted(zd.keys())


def mean_file(n):
    return os.path.join(HERE, "..", "..", "cache", f"jam21_replica_mean_zeus_binavg_n{n}.npz")


def _prep_worker(args):
    k, q2_list = args
    OZ._ZLF.MEAN_CACHE = os.path.join(HERE, "..", "..", "cache", f"_mean_part{k}.npz")
    OZ._ZLF._MEAN = {}
    for q2 in q2_list:
        OZ._ZLF._mean_arrays("JAM21PionPDFnlo", float(q2), OZ.X_FINE, False)
    return OZ._ZLF.MEAN_CACHE


def prep(n, workers=6):
    from multiprocessing import Pool
    os.makedirs(os.path.join(HERE, "..", "..", "cache"), exist_ok=True)
    q2_all = np.concatenate(list(q2_samples(n).values()))
    jobs = [(k, q2_all[k::workers]) for k in range(workers)]
    with Pool(workers) as pool:
        parts = pool.map(_prep_worker, jobs)
    merged = {}
    for f in parts:
        with np.load(f) as z:
            merged.update({key: z[key] for key in z.files})
        os.remove(f)
    np.savez(mean_file(n), **merged)
    print(f"prep done: {len(merged)} Q^2 values -> {mean_file(n)}")


class MeanPDFAt(OZ.MeanPDF):
    def __init__(self, set_name, q2_list, n):
        self.set_name = set_name
        self._m0 = lhapdf.mkPDF(set_name, 0)
        OZ._ZLF.MEAN_CACHE = mean_file(n)
        OZ._ZLF._MEAN = None
        for q2 in q2_list:
            OZ.PDF_CACHE[(id(self), float(q2))] = OZ._ZLF._mean_arrays(set_name, float(q2), OZ.X_FINE, False)


def binavg(pdf, flux_fn, cells, n, label=""):
    """{(iq, ix): F2LN[N_XL]} averaged over the (x_B, Q^2) bin."""
    t0 = time.time()
    qs = q2_samples(n)
    out, n_used = {}, {}
    for (iq, ix) in cells:
        _xnom, xlo, xhi = OZ.X_BINS_ZEUS[ix]
        xs = log_mid(xlo, xhi, n)
        pts = [(x, q2) for q2 in qs[iq] for x in xs if q2 / (OZ.S * x) <= 1.0]
        curve = np.full(OZ.N_XL, np.nan)
        for i, xl_c in enumerate(OZ.XL_CTRS):
            if not pts:
                continue
            fl = 2.0 * flux_fn(float(xl_c))
            vals = [fl * OZ.f2_nlo(pdf, float(x / (1.0 - xl_c)), float(q2)) if x / (1.0 - xl_c) < 1.0 else 0.0
                    for x, q2 in pts]
            curve[i] = float(np.mean(vals))
        out[(iq, ix)] = curve
        n_used[(iq, ix)] = len(pts)
        print(f"  [{label}] cell (iq={iq}, ix={ix}): {len(pts)}/{n * n} points with y <= 1   "
              f"{time.time() - t0:6.0f} s", flush=True)
    return out, n_used


def make_pdf_and_flux(pdf_key, mode, n):
    setn, lab = SETS[pdf_key]
    if mode == "new" and pdf_key == "jam21":
        q2_all = np.concatenate(list(q2_samples(n).values()))
        return MeanPDFAt(setn, q2_all, n), OZ.make_flux(OZ.JAM21_TABLE1_LAMBDA)
    lam = 1.31 if (mode == "previous" or pdf_key == "jam21") else OZ.best_lambdas()[lab]
    return lhapdf.mkPDF(setn, 0), OZ.make_flux(lam)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", action="store_true", help="build the JAM21 replica-mean grids only")
    ap.add_argument("--pdf", choices=list(SETS))
    ap.add_argument("--mode", choices=["previous", "new"])
    ap.add_argument("--nsub", type=int, default=N_DEFAULT)
    ap.add_argument("--cells", default=None, help="test: 'iq,ix;iq,ix'")
    a = ap.parse_args()

    if a.prep:
        prep(a.nsub)
        return
    cells = ([tuple(map(int, c.split(","))) for c in a.cells.split(";")] if a.cells else data_cells())
    pdf, flux_fn = make_pdf_and_flux(a.pdf, a.mode, a.nsub)
    h, n_used = binavg(pdf, flux_fn, cells, a.nsub, label=f"{a.pdf} {a.mode} n={a.nsub}")
    os.makedirs(os.path.join(HERE, "..", "..", "cache"), exist_ok=True)
    tag = "_test" if a.cells else ""
    name = f"sullivan_zeus_{a.pdf}_n{a.nsub}{tag}.npz" if a.mode == "new" else f"sullivan_zeus_{a.pdf}_{a.mode}_n{a.nsub}{tag}.npz"
    path = os.path.join(HERE, name)
    np.savez(path, **{f"{iq}|{ix}": v for (iq, ix), v in h.items()},
             **{f"npts|{iq}|{ix}": np.array(n_used[(iq, ix)]) for (iq, ix) in n_used})
    print(f"-> {path}")


if __name__ == "__main__":
    main()
