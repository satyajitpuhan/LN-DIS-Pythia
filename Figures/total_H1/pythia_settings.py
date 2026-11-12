"""H1 F2LN(3) vs x_L for the five Pythia settings: Pythia alone (Fig. 20) and
Pythia + Sullivan with JAM21 (Fig. 22).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import grids

m = grids.load_original("h1_five_settings")
grids.check_h1_cuts(m)
h1 = m.load_hdat_f2ln(grids.H1_DAT)
cells = sorted(c for c, v in h1.items() if v is not None)
jam21 = grids.sullivan_h1(cells)["jam21"]

results = {key: grids.h1_pythia(m, key) for key, _lab, _c in m.PDF_SETS}
m.finalize_pdf_set(results, jam21, h1)
grids.save_as(lambda out: m.make_grid_bg_only(results, h1, jam21, out),
              os.path.join(HERE, "sum_compare_f2ln_grid7x7_dis_background_only"))
grids.save_as(lambda out: m.make_grid_bg_jam(results, h1, jam21, out),
              os.path.join(HERE, "sum_compare_f2ln_grid7x7_dis_background_jam21"))
m.save_chi2_table(m.compute_overall_chi2(results, h1),
                  os.path.join(HERE, "sum_compare_f2ln_grid7x7_dis_background_jam21_chi2.txt"),
                  title="H1 data vs. Pythia setting + Sullivan with JAM21 (bin averaged), chi^2/N")
