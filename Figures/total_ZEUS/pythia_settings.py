"""ZEUS F2LN(3) vs x_L for the five Pythia settings: Pythia alone (Fig. 23) and
Pythia + Sullivan with JAM21 (Fig. 25).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import grids

m = grids.load_original("zeus_five_settings")
zeus = m.load_zeus_f2ln(grids.ZEUS_DAT, f2p_table_file=grids.F2P)
jam21 = grids.sullivan_zeus()["jam21"]

results = {key: grids.zeus_pythia(m, grids.ZEUS_KEY[key]) for key, _lab, _c in m.PDF_SETS}
m.finalize_pdf_set(results, jam21, zeus)
grids.save_as(lambda out: m.make_grid_bg_only(results, zeus, jam21, out),
              os.path.join(HERE, "compare_f2ln_zeus_f2ln_grid_zeus_dis_background_only"))
grids.save_as(lambda out: m.make_grid_bg_jam(results, zeus, jam21, out),
              os.path.join(HERE, "compare_f2ln_zeus_f2ln_grid_zeus_dis_background_jam21"))
m.save_chi2_table(m.compute_overall_chi2(results, zeus),
                  os.path.join(HERE, "compare_f2ln_zeus_f2ln_grid_zeus_dis_background_jam21_chi2.txt"),
                  title="ZEUS data vs. Pythia setting + Sullivan with JAM21 (bin averaged), chi^2/N")
