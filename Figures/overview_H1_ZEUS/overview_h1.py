"""H1 F2LN(3) vs x_L averaged over x_B and Q^2 (Fig. 9, top): Pythia + Sullivan for the
five Pythia settings (columns) and the three pion PDFs (rows). chi^2/N is computed from the
individual data points.
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
sullivan = grids.sullivan_h1(cells)

results = {key: grids.h1_pythia(m, key) for key, _lab, _c in m.PDF_SETS}
m.finalize_pdf_set(results, sullivan["jam21"], h1)
pythia = {lab: results[key]["f2ln"] for key, lab, _c in m.PDF_SETS}
grids.save_as(lambda out: m.make_publication_overview_grid(pythia, sullivan, h1, out),
              os.path.join(HERE, "compare_f2ln_h1_ctre_three_all_combinations_overview"))
