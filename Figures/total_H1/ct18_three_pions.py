"""H1 F2LN(3) vs x_L in the (x_B, Q^2) bins with the "CT18 NLO" Pythia setting:
Pythia + Sullivan for the three pion PDFs (Fig. 10) and the separate contributions (Fig. 21).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import grids  # noqa: E402

m = grids.load_original("h1_ct18_three_pions")
grids.check_h1_cuts(m)
h1 = m.load_hdat_f2ln(grids.H1_DAT)
cells = sorted(c for c, v in h1.items() if v is not None)
sullivan = grids.sullivan_h1(cells)
ct18 = grids.h1_pythia(m, "ct18")["f2ln"]
grids.set_colors(m, grids.PAPER_COLORS)

grids.save_as(lambda out: m.make_grid_ct18_plus_pions(ct18, sullivan, h1, out),
              os.path.join(HERE, "ct18_three_pion_f2ln_grid7x7_ct18_plus_three_pions"))
grids.save_as(lambda out: m.make_grid_three_pions(ct18, sullivan, h1, out),
              os.path.join(HERE, "ct18_three_pion_f2ln_grid7x7_three_pion_pdfs"))
m.save_chi2_table(m.compute_chi2_ct18_plus_pions(ct18, sullivan, h1),
                  os.path.join(HERE, "ct18_three_pion_f2ln_grid7x7_ct18_plus_three_pions_chi2.txt"),
                  title="H1 data vs. CT18 NLO + Sullivan (bin averaged), chi^2/N")
