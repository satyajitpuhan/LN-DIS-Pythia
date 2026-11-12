"""ZEUS F2LN(3) vs x_L in the (x_B, Q^2) bins with the "CT18 NLO + Remnant" Pythia setting:
Pythia + Sullivan for the three pion PDFs (Fig. 11) and the separate contributions (Fig. 24).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import grids  # noqa: E402

m = grids.load_original("zeus_three_pions")
zeus = m.load_zeus_f2ln(grids.ZEUS_DAT, f2p_table_file=grids.F2P)
sullivan = grids.sullivan_zeus()
remnant = grids.zeus_pythia(m, "ct18re")["f2ln"]
grids.set_colors(m, grids.PAPER_COLORS)

grids.save_as(lambda out: m.make_grid_ctre_plus_pions_zeus(remnant, sullivan, zeus, out),
              os.path.join(HERE, "compare_f2ln_zeus_ctre_three_f2ln_grid_zeus_ctre_plus_three_pions"))
grids.save_as(lambda out: m.make_grid_three_pions_zeus(remnant, sullivan, zeus, out),
              os.path.join(HERE, "compare_f2ln_zeus_ctre_three_f2ln_grid_zeus_three_pion_pdfs"))
m.save_chi2_table(m.compute_chi2_ctre_plus_pions(remnant, sullivan, zeus),
                  os.path.join(HERE, "compare_f2ln_zeus_ctre_three_f2ln_grid_zeus_ctre_plus_three_pions_chi2.txt"),
                  title="ZEUS data vs. CT18NLO + Remnant + Sullivan (bin averaged), chi^2/N")
