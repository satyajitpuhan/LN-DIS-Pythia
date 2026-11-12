"""ZEUS F2LN(3) vs x_L averaged over x_B and Q^2 (Fig. 9, bottom), as overview_h1.py."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import grids  # noqa: E402

m = grids.load_original("zeus_three_pions")
zeus = m.load_zeus_f2ln(grids.ZEUS_DAT, f2p_table_file=grids.F2P)
sullivan = grids.sullivan_zeus()
grids.set_colors(m, grids.PAPER_COLORS)

colors = {s["label"]: s["color"] for s in m.PYTHIA_FILES}
labels = {"default": "Pythia default", "ct18": "CT18NLO", "ct18re": "CT18NLO + Remnant",
          "ct18tune": "CT18NLO + Tuning", "ct18gluon": "CT18NLO + Gluon"}
pythia = {lab: (grids.zeus_pythia(m, key)["f2ln"], lab, colors[lab]) for key, lab in labels.items()}
grids.save_as(lambda out: m.make_publication_overview_grid(pythia, sullivan, zeus, out),
              os.path.join(HERE, "compare_f2ln_zeus_ctre_three_all_combinations_overview"))
