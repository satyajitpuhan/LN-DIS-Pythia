# Figures

Codes for the figures of the paper, one folder per figure. Each folder has the script(s)
and the figure as it is in the paper. Run the scripts from their own folder, for example

    cd pion_flux
    python3 pion_flux.py

| folder | figure | script |
|---|---|---|
| pion_flux | Fig. 2 | pion_flux.py |
| pion_structure_function | Fig. 3 | f2pi_q2_10.py |
| F2LN_H1 | Fig. 4 | f2ln_h1.py |
| F2LN_ZEUS | Fig. 6 | f2ln_zeus.py |
| data_over_theory | Figs. 5, 7, 18, 19 | data_over_theory.py |
| dsigma_dxL | Fig. 8 | dsigma_dxL_h1.py, dsigma_dxL_zeus.py |
| overview_H1_ZEUS | Fig. 9 | overview_h1.py, overview_zeus.py |
| total_H1 | Figs. 10, 21 | ct18_three_pions.py |
| | Figs. 20, 22 | pythia_settings.py |
| total_ZEUS | Figs. 11, 24 | ct18_remnant_three_pions.py |
| | Figs. 23, 25 | pythia_settings.py |
| EIC | Fig. 12 | eic_event_yield.py |
| | Fig. 13 | eic_dsigma_dxL.py |
| | Figs. 14-16 | eic_f2ln.py |
| lambda_fit | Fig. 17, Table V | fit_lambda.py |

The other folders:

- data: the H1 and ZEUS leading-neutron data (h1.dat, zeus.dat) and F2p from CT18NLO at
  the ZEUS points, used to convert the measured ratio r into F2LN(3)
- common: code used by several figures (pion flux, NLO F2 of the pion, the fit of the
  cutoff, and the H1/ZEUS plotting code)
- inputs: the Pythia histograms in the H1 and ZEUS bins and the bin averaged Sullivan term,
  together with the scripts that make them

## Pion PDFs and cutoffs

We use JAM21 (JAM21PionPDFnlo), xFitter (xFitterPI_NLO_VAR) and GRV (GRVPI1) from LHAPDF.
For JAM21 we always take the mean of the replicas as the central PDF. Member 0 of the
JAM21PionPDFnlo set is not the replica mean (it differs by 2-20%), so it is not used. For
xFitter and GRV member 0 is the central set. The only exception are the EIC projections
(Figs. 14-16), which still use member 0 of JAM21.

For JAM21 the cutoffs of the pion flux are the JAM18 values. For xFitter and GRV they come
from the fit in lambda_fit (lambda_fit_results.csv), which uses the two largest x_L bins of
H1 and ZEUS with the correlated ZEUS uncertainties as nuisance parameters.

## Order

Run lambda_fit first if you want to redo the fit; pion_flux and data_over_theory read its
results (lambda_fit_results.csv is included). The other figures can be made in any order.

The grid figures (Figs. 9-11 and 20-25) take the Pythia part from inputs/pythia_histograms
and the Sullivan part from inputs/sullivan_binavg. The Sullivan term there is averaged over each (x_B, Q^2) bin on 12 x 12 points (equal steps in
ln x_B and ln Q^2, points with y > 1 left out); for H1 it is also averaged over 12 points in
each x_L bin. To make these inputs again:

    cd inputs/sullivan_binavg
    python3 make_sullivan_h1.py --prep
    python3 make_sullivan_h1.py
    python3 make_sullivan_zeus.py --prep
    python3 make_sullivan_zeus.py --pdf jam21 --mode new     # and xfitter, grv

Figs. 8 and 12-16 and inputs/pythia_histograms/make_pythia_histograms.py read the Pythia
ROOT files (about 50 GB each, not included). Make them with the codes in ../Pythia; by default
the scripts look for them in ../Pythia/output, or set PYTHIA_OUT.

## Requirements

Python 3 with numpy, scipy, matplotlib and the LHAPDF python module, with the sets
JAM21PionPDFnlo, xFitterPI_NLO_VAR, GRVPI1 and CT18NLO. uproot is needed only to read the
Pythia ROOT files. If cupy is installed, F2 of the pion is computed on the GPU; all our
calculations were done on an NVIDIA GeForce RTX 5080 (16 GB, CUDA 13.2).
