# Leading neutrons in DIS at HERA and the EIC

Codes for

W.-C. Chang, C.-Y. Hsieh, S. Puhan, "Leading-Neutron Electroproduction at HERA and the EIC:
Sullivan Process, Target Fragmentation, and Pion PDFs", arXiv:2609.03368

There are two parts:

- Pythia: the Pythia8 codes for the DIS target fragmentation. One folder per Pythia setting
  (default, CT18NLO, CT18NLO_Remnant, CT18NLO_Tuning, CT18NLO_Gluon), each with a code for H1
  and for ZEUS, and for CT18NLO also the EIC one.
- Figures: the codes for the figures of the paper, one folder per figure (pion flux, pion
  structure function, F2LN for H1 and ZEUS, data/theory ratios, total H1 and ZEUS, EIC, the
  fit of the cutoff). Each folder also has the figure as it is in the paper.

More details are in Pythia/README.md and Figures/README.md.

## Requirements

The versions we used are given in brackets.

- Pythia 8.317, compiled with LHAPDF6. With 8.310 the H1 CT18NLO codes and all ZEUS codes
  stop at init, because PhaseSpace:Q2Max is not there.
- LHAPDF (6.5.5) with the sets JAM21PionPDFnlo, xFitterPI_NLO_VAR, GRVPI1 and CT18NLO
- ROOT (6.36) and a C++17 compiler, for the Pythia codes
- Python 3 (3.14) with numpy (2.4), scipy (1.17), matplotlib (3.10) and the LHAPDF python
  module
- uproot (5.7), only to read the Pythia ROOT files
- optional: an NVIDIA GPU with cupy (14.1) to compute F2 of the pion on the GPU (we used an
  NVIDIA GeForce RTX 5080), PyMuPDF to crop the PDF of the fit table

## Hardware

All calculations of this work (F2 of the pion, the Sullivan term, the bin averages, the fits
and the figures) were done with cupy on an NVIDIA GeForce RTX 5080 GPU (16 GB, CUDA 13.2).
Without cupy the same code runs on the CPU, only slower. The Pythia samples were generated
on the CPU.

## How to run

Pythia samples (only needed for Figs. 8 and 12-16, or to remake the Pythia histograms):

    cd Pythia
    make
    CT18NLO/H1/run.sh            # 5x10^8 events, about one day, ~50 GB

The root files go to Pythia/output. For a quick test use N=10000 CT18NLO/H1/run.sh. Make sure
that the Pythia 8.317 library comes first in LD_LIBRARY_PATH.

Figures: go to the folder and run the script, for example

    cd Figures/total_H1
    python3 ct18_three_pions.py

The Pythia histograms in the H1 and ZEUS bins and the bin averaged Sullivan term are included
(Figures/inputs), so all figures except Figs. 8 and 12-16 run without the ROOT files; the
H1 and ZEUS grids take a few seconds and less than 1.5 GB of memory. The fit of the cutoff takes about 15 s, Fig. 3 about 30 s.
The first run of lambda_fit and data_over_theory computes the JAM21 replica mean on a fine x
grid and keeps it in Figures/cache; this takes much longer, the next runs are fast. Making
the bin averaged Sullivan term again (Figures/inputs/sullivan_binavg) takes about 20 minutes
per pion PDF at most.

| folder in Figures | figure in the paper |
|---|---|
| pion_flux | 2 |
| pion_structure_function | 3 |
| F2LN_H1, F2LN_ZEUS | 4, 6 |
| data_over_theory | 5, 7, 18, 19 |
| dsigma_dxL | 8 |
| overview_H1_ZEUS | 9 |
| total_H1 | 10, 20, 21, 22 |
| total_ZEUS | 11, 23, 24, 25 |
| EIC | 12-16 |
| lambda_fit | 17 and Table V |

## Pythia settings

| folder | settings on top of the Pythia default |
|---|---|
| default | none, proton PDF NNPDF2.3 LO |
| CT18NLO | PDF:pSet = LHAPDF6:CT18NLO |
| CT18NLO_Remnant | CT18NLO, BeamRemnants:remnantMode = 1, ColourReconnection:mode = 1 |
| CT18NLO_Tuning | CT18NLO and the forward physics tune of Fieg et al., PRD 109, 016010 (2024) |
| CT18NLO_Gluon | CT18NLO, ColourReconnection:mode = 2 |

Beams: e+p, 27.5 x 920 GeV (H1), 27.5 x 820 GeV (ZEUS), 5x41, 10x100 and 18x275 GeV (EIC),
neutral current DIS (WeakBosonExchange:all = on). Generator cuts: Q2 > 6 GeV^2 for H1
(Q2 < 100 for the CT18NLO settings), 4 < Q2 < 10^4 GeV^2 for ZEUS, Q2 > 1 GeV^2 for the EIC;
mHatMin = 0 for ZEUS and the EIC. 5x10^8 events per sample. The leading neutron is the
final-state neutron with the highest energy, x_L = E_n/E_p.

## Pion PDFs and pion flux

- JAM21: we always use the mean of the replicas as the central PDF. Member 0 of the
  JAM21PionPDFnlo set is not the replica mean (it differs by 2-20%), so it is not used. The
  only exception are the EIC projections (Figs. 14-16), which still use member 0.
- xFitter: member 0, with the envelope of the variations as the band
- GRV: GRVPI1

The pion flux is integrated up to p_T = 0.2 GeV for H1 and p_T = 0.65 x_L GeV for ZEUS.
Cutoffs Lambda (GeV) of the five regulators (s-exp, t-exp, monopole, Regge, Pauli-Villars):

- JAM21: 1.31, 0.58, 0.52, 0.78, 0.25 (the JAM18 values)
- xFitter: 1.159, 0.552, 0.504, 0.799, 0.241 (our fit)
- GRV: 1.165, 0.497, 0.451, 0.620, 0.222 (our fit)

The comparisons with the full H1 and ZEUS data (Figs. 9-11 and 20-25) use the s-exp flux.

## Fit of the cutoff

The fit (Figures/lambda_fit) uses the two largest x_L bins of H1 (0.815, 0.905; 58 points)
and ZEUS (0.85, 0.94; 50 points), where target fragmentation is neglected as in the JAM
analyses, with the chi^2 of Cao et al., PRD 103, 114014 (2021), Eq. 17:

    chi2_e = sum_i [(d_i - sum_k r_k beta_k,i - t_i/n_e) / alpha_i]^2 + sum_k r_k^2 + ((1 - n_e)/dn_e)^2

For H1 alpha is stat + syst in quadrature and dn = 0.10. For ZEUS alpha is the statistical
error, dn = 0.088, and there are two correlated uncertainties: the acceptance (6%) and the
neutron energy scale (6.5% at x_L = 0.85, 25.5% at 0.94). The r_k are common to both ZEUS x_L
bins. For JAM21 the cutoff is kept at the JAM18 value; leaving it free gives
Lambda(s-exp) = 1.307 +0.071 -0.065 GeV.

## Sullivan term and chi^2 in the H1 and ZEUS figures

The Sullivan term is averaged over each (x_B, Q^2) bin like the Pythia part: 12 x 12 points,
equal steps in ln x_B and ln Q^2, points with y > 1 left out. For H1 it is also averaged over
12 points in each x_L bin, for ZEUS it is taken at the x_L bin centre. Pythia and Sullivan
are simply added, no normalisation factors.

chi^2/N in Figs. 9-11 is sum (d - t)^2 / sigma^2 over the data points, with sigma = stat + syst
in quadrature for H1 (N = 203) and the statistical error for ZEUS (N = 288). No
normalisations, correlations or theory uncertainties are included.
