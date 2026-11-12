# LN-DIS-Pythia

Pythia8 codes we used to simulate the DIS target fragmentation (leading neutrons) in

W.-C. Chang, C.-Y. Hsieh, S. Puhan, "Leading-Neutron Electroproduction at HERA and the EIC:
Sullivan Process, Target Fragmentation, and Pion PDFs", arXiv:2609.03368

The folders are arranged by Pythia setting. Inside each one there is a code for H1 and for
ZEUS, and for CT18NLO also the EIC one:

    default/          H1  ZEUS
    CT18NLO/          H1  ZEUS  EIC
    CT18NLO_Remnant/  H1  ZEUS
    CT18NLO_Tuning/   H1  ZEUS
    CT18NLO_Gluon/    H1  ZEUS

What is changed with respect to the Pythia default:

- default: nothing, the proton PDF is the Pythia default (NNPDF2.3 LO)
- CT18NLO: PDF:pSet = LHAPDF6:CT18NLO
- CT18NLO_Remnant: CT18NLO + BeamRemnants:remnantMode = 1, ColourReconnection:mode = 1
- CT18NLO_Tuning: CT18NLO + the forward physics tune of Fieg et al., PRD 109, 016010 (2024)
- CT18NLO_Gluon: CT18NLO + ColourReconnection:mode = 2

Beams are e+p with 27.5 x 920 GeV for H1, 27.5 x 820 GeV for ZEUS and 5x41, 10x100,
18x275 GeV for the EIC. Generator level Q2 cuts: Q2 > 6 GeV^2 for H1 (and Q2 < 100 for the
CT18NLO runs), 4 < Q2 < 10^4 GeV^2 for ZEUS and Q2 > 1 GeV^2 for the EIC. For ZEUS and the EIC
mHatMin is set to 0. For H1 we kept the default mHatMin = 4 GeV, which only removes
x_B < 1.6e-4.

## Requirements

- Pythia 8.317 compiled with LHAPDF6. With 8.310 the H1 CT18NLO and all ZEUS codes stop at
  init because PhaseSpace:Q2Max is not there.
- LHAPDF6 with the CT18NLO set
- ROOT

## Compile

    make

If you have more than one Pythia installed, use the 8.317 one

    PATH=/path/to/pythia8317/bin:$PATH make

and put /path/to/pythia8317/lib first in LD_LIBRARY_PATH, otherwise the code picks up the
other Pythia library when you run it.

## Run

Go to a folder and start run.sh, for example

    cd CT18NLO/H1
    ./run.sh

This generates 5x10^8 events as in the paper (about one day, the root file is ~50 GB). For a
quick test

    N=10000 ./run.sh

The output goes to the same folder, or to OUT=/some/path if you set it. The codes can also be
run directly,

    ./dis_h1 Ee Ep nEvents out.root CT18NLO

where the last argument is the LHAPDF proton set. Give CT18NLO for all the CT18NLO settings,
some of the codes use a different set if it is missing. The default codes don't need it.

## Output

The root file has a tree "tree" with the DIS variables (Q2, xB, y, W2, ...) from the scattered
positron and the variables of the leading neutron (the neutron with the highest energy): xL,
pT, t, theta_n, ... for every event with at least one neutron. No cuts are applied here, the
H1, ZEUS and EIC cuts are done later in the analysis. sigmaGen_nb and nGenerated are saved in
the same file, so the cross section after cuts is sigmaGen_nb * N_selected / nGenerated. The
running value of sigmaGen is also written to out_sigma.txt.
