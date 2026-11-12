# Leading neutrons in DIS at HERA and the EIC

Codes for

W.-C. Chang, C.-Y. Hsieh, S. Puhan, "Leading-Neutron Electroproduction at HERA and the EIC:
Sullivan Process, Target Fragmentation, and Pion PDFs", arXiv:2609.03368

- Pythia: the Pythia8 codes for the DIS target fragmentation, one folder per Pythia setting
  (default, CT18NLO, CT18NLO_Remnant, CT18NLO_Tuning, CT18NLO_Gluon), each with H1 and ZEUS,
  and the EIC samples for CT18NLO
- Figures: the codes for the figures of the paper, one folder per figure (pion flux, pion
  structure function, F2LN for H1 and ZEUS, data/theory ratios, total H1 and ZEUS, EIC, ...)

For JAM21 we always use the mean of the replicas as the central PDF, since member 0 of the
JAM21PionPDFnlo set is not the replica mean. The EIC projections (Figs. 14-16) are the only
figures that still use member 0. See the README in each folder for details.
