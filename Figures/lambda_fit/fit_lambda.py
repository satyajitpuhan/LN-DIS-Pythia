"""Fit of the pion-flux cutoff to the two largest x_L bins of H1 and ZEUS.

Writes chi2_vs_lambda.pdf (Fig. 17), the table of the fit results (Table V) and
lambda_fit_results.csv, from which the other figures take the refitted cutoffs.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import lambda_fit  # noqa: E402

lambda_fit.main()
