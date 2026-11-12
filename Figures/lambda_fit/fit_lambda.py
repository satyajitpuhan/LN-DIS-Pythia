"""Fit of the pion-flux cutoff to the two largest x_L bins of H1 and ZEUS."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import lambda_fit

lambda_fit.main()
