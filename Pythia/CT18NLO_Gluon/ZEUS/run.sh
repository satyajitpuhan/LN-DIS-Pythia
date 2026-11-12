#!/bin/bash
# ZEUS, CT18NLO + gluon move
# N = number of events (5e8 in the paper), OUT = output folder (default Pythia/output)
set -e
cd "$(dirname "$0")"
N=${N:-500000000}
OUT=${OUT:-../../output}
mkdir -p "$OUT"

./dis_zeus 27.5 820 "$N" "$OUT/zeus_ct18nlo_gluon.root" CT18NLO
