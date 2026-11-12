#!/bin/bash
# H1, CT18NLO + remnant
# N = number of events (5e8 in the paper), OUT = output folder (default Pythia/output)
set -e
cd "$(dirname "$0")"
N=${N:-500000000}
OUT=${OUT:-../../output}
mkdir -p "$OUT"

./dis_h1 27.5 920 "$N" "$OUT/h1_ct18nlo_remnant.root" CT18NLO
