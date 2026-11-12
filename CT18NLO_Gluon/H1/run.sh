#!/bin/bash
# H1, CT18NLO + gluon move
# N = number of events (5e8 in the paper), OUT = output folder
set -e
cd "$(dirname "$0")"
N=${N:-500000000}
OUT=${OUT:-.}

./dis_h1 27.5 920 "$N" "$OUT/h1_ct18nlo_gluon.root" CT18NLO
