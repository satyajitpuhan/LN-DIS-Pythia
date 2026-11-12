#!/bin/bash
# ZEUS, CT18NLO + forward tune
# N = number of events (5e8 in the paper), OUT = output folder
set -e
cd "$(dirname "$0")"
N=${N:-500000000}
OUT=${OUT:-.}

./dis_zeus 27.5 820 "$N" "$OUT/zeus_ct18nlo_tuning.root" CT18NLO
