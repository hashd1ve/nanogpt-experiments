#!/bin/bash
# Reproduce the core 2x2 arms of the grokking-muon experiment (frac=0.3, seed 1337).
# Writes results/curves/<tag>.csv. Use --device cpu|mps|cuda (default mps).
cd "$(dirname "$0")" || exit 1
DEV="${1:-mps}"
run () { python -u harness.py --frac 0.3 --steps 4000 --eval_every 100 --geom_every 100 \
         --device "$DEV" --opt "$1" --wd "$2" --emb_wd "$2" --tag "$3"; }
run adamw 0   A_adamw_nowd
run adamw 1.0 B_adamw_wd
run muon  0   C_muon_nowd
run muon  1.0 D_muon_wd
