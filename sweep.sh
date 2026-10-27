#!/usr/bin/env bash
# Local helper. Runs the official evaluation commands over several seeds.
# Usage: ./sweep.sh <smallGrid|mediumClassic|mediumMaze> <seed1> <seed2> ...
# PY overrides the interpreter; the default is the marker-matching 3.12.7 conda env.

PY="${PY:-/Users/omxr/opt/anaconda3/envs/pacman/bin/python}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
LAYOUT=$1; shift

if [ -z "$LAYOUT" ] || [ $# -eq 0 ]; then
  echo "Usage: $0 <layout> <seed> [seed ...]"; exit 1
fi

cd "$ROOT/src" || exit 1
echo "Python: $($PY --version 2>&1)"

for SEED in "$@"; do
  if [ "$LAYOUT" = "mediumMaze" ]; then
    ARGS=(-q -p MDPAgent -l mediumMaze --seed "$SEED" --maxMoves 300)
  else
    ARGS=(-q -n 25 -p MDPAgent -l "$LAYOUT" --seed "$SEED")
  fi
  START=$(date +%s)
  OUT=$("$PY" pacman.py "${ARGS[@]}" 2>&1)
  END=$(date +%s)
  echo "=== $LAYOUT seed=$SEED time=$((END-START))s ==="
  echo "$OUT" | grep -E "Win Rate|Record|Pacman emerges victorious|Traceback|Error|Moves:" | tail -n 5
done
