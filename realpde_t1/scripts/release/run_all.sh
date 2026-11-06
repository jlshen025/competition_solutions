#!/usr/bin/env bash
# One-command retrain of the RealPDE Track-1 entry from the competition release alone.
#
#   bash scripts/release/run_all.sh all                 # everything, sequentially
#   bash scripts/release/run_all.sh prep                # 1. release .h5 -> 32x64 float32 memmaps
#   bash scripts/release/run_all.sh train <NAME>        # 2. one net (8 of them, independent -> parallelisable)
#   bash scripts/release/run_all.sh calibrate           # 3. width table + variance head + archive
#
# NAMES: shipA_s250 shipB_s250 shipA_s350 shipB_s350   (all real data, these four ship)
#        cvA_s250   cvB_s250   cvA_s270   cvB_s270     (Re 13950+22875 held out; only
#                                                       used to fit the width table)
# Recipe A = direct delta predictor, recipe B = the same with a 2-step rollout (--ar-m 10).
#
# Environment (all optional):
#   RPDE_RAW     release dir holding train_real/ and train_sim/ (the HF dataset
#                AI4Science-WestlakeU/RealPDE-Competition-Data)
#   RPDE_DATA32  where step 1 writes / step 2 reads the preprocessed memmaps
#   RPDE_WORK    working dir (default: ./release_work) -> runs/<NAME>, calib/, final zip
#   RUNS_DIR / RUN_PREFIX   where step 3 finds the trained runs (default $RPDE_WORK/runs, "")
#   EXP_OUTPUT_DIR          if set, `train` writes there instead of $RUNS_DIR/<NAME>
#   SHIP_SEEDS / CV_SEEDS   two seeds each (default "250 350" / "250 270")
#   RPDE_STEPS   training steps per net (default 60000 = the ranked entry; smaller only for smoke tests)
#   FUSED / FWD_DTYPE       inference layout written by `calibrate` (default 2 / float16;
#                           FUSED=0 FWD_DTYPE= gives serial fp32 members)
#   EXPECT       optional reference archive that `calibrate` compares against
#
# Needs: python3 with torch (>=2.2, CUDA) + numpy; h5py for step 1 only. One GPU
# per training run (~16 GB is plenty; 30-45 min each on an H100 MIG 2g slice).
# No network access, no external data, no pretrained weights anywhere.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
WORK="${RPDE_WORK:-$ROOT/release_work}"
RUNS_DIR="${RUNS_DIR:-$WORK/runs}"
RUN_PREFIX="${RUN_PREFIX:-}"
PY="${PYTHON:-python}"

COMMON="--split real --steps ${RPDE_STEPS:-60000} --bs 32 --width 64 --depth 8 --spectral 0 --delta 1 \
--lam-tke 0.15 --lam-mvpe 0.249 --lam-nll 0.2 --lam-cov 3.0 --cov-k 2.0 \
--holdout-re 13950,22875 --select composite \
--aug-krange 0.75,1.6 --aug-kmode interp --aug-frac 0.125 --aug-flip 0.125"

# Seeds: SHIP_SEEDS / CV_SEEDS (two each) override the defaults the ranked entry used.
SHIP_SEEDS="${SHIP_SEEDS:-250 350}"; CV_SEEDS="${CV_SEEDS:-250 270}"
SHIP=""; for s_ in $SHIP_SEEDS; do SHIP="$SHIP shipA_s$s_ shipB_s$s_"; done   # archive order = model0..3
CV="";   for s_ in $CV_SEEDS;   do CV="$CV cvA_s$s_ cvB_s$s_"; done

train_one() {
  local name="$1" extra="" all=0
  case "$name" in
    shipA_s*) all=1 ;;
    shipB_s*) all=1; extra="--ar-m 10" ;;
    cvA_s*)   all=0 ;;
    cvB_s*)   all=0; extra="--ar-m 10" ;;
    *) echo "unknown run name $name"; exit 2 ;;
  esac
  local seed="${name##*_s}"
  local out="${EXP_OUTPUT_DIR:-$RUNS_DIR/$RUN_PREFIX$name}"
  mkdir -p "$out"
  echo "TRAIN $name seed=$seed train_all=$all -> $out"
  EXP_OUTPUT_DIR="$out" $PY scripts/train.py $COMMON --train-all $all $extra --seed "$seed"
  test -s "$out/model.pth" || { echo "TRAIN FAILED: no model.pth in $out"; exit 1; }
}

calibrate() {
  local C="$WORK/calib"; mkdir -p "$C/newtab" "$C/vfit"
  local ship_csv="" ship_plus="" cv_csv=""
  for n in $SHIP; do ship_csv="$ship_csv,$RUNS_DIR/$RUN_PREFIX$n"; ship_plus="$ship_plus+$RUNS_DIR/$RUN_PREFIX$n"; done
  for n in $CV;   do cv_csv="$cv_csv,$RUNS_DIR/$RUN_PREFIX$n"; done
  ship_csv="${ship_csv#,}"; ship_plus="${ship_plus#+}"; cv_csv="${cv_csv#,}"
  echo "CALIBRATE ship=$ship_csv"
  echo "          cv=$cv_csv"
  # 3a. calibration-free base archive: carries beta/kappa for 3b/3c
  $PY scripts/release/assemble.py --base --runs "$ship_csv" --out "$C/base.zip"
  # 3b. width table on the CV nets' held-out residuals (never in-sample)
  EXP_OUTPUT_DIR="$C/newtab" $PY scripts/newtab.py --runs "$cv_csv" --zip "$C/base.zip" --target 94.951
  # 3c. variance-head ridge fit on the SHIP nets over all training windows
  EXP_OUTPUT_DIR="$C/vfit" $PY scripts/vfit.py --runs "$ship_plus" --zip "$C/base.zip" --kind poly2 --dose 0.3
  # 3d. the submission archive (calibration settings = assemble.py defaults)
  #     with the fused fp16 inference path: one grouped trunk for the 4 members
  #     in fp16, about 1.7x faster than serial fp32 members at equal accuracy.
  EXP_OUTPUT_DIR="$C" $PY scripts/release/assemble.py --runs "$ship_csv" \
      --table "$C/newtab/results.json" --vfit "$C/vfit/results.json" \
      --fused "${FUSED:-2}" --fwd-dtype="${FWD_DTYPE-float16}" \
      --out "$WORK/submission.zip" ${EXPECT:+--expect "$EXPECT"}
  echo "DONE -> $WORK/submission.zip"
}

case "${1:-all}" in
  prep)      $PY scripts/prep_data.py ;;
  train)     train_one "${2:?train needs a NAME}" ;;
  calibrate) calibrate ;;
  all)       $PY scripts/prep_data.py
             for n in $SHIP $CV; do train_one "$n"; done
             calibrate ;;
  *) echo "usage: $0 {all|prep|train NAME|calibrate}"; exit 2 ;;
esac
