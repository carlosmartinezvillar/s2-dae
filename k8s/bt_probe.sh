#!/bin/bash

set -euo pipefail

: "${RUN_NAME:?Set RUN_NAME env var, e.g. bt_probe}"
: "${WORKERS:?Set WORKERS env var}"
: "${MODEL_ID:?Set MODEL_ID env var}"

# 10# FORCES BASE 10 -- printf TREATS A LEADING ZERO AS OCTAL, SO A PADDED
# MODEL_ID LIKE 008 ERRORS OUT AND 010 SILENTLY BECOMES 008.
MODEL_ID_PADDED=$(printf "%03d" "$((10#$MODEL_ID))")

# RESOLVED FROM THIS SCRIPT, NOT THE CWD: THE JOB TEMPLATE ALREADY cd's INTO
# THE CLONE BEFORE CALLING US, SO A RELATIVE PATH WOULD LOOK ONE LEVEL TOO DEEP.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CHIPS_PVC="/chips_pvc"
MODEL_PVC="/model_pvc"
LOG_PVC="/log_pvc"
CACHE_DIR="/cache"

DATA_DIR="${CACHE_DIR}"
NET_DIR="${CACHE_DIR}/model"
LOG_DIR="${CACHE_DIR}/logs"
PARAMS_FILE="./hpo/${RUN_NAME}.json"

# PARAMS ROW LOOKUP: hp[key] IF SET, ELSE <default> IF GIVEN, ELSE hp['id']
# (bt_id FALLS BACK TO id, THE SAME RULE AS train_condition.py)
hp_get() {
  python3 -c '
import json,sys
rows = {r["id"]:r for r in map(json.loads,filter(str.strip,open(sys.argv[1])))}
row  = rows[int(sys.argv[2])]
print(row.get(sys.argv[3],sys.argv[4] if len(sys.argv) > 4 else row["id"]))
' "$PARAMS_FILE" "$MODEL_ID" "$@"
}

# RUN WHOSE CHECKPOINTS HOLD THE PRE-TRAINED BT ENCODERS (A bt_pretrain.sh OUTPUT):
# hp['bt_run'] IF SET (e.g. bt_pretrain_long), ELSE THE bt_pretrain SWEEP
BT_RUN="$(hp_get bt_run bt_pretrain)"

MODEL_OUT_DIR="${MODEL_PVC}/${RUN_NAME}"
LOG_OUT_DIR="${LOG_PVC}/${RUN_NAME}"

DONE_MARKER="${LOG_OUT_DIR}/model_${MODEL_ID_PADDED}.done"
FAIL_MARKER="${LOG_OUT_DIR}/model_${MODEL_ID_PADDED}.failed"
STDOUT_LOG_LOCAL="${LOG_DIR}/model_${MODEL_ID_PADDED}_stdout.log"
STDOUT_LOG_PVC="${LOG_OUT_DIR}/model_${MODEL_ID_PADDED}_stdout.log"

# RUN_NAME MUST NOT BE THE BT RUN: OUTPUTS WOULD LAND IN THE BT FOLDERS (on_exit
# DELETES *_<id>_e*.pth.tar THERE, THE rm BELOW DROPS ITS MARKERS) AND JOB NAMES WOULD
# COLLIDE WITH THE PRETRAIN JOBS. THE PROBE USES hpo/bt_probe.json -> bt_pretrain.json.
[ "$RUN_NAME" != "$BT_RUN" ] || { echo "RUN_NAME must differ from $BT_RUN"; exit 1; }

mkdir -p "$MODEL_OUT_DIR" "$LOG_OUT_DIR"

# A RERUN STARTS FROM A CLEAN STATE: NO STALE MARKERS, NO OLD STDOUT APPENDED TO
rm -f "$DONE_MARKER" "$FAIL_MARKER" "$STDOUT_LOG_PVC"

echo "Copying dataset files..."
for split in training validation; do
  cp "${CHIPS_PVC}/${split}_bands.npy" "$CACHE_DIR/"
  cp "${CHIPS_PVC}/${split}_masks.npy" "$CACHE_DIR/"
done

mkdir -p "$NET_DIR" "$LOG_DIR"
cd "$REPO_DIR"

# MIRROR STDOUT TO THE PVC IN THE BACKGROUND -- tee writes to local disk only,
# so the training loop never blocks on network I/O; this process copies new
# lines over to the PVC as they appear, keeping crash logs visible without
# the per-print latency hit.
touch "$STDOUT_LOG_LOCAL"
tail -F "$STDOUT_LOG_LOCAL" >> "$STDOUT_LOG_PVC" &
TAIL_PID=$!

# PRE-TRAINED CHECKPOINTS COPIED INTO NET_DIR -- DROPPED BEFORE THE COPY BACK
STAGED=()

on_exit() {
  status=$?
  kill "$TAIL_PID" 2>/dev/null || true
  cp "$STDOUT_LOG_LOCAL" "$STDOUT_LOG_PVC" 2>/dev/null || true
  if [ "$status" -eq 0 ]; then
    # FILENAMES CARRY THE EPOCH, SO cp ALONE WOULD LEAVE A PREVIOUS RUN'S
    # CHECKPOINTS NEXT TO THESE. ONLY DROPPED ON SUCCESS, SO A FAILED RERUN
    # KEEPS THE LAST GOOD ONES.
    rm -f "${STAGED[@]}"
    rm -f "${MODEL_OUT_DIR}"/*_"${MODEL_ID_PADDED}"_e*.pth.tar
    cp -v "${NET_DIR}"/*.pth.tar "$MODEL_OUT_DIR/" 2>/dev/null || true
    cp -v "${LOG_DIR}"/*.tsv "$LOG_OUT_DIR/" 2>/dev/null || true
    touch "$DONE_MARKER"
  else
    touch "$FAIL_MARKER"
  fi
  exit "$status"
}
trap on_exit EXIT

# kubectl delete SENDS SIGTERM: EXIT THROUGH on_exit SO THE .failed MARKER IS WRITTEN
trap 'exit 143' TERM INT

# COPY THE LATEST BT CHECKPOINT FOR THIS ROW -- train_condition.py READS IT FROM --net-dir
BT_ID_PADDED=$(printf "%03d" "$(hp_get bt_id)")
BT_CKPTS=("${MODEL_PVC}/${BT_RUN}/bt_${BT_ID_PADDED}_e"*.pth.tar)
[ -f "${BT_CKPTS[-1]}" ] || { echo "No bt_${BT_ID_PADDED}_e*.pth.tar in ${MODEL_PVC}/${BT_RUN}"; exit 1; }
cp -v "${BT_CKPTS[-1]}" "$NET_DIR/"
STAGED+=("${NET_DIR}/$(basename "${BT_CKPTS[-1]}")")

# BACKGROUND + wait: BASH DEFERS TRAPS UNTIL A FOREGROUND COMMAND RETURNS, BUT
# wait IS INTERRUPTED BY THEM. wait $! STILL RETURNS THE PIPEFAIL STATUS.
python3 -u train_condition.py \
  --data-dir "$DATA_DIR" \
  --net-dir "$NET_DIR" \
  --log-dir "$LOG_DIR" \
  --num-workers "$WORKERS" \
  --params "$PARAMS_FILE" \
  --id "$MODEL_ID" \
  --probe-only 2>&1 | tee "$STDOUT_LOG_LOCAL" &
wait $!
