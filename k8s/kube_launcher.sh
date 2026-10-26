#!/bin/bash

# Launch one Kubernetes Job per model id over a range of ids from a hyperparameter file.
#
#   ./k8s/kube_launcher.sh ./hpo/bt_pretrain.json 0 239
#
# WHAT HAPPENS, END TO END
#   1. This script (on your laptop) reads the template k8s/train_template.yml and
#      renders one copy per id, changing only the fields listed below.
#   2. Each rendered Job is submitted with kubectl. On the cluster the pod runs
#        git clone .../s2-dae && cd s2-dae && exec bash ./k8s/bt_pretrain.sh
#   3. bt_pretrain.sh reads RUN_NAME / MODEL_ID / WORKERS from the env, copies the
#      .npy dataset off the PVC, and runs one pretraining job:
#        ./hpo/$RUN_NAME.json, row with "id" == $MODEL_ID
#      Checkpoints/logs are copied to /model_pvc/$RUN_NAME and /log_pvc/$RUN_NAME.
#
# FIELDS REWRITTEN PER JOB (found by YAML structure, no placeholder tokens needed)
#
#   metadata.name        -> <run>-<id>      e.g. bt-pretrain-007 (RFC 1123: lowercase, no '_')
#   env MODEL_ID value   -> the id          e.g. 7               (matches 'id' in the params file)
#   env RUN_NAME value   -> the run         e.g. bt_pretrain     (script reads ./hpo/$RUN_NAME.json)
#   ./k8s/<name>.sh      -> ./k8s/$SCRIPT   e.g. bt_pretrain.sh  (the script the pod executes)
#
# Every env value is emitted quoted: the Kubernetes API requires EnvVar.value to
# be a string and rejects a bare integer.
#
# Env (all optional):
#   TEMPLATE=<path>   job template to render     (default: k8s/train_template.yml)
#   SCRIPT=<name>     script in k8s/ the pod runs (default: bt_pretrain.sh)
#   DRY_RUN=1         render and print, submit nothing
#   ASSUME_YES=1      skip the confirmation prompt
#   ALLOW_DIRTY=1     submit even if the remote does not match the working tree
#
# Examples:
#   DRY_RUN=1 ./k8s/kube_launcher.sh ./hpo/bt_pretrain.json 0 1     # inspect two manifests
#   ./k8s/kube_launcher.sh ./hpo/bt_pretrain.json 120 239           # ViT rows only
#
# Useful follow-ups:
#   kubectl get jobs | grep bt-pretrain                 # status
#   kubectl logs job/bt-pretrain-007 -f                  # follow one run
#   kubectl delete job bt-pretrain-007                   # remove one (needed before resubmitting it)

# -e: stop on any error, -u: error on unset vars, -o pipefail: a failing stage fails the pipe
set -euo pipefail

################################################################################
# SETTINGS
################################################################################
TEMPLATE="${TEMPLATE:-k8s/train_template.yml}"
SCRIPT="${SCRIPT:-bt_pretrain.sh}"
DRY_RUN="${DRY_RUN:-0}"
ASSUME_YES="${ASSUME_YES:-0}"
ALLOW_DIRTY="${ALLOW_DIRTY:-0}"

# THE CONTAINER LOOKS FOR PARAMS IN ./$PARAMS_DIRNAME/$RUN_NAME.json (SEE bt_pretrain.sh)
PARAMS_DIRNAME="hpo"

die() { echo "error: $*" >&2; exit 1; }

################################################################################
# GIT CHECK
################################################################################
# THE JOB TEMPLATE RUNS "git clone ... && bash ./k8s/bt_pretrain.sh", SO THE CLUSTER
# EXECUTES WHAT IS ON THE REMOTE, NOT THE WORKING TREE. CATCH THE MISMATCH
# BEFORE A WHOLE SWEEP TRAINS AGAINST STALE CODE.
check_git_sync() {
  git rev-parse --is-inside-work-tree >/dev/null 2>&1 || return 0

  local problems=() dirty ahead upstream

  # THE CONTAINER READS ./hpo/$RUN_NAME.json AND RUNS ./k8s/$SCRIPT OUT OF THE CLONE
  if ! git ls-files --error-unmatch "$PARAMS_FILE" >/dev/null 2>&1; then
    problems+=("$PARAMS_FILE is untracked, so the clone will not contain it")
  fi
  if ! git ls-files --error-unmatch "k8s/$SCRIPT" >/dev/null 2>&1; then
    problems+=("k8s/$SCRIPT is untracked, so the clone will not contain it")
  fi

  # ANY TRACKED EDIT MEANS THE CLONE DIFFERS FROM WHAT YOU ARE LOOKING AT
  dirty="$(git diff --name-only HEAD 2>/dev/null | tr '\n' ' ')"
  if [ -n "$dirty" ]; then
    problems+=("uncommitted changes in: ${dirty% }")
  fi

  # LOCAL COMMITS NOT ON THE REMOTE YET (CACHED REF, NO FETCH -- RUN git fetch
  # FIRST IF YOU WANT THIS TO BE AUTHORITATIVE)
  upstream="$(git rev-parse --abbrev-ref '@{upstream}' 2>/dev/null || true)"
  if [ -n "$upstream" ]; then
    ahead="$(git rev-list --count '@{upstream}..HEAD' 2>/dev/null || echo 0)"
    if [ "$ahead" != "0" ]; then
      problems+=("$ahead local commit(s) not pushed to $upstream")
    fi
  else
    problems+=("branch has no upstream, cannot tell what the remote holds")
  fi

  if [ "${#problems[@]}" -eq 0 ]; then
    echo "git         : working tree matches $upstream"
    return 0
  fi

  echo >&2
  echo "error: the template clones the remote, so the cluster would NOT run your current code:" >&2
  local item
  for item in "${problems[@]}"; do echo "  - $item" >&2; done
  echo "       commit and push, or set ALLOW_DIRTY=1 to submit anyway." >&2
  return 1
}

################################################################################
# ARGUMENTS
################################################################################
usage() {
  echo "usage: $0 <params-file> <start-id> <end-id>" >&2
  echo "   e.g: $0 ./${PARAMS_DIRNAME}/bt_pretrain.json 0 239" >&2
  exit 2
}

[ "$#" -eq 3 ] || usage
PARAMS_FILE="$1"
START_ID="$2"
END_ID="$3"

# VALIDATE ARGS
[ -f "$PARAMS_FILE" ]  || die "params file not found: $PARAMS_FILE"
[ -f "$TEMPLATE" ]     || die "template not found: $TEMPLATE (override with TEMPLATE=<path>)"
[ -f "k8s/$SCRIPT" ]   || die "script not found: k8s/$SCRIPT (override with SCRIPT=<name>; run from the repo root)"
[[ "$START_ID" =~ ^[0-9]+$ ]] || die "start-id must be a non-negative integer, got '$START_ID'"
[[ "$END_ID"   =~ ^[0-9]+$ ]] || die "end-id must be a non-negative integer, got '$END_ID'"
[ "$START_ID" -le "$END_ID" ] || die "start-id ($START_ID) is greater than end-id ($END_ID)"

# RUN_NAME IS THE PARAMS BASENAME -- bt_pretrain.sh REBUILDS THE PATH AS ./hpo/$RUN_NAME.json,
# SO A FILE OUTSIDE hpo/ OR WITH A DIFFERENT NAME WOULD NOT RESOLVE INSIDE THE CONTAINER
RUN_NAME="$(basename "$PARAMS_FILE" .json)"
[ "$(basename "$(dirname "$PARAMS_FILE")")" = "$PARAMS_DIRNAME" ] \
  || die "params file must live in ${PARAMS_DIRNAME}/ so the container can find ./${PARAMS_DIRNAME}/${RUN_NAME}.json"

################################################################################
# RENDER ONE MANIFEST
################################################################################
# REWRITE metadata.name, THE RUN_NAME / MODEL_ID env VALUES, AND THE ./k8s/<x>.sh
# CALL IN THE CONTAINER COMMAND. TARGETS THE YAML STRUCTURE, SO A TEMPLATE WITH
# REAL VALUES IN THOSE FIELDS WORKS WITHOUT PLACEHOLDER TOKENS. INPUTS COME IN
# THROUGH ENV VARS (JOB_NAME, MODEL_ID, RUN_NAME, SCRIPT, TEMPLATE, RENDERED).
render() {
  python3 - <<'PYRENDER'
import os, re, sys

job, mid, run = os.environ['JOB_NAME'], os.environ['MODEL_ID'], os.environ['RUN_NAME']
script = os.environ['SCRIPT']
lines  = open(os.environ['TEMPLATE']).read().split('\n')
hits   = {'name': 0, 'RUN_NAME': 0, 'MODEL_ID': 0, 'script': 0}
quoted = []

out, in_meta, key = [], False, None
for line in lines:
    # TOP-LEVEL metadata BLOCK -- ITS FIRST name: IS THE JOB NAME. THE OTHER
    # name: KEYS (container, env, volumes) SIT DEEPER AND ARE LEFT ALONE.
    if re.match(r'^metadata:\s*$', line):
        in_meta = True
        out.append(line)
        continue
    if in_meta:
        if re.match(r'^\S', line):
            in_meta = False
        else:
            m = re.match(r'^(\s+)name:\s*(\S.*)$', line)
            if m:
                out.append('%sname: %s' % (m.group(1), job))
                hits['name'] += 1
                in_meta = False
                continue

    # CONTAINER COMMAND: POINT "bash ./k8s/<anything>.sh" AT $SCRIPT
    if re.search(r'\./k8s/[A-Za-z0-9_.-]+\.sh', line):
        line = re.sub(r'\./k8s/[A-Za-z0-9_.-]+\.sh', './k8s/' + script, line)
        hits['script'] += 1
        out.append(line)
        continue

    # env ENTRY: REMEMBER THE KEY, THEN REWRITE THE value: LINE THAT FOLLOWS IT
    m = re.match(r'^\s*-\s*name:\s*([A-Za-z_][A-Za-z0-9_]*)\s*$', line)
    if m:
        key = m.group(1)
        out.append(line)
        continue
    if key is not None:
        m = re.match(r'^(\s*)value:\s*(.*)$', line)
        if m:
            indent, cur = m.group(1), m.group(2).strip()
            bare = cur[1:-1] if len(cur) > 1 and cur[0] == cur[-1] and cur[0] in ('"', "'") else cur
            if key == 'RUN_NAME':
                bare = run
                hits['RUN_NAME'] += 1
            elif key == 'MODEL_ID':
                bare = mid
                hits['MODEL_ID'] += 1
            elif cur == bare:
                quoted.append(key)
            out.append('%svalue: "%s"' % (indent, bare))
            key = None
            continue
        key = None
    out.append(line)

# EACH FIELD MUST APPEAR EXACTLY ONCE, OTHERWISE THE TEMPLATE IS NOT WHAT WE EXPECT
for field, n in hits.items():
    if n != 1:
        sys.exit('error: expected exactly one %s field in the template, found %d' % (field, n))
if quoted:
    sys.stderr.write('  note: quoted bare env value(s) %s -- worth fixing in the template\n'
                     % ', '.join(quoted))
open(os.environ['RENDERED'], 'w').write('\n'.join(out).rstrip('\n') + '\n')  # TRAILING NEWLINE KEEPS DRY-RUN OUTPUT READABLE
PYRENDER
}

################################################################################
# WHICH IDS TO LAUNCH
################################################################################
# IDS PRESENT IN THE PARAMS FILE (JSON LINES, ONE OBJECT PER MODEL), AS " 0 1 2 ".
# A DELIMITED STRING RATHER THAN declare -A: macOS /bin/bash IS 3.2, WHICH HAS
# NO ASSOCIATIVE ARRAYS.
HAS_IDS=" $(python3 -c '
import json,sys
for line in open(sys.argv[1]):
    line = line.strip()
    if line:
        print(json.loads(line)["id"])
' "$PARAMS_FILE" | tr '\n' ' ')"
[ -n "${HAS_IDS// /}" ] || die "no ids found in $PARAMS_FILE"

# SPLIT THE REQUESTED RANGE INTO WHAT EXISTS AND WHAT DOES NOT
TO_LAUNCH=()
MISSING=()
for ((id=START_ID; id<=END_ID; id++)); do
  case "$HAS_IDS" in *" $id "*) TO_LAUNCH+=("$id");; *) MISSING+=("$id");; esac
done
[ "${#TO_LAUNCH[@]}" -gt 0 ] || die "none of the ids ${START_ID}..${END_ID} exist in $PARAMS_FILE"

# k8s OBJECT NAMES ARE RFC 1123 DNS LABELS: LOWERCASE ALPHANUMERIC AND '-' ONLY.
# RUN_NAME VALUES LIKE 'bt_pretrain' CONTAIN AN UNDERSCORE, WHICH THE API SERVER REJECTS.
sanitize() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | tr '_' '-' \
    | sed -E 's/[^a-z0-9-]+/-/g; s/^-+//; s/-+$//'
}
NAME_PREFIX="$(sanitize "$RUN_NAME")"
[ -n "$NAME_PREFIX" ] || die "run name '$RUN_NAME' sanitizes to an empty job name prefix"

################################################################################
# PLAN + SAFETY CHECKS
################################################################################
echo "params file : $PARAMS_FILE"
echo "run name    : $RUN_NAME   (job names: ${NAME_PREFIX}-XXX)"
echo "template    : $TEMPLATE"
echo "script      : k8s/$SCRIPT"
echo "ids         : ${#TO_LAUNCH[@]} job(s) over ${START_ID}..${END_ID}"
[ "${#MISSING[@]}" -gt 0 ] && echo "skipping    : ${#MISSING[@]} id(s) absent from the params file: ${MISSING[*]}"
if [ "$DRY_RUN" = "1" ]; then
  echo "context     : (dry run, nothing will be submitted)"
else
  echo "context     : $(kubectl config current-context 2>/dev/null || echo 'unknown')"
fi

# THE CLUSTER RUNS THE REMOTE, NOT THIS DIRECTORY
if [ "$DRY_RUN" != "1" ]; then
  if ! check_git_sync && [ "$ALLOW_DIRTY" != "1" ]; then
    exit 1
  fi
fi

# CONFIRM BEFORE PUSHING JOBS TO A CLUSTER
if [ "$DRY_RUN" != "1" ] && [ "$ASSUME_YES" != "1" ]; then
  # A FAILED read UNDER set -e WOULD KILL THE SCRIPT SILENTLY, SO SAY WHY
  if ! read -r -p "Submit ${#TO_LAUNCH[@]} job(s) to the context above? [y/N] " reply; then
    die "no terminal to confirm on -- set ASSUME_YES=1 to submit non-interactively"
  fi
  case "$reply" in [yY]|[yY][eE][sS]) ;; *) echo "aborted"; exit 1;; esac
fi

################################################################################
# SUBMIT
################################################################################
# RENDER OUTSIDE THE REPO SO A STRAY MANIFEST CAN NEVER BE COMMITTED.
# ONE TEMP FILE IS REUSED FOR EVERY JOB AND REMOVED ON EXIT.
RENDERED="$(mktemp -t "kube_launcher_${NAME_PREFIX}_XXXXXX.yaml")"
cleanup() { rm -f "$RENDERED"; }
trap cleanup EXIT

SUBMITTED=0
FAILED=()
for MODEL_ID in "${TO_LAUNCH[@]}"; do
  MODEL_ID_PADDED=$(printf "%03d" "$MODEL_ID")
  JOB_NAME="${NAME_PREFIX}-${MODEL_ID_PADDED}"

  # DNS LABELS CAP AT 63 CHARACTERS
  [ "${#JOB_NAME}" -le 63 ] || die "job name '$JOB_NAME' exceeds the 63 character limit"

  JOB_NAME="$JOB_NAME" MODEL_ID="$MODEL_ID" RUN_NAME="$RUN_NAME" SCRIPT="$SCRIPT" \
  TEMPLATE="$TEMPLATE" RENDERED="$RENDERED" render \
    || die "could not render $TEMPLATE for $JOB_NAME"

  # DRY RUN: SHOW THE MANIFEST INSTEAD OF SUBMITTING IT
  if [ "$DRY_RUN" = "1" ]; then
    echo "--- $JOB_NAME (MODEL_ID=$MODEL_ID, RUN_NAME=$RUN_NAME) ---"
    cat "$RENDERED"
    SUBMITTED=$((SUBMITTED+1))
    continue
  fi

  # create, NOT apply: a Job spec is largely immutable, so a name collision should
  # fail loudly here rather than be patched into a confusing error by the API server.
  # TO RERUN AN ID, DELETE ITS OLD JOB FIRST: kubectl delete job <run>-<id>
  if kubectl create -f "$RENDERED"; then
    SUBMITTED=$((SUBMITTED+1))
  else
    echo "error: failed to submit $JOB_NAME (id $MODEL_ID)" >&2
    cp "$RENDERED" "${RENDERED%.yaml}_${MODEL_ID_PADDED}_failed.yaml"
    echo "       rendered manifest kept at ${RENDERED%.yaml}_${MODEL_ID_PADDED}_failed.yaml" >&2
    FAILED+=("$MODEL_ID")
  fi
done

################################################################################
# SUMMARY
################################################################################
echo
if [ "$DRY_RUN" = "1" ]; then
  echo "dry run: rendered $SUBMITTED manifest(s), submitted none"
else
  echo "submitted $SUBMITTED of ${#TO_LAUNCH[@]} job(s)"
fi
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "failed ids: ${FAILED[*]}" >&2
  exit 1
fi
