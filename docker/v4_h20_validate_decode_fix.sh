#!/usr/bin/env bash
# ---------------------------------------------------------------------------- #
#  Validate the DeepSeek-V4-Flash multi-GPU decode-deadlock fix on H20 (sm90).
#
#  WHAT THIS CHECKS
#  ----------------
#  The branch adds three collective-safety fixes to the DP-decode path:
#    Fix #1  collective-safe _sync_decode_uuids_tensor / _sync_completion_status_tensor
#            (no early-return before the all_reduce -> idle ranks no longer skip
#             a decode-entry collective the other ranks run).
#    Fix #2  global all_reduce(MAX) break-validation before the decode-entry
#            `break` (ranks leave the loop together; fail-fast on desync).
#    Markers BATCHGEN_DECODE_DEADLOCK_TRACE=1 emits per-rank fd-2 markers at the
#            decode-entry syncs and the per-layer MoE collectives.
#
#  Pre-fix behaviour: decode HANGS on the first inference (7 ranks GPU-spin in
#  _ep_all_gather, 1 rank asleep on a futex). The fix was validated on H20 MP8:
#  with the markers on, decode advances through all 43 layers with all 8 ranks
#  in lockstep (per-rank last marker identical / one async step apart).
#
#  PASS SIGNAL: the per-rank [DDL] markers PROGRESS in lockstep (see `diagnose`)
#  — a deadlocked group freezes at one marker. NOTE: a curl timeout alone does
#  NOT mean a hang: the first successful decode triggers a cold torch-JIT compile
#  that can exceed the curl timeout. Use `diagnose` to tell a real hang (frozen
#  markers) from slow-but-progressing JIT (advancing markers). Pre-warm the JIT
#  cache with `v4_h20_rebuild_and_launch.sh warmup` to avoid the cold-compile
#  stall during the smoke.
#
#  USAGE (on the H20 node, after syncing current source — see
#  v4_h20_rebuild_and_launch.sh header for the rsync line)
#  --------------------------------------------------------------------------
#    bash docker/v4_h20_validate_decode_fix.sh build     # rebuild from HEAD
#    bash docker/v4_h20_validate_decode_fix.sh run        # launch+wait+decode+verdict
#    bash docker/v4_h20_validate_decode_fix.sh diagnose    # wchan + markers if hung
#    bash docker/v4_h20_validate_decode_fix.sh stop
#  One-shot:
#    bash docker/v4_h20_validate_decode_fix.sh all         # build+run+(diagnose on hang)
#
#  All config is inherited from v4_h20_rebuild_and_launch.sh; override via env
#  the same way (DEVICES, WORLD_SIZE, CKPT_DIR, ...). Set WORLD_SIZE=8 to repro
#  the MP8 split.
# ---------------------------------------------------------------------------- #
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNBOOK="$HERE/v4_h20_rebuild_and_launch.sh"
[ -f "$RUNBOOK" ] || { echo "missing $RUNBOOK"; exit 1; }

CONTAINER="${CONTAINER:-leyang-v4-fullrun}"
PORT="${PORT:-10944}"
LOG="/tmp/v4_h20_server.log"
DECODE_TOKENS="${DECODE_TOKENS:-32}"
SMOKE_TIMEOUT="${SMOKE_TIMEOUT:-180}"

cmd_build() { bash "$RUNBOOK" build; }

cmd_launch() {
  # Inject the trace flag into the server process via the runbook's RUN_ENV_EXTRA hook.
  RUN_ENV_EXTRA="BATCHGEN_DECODE_DEADLOCK_TRACE=1 ${RUN_ENV_EXTRA:-}" \
    bash "$RUNBOOK" launch
}

cmd_wait() { bash "$RUNBOOK" wait; }

# A decode that emits DECODE_TOKENS tokens — the operation that hung pre-fix.
cmd_decode_smoke() {
  echo ">>> Decode smoke: $DECODE_TOKENS tokens, timeout ${SMOKE_TIMEOUT}s (this is the op that hung pre-fix)"
  local body resp rc
  body="{\"prompts\":[\"Write three sentences about the ocean.\"],\"max_output_len\":$DECODE_TOKENS,\"temperature\":0.0}"
  resp=$(docker exec "$CONTAINER" bash -lc \
    "curl -s -m $SMOKE_TIMEOUT -X POST http://127.0.0.1:$PORT/v1/inference \
       -H 'Content-Type: application/json' -d '$body'")
  rc=$?
  echo ">>> curl rc=$rc"
  echo ">>> response: $resp"
  if [ "$rc" -ne 0 ]; then
    echo ">>> curl did not return within ${SMOKE_TIMEOUT}s. This is NOT proof of a"
    echo "    hang — a cold torch-JIT compile can exceed the timeout. Run"
    echo "    '$0 diagnose' and check whether the [DDL] markers are PROGRESSING"
    echo "    (slow JIT, fix OK) or FROZEN (real deadlock)."
    return 1
  fi
  # A non-empty completion field means decode actually produced tokens.
  if echo "$resp" | grep -qE '"(text|output|completion|generated_text)"\s*:\s*"[^"]'; then
    echo ">>> PASS: decode returned tokens."
    return 0
  fi
  echo ">>> AMBIGUOUS: curl returned but no obvious token field. Inspect response above."
  return 2
}

# If decode hangs, snapshot WHY: per-rank wchan + the last marker each rank hit.
cmd_diagnose() {
  echo "=========================================================="
  echo ">>> DIAGNOSE: server process states (R = GPU-spin in collective, futex = asleep pre/post collective)"
  docker exec "$CONTAINER" bash -lc '
    for pid in $(pgrep -f launch_http_server); do
      st=$(cat /proc/$pid/stat 2>/dev/null | awk "{print \$3}")
      wch=$(cat /proc/$pid/wchan 2>/dev/null)
      echo "pid=$pid state=$st wchan=${wch:-<running>}"
    done'
  echo "----------------------------------------------------------"
  echo ">>> Last decode-deadlock marker per rank (tail of server log):"
  docker exec "$CONTAINER" bash -lc "grep '\[DDL\]' $LOG | tail -40"
  echo "----------------------------------------------------------"
  echo ">>> Per-rank: very last [DDL] marker (where each rank stalled):"
  docker exec "$CONTAINER" bash -lc "
    grep '\[DDL\]' $LOG | sed -E 's/.*rank=([0-9?]+) /\1\t/' \
      | awk -F'\t' '{last[\$1]=\$2} END {for (r in last) print \"rank \" r \": \" last[r]}' | sort -V"
  echo "=========================================================="
  echo ">>> INTERPRETATION GUIDE:"
  echo "  - Run diagnose TWICE a few seconds apart and compare the per-rank last"
  echo "    markers. If they ADVANCE (e.g. L=9 -> L=20), decode is progressing —"
  echo "    not a hang, just slow (cold JIT). PASS."
  echo "  - If the markers are FROZEN at the same point across both snapshots, it"
  echo "    is a real deadlock. Where they froze localises it:"
  echo "      * a rank stuck at 'sync_*:before_*reduce' while others passed"
  echo "        'after' => a collective-skip is live (should be impossible after"
  echo "        the fix — re-check the diff landed in the running image)."
  echo "      * N-1 ranks at 'moe:before_*_ag' with the owner at"
  echo "        'decode_cont:before_v4_metadata' / a KV path => a NEW MoE-internal"
  echo "        stall, distinct from the entry-sync bug this fix addresses."
}

cmd_stop() { bash "$RUNBOOK" stop; }

cmd_run() {
  cmd_launch || return 1
  cmd_wait   || { echo ">>> server never became ready (not a decode-deadlock issue; see $LOG)"; return 1; }
  if cmd_decode_smoke; then
    echo ">>> VERDICT: DECODE FIX VALIDATED — smoke returned tokens."
    return 0
  fi
  echo ">>> Smoke did not return tokens. Capturing markers/wchan to tell a real"
  echo ">>> hang (frozen markers) from slow cold-JIT (advancing markers)."
  cmd_diagnose
  return 2
}

case "${1:-help}" in
  build)        cmd_build ;;
  launch)       cmd_launch ;;
  wait)         cmd_wait ;;
  decode-smoke) cmd_decode_smoke ;;
  diagnose)     cmd_diagnose ;;
  run)          cmd_run ;;
  stop)         cmd_stop ;;
  all)          cmd_build && cmd_run ;;
  *) echo "usage: $0 {build|launch|wait|decode-smoke|diagnose|run|stop|all}"; exit 1 ;;
esac
