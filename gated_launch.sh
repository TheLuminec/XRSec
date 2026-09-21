#!/usr/bin/env bash
# gated_launch.sh - run ONE job under a hard memory cap, after proving the cap works.
#
# Written after Miami was driven to 100% RAM and its data volume corrupted (2026-09-20). The
# user's instruction: don't let it happen again. A pre-check alone cannot deliver that - a job
# that passes the check can still grow - so the job runs inside a systemd --user scope with
# MemoryMax set and swap disabled, and the kernel kills it before the machine thrashes.
#
# WHAT THIS DOES AND DOES NOT PREVENT (Miami's review, 2026-09-21): it bounds ONE job. It does
# not bound the machine - the desktop, a browser, or an UNGATED job started beside it are outside
# the cap. The lock, the active-scope check and the heavy-python check below refuse a second job
# through this script and refuse to start beside a heavy ungated one; they cannot stop a job
# launched by other means afterwards. Say "one job cannot eat the box", never "100% RAM cannot recur".
#
#   bash gated_launch.sh selftest
#   XRSEC_MARKER_DIR=/abs/path bash gated_launch.sh run <marker-name> -- <command ...>
#
# Every probe this script relies on must exist or it REFUSES (exit 2): a guard whose failure mode
# is to pass is worse than no guard (CLAUDE.md, the bc / kill -0 / flock / RSS family). The cap is
# proven in BOTH directions on this machine on every launch, AFTER the free-memory check and with
# the positive control itself capped, so the fixture is never an unguarded allocation.
#
# Env: XRSEC_MARKER_DIR (REQUIRED, absolute) - "<name>.done" written here with rc=, oom_kill=,
#          peak_mb= and timing; also holds the lock. Must NOT be on a volume the user has halted.
#      XRSEC_MIN_AVAIL_GB (default 30) - refuse unless MemAvailable is at least this;
#      XRSEC_HEADROOM_GB  (default 8)  - cap = MemAvailable - headroom.
# Linux + systemd --user with the memory controller delegated. Not for Git Bash / Windows.
set -u
MIN_AVAIL_GB="${XRSEC_MIN_AVAIL_GB:-30}"
HEADROOM_GB="${XRSEC_HEADROOM_GB:-8}"
MARKER_DIR="${XRSEC_MARKER_DIR:-}"
PY="$(command -v python3 || true)"

refuse() { echo "REFUSE: $*" >&2; exit 2; }
need() { for t in "$@"; do command -v "$t" >/dev/null 2>&1 || refuse "$t missing - a probe this guard needs is unusable"; done; }
mem_avail_gb() { awk '/MemAvailable:/ {printf "%d", $2/1048576}' /proc/meminfo; }

preflight() {
  need systemd-run systemctl pgrep ps flock awk grep
  [ -n "$PY" ] || refuse "python3 missing for the fixture"
  grep -qw memory "/sys/fs/cgroup/user.slice/user-$(id -u).slice/cgroup.controllers" 2>/dev/null \
    || refuse "memory controller not delegated to the user slice - MemoryMax would be ignored"
}

selftest() {
  preflight
  local avail; avail="$(mem_avail_gb)"
  [ "$avail" -ge 6 ] || refuse "MemAvailable ${avail} GB too low even for the fixture"
  # 1. a 1.5 GB bloater under a 512M cap MUST die (137 = SIGKILL from the cgroup OOM killer)
  systemd-run --user --scope -q -p MemoryMax=512M -p MemorySwapMax=0 "$PY" -c \
    'a=[bytearray(100<<20) for _ in range(15)]; print("SURVIVED")' >/dev/null 2>&1; local rc_kill=$?
  # 2. a 100 MB job under the same cap MUST pass
  systemd-run --user --scope -q -p MemoryMax=512M -p MemorySwapMax=0 "$PY" -c \
    'a=bytearray(100<<20); print("ok")' >/dev/null 2>&1; local rc_pass=$?
  # 3. positive control: the same bloater under a 4G cap MUST survive - proves the scope is not
  #    killing everything, and is itself capped so the fixture never allocates unguarded
  systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 "$PY" -c \
    'a=[bytearray(100<<20) for _ in range(15)]; print("SURVIVED")' >/dev/null 2>&1; local rc_ctrl=$?
  echo "selftest: 512M-capped bloater rc=$rc_kill (need 137)  512M-capped small job rc=$rc_pass (need 0)  4G-capped bloater rc=$rc_ctrl (need 0)"
  [ "$rc_kill" -eq 137 ] && [ "$rc_pass" -eq 0 ] && [ "$rc_ctrl" -eq 0 ] || refuse "memory cap not proven in both directions"
  echo "selftest: PASS"
}

one_job_only() {
  # (a) any other gated job: an active xrsec-*.scope
  local scopes; scopes="$(systemctl --user list-units --plain --no-legend --state=active 'xrsec-*' 2>/dev/null | grep -c '\.scope' || true)"
  [ "${scopes:-1}" -eq 0 ] || refuse "${scopes} xrsec-*.scope unit(s) present - one gated job at a time"
  # (b) any heavy ungated python of ours, whatever its script is called (Rack's run.py, eval_harness.py, prepare_*, ...)
  local heavy; heavy="$(ps -u "$(id -u)" -o rss=,comm= 2>/dev/null | awk '$2 ~ /python/ && $1 > 1048576 {n++} END {print n+0}')"
  [ -n "$heavy" ] || refuse "ps probe returned nothing"
  [ "$heavy" -eq 0 ] || refuse "${heavy} python process(es) over 1 GB RSS already running - not starting beside them"
  # (c) any pipeline process by name, gated or not
  # anchored on an interpreter and bracket-broken on every bare word, so the enclosing shell that
  # carries this very pattern in its own command line cannot match it (it did, on AVALON, 2026-09-21).
  # The launcher's own ancestry is excluded (the invoking shell quotes the job's script name), and so
  # is every process whose command line is byte-identical to this script's own - each $( ) or pipe
  # fork of this script carries that line and is not a job.
  local anc=" $$ " pid=$$
  while [ "$pid" -gt 1 ]; do pid="$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')"; [ -n "$pid" ] || break; anc="$anc$pid "; done
  local me; me="$(tr '\0' ' ' </proc/$$/cmdline)"
  local hits; hits="$(pgrep -f '(^|/)(python[0-9.]*|bash) .*(model/main[.]py|run[.]py|eval[_]harness|score[_]nymeria|across_xr[_]alignment|prepare[_][a-z_]+[.]py|queue[_]runner[.]sh)' 2>/dev/null)"; local prc=$?
  [ "$prc" -le 1 ] || refuse "pgrep failed (rc $prc)"
  local named=0 h other
  for h in $hits; do
    case "$anc" in *" $h "*) continue;; esac
    other="$(tr '\0' ' ' </proc/$h/cmdline 2>/dev/null)"; [ -n "$other" ] || continue   # gone already
    [ "$other" = "$me" ] && continue
    named=$((named+1)); [ -n "${XRSEC_DEBUG:-}" ] && echo "  by-name hit: pid $h: ${other:0:120}" >&2
  done
  [ "$named" -eq 0 ] || refuse "${named} pipeline process(es) alive by name (excluding this launcher's own ancestry and forks; XRSEC_DEBUG=1 lists them) - one job at a time"
}

inner() {  # runs INSIDE the scope: run the job, then read this cgroup's OOM count and peak before exiting
  local side="$1"; shift; [ "${1:-}" = "--" ] && shift
  # after the kernel OOM-kills the job, systemd's default OOMPolicy=stop TERMs everything left in the
  # scope - including this wrapper, which then never read memory.events (seen on AVALON: rc 143,
  # sidecar missing). Ignore TERM here; the read takes milliseconds and we exit on our own.
  trap '' TERM
  "$@"; local rc=$?
  local cg; cg="/sys/fs/cgroup$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup)"
  local oom peak; oom="$(awk '$1=="oom_kill"{print $2}' "$cg/memory.events" 2>/dev/null || echo NA)"
  peak="$(awk '{printf "%d", $1/1048576}' "$cg/memory.peak" 2>/dev/null || echo NA)"
  printf 'oom_kill=%s peak_mb=%s cgroup=%s\n' "${oom:-NA}" "${peak:-NA}" "$cg" > "$side"
  exit "$rc"
}

run() {
  local name="$1"; shift; [ "${1:-}" = "--" ] && shift
  [ $# -gt 0 ] || refuse "no command given"
  [ -n "$MARKER_DIR" ] || refuse "XRSEC_MARKER_DIR is unset - it must be an absolute path off any halted volume"
  case "$MARKER_DIR" in /*) ;; *) refuse "XRSEC_MARKER_DIR must be absolute, got '$MARKER_DIR'";; esac
  preflight
  mkdir -p "$MARKER_DIR" || refuse "cannot create $MARKER_DIR"
  exec 9>"$MARKER_DIR/.gated_launch.lock" || refuse "cannot open lock"
  flock -n 9 || refuse "another gated_launch holds $MARKER_DIR/.gated_launch.lock"
  local avail; avail="$(mem_avail_gb)"
  [ "$avail" -ge "$MIN_AVAIL_GB" ] || refuse "MemAvailable ${avail} GB < ${MIN_AVAIL_GB} GB (system figure, never RSS) - checked BEFORE the fixture"
  one_job_only
  selftest
  avail="$(mem_avail_gb)"; local cap=$(( avail - HEADROOM_GB ))
  [ "$cap" -ge 8 ] || refuse "cap ${cap} GB too small"
  local unit="xrsec-${name}" side="$MARKER_DIR/${name}.cgroup" marker="$MARKER_DIR/${name}.done"
  rm -f "$side"; local start; start="$(date -Is)"; local t0=$SECONDS
  echo "launch: MemAvailable=${avail}G cap=${cap}G swap=off unit=${unit}.scope marker=${marker} cmd: $*"
  systemctl --user reset-failed "$unit.scope" 2>/dev/null   # a previous OOM leaves the name in "failed"
  systemd-run --user --scope -q --unit="$unit" -p "MemoryMax=${cap}G" -p MemorySwapMax=0 -p OOMPolicy=continue \
    bash "$0" __inner "$side" -- "$@"; local rc=$?
  systemctl --user reset-failed "$unit.scope" 2>/dev/null
  local cginfo; cginfo="$(cat "$side" 2>/dev/null || echo 'oom_kill=NA peak_mb=NA (sidecar missing: inner wrapper did not survive)')"
  printf 'rc=%d %s start=%s end=%s elapsed_s=%d cap_gb=%d cmd=%s\n' "$rc" "$cginfo" "$start" "$(date -Is)" "$((SECONDS-t0))" "$cap" "$*" > "$marker"
  echo "done: rc=$rc $cginfo  marker=$marker  (rc 137 alone does not name a cause; oom_kill does)"
  exit "$rc"
}

case "${1:-}" in
  selftest) selftest ;;
  run) shift; run "$@" ;;
  __inner) shift; inner "$@" ;;
  *) echo "usage: $0 selftest | XRSEC_MARKER_DIR=/abs bash $0 run <marker-name> -- <command ...>" >&2; exit 64 ;;
esac
