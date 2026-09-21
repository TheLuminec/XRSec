#!/usr/bin/env bash
# gated_launch.sh - run ONE job under a hard memory cap, after proving the cap works.
#
# Written after Miami was driven to 100% RAM and its data volume corrupted (2026-09-20). The
# user's instruction: don't let it happen again. A pre-check alone cannot deliver that - a job
# that passes the check can still grow - so the job runs inside a systemd --user scope with
# MemoryMax set and swap disabled, and the kernel kills it before the machine thrashes. The cap
# is proven in BOTH directions on this machine, every launch, before the real command starts:
# a bloater must die under it and a small job must survive; if either fails, or systemd-run is
# absent, this script REFUSES (exit 2) rather than falling through. A guard whose failure mode
# is to pass is worse than no guard (CLAUDE.md, the bc / kill -0 / flock / RSS family).
#
#   bash gated_launch.sh selftest
#   bash gated_launch.sh run <marker-name> -- <command ...>
#
# Env: XRSEC_MIN_AVAIL_GB (default 30) - refuse unless MemAvailable is at least this;
#      XRSEC_HEADROOM_GB (default 8)   - cap = MemAvailable - headroom;
#      XRSEC_MARKER_DIR (default ./markers) - "<name>.done" written with rc= and timing.
# Linux + systemd --user with the memory controller delegated (cpu memory pids in
# /sys/fs/cgroup/user.slice/user-<uid>.slice/cgroup.controllers). Not for Git Bash / Windows.
set -u
MIN_AVAIL_GB="${XRSEC_MIN_AVAIL_GB:-30}"
HEADROOM_GB="${XRSEC_HEADROOM_GB:-8}"
MARKER_DIR="${XRSEC_MARKER_DIR:-./markers}"

mem_avail_gb() { awk '/MemAvailable:/ {printf "%d", $2/1048576}' /proc/meminfo; }
refuse() { echo "REFUSE: $*" >&2; exit 2; }

selftest() {
  command -v systemd-run >/dev/null || refuse "systemd-run missing - no cap is possible here"
  grep -qw memory "/sys/fs/cgroup/user.slice/user-$(id -u).slice/cgroup.controllers" 2>/dev/null \
    || refuse "memory controller not delegated to user slice - MemoryMax would be ignored"
  local py; py="$(command -v python3)" || refuse "python3 missing for the fixture"
  # 1. a 1.5 GB bloater under a 512M cap MUST die (rc 137 = SIGKILL from the OOM killer)
  systemd-run --user --scope -q -p MemoryMax=512M -p MemorySwapMax=0 "$py" -c \
    'a=[bytearray(100<<20) for _ in range(15)]; print("SURVIVED")' >/dev/null 2>&1
  local rc_kill=$?
  # 2. a 100 MB job under the same cap MUST pass
  systemd-run --user --scope -q -p MemoryMax=512M -p MemorySwapMax=0 "$py" -c \
    'a=bytearray(100<<20); print("ok")' >/dev/null 2>&1
  local rc_pass=$?
  # 3. positive control: the same bloater with NO cap must survive, or the fixture proves nothing
  "$py" -c 'a=[bytearray(100<<20) for _ in range(15)]; print("SURVIVED")' >/dev/null 2>&1
  local rc_ctrl=$?
  echo "selftest: capped bloater rc=$rc_kill (need 137)  capped small job rc=$rc_pass (need 0)  uncapped bloater rc=$rc_ctrl (need 0)"
  [ "$rc_kill" -eq 137 ] && [ "$rc_pass" -eq 0 ] && [ "$rc_ctrl" -eq 0 ] || refuse "memory cap not proven in both directions"
  echo "selftest: PASS"
}

run() {
  local name="$1"; shift
  [ "${1:-}" = "--" ] && shift
  [ $# -gt 0 ] || refuse "no command given"
  selftest
  local avail; avail="$(mem_avail_gb)"
  [ "$avail" -ge "$MIN_AVAIL_GB" ] || refuse "MemAvailable ${avail} GB < ${MIN_AVAIL_GB} GB (system figure, never RSS)"
  local others; others="$(pgrep -fc 'model/main[.]py|score_nymeria[.]py|across_xr_alignment[.]py' || true)"
  [ "${others:-0}" -eq 0 ] || refuse "${others} other pipeline process(es) alive - one job at a time"
  local cap=$(( avail - HEADROOM_GB ))
  [ "$cap" -ge 8 ] || refuse "cap ${cap} GB too small"
  mkdir -p "$MARKER_DIR" || refuse "cannot create $MARKER_DIR"
  local start; start="$(date -Is)"; local t0=$SECONDS
  echo "launch: MemAvailable=${avail}G cap=${cap}G swap=off marker=${MARKER_DIR}/${name}.done cmd: $*"
  systemd-run --user --scope -q -p "MemoryMax=${cap}G" -p MemorySwapMax=0 "$@"
  local rc=$?
  printf 'rc=%d start=%s end=%s elapsed_s=%d cap_gb=%d cmd=%s\n' "$rc" "$start" "$(date -Is)" "$((SECONDS-t0))" "$cap" "$*" > "${MARKER_DIR}/${name}.done"
  echo "done: rc=$rc (137 = killed by the cap)  marker=${MARKER_DIR}/${name}.done"
  exit "$rc"
}

case "${1:-}" in
  selftest) selftest ;;
  run) shift; run "$@" ;;
  *) echo "usage: $0 selftest | run <marker-name> -- <command ...>" >&2; exit 64 ;;
esac
