#!/usr/bin/env bash
# One-job-at-a-time queue runner for long XRSec jobs, on any machine.
#
# Upstreamed from the Miami server at the coordinator's request because DESKTOP-C lost a
# night's GPU to precisely the failure the lock prevents. Run `queue_runner.sh selftest`
# on a new machine before trusting it - the verification is a COMMAND rather than a
# procedure in a comment, because a procedure in a comment is one nobody runs.
#
# WHY FLOCK RATHER THAN CARE. DESKTOP-C ran two chain wrappers concurrently for a whole
# night: the harness killed the first chain's *tracking* while its bash process kept
# launching training jobs, the two shared .done markers, each skipped what the other
# finished, and a race started the same config twice. The fix is not a better marker
# scheme - markers are what they interleaved through. It is an exclusive lock, so a second
# runner CANNOT start. `flock -n` makes the failure immediate and loud instead of
# concurrent and silent.
#
# WHY A HEARTBEAT RATHER THAN A PIDFILE ALONE. A pidfile says a pid was written once. A
# stale pidfile whose pid has been recycled reads as "alive" and is the same class of error
# as trusting the harness's tracking. The heartbeat file's mtime answers "is it alive"
# from outside this session, by anyone, without asking the harness anything.
#
# Usage:
# WHY A MARKER RECORDS A STATE RATHER THAN AN EVENT. A marker used to mean only "do not run
# this again", so four different situations wrote the same thing: a clean completion, a
# failure whose retry was suppressed, a job parked by hand, and a marker someone touched to
# skip something. `4bca77dac0620a1c` (M-C2-lo) was PARKED on 2026-09-11 and never executed,
# and by marker alone it was indistinguishable from the 28 jobs that ran - so "29 done" was
# 28 runs plus one job that never started. A `.done` marker must mean ran-and-exited-0 and
# nothing else. Markers are now classified by their CONTENT (`rc=0`, a `.failed` sidecar, a
# `parked=` field, or none of those), and a marker that asserts nothing is reported as
# `unverified` rather than counted as a completion. The failure mode of the old scheme was
# to over-report success, which is the direction that costs an experiment.
#
# Usage:
#   queue_runner.sh add "<command>"      append a job
#   queue_runner.sh park "<command>" "<reason>"   record a job as deliberately NOT run
#   queue_runner.sh run                  start the runner; exits when the queue drains
#   queue_runner.sh daemon               same, but WAITS on an empty queue (for a service)
#   queue_runner.sh status               liveness, current job, queue depth  -- safe from anywhere
#   queue_runner.sh stop                 ask the runner to finish the current job and exit
#   queue_runner.sh selftest             verify the lock in BOTH directions, in a temp root
#   queue_runner.sh selftest-markers     verify marker classification; needs no flock/setsid,
#                                        so it runs on a Windows/Git-Bash node where `run` cannot
set -uo pipefail

ROOT="${XRSEC_QUEUE_ROOT:-/run/media/feng/Data/CalebProject/scratch/queue}"
QUEUE="$ROOT/queue.txt"
DONE_DIR="$ROOT/done"
LOG_DIR="$ROOT/logs"
LOCK="$ROOT/runner.lock"
PIDFILE="$ROOT/runner.pid"
HEARTBEAT="$ROOT/runner.heartbeat"
CURRENT="$ROOT/current.txt"
STOPFILE="$ROOT/stop"
HEARTBEAT_SECONDS=30
IDLE_POLL_SECONDS=15     # how often a daemon-mode runner re-reads an empty queue
STALE_AFTER=120          # heartbeat older than this => not alive, whatever the pidfile says

# FAIL LOUDLY IF THE ROOT IS UNREACHABLE, rather than continuing with an invisible queue.
# The default ROOT is Miami-specific. On any other machine, without XRSEC_QUEUE_ROOT set,
# `mkdir -p` failed, `touch` failed, and `status` then read a NONEXISTENT queue file and
# reported "0 pending of 0 total" - which is exactly what a genuinely drained queue looks
# like. An unreadable queue must never render as an empty one; that is the same
# absence-reported-as-a-benign-state defect as the marker scheme below, one layer down.
# The selftests build their own temp root and must not need a usable default one - a guard
# that blocks the check written to verify it is the other failure direction, and enabling a
# guard without testing that direction is how a guard stops legitimate work.
case "${1:-status}" in
    selftest|selftest-markers) SKIP_ROOT_CHECK=1 ;;
    *) SKIP_ROOT_CHECK=0 ;;
esac
if [ "$SKIP_ROOT_CHECK" = "1" ]; then
    mkdir -p "$ROOT" "$DONE_DIR" "$LOG_DIR" 2>/dev/null || true
    touch "$QUEUE" 2>/dev/null || true
elif ! mkdir -p "$ROOT" "$DONE_DIR" "$LOG_DIR" 2>/dev/null; then
    echo "FATAL: cannot create queue root '$ROOT'." >&2
    echo "       The built-in default is specific to the Miami server. Set XRSEC_QUEUE_ROOT" >&2
    echo "       to a path on THIS machine. Refusing to continue, because a queue this" >&2
    echo "       script cannot read would be reported as a queue with nothing left in it." >&2
    exit 2
fi
chmod 755 "$ROOT" "$DONE_DIR" "$LOG_DIR" 2>/dev/null || true
if [ "$SKIP_ROOT_CHECK" != "1" ] && { ! touch "$QUEUE" 2>/dev/null || [ ! -r "$QUEUE" ]; }; then
    echo "FATAL: queue file '$QUEUE' is not readable/writable. See above; same reason." >&2
    exit 2
fi

job_key() { printf '%s' "$1" | sha256sum | cut -c1-16; }

cmd_add() {
    [ $# -ge 1 ] || { echo "add: need a command" >&2; return 2; }
    printf '%s\n' "$1" >> "$QUEUE"
    echo "queued (key $(job_key "$1")): $1"
}

# Classify a job's marker by CONTENT, never by existence. Existence was the whole defect:
# it collapsed "ran and succeeded" with "parked and never started".
#   ok         -- a marker asserting rc=0. The only state that means the job ran and passed.
#   failed     -- a .failed sidecar exists (or the marker records a non-zero rc)
#   parked     -- deliberately not run, with a reason recorded
#   unverified -- a marker exists but asserts nothing: legacy, or touched by hand. NOT a pass.
#   none       -- no marker; still to do
classify_marker() {
    local key="$1" m="$DONE_DIR/$1"
    [ -f "$m.failed" ] && { echo failed; return; }
    [ -f "$m" ] || { echo none; return; }
    grep -q '^parked=' "$m" 2>/dev/null && { echo parked; return; }
    grep -q '^rc=0$'   "$m" 2>/dev/null && { echo ok; return; }
    grep -q '^rc=[1-9]' "$m" 2>/dev/null && { echo failed; return; }
    echo unverified
}

# Park a job: record that it is deliberately not being run, and why. The marker keeps the
# plain-key name so `run` skips it exactly as it skips a completion, and carries `parked=`
# so nothing ever reads it as one. The .parked sidecar exists to be greppable from outside
# without parsing, mirroring how .failed works.
cmd_park() {
    [ $# -ge 2 ] || { echo 'park: need a command AND a reason -- a park without a reason is the bug this fixes' >&2; return 2; }
    local job="$1" reason="$2" key
    key=$(job_key "$job")
    if [ -f "$DONE_DIR/$key" ] && grep -q '^rc=0$' "$DONE_DIR/$key" 2>/dev/null; then
        echo "REFUSING: $key already has a completion marker (rc=0). Parking it would overwrite" >&2
        echo "          the record of a job that actually ran. Delete the marker first if that is" >&2
        echo "          really what you want." >&2
        return 1
    fi
    printf 'job=%s\nparked=%s\nreason=%s\n' "$job" "$(date -Is)" "$reason" > "$DONE_DIR/$key"
    printf 'job=%s\nparked=%s\nreason=%s\n' "$job" "$(date -Is)" "$reason" > "$DONE_DIR/$key.parked"
    grep -qxF "$job" "$QUEUE" 2>/dev/null || printf '%s\n' "$job" >> "$QUEUE"
    echo "parked (key $key): $job"
    echo "  reason: $reason"
}

# Walk the queue once and classify every entry. Counting is done here, over the queue's own
# lines, rather than by `ls | wc -l` over the marker directory -- which double-counted every
# failure, because a failure writes both $key and $key.failed. Measured on a 3-job fixture:
# the old line reported "done: 4".
tally() {
    T_pending=0; T_ok=0; T_failed=0; T_parked=0; T_unverified=0; T_total=0
    local line state
    while IFS= read -r line; do
        [ -z "${line// }" ] && continue
        T_total=$((T_total + 1))
        state=$(classify_marker "$(job_key "$line")")
        case "$state" in
            ok)         T_ok=$((T_ok + 1)) ;;
            failed)     T_failed=$((T_failed + 1)) ;;
            parked)     T_parked=$((T_parked + 1)) ;;
            unverified) T_unverified=$((T_unverified + 1)) ;;
            none)       T_pending=$((T_pending + 1)) ;;
        esac
    done < "$QUEUE"
}

# Liveness read from the process table and the heartbeat, never from a harness's opinion.
runner_alive() {
    [ -f "$PIDFILE" ] || return 1
    local pid; pid=$(cat "$PIDFILE" 2>/dev/null) || return 1
    [ -n "$pid" ] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    # A recycled pid passes kill -0. The heartbeat is what distinguishes it.
    if [ -f "$HEARTBEAT" ]; then
        local age=$(( $(date +%s) - $(stat -c %Y "$HEARTBEAT") ))
        [ "$age" -le "$STALE_AFTER" ] || return 1
    else
        return 1
    fi
    return 0
}

cmd_status() {
    if [ "${1:-}" = "--json" ]; then cmd_status_json; return 0; fi
    echo "root:      $ROOT"
    echo "pidfile:   $PIDFILE"
    echo "heartbeat: $HEARTBEAT   (mtime is the liveness signal; stale after ${STALE_AFTER}s)"
    if runner_alive; then
        local pid age
        pid=$(cat "$PIDFILE")
        age=$(( $(date +%s) - $(stat -c %Y "$HEARTBEAT") ))
        echo "runner:    ALIVE (pid $pid, heartbeat ${age}s ago)"
        echo "current:   $(cat "$CURRENT" 2>/dev/null || echo '(between jobs)')"
    else
        echo "runner:    NOT RUNNING"
        if [ -f "$PIDFILE" ]; then
            local pid; pid=$(cat "$PIDFILE" 2>/dev/null)
            if kill -0 "$pid" 2>/dev/null; then
                echo "           WARNING: pid $pid is alive but its heartbeat is stale."
                echo "           That is a hung runner, not a dead one - investigate before starting another."
            else
                echo "           (stale pidfile for dead pid $pid)"
            fi
        fi
    fi
    # The queue is append-only and completion is tracked by markers, so the raw line
    # count is NOT what is left to do. Reporting only that number is how a drained queue
    # reads as a full one.
    tally
    echo "queued:    $T_pending pending of $T_total total"
    echo "ran ok:    $T_ok        (marker asserts rc=0)"
    echo "failed:    $T_failed"
    [ "$T_parked" -gt 0 ] && echo "parked:    $T_parked        (deliberately not run; see done/*.parked for reasons)"
    if [ "$T_unverified" -gt 0 ]; then
        echo "UNVERIFIED: $T_unverified   <-- a marker exists but asserts nothing. This is NOT a completion."
        echo "            Legacy marker, or one created by hand. Do not read it as 'ran'."
        local line key
        while IFS= read -r line; do
            [ -z "${line// }" ] && continue
            key=$(job_key "$line")
            [ "$(classify_marker "$key")" = "unverified" ] && echo "            $key :: $line"
        done < "$QUEUE"
    fi
    # A marker whose key matches no queue line: the queue was edited after the job ran, so
    # the record and the queue disagree about what was ever asked for.
    local orphans=0 f base
    for f in "$DONE_DIR"/*; do
        [ -f "$f" ] || continue
        base=$(basename "$f"); base=${base%.failed}; base=${base%.parked}
        grep -q "$base" <(while IFS= read -r l; do [ -z "${l// }" ] || job_key "$l"; done < "$QUEUE") || orphans=$((orphans + 1))
    done
    [ "$orphans" -gt 0 ] && echo "orphaned:  $orphans marker file(s) with no matching queue line"
    [ -f "$STOPFILE" ] && echo "stop:      REQUESTED"
    return 0
}

cmd_status_json() {
    local alive=false pid="" age=-1 current=""
    runner_alive && alive=true
    [ -f "$PIDFILE" ] && pid=$(cat "$PIDFILE" 2>/dev/null)
    [ -f "$HEARTBEAT" ] && age=$(( $(date +%s) - $(stat -c %Y "$HEARTBEAT") ))
    [ -f "$CURRENT" ] && current=$(tr -d '"' < "$CURRENT" | tr '\n' ' ')
    tally
    # `done` is kept as a key for existing readers but now means ran-and-exited-0 ONLY.
    # `unverified` is reported separately and must never be folded into it.
    printf '{"alive":%s,"pid":"%s","heartbeat_age_s":%s,"stale_after_s":%s,"current":"%s","pending":%s,"total":%s,"done":%s,"failed":%s,"parked":%s,"unverified":%s,"root":"%s"}\n' \
        "$alive" "$pid" "$age" "$STALE_AFTER" "$current" "$T_pending" "$T_total" \
        "$T_ok" "$T_failed" "$T_parked" "$T_unverified" "$ROOT"
}

cmd_stop() { touch "$STOPFILE"; echo "stop requested; the runner exits after the current job"; }

cmd_run() {
    exec 200>"$LOCK"
    if ! flock -n 200; then
        echo "REFUSING: another runner holds $LOCK. This is the guard working." >&2
        cmd_status >&2
        return 1
    fi
    rm -f "$STOPFILE"
    echo $$ > "$PIDFILE"
    touch "$HEARTBEAT"
    chmod 644 "$PIDFILE" "$HEARTBEAT" 2>/dev/null || true

    # `exec 200>&-` is load-bearing, not tidiness. The heartbeat subshell inherits every
    # open fd including 200, which IS the lock - and it spawns `sleep`, which inherits it
    # too. Killing the subshell can orphan that `sleep`, and an orphaned holder keeps the
    # lock for up to HEARTBEAT_SECONDS after the runner exits, so the next runner is
    # REFUSED by a process that is only sleeping. Closing the fd here means no descendant
    # can hold the lock at all. Found by `selftest` step 5 on its first run: the lock
    # blocked correctly and then failed to admit a fresh runner, which is exactly the
    # asymmetry a one-directional check cannot see.
    ( exec 200>&-; while kill -0 $$ 2>/dev/null; do touch "$HEARTBEAT"; sleep "$HEARTBEAT_SECONDS"; done ) &
    local hb=$!
    # shellcheck disable=SC2064
    trap "kill $hb 2>/dev/null; rm -f '$PIDFILE' '$CURRENT'; echo 'runner exiting'" EXIT INT TERM

    echo "runner up, pid $$, lock held on $LOCK"
    while :; do
        [ -f "$STOPFILE" ] && { echo "stop requested"; break; }
        local job=""
        while IFS= read -r line; do
            [ -z "${line// }" ] && continue
            [ -f "$DONE_DIR/$(job_key "$line")" ] && continue
            job="$line"; break
        done < "$QUEUE"
        if [ -z "$job" ]; then
            if [ "${DAEMON:-0}" = "1" ]; then
                # Idle, not finished. The heartbeat keeps ticking, so an idle runner is
                # still visibly ALIVE from outside - which is what distinguishes "waiting
                # for work" from "died quietly", a distinction DESKTOP-C did not have.
                sleep "$IDLE_POLL_SECONDS"
                continue
            fi
            echo "queue drained"
            break
        fi

        local key stamp log
        key=$(job_key "$job")
        stamp=$(date +%Y%m%d-%H%M%S)
        log="$LOG_DIR/${stamp}_${key}.log"
        printf '%s\n' "$job" > "$CURRENT"
        echo "[$(date -Is)] START $key :: $job" | tee -a "$log"

        local rc=0
        bash -c "$job" >> "$log" 2>&1 || rc=$?

        if [ "$rc" -eq 0 ]; then
            # The marker records HOW it finished, so a later reader can tell a real
            # completion from a crash without opening the log.
            printf 'job=%s\nrc=0\nfinished=%s\nlog=%s\n' "$job" "$(date -Is)" "$log" > "$DONE_DIR/$key"
            echo "[$(date -Is)] DONE  $key" | tee -a "$log"
        else
            printf 'job=%s\nrc=%s\nfailed=%s\nlog=%s\n' "$job" "$rc" "$(date -Is)" "$log" > "$DONE_DIR/$key.failed"
            echo "[$(date -Is)] FAIL  $key rc=$rc" | tee -a "$log"
            # A failure marks .failed, not .done, so it is retried on the next run rather
            # than silently skipped - the sweep convention, and the safe direction.
            # The retry-suppression marker carries `rc=` too: it used to hold the bare job
            # line, which asserted nothing, so it classified the same as a hand-made park.
            printf 'job=%s\nrc=%s\nsuppressed_retry=%s\nlog=%s\n' "$job" "$rc" "$(date -Is)" "$log" > "$DONE_DIR/$key"
            echo "  (marked done to avoid a retry loop; delete $DONE_DIR/$key to requeue)" | tee -a "$log"
        fi
        : > "$CURRENT"
    done
    return 0
}


# --------------------------------------------------------------------------------------
# The lock, verified in both directions.
#
# Checking only that a second runner is refused cannot distinguish a working lock from a
# runner that never starts at all - both produce "no second runner". So this also asserts
# that a fresh runner IS admitted once the lock is released, that jobs actually ran, and
# that a stale heartbeat reads as NOT alive even while the pid is still in the process
# table, which is the recycled-pid case a pidfile alone gets wrong.
cmd_selftest() {
    local tmp self failures=0
    tmp=$(mktemp -d) || { echo "cannot mktemp" >&2; return 2; }
    self=$(readlink -f "$0")
    echo "selftest root: $tmp"

    check() {  # check <description> <expected> <actual>
        if [ "$2" = "$3" ]; then
            echo "  PASS  $1"
        else
            echo "  FAIL  $1 (expected '$2', got '$3')"
            failures=$((failures + 1))
        fi
    }

    XRSEC_QUEUE_ROOT="$tmp" "$self" add "sleep 6; echo slow-job-ran" >/dev/null
    XRSEC_QUEUE_ROOT="$tmp" "$self" add "echo second-job-ran" >/dev/null

    setsid nohup env XRSEC_QUEUE_ROOT="$tmp" "$self" run >"$tmp/runner.out" 2>&1 &
    sleep 2

    echo "[1] runner reports alive while working"
    check "status says ALIVE" "true" \
        "$(XRSEC_QUEUE_ROOT="$tmp" "$self" status --json | grep -o '"alive":[a-z]*' | cut -d: -f2)"

    echo "[2] BLOCKS when it should - a second runner must refuse"
    XRSEC_QUEUE_ROOT="$tmp" "$self" run >"$tmp/second.out" 2>&1
    check "second runner exit code" "1" "$?"
    check "second runner said REFUSING" "yes" \
        "$(grep -q REFUSING "$tmp/second.out" && echo yes || echo no)"

    echo "[3] a stale heartbeat reads as NOT alive even with the pid alive"
    # A recycled pid passes kill -0. Only heartbeat staleness distinguishes it, so a
    # pidfile-only liveness check would report this hung state as healthy.
    touch -d "-1 hour" "$tmp/runner.heartbeat"
    check "stale heartbeat => alive false" "false" \
        "$(XRSEC_QUEUE_ROOT="$tmp" "$self" status --json | grep -o '"alive":[a-z]*' | cut -d: -f2)"
    check "pid is nonetheless still running" "yes" \
        "$(kill -0 "$(cat "$tmp/runner.pid" 2>/dev/null)" 2>/dev/null && echo yes || echo no)"

    echo "[4] waiting for the queue to drain"
    local waited=0
    while [ -f "$tmp/runner.pid" ] && kill -0 "$(cat "$tmp/runner.pid" 2>/dev/null)" 2>/dev/null; do
        sleep 1; waited=$((waited + 1))
        [ "$waited" -gt 60 ] && { echo "  FAIL  runner did not exit within 60s"; failures=$((failures + 1)); break; }
    done
    check "both jobs left done markers" "2" "$(ls -1 "$tmp/done" 2>/dev/null | grep -cv '\.failed$')"
    check "jobs ran (slow job output present)" "yes" \
        "$(grep -rq slow-job-ran "$tmp/logs" && echo yes || echo no)"
    check "markers record rc=0" "2" "$(grep -l '^rc=0$' "$tmp"/done/* 2>/dev/null | wc -l)"

    echo "[5] PASSES when it should - a fresh runner is admitted once the lock is released"
    XRSEC_QUEUE_ROOT="$tmp" "$self" add "echo third-job-ran" >/dev/null
    XRSEC_QUEUE_ROOT="$tmp" "$self" run >"$tmp/third.out" 2>&1
    check "fresh runner exit code" "0" "$?"
    check "third job ran" "yes" "$(grep -rq third-job-ran "$tmp/logs" && echo yes || echo no)"

    echo "[6] a FAILING job is marked, not silently completed"
    XRSEC_QUEUE_ROOT="$tmp" "$self" add "exit 3" >/dev/null
    XRSEC_QUEUE_ROOT="$tmp" "$self" run >/dev/null 2>&1
    check "a .failed marker exists" "1" "$(ls -1 "$tmp"/done/*.failed 2>/dev/null | wc -l)"
    check "the failed marker records rc=3" "yes" \
        "$(grep -hq '^rc=3$' "$tmp"/done/*.failed 2>/dev/null && echo yes || echo no)"

    rm -rf "$tmp"
    echo
    if [ "$failures" -eq 0 ]; then
        echo "SELFTEST PASSES: lock blocks a second runner AND admits a fresh one;"
        echo "                 stale heartbeat overrides a live pid; failures are marked."
        return 0
    fi
    echo "SELFTEST FAILED with $failures problem(s) - do not queue real work on this machine."
    return 1
}

# --------------------------------------------------------------------------------------
# Marker classification, verified in BOTH directions and WITHOUT flock/setsid.
#
# Split out from cmd_selftest deliberately. `run` needs flock and the lock test needs
# setsid, and NEITHER EXISTS IN GIT BASH FOR WINDOWS - so on a Windows node `selftest` can
# only fail, and a guard that cannot run on a machine has never been shown to work there.
# The marker half needs nothing but coreutils, so it runs everywhere the queue is read.
#
# Both directions matter here for the usual reason: a classifier that called everything
# `unverified` would "catch" the parked job and be useless, so the test asserts that a real
# completion reads `ok` just as hard as it asserts the park does not.
cmd_selftest_markers() {
    local tmp self failures=0
    tmp=$(mktemp -d) || { echo "cannot mktemp" >&2; return 2; }
    self=$(readlink -f "$0")
    echo "marker selftest root: $tmp"

    check() {
        if [ "$2" = "$3" ]; then echo "  PASS  $1"
        else echo "  FAIL  $1 (expected '$2', got '$3')"; failures=$((failures + 1)); fi
    }
    q() { XRSEC_QUEUE_ROOT="$tmp" "$self" "$@"; }

    q add "echo ran-clean"   >/dev/null
    q add "echo did-fail"    >/dev/null
    q add "echo was-parked"  >/dev/null
    q add "echo legacy-hand" >/dev/null
    q add "echo never-run"   >/dev/null

    local k_ok k_fail k_park k_legacy
    k_ok=$(printf '%s' "echo ran-clean"  | sha256sum | cut -c1-16)
    k_fail=$(printf '%s' "echo did-fail" | sha256sum | cut -c1-16)
    k_park=$(printf '%s' "echo was-parked"  | sha256sum | cut -c1-16)
    k_legacy=$(printf '%s' "echo legacy-hand" | sha256sum | cut -c1-16)

    # exactly as cmd_run writes a success
    printf 'job=echo ran-clean\nrc=0\nfinished=x\nlog=y\n' > "$tmp/done/$k_ok"
    # exactly as cmd_run writes a failure (sidecar + suppression marker)
    printf 'job=echo did-fail\nrc=4\nfailed=x\nlog=y\n' > "$tmp/done/$k_fail.failed"
    printf 'job=echo did-fail\nrc=4\nsuppressed_retry=x\nlog=y\n' > "$tmp/done/$k_fail"
    # a real park, through the command
    q park "echo was-parked" "waiting on the M-zero screen" >/dev/null
    # the 4bca case: a marker touched by hand, asserting nothing
    touch "$tmp/done/$k_legacy"

    echo "[1] each state classifies as itself - the PASSES-when-it-should direction"
    check "clean completion => ok"      "ok"         "$(XRSEC_QUEUE_ROOT=$tmp bash -c "source '$self' >/dev/null 2>&1; classify_marker $k_ok" 2>/dev/null || echo ERR)"
    check "failure => failed"           "failed"     "$(XRSEC_QUEUE_ROOT=$tmp bash -c "source '$self' >/dev/null 2>&1; classify_marker $k_fail" 2>/dev/null || echo ERR)"
    check "explicit park => parked"     "parked"     "$(XRSEC_QUEUE_ROOT=$tmp bash -c "source '$self' >/dev/null 2>&1; classify_marker $k_park" 2>/dev/null || echo ERR)"
    check "hand-touched => unverified"  "unverified" "$(XRSEC_QUEUE_ROOT=$tmp bash -c "source '$self' >/dev/null 2>&1; classify_marker $k_legacy" 2>/dev/null || echo ERR)"

    echo "[2] the tally counts the queue, not the marker directory"
    local js; js=$(q status --json)
    check "total"      "5" "$(echo "$js" | grep -o '"total":[0-9]*'      | cut -d: -f2)"
    check "done (rc=0 only)" "1" "$(echo "$js" | grep -o '"done":[0-9]*' | cut -d: -f2)"
    check "failed"     "1" "$(echo "$js" | grep -o '"failed":[0-9]*'     | cut -d: -f2)"
    check "parked"     "1" "$(echo "$js" | grep -o '"parked":[0-9]*'     | cut -d: -f2)"
    check "unverified" "1" "$(echo "$js" | grep -o '"unverified":[0-9]*' | cut -d: -f2)"
    check "pending"    "1" "$(echo "$js" | grep -o '"pending":[0-9]*'    | cut -d: -f2)"
    # 6 marker files for 5 jobs; the old `ls | wc -l` reported that as "done".
    check "marker FILES outnumber jobs (why ls was wrong)" "6" "$(ls -1 "$tmp/done" | wc -l)"

    echo "[3] a park is visible from outside without parsing, and carries its reason"
    check ".parked sidecar exists" "1" "$(ls -1 "$tmp"/done/*.parked 2>/dev/null | wc -l)"
    check "reason recorded" "yes" \
        "$(grep -q 'reason=waiting on the M-zero screen' "$tmp/done/$k_park.parked" && echo yes || echo no)"

    echo "[4] BLOCKS when it should - parking a completed job is refused"
    q park "echo ran-clean" "should be refused" >/dev/null 2>&1
    check "park over a completion exits 1" "1" "$?"
    check "the completion marker is untouched" "yes" \
        "$(grep -q '^rc=0$' "$tmp/done/$k_ok" && echo yes || echo no)"
    q park "echo no-reason-given" >/dev/null 2>&1
    check "park without a reason exits 2" "2" "$?"

    echo "[5] status names the unverified job rather than only counting it"
    check "unverified key is printed" "yes" \
        "$(q status | grep -q "$k_legacy" && echo yes || echo no)"

    rm -rf "$tmp"
    echo
    if [ "$failures" -eq 0 ]; then
        echo "MARKER SELFTEST PASSES: ok/failed/parked/unverified each classify as themselves;"
        echo "                        the tally counts jobs not files; a park needs a reason and"
        echo "                        cannot overwrite a completion."
        return 0
    fi
    echo "MARKER SELFTEST FAILED with $failures problem(s)."
    return 1
}

case "${1:-status}" in
    add)    shift; cmd_add "$@" ;;
    park)   shift; cmd_park "$@" ;;
    selftest-markers) cmd_selftest_markers ;;
    run)    cmd_run ;;
    daemon) DAEMON=1 cmd_run ;;
    status) shift; cmd_status "$@" ;;
    stop)   cmd_stop ;;
    selftest) cmd_selftest ;;
    *)      echo "usage: $0 {add <cmd>|run|daemon|status [--json]|stop|selftest}" >&2; exit 2 ;;
esac
