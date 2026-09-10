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
#   queue_runner.sh add "<command>"      append a job
#   queue_runner.sh run                  start the runner; exits when the queue drains
#   queue_runner.sh daemon               same, but WAITS on an empty queue (for a service)
#   queue_runner.sh status               liveness, current job, queue depth  -- safe from anywhere
#   queue_runner.sh stop                 ask the runner to finish the current job and exit
#   queue_runner.sh selftest             verify the lock in BOTH directions, in a temp root
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

mkdir -p "$ROOT" "$DONE_DIR" "$LOG_DIR"
chmod 755 "$ROOT" "$DONE_DIR" "$LOG_DIR" 2>/dev/null || true
touch "$QUEUE"

job_key() { printf '%s' "$1" | sha256sum | cut -c1-16; }

cmd_add() {
    [ $# -ge 1 ] || { echo "add: need a command" >&2; return 2; }
    printf '%s\n' "$1" >> "$QUEUE"
    echo "queued (key $(job_key "$1")): $1"
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
    local total=0 pending=0 line
    while IFS= read -r line; do
        [ -z "${line// }" ] && continue
        total=$((total + 1))
        [ -f "$DONE_DIR/$(job_key "$line")" ] || pending=$((pending + 1))
    done < "$QUEUE"
    echo "queued:    $pending pending of $total total"
    echo "done:      $(ls -1 "$DONE_DIR" 2>/dev/null | wc -l)"
    [ -f "$STOPFILE" ] && echo "stop:      REQUESTED"
    return 0
}

cmd_status_json() {
    local alive=false pid="" age=-1 current=""
    runner_alive && alive=true
    [ -f "$PIDFILE" ] && pid=$(cat "$PIDFILE" 2>/dev/null)
    [ -f "$HEARTBEAT" ] && age=$(( $(date +%s) - $(stat -c %Y "$HEARTBEAT") ))
    [ -f "$CURRENT" ] && current=$(tr -d '"' < "$CURRENT" | tr '\n' ' ')
    local total=0 pending=0 line
    while IFS= read -r line; do
        [ -z "${line// }" ] && continue
        total=$((total + 1))
        [ -f "$DONE_DIR/$(job_key "$line")" ] || pending=$((pending + 1))
    done < "$QUEUE"
    printf '{"alive":%s,"pid":"%s","heartbeat_age_s":%s,"stale_after_s":%s,"current":"%s","pending":%s,"total":%s,"done":%s,"root":"%s"}\n' \
        "$alive" "$pid" "$age" "$STALE_AFTER" "$current" "$pending" "$total" \
        "$(ls -1 "$DONE_DIR" 2>/dev/null | wc -l)" "$ROOT"
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
            printf '%s\n' "$job" > "$DONE_DIR/$key"
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

case "${1:-status}" in
    add)    shift; cmd_add "$@" ;;
    run)    cmd_run ;;
    daemon) DAEMON=1 cmd_run ;;
    status) shift; cmd_status "$@" ;;
    stop)   cmd_stop ;;
    selftest) cmd_selftest ;;
    *)      echo "usage: $0 {add <cmd>|run|daemon|status [--json]|stop|selftest}" >&2; exit 2 ;;
esac
