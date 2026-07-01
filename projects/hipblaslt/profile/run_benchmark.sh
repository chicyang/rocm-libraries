#!/usr/bin/env bash
#
# 偵測 GPU 無人使用時,自動跑 benchmark 量測。
#
# 邏輯:
#   - 開始前:等到 GPU 完全閒置 (除了本 script 自己以外沒有任何 process 在用 GPU)。
#   - 量測中:持續偵測「我以外」的新使用者。一旦出現外人使用,這次量測作廢,重新量測。
#   - 收集到指定次數的「有效」量測為止 (預設 10 次)。
#
# 用法:
#   ./run_benchmark.sh 1           # 行為 1: run.sh -> result/log_0, log_1, ...
#   ./run_benchmark.sh 2           # 行為 2: rocprofv3 counter -> result/counters/counter_0, ...
#   ./run_benchmark.sh 1 -n 5      # 只要 5 次有效量測
#   GPU_ID=0 ./run_benchmark.sh 1  # 指定要偵測 / 使用的 GPU id (預設 0)
#
set -u

# ------------------------------------------------------------------ #
# 設定
# ------------------------------------------------------------------ #
BASE_DIR="$HOME/rocm-libraries/projects/hipblaslt/tensilelite/my-custom-build"
RESULT_DIR="$BASE_DIR/result"
RUN_SH="./1_BenchmarkProblems/Cijk_Alik_Bljk_BBS_BH_UserArgs_00/00_Final/build/run.sh"
COUNTER_YAML="$HOME/rocm-libraries/projects/hipblaslt/profile/counter_mi450.yaml"
PMC="TX_PERF_SEL_VMW_LDS_BANKCONF_LOAD_CNT,TX_PERF_SEL_VMW_CROSS_PORT_SEGMENT_CONFLICT_LDS_STALLED_CYCLES"

# 本次量測會用到的程式絕對路徑 (用來辨識「屬於我這次量測」的 GPU 進程,
# 因為 rocprofv3 會把 run.sh/client fork 到獨立 process group 甚至 reparent
# 給 init,單靠父子鏈追不到)。
RUN_SH_ABS="$BASE_DIR/1_BenchmarkProblems/Cijk_Alik_Bljk_BBS_BH_UserArgs_00/00_Final/build/run.sh"
CLIENT_BIN="$BASE_DIR/tensilelite/client/tensilelite-client"

GPU_ID="${GPU_ID:-0}"          # 要偵測 / 使用的 GPU
NUM_VALID="${NUM_VALID:-10}"   # 想要的有效量測次數
IDLE_POLL_SEC=2                # 等待閒置時的輪詢間隔 (秒)
BUSY_POLL_SEC=1                # 量測中偵測外人的輪詢間隔 (秒)
IDLE_CONFIRM=2                 # 需連續幾次都閒置才視為真正閒置

# amd-smi 必須用 sudo 才看得到「所有使用者」的 GPU process。
# 非 root 執行 amd-smi process 時,只列得到部分可見的 process,別的使用者
# (例如以 root 或其他帳號跑的工作) 會被過濾掉,導致誤判 GPU 閒置而與人搶用,
# 污染效能量測。用絕對路徑避免 sudo 環境下 PATH 找不到 amd-smi。
AMD_SMI_BIN="${AMD_SMI_BIN:-/opt/rocm/bin/amd-smi}"
AMD_SMI=(sudo -n "$AMD_SMI_BIN")

# GPU kernel 崩潰偵測:某些 solution 在資源競爭下會偶發踩到非法記憶體位址
# (Error 700 / hipErrorIllegalAddress),污染整個 HIP context 後 abort (exit 134,
# 即 128+SIGABRT)。一旦發生,該次 log 從崩潰點之後的所有數據都是 INVALID/-nan,
# 不能採用,必須作廢重測。以下為判定用的 exit code 與 log 關鍵字。
CRASH_EXIT_CODE=134            # 128 + SIGABRT,client 被 abort() (core dumped)
CRASH_LOG_PATTERN='hipErrorIllegalAddress|illegal memory access|terminate called|core dumped|Aborted'

# ------------------------------------------------------------------ #
# 解析參數
# ------------------------------------------------------------------ #
MODE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        1|2) MODE="$1"; shift ;;
        -n|--num) NUM_VALID="$2"; shift 2 ;;
        -g|--gpu) GPU_ID="$2"; shift 2 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "未知參數: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$MODE" ]]; then
    echo "用法: $0 <1|2> [-n 有效次數] [-g gpu_id]" >&2
    echo "  1 = run.sh (log_N)   2 = rocprofv3 counter (counter_N)" >&2
    exit 1
fi

cd "$BASE_DIR" || { echo "找不到目錄: $BASE_DIR" >&2; exit 1; }
mkdir -p "$RESULT_DIR"
[[ "$MODE" == "2" ]] && mkdir -p "$RESULT_DIR/counters"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ------------------------------------------------------------------ #
# 啟動自我檢查:確認 sudo -n amd-smi 能免密碼執行。
# 若不行 (需要密碼 / 沒權限),amd-smi 就只看得到部分 process,無法可靠偵測
# 外人使用 GPU。此時直接報錯退出,而不是默默地漏看別人繼續搶跑。
# ------------------------------------------------------------------ #
if ! "${AMD_SMI[@]}" version >/dev/null 2>&1; then
    echo "錯誤: 無法免密碼執行 '${AMD_SMI[*]}'。" >&2
    echo "      本 script 需要 sudo 權限才能看到所有使用者的 GPU process;" >&2
    echo "      否則會漏看別人而誤判 GPU 閒置,導致與他人搶用、污染量測。" >&2
    echo "  請設定 NOPASSWD sudo (例如在 /etc/sudoers.d/ 加一行):" >&2
    echo "      $(whoami) ALL=(root) NOPASSWD: $AMD_SMI_BIN" >&2
    echo "  或改用 AMD_SMI_BIN=<path> 指定可用的 amd-smi 路徑。" >&2
    exit 1
fi

# 記錄目前正在跑的量測 process group,若本 script 被中斷 (Ctrl-C / kill),
# 一併清掉,避免 run.sh -> tensilelite-client 變孤兒繼續佔 GPU。
CURRENT_PGID=""
_CLEANING=0
cleanup_on_exit() {
    # 避免連按 Ctrl+C 或 EXIT/INT 重入把清理流程自己打斷
    [[ "$_CLEANING" == "1" ]] && return
    _CLEANING=1
    # 進到清理就不再理會後續中斷訊號,確保清乾淨
    trap '' INT TERM

    if [[ -n "$CURRENT_PGID" ]]; then
        echo ""
        log "收到中斷,正在終止量測並清理 GPU 上的 run.sh / client ..."
        # 量測是用 setsid 起在獨立 session,收不到終端的 Ctrl+C,
        # 這裡直接對整個 process group 送 SIGKILL,並用絕對路徑補殺
        # rocprofv3 逃逸出去 / reparent 給 init 的 run.sh 與 client。
        kill -KILL -- "-$CURRENT_PGID" 2>/dev/null
        pkill -KILL -f "$RUN_SH_ABS" 2>/dev/null
        pkill -KILL -f "$CLIENT_BIN" 2>/dev/null
        pkill -KILL -f "rocprofv3.*$RUN_SH_ABS" 2>/dev/null

        # 確認真的清乾淨 (最多等約 5 秒)
        local i
        for i in $(seq 1 5); do
            if ! pgrep -f "$RUN_SH_ABS" >/dev/null 2>&1 \
               && ! pgrep -f "$CLIENT_BIN" >/dev/null 2>&1; then
                break
            fi
            pkill -KILL -f "$RUN_SH_ABS" 2>/dev/null
            pkill -KILL -f "$CLIENT_BIN" 2>/dev/null
            sleep 1
        done
        log "已清理完成。"
        CURRENT_PGID=""
    fi
}
# INT/TERM: 清理後主動以非 0 結束 (128+訊號);EXIT: 收尾兜底
trap 'cleanup_on_exit; exit 130' INT
trap 'cleanup_on_exit; exit 143' TERM
trap cleanup_on_exit EXIT

# ------------------------------------------------------------------ #
# 取得目前 GPU 上「我以外」的 process PID 清單
#   $$        = 本 script
#   $BASHPID  = (相同,保險)
# 排除本 script 及其所有子孫 process。
# ------------------------------------------------------------------ #
descendant_pids() {
    # 印出 $1 及其所有子孫 PID
    local root="$1" child
    echo "$root"
    for child in $(pgrep -P "$root" 2>/dev/null); do
        descendant_pids "$child"
    done
}

# 判斷某個 GPU 上的 PID 是不是「屬於本次量測」。
# 認自己的兩個條件 (滿足任一即算自己):
#   (1) 在本 script 的 process tree 底下 (涵蓋模式 1 直接 fork 的 run.sh/client)
#   (2) 其 cmdline 明確是本次量測會用到的程式 (涵蓋模式 2:rocprofv3 會把
#       run.sh/client fork 到獨立 process group、甚至 reparent 給 init,
#       導致父子鏈追不到,但這些進程本質仍是「我這次量測的一部分」)。
#       比對用本 build 目錄下的絕對路徑,避免誤把別人的 process 認成自己。
is_mine_pid() {
    local pid="$1" mine_tree="$2" cmd
    # 條件 (1):在我的 process tree
    case "$mine_tree" in
        *" $pid "*) return 0 ;;
    esac
    # 條件 (2):cmdline 屬於本次量測的程式 (絕對路徑比對)
    cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)"
    [[ -z "$cmd" ]] && return 1
    case "$cmd" in
        *"$RUN_SH_ABS"*) return 0 ;;               # 本 build 的 run.sh
        *"$CLIENT_BIN"*) return 0 ;;               # 本 build 的 tensilelite-client
        *rocprofv3*"$RUN_SH_ABS"*) return 0 ;;     # rocprofv3 -- run.sh
    esac
    return 1
}

gpu_external_pids() {
    # 回傳:GPU 上不屬於本次量測的 PID (以空白分隔),或 PARSE_ERR
    local mine_tree pid
    mine_tree=" $(descendant_pids "$$" | tr '\n' ' ') "

    # 解析 amd-smi JSON。輸出:
    #   一行行的 PID (正常)
    #   "PARSE_ERR"  代表 amd-smi 輸出無法解析 (視為未知,保守處理,不當成閒置)
    local gpu_pids
    gpu_pids="$("${AMD_SMI[@]}" process --gpu "$GPU_ID" --json 2>/dev/null | python3 -c '
import sys, json

def walk_pids(obj, out):
    # 遞迴走訪任意結構,只要遇到 dict 有整數 pid 就收集。
    if isinstance(obj, dict):
        pid = obj.get("pid")
        if isinstance(pid, int):
            out.append(pid)
        for v in obj.values():
            walk_pids(v, out)
    elif isinstance(obj, list):
        for v in obj:
            walk_pids(v, out)

raw = sys.stdin.read().strip()
if not raw:
    print("PARSE_ERR"); sys.exit(0)
try:
    data = json.loads(raw)
except Exception:
    print("PARSE_ERR"); sys.exit(0)

pids = []
walk_pids(data, pids)
print(" ".join(str(x) for x in sorted(set(pids))))
')"

    if [[ "$gpu_pids" == *PARSE_ERR* ]]; then
        echo "PARSE_ERR"
        return 0
    fi

    for pid in $gpu_pids; do
        [[ -z "$pid" ]] && continue
        if ! is_mine_pid "$pid" "$mine_tree"; then
            echo "$pid"
        fi
    done
}

# 等到 GPU 閒置 (連續 IDLE_CONFIRM 次都沒有外人)
# PARSE_ERR (amd-smi 讀取失敗) 保守視為「非閒置」,重置確認計數繼續等。
wait_until_idle() {
    local confirm=0 ext
    while true; do
        ext="$(gpu_external_pids)"
        if [[ "$ext" == *PARSE_ERR* ]]; then
            log "amd-smi 讀取異常,無法確認 GPU 狀態,保守等待中..."
            confirm=0
        elif [[ -z "$ext" ]]; then
            confirm=$((confirm + 1))
            if [[ $confirm -ge $IDLE_CONFIRM ]]; then
                return 0
            fi
        else
            if [[ $confirm -gt 0 ]] || [[ $((RANDOM % 15)) -eq 0 ]]; then
                log "GPU $GPU_ID 有其他使用者 (PID: $(echo $ext | tr '\n' ' ')),等待閒置中..."
            fi
            confirm=0
        fi
        sleep "$IDLE_POLL_SEC"
    done
}

# ------------------------------------------------------------------ #
# 執行一次量測。回傳 0 = 有效; 1 = 期間有外人 -> 作廢。
#   $1 = 這次量測的輸出索引 N
# ------------------------------------------------------------------ #
# 徹底終止一整棵量測 process tree。
# 量測是用 setsid 起在自己的 process group (PGID == cmd_pid),
# 這裡直接對整個 group 送訊號,確保 run.sh -> tensilelite-client 這種
# 深層孫進程也一併被清掉,不會脫離變孤兒繼續佔 GPU。
kill_tree() {
    local pgid="$1"
    kill -TERM -- "-$pgid" 2>/dev/null
    # 模式 2:rocprofv3 會把 run.sh/client fork 到別的 group / reparent 給 init,
    # setsid group 殺不到,需額外用絕對路徑找出來一起清 (只清本 build 的)。
    pkill -TERM -f "$RUN_SH_ABS"  2>/dev/null
    pkill -TERM -f "$CLIENT_BIN"  2>/dev/null
    # 等它們收手 (最多約 8 秒)
    local i
    for i in $(seq 1 8); do
        if ! kill -0 -- "-$pgid" 2>/dev/null \
           && ! pgrep -f "$RUN_SH_ABS" >/dev/null 2>&1 \
           && ! pgrep -f "$CLIENT_BIN" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    kill -KILL -- "-$pgid" 2>/dev/null
    pkill -KILL -f "$RUN_SH_ABS" 2>/dev/null
    pkill -KILL -f "$CLIENT_BIN" 2>/dev/null
}

run_once() {
    local idx="$1"
    local out_file cmd_pid ext rc

    if [[ "$MODE" == "1" ]]; then
        out_file="$RESULT_DIR/log_${idx}"
        log "開始量測 log_${idx} ..."
        # setsid: 讓 run.sh 及其所有子孫獨立成一個 process group,方便整組清理
        setsid bash -c '"$0" > "$1" 2>&1' "$RUN_SH" "$out_file" &
        cmd_pid=$!
    else
        out_file="$RESULT_DIR/counters/counter_${idx}"
        log "開始量測 counter_${idx} ..."
        setsid bash -c 'rocprofv3 -E "$0" --pmc "$1" --output-format csv -d "$2" -- "$3" > "$4" 2>&1' \
            "$COUNTER_YAML" "$PMC" "$out_file" "$RUN_SH" "${out_file}.stdout.log" &
        cmd_pid=$!
    fi
    # setsid 起的子行程,其 PGID 等於自己的 PID (cmd_pid)
    local pgid="$cmd_pid"
    CURRENT_PGID="$pgid"

    # 量測進行中:持續偵測外人 (PARSE_ERR 保守作廢)
    local invalid=0
    while kill -0 "$cmd_pid" 2>/dev/null; do
        ext="$(gpu_external_pids)"
        if [[ "$ext" == *PARSE_ERR* ]]; then
            log "量測中 amd-smi 讀取異常,無法保證乾淨 -> 這次量測作廢!"
            invalid=1
            kill_tree "$pgid"
            break
        elif [[ -n "$ext" ]]; then
            log "偵測到外人使用 GPU (PID: $(echo $ext | tr '\n' ' ')) -> 這次量測作廢!"
            invalid=1
            kill_tree "$pgid"
            break
        fi
        sleep "$BUSY_POLL_SEC"
    done

    wait "$cmd_pid" 2>/dev/null
    rc=$?
    CURRENT_PGID=""

    if [[ "$invalid" == "1" ]]; then
        # 防禦性:確認整組真的清乾淨了
        kill_tree "$pgid"
        rm -rf "$out_file" "${out_file}.stdout.log" 2>/dev/null
        return 1
    fi

    # ---------------------------------------------------------------- #
    # GPU kernel 崩潰偵測 (作廢重測)
    #   偶發的 illegal memory access 會讓 client abort (exit 134),之後整份
    #   log 的量測數據全變 INVALID/-nan,不能採用。這裡在「量測正常結束
    #   (非外人打斷)」之後檢查兩種崩潰跡象:
    #     (1) exit code == 134 (SIGABRT / core dumped)
    #     (2) client 輸出出現 illegal memory access 等關鍵字
    #   任一成立即作廢,比照外人打斷處理,回傳 1 讓主流程重測。
    # 模式 1:client 輸出即 $out_file;模式 2:rocprofv3 下 client 輸出在 .stdout.log。
    # ---------------------------------------------------------------- #
    local scan_file="$out_file"
    [[ "$MODE" == "2" ]] && scan_file="${out_file}.stdout.log"

    local crashed=0
    if [[ "$rc" -eq "$CRASH_EXIT_CODE" ]]; then
        crashed=1
        log "偵測到 GPU kernel 崩潰: exit code = $rc (SIGABRT) -> 這次量測作廢!"
    elif [[ -f "$scan_file" ]] && grep -Eq "$CRASH_LOG_PATTERN" "$scan_file"; then
        crashed=1
        log "偵測到 GPU kernel 崩潰: log 出現 illegal memory access -> 這次量測作廢!"
    fi

    if [[ "$crashed" == "1" ]]; then
        kill_tree "$pgid"
        rm -rf "$out_file" "${out_file}.stdout.log" 2>/dev/null
        return 1
    fi

    if [[ "$rc" -ne 0 ]]; then
        log "警告: 量測程式 exit code = $rc (輸出仍保留於 $out_file)"
    fi
    return 0
}

# ------------------------------------------------------------------ #
# 主流程
# ------------------------------------------------------------------ #
log "模式 $MODE | GPU $GPU_ID | 目標有效量測次數: $NUM_VALID"
valid=0
attempt=0
while [[ $valid -lt $NUM_VALID ]]; do
    attempt=$((attempt + 1))

    log "等待 GPU $GPU_ID 閒置 (準備第 $((valid)) -> 第 $((valid + 1)) 次有效量測, 總嘗試 #$attempt)..."
    wait_until_idle
    log "GPU $GPU_ID 已閒置,開始量測。"

    if run_once "$valid"; then
        valid=$((valid + 1))
        log "第 $valid / $NUM_VALID 次有效量測完成。"
    else
        log "量測作廢,將重新等待閒置並重測。"
    fi
done

log "全部完成!共取得 $NUM_VALID 次有效量測 (總嘗試 $attempt 次)。"
if [[ "$MODE" == "1" ]]; then
    log "輸出: $RESULT_DIR/log_0 .. log_$((NUM_VALID - 1))"
else
    log "輸出: $RESULT_DIR/counters/counter_0 .. counter_$((NUM_VALID - 1))"
fi
