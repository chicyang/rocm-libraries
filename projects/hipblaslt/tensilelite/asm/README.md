# TDM iterate-edge un-shift — 組語對照

兩份都是同一個 kernel，差別只在 un-shift 的產碼策略。

```
MatrixInstruction  [16, 16, 32, 1,  1, 8, 8,  2, 2]
VectorWidth        8 (A 和 B)
SourceSwap         true
MacroTile          256 x 256      DepthU 128      bf16 TN
TDMInst 3          TDMIterateMode 3  (coord0 與 coord1 兩個位移都開)
size               [249, 256, 1, 128]      ->  delta = (-249) mod 8 = 7
```

| 檔案 | 策略 | `s_and`（二進位） | `s_cmp_eq`（展開） | `v_mov_b64` | 非零 `s_wait_dscnt` |
|---|---|---:|---:|---:|---:|
| `unshift_v1_binary-3pass.s` | 二進位分解 3 個 pass | 6 | 0 | 0 | 13 |
| `unshift_v2_unrolled-b64-pipelined.s` | 7 份展開 + b64 合併 + bpermute pipeline | 0 | 14 | 896 | 279 |

（`v1` 的 13 個非零 `s_wait_dscnt` 來自主迴圈，不在 un-shift 區段內。）

## 兩者的差別

**v1 → v2 有三個獨立的改動**

1. **7 份展開**取代二進位分解。`delta` 是 runtime 值，以前拆成 `4+2+1` 三趟
   累積、每趟都重寫全部 512 個 accumulator；現在每個 `delta` 值各一份直線碼，
   一趟到位。
2. **`v_mov_b64` 合併**。相差 8 的兩個 `coord1` 欄在實體暫存器上連號，可以把
   兩個 `v_mov_b32` 併成一個。只有 `coord0` 方向的位移能合併——`coord1` 方向
   的合併夥伴和位移同軸，會產生環狀相依。
3. **`ds_bpermute` 軟體 pipeline**。以前每批都 `s_wait_dscnt 0` 排空管線；
   現在雙緩衝，下一批先發出去，`s_wait_dscnt 2d` 只等最舊那一批。

## 找 un-shift 區段

```
grep -n TDMIterUnshift <檔案>
```

| 想看什麼 | 找什麼 |
|---|---|
| delta / cOwn 的計算 | `rows = Size - wg*MT`、`delta = (-rows) mod` |
| 展開的分派 | `s_cmp_eq_u32 ... // delta == d` |
| 跨 lane 抓值 | `ds_bpermute_b32 ... // neighbour coal[k] prep[p]` |
| pipeline 的等待 | `s_wait_dscnt` 後面接非零值 |
| b64 合併 | `v_mov_b64 v[vgprValuC+n : vgprValuC+n+1]` |

## 重新產生

用的 yaml 就在旁邊：`tdm_edge_MT256x256x128_VW8_delta7.yaml`。不需要 GPU：

```bash
cd projects/hipblaslt/tensilelite
<venv python> Tensile/bin/Tensile \
    ../../../docs/plans/asm/tdm_edge_MT256x256x128_VW8_delta7.yaml \
    <outdir> --prebuilt-client=/nonexistent
# .s 在 <outdir>/1_BenchmarkProblems/*/00_Final/caches/*/source/build_tmp/SOURCE/assembly/
```

`SourceSwap: [false, true]` 會產出兩個變體，這裡收的是 kernel name 帶 `_SS1_`
（`SourceSwap=true`）那一個。對照組語裡的 kernel name 確認拿對檔案：

```
MT256x256x128 ... LBSPPA2048 ... LPA8 ... SS1 ... TDMI3_TDMIM3 ... VWA8_VWB8
```

## 驗證狀態

`v2` 在 gfx1250 上跑過五批全過（參考組態 + `tt1`/`tt0` × beta 0/2），0 個 INVALID。
