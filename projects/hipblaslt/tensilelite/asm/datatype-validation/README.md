# TDM iterate-edge — 各 input datatype 的 GPU 驗證紀錄

驗證 edge shift 的正確性是否隨 input datatype 改變。跑在 gfx1250
(`heliosr-2b805-b8-3.aus-b200.dcgpu`)，程式碼版本 `059a47c606`。

## 結果

| yaml | input | ComputeDataType | WMMA opcode | 結果 |
|---|---|---|---|---|
| `f16_hpa_MT256x256x128.yaml` | f16 (HPA) | s | `v_wmma_f32_16x16x32_f16` | 6/6 PASSED |
| `fp8_MT256x256x256.yaml` | fp8 | s | `v_wmma_scale_f32_16x16x128_f8f6f4` | 6/6 PASSED |
| `fp4_MT256x256x512.yaml` | fp4 | s | `v_wmma_scale_f32_16x16x128_f8f6f4` | 6/6 PASSED |
| `f32_MT128x128x128.yaml` | f32 | s | `v_wmma_f32_16x16x4_f32` | 6/6 PASSED |
| `bf16_batched.yaml` | bf16, batch 2..8 | s | `v_wmma_f32_16x16x32_bf16` | 6/6 PASSED |
| `f16_nonhpa_REJECTED.yaml` | f16 (非 HPA) | h | — | 0 valid solutions |

bf16 是既有的參考組態，另外驗證過（見上層 `README.md`）。

f32 的 `VectorWidth` 上限是 4（`VW x bpe <= 16`），所以 `tile_dim1` 只有 4，
奇偶 delta 都測得到但最大只到 3。MacroTile 也得縮到 128x128，否則
`DepthU 128` 的 f32 會超出 LDS。另外 `HalfPLR` 必須設 0 —— 參考組態的 3 會讓
`getHalfPLRValuStr` 在 f32 上 IndexError，那是 localReadDo 的問題，和 edge
shift 無關。

fp4 需要 `DepthU 512` 才進得了 iterate mode —— `LdsBlockSizePerPad` 是
`roundUp(DepthU x bpe x VW, 256)`，而 iterate mode 要求它大於 1024。fp4 的
`bpe` 只有 0.5，DepthU 256 只能拿到 1024。

## size 的選法

`delta = (-size) mod tile_dim1`，`delta = 0` 時整段 un-shift 被跳過。所以
**光看組語有生出 un-shift 不代表跑到了** —— size 必須讓兩個維度的 delta 都非零，
否則只驗到其中一個 tensor。

三個 datatype 的 `tile_dim1` 都是 8（`LBSPP / (DepthU x bpe)` = 2048/256）。

| size (M, N) | deltaA | deltaB |
|---|---:|---:|
| 249, 250 | 7 | 6 |
| 250, 249 | 6 | 7 |
| 249, 249 | 7 | 7 |
| 255, 255 | 1 | 1 |
| 252, 254 | 4 | 2 |
| 256, 256 | 0 | 0 (對照組) |

fp4 另外一組，因為它的 `AssertFreeElementMultiple` 是 2（packed 型別的對齊
要求），M/N 必須是偶數，**奇數 delta 測不到**：

| size (M, N) | deltaA | deltaB |
|---|---:|---:|
| 250, 252 | 6 | 4 |
| 252, 250 | 4 | 6 |
| 254, 254 | 2 | 2 |
| 250, 254 | 6 | 2 |
| 246, 246 | 2 | 2 |
| 256, 256 | 0 | 0 (對照組) |

## yaml 的兩個設定

- `SourceSwap: [true]` —— 只產 SS1。`[false, true]` 會產兩個變體，而
  `PrintWinnersOnly: True` 只印贏家，驗證表上會看不到 SS1。
- `PrintWinnersOnly: False` —— 讓每個 solution 的 validation 欄都列出來。

## f16 非 HPA 為什麼生不出來

兩道 half-ECC 守衛，都只在 `HighPrecisionAccumulate == False` 時生效，和 edge shift 無關：

- `Solution.py:2754` — `Archs with HasEccHalf require AF0EM%2==0 except for HPA kernels`
- `Solution.py:4394` — `HalfEcc requires HPA if glvw = 1`

繞過第一道（`AssertFree0ElementMultiple: 2`）之後撞第二道。歸檔的 yaml 是繞過
第一道後的版本，留著記錄第二道長什麼樣。

這條路徑是唯一能拿到非 f32 accumulator 的方式 —— WMMA 的輸出型別跟
`ComputeDataType` 走（`Common/MatrixInstructionNaming.py:153`），不是天生 f32。

## 重新產生

```bash
cd projects/hipblaslt/tensilelite/my-custom-build
source ~/venv/bin/activate
bash Tensile.sh <yaml> ./out --prebuilt-client=./tensilelite/client/tensilelite-client
```

讀 validation 欄：

```bash
awk '/^run,problem-progress/{h=1;next} h&&/^[0-9]/{print}' <log>
```
