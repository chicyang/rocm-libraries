# TDM iterate mode 位址越界導致 GPU hang — 調查紀錄

- 對象:gfx1250,solution 221(256 顆 tuning sweep 中的第 221 顆)
- kernel:`Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x240x64_..._TDMI3_..._WS32_WG64_2_1`
- 原始問題尺寸:M=4096, N=2304, K=214336(bf16, TN)
- 狀態:**根因已確認、修正已完成並驗證**;仍有一條相似路徑未評估(見「未解缺口」)

---

## 1. 摘要

TDM iterate mode 的**迭代次數是編譯期常數**,但每次迭代的 global address 前進**不受 descriptor 的 dimension 欄位約束**。當 N 不是 MacroTile1(240)的倍數時,最後一個 N 方向 workgroup 的第二個 wave component 實際只擁有少數 rows,卻照樣走完 120 rows 的位址,越界

```
越界 rows = 240 − (N mod 240)
```

與 M、K 無關。越界只會產生 page fault,**不會算錯數值**(dimension 欄位正確夾住了資料讀取),因此任何數值驗證都會 PASS。

修正:改為從「餵給 dimension 欄位的同一個執行期數值」推導迭代次數。改動 `+35 −8`,單一檔案 `Tensile/KernelWriterAssembly.py`。

---

## 2. 症狀

同事回報了兩種表現,實為**同一個 page fault 的兩種呈現**:

| 來源 | 表現 |
|---|---|
| 原始 tuning sweep | GPU hang,完全沒有輸出 |
| 隔離成單一 kernel 後跑 tensilelite | 印出 `an illegal memory access was encountered`(在 `hipEventDestroy` / `hipModuleUnload` 階段),process 存活 |

本機重現到的是**第一種**(靜默 hang)。差異原因:本機 `XNACK enabled: NO`,no-retry fault 直接鎖死 queue,exception 沒機會遞給 process,HIP 永不返回,只能靠 timeout 砍掉(`exit=137` = 128+9,是 timeout 的 SIGKILL,**不代表錯誤類型**)。

事後 GPU 進入不可用狀態:下一次執行連 code object 都載不進去,`rocm-smi --gpureset` 自己卡在 D state,需重開機。

---

## 3. 根因

### 3.1 缺陷位置

`_emitTdmIterateInit`(`Tensile/KernelWriterAssembly.py`)把迭代次數寫成編譯期常數:

```python
iter_count = rows_per_il // tile_dim1          # 240//2 // 2 = 60
...
mod.add(SMovB32(sgpr(sIter), hex(iter_count - 1), ...))   # → s_mov_b32 s16, 0x3b
```

`mt`(=`kernel["MacroTile1"]`=240)、`numComp`(=`NumWaves//2`=2)、`dim1Divisor`(=1,因 `TDMS0`)全是編譯期值,所以 `rows_per_il = 240//2 = 120` 對**每個 component 都相同**。沒有「不同 component 拿到不同 mt」的機制,而且即使有也修不掉問題(見 3.3)。

### 3.2 wave / component 結構

WS32(wave32)+ WG64_2_1 → 128 threads → `NumWaves=4`,`numComp = 4//2 = 2`。
偶數 wave 載 A,奇數 wave 載 B。B 的 `wId = fTid >> 6`(= `fTid/32/2` = waveIdx/2)→ wave1→comp0、wave3→comp1。
每個 component 負責 tile 內 120 rows,2 × 120 = 240 = MT_N,**這個切法是正確的**。

### 3.3 為何編譯期常數不可能正確

以 N=384 為例(MT_N=240 → N 方向 2 個 workgroup):

| workgroup | component | 負責 global rows | 實際存在 | 正確迭代 | 舊碼 |
|---|---|---|---|---|---|
| wg1=0 | 0 | 0–119 | 120 | 60 | 60 ✓ |
| wg1=0 | 1 | 120–239 | 120 | 60 | 60 ✓ |
| wg1=1 | 0 | 240–359 | 120 | 60 | 60 ✓ |
| wg1=1 | 1 | 360–479 | **24**(只到 383) | **12** | 60 ✗ |

四組裡三組正確,只有最後一組不對;而「哪一組不對、差多少」取決於**執行期的 N**。正確的迭代次數是 `(component, workgroup, N)` 的函數,不是 component 的屬性,因此不可能用編譯期常數表達。

### 3.4 執行期正確值原本就存在

同一個 commit 已經正確算出 per-wave 剩餘 rows 並餵給 dimension 欄位:

```python
# initTDMDescriptorWaveSeparatedImpl
mod.add(SMulI32(sgpr(tmpSgprIdx), mt, sgpr(wgIdx)))            # 240*wg1
mod.add(SSubI32(sgpr(tmpSgprIdx), sgpr(size), sgpr(tmpSgprIdx)))  # N − 240*wg1
...
mod.add(SSubU32(sgpr(dim1), sgpr(dim1), sgpr(tmpSgprWaveOffset), "consider multiple waves"))  # − wId*120
mod.add(SCMovB32(sgpr(dim1), 0, "set to 0 for waves that no enough data to load"))            # clamp
```

對應組語 line 1316–1317、1329、1331,暫存器為 **s16**。

問題在於舊碼的 `if isTdmIter:` 呼叫點落在持有 `tmpSgprIdx` 的 `with self.allocTmpSgpr(1, tag="…tmpSgprRes2")` 區塊**外面**,呼叫時 s16 已還給配置器;`_emitTdmIterateInit` 自己的 `allocTmpSgpr(2)` 又把同一顆 s16 當 `sIter` 拿回來,於是產生 `s_mov_b32 s16, 0x3b`。

> 註:這個覆寫**本身無害** —— `setTensorDim1` 在組語 line 1335/1337 已把 s16 讀走用完。它的意義是**症狀**:證明剩餘 rows 的值在呼叫點已不在有效範圍,舊碼就算想用也拿不到。

### 3.5 責任歸屬

```
2d560f932ce  chicyang  2026-06-08  [tensilelite] Support TDM iterate mode (#7976)   ← 引入缺陷
aa17c3a4209  chicyang  2026-06-25  Refine the reject for LDS (#8759)                ← 僅搬動兩行,非肇因
```

缺陷隨功能本身引入,非後續重構造成。

---

## 4. 為什麼難以發現

1. **數值永遠正確**:dimension 欄位正確夾住每次載入的資料,越界只影響位址。任何逐元素驗證都會 PASS。
2. **需要 buffer 剛好無 slack**:若同一輪測多個尺寸、以最大 N 配置記憶體,越界落在 allocation 內就完全靜默。實測 N∈[240,480,2160,2400,2304] 一起跑時全部通過,因為最大配置是 2400,留了 96 rows 的空間。
3. **觸發條件窄**:需同時滿足 iterate mode 開啟(`TDMI3`)且 N 不是 MT_N 的倍數。
4. 該 solution 的 `AssertFree1ElementMultiple = [1]`,對外宣告支援任意 N,所以沒有 assert 擋住。

---

## 5. 最小重現

關鍵手法是 `BoundsCheck: 3`(GuardPageBack)—— 它把矩陣尾端貼齊 allocation 尾端,使越界**必然**踩到未映射記憶體,因此**不依賴問題規模**。餘數只要保持 144 即可等價複現:`384 − 240×1 = 144`,與 `2304 − 240×9 = 144` 相同。

規模從原始尺寸縮減:N 6×、K 209×、workgroup 160→2、launch 1 次、B 僅 768 KB。

config:`repro_local/min_repro.yaml`

```yaml
GlobalParameters: {..., NumElementsToValidate: 0, BoundsCheck: 3,
                   NumBenchmarks: 1, SyncsPerBenchmark: 1, EnqueuesPerSync: 1, NumWarmups: 0, ...}
ProblemSizes:
  - Exact: [256, 384, 1, 1024, 256, 256, 1024, 1024]   # [M, N, Batch, K, ldC, ldD, ldA, ldB]
```

執行:

```bash
cd projects/hipblaslt/tensilelite
export PYTHONPATH="$PWD/rocisa/build/cp312-cp312-linux_x86_64/install/platlib:$PWD"
./Tensile/bin/Tensile case_001/repro_local/min_repro.yaml case_001/repro_local/min_out --build-only
INI=$(echo case_001/repro_local/min_out/1_BenchmarkProblems/*/00_Final/caches/*/source/ClientParameters.ini)
AMD_SERIALIZE_KERNEL=3 timeout -s KILL 120 ./build_tmp/tensilelite/client/tensilelite-client --config-file "$INI"
```

修正前:掛住直到 timeout,log 停在 `Log level: Debug`(28 行),無任何錯誤訊息。

---

## 6. fault 位址證據

dmesg 明確歸因到 client 的子 process,且位址精準命中預測:

```
amdgpu: [gfxhub0] no-retry page fault (pasid:747)
  Process tensilelite-cli pid 166688          ← client 166685 的子 process
  in page starting at address 0x00007d90eb800000
  ... 801000, 802000, 803000, 804000, 805000, 806000（連續往上）
```

兩條獨立推導都指向同一個 B 結尾位址:

| 推導路徑 | 計算 | 結果 |
|---|---|---|
| 2 MB hipMalloc + 大小 | `0x7d90eb600000 + 0x200000` | `0x7d90eb800000` |
| B 基底 + B 大小 | `0x7d90eb740000 + 0xC0000`(786,432 B) | `0x7d90eb800000` |

**實測第一個 fault page 就是 `0x7d90eb800000`**,與 B 的最後一個有效位元組不差一個 byte。

---

## 7. 修正

從餵給 dimension 欄位的同一個執行期數值推導迭代次數,兩者由構造保證一致 —— dimension 夾到哪裡,位址就只走到哪裡。同時把呼叫點移入持有該暫存器的 `with` 區塊,避免它被當成 iterate-init 自己的暫存配走。

產生的組語:

```
s_min_u32  s18, s16, 120   // min(本 wave 剩餘 rows, rows_per_il)
s_add_u32  s18, s18, 1     // ceil
s_lshr_b32 s18, s18, 1     // / tile_dim1(2)
s_max_u32  s18, s18, 1     // 沒有 rows 也發一次(dimension 夾住讀取)
s_sub_u32  s18, s18, 1     // 欄位編碼 n-1
```

驗證過的性質:

- 硬寫的 `0x3b` 消失
- `s_sub_u32` → `s_cmov_b32` 的 SCC 相鄰依賴**未**被 scheduler 插斷(此 codegen 會重排指令,必須回頭讀組語確認)
- 滿 tile 時算出 field=59(即 `0x3b`),與原行為**完全相同**,對齊尺寸無效能回歸

---

## 8. 驗證結果

### 8.1 數值正確性(`fix_validate.yaml`,`NumElementsToValidate: -1` 逐元素)

N = 240 / 300 / 361 / 384 / 480 / 2304 → **6/6 PASSED**

其中兩個刻意加入的邊界:

- `N=300`:餘數 60 ≤ 120,component 1 剩 **0** rows → 驗證 `s_max_u32` 下限分支
- `N=361`:餘數 121 為**奇數**,component 1 剩 **1** row → 驗證 ceil 除法

### 8.2 無越界(GuardPageBack,零 fault)

N = 384 / 361 / 300 / 2304,以及原始尺寸 4096×2304×214336 → 全部 `exit=0`,dmesg **零筆新 page fault**。

`N=361` 的結果額外證明:**dimension 欄位在單次迭代內部確實會夾住位址**,只有跨迭代的位址前進不受管。因此 ceil 到 2 rows 不會留下殘餘越界,修法是完整的,而非把 96 rows 的越界縮成 1 row。

### 8.3 原始尺寸效能(修正後,兩次獨立執行)

| log | time-us | gflops | mem-read-bytes |
|---|---|---|---|
| `gp_orig_run.log` | 2355.61 | 1.717e6(≈1.72 PFLOPS) | 32,677,507,890 |
| `orig_fix_run.log` | 2343.15 | 1.727e6(≈1.73 PFLOPS) | 32,677,507,890 |

### 8.4 同時間點 A/B(排除他人干擾)

為排除「其他人的 job 把 GPU 弄掛」的可能,同一分鐘、同一 GPU、同一 config,只換 code object:

| | Phase A(修正版 `26380f62`) | Phase B(未修版 `33d43f11`) |
|---|---|---|
| 起跑時間 | 01:38:09 | 01:38:09 |
| exit | **0**(同一秒完成) | **137**(掛滿 90s 被 SIGKILL) |
| dmesg 新增 page fault | **0 筆** | **10 筆** |
| faulting process | 無 | **全部 10 筆 = `tensilelite-cli pid 179856`** |
| 結果列 | 1 | 0 |

決定性證據:三個 dmesg 快照 **md5 完全相同**(`fc1eea1af525`):

```
ab_A_fixed_before.txt   fc1eea1af525
ab_A_fixed_after.txt    fc1eea1af525   ← 修正版跑完,kernel log 一個位元組都沒變
ab_B_orig_before.txt    fc1eea1af525   ← 未修版開跑前,仍完全相同
```

若期間有他人的 job 在 fault,dmesg 必然增長 —— 它沒有。且 Phase B 的 delta 中**只有一個 process**,10 個 fault 頁**連續**(`0x709e2e000000` → `0x709e2e009000`),起始位址 2 MB 對齊,符合 allocation 邊界。

> 註:兩次 code object 中 `Kernels.so-000-gfx1250.hsaco` 相同(共用 cache 產物),但 client 實際載入的 `TensileLibrary_gfx1250.co` 不同 —— 已確認 A/B 有效。

---

## 9. 未解缺口

1. **另一條呼叫點未評估**:`initTDMDescriptor`(非 wave-separated)仍使用編譯期常數。該路徑的 `setTensorDim1` 餵的是**完整 tensor size** 而非 per-wave 剩餘量,語意不同,在沒有證據前未修改。**值得後續確認是否有同類問題。**
2. **越界終點未量到**:所有 fault 記錄都帶 `MORE_FAULTS: 0x1`,表示 driver 合併或丟棄了其餘 fault。只觀察到 10 個連續頁(40 KB),理論越界為 48 頁(192 KB)。起點精確吻合、修正後歸零,但完整範圍無法從 dmesg 確認。
3. **未複現「有印出錯誤訊息」的變體**:需要 XNACK 開啟的機器。修正已由靜默 hang 這一側證明。

---

## 10. 環境注意事項

- **PYTHONPATH**:重開機後會遺失,`Tensile/bin/Tensile` 會以 `ImportError: cannot import name 'rocIsa' from 'rocisa'` 失敗(與程式改動無關)。`rocisa` 未系統安裝,且 `tensilelite/rocisa/` 沒有 `__init__.py`(僅 namespace portion,不會遮蔽真正的套件):

  ```bash
  export PYTHONPATH="$PWD/rocisa/build/cp312-cp312-linux_x86_64/install/platlib:$PWD"
  ```

- **不要動 `~/rocm-libraries`**:該樹的 rocisa 是 editable 安裝且 `editable.rebuild=true`,`import rocisa` 會觸發重建並汙染他人環境。本調查全程使用 `rocm-libraries2` 的獨立 build。
- **dmesg 需 sudo**:`kernel.dmesg_restrict=1`,`sudo -n` 免密碼可用。**證據請存到 case 目錄**,重開機會清掉 kernel log(本調查曾因此遺失一批 fault 記錄)。
- **這台機器上其他使用者也會產生 page fault**(觀察到多個不同 pasid),歸因**必須**依 dmesg 的 `Process … pid`,不能只看時間相近。
- **每次觸發 fault 後 GPU 需 reset**;殘留 process 會是 zombie(`Zl`)並可能仍握著 GPU context。
- `git` 反覆警告需要 gc,跨歷史查詢(如 `git log -S`)極慢;請改用範圍限定的 `git blame -L`,並加 `-c gc.auto=0` 抑制警告。

---

## 11. 檔案索引(`case_001/repro_local/`)

| 檔案 | 用途 |
|---|---|
| `min_repro.yaml` | **最小重現**:N=384, K=1024, GuardPageBack |
| `min_ctrl.yaml` | 對照組:N=480(240 的倍數,不觸發) |
| `fix_validate.yaml` | 數值回歸:6 個 N 逐元素驗證 |
| `gp_361.yaml` / `gp_300.yaml` / `gp_2304.yaml` | guard page 邊界測試 |
| `gp_orig.yaml` / `orig_fix.yaml` | 原始尺寸 4096×2304×214336 |
| `min_run.log` / `fault_run.log` | 修正前的靜默 hang(28 行,停在 `Log level: Debug`) |
| `ab_A_fixed_*.txt` / `ab_B_orig_*.txt` | 同時間點 A/B 的 dmesg before/after/delta 與 run log |
| `fix_out/` / `min_out/` | 已修 / 未修的 build 產物(含可讀組語,`KeepBuildTmp: true`) |
| `asm_keep/` | 修正前的完整註解組語(79,270 行) |
| `clsfalse.yaml` | `CompactLoopStore: false` 交叉驗證(見第 12 節) |
| `clsfalse_out/` / `clsfix_out/` | CLS=false 的未修 / 已修 build 產物 |
| `clsfalse_run.log` / `clsfalse_delta.txt` | 未修 + CLS=false 仍 fault 的 run log 與 dmesg delta |
| `bb_fix_out.log` / `bb_clsfix_out.log` | 已修版 CLS=true / CLS=false 背對背通過的 run log |

---

## 12. 交叉驗證:`CompactLoopStore` 是否才是真正原因?

同事在**另一份 yaml** 遇到類似症狀,並發現把 `CompactLoopStore: [true]` 改成 `false` 就正常。由於 `solution_221.yaml`(line 118)本身就帶 `CompactLoopStore: [true]`,這個 workaround 可以直接套在本 case 上實測,不需推測。

### 12.1 靜態檢查:CLS=false 並未關掉 TDM iterate

未修版 + `CompactLoopStore: false` 產生的組語中,越界機制**完好無損**:

| 特徵 | 出現次數 |
|---|---|
| `s_mov_b32 s?, 0x3b`(硬寫的 60 次迭代) | 1 |
| `set iterate_enable` | 1 |
| `tensor_load_to_lds` | 5 |

### 12.2 2×2 實測(N=384, GuardPageBack)

| 組合 | 新增 page fault | 結果 |
|---|---|---|
| 未修 + CLS=**true** | 10 | hang,0 筆結果 |
| 未修 + CLS=**false** | **10**(`Process tensilelite-cli pid 286387`) | hang,0 筆結果 |
| 已修 + CLS=**true** | 0 | PASS |
| 已修 + CLS=**false** | 0 | PASS |

**fault 只跟 TDM iterate 的修正連動,與 `CompactLoopStore` 無關。** 關掉 CLS 對 kernel 221 完全無效。

> 過程註記:06:03 與 06:06 兩次「已修 + CLS=false」曾 timeout 但 **page fault 為 0**。當時 load average 為 114(23 個使用者)。待負載降低後,以先前確定會通過的 `fix_out` 當基準線確認 GPU 未卡死(0.6s、exit=0),再背對背重跑,兩個已修版本皆通過並產出真實結果列 `(256,384,1,1024) BFloat16`。故該 timeout 為環境爭用,不可重現。

### 12.3 機制上為何互不相干

兩者屬於不同子系統,程式碼路徑不相交:

- **`CompactLoopStore`(寫出端)**:`KernelWriterAssembly.py:16399` 配置 `CLSm0Base` / `CLSLoopCounter`;以 `v_movrelsd_2_b32` 用 M0 索引讀 accumulator VGPR;在 store batch 上跑倒數迴圈;改變 D/C 的 `incrementToNextRow` primer 鏈。全部發生在 **epilogue 寫 D** 的階段。
- **TDM iterate(讀入端)**:A/B 的 `tensor_load_to_lds` descriptor 與迭代次數,發生在 **main loop 讀取** 階段。

CLS 的定址 bug 會表現為對 **D 的越界寫入**,同樣會是 illegal memory access —— 症狀相同,根因不同。

### 12.4 尚待同事確認(無法單方面判定)

同事的 case 有兩種可能,需要他那份 yaml 才能區分:

1. **真的是另一個 bug**:CLS store 路徑自身的定址問題(與本 case 無關的獨立缺陷)。
2. **同一個 TDM bug 被意外遮住**:改 `CompactLoopStore` 會改變產生/勝出的 solution 集合,可能只是碰巧換掉了那顆有問題的 kernel,而非修掉根因。

判別方式:檢查他那顆出問題的 solution 名稱是否含 **`TDMI`**(TDM iterate 開啟),以及 **N 是否為 MacroTile N 的倍數**。若含 TDMI 且 N 不是倍數,則很可能是同一個根因,`CompactLoopStore=false` 只是遮住症狀。

---

## 13. 對抗式 review 結果(attack / defense)

對本修正做了一輪敵對審查。以下每一項都標明**驗證狀態**,未經實證的不當成結論。

### 13.1 S1 — 修正本身可能不完整(最高優先,未實測)

**我先前的驗證論證有漏洞,這點必須更正。** §8.2 曾主張「N=361 通過,證明 ceil 的殘留無害」。但 N=361 時該 component 的剩餘列數 R=1 而 `tile_dim1=2` —— 它是因為 **R < tile_dim1** 被夾住,不是因為 ceil 無害。兩者不同。

用既有證據可以反推硬體行為:若 fence 隨位址前進而遞減,原本 bug 中超出 fence 的步就會讀 0 列、不觸碰記憶體、**不會 fault** —— 但實測確實 fault(§4)。故 fence 更可能固定,每步讀滿 `min(tile_dim1, dimension)`。在該模型下,只要 `R ≥ tile_dim1` 且 `R % tile_dim1 ≠ 0`,**最後一步會多讀 `tile_dim1 − (R mod tile_dim1)` 列**。

已驗證的 6 個 N 全部避開這個窗口:240 / 480 / 2304 整除;300 的 R=60 為偶數;384 的 R=24 為偶數;361 的 R=1 過小。**等於這個殘留從未被測到。**

判定測試:**N=243**(R=3、`ceil(3/2)=2` 步讀 4 列、只有 3 列存在 → 預期 1 列 = 2048 B 越界)。config 為 `atk_s1.yaml`,已建置成功(runtime 路徑在位),**待 guard page 實測**。

### 13.2 A1 — TDMSplit 的第二次 load 沿用同一個迭代次數(已確認為真)

`TDMSplit` 下 `globalReadDo` 會從同一個 descriptor 發**兩次** load。兩次之間位址已被推進、dimension 被改寫成 `H1 = H0 − halfRows`,但 `iterations` 沿用不變:

```11532:11537:tensilelite/Tensile/KernelWriterAssembly.py
            imod.middle.add(SSubU32(sgpr(h1), sgpr(h0), sgpr(hr), "H1 = H0 - halfRows"))
            imod.middle.add(SCSelectB32(sgpr(h1), 0, sgpr(h1), "clamp H1 to 0"))
            imod.middle.add(comp.setTensorDim1(group1, h1, self))
            comp.setMemToken([self.states.memTokenLdsSplit[tdmParity][1]])
            imod.middle.add(comp.issueLoad("tdmAGroup0", "tdmAGroup1", tdmAGroup2, tdmAGroup3))
            imod.middle.add(comp.setTensorDim1(group1, h0, self))
```

組語實證(`atk_a1.yaml` = `min_repro.yaml` + `TDMSplit: [true]`,建置成功):

| 項目 | 觀察 |
|---|---|
| `tensor_load_to_lds` | **10** 條(非 split 為 5)|
| `s_min_u32`(迭代次數計算) | **僅 1 次**,在 asm 1340 |
| `setIterations` 寫入 | **僅 1 次**,在 asm 1346 |
| 五對 load(1512/1537、2181/2206、2414/2435、2625/2646、7138/7169) | 中間**沒有**任何 `s[tdmBGroup2+3]` 重寫 |
| asm 7162 / 7170 | `// TDM set tensor dim 1` 包住第二次 load → dimension 每半改寫 |

即「dimension 跟著變、iterations 不跟」的同一個病換了地點。`TDMSplit` 只是布林參數,`Solution.py` 沒有任何規則把它與 iterate mode 綁定,且既有測試(`Tests/common/streamk/gfx1250/core/sk_bgemm_tdm_split.yaml`)本來就在 sweep `TDMSplit: [False, True]`。

**這不是本修正造成的退步**:修正前第二次 load 用的是常數,同樣過長;修正後只會更短或相同,絕不更長。屬於「沒修到」。

### 13.3 A2 — 非 2 次方 `tile_dim1` 靜默退回常數(可達性未證實)

`runtimeIterCount`(line 19178)在 `tile_dim1` 非 2 次方時會**安靜地**退回 `SMovB32(sIter, iter_count-1)`,而這正是造成 hang 的那條指令。函式內其他前置條件(19149 / 19166 / 19172)全都 `raise RuntimeError`,只有這一個靜默。

嘗試建構可達配置,兩次都被拒:

| 配方 | 結果 |
|---|---|
| `LdsBlockSizePerPadB: 768` + `LdsPadB: 0` | `LdsPadB=0` 連帶使 `LdsBlockSizePerPadB` 被歸零(`Solution.py:3071`)→ `tile_dim1(0)` → 撞上**會大聲 raise** 的守衛 |
| `LdsBlockSizePerPadB: 3072` + `LdsPadB: 8`(`tile_dim1=24`,非 2 次方,整除 `rows_per_il=120`,且 `24 % 8 == 0`) | `reject: can't pad by addrVgpr or instOffset`(pad 位址模擬) |

`Solution.py:3345-3352` 確實對 iterate 模式的 A/B 跳過 2 次方檢查,但其他 pad 檢查仍把 `LdsBlockSizePerPad` 限制在剛好產生 2 次方 `tile_dim1` 的值。兩次失敗不足以證明不可達,**但可達性比原先聲稱的困難得多**。

無論可達與否,**程式碼品質的反對意見仍然成立**:在一個所有前置條件都 raise 的函式裡,獨獨這一個靜默退回已知會 hang 的路徑,是錯誤的失敗模式。應改為 `raise` 或實作通用除法。

### 13.4 A3 — 非 wave-separated 路徑(NumWaves=1)完全未修(驗證中)

`initTDMDescriptor` 的呼叫點(line 19457-19461)不傳 `remainRowsSgpr`,仍用編譯期常數。§9.1 原本列為「未評估」,現已確認**repo 內有現成測試可達**:

`Tests/common/gemm/gfx12/tdm_gfx1250.yaml` 第 221 行該組為 `TDMIterateMode: [-1, 0, 2]`,其 MatrixInstruction 清單含 `[16,16,32,1,1,1,1,1,1]` → MT 16×16、**NumWaves=1**,而同組 line 256 有 `Exact: [127, 127, 1, 128]`,127 % 16 = 15 為 partial tile。(以上皆獨立核對過。)

未決的關鍵:該路徑的 dimension 取**完整 tensor size**(line 19420),base 卻已被推進到本 workgroup 的 tile,座標語意與 WS 路徑不同,硬體是否夾得住需實證。

### 13.5 gfx12 YAML 全面掃描

63 個 YAML 中僅 **2 組**會踩到(iterate 開啟且對應維度非 MacroTile 倍數):

| 檔案 / 組 | 尺寸 | MT_N | NumWaves | 本修正 |
|---|---|---|---|---|
| `mxf8_gfx1250.yaml` TN g3 | `[256, 257, 1, 2048]` | 128 / 256 | 4 | 已覆蓋 |
| `tdm_gfx1250.yaml` NN g3 | `[127, 127, 1, 128]` | 32 | 4 | 已覆蓋 |
| `tdm_gfx1250.yaml` NN g3 | `[127, 127, 1, 128]` | 16 | **1** | **未覆蓋** |

這兩組本來就在 repo 裡且應為通過 —— 與 §5 一致:沒有 guard page 時越界落在配置餘裕內,不 fault、數值也正確,故此 bug 在一般 CI 下是**靜默**的。這兩個尺寸是現成的回歸測試素材(本修正目前**沒有** repo 內回歸測試,`fix_validate.yaml` 只在 case 目錄)。

### 13.6 敵對審查判定為安全的項目

以下經獨立查核確認無誤,其中前兩項與我自己的 review 結論一致:

| 項目 | 理由 |
|---|---|
| 搬移呼叫位置造成的 clobber | `setIterationEnabled` 只動 `Group1+0` bit 19;之後的 `setTensorTile0/1`、`setTensorStride0` 只動 `Group1+3/+4/+5/+6`。組語確認 `set iterate_enable` 後僅 `+3`、`+4` 被動 |
| SGPR 壓力 | 峰值不變,前後皆 3 個暫存器 |
| SCC 相鄰性 | `SSubU32`/`SCMovB32` 仍相鄰;新增的 SCC 寫入全在其後,且中間的 `setTensorDim1` 不消費 SCC。**這段程式碼沒有任何重排 pass 會經過**(`makeSchedule` 只排 unroll loop 內的 global-read / local-write) |
| iterate 路徑的 tail loop | `tdmDescIdx=1`,`resetTensorDimForTail` 改寫的是 **dim0**(K 範圍),不是 dim1,故不會再造成 dimension/iterations 不一致 |
| MXS / Metadata / Sparse | `_TDMIterateMode` 只對 A/B 設定,`isTdmIter` 對這些為 False,不會進入該函式 |
| fp4 / fp6 | `global_inc` 的位元組換算不影響**列**計數,`tile_dim1` 與 `rows_per_il` 皆為純列數 |
| 數值範圍 / 溢位 / 字面值 | `SMinU32` 已把上限壓在編譯期 `rows_per_il`,`SAddU32` 不會溢位;結果受 `iter_count ≤ 256` 保證,符合 16-bit 欄位;每條 SOP2 最多一個字面值;`tile_dim1 == 1` 退化為位移 0 仍正確 |
| `numComp == 1`(NumWaves=2) | `wId=0` → waveOffset=0 → `dim1 = size − mt·wg`,仍是正確的 per-wave 餘量 |

### 13.7 待辦優先序

1. **N=243 guard page 實測**(13.1)—— 唯一能判定本修正是否完整的測試
2. A3 非 WS 路徑實證(13.4)
3. A1 TDMSplit 修復:在第二次 load 前依 `H1` 重算 `iterations`(13.2)
4. A2 改為 `raise` 或通用除法(13.3)
5. 把 `[256,257,1,2048]` 與 `[127,127,1,128]` 加成 repo 內回歸測試(13.5)
