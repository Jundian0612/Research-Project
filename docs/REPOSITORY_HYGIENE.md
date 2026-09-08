# GitHub 檔案整理建議

2026-09-08 依本機 Git index 檢查。GitHub 網頁此次未成功取得，以下不是遠端最新 main 的獨立核對。
大小為工作目錄中已追蹤檔案的表面大小加總（符號連結可能重複計算），不是 Git 壓縮大小或下載量。

| 項目 | 本機追蹤狀態 | 建議 |
|---|---|---|
| `.conda/` | 8,547 個檔案，約 477 MiB | 停止追蹤；每台重新建立環境 |
| `.venv/` | 4 個檔案，約 23 MiB | 停止追蹤；含裝置相關 Python 執行檔 |
| `.venv-wsl/` | 已在工作目錄忽略 | 保留本機，勿上傳 |
| `*:Zone.Identifier` | 57 個檔案 | Windows 下載附加資訊，可停止追蹤 |
| `project/nc4/` | 548 個檔案，約 1,145 MiB | 舊專案資料；先確認來源和備份，再考慮外移 |
| `project/photo/` | 534 個檔案，約 67 MiB | 舊圖表；可保留報告用圖，不能僅憑名稱全刪 |
| `weather2k_exp/` | 約 238 MiB | 保留程式、參數、summary、trial validation；逐點 forecast 可另規劃封存 |
| `STDK_QConvLSTM_reproduction_results/code/` | 目前 QConvLSTM 直接引用 | 必須保留，不能當成無用歷史輸出 |
| `spatial-adapter` | Git submodule | 保留並使用 parent repo 鎖定的 commit |
| `geospatial-neural-adapter-dev/` | 舊套件原始碼 | 當前 QConvLSTM 不需要安裝；仍可能供歷史 notebook 使用，先保留 |
| `weather2k.npy`、checkpoints、logs | 原始資料目錄及多種產物已有忽略規則 | 本機或獨立儲存；跨機器自行準備資料 |

本次新增環境、cache、Zone.Identifier 與 `.env` 忽略規則，保留原有 `.gitignore` 修改。
最初文件整理沒有刪除資料、trial、Python 環境或歷史研究結果。後續已執行停止追蹤操作，
環境檔的刪除紀錄目前在 Git 暫存區；這與刪除本機環境不同。

## 停止追蹤環境檔（另外執行一次）

`.gitignore` 不會自動移除已追蹤檔案。等目前實驗結束，在一台裝置整理：

```bash
git rm -r --cached -- .conda .venv
git ls-files -z -- '*:Zone.Identifier' | xargs -0 -r git rm --cached --
git add .gitignore
git diff --cached --stat
```

`--cached` 在執行這台保留本機檔案，只移除追蹤；確認暫存變更後再 commit / push。
**另一台 pull 這個刪除 commit 時，Git 可能真的刪掉那台已追蹤的環境檔**：
先停訓練、記錄原環境版本，再建立新環境（例如已忽略的 `.venv-weather2k`），確認可用後才 pull。
不要在另一台仍使用 `.conda` / `.venv` 訓練時同步此清理。

這只讓後續版本乾淨，舊 commit 仍包含環境檔，普通完整 clone 不一定因此縮小。
若日後要縮減歷史，需要另行規劃 history rewrite，並協調兩台 checkout；本次不改寫歷史。

`project/nc4/` 沒有在本次刪除或整批忽略；移出前應先確認舊 notebook 的依賴、資料備份與取得方式。
實驗結果也不要只因檔案數多就刪除：trial validation JSON 是跨機器跳過已完成 runs 的依據，
root interface best JSON 是 Stage 2 的 operational input，兩者均有用途。

## 終端出現大量紅色減號，正常嗎？

正常。執行 `git diff --cached --stat` 後，Git 會列出每個待提交檔案的變更。
例如 `.conda/include/unicode/numsys.h | 220 ---` 表示該檔案在 Git 的下一個版本移除 220 行，
不是程式錯誤。這次 `.conda` 原本有 8,547 個追蹤檔案，所以畫面會非常長。
`git status --short` 左側的 `D` 表示已暫存刪除；使用 `git rm --cached` 時，本機檔案仍保留。

畫面底部若出現 `:` 或 `(END)`，表示正在 Git 的分頁器裡。按 **`q`** 回到終端提示符，
不需要按 `Ctrl+c`，也不需要再次執行移除指令。

只看一行總計可改用：

```bash
git --no-pager diff --cached --shortstat
```

若要查看環境檔以外的待提交內容：

```bash
git --no-pager diff --cached --name-status -- . ':!.conda/**' ':!.venv/**' ':!*:Zone.Identifier'
```

預期是 `.gitignore`、README、Markdown 文件及 requirements。
模型程式、資料、已封存結果不應出現在本次刪除清單。
完成檢查後，將本文件的最新說明重新加入暫存區：

```bash
git add docs/REPOSITORY_HYGIENE.md
git diff --cached --check
git commit -m "Remove tracked environments and document Weather2K setup"
git push origin main
```

以上 commit / push 指令適用於目前在 `main` 且暫存區已檢查完的情況。
另一台裝置 pull 前，務必先完成前節的新環境準備。
