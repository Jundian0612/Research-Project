# Weather2K 新裝置安裝與雙裝置工作流程

適用：Ubuntu / WSL2、目前 `weather2k_exp` 的 STDK + QConvLSTM；其他 baseline 另列選配依賴。
每台機器各自建立 Python 環境，Git 同步程式、設定及結果。不要複製 `.venv` / `.conda`。
以下指令都在 Bash 執行。2026-09-08 文件依本機程式整理；尚未在全新 GPU 主機完整重建驗證。

## 1. 系統工具與專案

Ubuntu / WSL Ubuntu：

```bash
sudo apt update
sudo apt install git git-lfs tmux python3 python3-venv python3-pip
git clone https://github.com/Jundian0612/Research-Project.git
cd Research-Project
git submodule update --init --recursive spatial-adapter
```

已有專案時不用再次 clone。先確認沒有正在執行的實驗、處理自己的未提交修改，再執行：

```bash
git status --short
git pull --ff-only
git submodule update --init --recursive spatial-adapter
```

只初始化 `spatial-adapter`，資料集在下一節獨立處理。不要用 `git submodule update --remote`，
避免兩台裝置取得不同的模型程式版本。目前 parent repo 記錄的 spatial-adapter commit 是
`2aea188f3b8d92f948663b6705f0a22850d6e4ee`；後續以自己 checkout 的 parent commit 為準。

## 2. 每台裝置建立環境

建議兩台使用同一個 Python minor 版本（例如 3.12）。先以 `python3 --version` 確認。
若已經有 `.venv-weather2k`，直接 activate，不要在正在訓練的環境重裝套件。

```bash
python3 -m venv .venv-weather2k
source .venv-weather2k/bin/activate
python -m pip install --upgrade pip
```

先安裝 PyTorch。NVIDIA GPU 機器先執行 `nvidia-smi`，確認驅動可見，
再到 [PyTorch 官方安裝選擇器](https://pytorch.org/get-started/locally/) 選擇 Linux / Pip / Python
及適合驅動的 CUDA wheel，在這個已啟用的環境執行其安裝指令（使用 `python -m pip`）。
若要延續同一批實驗，優先使用原訓練裝置記錄的 torch 版本和 wheel 來源，
可由 [歷史版本頁](https://pytorch.org/get-started/previous-versions/) 查對應指令。
不要僅因換機就升級模型依賴。Windows + WSL2 必須先在 Windows 配置可供 WSL 使用的 GPU 驅動。

只有 CPU、僅打算做小規模檢查時，可用：

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

再裝核心依賴：

```bash
python -m pip install -r weather2k_exp/requirements-core.txt
python -m pip check
python -c "import sys, torch; print(sys.executable); print(torch.__version__, torch.version.cuda); print('CUDA:', torch.cuda.is_available())"
```

GPU 正式訓練應看到 `CUDA: True`。目前程式會在 CUDA 不可用時自動改跑 CPU，
所以程式啟動成功不代表正在用 GPU。此文件編寫環境為 Python 3.12、torch 2.14.0+cu130，
但沙盒回報 CUDA 不可用；這不是已驗證的 GPU 重現環境，也不能據此判定主機本身沒有 GPU。

目前 QConvLSTM 用 PyTorch，透過檔案路徑載入
`spatial-adapter/examples/baselines/stdk/st_interp.py` 與
`STDK_QConvLSTM_reproduction_results/code/STDK_QConvLSTM_reproduction.py`。
這條實驗路徑不需要 TensorFlow，也不需要編譯安裝整個 spatial-adapter 套件。

要跑 SVGP、DLinear+FRK 或 notebook，再安裝：

```bash
python -m pip install -r weather2k_exp/requirements-baselines.txt
python -m pip check
python -c "import gpytorch, optuna; from darts.models import DLinearModel; print('baseline imports OK')"
python -m ipykernel install --user --name weather2k --display-name 'Python (weather2k)'
```

這兩份 requirements 是依賴清單，並非所有舊實驗的精確 lockfile；baseline 額外依賴未在本次環境安裝驗證。
VS Code 的 Python interpreter / notebook kernel 選這個環境；終端仍需自行 activate。

## 3. 準備同一份 Weather2K 資料

目前 tuning runner 使用模型預設資料路徑：

```text
Research-Project/Josh's Weather2K/Weather2K/weather2k.npy
```

由 [Weather2K 資料集頁](https://huggingface.co/datasets/BUPT-PRIS-727/Weather2K)
取得授權與資料；如果頁面要求申請存取，先完成該流程。每台裝置自行準備，原始資料不 push 到本專案。
repo 已把資料來源設為 submodule；有存取權且 Git 憑證已配置時可執行：

```bash
git lfs install
git submodule update --init "Josh's Weather2K/Weather2K"
git -C "Josh's Weather2K/Weather2K" lfs pull
```

也可以將已合法取得的同版本 `weather2k.npy` 放到上述位置。檔案必須是真正的 NumPy 資料，
不是 Git LFS pointer。上游若變更資料格式或變數順序，不能直接視為和舊實驗相同。

```bash
python -c "import numpy as np; a=np.load(\"Josh's Weather2K/Weather2K/weather2k.npy\", mmap_mode='r'); print(a.shape, a.dtype)"
sha256sum "Josh's Weather2K/Weather2K/weather2k.npy"
```

兩台的 shape、dtype、SHA256 要一致；hash 請保存到實驗紀錄。

## 4. 執行前確認

```bash
python weather2k_exp/2K_STDK_QConvLSTM.py --help
python weather2k_exp/tune_stdk_qconvlstm_hyperparams.py --help
```

新環境先用隔離的目錄做小型流程檢查（會真的訓練，但採縮小設定）：

```bash
python -u weather2k_exp/tune_stdk_qconvlstm_hyperparams.py \
  --stage all --smoke-test --seeds 41 \
  --max-interface-trials 1 --max-capacity-trials 1 \
  --output-dir /tmp/weather2k-smoke
```

Smoke 結果不是正式選參結果；此模式不更新正式參數檔。

## 5. tmux 執行 Stage 2

```bash
tmux new -s weather2k
```

在 tmux 裡切到 clone 的專案根目錄（例如 `cd ~/Research-Project`），再執行：

```bash
source .venv-weather2k/bin/activate
mkdir -p logs
set -o pipefail
PYTHONUNBUFFERED=1 python -u weather2k_exp/tune_stdk_qconvlstm_hyperparams.py \
  --stage capacity --seeds 41 42 \
  2>&1 | tee -a logs/stage2_capacity.log
```

`PYTHONUNBUFFERED=1` 也讓子訓練程序即時輸出，`tee -a` 保留先前紀錄，
`pipefail` 讓 pipeline 能反映 Python 失敗。結束後立即用 `echo $?` 查看退出碼；
並確認 capacity summary 與 best params 已產生，不能只看退出碼或最後一行參數。

- 脫離畫面：`Ctrl+b`，放開，再按 `d`。
- 回來：`tmux attach -t weather2k`。
- 查看 sessions：`tmux ls`。
- tmux 可讓程序在終端斷線後繼續；關機、WSL 關閉、休眠仍會中斷或暫停運算。

Stage 2 預設讀取 `weather2k_exp/qconvlstm_interface_best_params.json`，
完成 4 組 capacity × 2 seeds，依 Val150 選參，最後更新正式
`2K_best_stdk_qconvlstm_params_500to100.json`。
改用 `--output-dir` 時，必須先將 interface best JSON 複製到該目錄，否則找不到 Stage 1 結果。
即使用不同 output-dir，非 smoke Stage 2 完成時仍會更新根目錄的正式參數檔。

## 6. 兩台裝置同步

最簡單的方式：同一輪 Stage 2 由一台跑完，另一台等 push 後 pull。
兩台可以做不同實驗，但不要同時修改同一組 trial、同名結果或正式參數檔。
若必須並行，分配不同實驗、分支及輸出目錄，再由一台統一整理；不要讓兩台各自選出「正式最佳參數」。
不要在訓練過程中 pull、切分支、更新 submodule 或更新套件，runner 後續 trials 會重新載入程式。

跑完後，先保存該次環境和程式版本。下面目錄名稱是範例，每批實驗使用不同名稱：

```bash
mkdir -p weather2k_exp/run_metadata/stage2_device_a
python -m pip freeze > weather2k_exp/run_metadata/stage2_device_a/pip-freeze.txt
python --version > weather2k_exp/run_metadata/stage2_device_a/python.txt
git rev-parse HEAD > weather2k_exp/run_metadata/stage2_device_a/code-commit.txt
git submodule status > weather2k_exp/run_metadata/stage2_device_a/submodules.txt
sha256sum "Josh's Weather2K/Weather2K/weather2k.npy" > weather2k_exp/run_metadata/stage2_device_a/data-sha256.txt
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')" > weather2k_exp/run_metadata/stage2_device_a/torch-device.txt
```

若程式有未提交修改，另保存 `git diff` 或先提交程式版本；單一 commit hash 不代表未提交程式。
freeze 是該機器的環境快照，可能包含本機路徑或特定 CUDA wheel，新機器安裝前需檢查；
請額外記錄實際 PyTorch 安裝指令與 wheel index，不能靠複製環境目錄重現。

裝置 A：

```bash
git status --short
git diff --stat
git add weather2k_exp
git diff --cached --stat
git diff --cached --name-only
git commit -m "Record completed Weather2K capacity tuning"
git push origin main
```

提交前確認暫存區只有要同步的完整結果、參數、程式及 metadata。
若目前在其他分支，改 push 該分支；若 push 因遠端更新被拒絕，先 fetch 並檢查差異、整合衝突，勿 force push。
安裝文件與 `.gitignore` 若也要提交，另以明確路徑加入暫存區。

裝置 B：在沒有實驗執行、工作目錄修改已處理後：

```bash
git pull --ff-only
git submodule update --init --recursive spatial-adapter
source .venv-weather2k/bin/activate
```

只同步結果不需重裝環境；依賴有變更才更新。跨裝置接續未完成 tuning 時，
必須帶上 operational interface JSON 與已完成的 `qconvlstm_capacity_trial*_seed*_validation.json`。
程式可跳過設定相符的已完成 seed runs，但不保存 model checkpoint，
中斷在一個 seed 中途時，該 seed 必須重新訓練。Stage 2 全部完成、參數鎖定後才執行正式五-seed test。
