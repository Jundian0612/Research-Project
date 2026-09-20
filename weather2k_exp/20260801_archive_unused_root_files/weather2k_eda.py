import os
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Weather2K EDA
# ============================================================
# 這個腳本使用和模型相同的資料載入方式：
# 原始資料 shape = (空間點, 13, 時間點)
# transpose 後變成 (空間點, 時間點, 13)
# 前 3 欄是位置資訊，後 10 欄是氣象變數。

SCRIPT_DIR = Path(__file__).resolve().parent
CWD = Path.cwd()

CANDIDATE_PATHS = [
    os.environ.get("WEATHER2K_NPY", ""),
    SCRIPT_DIR / "../Josh's Weather2K/Weather2K/weather2k.npy",
    SCRIPT_DIR / "../../Josh's Weather2K/Weather2K/weather2k.npy",
    CWD / "Josh's Weather2K/Weather2K/weather2k.npy",
    CWD / "../Josh's Weather2K/Weather2K/weather2k.npy",
]

WEATHER_VAR_NAMES = [
    "air_pressure",
    "air_temperature",
    "relative_humidity",
    "wind_speed",
    "wind_direction",
    "precipitation",
    "solar_radiation",
    "dew_point_temperature",
    "cloud_cover",
    "visibility",
]

OUT_DIR = SCRIPT_DIR / "weather2k_eda"
MAX_HIST_SAMPLES = 500_000
MAX_CORR_SAMPLES = 300_000
RANDOM_SEED = 41


def _find_dataset_path() -> Path:
    for candidate in CANDIDATE_PATHS:
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        if path.is_file():
            return path
    raise FileNotFoundError(f"找不到 weather2k.npy，已搜尋：{CANDIDATE_PATHS}")


def _make_time_index(ntime: int) -> pd.DatetimeIndex:
    date_start = datetime(2017, 1, 1)
    time_list = []
    current = date_start
    for _ in range(ntime):
        time_list.append(current)
        current += timedelta(hours=3)
    return pd.to_datetime(np.array(time_list))


def _safe_values(x: np.ndarray) -> np.ndarray:
    values = x.astype(np.float64).ravel()
    return values[np.isfinite(values)]


def _sample_values(values: np.ndarray, max_samples: int, rng: np.random.Generator) -> np.ndarray:
    if values.size <= max_samples:
        return values
    idx = rng.choice(values.size, size=max_samples, replace=False)
    return values[idx]


def _df_to_markdown(df: pd.DataFrame, float_digits: int = 6) -> str:
    """輸出簡單 Markdown table，避免依賴 pandas 的 tabulate optional package。"""
    table = df.copy()
    for col in table.columns:
        if pd.api.types.is_float_dtype(table[col]):
            table[col] = table[col].map(lambda x: f"{x:.{float_digits}f}")
    headers = [str(col) for col in table.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in table.columns) + " |")
    return "\n".join(lines)


def build_summary(data: np.ndarray) -> pd.DataFrame:
    rows = []
    for var_idx, var_name in enumerate(WEATHER_VAR_NAMES):
        values = data[:, :, var_idx].astype(np.float64).ravel()
        finite = np.isfinite(values)
        clean = values[finite]

        rows.append(
            {
                "variable": var_name,
                "missing_rate": float(1.0 - finite.mean()),
                "min": float(np.min(clean)),
                "p01": float(np.percentile(clean, 1)),
                "p05": float(np.percentile(clean, 5)),
                "mean": float(np.mean(clean)),
                "std": float(np.std(clean)),
                "p50": float(np.percentile(clean, 50)),
                "p95": float(np.percentile(clean, 95)),
                "p99": float(np.percentile(clean, 99)),
                "max": float(np.max(clean)),
            }
        )
    return pd.DataFrame(rows)


def plot_histograms(data: np.ndarray, rng: np.random.Generator) -> None:
    fig, axes = plt.subplots(5, 2, figsize=(14, 18))
    axes = axes.ravel()

    for var_idx, var_name in enumerate(WEATHER_VAR_NAMES):
        values = _safe_values(data[:, :, var_idx])
        sample = _sample_values(values, MAX_HIST_SAMPLES, rng)
        axes[var_idx].hist(sample, bins=80, color="#4C78A8", alpha=0.85)
        axes[var_idx].set_title(var_name)
        axes[var_idx].set_ylabel("count")
        axes[var_idx].grid(True, alpha=0.25)

    fig.suptitle("Weather2K variable distributions")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "variable_histograms.png", dpi=180)
    plt.close(fig)


def plot_time_means(data: np.ndarray, time_index: pd.DatetimeIndex) -> None:
    fig, axes = plt.subplots(5, 2, figsize=(16, 18), sharex=True)
    axes = axes.ravel()

    for var_idx, var_name in enumerate(WEATHER_VAR_NAMES):
        time_mean = np.nanmean(data[:, :, var_idx], axis=0)
        axes[var_idx].plot(time_index, time_mean, linewidth=0.8, color="#F58518")
        axes[var_idx].set_title(var_name)
        axes[var_idx].grid(True, alpha=0.25)

    fig.suptitle("Spatial mean time series")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "spatial_mean_timeseries.png", dpi=180)
    plt.close(fig)


def plot_correlation(data: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    nloc, ntime, nvar = data.shape
    total_pairs = nloc * ntime
    sample_size = min(MAX_CORR_SAMPLES, total_pairs)
    flat_idx = rng.choice(total_pairs, size=sample_size, replace=False)
    loc_idx = flat_idx // ntime
    time_idx = flat_idx % ntime

    sampled = data[loc_idx, time_idx, :].astype(np.float64)
    corr = pd.DataFrame(sampled, columns=WEATHER_VAR_NAMES).corr()
    corr.to_csv(OUT_DIR / "variable_correlation.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(np.arange(len(WEATHER_VAR_NAMES)))
    ax.set_yticks(np.arange(len(WEATHER_VAR_NAMES)))
    ax.set_xticklabels(WEATHER_VAR_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(WEATHER_VAR_NAMES)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Variable correlation")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "variable_correlation.png", dpi=180)
    plt.close(fig)

    return corr


def plot_spatial_means(data: np.ndarray, loc: np.ndarray) -> None:
    lat = loc[:, 0]
    lon = loc[:, 1]

    fig, axes = plt.subplots(5, 2, figsize=(14, 18))
    axes = axes.ravel()

    for var_idx, var_name in enumerate(WEATHER_VAR_NAMES):
        spatial_mean = np.nanmean(data[:, :, var_idx], axis=1)
        sc = axes[var_idx].scatter(lon, lat, c=spatial_mean, s=8, cmap="viridis")
        axes[var_idx].set_title(var_name)
        axes[var_idx].set_xlabel("lon")
        axes[var_idx].set_ylabel("lat")
        fig.colorbar(sc, ax=axes[var_idx], fraction=0.046, pad=0.04)

    fig.suptitle("Spatial mean by station")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "spatial_mean_maps.png", dpi=180)
    plt.close(fig)


def write_readme(summary: pd.DataFrame, corr: pd.DataFrame, raw_shape: tuple[int, ...], data_shape: tuple[int, ...]) -> None:
    lines = [
        "# Weather2K EDA",
        "",
        "這份 EDA 使用和模型相同的資料載入方式。",
        "",
        f"- raw shape: `{raw_shape}`",
        f"- model data shape: `{data_shape}` = `(空間點, 時間點, 變數)`",
        f"- variables: `{', '.join(WEATHER_VAR_NAMES)}`",
        "",
        "## Summary",
        "",
        _df_to_markdown(summary),
        "",
        "## 輸出圖檔",
        "",
        "- `variable_histograms.png`：每個變數的分布。",
        "- `spatial_mean_timeseries.png`：每個變數跨空間平均後的時間序列。",
        "- `variable_correlation.png`：10 個變數的相關矩陣。",
        "- `spatial_mean_maps.png`：每個測站時間平均後的空間分布。",
        "- `variable_correlation.csv`：相關矩陣數值。",
        "",
        "## 初步提醒",
        "",
        "EDA 只先檢查數值範圍、分布、時間趨勢和相關性。",
        "如果某些變數的範圍和名稱直覺不一致，建議再確認 Weather2K 原始欄位順序或單位定義。",
        "",
        "## Correlation",
        "",
        _df_to_markdown(corr.reset_index().rename(columns={"index": "variable"}), float_digits=4),
        "",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    dataset_path = _find_dataset_path()
    raw = np.load(dataset_path)
    loaded = np.transpose(raw, (0, 2, 1))
    loc = loaded[:, 0, :3]
    data = loaded[:, :, 3:]

    if data.shape[2] != len(WEATHER_VAR_NAMES):
        raise ValueError(
            f"資料變數數量 = {data.shape[2]}，但 WEATHER_VAR_NAMES = {len(WEATHER_VAR_NAMES)}，請確認欄位順序"
        )

    time_index = _make_time_index(data.shape[1])

    summary = build_summary(data)
    summary.to_csv(OUT_DIR / "variable_summary.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "variable_summary.md").write_text(_df_to_markdown(summary), encoding="utf-8")

    plot_histograms(data, rng)
    plot_time_means(data, time_index)
    corr = plot_correlation(data, rng)
    plot_spatial_means(data, loc)
    write_readme(summary, corr, raw.shape, data.shape)

    print(f"Dataset: {dataset_path}")
    print(f"raw shape: {raw.shape}")
    print(f"model data shape: {data.shape}")
    print(summary.to_string(index=False))
    print(f"Saved EDA outputs: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
