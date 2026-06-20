import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Plot DLinear + differentiable FRK loss history from JSON.")
    parser.add_argument("json_path", type=Path, help="Path to DLinear metrics JSON.")
    parser.add_argument("--seed", type=int, default=None, help="Seed to plot. Defaults to the first seed run.")
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path.")
    args = parser.parse_args()

    with open(args.json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    seed_runs = payload.get("seed_runs", [])
    if not seed_runs:
        raise ValueError(f"No seed_runs found in {args.json_path}")

    if args.seed is None:
        run = seed_runs[0]
    else:
        matched = [item for item in seed_runs if int(item.get("seed", -1)) == int(args.seed)]
        if not matched:
            raise ValueError(f"Seed {args.seed} not found in {args.json_path}")
        run = matched[0]

    history = run.get("loss_history", [])
    if not history:
        raise ValueError("No loss_history found. Please rerun 2K_DLinear_FRK_hybridloss.py first.")

    df = pd.DataFrame(history)
    seed = int(run.get("seed", -1))
    loss_summary = run.get("loss_summary", {})
    alpha_obs = loss_summary.get("alpha_obs", None)
    lambda_unobs = loss_summary.get("lambda_unobs", None)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(df["epoch"], df["loss_obs_scaled"], label="loss_obs_scaled", linewidth=2)
    axes[0].plot(df["epoch"], df["loss_unobs_scaled"], label="loss_unobs_scaled", linewidth=2)
    if "weighted_loss_obs_scaled" in df.columns:
        axes[0].plot(df["epoch"], df["weighted_loss_obs_scaled"], label="weighted_loss_obs_scaled", linestyle="--", linewidth=2)
    if "weighted_loss_unobs_scaled" in df.columns:
        axes[0].plot(df["epoch"], df["weighted_loss_unobs_scaled"], label="weighted_loss_unobs_scaled", linestyle="--", linewidth=2)
    axes[0].plot(df["epoch"], df["loss_total_scaled"], label="loss_total_scaled", linewidth=2)
    axes[0].set_ylabel("Training loss (standardized)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["epoch"], df["val_4plus5_rmse_raw"], label="val_4plus5_rmse_raw", color="tab:red", linewidth=2)
    best_rows = df[df.get("is_best", False) == True]
    if not best_rows.empty:
        best = best_rows.iloc[-1]
        axes[1].scatter([best["epoch"]], [best["val_4plus5_rmse_raw"]], color="black", zorder=3, label="best")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation RMSE (raw)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    weight_text = ""
    if alpha_obs is not None and lambda_unobs is not None:
        weight_text = f", alpha={alpha_obs}, lambda={lambda_unobs}"
    fig.suptitle(f"DLinear + differentiable FRK loss history, seed {seed}{weight_text}")
    fig.tight_layout()

    out_path = args.out
    if out_path is None:
        out_path = args.json_path.with_name(f"{args.json_path.stem}_seed{seed}_loss.png")
    fig.savefig(out_path, dpi=180)
    print(f"Saved loss plot: {out_path.resolve()}")


if __name__ == "__main__":
    main()
