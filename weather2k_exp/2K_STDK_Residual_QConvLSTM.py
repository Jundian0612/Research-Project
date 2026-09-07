#!/usr/bin/env python3
"""Run the Weather2K Spatial-adapter STDK + residual QConvLSTM ablation.

This keeps the data split, STDK grids, released QConvLSTM architecture and
optimization arguments of 2K_STDK_QConvLSTM.py, but changes the prediction to
direct STDK q50 forecast + learned QConvLSTM residual.  Output names use the
shared_residual_qconvlstm prefix and cannot overwrite the paper-aligned run.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path


MODEL = Path(__file__).resolve().with_name("2K_STDK_QConvLSTM.py")


def main():
    if "--prediction-mode" in sys.argv:
        raise SystemExit(
            "2K_STDK_Residual_QConvLSTM.py fixes --prediction-mode=residual; "
            "do not pass --prediction-mode explicitly"
        )
    sys.argv.extend(("--prediction-mode", "residual"))
    runpy.run_path(str(MODEL), run_name="__main__")


if __name__ == "__main__":
    main()
