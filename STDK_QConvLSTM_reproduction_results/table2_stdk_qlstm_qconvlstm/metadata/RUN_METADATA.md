# Run metadata

- Purpose: author-data STDK, STDK+QLSTM and STDK+QConvLSTM comparison
- Data: released 100 x 500 simulation files
- Train/test: times 1--495 / 496--500
- Seed phase: seed 41 first; seeds 42--45 resume after validation
- QLSTM: author notebook settings, with the missing `50k_lstm_data.csv` bridge reconstructed from fitted STDK q50
- QConvLSTM: curated completed paired forecasts
- Command: `.venv-wsl/bin/python -u STDK_QConvLSTM_reproduction_results/code/compare_stdk_qlstm_qconvlstm.py --seeds 41`
