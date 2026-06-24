# UniSI server scripts

Self-contained scripts to run the thesis CAPACITY experiments on the UniSI **CUDA** server.
The box is periodically reset, so always start from `00_bootstrap.sh` after a fresh `git clone`.
Nothing here is auto-committed — copy results off the server with `05_save_results.sh` + `scp`.

Run in order (each is idempotent):

| Script | What it does |
|---|---|
| `00_bootstrap.sh` | Install pixi, `pixi install -e cuda`, verify `import bio, torch` + CUDA. |
| `01_smoke_test.sh` | Fast unit tests + tiny integrate + grouped-CV mini run — confirm the pipeline works. |
| `02_build_datasets.sh` | (Optional) build/inspect the combined old+new featurized dataset. |
| `03_run_experiments.sh` | Run the sweep (datasets × architectures, grouped CV) → `RESULTS/thesis_experiments/`. |
| `04_rank.sh` | Rebuild the Q2/MAE/RMSE leaderboard. |
| `05_save_results.sh` | Tar `RESULTS/thesis_experiments/` for scp. |

`03_run_experiments.sh` passes args through, e.g.:
```bash
./unisi_scripts/03_run_experiments.sh --datasets original opus deepseek pool old_plus_new
./unisi_scripts/03_run_experiments.sh --architectures hd_16_8_4_4_4 hd_64_32_16_8_4
```

Notes:
- The old `RESULTS/mlp_experiments/` (LOOCV results) is never touched.
- Tiny per-model splits (`kimi`, `gemma`) are auto-handled: grouped folds are capped to the number
  of polymer–drug groups, and datasets with <2 groups are skipped with a warning.
- First featurization of each dataset is heavy (minutes); it is cached afterwards.
