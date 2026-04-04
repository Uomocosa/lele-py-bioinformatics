from pathlib import Path
import bio

ExperimentConfig = bio.mlp_experiment.Config
def get_log_files(exp_config: ExperimentConfig, save_dir: Path):
    return {
        "traing_epochs": save_dir / f"{exp_config.name}" / "traing_epochs.jsonl",
        "fold_predictions": save_dir / f"{exp_config.name}" / "fold_predictions.jsonl",
        "fold_metrics": save_dir / f"{exp_config.name}" / "fold_metrics.jsonl",
        "aggregate": save_dir / f"{exp_config.name}" / "aggregated_results.jsonl",
    }
