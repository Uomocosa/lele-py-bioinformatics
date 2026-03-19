from pathlib import Path
from experiment.__call__ import Experiment

def get_log_files(experiment: Experiment, save_dir: Path):
    return {
        "traing_epochs": save_dir / f"{experiment.name}" / "traing_epochs.jsonl",
        "fold_predictions": save_dir / f"{experiment.name}" / "fold_predictions.jsonl",
        "fold_metrics": save_dir / f"{experiment.name}" / "fold_metrics.jsonl",
        "aggregate": save_dir / f"{experiment.name}" / "aggregated_results.jsonl",
    }
