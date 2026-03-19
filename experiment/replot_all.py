from pathlib import Path
import experiment
from loguru import logger

def replot_all(dir: Path):
    for experiment_dir in dir.iterdir():
        if not experiment_dir.is_dir(): continue
        config_file = experiment_dir / "config.yaml"
        if not config_file.exists(): continue
        logger.debug(f"Replotting {experiment_dir}")
        exp = experiment.from_config(config_file)
        log_files = experiment.get_log_files(exp, dir)
        experiment.plot_learning_curves(exp, dir, log_files["traing_epochs"])
        experiment.plot_fold_variance(exp, dir, log_files["fold_metrics"])
        experiment.plot_parity(exp, dir, log_files["fold_predictions"]) 
        

import pytest
@pytest.mark.above10s
def test_run():
    import sys
    import bio
    logger.remove()
    logger.add(
        sys.stderr,
        format = bio.__global__.LOGURU_SIMPLE_FORMAT,
        filter = {
            "bio.ML.MLPMethod.train_model": "WARNING",
        },
        level = "INFO"
    )
    dir = Path(__file__).parent
    replot_all(dir)
