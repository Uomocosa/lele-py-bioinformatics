from bio.ML import MLP
import warnings
import torch
import lele, bio
from loguru import logger


def save_model(model: MLP):
    dir = model.config.best_model_save_dir
    if not dir: warnings.warn("Directory not specified."); return
    dir.mkdir(parents=True, exist_ok=True)
    unique_dir = model.config.best_model_save_dir / lele.String.unique()
    logger.debug(f"unique_dir: {unique_dir}")
    unique_dir.mkdir(parents=True, exist_ok=False)
    torch.save(model.state_dict(), unique_dir / "model.pt")
    config_dict = bio.Dataset.serialize_dataclass_instance(model.config)
    lele.Json.save_dict_to_jsonc_file(config_dict, unique_dir / "model_config.jsonc")


import pytest
@pytest.mark.todo
def test_():
    pass
