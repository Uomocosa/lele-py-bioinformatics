import re
import pydantic
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict
import platform
import psutil
from loguru import logger
import lele, bio
from bio import Dataset
from bio.MinGPT.__global__ import HELPER_DIR

@dataclass
class Options:
    save_model_config: bool = True
    log_loss_data: bool = True
    save_best_model: bool = True
    save_checkpoint_every_n_iters: Optional[int] = 1000
    checkpoint_dir: Path = HELPER_DIR / "checkpoints"


"""
TODO: Rename it to Config, change all mentions to MinGPT.Config
"""
@dataclass
class ModelConfig:
    seed: int = 42
    model_type: str = 'gpt-nano'
    block_size: int = 128  # Context length
    epochs: int = 10
    learning_rate: float = 6e-4
    batch_size: int = 64
    num_workers: int = 0
    starting_state_dict: Optional[Path] = None 
    early_stop_patience: Optional[int] = 1000 
    options: Options = field(default_factory=Options)
    dataset: lele.type(Dataset.Config) = field(default_factory=Dataset.Config)
    
    def __post_init__(self):
        bio.ML.set_seed(self.seed)
        
    def save_if_requested(self):
        if not self.options.save_model_config: return
        self.save(self.options.checkpoint_dir/"model_config_used.jsonc")
        
    def save(self, path:Path):
        path = lele.P(path)
        config_dict = asdict(self)
        config_dict['device_info'] = lele.Metaprogramming.get_device_specs()
        lele.Json.save_dict_to_jsonc_file(
            config_dict, path, header="Configuration used:"
        )
        logger.debug(f"Config saved to: {path}")
    
    
    
def load(path: str, add_unique_id=True) -> ModelConfig:
    config_dict = lele.Json.get_dict_from_jsonc(lele.P(path))
    logger.debug(f"config_dict: {config_dict}")
    try:
        adapter = pydantic.TypeAdapter(ModelConfig)
        config = adapter.validate_python(config_dict)
    except pydantic.ValidationError as e:
        for error in e.errors():
            if error['type'] == 'assertion_error':
                logger.error(f"Assertion Failed in logic:\nLocation: {error['loc']}\nInput: {error['input']}")
        raise e
    config.options.checkpoint_dir = lele.P(config.options.checkpoint_dir)
    if add_unique_id:
        config.options.checkpoint_dir = config.options.checkpoint_dir/lele.String.unique()
        config.options.checkpoint_dir.mkdir(exist_ok=False, parents=True)
    return config



def test_():
    from bio.MinGPT.__global__ import MIN_GPT_CONFIG_FILE
    config = load(MIN_GPT_CONFIG_FILE)
    print(config)
