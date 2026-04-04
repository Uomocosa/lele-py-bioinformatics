"""TODO:
    - At the moment I cannot load an 'unserializable data' from a json file
        - 'unserializable data': like functions.
    - Rename 'ModelConfig' to just 'Config', change all mentions to 'MinGPT.Config'
"""
import re
import pydantic
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict
import platform
import psutil
from loguru import logger
import lele, bio
from bio.__global__ import BIOINFORMATICS_DIR
from bio.MinGPT.__global__ import HELPER_DIR
DatasetConfig = bio.Dataset.Config.Config

@dataclass
class Options:
    save_model_config: bool = True
    log_loss_data: bool = True
    save_best_model: bool = True
    save_checkpoint_every_n_iters: Optional[int] = 1000
    checkpoint_dir: Path = HELPER_DIR / "checkpoints"


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
    dataset: DatasetConfig = field(default_factory= lambda: DatasetConfig(
        train_validation_test_pecentages = (0.8, 0.2, 0.0)
    ))
    
    def __post_init__(self):
        bio.ML.set_seed(self.seed)
        
    def save_if_requested(self):
        if not self.options.save_model_config: return
        self.save(self.options.checkpoint_dir/"model_config_used.jsonc")
        
    def save(self, path:Path):
        path = lele.P(path)
        config_dict = asdict(self)
        if config_dict.get('dataset') and config_dict['dataset'].get('csv_file'):
            abs_csv = Path(config_dict['dataset']['csv_file'])
            try:
                # Converts "C:\...\DATASETS\PI1M\PI1M.csv" -> "DATASETS/PI1M/PI1M.csv"
                rel_csv = abs_csv.relative_to(BIOINFORMATICS_DIR)
                config_dict['dataset']['csv_file'] = rel_csv.as_posix() 
            except ValueError:
                pass # If the path isn't inside the repo, leave it as is
        if config_dict.get('options') and config_dict['options'].get('checkpoint_dir'):
            abs_ckpt = Path(config_dict['options']['checkpoint_dir'])
            try:
                rel_ckpt = abs_ckpt.relative_to(BIOINFORMATICS_DIR)
                config_dict['options']['checkpoint_dir'] = rel_ckpt.as_posix()
            except ValueError:
                pass
        config_dict['device_info'] = lele.Metaprogramming.get_device_specs()
        lele.Json.save_dict_to_jsonc_file(
            config_dict, path, header="Configuration used:"
        )
        logger.debug(f"Config saved to: {path}")
    
    
    
def load(path: str, add_unique_id=True) -> ModelConfig:
    config_dict = lele.Json.get_dict_from_jsonc(lele.P(path))
    config_dict = remove_unserializable_data(config_dict)
    if 'dataset' in config_dict and 'csv_file' in config_dict['dataset']:
        saved_path = Path(config_dict['dataset']['csv_file'])
        if not saved_path.is_absolute():
            config_dict['dataset']['csv_file'] = BIOINFORMATICS_DIR / saved_path
    if 'options' in config_dict and 'checkpoint_dir' in config_dict['options']:
        saved_path = Path(config_dict['options']['checkpoint_dir'])
        if not saved_path.is_absolute():
            config_dict['options']['checkpoint_dir'] = BIOINFORMATICS_DIR / saved_path
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

def remove_unserializable_data(dictionary: dict) -> dict:
    dictionary = {k: v for k, v in dictionary.items() if not "not serializable" in str(v)}
    return dictionary

def test_():
    from bio.MinGPT.__global__ import MIN_GPT_CONFIG_FILE
    config = load(MIN_GPT_CONFIG_FILE)
    print(config)
