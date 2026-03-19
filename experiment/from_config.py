import yaml
import tyro
from types import SimpleNamespace
from pathlib import Path
from experiment.__call__ import Experiment, DatasetConfig, ModelConfig, FeaturizerConfig


def from_config(config_file: Path) -> Experiment:
    """Loads a plain YAML file and reconstructs a full Experiment dataclass."""
    assert config_file.exists(), f"Config file not found: {config_file}"
    assert config_file.suffix == ".yaml", f"Config file must be a YAML file: {config_file}"
    
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)
        
    ds_dict = config_dict.get("dataset", {})
    mod_dict = config_dict.get("model", {})
    feat_dict = config_dict.get("features", {})
    
    if "csv_file" in ds_dict and ds_dict["csv_file"] is not None:
        ds_dict["csv_file"] = Path(ds_dict["csv_file"])
        
    return Experiment(
        name=config_dict.get("name", "experiment_0"),
        yaml_base_config=config_dict.get("yaml_base_config"),
        dataset=DatasetConfig(**ds_dict),
        model=ModelConfig(**mod_dict),
        features=FeaturizerConfig(**feat_dict),
        seed=config_dict.get("seed")
    )


def test_():
    config_file = Path(__file__).parent / "experiment_0" / "config.yaml"
    exp = from_config(config_file)
    print(exp)
