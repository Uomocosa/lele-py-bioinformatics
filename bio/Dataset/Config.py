from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Callable
import pandas as pd
import math
import lele
from bio.Dataset.__global__ import ZINC_BASE_CSV
print(lele)
print(lele is None)
print(dir(lele))

@dataclass
class Config:
    csv_file: Path = ZINC_BASE_CSV
    train_validation_test_pecentages: Tuple[float, float, float] = (0.6, 0.2, 0.2)
    max_size: Optional[int] = None
    process_raw_smiles: Optional[Callable[pd.Series, pd.Series]] = None
    
    def __post_init__(self):
        err_msg =  f"train_validation_test_pecentages must sum to 1.0\n"
        err_msg += f"You defined: {self.train_validation_test_pecentages}"
        assert math.isclose(sum(self.train_validation_test_pecentages), 1.0), err_msg


import pytest
def test_():
    Config(
        csv_file = ".",
        train_validation_test_pecentages=(0.1, 0.1, 0.8),
    )

def test_failing_():
    with pytest.raises(AssertionError):
        Config(
            csv_file = ".",
            train_validation_test_pecentages=(5.0, 0.1, 0.8),
        )
