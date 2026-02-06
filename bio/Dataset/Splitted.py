import torch
from dataclasses import dataclass

@dataclass
class Splitted:
    train: torch.utils.data.Dataset
    validation: torch.utils.data.Dataset
    test: torch.utils.data.Dataset


def test_():
    pass
