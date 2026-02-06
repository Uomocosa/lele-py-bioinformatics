from pathlib import Path
import os
import lele
from loguru import logger

def load_text_dataset(path: Path, n_lines=None):
    logger.warning("DEPRECATED: Use 'bio.Dataset.UnlabeledSmiles.load()' instead")
    datasets = lele.Metaprogramming.try_import("datasets")
    path = lele.P(path)
    assert path.exists(), f"Path does not exist: {path}"
    assert os.path.getsize(path) > 0, f"The data file is empty"
    if n_lines: gen_ = lambda: partial_generator(path, n_lines)
    else      : gen_ = lambda: generator(path)
    dataset = datasets.Dataset.from_generator(gen_)
    return dataset


def generator(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped_line = line.strip()
            if stripped_line: yield {"text": stripped_line}


def partial_generator(path, n_lines):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            stripped_line = line.strip()
            if i >= n_lines: break
            if stripped_line: yield {"text": stripped_line}


import pytest
@pytest.mark.skip(reason="Deprecated")
def test_():
    from bio.Dataset.__global__ import ZINC_BASE_CSV
    N_LINES = 10
    load_text_dataset(ZINC_BASE_CSV, n_lines=N_LINES)
