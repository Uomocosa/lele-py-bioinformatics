import torch
import numpy
import random
import os


def set_seed(seed: int):
    """
    set_seed(42) # do this at the beginning of the program to make it determinisitc
    IMPORTANT: if you also use torch.DataLoader with worker > 0 also see lele.Lerning.set_torch_dataloader_worker_seed
    """
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        print("torch.use_deterministic_algorithms not available.")


import pytest
def test_():
    SEED = 42
    set_seed(SEED) 
    assert random.random() == 0.6394267984578837
    assert numpy.random.rand(1, 1)[0] == pytest.approx(numpy.array([0.37454012]))
    assert torch.rand(1, 1)[0] == pytest.approx(torch.tensor([0.88226926]))
