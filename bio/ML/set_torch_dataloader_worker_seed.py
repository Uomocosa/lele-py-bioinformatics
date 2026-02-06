import numpy
import random
import torch

import numpy
import random
import torch

def set_torch_dataloader_worker_seed(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)



# import pytest
# @pytest.mark.verbose
def test_():
    SEED = 42
    my_dataset = [1, 2, 3] # Here of course you have to use an actual dataset
    
    train_loader = torch.utils.data.DataLoader(
        my_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        worker_init_fn=set_torch_dataloader_worker_seed,    # <<<
        generator=torch.Generator().manual_seed(SEED)       # <<<
    )
