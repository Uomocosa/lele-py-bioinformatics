import torch

def get_torch_device() -> str:
    if torch.cuda.is_available(): return 'cuda'
    elif torch.backends.mps.is_available(): 
        # Apple Silicon GPU (Metal Performance Shaders)
        return 'mps'
    else: return 'cpu'

def test_():
    assert get_torch_device() in ['cpu', 'cuda', 'mps']
