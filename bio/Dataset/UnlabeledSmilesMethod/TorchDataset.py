import bio
import torch
from deepchem.feat.smiles_tokenizer import SmilesTokenizer

class TorchDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        smiles_data: bio.Dataset.UnlabeledSmiles, 
        tokenizer: SmilesTokenizer, 
        block_size: int
    ):
        self.smiles_data = smiles_data
        self.tokenizer = tokenizer
        self.block_size = block_size
        # Ensure tokenizer has pad/eos attributes or hardcode them if you know them
        # self.pad_id = getattr(tokenizer, 'pad_token_id', 0) 
        # self.eos_id = getattr(tokenizer, 'eos_token_id', <YOUR_EOS_ID>)
        
        # Assuming for now based on common simple tokenizers:
        self.pad_id = 0 
        self.eos_id = tokenizer.vocab_size - 1 # Or however your tokenizer defines it

    def __len__(self): 
        return len(self.smiles_data)

    def __getitem__(self, idx):
        smiles = self.smiles_data[idx]
        tokens = self.tokenizer.encode(smiles) # List[int]
        max_len = self.block_size 
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        if len(tokens) < max_len:
            tokens = tokens + [self.pad_id] * (max_len - len(tokens))
            
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        # Masking: We don't want to calculate loss on Pad tokens
        # Replace pad_id in Y with -1 -> mingpt will exclude it
        y[y == self.pad_id] = -1
        return x, y


def test_usage():
    from bio.Dataset.__global__ import ZINC_BASE_CSV
    BLOCK_SIZE = 8
    smiles = bio.Dataset.UnlabeledSmiles.load(ZINC_BASE_CSV, max_dataset_size=100)
    dataset = smiles.to_torch_dataset(BLOCK_SIZE)
    print(f"dataset: {dataset}")
    for i in range(0,100): print(f"dataset[{i}]: {dataset[i]}")


def test_internal():
    from bio.Dataset.__global__ import ZINC_BASE_CSV, SMILES_VOCABULARY
    BLOCK_SIZE = 8
    smiles = bio.Dataset.UnlabeledSmiles.load(ZINC_BASE_CSV, max_dataset_size=100)
    tokenizer = SmilesTokenizer(SMILES_VOCABULARY)
    dataset = TorchDataset(smiles, tokenizer, BLOCK_SIZE)
    print(f"dataset: {dataset}")
    for i in range(0,100): print(f"dataset[{i}]: {dataset[i]}")
