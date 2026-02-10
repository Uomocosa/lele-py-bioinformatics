![3.11](https://github.com/Uomocosa/lele-py-bioinformatics/actions/workflows/test-3-11.yml/badge.svg)

# How to use
1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. `git clone https://github.com/Uomocosa/lele-py-bioinformatics`
3. `uv sync`
4. `uv pip install --upgrade torch --torch-backend=auto` (Optional, if you want to use GPU)
5. `uv tool install --editable . --python 3.11`
<!--6. `cacca_train`-->

# My personal bioinformatics library
- I used this library and apps for my master's thesis.
- It uses my [new-python-import-system](https://github.com/Uomocosa/new-python-import-system). The imports might seem more _magical_ than the usual.

# Data and Outputs
- The **datasets** used can be found in the `DATASET/` directory.
- The **models and checkpoints** created are in the `SMILES_checkpoints/`, `PSMILES_checkpoints/`, `COMBINED_checkpoints/` directories.
    - Each model has a `model_config_used.json` file, read it to understand the configuration used to train the model.
    - There are also "toy" models and checkpoints, in the corresponding directories `SMILES_checkpoints_test/`, ... directories.
- The **generated SMILES** can be found in the `SMILES_checkpoints/generate_.../` directory.
- The **generated P-SMILES** can be found in the `PSMILES_checkpoints/generate_.../` directory.
- And so on ...
    - Again, there is a `generate_config.json` file that lists the configuration used to generate the data.
