In this folder we can find the mingpt model *"pee_smiles_generator"* found via the `pixi run -e cuda train_pee_smile_model`

We have also a plot of the training of the model

Finally we have the `generate_*` folder/s that contain all runs of `pixi run -e cuda generate_random_psmiles`. Here we have many subfolders containing:
- The `generated_smiles.csv` (all the generated p-smiles)
- The `valid_smiles.csv` (only the smiles that passed the `config.is_valid_smile` function)

For more information look at `bio.pee_smiles_generate.py`
