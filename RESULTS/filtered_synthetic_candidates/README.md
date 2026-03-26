1. We create random valid p-smiles strings using `pixi run -e cuda generate_random_psmiles`
2. We run `pixi run -e cuda pee_smile_filter --target_molecule <target_molecule_name>`
    - For example you could run: `pixi run -e cuda pee_smile_filter --target_molecule aspirin`

Here we save the resulting p-smiles that pass (or "survive") the filter and the filter config used.

***Note!*** You can see what options the filter has with: `pixi run pee_smile_filter_with_config --help` (This is a more complete function, for more information see `bio.pee_smile_filter.py`)
