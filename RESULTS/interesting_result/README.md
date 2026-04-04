If we run:
- `pixi run -e cuda find_polymer_for_target_molecule "aspirin" --model_config.batch_size=256 --model_config.polymers_to_generate_per_loop=1024`
- `pixi run -e cuda find_polymer_for_target_molecule "lisinpropil" --model_config.batch_size=256 --model_config.polymers_to_generate_per_loop=1024`
- `pixi run -e cuda find_polymer_for_target_molecule "metformin" --model_config.batch_size=256 --model_config.polymers_to_generate_per_loop=1024`
The rusults all inlcude the same polymer that has a REALLY high predicted capacity:
- `*Nc1ccc(NC(=O)c2ccc(C(=O)NNC(=O)c3ccc(*)cc3)cc2)cc1`
(The prediction is probably too high with respect to the concentration used (12.5) in a real-world scenario, this is due to the simplicity of the MLP used and the scarcity of data)
