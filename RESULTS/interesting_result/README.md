If we run:
- `pixi run -e cuda find_polymer_for_target_molecule "aspirin"`
- `pixi run -e cuda find_polymer_for_target_molecule "lisinpropil"`
- `pixi run -e cuda find_polymer_for_target_molecule "metformin"`
The rusults all inlcude the same polymer that has a REALLY high predicted capacity:
- `*Nc1ccc(NC(=O)c2ccc(C(=O)NNC(=O)c3ccc(*)cc3)cc2)cc1`
(The prediction is probably too high with respect to a real-case scenario, this is due to the simplicity of the MLP used and the scarcity of data)
