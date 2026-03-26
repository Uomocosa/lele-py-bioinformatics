After finding and training the PeeSmileCapacityPredictor (mlp architecture), we use it to predict all the random-valid smiles found by the "pee_smiles_generator" model (mingpt architecture).
Then we find the best-performing p-smiles accoring to the `PeeSmileCapacityPredictor` for a specific `target_molecule`. 
Finally we plot the best-performing p-smiles by first fixing the `water_ph` and changing the `concentration`, then by fixing the `concentration` and changing the `water_ph`.
