Results of all leave-one-out cross-validation (**LOOCV**) test from the `bio.mlp_experiment.run_all_experiments.py` function.

Each experiment folder has:
1. An `exp_config.yaml` file containing the config used for the experiment, for more information look at the `bio.mlp_experiment` folder
2. A `parity_plot.png` showing the accuracy of the model for each fold prediction.
3. A `plot_learning_curves.png` showing a mean of the learning epoch required by the LOOCV method
4. A `plot_fold_variance.png` useless since we cannot plot if for LOOCV but only for k-fold cross-validation.
