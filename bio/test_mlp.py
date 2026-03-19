import bio
from bio.pee_smiles_filter import FilterConfig
FeaturizerOptions = bio.Dataset.PDCCMethod.featurize.Options


import pytest
@pytest.mark.above10s
def test_():
    setup_loguru()
    config = FilterConfig()
    config.csv_train_data = TRAIN_CSV_FILE
    config.max_size = 10
    # config.max_size = 100
    config.target_molecule = "CC(=O)OC1=CC=CC=C1C(=O)O" # Aspirin
    config.water_ph = 8.2
    config.featurizer_options = FeaturizerOptions(
        molecule_features_to_calculate = [
            'logp', 
            'logd', 
            'homo_lumo_eV', 
            'net_charge', 
            # 'fingerprint',
        ],
        polymer_features_to_calculate = [
            'logp', 
            'logd', 
            'homo_lumo_eV', 
            'net_charge', 
            # 'fingerprint'
        ],
    )
    df = run_with_config(config)
    
    
    
    
def setup_loguru():
    logger.remove()
    logger.add(
        sys.stderr,
        format = bio.__global__.LOGURU_SIMPLE_FORMAT,
        filter = {
            # "bio.ML.MLPMethod.train_model": "WARNING",
        },
        level = "INFO"
    )
