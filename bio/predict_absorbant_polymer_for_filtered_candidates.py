from bio.Bioinformatics import PeeSmileCapacityPredictor
import pandas as pd
from loguru import logger
CANDIDATES_DIR = RESULTS_DIR / "filtered_synthetic_candidates"
# MODEL_DIR = RESULTS_DIR / "PeeSmileCapacityPredictor"


def predict_absorbant_polymer_for_filtered_candidates(
    target_molecule_name: str, 
    water_ph: float = 6.5, 
    concentration: float = 12.5
) -> pd.DataFrame:
    """Predicts capacity for filtered candidates and ranks them from best to worst."""
    candidates_csv = CANDIDATES_DIR / f"target_{target_molecule_name}.csv"
    
    if not candidates_csv.exists():
        logger.error(f"Could not find filtered candidates at {candidates_csv}. Did you run the filters first?")
        return pd.DataFrame()
        
    df = pd.read_csv(candidates_csv)
    if df.empty:
        logger.warning(f"The filtered candidates dataframe for {target_molecule_name} is empty.")
        return df

    # 2. Add the required environmental variables for the predictor
    df['WATER_PH'] = water_ph
    df['CONCENTRATION'] = concentration

    # 3. Load the Prediction Model
    pscp = PeeSmileCapacityPredictor()
    trained_model = pscp.load_trained_model()
    
    # Fallback just in case the model hasn't been saved yet
    if trained_model is None:
        logger.info("No saved model found, retrieving and saving a new trained model...")
        trained_model = pscp.get_trained_model()
        pscp.save_trained_model(trained_model)
        
    assert trained_model is not None, "Failed to load trained model"
    if hasattr(trained_model, "eval"): trained_model.eval()

    # 4. Predict
    # Subset only the columns the predictor needs to avoid confusing the model
    predict_df = df[['POLYMER_USED', 'DRUG', 'WATER_PH', 'CONCENTRATION']].copy()
    
    logger.info(f"Running capacity predictions on {len(df)} candidates for {target_molecule_name}...")
    predictions = trained_model.predict(predict_df)
    
    # 5. Rank and Sort (Assuming higher capacity is better)
    df['PREDICTED_CAPACITY'] = predictions
    df = df.sort_values(by='PREDICTED_CAPACITY', ascending=False).reset_index(drop=True)
    
    # 6. Save the ranked results
    ranked_csv = save_dir / f"target_{target_molecule_name}_ranked.csv"
    
    # Reorder columns to put the target metric up front for easy viewing
    cols = df.columns.tolist()
    front_cols = ['POLYMER_USED', 'DRUG', 'PREDICTED_CAPACITY', 'WATER_PH', 'CONCENTRATION']
    for col in front_cols:
        if col in cols: cols.remove(col)
    df = df[front_cols + cols]
    
    df.to_csv(ranked_csv, index=False, float_format="%.4f")
    logger.info(f"Saved ranked candidates to {ranked_csv}")
    
    # 7. Celebrate the winners
    logger.info("\n" + "="*70 + f"\n🏆 TOP 5 PREDICTED POLYMERS FOR {target_molecule_name.upper()}\n" + "="*70)
    for i, row in df.head(5).iterrows():
        logger.info(f"Rank {i+1}: Capacity = {row['PREDICTED_CAPACITY']:.4f} | Polymer: {row['POLYMER_USED']}")
    logger.info("="*70)
        
    return df
