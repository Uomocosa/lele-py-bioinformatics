def test_usage():
    pscp = PeeSmileCapacityPredictor()
    trained_model = pscp.get_trained_model()
    pscp.save_trained_model(trained_model)
    trained_model = None
    trained_model = pscp.load_trained_model()
    assert trained_model is not None, "Failed to load trained model"
    
    trained_model.eval()
    input_df = pd.DataFrame({
        'POLYMER_USED': ["*/CCC[Fe]CCCC(=O)OCCCCOCCCNCC(*)=O"],
        'DRUG': ["CC(=O)OC1=CC=CC=C1C(=O)O"],
        'WATER_PH': [6.5],
        'CONCENTRATION': [12.5],
    })
    prediction = trained_model.predict(input_df)
    logger.info(f"Predicted Capacity: {prediction:.4f}")
