# SHAP Analysis Results: High Capacity Prediction Explanation

This folder contains the output of SHAP (SHapley Additive exPlanations) analysis performed on a `PeeSmileCapacityPredictor` model to understand why a specific polymer/drug combination produces a high capacity prediction.

## Files

| File | Description |
|------|-------------|
| `shap_summary.png` | Beeswarm plot showing the distribution of SHAP values for each feature |
| `shap_summary_bar.png` | Bar chart showing global feature importance |
| `top_features.csv` | Complete ranked list of all features by SHAP value |
| `feature_comparison.csv` | Side-by-side comparison of feature values for high vs normal prediction |

## Test Cases Analyzed

- **High-prediction polymer**: `*Nc1ccc(NC(=O)c2ccc(C(=O)NNC(=O)c3ccc(*)cc3)cc2)cc1` (a polymer with aromatic rings and amide groups)
- **Normal polymer**: `*CC(C)(C(=O)OCCO)*` (a simple PEG-like polymer)
- **Drug**: `CN(C)C(=N)N=C(N)N` (metformin)
- **pH**: 8.2
- **Concentration**: 12.5

## Key Findings

### Top Features Driving High Prediction

| Rank | Feature | SHAP Value | High Value | Normal Value | Interpretation |
|------|---------|------------|------------|--------------|----------------|
| 1 | `fingerprint_H_bit_12` | +17.29 | 1.0 | 0.0 | Polymer fingerprint bit - present only in high |
| 2 | `fingerprint_H_bit_177` | +16.94 | 1.0 | 0.0 | Polymer fingerprint bit - present only in high |
| 3 | `poly_num_aromatic_rings_sum_backbonefeaturizer` | +9.18 | 3.0 | 0.0 | High polymer has 3 aromatic rings vs none in normal |
| 4 | `fingerprint_H_bit_120` | +7.88 | 1.0 | 0.0 | Polymer fingerprint bit - present only in high |
| 5 | `fingerprint_H_bit_159` | +7.70 | 0.0 | 0.0 | Fingerprint present but same value (context matters) |
| 6 | `drug_fingerprint_bit_111` | +7.48 | 1.0 | 1.0 | Shared drug fingerprint (context-dependent) |
| 7 | `poly_num_rings_sum_backbonefeaturizer` | +7.36 | 3.0 | 0.0 | High polymer has 3 rings vs 0 in normal |
| 8 | `poly_smr_vsa5_sum_backbonefeaturizer` | +6.93 | 0.0 | 20.3 | **Negative contribution** - high has smaller VSA |
| 9 | `logd_at_WATER_PH_H` | +6.49 | 4.16 | 0.18 | High polymer more lipophilic at pH 8.2 |
| 10 | `drug_fingerprint_bit_175` | +6.28 | 0.0 | 0.0 | Drug fingerprint (context-dependent) |

### Key Insights

1. **Polymer aromatic rings are critical**: The high-capacity polymer has **3 aromatic rings** while the normal polymer has none. This is a major contributor to the high prediction.

2. **Polymer fingerprints dominate**: The top 5 features are all polymer-related fingerprints, indicating that specific molecular substructures in the polymer are the primary drivers of capacity prediction.

3. **Lipophilicity matters**: The `logd_at_WATER_PH_H` feature (logD at pH 8.2) is significantly higher for the high polymer (4.16 vs 0.18), suggesting increased lipophilicity contributes to higher capacity.

4. **Molecular size (SMR VSA5)**: The negative SHAP value for `poly_smr_vsa5` indicates that the smaller molecular surface area of the high polymer (0 vs 20.3) contributes positively to the prediction.

## Interpretation

The model learned that polymers with:
- **Aromatic ring structures** (3 rings)
- **Higher lipophilicity** at physiological pH
- **Smaller molecular surface area**

...tend to have higher drug delivery capacity. This makes sense from a pharmaceutical perspective - aromatic polymers may provide better π-π stacking interactions with drugs, and increased lipophilicity can enhance drug loading.

## Running the Analysis

To regenerate these results:

```bash
python -m bio.explain_high_prediction
```

Or from Python:

```python
from bio.explain_high_prediction import main
main()
```