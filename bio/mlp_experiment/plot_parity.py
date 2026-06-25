import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
from pathlib import Path
import bio
from bio.__global__ import PSMILES_DICT, SMILES_DICT

# Create reverse lookup dictionaries: SMILES -> Readable Name
REVERSE_PSMILES = {v: k for k, v in PSMILES_DICT.items()}
REVERSE_SMILES = {v: k for k, v in SMILES_DICT.items()}

ExperimentConfig = bio.mlp_experiment.Config
def plot_parity(exp_config: ExperimentConfig, save_dir: Path, log_file: Path):
    if not log_file.exists(): return

    # 1. Load predictions from the log
    df = pd.read_json(log_file, lines=True)
    if df.empty or "log_type" not in df.columns:
        return
    df = df[df["log_type"] == "prediction_trace"]
    if df.empty:
        return

    # 2. Translate SMILES to Readable Names
    def get_pair_name(row):
        poly_name = REVERSE_PSMILES.get(row.get("polymer", ""), "Unknown Polymer")
        drug_name = REVERSE_SMILES.get(row.get("drug", ""), "Unknown Drug")
        return f"{poly_name}\n+ {drug_name}"
    
    df["pair_name"] = df.apply(get_pair_name, axis=1)

    # 3. Calculate Absolute Error to find the worst predictions
    df["error"] = abs(df["actual"] - df["predicted"])

    # 4. Create the Scatter Plot
    plt.figure(figsize=(10, 10))
    sns.scatterplot(data=df, x="actual", y="predicted", hue="fold", alpha=0.8, s=50)

    # 5. Add the Parity Line
    min_val = min(df["actual"].min(), df["predicted"].min())
    max_val = max(df["actual"].max(), df["predicted"].max())
    padding = (max_val - min_val) * 0.1
    
    plt.plot(
        [min_val - padding, max_val + padding], 
        [min_val - padding, max_val + padding], 
        'r--', label='Perfect Prediction', zorder=0
    )

    # 6. Extract the Worst 20 Predictions and add Text Labels
    df_to_label = df.nlargest(20, "error")
    
    texts = []
    for _, row in df_to_label.iterrows():
        texts.append(
            plt.text(
                row["actual"], row["predicted"], str(row["pair_name"]), 
                fontsize=8, color='black', ha='center', va='center'
            )
        )
    
    # Repel labels so they don't overlap
    if texts:
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.6))

    # 7. Finalize and Save
    plt.xlabel("Actual Capacity")
    plt.ylabel("Predicted Capacity")
    plt.title(f"{exp_config.name} - Parity Plot")
    plt.legend()
    plt.tight_layout()
    
    save_path = save_dir / exp_config.name / "parity_plot.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


import pytest
@pytest.mark.todo
def test_():
    pass
