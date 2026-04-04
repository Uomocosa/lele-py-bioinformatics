# AHC Clustering Analysis Results
Generated from `pixi run ahc_clustering`

This directory contains the output of the Agglomerative Hierarchical Clustering (AHC) pipeline. These files provide insight into how the data points (drugs/polymers) were grouped and which features drive those groupings.

## 📊 Visualizations

* **`ahc_dendrogram.png`**: The "family tree" of the dataset. It illustrates the hierarchical merging process, showing how individual samples are grouped into clusters based on similarity.
* **`cluster_plot.jpg`**: A 2D scatter plot (typically using the first two Principal Components) that shows the spatial separation and overlap of the identified clusters.
* **`capacity_by_cluster.png`**: A performance-focused plot (likely a boxplot) showing the distribution of the target "capacity" metric within each cluster. This helps identify which clusters are "high performers."
* **`cluster_feature_heatmap.jpg`**: A grid visualization showing the relative intensity of specific features across all clusters. Great for spotting patterns like "Cluster 5 has high molecular weight but low solubility."
* **`cluster_decision_tree.jpg`**: An interpretability tool. This is a decision tree trained to predict the cluster labels, effectively providing a "rulebook" (e.g., *if feature A > 0.5, then Cluster 1*) for how the AHC divided the data.

## 📁 Data & Feature Analysis

* **`cluster_profile_top_10.txt`**: A summary report listing the 10 most defining features for every cluster. This is the "ID card" for each group.
* **`pca_loadings_full.csv`**: A matrix showing the weight (contribution) of every original feature to each Principal Component ($PC_1$, $PC_2$, etc.).
* **`pca_top_drivers_PC1.csv` & `pca_top_drivers_PC2.csv`**: Focused lists of the most influential features for the first and second Principal Components. These tell you exactly what physical or chemical properties are responsible for the most variance in your dataset.
