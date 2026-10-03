# HX-IDS: Hierarchical Explainable AI for Intrusion Detection

Research codebase for a novel **Hierarchical Explainable AI (HX)** algorithm that maps
feature importance to OSI layers (Network / Transport / Application), benchmarked against
flat XAI (SHAP, LIME) on CIC-IDS-2017.

## Core idea

For a given prediction, total explainability is a weighted sum of layer-wise scores:

```
E_total = Σ_{l ∈ L} w_l · E_l ,   L = {Network, Transport, Application}
```

where `E_l` is the sum of local feature |SHAP| contributions within layer `l`, and `w_l`
is an optional dynamic weight learned from class-conditional attack signatures
(default 1.0 = unweighted baseline). Because SHAP is additive, layer aggregation is
**lossless** with respect to the underlying attribution — verified by the additive-fidelity
metric (R² ≈ 1.0).





## Repository layout

```
├── data/raw/                 # CIC-IDS-2017 CSVs (git-ignored)
├── notebooks/                # Step 1: prototyping (self-contained, executed)
│   ├── 01_eda_and_preprocessing.ipynb    # NaN/Inf handling, imbalance, layer map
│   ├── 02_baseline_training.ipynb        # RF/XGBoost + flat SHAP/LIME plots
│   └── 03_hx_algorithm_prototype.ipynb   # E_total = Σ w_l·E_l built up visually
```

## Evaluation metrics

Classification: Accuracy, Precision, Recall, F1 (macro), ROC-AUC (OvR).

Explainability:

- **Fidelity** — (a) additive R² between layer-sum reconstruction and model margins;
  (b) occlusion: permuting the top-ranked layer must disrupt predictions far more than
  the bottom-ranked layer (`fidelity_ratio`).
- **Sparsity** — entropy-based *effective cardinality*: attribution units an analyst must
  inspect, flat (~70 features) vs HX (≤4 layers) → `cognitive_load_reduction`.
- **Stability** — cosine similarity and top-layer agreement of layer scores under benign
  Gaussian perturbations (σ = 5% of feature std).

## Key assumptions

80/20 stratified split, `random_state=42` everywhere. Resampling (benign undersampling +
SMOTE for classes < 2000 samples) applied to the **training split only**. Scaling fit on
train only. XGBoost is the default XAI backbone (TreeSHAP-exact); RF supported identically.
