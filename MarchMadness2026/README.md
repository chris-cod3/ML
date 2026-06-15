# NCAA Tournament Prediction Model

Machine learning pipeline that predicts NCAA Men's Basketball Tournament matchups using
pairwise team comparison. An ensemble of seven models is trained on 23 seasons of box-score
and ranking data, then used to fill out full brackets and produce a shareable dashboard.

## Results

- **~73% game accuracy / 0.781 AUC** (XGBoost) on a held-out 2023–2025 test set
- Beats the "chalk" baseline (always pick the higher seed) of ~71%
- Models: Logistic Regression, Random Forest, SVM, MLP, XGBoost, LightGBM, CatBoost (+ ensemble)

The headline artifact is a single-page dashboard of the 2026 bracket prediction:
`results/dashboard/ncaa_2026_dashboard.png`.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Build the matchup dataset from Kaggle box scores + Massey ordinals
python build_dataset.py
python add_massey_ranks.py

# 2. Train + tune the models (writes models/*.pkl)
python retrain_models.py

# 3. Generate predictions and visualizations
python predict_2026.py              # 2026 bracket, all models + ensemble
python generate_all_brackets.py     # historical brackets (predicted vs actual)
python generate_dashboard.py        # single-page dashboard (PNG/PDF/HTML)
```

> **Data note:** the `data/` directory is git-ignored. Raw inputs come from the public
> [Kaggle March Machine Learning Mania](https://www.kaggle.com/c/march-machine-learning-mania-2025)
> CSVs (box scores, seeds, Massey ordinals) plus the scrapers in `src/data_collection/`.

## Project Structure

```
build_dataset.py          # Build per-team/per-season stats + matchup dataset
add_massey_ranks.py       # Merge Massey ranking systems into matchups
retrain_models.py         # Train + hyperparameter-tune the 7 models
predict_2026.py           # 2026 bracket predictions (uses BracketSimulator)
generate_all_brackets.py  # Bracket images for every year & model
generate_dashboard.py     # LinkedIn-ready summary dashboard

src/
  data_collection/        # ESPN / Sports-Reference / NCAA scrapers, leakage fixes
  prediction/             # BracketSimulator (matchup -> win prob -> bracket)
  visualization/          # Season stat-leader charts
  utils/                  # Plotting helpers
scripts/                  # Standalone demo/inspection scripts
notebooks/                # Executed exploration & modeling notebooks
models/                   # Saved trained models (*.pkl)
results/                  # Brackets, dashboards, and other visualizations
```

## Model Approach

Binary classification via pairwise team comparison:

- **Input:** Team A stats, Team B stats, and their differential features (incl. Massey ranks)
- **Output:** probability that Team A beats Team B
- **Training:** 2003–2022 regular-season + tournament games; tested on 2023–2025
- **Ensemble:** seven classifiers averaged to produce the final bracket

## Data Sources

- Kaggle "March Machine Learning Mania" datasets (box scores, seeds, Massey ordinals)
- ESPN public APIs
- Sports-Reference (advanced & basic team stats)
