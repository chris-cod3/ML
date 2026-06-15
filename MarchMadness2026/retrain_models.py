"""
Retrain 4 models (LogReg, XGBoost, RF, LightGBM) with 11-feature set.
  Data: Kaggle box scores + Massey ordinals (2003-2025)
  Train: 2003-2022 (~1,250 games)
  Test:  2023-2025 (~200 games)

Includes chalk baseline (always pick higher seed) for comparison.
Full hyperparameter re-tuning via RandomizedSearchCV.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import warnings
warnings.filterwarnings('ignore')

np.random.seed(99)

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / 'models'
MODELS_DIR.mkdir(exist_ok=True)

# ── 1. Load Data ──────────────────────────────────────────────────

df = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'kaggle_tourney.csv')

features = [
    'diff_rank_LMC', 'diff_ortg', 'diff_orb_pct', 'diff_drtg',
    'diff_rank_vol_POM', 'diff_ortg_std',
    'diff_win_pct', 'diff_ft_pct', 'diff_tov_pct',
]

# Fill any NaN (ranking gaps in early years)
for f in features:
    df[f] = df[f].fillna(0)

# Train: 2009-2022, Test: 2023-2025
train_mask = (df['season'] >= 2009) & (df['season'] <= 2022) & (df['season'] != 2020)
test_mask = df['season'].isin([2023, 2024, 2025])

X_train = df.loc[train_mask, features].values
X_test = df.loc[test_mask, features].values
y_train = df.loc[train_mask, 'home_win'].values
y_test = df.loc[test_mask, 'home_win'].values

# Scaler fit ONLY on training data (used for LogReg only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=99)

print(f'Training: {len(y_train)} games (seasons {sorted(df.loc[train_mask, "season"].unique())})')
print(f'Test: {len(y_test)} games (seasons {sorted(df.loc[test_mask, "season"].unique())})')
print(f'Features: {len(features)}')

# ── 2. Chalk Baseline ────────────────────────────────────────────

# "Chalk" = always pick the higher seed. On equal seeds, coin flip (random).
seed_diff_test = df.loc[test_mask, 'seed_diff'].values
chalk_preds_test = (seed_diff_test > 0).astype(int)
ties = seed_diff_test == 0
np.random.seed(99)
chalk_preds_test[ties] = np.random.randint(0, 2, size=ties.sum())

seed_diff_train = df.loc[train_mask, 'seed_diff'].values
chalk_preds_train = (seed_diff_train > 0).astype(int)
ties_tr = seed_diff_train == 0
chalk_preds_train[ties_tr] = np.random.randint(0, 2, size=ties_tr.sum())

chalk_train_acc = accuracy_score(y_train, chalk_preds_train)
chalk_test_acc = accuracy_score(y_test, chalk_preds_test)
print(f'\nChalk baseline — Train: {chalk_train_acc:.1%}  Test: {chalk_test_acc:.1%}')

# ── 3. Tune XGBoost ──────────────────────────────────────────────

xgb_param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [2, 3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 2, 5],
}

print('\nTuning XGBoost...')
xgb_search = RandomizedSearchCV(
    XGBClassifier(random_state=99, eval_metric='logloss', verbosity=0),
    xgb_param_grid, n_iter=100, cv=cv, scoring='accuracy',
    n_jobs=-1, random_state=99
)
xgb_search.fit(X_train, y_train)
print(f'  Best CV: {xgb_search.best_score_:.3f}  Params: {xgb_search.best_params_}')

# ── 4. Tune Random Forest ────────────────────────────────────────

rf_param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [3, 4, 5, 6, 7, None],
    'min_samples_split': [5, 10, 15, 20],
    'min_samples_leaf': [2, 4, 6, 8],
    'max_features': ['sqrt', 'log2', 0.5, 0.7],
}

print('\nTuning Random Forest...')
rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=99, n_jobs=-1),
    rf_param_grid, n_iter=100, cv=cv, scoring='accuracy',
    n_jobs=-1, random_state=99
)
rf_search.fit(X_train, y_train)
print(f'  Best CV: {rf_search.best_score_:.3f}  Params: {rf_search.best_params_}')

# ── 5. Tune LightGBM ────────────────────────────────────────────

lgbm_param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [2, 3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [7, 15, 31],
    'min_child_samples': [5, 10, 20, 30],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 2, 5],
}

print('\nTuning LightGBM...')
lgbm_search = RandomizedSearchCV(
    LGBMClassifier(random_state=99, verbosity=-1),
    lgbm_param_grid, n_iter=100, cv=cv, scoring='accuracy',
    n_jobs=-1, random_state=99
)
lgbm_search.fit(X_train, y_train)
print(f'  Best CV: {lgbm_search.best_score_:.3f}  Params: {lgbm_search.best_params_}')

# ── 5b. Tune CatBoost ────────────────────────────────────────────

cat_param_grid = {
    'iterations': [100, 200, 300],
    'depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'l2_leaf_reg': [1, 3, 5, 7],
    'border_count': [32, 64, 128],
}

print('\nTuning CatBoost...')
cat_search = RandomizedSearchCV(
    CatBoostClassifier(random_state=99, verbose=0, allow_writing_files=False),
    cat_param_grid, n_iter=50, cv=cv, scoring='accuracy',
    n_jobs=-1, random_state=99
)
cat_search.fit(X_train, y_train)
print(f'  Best CV: {cat_search.best_score_:.3f}  Params: {cat_search.best_params_}')

# ── 5c. Tune SVM (RBF) ───────────────────────────────────────────

svm_param_grid = {
    'C': [0.1, 0.5, 1, 2, 5, 10],
    'gamma': ['scale', 'auto', 0.01, 0.05, 0.1],
    'kernel': ['rbf'],
}

print('\nTuning SVM...')
svm_search = RandomizedSearchCV(
    SVC(probability=True, random_state=99),
    svm_param_grid, n_iter=30, cv=cv, scoring='accuracy',
    n_jobs=-1, random_state=99
)
svm_search.fit(X_train_scaled, y_train)
print(f'  Best CV: {svm_search.best_score_:.3f}  Params: {svm_search.best_params_}')

# ── 5d. Tune MLP ─────────────────────────────────────────────────

mlp_param_grid = {
    'hidden_layer_sizes': [(16,), (32,), (16, 8), (32, 16), (64, 32)],
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    'learning_rate_init': [0.001, 0.005, 0.01],
    'activation': ['relu', 'tanh'],
}

print('\nTuning MLP...')
mlp_search = RandomizedSearchCV(
    MLPClassifier(max_iter=500, early_stopping=True, random_state=99),
    mlp_param_grid, n_iter=30, cv=cv, scoring='accuracy',
    n_jobs=-1, random_state=99
)
mlp_search.fit(X_train_scaled, y_train)
print(f'  Best CV: {mlp_search.best_score_:.3f}  Params: {mlp_search.best_params_}')

# ── 6. Train Logistic Regression ──────────────────────────────────

print('\nTraining Logistic Regression...')
lr = LogisticRegression(max_iter=1000, random_state=99)
lr.fit(X_train_scaled, y_train)

# ── 6. Evaluate All Models ────────────────────────────────────────

def symmetric_predict(model, X):
    """Test-time averaging: predict both directions, average for symmetry."""
    prob_fwd = model.predict_proba(X)[:, 1]
    prob_rev = model.predict_proba(-X)[:, 1]
    return (prob_fwd + (1 - prob_rev)) / 2


def evaluate(model, X_tr, X_te, y_tr, y_te, name):
    y_prob_tr = symmetric_predict(model, X_tr)
    y_prob_te = symmetric_predict(model, X_te)
    y_pred_tr = (y_prob_tr >= 0.5).astype(int)
    y_pred_te = (y_prob_te >= 0.5).astype(int)
    train_acc = accuracy_score(y_tr, y_pred_tr)
    test_acc = accuracy_score(y_te, y_pred_te)
    auc = roc_auc_score(y_te, y_prob_te)
    overfit = train_acc - test_acc
    return {'name': name, 'train_acc': train_acc, 'test_acc': test_acc,
            'auc': auc, 'overfit': overfit}

results = [
    evaluate(lr, X_train_scaled, X_test_scaled, y_train, y_test, 'LogReg'),
    evaluate(xgb_search.best_estimator_, X_train, X_test, y_train, y_test, 'XGBoost'),
    evaluate(rf_search.best_estimator_, X_train, X_test, y_train, y_test, 'RF'),
    evaluate(lgbm_search.best_estimator_, X_train, X_test, y_train, y_test, 'LightGBM'),
    evaluate(cat_search.best_estimator_, X_train, X_test, y_train, y_test, 'CatBoost'),
    evaluate(svm_search.best_estimator_, X_train_scaled, X_test_scaled, y_train, y_test, 'SVM'),
    evaluate(mlp_search.best_estimator_, X_train_scaled, X_test_scaled, y_train, y_test, 'MLP'),
]

results_df = pd.DataFrame(results).sort_values('test_acc', ascending=False)

print(f'\n{"="*70}')
print(f'RESULTS — Train: 2009-2022 ({len(y_train)} games) | Test: 2023-2025 ({len(y_test)} games)')
print(f'{"="*70}')
print(f'{"Model":<20s} {"Train":>7s} {"Test":>7s} {"AUC":>7s} {"Overfit":>8s}')
print('-'*70)
print(f'{"Chalk (seed)":<20s} {chalk_train_acc:>6.1%} {chalk_test_acc:>6.1%} {"N/A":>7s} {chalk_train_acc - chalk_test_acc:>7.1%}')
for _, r in results_df.iterrows():
    print(f'{r["name"]:<20s} {r["train_acc"]:>6.1%} {r["test_acc"]:>6.1%} '
          f'{r["auc"]:>6.3f} {r["overfit"]:>7.1%}')
print(f'{"="*70}')

# Per-year breakdown for test years
print('\nPer-year test accuracy:')
test_configs = [
    ('LogReg', lr, True),
    ('XGBoost', xgb_search.best_estimator_, False),
    ('RF', rf_search.best_estimator_, False),
    ('LightGBM', lgbm_search.best_estimator_, False),
    ('CatBoost', cat_search.best_estimator_, False),
    ('SVM', svm_search.best_estimator_, True),
    ('MLP', mlp_search.best_estimator_, True),
]

test_years = [2023, 2024, 2025]
hdr = ''.join(f' {y:>7}' for y in test_years)
print(f'{"Model":<20s}{hdr}')
print('-'*55)

# Chalk per year
chalk_accs = []
for year in test_years:
    mask = df['season'] == year
    sd_yr = df.loc[mask, 'seed_diff'].values
    y_yr = df.loc[mask, 'home_win'].values
    chalk_yr = (sd_yr > 0).astype(int)
    ties_yr = sd_yr == 0
    chalk_yr[ties_yr] = np.random.randint(0, 2, size=ties_yr.sum())
    chalk_accs.append(accuracy_score(y_yr, chalk_yr))
vals = ''.join(f' {a:>6.1%}' for a in chalk_accs)
print(f'{"Chalk (seed)":<20s}{vals}')

for mname, model, scaled in test_configs:
    accs = []
    for year in test_years:
        mask = df['season'] == year
        X = df.loc[mask, features].values
        if scaled:
            X = scaler.transform(X)
        y = df.loc[mask, 'home_win'].values
        prob = symmetric_predict(model, X)
        acc = accuracy_score(y, (prob >= 0.5).astype(int))
        accs.append(acc)
    vals = ''.join(f' {a:>6.1%}' for a in accs)
    print(f'{mname:<20s}{vals}')

# Upset analysis
print('\nUpset prediction analysis (test set):')
test_df = df[test_mask].copy()
test_df['upset'] = ((test_df['home_seed'] > test_df['away_seed']) & (test_df['home_win'] == 1)) | \
                   ((test_df['away_seed'] > test_df['home_seed']) & (test_df['home_win'] == 0))
upset_mask_arr = test_df['upset'].values
chalk_mask_arr = ~upset_mask_arr
print(f'Upsets in test set: {upset_mask_arr.sum()} / {len(y_test)} ({upset_mask_arr.mean():.1%})')

# Chalk upset analysis
chalk_upset_acc = accuracy_score(y_test[upset_mask_arr], chalk_preds_test[upset_mask_arr])
chalk_chalk_acc = accuracy_score(y_test[chalk_mask_arr], chalk_preds_test[chalk_mask_arr])
print(f'  {"Chalk (seed)":<20s} Upset: {chalk_upset_acc:.1%}  Chalk: {chalk_chalk_acc:.1%}')

for mname, model, scaled in test_configs:
    X = df.loc[test_mask, features].values
    if scaled:
        X = scaler.transform(X)
    prob = symmetric_predict(model, X)
    preds = (prob >= 0.5).astype(int)
    upset_acc = accuracy_score(y_test[upset_mask_arr], preds[upset_mask_arr])
    chalk_acc = accuracy_score(y_test[chalk_mask_arr], preds[chalk_mask_arr])
    print(f'  {mname:<20s} Upset: {upset_acc:.1%}  Chalk: {chalk_acc:.1%}')

# ── 7. Save Models ────────────────────────────────────────────────

# Clean old models
for f in MODELS_DIR.glob('*.pkl'):
    f.unlink()

joblib.dump(lr, MODELS_DIR / 'logistic_regression.pkl')
joblib.dump(xgb_search.best_estimator_, MODELS_DIR / 'xgboost_tuned.pkl')
joblib.dump(rf_search.best_estimator_, MODELS_DIR / 'random_forest_tuned.pkl')
joblib.dump(lgbm_search.best_estimator_, MODELS_DIR / 'lightgbm_tuned.pkl')
joblib.dump(cat_search.best_estimator_, MODELS_DIR / 'catboost_tuned.pkl')
joblib.dump(svm_search.best_estimator_, MODELS_DIR / 'svm_tuned.pkl')
joblib.dump(mlp_search.best_estimator_, MODELS_DIR / 'mlp_tuned.pkl')
joblib.dump(scaler, MODELS_DIR / 'scaler.pkl')
joblib.dump(features, MODELS_DIR / 'features.pkl')

print(f'\nSaved {len(list(MODELS_DIR.glob("*.pkl")))} model files to {MODELS_DIR}')
print('Done!')
