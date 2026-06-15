"""
Generate a single-page PDF dashboard for LinkedIn showcasing 2026 NCAA
Tournament predictions.

Uses LogReg as the featured model. Draws bracket directly in dark theme.
Computes historical accuracy and feature importance from model artifacts.
Data source: Kaggle (same as generate_all_brackets.py).

Output: results/dashboard/ncaa_2026_dashboard.pdf
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Rectangle
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'
DASH_DIR = PROJECT_ROOT / 'results' / 'dashboard'
DASH_DIR.mkdir(parents=True, exist_ok=True)

# ── Colors ────────────────────────────────────────────────────────

BG       = '#0d1117'
CARD_BG  = '#161b22'
BORDER   = '#30363d'
WHITE    = '#e6edf3'
MUTED    = '#8b949e'
GOLD     = '#f0c040'
BLUE     = '#58a6ff'
DK_BLUE  = '#1f6feb'
GREEN    = '#3fb950'
LINE_CLR = '#3d444d'
SEED_CLR = '#58a6ff'
PICK_CLR = '#7ee787'
R64_CLR  = '#c9d1d9'
CHAMP_CLR= '#f0c040'
PURPLE   = '#c084fc'
CYAN     = '#56d4dd'

# Per-model colors used as a legend across the performance table
# and the multi-model consensus bracket alts.
MODEL_COLORS = {
    'LR':  '#4493f8',  # blue
    'XGB': '#f85149',  # red
    'RF':  '#3fb950',  # green
    'LGB': '#bc8cff',  # purple
    'CB':  '#f0c040',  # gold
    'SVM': '#56d4dd',  # cyan
    'MLP': '#ff8700',  # orange
}
MODEL_NAME_TO_ABBR = {
    'LogReg': 'LR', 'XGBoost': 'XGB', 'RF': 'RF', 'LightGBM': 'LGB',
    'CatBoost': 'CB', 'SVM': 'SVM', 'MLP': 'MLP',
}

# ── 2026 Bracket (uses Kaggle TeamNames from MTeams.csv) ─────────

BRACKET_2026 = [
    # East Region
    [
        {'name': 'Duke', 'seed': 1}, {'name': 'Connecticut', 'seed': 2},
        {'name': 'Michigan State', 'seed': 3}, {'name': 'Kansas', 'seed': 4},
        {'name': "St John's", 'seed': 5}, {'name': 'Louisville', 'seed': 6},
        {'name': 'UCLA', 'seed': 7}, {'name': 'Ohio St', 'seed': 8},
        {'name': 'TCU', 'seed': 9}, {'name': 'UCF', 'seed': 10},
        {'name': 'South Florida', 'seed': 11}, {'name': 'Northern Iowa', 'seed': 12},
        {'name': 'Cal Baptist', 'seed': 13}, {'name': 'N Dakota St', 'seed': 14},
        {'name': 'Furman', 'seed': 15}, {'name': 'Siena', 'seed': 16},
    ],
    # South Region
    [
        {'name': 'Florida', 'seed': 1}, {'name': 'Houston', 'seed': 2},
        {'name': 'Illinois', 'seed': 3}, {'name': 'Nebraska', 'seed': 4},
        {'name': 'Vanderbilt', 'seed': 5}, {'name': 'North Carolina', 'seed': 6},
        {'name': "St Mary's CA", 'seed': 7}, {'name': 'Clemson', 'seed': 8},
        {'name': 'Iowa', 'seed': 9}, {'name': 'Texas A&M', 'seed': 10},
        {'name': 'VCU', 'seed': 11}, {'name': 'McNeese St', 'seed': 12},
        {'name': 'Troy', 'seed': 13}, {'name': 'Penn', 'seed': 14},
        {'name': 'Idaho', 'seed': 15}, {'name': 'Prairie View', 'seed': 16},
    ],
    # West Region
    [
        {'name': 'Arizona', 'seed': 1}, {'name': 'Purdue', 'seed': 2},
        {'name': 'Gonzaga', 'seed': 3}, {'name': 'Arkansas', 'seed': 4},
        {'name': 'Wisconsin', 'seed': 5}, {'name': 'BYU', 'seed': 6},
        {'name': 'Miami FL', 'seed': 7}, {'name': 'Villanova', 'seed': 8},
        {'name': 'Utah St', 'seed': 9}, {'name': 'Missouri', 'seed': 10},
        {'name': 'Texas', 'seed': 11}, {'name': 'High Point', 'seed': 12},
        {'name': 'Hawaii', 'seed': 13}, {'name': 'Kennesaw St', 'seed': 14},
        {'name': 'Queens', 'seed': 15}, {'name': 'LIU', 'seed': 16},
    ],
    # Midwest Region
    [
        {'name': 'Michigan', 'seed': 1}, {'name': 'Iowa St', 'seed': 2},
        {'name': 'Virginia', 'seed': 3}, {'name': 'Alabama', 'seed': 4},
        {'name': 'Texas Tech', 'seed': 5}, {'name': 'Tennessee', 'seed': 6},
        {'name': 'Kentucky', 'seed': 7}, {'name': 'Georgia', 'seed': 8},
        {'name': 'St Louis', 'seed': 9}, {'name': 'Santa Clara', 'seed': 10},
        {'name': 'SMU', 'seed': 11}, {'name': 'Akron', 'seed': 12},
        {'name': 'Hofstra', 'seed': 13}, {'name': 'Wright St', 'seed': 14},
        {'name': 'Tennessee St', 'seed': 15}, {'name': 'Howard', 'seed': 16},
    ],
]

REGION_NAMES = ['East', 'South', 'West', 'Midwest']

# ── 2026 Actual Tournament Results (winners by round, in matchup order) ──
# Order matches MATCHUP_ORDER pairings: (1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15)
ACTUAL_2026 = {
    0: {  # East
        'R64': ['Duke', 'TCU', "St John's", 'Kansas', 'Louisville', 'Michigan State', 'UCLA', 'Connecticut'],
        'R32': ['Duke', "St John's", 'Michigan State', 'Connecticut'],
        'S16': ['Duke', 'Connecticut'],
        'E8':  ['Connecticut'],
    },
    1: {  # South
        'R64': ['Florida', 'Iowa', 'Vanderbilt', 'Nebraska', 'VCU', 'Illinois', 'Texas A&M', 'Houston'],
        'R32': ['Iowa', 'Vanderbilt', 'Illinois', 'Houston'],
        'S16': ['Iowa', 'Illinois'],
        'E8':  ['Illinois'],
    },
    2: {  # West
        'R64': ['Arizona', 'Utah St', 'High Point', 'Arkansas', 'Texas', 'Gonzaga', 'Miami FL', 'Purdue'],
        'R32': ['Arizona', 'Arkansas', 'Texas', 'Purdue'],
        'S16': ['Arizona', 'Purdue'],
        'E8':  ['Arizona'],
    },
    3: {  # Midwest
        'R64': ['Michigan', 'St Louis', 'Texas Tech', 'Alabama', 'Tennessee', 'Virginia', 'Kentucky', 'Iowa St'],
        'R32': ['Michigan', 'Alabama', 'Tennessee', 'Iowa St'],
        'S16': ['Michigan', 'Tennessee'],
        'E8':  ['Michigan'],
    },
}
# F4 winners by pair index: pair 0 = regions [East, South], pair 1 = regions [West, Midwest]
F4_2026 = ['Connecticut', 'Michigan']
CHAMP_2026 = 'Michigan'


def build_actual_tree_2026(bracket):
    """Build an actual-results tree in the same shape as a prediction tree."""
    teams_by_name = [{t['name']: t for t in region} for region in bracket]
    tree = {'R64_teams': {}}
    for ri in range(4):
        region_sorted = sorted(bracket[ri], key=lambda x: x['seed'])
        ordered = []
        for i in range(0, 16, 2):
            ordered.append(region_sorted[MATCHUP_ORDER[i]])
            ordered.append(region_sorted[MATCHUP_ORDER[i + 1]])
        tree['R64_teams'][ri] = ordered

    tree['R64_winners'] = {}
    tree['R32_winners'] = {}
    tree['S16_winners'] = {}
    tree['E8_winners']  = {}
    for ri in range(4):
        a = ACTUAL_2026[ri]
        tree['R64_winners'][ri] = [teams_by_name[ri][n] for n in a['R64']]
        tree['R32_winners'][ri] = [teams_by_name[ri][n] for n in a['R32']]
        tree['S16_winners'][ri] = [teams_by_name[ri][n] for n in a['S16']]
        tree['E8_winners'][ri]  = [teams_by_name[ri][n] for n in a['E8']]

    f4_winners = []
    for pair_idx, (ra, rb) in enumerate([(0, 1), (2, 3)]):
        wn = F4_2026[pair_idx]
        team = teams_by_name[ra].get(wn) or teams_by_name[rb].get(wn)
        f4_winners.append(team)
    tree['F4_winners'] = f4_winners

    for ri in range(4):
        if CHAMP_2026 in teams_by_name[ri]:
            tree['Championship_winner'] = teams_by_name[ri][CHAMP_2026]
            break
    return tree

ALL_MODELS = {
    'LogReg': {'model': 'logistic_regression.pkl', 'scaler': 'scaler.pkl', 'features': 'features.pkl'},
    'XGBoost': {'model': 'xgboost_tuned.pkl', 'scaler': 'scaler.pkl', 'features': 'features.pkl'},
    'RF': {'model': 'random_forest_tuned.pkl', 'scaler': 'scaler.pkl', 'features': 'features.pkl'},
    'LightGBM': {'model': 'lightgbm_tuned.pkl', 'scaler': 'scaler.pkl', 'features': 'features.pkl'},
    'CatBoost': {'model': 'catboost_tuned.pkl', 'scaler': 'scaler.pkl', 'features': 'features.pkl'},
    'SVM': {'model': 'svm_tuned.pkl', 'scaler': 'scaler.pkl', 'features': 'features.pkl'},
    'MLP': {'model': 'mlp_tuned.pkl', 'scaler': 'scaler.pkl', 'features': 'features.pkl'},
}

# Map Kaggle region codes to human-readable names per year
REGION_CODE_NAMES = {
    2018: {'W': 'South', 'X': 'West', 'Y': 'East', 'Z': 'Midwest'},
    2019: {'W': 'East', 'X': 'West', 'Y': 'South', 'Z': 'Midwest'},
    2021: {'W': 'West', 'X': 'East', 'Y': 'South', 'Z': 'Midwest'},
    2022: {'W': 'West', 'X': 'East', 'Y': 'South', 'Z': 'Midwest'},
    2023: {'W': 'South', 'X': 'East', 'Y': 'Midwest', 'Z': 'West'},
    2024: {'W': 'South', 'X': 'East', 'Y': 'Midwest', 'Z': 'West'},
    2025: {'W': 'East', 'X': 'South', 'Y': 'Midwest', 'Z': 'West'},
}

TEST_YEARS = [2023, 2024, 2025]
MATCHUP_ORDER = [0, 15, 7, 8, 4, 11, 3, 12, 5, 10, 2, 13, 6, 9, 1, 14]

SHORT_NAMES = {
    'Connecticut': 'UConn', 'North Carolina': 'UNC',
    'Michigan St': 'Mich St', 'Michigan State': 'Mich St', 'Mississippi St': 'Miss St',
    'San Diego St': 'SDSU', 'South Dakota St': 'SD State',
    'FL Atlantic': 'FAU', 'W Kentucky': 'W Kentucky',
    'Long Beach St': 'Long Beach', 'Iowa St': 'Iowa St',
    'Colorado St': 'Colorado St', 'South Carolina': 'S Carolina',
    'Utah St': 'Utah St', 'James Madison': 'James Mad.',
    'TCU': 'TCU', 'BYU': 'BYU',
    "St Mary's CA": "St Mary's",
    "St Peter's": "St Peter's",
    'NC State': 'NC State', 'Texas A&M': 'Texas A&M',
    'VCU': 'VCU', 'SMU': 'SMU',
    'Prairie View': 'PV A&M', 'Cal Baptist': 'Cal Baptist',
    'LIU': 'LIU', 'Queens': 'Queens',
    "St John's": "St John's", 'N Dakota St': 'ND State',
    'Northern Iowa': 'N Iowa', 'Kennesaw St': 'Kennesaw',
    'Tennessee St': 'Tenn St', 'South Florida': 'S Florida',
    'Wright St': 'Wright St', 'Loyola-Chicago': 'Loyola-Chi',
    'Penn St': 'Penn St', 'Kansas St': 'Kansas St',
    'Murray St': 'Murray St', 'NM State': 'NM State',
    'Boise St': 'Boise St', 'Wichita St': 'Wichita St',
    'Montana St': 'Montana St', 'Cleveland St': 'Cleveland St',
    'Oregon St': 'Oregon St', 'Ohio St': 'Ohio St',
    'Georgia St': 'Georgia St', 'North Texas': 'N Texas',
    'Florida St': 'Florida St', 'Texas St': 'Texas St',
    'Kent St': 'Kent St', 'Missouri St': 'Missouri St',
    'Ark Little Rock': 'Ark LR', 'CS Bakersfield': 'CS Bakers.',
    'CS Fullerton': 'CS Fuller.', 'E Washington': 'E Wash',
    'G Washington': 'G Wash', 'N Kentucky': 'N Kentucky',
    'SIU Edwardsville': 'SIUE', 'SE Missouri St': 'SE Missouri',
    'Appalachian St': 'App St', 'F Dickinson': 'FDU',
    'MTSU': 'MTSU', 'Col Charleston': 'Charleston',
    'Miami FL': 'Miami', 'Morehead St': 'Morehead',
    'Norfolk St': 'Norfolk St', 'Alabama St': 'Alabama St',
    'McNeese St': 'McNeese', 'WI Green Bay': 'Green Bay',
    'WI Milwaukee': 'Milwaukee', 'High Point': 'High Pt',
    'Sacramento St': 'Sacramento', 'Weber St': 'Weber St',
    'St Louis': 'St Louis',
}

FEATURE_DESCRIPTIONS = {
    'diff_rank_POM': 'KenPom Ranking',
    'diff_rank_LMC': 'LMC Ranking',
    'diff_pace': 'Pace (possessions/game)',
    'diff_three_par': '3-Point Attempt %',
    'diff_ortg': 'Offensive Rating (pts/100 poss)',
    'diff_drtg': 'Defensive Rating (pts allowed/100 poss)',
    'diff_ft_pct': 'Free Throw %',
    'diff_tov_pct': 'Turnover %',
    'diff_orb_pct': 'Offensive Rebound %',
    'diff_trb_pct': 'Total Rebound %',
    'diff_ast_rate': 'Assist %',
    'diff_stl_rate': 'Steal %',
    'diff_blk_rate': 'Block %',
    'diff_pts_against': 'Points Allowed',
    'diff_win_pct': 'Win Percentage',
    'diff_efg_pct': 'Effective FG %',
    'seed_diff': 'Seed Difference',
    'diff_margin_std': 'Margin Volatility (std dev)',
    'diff_ortg_std': 'Offensive Rating Volatility',
    'diff_drtg_std': 'Defensive Rating Volatility',
    'diff_ct_winpct': 'Conf. Tourney Win %',
    'diff_ct_wins': 'Conf. Tourney Wins',
    'diff_ct_margin': 'Conf. Tourney Avg Margin',
    'diff_rank_vol_POM': 'KenPom Ranking Volatility',
    'diff_rank_vol_LMC': 'LMC Ranking Volatility',
    'diff_rank_trend_POM': 'KenPom Ranking Trend',
    'diff_rank_trend_LMC': 'LMC Ranking Trend',
}


def shorten(name):
    return SHORT_NAMES.get(name, name)


# ── Kaggle data loading (mirrors generate_all_brackets.py) ────────

# Feature column mapping: feature name -> column in kaggle_team_stats.csv
FEATURE_COL_MAP = {
    'diff_pace': 'Pace',
    'diff_ortg': 'ORtg',
    'diff_drtg': 'DRtg',
    'diff_efg_pct': 'EFG_pct',
    'diff_tov_pct': 'TOV_pct',
    'diff_ft_pct': 'FT_pct',
    'diff_three_par': 'ThreeP_rate',
    'diff_orb_pct': 'ORB_pct',
    'diff_trb_pct': 'TRB_pct',
    'diff_ast_rate': 'Ast_rate',
    'diff_stl_rate': 'Stl_rate',
    'diff_blk_rate': 'Blk_rate',
    'diff_win_pct': 'WinPct',
    'diff_pts_against': 'PtsAgainst',
    'diff_rank_POM': 'Rank_POM',
    'diff_rank_LMC': 'Rank_LMC',
    'diff_margin_std': 'Margin_std',
    'diff_ortg_std': 'ORtg_std',
    'diff_drtg_std': 'DRtg_std',
    'diff_ct_winpct': 'CT_WinPct',
    'diff_ct_margin': 'CT_AvgMargin',
    'diff_ct_wins': 'CT_Wins',
    'diff_rank_vol_POM': 'RankVol_POM',
    'diff_rank_vol_LMC': 'RankVol_LMC',
    'diff_rank_trend_POM': 'RankTrend_POM',
    'diff_rank_trend_LMC': 'RankTrend_LMC',
}

RANK_FEATURES = {'diff_rank_POM', 'diff_rank_LMC'}


def compute_matchup_features(team_a_stats, team_b_stats, seed_a, seed_b, features):
    """Compute feature vector for a matchup (team_a as 'home').

    Convention (matches build_dataset.py):
      - Regular stats: home - away (positive = home is better)
      - Rankings: away_rank - home_rank (positive = home is better)
      - seed_diff: away_seed - home_seed (positive = home has better seed)
    """
    def _g(stats, key):
        v = stats.get(key, 0)
        return 0 if pd.isna(v) else float(v)

    feat_vals = []
    for feat in features:
        if feat == 'seed_diff':
            feat_vals.append(seed_b - seed_a)
        elif feat == 'diff_ortg_adj':
            adj_a = _g(team_a_stats, 'ORtg') - _g(team_a_stats, 'ORtg_std')
            adj_b = _g(team_b_stats, 'ORtg') - _g(team_b_stats, 'ORtg_std')
            feat_vals.append(adj_a - adj_b)
        elif feat == 'diff_drtg_adj':
            adj_a = _g(team_a_stats, 'DRtg') + _g(team_a_stats, 'DRtg_std')
            adj_b = _g(team_b_stats, 'DRtg') + _g(team_b_stats, 'DRtg_std')
            feat_vals.append(adj_a - adj_b)
        elif feat in FEATURE_COL_MAP:
            col = FEATURE_COL_MAP[feat]
            va = team_a_stats.get(col, 0)
            vb = team_b_stats.get(col, 0)
            va = 0 if pd.isna(va) else float(va)
            vb = 0 if pd.isna(vb) else float(vb)
            if feat in RANK_FEATURES:
                feat_vals.append(vb - va)  # away_rank - home_rank
            else:
                feat_vals.append(va - vb)  # home - away
        else:
            feat_vals.append(0)
    return np.array(feat_vals).reshape(1, -1)


class DirectPredictor:
    def __init__(self, model_path, scaler_path, features_path, team_stats_df, season):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.features = joblib.load(features_path)
        model_type = type(self.model).__name__
        self.needs_scaling = model_type in ('LogisticRegression', 'SVC', 'MLPClassifier')
        ss = team_stats_df[team_stats_df.Season == season]
        self.stats_by_id = {row.TeamID: row for _, row in ss.iterrows()}

    def _symmetric_prob(self, X):
        """Test-time averaging: predict both directions, average for symmetry."""
        prob_fwd = self.model.predict_proba(X)[0, 1]
        prob_rev = self.model.predict_proba(-X)[0, 1]
        return (prob_fwd + (1 - prob_rev)) / 2

    def predict(self, team_a, team_b):
        tid_a = team_a.get('team_id')
        tid_b = team_b.get('team_id')
        stats_a = self.stats_by_id.get(tid_a)
        stats_b = self.stats_by_id.get(tid_b)
        if stats_a is None or stats_b is None:
            if team_a['seed'] <= team_b['seed']:
                return team_a, 0.5 + (team_b['seed'] - team_a['seed']) * 0.03
            else:
                return team_b, 0.5 + (team_a['seed'] - team_b['seed']) * 0.03
        X = compute_matchup_features(stats_a, stats_b, team_a['seed'], team_b['seed'], self.features)
        if self.needs_scaling:
            X = self.scaler.transform(X)
        prob_a = self._symmetric_prob(X)
        if prob_a >= 0.5:
            return team_a, prob_a
        else:
            return team_b, 1 - prob_a


class EnsemblePredictor:
    def __init__(self, model_configs, team_stats_df, season):
        self.predictors = []
        for cfg in model_configs:
            self.predictors.append(DirectPredictor(
                cfg['model'], cfg['scaler'], cfg['features'], team_stats_df, season))

    def predict(self, team_a, team_b):
        probs_a = []
        for p in self.predictors:
            stats_a = p.stats_by_id.get(team_a.get('team_id'))
            stats_b = p.stats_by_id.get(team_b.get('team_id'))
            if stats_a is None or stats_b is None:
                if team_a['seed'] <= team_b['seed']:
                    probs_a.append(0.5 + (team_b['seed'] - team_a['seed']) * 0.03)
                else:
                    probs_a.append(0.5 - (team_a['seed'] - team_b['seed']) * 0.03)
                continue
            X = compute_matchup_features(stats_a, stats_b, team_a['seed'], team_b['seed'], p.features)
            if p.needs_scaling:
                X = p.scaler.transform(X)
            probs_a.append(p._symmetric_prob(X))
        avg_prob = np.mean(probs_a)
        if avg_prob >= 0.5:
            return team_a, avg_prob
        else:
            return team_b, 1 - avg_prob


def create_predictor(model_cfg, team_stats_df, season):
    if 'ensemble' in model_cfg:
        configs = [
            {'model': str(MODELS_DIR / c['model']),
             'scaler': str(MODELS_DIR / c['scaler']),
             'features': str(MODELS_DIR / c['features'])}
            for c in model_cfg['ensemble']
        ]
        return EnsemblePredictor(configs, team_stats_df, season)
    else:
        return DirectPredictor(
            str(MODELS_DIR / model_cfg['model']),
            str(MODELS_DIR / model_cfg['scaler']),
            str(MODELS_DIR / model_cfg['features']),
            team_stats_df, season)


def simulate_bracket(bracket, predictor):
    """Simulate full tournament. Returns prediction tree."""
    r64_teams = {}
    for ri, region in enumerate(bracket):
        region_sorted = sorted(region, key=lambda x: x['seed'])
        ordered = []
        for i in range(0, 16, 2):
            ordered.append(region_sorted[MATCHUP_ORDER[i]])
            ordered.append(region_sorted[MATCHUP_ORDER[i + 1]])
        r64_teams[ri] = ordered

    tree = {'R64_teams': r64_teams}

    tree['R64_winners'] = {}
    for ri in range(4):
        winners = []
        for i in range(0, 16, 2):
            ta = r64_teams[ri][i]
            tb = r64_teams[ri][i + 1]
            winner, prob = predictor.predict(ta, tb)
            winner = dict(winner)
            winner['prob'] = prob
            winners.append(winner)
        tree['R64_winners'][ri] = winners

    tree['R32_winners'] = {}
    for ri in range(4):
        winners = []
        prev = tree['R64_winners'][ri]
        for i in range(0, 8, 2):
            winner, prob = predictor.predict(prev[i], prev[i + 1])
            winner = dict(winner)
            winner['prob'] = prob
            winners.append(winner)
        tree['R32_winners'][ri] = winners

    tree['S16_winners'] = {}
    for ri in range(4):
        winners = []
        prev = tree['R32_winners'][ri]
        for i in range(0, 4, 2):
            winner, prob = predictor.predict(prev[i], prev[i + 1])
            winner = dict(winner)
            winner['prob'] = prob
            winners.append(winner)
        tree['S16_winners'][ri] = winners

    tree['E8_winners'] = {}
    for ri in range(4):
        prev = tree['S16_winners'][ri]
        winner, prob = predictor.predict(prev[0], prev[1])
        winner = dict(winner)
        winner['prob'] = prob
        tree['E8_winners'][ri] = [winner]

    f4_winners = []
    for pair in [(0, 1), (2, 3)]:
        ta = tree['E8_winners'][pair[0]][0]
        tb = tree['E8_winners'][pair[1]][0]
        winner, prob = predictor.predict(ta, tb)
        winner = dict(winner)
        winner['prob'] = prob
        f4_winners.append(winner)
    tree['F4_winners'] = f4_winners

    winner, prob = predictor.predict(f4_winners[0], f4_winners[1])
    winner = dict(winner)
    winner['prob'] = prob
    tree['Championship_winner'] = winner

    return tree


# ── Kaggle bracket building for historical years ──────────────────

def load_kaggle_data():
    seeds = pd.read_csv(DATA_DIR / 'MNCAATourneySeeds.csv')
    tourney = pd.read_csv(DATA_DIR / 'MNCAATourneyDetailedResults.csv')
    teams = pd.read_csv(DATA_DIR / 'MTeams.csv')
    team_stats = pd.read_csv(PROCESSED_DIR / 'kaggle_team_stats.csv')
    seeds['Region'] = seeds.Seed.str[0]
    seeds['SeedNum'] = seeds.Seed.str[1:3].astype(int)
    seeds['PlayIn'] = seeds.Seed.str[3:]
    id_to_name = dict(zip(teams.TeamID, teams.TeamName))
    return seeds, tourney, teams, team_stats, id_to_name


def get_play_in_winners(tourney_df, season):
    playin = tourney_df[(tourney_df.Season == season) & (tourney_df.DayNum <= 135)]
    return set(playin.WTeamID.tolist())


def build_bracket_from_seeds(seeds_df, season, id_to_name, tourney_df):
    year_seeds = seeds_df[seeds_df.Season == season].copy()
    playin_winners = get_play_in_winners(tourney_df, season)
    regions = {}
    for _, row in year_seeds.iterrows():
        rc = row['Region']
        if rc not in regions:
            regions[rc] = {}
        seed_num = row['SeedNum']
        team_id = row['TeamID']
        playin_tag = row['PlayIn']
        if playin_tag in ('a', 'b'):
            if team_id in playin_winners:
                regions[rc][seed_num] = team_id
        else:
            regions[rc][seed_num] = team_id
    bracket = []
    region_codes = sorted(regions.keys())
    for rc in region_codes:
        region_teams = []
        for seed_num in range(1, 17):
            tid = regions[rc].get(seed_num)
            if tid is not None:
                name = id_to_name.get(tid, f'Team_{tid}')
                region_teams.append({'name': name, 'seed': seed_num, 'team_id': tid})
            else:
                region_teams.append({'name': f'Unknown_{seed_num}', 'seed': seed_num, 'team_id': None})
        bracket.append(region_teams)
    return bracket, region_codes


def build_actual_tree(bracket, tourney_df, season, seeds_df, id_to_name):
    """Build actual result tree from Kaggle tournament data."""
    year_games = tourney_df[tourney_df.Season == season].copy()
    year_games = year_games[year_games.DayNum >= 136].sort_values('DayNum')

    year_seeds = seeds_df[seeds_df.Season == season].copy()
    tid_to_seed = {}
    for _, row in year_seeds.iterrows():
        tid_to_seed[row['TeamID']] = (row['Region'], row['SeedNum'])

    actual_results = []
    for _, g in year_games.iterrows():
        w_region, w_seed = tid_to_seed.get(g.WTeamID, ('?', 0))
        l_region, l_seed = tid_to_seed.get(g.LTeamID, ('?', 0))
        actual_results.append({
            'day': g.DayNum,
            'winner_id': g.WTeamID,
            'winner_name': id_to_name.get(g.WTeamID, '?'),
            'winner_seed': w_seed,
            'loser_id': g.LTeamID,
            'loser_name': id_to_name.get(g.LTeamID, '?'),
            'loser_seed': l_seed,
        })

    # Build R64 ordered teams
    r64_teams = {}
    for ri, region in enumerate(bracket):
        region_sorted = sorted(region, key=lambda x: x['seed'])
        ordered = []
        for i in range(0, 16, 2):
            ordered.append(region_sorted[MATCHUP_ORDER[i]])
            ordered.append(region_sorted[MATCHUP_ORDER[i + 1]])
        r64_teams[ri] = ordered

    tree = {'R64_teams': r64_teams}

    # Map team_id to (region_idx, slot)
    tid_to_slot = {}
    for ri in range(4):
        for si, team in enumerate(r64_teams[ri]):
            if team.get('team_id'):
                tid_to_slot[team['team_id']] = (ri, si)

    results_by_day = sorted(actual_results, key=lambda x: x['day'])
    round_sizes = [32, 16, 8, 4, 2, 1]
    round_keys = ['R64_winners', 'R32_winners', 'S16_winners', 'E8_winners', 'F4_winners', 'Championship_winner']

    idx = 0
    for rnd_idx, (key, size) in enumerate(zip(round_keys, round_sizes)):
        rnd_games = results_by_day[idx:idx + size]
        idx += size

        if key == 'R64_winners':
            tree[key] = {ri: [None]*8 for ri in range(4)}
            for g in rnd_games:
                wid = g['winner_id']
                if wid in tid_to_slot:
                    ri, si = tid_to_slot[wid]
                    tree[key][ri][si // 2] = {
                        'name': g['winner_name'], 'seed': g['winner_seed'], 'team_id': wid}
        elif key == 'R32_winners':
            tree[key] = {ri: [None]*4 for ri in range(4)}
            for g in rnd_games:
                wid = g['winner_id']
                if wid in tid_to_slot:
                    ri, si = tid_to_slot[wid]
                    tree[key][ri][si // 4] = {
                        'name': g['winner_name'], 'seed': g['winner_seed'], 'team_id': wid}
        elif key == 'S16_winners':
            tree[key] = {ri: [None]*2 for ri in range(4)}
            for g in rnd_games:
                wid = g['winner_id']
                if wid in tid_to_slot:
                    ri, si = tid_to_slot[wid]
                    tree[key][ri][si // 8] = {
                        'name': g['winner_name'], 'seed': g['winner_seed'], 'team_id': wid}
        elif key == 'E8_winners':
            tree[key] = {ri: [None] for ri in range(4)}
            for g in rnd_games:
                wid = g['winner_id']
                if wid in tid_to_slot:
                    ri, _ = tid_to_slot[wid]
                    tree[key][ri][0] = {
                        'name': g['winner_name'], 'seed': g['winner_seed'], 'team_id': wid}
        elif key == 'F4_winners':
            tree[key] = [None, None]
            for g in rnd_games:
                wid = g['winner_id']
                if wid in tid_to_slot:
                    ri, _ = tid_to_slot[wid]
                    semi_idx = 0 if ri in (0, 1) else 1
                    tree[key][semi_idx] = {
                        'name': g['winner_name'], 'seed': g['winner_seed'], 'team_id': wid}
        elif key == 'Championship_winner':
            if rnd_games:
                g = rnd_games[0]
                tree[key] = {
                    'name': g['winner_name'], 'seed': g['winner_seed'], 'team_id': g['winner_id']}

    return tree


def compute_accuracy(pred_tree, actual_tree):
    def same(a, b):
        if a is None or b is None:
            return False
        if 'team_id' in a and 'team_id' in b and a.get('team_id') and b.get('team_id'):
            return a['team_id'] == b['team_id']
        return a.get('name', '').lower() == b.get('name', '').lower()

    stats, total_c, total_g = {}, 0, 0
    for label, key in [('R64', 'R64_winners'), ('R32', 'R32_winners'),
                        ('S16', 'S16_winners'), ('E8', 'E8_winners')]:
        correct, count = 0, 0
        for ri in range(4):
            preds = pred_tree.get(key, {}).get(ri, [])
            actuals = actual_tree.get(key, {}).get(ri, [])
            for i, p in enumerate(preds):
                a = actuals[i] if i < len(actuals) else None
                count += 1
                if same(p, a):
                    correct += 1
        stats[label] = (correct, count)
        total_c += correct
        total_g += count

    f4_c = sum(1 for i, p in enumerate(pred_tree.get('F4_winners', []))
               if same(p, (actual_tree.get('F4_winners', []) + [None]*2)[i]))
    stats['F4'] = (f4_c, 2)
    total_c += f4_c
    total_g += 2

    ch_c = 1 if same(pred_tree.get('Championship_winner'),
                      actual_tree.get('Championship_winner')) else 0
    stats['Champ'] = (ch_c, 1)
    total_c += ch_c
    total_g += 1
    stats['Overall'] = (total_c, total_g)
    return stats


# ── Accuracy and prediction helpers ───────────────────────────────

class ChalkPredictor:
    """Always picks the higher (lower number) seed. Coin flip on ties."""
    def predict(self, team_a, team_b):
        if team_a['seed'] < team_b['seed']:
            return team_a, 0.5 + (team_b['seed'] - team_a['seed']) * 0.03
        elif team_b['seed'] < team_a['seed']:
            return team_b, 0.5 + (team_a['seed'] - team_b['seed']) * 0.03
        else:
            # Coin flip on equal seeds
            if np.random.random() < 0.5:
                return team_a, 0.50
            else:
                return team_b, 0.50


def get_test_accuracy():
    """Compute bracket simulation accuracy on test years using Kaggle data.
    Returns (results, round_stats) where round_stats[year][model][round] = (correct, total)
    and champ_picks[year][model] = {'pred': team_name, 'actual': team_name, 'correct': bool}."""
    print("Computing test accuracy (2023-2025)...")
    seeds, tourney, teams, team_stats, id_to_name = load_kaggle_data()
    results = {}
    round_stats = {}  # year -> model -> round -> (correct, total)
    champ_picks = {}  # year -> model -> {'pred': name, 'actual': name, 'correct': bool}

    for year in TEST_YEARS:
        bracket, region_codes = build_bracket_from_seeds(seeds, year, id_to_name, tourney)
        actual_tree = build_actual_tree(bracket, tourney, year, seeds, id_to_name)
        actual_champ = actual_tree.get('Championship_winner', {})
        actual_champ_name = actual_champ.get('name', '?') if actual_champ else '?'

        round_stats[year] = {}
        champ_picks[year] = {}

        # Chalk baseline
        chalk_pred = simulate_bracket(bracket, ChalkPredictor())
        chalk_stats = compute_accuracy(chalk_pred, actual_tree)
        ov_c, ov_t = chalk_stats['Overall']
        pct = 100 * ov_c / ov_t
        chalk_name = 'Chalk (higher seed, coin flip on equal)'
        if chalk_name not in results:
            results[chalk_name] = {}
        results[chalk_name][year] = pct
        round_stats[year][chalk_name] = {}
        for rnd in ['S16', 'E8', 'F4', 'Champ']:
            round_stats[year][chalk_name][rnd] = chalk_stats[rnd]
        chalk_champ = chalk_pred.get('Championship_winner', {})
        chalk_champ_name = chalk_champ.get('name', '?') if chalk_champ else '?'
        champ_picks[year][chalk_name] = {
            'pred': chalk_champ_name, 'actual': actual_champ_name,
            'correct': chalk_champ_name.lower() == actual_champ_name.lower()
        }
        print(f"  {year} {chalk_name:<20s} {pct:.1f}%")

        # ML models
        for model_name, model_cfg in ALL_MODELS.items():
            predictor = create_predictor(model_cfg, team_stats, year)
            pred_tree = simulate_bracket(bracket, predictor)
            stats = compute_accuracy(pred_tree, actual_tree)
            ov_c, ov_t = stats['Overall']
            pct = 100 * ov_c / ov_t
            if model_name not in results:
                results[model_name] = {}
            results[model_name][year] = pct
            round_stats[year][model_name] = {}
            for rnd in ['S16', 'E8', 'F4', 'Champ']:
                round_stats[year][model_name][rnd] = stats[rnd]
            pred_champ = pred_tree.get('Championship_winner', {})
            pred_champ_name = pred_champ.get('name', '?') if pred_champ else '?'
            champ_picks[year][model_name] = {
                'pred': pred_champ_name, 'actual': actual_champ_name,
                'correct': pred_champ_name.lower() == actual_champ_name.lower()
            }
            print(f"  {year} {model_name:<20s} {pct:.1f}%")

    return results, round_stats, champ_picks


def get_2026_predictions_all():
    """Get prediction trees for ALL models on 2026 bracket."""
    print("Computing 2026 predictions for all models...")
    team_stats = pd.read_csv(PROCESSED_DIR / 'kaggle_team_stats.csv')
    teams = pd.read_csv(DATA_DIR / 'MTeams.csv')
    id_to_name = dict(zip(teams.TeamID, teams.TeamName))
    name_to_id = {v: k for k, v in id_to_name.items()}

    # Map team names to TeamIDs
    for region in BRACKET_2026:
        for team in region:
            tid = name_to_id.get(team['name'])
            team['team_id'] = tid
            if tid is None:
                print(f"  WARNING: No TeamID for {team['name']}")

    all_trees = {}
    for model_name, model_cfg in ALL_MODELS.items():
        predictor = create_predictor(model_cfg, team_stats, 2026)
        all_trees[model_name] = simulate_bracket(BRACKET_2026, predictor)
        champ = all_trees[model_name]['Championship_winner']
        print(f"  {model_name:<20s} -> ({champ['seed']}) {champ['name']}")

    return all_trees


# ── Dark-themed bracket drawing ───────────────────────────────────

def draw_bracket_dark(ax, tree, region_names, actual_tree=None):
    """Draw NCAA bracket prediction on dark background.
    If actual_tree is provided, picks are colored green (correct) or red (wrong),
    with the actual winner shown italic-muted under wrong picks."""
    FS       = 14
    FS_CONF  = 12
    FS_SM    = 10
    FS_HDR   = 15
    FS_REG   = 20

    RIGHT_CLR = GREEN
    WRONG_CLR = '#f85149'
    ACTUAL_CLR = '#9aa4b2'

    RH    = 16
    GAP   = 1.5
    COL_W = 1.7
    TW    = 1.45
    RX    = [i * COL_W for i in range(6)]

    total_h = 4 * RH + 3 * GAP
    ystarts = [3*(RH+GAP), 2*(RH+GAP), (RH+GAP), 0]

    ax.set_xlim(-1.0, RX[5] + 2.5)
    ax.set_ylim(-0.5, total_h + 3.0)
    ax.set_facecolor(CARD_BG)
    ax.axis('off')

    def lbl(t):
        return f"({t['seed']:>2}) {shorten(t['name'])}"

    def conf_color(prob):
        if prob >= 0.75: return GREEN
        elif prob >= 0.65: return '#d4820a'
        else: return MUTED

    def same(a, b):
        if a is None or b is None:
            return False
        if a.get('team_id') and b.get('team_id'):
            return a['team_id'] == b['team_id']
        return a.get('name', '').lower() == b.get('name', '').lower()

    def get_actual(rkey, idx, ri=None):
        if actual_tree is None:
            return None
        if rkey == 'Championship_winner':
            return actual_tree.get(rkey)
        if rkey == 'F4_winners':
            seq = actual_tree.get(rkey, [])
            return seq[idx] if idx < len(seq) else None
        seq = actual_tree.get(rkey, {}).get(ri, [])
        return seq[idx] if idx < len(seq) else None

    def is_correct(team, rkey, idx, ri=None):
        return same(team, get_actual(rkey, idx, ri))

    def put(x, y, team, c=R64_CLR, bold=False, fs_override=None, prob=None):
        ax.text(x, y, lbl(team), fontsize=fs_override or FS, color=c,
                fontweight='bold' if bold else 'normal',
                va='center', fontfamily='monospace', clip_on=False)
        if prob is not None:
            ax.text(x, y - 1.0, f"  {prob:.0%} conf",
                    fontsize=FS_CONF, color=conf_color(prob),
                    va='center', fontfamily='monospace', clip_on=False)

    def put_pick(x, y, team, rkey, idx, ri=None, bold=False, fs_override=None,
                 fallback_color=MUTED, show_actual_dy=-0.7, show_conf=True):
        """Draw a pick. If actual_tree set, color green/red, show conf below,
        and show actual winner under wrong picks."""
        prob = team.get('prob') if show_conf else None
        if actual_tree is None:
            put(x, y, team, c=fallback_color, bold=bold, fs_override=fs_override, prob=prob)
            return

        correct = is_correct(team, rkey, idx, ri)
        c = RIGHT_CLR if correct else WRONG_CLR
        fw = 'bold' if bold else 'normal'

        ax.text(x, y, lbl(team), fontsize=fs_override or FS, color=c,
                fontweight=fw, va='center', fontfamily='monospace', clip_on=False)

        # Sub-line: confidence (and actual winner if wrong)
        if correct:
            if prob is not None:
                ax.text(x, y + show_actual_dy, f"  Conf: {prob:.0%}",
                        fontsize=FS_SM, color=conf_color(prob),
                        va='center', fontfamily='monospace', clip_on=False)
        else:
            actual = get_actual(rkey, idx, ri)
            parts = []
            if prob is not None:
                parts.append(f"Conf: {prob:.0%}")
            if actual:
                parts.append(lbl(actual))
            if parts:
                ax.text(x, y + show_actual_dy, "  " + " ".join(parts),
                        fontsize=FS_SM, color=ACTUAL_CLR, fontstyle='italic',
                        va='center', fontfamily='monospace', clip_on=False)

    def bracket_line(x1, y1, y2, x2, yo):
        mx = (x1 + x2) / 2
        kw = dict(color=LINE_CLR, lw=0.8, solid_capstyle='round')
        ax.plot([x1, mx], [y1, y1], **kw)
        ax.plot([x1, mx], [y2, y2], **kw)
        ax.plot([mx, mx], [y1, y2], **kw)
        ax.plot([mx, x2], [yo, yo], **kw)

    for ri in range(4):
        yb = ystarts[ri]
        ax.text(-0.7, yb + RH/2, region_names[ri], fontsize=FS_REG,
                fontweight='bold', rotation=90, va='center', ha='center',
                color=GOLD)

        r64 = tree['R64_teams'][ri]
        for i, t in enumerate(r64):
            put(RX[0], yb + RH - 0.5 - i, t, c=MUTED, fs_override=12)

        w1 = tree.get('R64_winners', {}).get(ri, [])
        for i, t in enumerate(w1):
            y = yb + RH - 1 - i*2
            put_pick(RX[1], y, t, 'R64_winners', i, ri, show_actual_dy=-0.85)
            bracket_line(RX[0]+TW, yb+RH-0.5 - i*2,
                                   yb+RH-1.5 - i*2, RX[1]-0.05, y)

        w2 = tree.get('R32_winners', {}).get(ri, [])
        for i, t in enumerate(w2):
            y = yb + RH - 2 - i*4
            put_pick(RX[2], y, t, 'R32_winners', i, ri, show_actual_dy=-1.0)
            if len(w1) > i*2+1:
                bracket_line(RX[1]+TW, yb+RH-1 - (i*2)*2,
                                       yb+RH-1 - (i*2+1)*2, RX[2]-0.05, y)

        w3 = tree.get('S16_winners', {}).get(ri, [])
        for i, t in enumerate(w3):
            y = yb + RH - 4 - i*8
            put_pick(RX[3], y, t, 'S16_winners', i, ri, show_actual_dy=-1.1)
            if len(w2) > i*2+1:
                bracket_line(RX[2]+TW, yb+RH-2 - (i*2)*4,
                                       yb+RH-2 - (i*2+1)*4, RX[3]-0.05, y)

        w4 = tree.get('E8_winners', {}).get(ri, [])
        if w4:
            y = yb + RH/2
            put_pick(RX[4], y, w4[0], 'E8_winners', 0, ri, show_actual_dy=-1.2)
            if len(w3) >= 2:
                bracket_line(RX[3]+TW, yb+RH-4, yb+RH-12, RX[4]-0.05, y)

    f4 = tree.get('F4_winners', [])
    y_top = (ystarts[0] + RH/2 + ystarts[1] + RH/2) / 2
    y_bot = (ystarts[2] + RH/2 + ystarts[3] + RH/2) / 2

    for idx, yy, regions in [(0, y_top, [0,1]), (1, y_bot, [2,3])]:
        if idx < len(f4):
            put_pick(RX[5], yy, f4[idx], 'F4_winners', idx, show_actual_dy=-1.2)
            for pri in regions:
                if tree.get('E8_winners', {}).get(pri):
                    bracket_line(RX[4]+TW, ystarts[pri]+RH/2,
                                           ystarts[pri]+RH/2, RX[5]-0.05, yy)

    ch = tree.get('Championship_winner')
    if ch and len(f4) >= 2:
        yc = (y_top + y_bot) / 2
        if actual_tree is None:
            put(RX[5]+0.2, yc, ch, c=CHAMP_CLR, bold=True, fs_override=FS+4,
                prob=ch.get('prob'))
        else:
            put_pick(RX[5]+0.2, yc, ch, 'Championship_winner', 0,
                     bold=True, fs_override=FS+4, show_actual_dy=-1.6)
        bracket_line(RX[5]+TW, y_top, y_bot, RX[5]+0.15, yc)

    hdr_y = max(ystarts) + RH + 1.8
    for i, h in enumerate(['R64', 'R32', 'Sweet 16', 'Elite 8', 'Final 4', 'Champ']):
        ax.text(RX[i]+0.6, hdr_y, h, fontsize=FS_HDR, ha='center',
                color=MUTED, fontweight='bold')

    # ── Accuracy footer (bottom right) ────────────────────────────
    if actual_tree is not None:
        # ESPN Tournament Challenge scoring: 10/20/40/80/160/320 by round
        ESPN_PTS = {'R64_winners': 10, 'R32_winners': 20, 'S16_winners': 40,
                    'E8_winners': 80, 'F4_winners': 160, 'Championship_winner': 320}

        def score_tree(t):
            c = 0
            n = 0
            espn = 0
            for rkey in ('R64_winners', 'R32_winners', 'S16_winners', 'E8_winners'):
                for ri in range(4):
                    preds = t.get(rkey, {}).get(ri, [])
                    for i, tm in enumerate(preds):
                        n += 1
                        if is_correct(tm, rkey, i, ri):
                            c += 1
                            espn += ESPN_PTS[rkey]
            for i, tm in enumerate(t.get('F4_winners', [])):
                n += 1
                if is_correct(tm, 'F4_winners', i):
                    c += 1
                    espn += ESPN_PTS['F4_winners']
            ch = t.get('Championship_winner')
            if ch:
                n += 1
                if is_correct(ch, 'Championship_winner', 0):
                    c += 1
                    espn += ESPN_PTS['Championship_winner']
            return c, n, espn

        correct, total, espn_score = score_tree(tree)
        pct = (correct / total * 100) if total else 0.0

        # Chalk accuracy: pick the better (lower) seed each round
        def chalk_winner(a, b):
            return a if a['seed'] <= b['seed'] else b

        chalk = {'R64_winners': {}, 'R32_winners': {},
                 'S16_winners': {}, 'E8_winners': {}}
        for ri in range(4):
            r64 = tree['R64_teams'].get(ri, [])
            chalk['R64_winners'][ri] = [chalk_winner(r64[i*2], r64[i*2+1])
                                         for i in range(len(r64)//2)]
            w1 = chalk['R64_winners'][ri]
            chalk['R32_winners'][ri] = [chalk_winner(w1[i*2], w1[i*2+1])
                                         for i in range(len(w1)//2)]
            w2 = chalk['R32_winners'][ri]
            chalk['S16_winners'][ri] = [chalk_winner(w2[i*2], w2[i*2+1])
                                         for i in range(len(w2)//2)]
            w3 = chalk['S16_winners'][ri]
            chalk['E8_winners'][ri] = [chalk_winner(w3[0], w3[1])] if len(w3) >= 2 else []
        f4_chalk = []
        for pair in [(0, 1), (2, 3)]:
            la = chalk['E8_winners'].get(pair[0], [])
            lb = chalk['E8_winners'].get(pair[1], [])
            if la and lb:
                f4_chalk.append(chalk_winner(la[0], lb[0]))
        chalk['F4_winners'] = f4_chalk
        chalk['Championship_winner'] = (chalk_winner(f4_chalk[0], f4_chalk[1])
                                         if len(f4_chalk) >= 2 else None)

        chalk_correct, chalk_total, chalk_espn = score_tree(chalk)
        chalk_pct = (chalk_correct / chalk_total * 100) if chalk_total else 0.0

        ax.text(RX[5] + 2.4, 1.5,
                f"Bracket accuracy: {correct}/{total}  ({pct:.1f}%)",
                fontsize=13, color=GOLD, fontweight='bold',
                ha='right', va='bottom', fontfamily='monospace')
        ax.text(RX[5] + 2.4, 0.3,
                f"ESPN score:       {espn_score} / 1920",
                fontsize=11, color=WHITE,
                ha='right', va='bottom', fontfamily='monospace')
        ax.text(RX[5] + 2.4, -0.9,
                f"Chalk accuracy:   {chalk_correct}/{chalk_total}  ({chalk_pct:.1f}%, {chalk_espn} pts)",
                fontsize=11, color=MUTED,
                ha='right', va='bottom', fontfamily='monospace')


# ── Dashboard drawing ─────────────────────────────────────────────

def _draw_all(fig, gs, pred_tree, all_trees, test_acc, model_avgs, sorted_models, featured='XGBoost', model_stats=None, actual_tree=None):
    FEATURED = featured
    champ = pred_tree['Championship_winner']
    n_models = len(all_trees)

    # ── Title ─────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0]); ax.set_facecolor(BG); ax.axis('off')
    ax.text(0.5, 0.5, 'MARCH MADNESS 2026 ML PREDICTIONS', fontsize=28,
            fontweight='bold', color=GOLD, ha='center', va='center',
            transform=ax.transAxes, fontfamily='sans-serif')

    # ── Bracket (drawn directly, dark theme) ──────────────────────
    ax_bracket = fig.add_subplot(gs[1])
    for sp in ax_bracket.spines.values(): sp.set_color(BORDER); sp.set_linewidth(1.5)
    draw_bracket_dark(ax_bracket, pred_tree, REGION_NAMES, actual_tree=actual_tree)

    # ── Compute consensus picks across all models (with model names) ──
    MODEL_SHORT = MODEL_NAME_TO_ABBR

    def _gather_picks(get_team_fn):
        """Gather picks: returns {team_name: {'team': t, 'count': N, 'models': [...]}}"""
        counts = {}
        for mname, tree in all_trees.items():
            t = get_team_fn(tree)
            if t:
                key = t['name']
                if key not in counts:
                    counts[key] = {'team': t, 'count': 0, 'models': []}
                counts[key]['count'] += 1
                counts[key]['models'].append(MODEL_SHORT.get(mname, mname))
        return counts

    def _best_and_others(counts):
        """Return (best_team, best_count, best_models, others_list)."""
        if not counts:
            return None, 0, [], []
        best = max(counts.values(), key=lambda x: x['count'])
        others = []
        for k, v in sorted(counts.items(), key=lambda x: -x[1]['count']):
            if k != best['team']['name']:
                others.append((v['team'], v['models']))
        return best['team'], best['count'], best['models'], others

    s16_consensus = {}
    for ri in range(4):
        s16_consensus[ri] = []
        for si in range(2):
            counts = _gather_picks(
                lambda tree, _ri=ri, _si=si: (
                    tree.get('S16_winners', {}).get(_ri, [])[ _si]
                    if _si < len(tree.get('S16_winners', {}).get(_ri, []))
                    and tree.get('S16_winners', {}).get(_ri, [])[_si]
                    else None
                )
            )
            s16_consensus[ri].append(_best_and_others(counts))

    e8_consensus = {}
    for ri in range(4):
        counts = _gather_picks(
            lambda tree, _ri=ri: (
                tree.get('E8_winners', {}).get(_ri, [])[0]
                if tree.get('E8_winners', {}).get(_ri, [])
                and tree.get('E8_winners', {}).get(_ri, [])[0]
                else None
            )
        )
        e8_consensus[ri] = _best_and_others(counts)

    f4_consensus = {}
    for semi_idx, label in [(0, 'top'), (1, 'bot')]:
        counts = _gather_picks(
            lambda tree, _si=semi_idx: (
                tree.get('F4_winners', [])[_si]
                if _si < len(tree.get('F4_winners', []))
                and tree.get('F4_winners', [])[_si]
                else None
            )
        )
        f4_consensus[label] = _best_and_others(counts)

    champ_picks = _gather_picks(lambda tree: tree.get('Championship_winner'))
    champ_best_team, champ_best_count, champ_best_models, champ_others = _best_and_others(champ_picks)

    # ── Consensus Bracket ──────────────────────────────────────────
    ax = fig.add_subplot(gs[3])
    ax.set_facecolor(CARD_BG)
    for sp in ax.spines.values(): sp.set_color(BORDER); sp.set_linewidth(1.5)
    ax.set_xlim(0, 22); ax.set_ylim(-0.6, 11); ax.axis('off')

    ax.text(11, 10.7, 'MULTI-MODEL CONSENSUS BRACKET',
            fontsize=16, fontweight='bold', color=GOLD, ha='center', va='center',
            fontfamily='sans-serif')
    ax.text(11, 10.2, f'Most-picked winner across all {n_models} models per matchup',
            fontsize=12, color=MUTED, ha='center', va='center',
            fontfamily='sans-serif')

    s16_spread = 1.2
    region_cy = [8.5 - ri * 2.0 for ri in range(4)]
    reg_label_x = 0.5
    s16_x = 1.8
    s16_count_x = 6.2
    e8_x = 8.0
    e8_count_x = 12.2
    f4_x = 14.0
    f4_count_x = 18.0
    champ_x_pos = 19.5

    hdr_y = 9.5
    for hx, hl in [(s16_x, 'ELITE 8'), (e8_x, 'FINAL 4')]:
        ax.text(hx, hdr_y, hl, fontsize=14, color=MUTED,
                fontweight='bold', ha='left', va='center', fontfamily='sans-serif')

    kw = dict(color=LINE_CLR, lw=1.2, solid_capstyle='round')

    def team_label(team):
        return f'({team["seed"]}) {shorten(team["name"])}'

    def _draw_alts(ax, x, y, others, fontsize=11):
        """Draw alternative picks horizontally as colored chunks separated by '·'.
        Each (team, model) is rendered in the model's color."""
        rows = [(shorten(t["name"]), m) for t, models in others for m in models]
        if not rows:
            return
        # Use a TextArea/HPacker so each chunk gets its own color but stays inline.
        from matplotlib.offsetbox import TextArea, HPacker, AnnotationBbox
        children = []
        for i, (name, m) in enumerate(rows):
            clr = MODEL_COLORS.get(m, MUTED)
            children.append(TextArea(name, textprops=dict(
                color=clr, fontsize=fontsize, fontfamily='monospace')))
            if i < len(rows) - 1:
                children.append(TextArea(' · ', textprops=dict(
                    color=MUTED, fontsize=fontsize, fontfamily='monospace')))
        box = HPacker(children=children, align='center', pad=0, sep=0)
        ab = AnnotationBbox(box, (x, y), frameon=False, box_alignment=(0, 0.5),
                            pad=0)
        ax.add_artist(ab)

    for ri in range(4):
        cy = region_cy[ri]
        s16_y_top = cy + s16_spread / 2
        s16_y_bot = cy - s16_spread / 2
        e8_y = cy

        rname_short = {'East': 'E', 'South': 'S', 'West': 'W', 'Midwest': 'MW'}
        rname = rname_short.get(REGION_NAMES[ri], REGION_NAMES[ri][0])
        ax.text(reg_label_x, cy, rname, fontsize=13,
                fontweight='bold', rotation=90, va='center', ha='center',
                color=GOLD, fontfamily='sans-serif', alpha=0.9)

        for si, sy in enumerate([s16_y_top, s16_y_bot]):
            team, count, models, others = s16_consensus[ri][si]
            if team:
                ax.text(s16_x, sy, team_label(team), fontsize=16, color=MUTED,
                        va='center', fontfamily='monospace')
                ax.text(s16_count_x, sy, f'({count}/{n_models})', fontsize=10,
                        color=MUTED, va='center', fontfamily='sans-serif')
                if others:
                    _draw_alts(ax, s16_x, sy - 0.42, others, fontsize=11)

        s16_line_right = 7.2
        e8_line_left = 7.7
        mx = (s16_line_right + e8_line_left) / 2
        ax.plot([s16_line_right, mx], [s16_y_top, s16_y_top], **kw)
        ax.plot([s16_line_right, mx], [s16_y_bot, s16_y_bot], **kw)
        ax.plot([mx, mx], [s16_y_top, s16_y_bot], **kw)
        ax.plot([mx, e8_line_left], [e8_y, e8_y], **kw)

        team, count, models, others = e8_consensus[ri]
        ax.text(e8_x, e8_y, team_label(team), fontsize=16, color=MUTED,
                va='center', fontfamily='monospace')
        ax.text(e8_count_x, e8_y, f'({count}/{n_models})', fontsize=10,
                color=MUTED, va='center', fontfamily='sans-serif')
        if others:
            _draw_alts(ax, e8_x, e8_y - 0.46, others, fontsize=11)

    e8_line_right = 13.2
    f4_line_left = 13.7
    for pair_idx, (ri_a, ri_b) in enumerate([(0, 1), (2, 3)]):
        ya, yb = region_cy[ri_a], region_cy[ri_b]
        ym = (ya + yb) / 2
        mx = (e8_line_right + f4_line_left) / 2
        ax.plot([e8_line_right, mx], [ya, ya], **kw)
        ax.plot([e8_line_right, mx], [yb, yb], **kw)
        ax.plot([mx, mx], [ya, yb], **kw)
        ax.plot([mx, f4_line_left], [ym, ym], **kw)

        label_key = 'top' if pair_idx == 0 else 'bot'
        team, count, models, others = f4_consensus[label_key]
        if team:
            ax.text(f4_x, ym, team_label(team), fontsize=16, color=MUTED,
                    va='center', fontfamily='monospace')
            ax.text(f4_count_x, ym, f'({count}/{n_models})', fontsize=10,
                    color=MUTED, va='center', fontfamily='sans-serif')
            if others:
                _draw_alts(ax, f4_x, ym - 0.46, others, fontsize=11)

    f4_tops = [(region_cy[0] + region_cy[1]) / 2, (region_cy[2] + region_cy[3]) / 2]
    champ_y = (f4_tops[0] + f4_tops[1]) / 2
    f4_line_right2 = 19.0
    champ_line_left = 19.3
    mx2 = (f4_line_right2 + champ_line_left) / 2
    ax.plot([f4_line_right2, mx2], [f4_tops[0], f4_tops[0]], **kw)
    ax.plot([f4_line_right2, mx2], [f4_tops[1], f4_tops[1]], **kw)
    ax.plot([mx2, mx2], [f4_tops[0], f4_tops[1]], **kw)
    ax.plot([mx2, champ_line_left], [champ_y, champ_y], **kw)

    ch_team, ch_count = champ_best_team, champ_best_count
    # CHAMP header above the convergence point
    ax.text((champ_x_pos + f4_line_right2) / 2, hdr_y, 'CHAMP', fontsize=13,
            color=MUTED, fontweight='bold', ha='center', va='center',
            fontfamily='sans-serif')
    ax.text(champ_x_pos, champ_y + 0.55, team_label(ch_team), fontsize=18,
            color=CHAMP_CLR, va='center', fontweight='bold', fontfamily='monospace')
    ax.text(champ_x_pos, champ_y + 0.05,
            f'({ch_count}/{n_models})', fontsize=10, color=MUTED,
            va='center', fontfamily='sans-serif')
    if champ_others:
        _draw_alts(ax, champ_x_pos, champ_y - 0.55, champ_others, fontsize=11)

    # ── Middle row: Accuracy Table + Features ──────────────────────
    gs_bottom = gs[2].subgridspec(1, 2, wspace=0.05, width_ratios=[1, 1])

    # ── Model Performance Table (left) ────────────────────────────
    ax = fig.add_subplot(gs_bottom[0])
    ax.set_facecolor(CARD_BG)
    for sp in ax.spines.values(): sp.set_color(BORDER); sp.set_linewidth(1.5)
    ax.axis('off')
    ax.text(0.5, 0.96, 'MODEL PERFORMANCE ON UNSEEN TOURNAMENTS', fontsize=16,
            fontweight='bold', color=GOLD, ha='center', va='top',
            transform=ax.transAxes, fontfamily='sans-serif')
    ax.text(0.5, 0.89,
            'Trained on 2009-2022 (864 games). Tested on 2023-2025 (201 games).',
            fontsize=11, color=MUTED, ha='center', va='top',
            transform=ax.transAxes, fontstyle='italic', fontfamily='sans-serif')

    col_x = [0.05, 0.48, 0.70, 0.90]
    headers = ['Model', 'Test Acc', 'AUC', 'Overfit']
    y_start = 0.82; row_h = 0.085
    for j, (x, h) in enumerate(zip(col_x, headers)):
        ha = 'left' if j == 0 else 'center'
        ax.text(x, y_start, h, fontsize=13, color=MUTED, ha=ha, va='center',
                transform=ax.transAxes, fontweight='bold', fontfamily='sans-serif')
    ax.plot([0.03, 0.97], [y_start - 0.03, y_start - 0.03],
            color=BORDER, lw=1, transform=ax.transAxes, clip_on=False)

    # Sort by test accuracy with chalk inserted between models that beat it and those that don't
    ms = model_stats or {}
    chalk_key = 'Chalk (higher seed, coin flip on equal)'
    chalk_acc = ms.get(chalk_key, {}).get('test_acc', 0)
    ml_names = [n for n in ms if n != chalk_key]
    ml_names.sort(key=lambda n: ms[n]['test_acc'], reverse=True)
    above = [n for n in ml_names if ms[n]['test_acc'] > chalk_acc]
    below = [n for n in ml_names if ms[n]['test_acc'] <= chalk_acc]
    row_order = above + ([chalk_key] if chalk_key in ms else []) + below

    for i, model_name in enumerate(row_order):
        s = ms[model_name]
        y = y_start - 0.06 - i * row_h
        is_f = (model_name == FEATURED)
        is_chalk = (model_name == 'Chalk (higher seed, coin flip on equal)')
        abbr = MODEL_NAME_TO_ABBR.get(model_name)
        if is_chalk:
            nc = MUTED
        else:
            nc = MODEL_COLORS.get(abbr, WHITE)
        wt = 'bold' if is_f else 'normal'

        if is_f:
            rect = Rectangle((0.02, y - 0.028), 0.96, row_h - 0.005,
                              facecolor='#1a2233', edgecolor='none',
                              transform=ax.transAxes)
            ax.add_patch(rect)

        # Separator lines around chalk (cutoff band)
        if is_chalk:
            if i > 0:
                sep_y_top = y + row_h / 2 + 0.005
                ax.plot([0.03, 0.97], [sep_y_top, sep_y_top],
                        color=BORDER, lw=0.7, transform=ax.transAxes, clip_on=False,
                        linestyle='--', alpha=0.5)
            if i < len(row_order) - 1:
                sep_y_bot = y - row_h / 2 - 0.005
                ax.plot([0.03, 0.97], [sep_y_bot, sep_y_bot],
                        color=BORDER, lw=0.7, transform=ax.transAxes, clip_on=False,
                        linestyle='--', alpha=0.5)

        FULL_NAMES = {
            'RF': 'RF (Random Forest)',
            'MLP': 'MLP (Multi-Layer Perceptron)',
            'SVM': 'SVM (Support Vector Machine)',
        }
        if is_chalk:
            display_name = 'Chalk (higher seed, coin flip on tie)'
        else:
            display_name = FULL_NAMES.get(model_name, model_name)
        name_fs = 11 if is_chalk else 12.5
        ax.text(col_x[0], y, display_name, fontsize=name_fs, color=nc,
                ha='left', va='center', transform=ax.transAxes,
                fontweight=wt, fontfamily='sans-serif')

        # Test Accuracy — green if beats chalk, red if doesn't
        if is_chalk:
            acc_color = MUTED
        elif s['test_acc'] > chalk_acc:
            acc_color = GREEN
        else:
            acc_color = '#f85149'  # red
        ax.text(col_x[1], y, f'{s["test_acc"]:.1f}%', fontsize=13, color=acc_color,
                ha='center', va='center', transform=ax.transAxes,
                fontweight='bold', fontfamily='monospace')

        # AUC
        if s['auc'] is not None:
            auc_color = GOLD if is_f else WHITE
            ax.text(col_x[2], y, f'{s["auc"]:.3f}', fontsize=13, color=auc_color,
                    ha='center', va='center', transform=ax.transAxes,
                    fontfamily='monospace')
        else:
            ax.text(col_x[2], y, 'N/A', fontsize=13, color='#484f58',
                    ha='center', va='center', transform=ax.transAxes,
                    fontfamily='monospace')

        # Overfit (skip for chalk)
        if not is_chalk:
            ov = s['overfit']
            ov_color = GOLD if is_f else WHITE
            ax.text(col_x[3], y, f'{ov:+.1f}%', fontsize=13, color=ov_color,
                    ha='center', va='center', transform=ax.transAxes,
                    fontfamily='monospace')
        else:
            ax.text(col_x[3], y, '\u2014', fontsize=13, color='#484f58',
                    ha='center', va='center', transform=ax.transAxes,
                    fontfamily='monospace')

    # LightGBM note
    last_y = y_start - 0.06 - len(row_order) * row_h
    ax.text(0.05, last_y - 0.02, f'*Bracket above uses {FEATURED} model',
            fontsize=9.5, color=GOLD, ha='left', va='top',
            transform=ax.transAxes, fontstyle='italic', fontfamily='sans-serif')

    # ── Model Methodology (right) ───────────────────────────────────
    model_features = joblib.load(MODELS_DIR / 'features.pkl')

    ax = fig.add_subplot(gs_bottom[1])
    ax.set_facecolor(CARD_BG)
    for sp in ax.spines.values(): sp.set_color(BORDER); sp.set_linewidth(1.5)
    ax.axis('off')
    ax.text(0.5, 0.96, 'MODEL METHODOLOGY', fontsize=16,
            fontweight='bold', color=GOLD, ha='center', va='top',
            transform=ax.transAxes, fontfamily='sans-serif')

    ax.plot([0.05, 0.95], [0.91, 0.91],
            color=BORDER, lw=1, transform=ax.transAxes, clip_on=False)

    # ── Left column: Core Predictor ──
    ax.text(0.08, 0.86, 'CORE PREDICTOR', fontsize=13,
            fontweight='bold', color=GOLD, ha='left', va='top',
            transform=ax.transAxes, fontfamily='sans-serif')
    ax.text(0.08, 0.78, 'Logistic Markov Chain\nComposite Ranking (LMC)',
            fontsize=13, color=WHITE, ha='left', va='top',
            transform=ax.transAxes, fontfamily='sans-serif', fontweight='bold',
            linespacing=1.3)
    lmc_desc = (
        'The LMC ranking aggregates\n'
        '100+ individual ranking\n'
        'systems into a single\n'
        'composite via a logistic\n'
        'regression / Markov chain\n'
        'model. It is the strongest\n'
        'single predictor of\n'
        'tournament outcomes in\n'
        'our feature set.'
    )
    ax.text(0.08, 0.64, lmc_desc,
            fontsize=11.5, color=MUTED, ha='left', va='top',
            transform=ax.transAxes, fontfamily='sans-serif', linespacing=1.4)

    # Vertical separator
    ax.plot([0.50, 0.50], [0.88, 0.15],
            color=BORDER, lw=0.5, transform=ax.transAxes, clip_on=False)

    # ── Right column: Supporting Features ──
    FEAT_DISPLAY = {
        'diff_ortg': 'Offensive Rating',
        'diff_ortg_std': 'Offensive Rating Volatility',
        'diff_drtg': 'Defensive Rating',
        'diff_drtg_std': 'Defensive Rating Volatility',
        'diff_orb_pct': 'Offensive Rebound %',
        'diff_trb_pct': 'Total Rebound %',
        'diff_stl_rate': 'Steal Rate',
        'diff_tov_pct': 'Turnover %',
        'diff_rank_vol_POM': 'KenPom Rank Volatility',
        'diff_rank_trend_POM': 'KenPom Rank Trend',
        'diff_rank_trend_LMC': 'LMC Ranking Trend',
        'diff_rank_vol_LMC': 'LMC Ranking Volatility',
        'diff_win_pct': 'Win Percentage',
        'diff_ct_wins': 'Conf Tourney Wins',
    }

    ax.text(0.55, 0.86, f'SUPPORTING FEATURES ({len(model_features) - 1})', fontsize=13,
            fontweight='bold', color=GOLD, ha='left', va='top',
            transform=ax.transAxes, fontfamily='sans-serif')

    y_pos = 0.78
    for f in model_features:
        if f == 'diff_rank_LMC':
            continue
        desc = FEAT_DISPLAY.get(f, FEATURE_DESCRIPTIONS.get(f, f))
        ax.text(0.57, y_pos, f'-  {desc}', fontsize=12.5, color=MUTED,
                ha='left', va='top', transform=ax.transAxes,
                fontfamily='sans-serif')
        y_pos -= 0.085

    # ── Footer ────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[4]); ax.set_facecolor(BG); ax.axis('off')
    ax.text(0.5, 0.5,
            'Built with Python, scikit-learn, XGBoost, LightGBM  |  Data: Kaggle March Machine Learning Mania',
            fontsize=10, color='#484f58', ha='center', va='center',
            transform=ax.transAxes, fontfamily='sans-serif')


def get_model_stats():
    """Compute game-level model stats (accuracy, AUC, log loss, overfit) from kaggle_tourney.csv."""
    from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
    print("Computing game-level model stats...")
    df = pd.read_csv(PROCESSED_DIR / 'kaggle_tourney.csv')
    features = joblib.load(MODELS_DIR / 'features.pkl')
    scaler = joblib.load(MODELS_DIR / 'scaler.pkl')

    # Derived stability-adjusted rating features
    if 'diff_ortg_adj' in features:
        df['diff_ortg_std'] = df['diff_ortg_std'].fillna(0)
        df['diff_ortg_adj'] = df['diff_ortg'] - df['diff_ortg_std']
    if 'diff_drtg_adj' in features:
        df['diff_drtg_std'] = df['diff_drtg_std'].fillna(0)
        df['diff_drtg_adj'] = df['diff_drtg'] + df['diff_drtg_std']

    for f in features:
        df[f] = df[f].fillna(0)

    train_mask = (df['season'] >= 2009) & (df['season'] <= 2022) & (df['season'] != 2020)
    test_mask = df['season'].isin([2023, 2024, 2025])

    X_train = df.loc[train_mask, features].values
    X_test = df.loc[test_mask, features].values
    y_train = df.loc[train_mask, 'home_win'].values
    y_test = df.loc[test_mask, 'home_win'].values

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Chalk baseline
    seed_diff_test = df.loc[test_mask, 'seed_diff'].values
    chalk_preds = (seed_diff_test > 0).astype(int)
    ties = seed_diff_test == 0
    np.random.seed(42)
    chalk_preds[ties] = np.random.randint(0, 2, size=ties.sum())
    chalk_test_acc = accuracy_score(y_test, chalk_preds)

    seed_diff_train = df.loc[train_mask, 'seed_diff'].values
    chalk_train_preds = (seed_diff_train > 0).astype(int)
    ties_tr = seed_diff_train == 0
    chalk_train_preds[ties_tr] = np.random.randint(0, 2, size=ties_tr.sum())
    chalk_train_acc = accuracy_score(y_train, chalk_train_preds)

    model_configs = [
        ('LogReg', 'logistic_regression.pkl', True),
        ('XGBoost', 'xgboost_tuned.pkl', False),
        ('RF', 'random_forest_tuned.pkl', False),
        ('LightGBM', 'lightgbm_tuned.pkl', False),
        ('CatBoost', 'catboost_tuned.pkl', False),
        ('SVM', 'svm_tuned.pkl', True),
        ('MLP', 'mlp_tuned.pkl', True),
    ]

    stats = {}
    stats['Chalk (higher seed, coin flip on equal)'] = {
        'test_acc': chalk_test_acc * 100,
        'train_acc': chalk_train_acc * 100,
        'auc': None,
        'log_loss': None,
        'overfit': (chalk_train_acc - chalk_test_acc) * 100,
    }

    for name, pkl_file, scaled in model_configs:
        model = joblib.load(MODELS_DIR / pkl_file)
        Xtr = X_train_scaled if scaled else X_train
        Xte = X_test_scaled if scaled else X_test

        # Test-time averaging: predict both directions, average for symmetry
        prob_tr_fwd = model.predict_proba(Xtr)[:, 1]
        prob_tr_rev = model.predict_proba(-Xtr)[:, 1]
        y_prob_tr = (prob_tr_fwd + (1 - prob_tr_rev)) / 2

        prob_te_fwd = model.predict_proba(Xte)[:, 1]
        prob_te_rev = model.predict_proba(-Xte)[:, 1]
        y_prob_te = (prob_te_fwd + (1 - prob_te_rev)) / 2

        y_pred_tr = (y_prob_tr >= 0.5).astype(int)
        y_pred_te = (y_prob_te >= 0.5).astype(int)

        train_acc = accuracy_score(y_train, y_pred_tr)
        test_acc_val = accuracy_score(y_test, y_pred_te)
        auc = roc_auc_score(y_test, y_prob_te)
        ll = log_loss(y_test, y_prob_te)

        stats[name] = {
            'test_acc': test_acc_val * 100,
            'train_acc': train_acc * 100,
            'auc': auc,
            'log_loss': ll,
            'overfit': (train_acc - test_acc_val) * 100,
        }
        print(f"  {name:<12s} Test: {test_acc_val:.1%}  AUC: {auc:.3f}  LogLoss: {ll:.3f}  Overfit: {train_acc - test_acc_val:+.1%}")

    print(f"  {'Chalk':<12s} Test: {chalk_test_acc:.1%}")
    return stats


def main():
    all_trees = get_2026_predictions_all()
    test_acc, _, _ = get_test_accuracy()
    model_stats = get_model_stats()
    model_avgs = {m: np.mean(list(yrs.values())) for m, yrs in test_acc.items()}
    sorted_models = sorted(model_avgs.items(), key=lambda x: x[1], reverse=True)

    # Use LightGBM for the featured (first) bracket
    featured = 'LightGBM'
    pred_tree = all_trees['LightGBM']
    print(f"\nFeatured model: {featured}")

    # Build actuals tree for highlighting correct/wrong picks
    actual_tree = build_actual_tree_2026(BRACKET_2026)

    individual_trees = all_trees

    fig = plt.figure(figsize=(18, 22), facecolor=BG)
    gs = GridSpec(5, 1, figure=fig,
                  height_ratios=[0.3, 7.2, 2.8, 2.8, 0.12],
                  hspace=0.008,
                  left=0.02, right=0.98, top=0.995, bottom=0.002)

    _draw_all(fig, gs, pred_tree, individual_trees, test_acc, model_avgs, sorted_models, featured, model_stats, actual_tree=actual_tree)

    out_pdf = DASH_DIR / 'ncaa_2026_dashboard.pdf'
    fig.savefig(out_pdf, format='pdf', facecolor=BG, bbox_inches='tight', dpi=200)
    print(f"\nPDF saved: {out_pdf}")

    out_png = DASH_DIR / 'ncaa_2026_dashboard.png'
    fig.savefig(out_png, format='png', facecolor=BG, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"PNG preview: {out_png}")


if __name__ == '__main__':
    main()
