"""
Generate bracket visualizations for all years and models.

Historical years (2003-2025): predictions vs actual results (green/red).
Future year (2026): prediction-only (dark blue).

Data source: Kaggle (MNCAATourneySeeds, MNCAATourneyDetailedResults, kaggle_team_stats).
No ESPN or SportsRef dependencies.

Output: results/visualizations/{year}/bracket_{year}_{model}.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import joblib
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'
VIZ_DIR = PROJECT_ROOT / 'results' / 'visualizations'

# ── Model configs ─────────────────────────────────────────────────
MODELS = {
    'LogReg': {
        'model': 'logistic_regression.pkl',
        'scaler': 'scaler.pkl',
        'features': 'features.pkl',
    },
    'XGBoost': {
        'model': 'xgboost_tuned.pkl',
        'scaler': 'scaler.pkl',
        'features': 'features.pkl',
    },
    'RandomForest': {
        'model': 'random_forest_tuned.pkl',
        'scaler': 'scaler.pkl',
        'features': 'features.pkl',
    },
    'LightGBM': {
        'model': 'lightgbm_tuned.pkl',
        'scaler': 'scaler.pkl',
        'features': 'features.pkl',
    },
    'CatBoost': {
        'model': 'catboost_tuned.pkl',
        'scaler': 'scaler.pkl',
        'features': 'features.pkl',
    },
    'SVM': {
        'model': 'svm_tuned.pkl',
        'scaler': 'scaler.pkl',
        'features': 'features.pkl',
    },
    'MLP': {
        'model': 'mlp_tuned.pkl',
        'scaler': 'scaler.pkl',
        'features': 'features.pkl',
    },
    'Ensemble': {
        'ensemble': [
            {'model': 'logistic_regression.pkl', 'scaler': 'scaler.pkl', 'features': 'features.pkl'},
            {'model': 'xgboost_tuned.pkl', 'scaler': 'scaler.pkl', 'features': 'features.pkl'},
            {'model': 'random_forest_tuned.pkl', 'scaler': 'scaler.pkl', 'features': 'features.pkl'},
            {'model': 'lightgbm_tuned.pkl', 'scaler': 'scaler.pkl', 'features': 'features.pkl'},
            {'model': 'catboost_tuned.pkl', 'scaler': 'scaler.pkl', 'features': 'features.pkl'},
            {'model': 'svm_tuned.pkl', 'scaler': 'scaler.pkl', 'features': 'features.pkl'},
            {'model': 'mlp_tuned.pkl', 'scaler': 'scaler.pkl', 'features': 'features.pkl'},
        ],
    },
}

HISTORICAL_YEARS = list(range(2003, 2020)) + list(range(2021, 2026))  # Skip 2020 (COVID)
PREDICTION_YEARS = [2026]

# Map Kaggle region codes (W/X/Y/Z) to human-readable region names per year
# These were determined from NCAA bracket announcements
REGION_CODE_NAMES = {
    2018: {'W': 'South', 'X': 'West', 'Y': 'East', 'Z': 'Midwest'},
    2019: {'W': 'East', 'X': 'West', 'Y': 'South', 'Z': 'Midwest'},
    2021: {'W': 'West', 'X': 'East', 'Y': 'South', 'Z': 'Midwest'},
    2022: {'W': 'West', 'X': 'East', 'Y': 'South', 'Z': 'Midwest'},
    2023: {'W': 'South', 'X': 'East', 'Y': 'Midwest', 'Z': 'West'},
    2024: {'W': 'South', 'X': 'East', 'Y': 'Midwest', 'Z': 'West'},
    2025: {'W': 'East', 'X': 'South', 'Y': 'Midwest', 'Z': 'West'},
}

# 2026 bracket (prediction only — manually entered from Selection Sunday)
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
REGION_NAMES_2026 = ['East', 'South', 'West', 'Midwest']


# ── Data Loading ──────────────────────────────────────────────────

def load_kaggle_data():
    """Load all Kaggle data needed for bracket generation."""
    seeds = pd.read_csv(DATA_DIR / 'MNCAATourneySeeds.csv')
    tourney = pd.read_csv(DATA_DIR / 'MNCAATourneyDetailedResults.csv')
    teams = pd.read_csv(DATA_DIR / 'MTeams.csv')
    team_stats = pd.read_csv(PROCESSED_DIR / 'kaggle_team_stats.csv')

    # Parse seed strings: 'W01a' -> region='W', seed_num=1
    seeds['Region'] = seeds.Seed.str[0]
    seeds['SeedNum'] = seeds.Seed.str[1:3].astype(int)
    seeds['PlayIn'] = seeds.Seed.str[3:]  # 'a', 'b', or ''

    # TeamID -> TeamName lookup
    id_to_name = dict(zip(teams.TeamID, teams.TeamName))

    return seeds, tourney, teams, team_stats, id_to_name


def get_play_in_winners(tourney_df, season):
    """Get play-in game winners (DayNum 134-135)."""
    playin = tourney_df[(tourney_df.Season == season) & (tourney_df.DayNum <= 135)]
    return set(playin.WTeamID.tolist())


def build_bracket_from_seeds(seeds_df, season, id_to_name, tourney_df):
    """Build 64-team bracket from seeds, resolving play-in games."""
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
            # Play-in team: only include if they won their play-in game
            if team_id in playin_winners:
                regions[rc][seed_num] = team_id
            elif seed_num not in regions[rc]:
                # Don't overwrite a winner already placed
                pass
        else:
            regions[rc][seed_num] = team_id

    # Build bracket as list of 4 regions, each with 16 teams
    bracket = []
    region_codes = sorted(regions.keys())  # W, X, Y, Z
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


def build_actual_results(tourney_df, season, seeds_df, id_to_name):
    """Build actual tournament results from Kaggle tourney data."""
    year_games = tourney_df[tourney_df.Season == season].copy()
    year_games = year_games[year_games.DayNum >= 136].sort_values('DayNum')

    year_seeds = seeds_df[seeds_df.Season == season].copy()
    tid_to_seed = {}
    for _, row in year_seeds.iterrows():
        tid_to_seed[row['TeamID']] = (row['Region'], row['SeedNum'])

    results = []
    for _, g in year_games.iterrows():
        w_region, w_seed = tid_to_seed.get(g.WTeamID, ('?', 0))
        l_region, l_seed = tid_to_seed.get(g.LTeamID, ('?', 0))
        results.append({
            'day': g.DayNum,
            'winner_id': g.WTeamID,
            'winner_name': id_to_name.get(g.WTeamID, '?'),
            'winner_seed': w_seed,
            'loser_id': g.LTeamID,
            'loser_name': id_to_name.get(g.LTeamID, '?'),
            'loser_seed': l_seed,
            'w_score': g.WScore,
            'l_score': g.LScore,
        })

    return results


# ── Feature computation (mirrors build_dataset.py) ───────────────

# Column mapping: feature name -> column in kaggle_team_stats.csv
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
    'diff_ct_winpct': 'CT_WinPct',
    'diff_ct_margin': 'CT_AvgMargin',
    'diff_ct_wins': 'CT_Wins',
    'diff_drtg_std': 'DRtg_std',
    'diff_rank_vol_POM': 'RankVol_POM',
    'diff_rank_vol_LMC': 'RankVol_LMC',
    'diff_rank_trend_POM': 'RankTrend_POM',
    'diff_rank_trend_LMC': 'RankTrend_LMC',
}


def compute_matchup_features(team_a_stats, team_b_stats, seed_a, seed_b, features):
    """Compute feature vector for a matchup (team_a as 'home').

    Convention (matches build_dataset.py):
      - Regular stats: home - away (positive = home is better)
      - Rankings: away_rank - home_rank (positive = home is better, since lower rank = better)
      - seed_diff: away_seed - home_seed (positive = home has better seed)
    """
    # Rankings need inverted sign (lower rank = better)
    RANK_FEATURES = {'diff_rank_POM', 'diff_rank_LMC'}

    feat_vals = []
    for feat in features:
        if feat == 'seed_diff':
            feat_vals.append(seed_b - seed_a)
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
    """Predict games using model + features directly from kaggle_team_stats."""

    def __init__(self, model_path, scaler_path, features_path, team_stats_df, season):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.features = joblib.load(features_path)
        model_type = type(self.model).__name__
        self.needs_scaling = model_type in ('LogisticRegression', 'SVC', 'MLPClassifier')

        # Index stats by TeamID for this season
        ss = team_stats_df[team_stats_df.Season == season]
        self.stats_by_id = {row.TeamID: row for _, row in ss.iterrows()}

    def _symmetric_prob(self, X):
        """Test-time averaging: predict both directions, average for symmetry."""
        prob_fwd = self.model.predict_proba(X)[0, 1]
        prob_rev = self.model.predict_proba(-X)[0, 1]
        return (prob_fwd + (1 - prob_rev)) / 2

    def predict(self, team_a, team_b):
        """Predict winner. Returns (winner_dict, probability)."""
        tid_a = team_a.get('team_id')
        tid_b = team_b.get('team_id')
        stats_a = self.stats_by_id.get(tid_a)
        stats_b = self.stats_by_id.get(tid_b)

        if stats_a is None or stats_b is None:
            # Fallback to seed-based
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
    """Average probabilities across multiple models."""

    def __init__(self, model_configs, team_stats_df, season):
        self.predictors = []
        for cfg in model_configs:
            p = DirectPredictor(cfg['model'], cfg['scaler'], cfg['features'],
                                team_stats_df, season)
            self.predictors.append(p)

    def predict(self, team_a, team_b):
        probs_a = []
        for p in self.predictors:
            tid_a = team_a.get('team_id')
            tid_b = team_b.get('team_id')
            stats_a = p.stats_by_id.get(tid_a)
            stats_b = p.stats_by_id.get(tid_b)

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
    """Create a predictor from model config dict."""
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
            team_stats_df, season
        )


# ── Bracket simulation ───────────────────────────────────────────

MATCHUP_ORDER = [0, 15, 7, 8, 4, 11, 3, 12, 5, 10, 2, 13, 6, 9, 1, 14]

def simulate_bracket(bracket, predictor):
    """Simulate full tournament. Returns prediction tree."""
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

    # R64
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

    # R32
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

    # S16
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

    # E8
    tree['E8_winners'] = {}
    for ri in range(4):
        prev = tree['S16_winners'][ri]
        winner, prob = predictor.predict(prev[0], prev[1])
        winner = dict(winner)
        winner['prob'] = prob
        tree['E8_winners'][ri] = [winner]

    # F4: region 0 vs 1, region 2 vs 3
    f4_winners = []
    for pair in [(0, 1), (2, 3)]:
        ta = tree['E8_winners'][pair[0]][0]
        tb = tree['E8_winners'][pair[1]][0]
        winner, prob = predictor.predict(ta, tb)
        winner = dict(winner)
        winner['prob'] = prob
        f4_winners.append(winner)
    tree['F4_winners'] = f4_winners

    # Championship
    winner, prob = predictor.predict(f4_winners[0], f4_winners[1])
    winner = dict(winner)
    winner['prob'] = prob
    tree['Championship_winner'] = winner

    return tree


# ── Build actual tree from tournament results ─────────────────────

def build_actual_tree_from_kaggle(bracket, actual_results, region_codes):
    """Build actual result tree from Kaggle tournament data."""
    # Build R64 ordered teams (same ordering as simulate_bracket)
    r64_teams = {}
    for ri, region in enumerate(bracket):
        region_sorted = sorted(region, key=lambda x: x['seed'])
        ordered = []
        for i in range(0, 16, 2):
            ordered.append(region_sorted[MATCHUP_ORDER[i]])
            ordered.append(region_sorted[MATCHUP_ORDER[i + 1]])
        r64_teams[ri] = ordered

    tree = {'R64_teams': r64_teams}

    # Map team_id to (region_idx, slot) for tracking through rounds
    tid_to_slot = {}
    for ri in range(4):
        for si, team in enumerate(r64_teams[ri]):
            if team.get('team_id'):
                tid_to_slot[team['team_id']] = (ri, si)

    # Sort results by day to process in order
    results_by_day = sorted(actual_results, key=lambda x: x['day'])

    # Group by approximate round based on DayNum
    days = sorted(set(r['day'] for r in results_by_day))
    # Standard structure: 2 days R64, 2 days R32, 2 days S16, 2 days E8, 1 day F4, 1 day Final
    # R64: first 32 games, R32: next 16, S16: next 8, E8: next 4, F4: next 2, Final: 1

    all_games = results_by_day
    round_sizes = [32, 16, 8, 4, 2, 1]
    round_names = ['R64', 'R32', 'S16', 'E8', 'F4', 'Championship']

    idx = 0
    for rnd_name, rnd_size in zip(round_names, round_sizes):
        rnd_games = all_games[idx:idx + rnd_size]
        idx += rnd_size

        if rnd_name == 'R64':
            tree['R64_winners'] = {ri: [None]*8 for ri in range(4)}
            for g in rnd_games:
                wid = g['winner_id']
                # Find which slot the winner was in
                if wid in tid_to_slot:
                    ri, si = tid_to_slot[wid]
                    slot_idx = si // 2
                    tree['R64_winners'][ri][slot_idx] = {
                        'name': g['winner_name'], 'seed': g['winner_seed'],
                        'team_id': g['winner_id']
                    }

        elif rnd_name == 'R32':
            tree['R32_winners'] = {ri: [None]*4 for ri in range(4)}
            for g in rnd_games:
                wid = g['winner_id']
                if wid in tid_to_slot:
                    ri, si = tid_to_slot[wid]
                    slot_idx = si // 4
                    tree['R32_winners'][ri][slot_idx] = {
                        'name': g['winner_name'], 'seed': g['winner_seed'],
                        'team_id': g['winner_id']
                    }

        elif rnd_name == 'S16':
            tree['S16_winners'] = {ri: [None]*2 for ri in range(4)}
            for g in rnd_games:
                wid = g['winner_id']
                if wid in tid_to_slot:
                    ri, si = tid_to_slot[wid]
                    slot_idx = si // 8
                    tree['S16_winners'][ri][slot_idx] = {
                        'name': g['winner_name'], 'seed': g['winner_seed'],
                        'team_id': g['winner_id']
                    }

        elif rnd_name == 'E8':
            tree['E8_winners'] = {ri: [None] for ri in range(4)}
            for g in rnd_games:
                wid = g['winner_id']
                if wid in tid_to_slot:
                    ri, _ = tid_to_slot[wid]
                    tree['E8_winners'][ri][0] = {
                        'name': g['winner_name'], 'seed': g['winner_seed'],
                        'team_id': g['winner_id']
                    }

        elif rnd_name == 'F4':
            tree['F4_winners'] = [None, None]
            for g in rnd_games:
                wid = g['winner_id']
                if wid in tid_to_slot:
                    ri, _ = tid_to_slot[wid]
                    # Regions 0,1 -> semi 0; Regions 2,3 -> semi 1
                    semi_idx = 0 if ri in (0, 1) else 1
                    tree['F4_winners'][semi_idx] = {
                        'name': g['winner_name'], 'seed': g['winner_seed'],
                        'team_id': g['winner_id']
                    }

        elif rnd_name == 'Championship':
            if rnd_games:
                g = rnd_games[0]
                tree['Championship_winner'] = {
                    'name': g['winner_name'], 'seed': g['winner_seed'],
                    'team_id': g['winner_id']
                }

    return tree


# ── Accuracy ──────────────────────────────────────────────────────

def compute_accuracy(pred_tree, actual_tree):
    def same(a, b):
        if a is None or b is None:
            return False
        # Compare by team_id if available, else by name
        if 'team_id' in a and 'team_id' in b and a['team_id'] and b['team_id']:
            return a['team_id'] == b['team_id']
        return a.get('name', '').lower() == b.get('name', '').lower()

    rounds = [
        ('R64', 'R64_winners', 32),
        ('R32', 'R32_winners', 16),
        ('S16', 'S16_winners', 8),
        ('E8',  'E8_winners',  4),
    ]

    stats = {}
    total_correct = 0
    total_games = 0

    for label, key, expected in rounds:
        correct = 0
        count = 0
        for ri in range(4):
            preds = pred_tree.get(key, {}).get(ri, [])
            actuals = actual_tree.get(key, {}).get(ri, [])
            for i, p in enumerate(preds):
                a = actuals[i] if i < len(actuals) else None
                count += 1
                if same(p, a):
                    correct += 1
        stats[label] = (correct, count)
        total_correct += correct
        total_games += count

    f4_correct = 0
    for i, p in enumerate(pred_tree.get('F4_winners', [])):
        actuals = actual_tree.get('F4_winners', [])
        a = actuals[i] if i < len(actuals) else None
        if same(p, a):
            f4_correct += 1
    stats['F4'] = (f4_correct, 2)
    total_correct += f4_correct
    total_games += 2

    ch_correct = 1 if same(pred_tree.get('Championship_winner'),
                           actual_tree.get('Championship_winner')) else 0
    stats['Champ'] = (ch_correct, 1)
    total_correct += ch_correct
    total_games += 1

    stats['Overall'] = (total_correct, total_games)
    return stats


# ── Display helpers ───────────────────────────────────────────────

SHORT_NAMES = {
    'Connecticut': 'UConn', 'North Carolina': 'UNC',
    'Michigan St': 'Mich St', 'Mississippi St': 'Miss St',
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
    'St Bonaventic': 'St Bona',
    'Ark Little Rock': 'Ark LR',
    'CS Bakersfield': 'CS Bakers.',
    'CS Fullerton': 'CS Fuller.',
    'E Washington': 'E Wash',
    'G Washington': 'G Wash',
    'N Kentucky': 'N Kentucky',
    'SIU Edwardsville': 'SIUE',
    'SE Missouri St': 'SE Missouri',
    'Appalachian St': 'App St',
    'F Dickinson': 'FDU',
    'MTSU': 'MTSU',
    'Col Charleston': 'Charleston',
    'Miami FL': 'Miami',
    'Morehead St': 'Morehead',
    'Norfolk St': 'Norfolk St',
    'Alabama St': 'Alabama St',
    'McNeese St': 'McNeese',
    'WI Green Bay': 'Green Bay',
    'WI Milwaukee': 'Milwaukee',
    'High Point': 'High Point',
    'Sacramento St': 'Sacramento',
    'Weber St': 'Weber St',
    'Wofford': 'Wofford',
}


def shorten(name):
    return SHORT_NAMES.get(name, name)


# ── Drawing ───────────────────────────────────────────────────────

def draw_bracket(ax, tree, title, region_names, compare_tree=None):
    FS       = 14
    FS_SM    = 11
    FS_HDR   = 14
    FS_REG   = 16
    FS_TITLE = 28
    PRED_CLR = '#1a237e'
    ACTUAL_CLR = '#1565c0'

    RH   = 16
    GAP  = 1.5
    COL_W = 1.7
    TW    = 1.45
    RX = [i * COL_W for i in range(6)]

    total_h = 4 * RH + 3 * GAP
    ystarts = [3*(RH+GAP), 2*(RH+GAP), (RH+GAP), 0]

    ax.set_xlim(-1.0, RX[5] + 2.5)
    ax.set_ylim(-3.5, total_h + 2.5)
    ax.axis('off')
    ax.set_title(title, fontsize=FS_TITLE, fontweight='bold', pad=16)

    def lbl(t):
        return f"({t['seed']:>2}) {shorten(t['name'])}"

    def same(a, b):
        if a is None or b is None:
            return False
        if 'team_id' in a and 'team_id' in b and a.get('team_id') and b.get('team_id'):
            return a['team_id'] == b['team_id']
        return a.get('name', '').lower() == b.get('name', '').lower()

    def get_actual(rkey, idx, ri=None):
        if compare_tree is None:
            return None
        if rkey == 'Championship_winner':
            return compare_tree.get(rkey)
        if rkey == 'F4_winners':
            cmp = compare_tree.get(rkey, [])
            return cmp[idx] if idx < len(cmp) else None
        cmp = compare_tree.get(rkey, {}).get(ri, [])
        return cmp[idx] if idx < len(cmp) else None

    def is_correct(team, rkey, idx, ri=None):
        actual = get_actual(rkey, idx, ri)
        return same(team, actual)

    def colour(team, rkey, idx, ri=None):
        if compare_tree is None:
            return PRED_CLR
        return '#006400' if is_correct(team, rkey, idx, ri) else '#cc0000'

    FS_CONF = 11

    def conf_color(prob):
        if prob >= 0.75:
            return '#1b8a1b'
        elif prob >= 0.65:
            return '#d4820a'
        else:
            return '#999999'

    def put(x, y, team, c='#111111', bold=False, fs_override=None, prob=None):
        ax.text(x, y, lbl(team), fontsize=fs_override or FS, color=c,
                fontweight='bold' if bold else 'normal',
                va='center', fontfamily='monospace', clip_on=False)
        if prob is not None:
            ax.text(x, y - 0.42, f"  {prob:.0%} conf",
                    fontsize=FS_CONF, color=conf_color(prob),
                    va='center', fontfamily='monospace', clip_on=False)

    def put_with_actual(x, y, team, rkey, idx, ri=None, bold=False, fs_override=None):
        c = colour(team, rkey, idx, ri)
        prob = team.get('prob')
        if compare_tree is not None and not is_correct(team, rkey, idx, ri):
            actual = get_actual(rkey, idx, ri)
            put(x, y, team, c=c, bold=bold, fs_override=fs_override, prob=None)
            if actual:
                ax.text(x, y - 0.50, f"  {lbl(actual)}",
                        fontsize=FS_SM, color=ACTUAL_CLR, fontstyle='italic',
                        va='center', fontfamily='monospace', clip_on=False)
                if prob is not None:
                    ax.text(x, y - 0.92, f"  {prob:.0%} conf",
                            fontsize=FS_CONF, color=conf_color(prob),
                            va='center', fontfamily='monospace', clip_on=False)
            elif prob is not None:
                ax.text(x, y - 0.50, f"  {prob:.0%} conf",
                        fontsize=FS_CONF, color=conf_color(prob),
                        va='center', fontfamily='monospace', clip_on=False)
        else:
            put(x, y, team, c=c, bold=bold, fs_override=fs_override, prob=prob)

    def bracket_line(x1, y1, y2, x2, yo):
        mx = (x1 + x2) / 2
        kw = dict(color='#888888', lw=1.0, solid_capstyle='round')
        ax.plot([x1, mx], [y1, y1], **kw)
        ax.plot([x1, mx], [y2, y2], **kw)
        ax.plot([mx, mx], [y1, y2], **kw)
        ax.plot([mx, x2], [yo, yo], **kw)

    # Draw four regions
    for ri in range(4):
        yb = ystarts[ri]
        ax.text(-0.7, yb + RH/2, region_names[ri], fontsize=FS_REG,
                fontweight='bold', rotation=90, va='center', ha='center',
                color='#37474f')

        r64 = tree['R64_teams'][ri]
        for i, t in enumerate(r64):
            put(RX[0], yb + RH - 0.5 - i, t)

        w1 = tree.get('R64_winners', {}).get(ri, [])
        for i, t in enumerate(w1):
            y = yb + RH - 1 - i*2
            put_with_actual(RX[1], y, t, 'R64_winners', i, ri)
            bracket_line(RX[0]+TW, yb+RH-0.5 - i*2,
                                   yb+RH-1.5 - i*2, RX[1]-0.05, y)

        w2 = tree.get('R32_winners', {}).get(ri, [])
        for i, t in enumerate(w2):
            y = yb + RH - 2 - i*4
            put_with_actual(RX[2], y, t, 'R32_winners', i, ri)
            if len(w1) > i*2+1:
                bracket_line(RX[1]+TW, yb+RH-1 - (i*2)*2,
                                       yb+RH-1 - (i*2+1)*2, RX[2]-0.05, y)

        w3 = tree.get('S16_winners', {}).get(ri, [])
        for i, t in enumerate(w3):
            y = yb + RH - 4 - i*8
            put_with_actual(RX[3], y, t, 'S16_winners', i, ri)
            if len(w2) > i*2+1:
                bracket_line(RX[2]+TW, yb+RH-2 - (i*2)*4,
                                       yb+RH-2 - (i*2+1)*4, RX[3]-0.05, y)

        w4 = tree.get('E8_winners', {}).get(ri, [])
        if w4:
            y = yb + RH/2
            put_with_actual(RX[4], y, w4[0], 'E8_winners', 0, ri, bold=True)
            if len(w3) >= 2:
                bracket_line(RX[3]+TW, yb+RH-4, yb+RH-12, RX[4]-0.05, y)

    # Final Four
    f4 = tree.get('F4_winners', [])
    y_top = (ystarts[0] + RH/2 + ystarts[1] + RH/2) / 2
    y_bot = (ystarts[2] + RH/2 + ystarts[3] + RH/2) / 2

    for idx, yy, regions in [(0, y_top, [0,1]), (1, y_bot, [2,3])]:
        if idx < len(f4):
            put_with_actual(RX[5], yy, f4[idx], 'F4_winners', idx, bold=True)
            for pri in regions:
                if tree.get('E8_winners', {}).get(pri):
                    bracket_line(RX[4]+TW, ystarts[pri]+RH/2,
                                           ystarts[pri]+RH/2, RX[5]-0.05, yy)

    # Champion
    ch = tree.get('Championship_winner')
    if ch and len(f4) >= 2:
        yc = (y_top + y_bot) / 2
        champ_label = 'CHAMPION' if compare_tree else 'PREDICTED CHAMPION'
        ax.text(RX[5]+1.2, yc+1.8, champ_label, fontsize=16,
                fontweight='bold', ha='center', va='bottom', color='#37474f')
        if compare_tree:
            put_with_actual(RX[5]+0.2, yc, ch, 'Championship_winner', 0,
                            bold=True, fs_override=FS+3)
        else:
            put(RX[5]+0.2, yc, ch, c='#b71c1c', bold=True, fs_override=FS+3,
                prob=ch.get('prob'))
        bracket_line(RX[5]+TW, y_top, y_bot, RX[5]+0.15, yc)

    # Round headers
    hdr_y = max(ystarts) + RH + 1.5
    for i, h in enumerate(['R64', 'R32', 'Sweet 16', 'Elite 8', 'Final 4', 'Champ']):
        ax.text(RX[i]+0.6, hdr_y, h, fontsize=FS_HDR, ha='center',
                color='#546e7a', fontweight='bold')

    # Footer
    if compare_tree is not None:
        stats = compute_accuracy(tree, compare_tree)
        ov_c, ov_t = stats['Overall']
        pct = 100 * ov_c / ov_t
        parts = []
        for rnd in ['R64', 'R32', 'S16', 'E8', 'F4', 'Champ']:
            c, t = stats[rnd]
            parts.append(f"{rnd}: {c}/{t}")
        breakdown = '  |  '.join(parts)
        ax.text(RX[0], -2.8, f"Overall: {ov_c}/{ov_t} ({pct:.1f}%)  —  {breakdown}",
                fontsize=14, fontfamily='monospace', fontweight='bold',
                color='#37474f', va='center', ha='left', clip_on=False)
    else:
        ax.text(RX[0], -2.8, 'PREDICTION (no results yet)',
                fontsize=18, fontfamily='monospace', fontweight='bold',
                color='#37474f', va='center', ha='left', clip_on=False)


# ── Generation functions ──────────────────────────────────────────

def get_region_names(year, region_codes):
    """Get human-readable region names for a year."""
    code_map = REGION_CODE_NAMES.get(year, {})
    if code_map:
        return [code_map.get(rc, rc) for rc in region_codes]
    # Default: just use the code letters
    return [f'Region {rc}' for rc in region_codes]


def generate_historical_bracket(year, model_name, model_cfg, out_dir,
                                 seeds, tourney, team_stats, id_to_name):
    """Generate bracket with predictions vs actual results."""
    bracket, region_codes = build_bracket_from_seeds(seeds, year, id_to_name, tourney)
    region_names = get_region_names(year, region_codes)
    actual_results = build_actual_results(tourney, year, seeds, id_to_name)

    predictor = create_predictor(model_cfg, team_stats, year)
    pred_tree = simulate_bracket(bracket, predictor)
    actual_tree = build_actual_tree_from_kaggle(bracket, actual_results, region_codes)
    stats = compute_accuracy(pred_tree, actual_tree)

    ov_c, ov_t = stats['Overall']
    pct = 100 * ov_c / ov_t
    champ_pred = pred_tree['Championship_winner']['name']
    champ_actual = actual_tree.get('Championship_winner', {}).get('name', '?')

    fig, ax = plt.subplots(1, 1, figsize=(20, 28))
    fig.patch.set_facecolor('white')
    title = f'{model_name} — {year}'
    draw_bracket(ax, pred_tree, title, region_names, compare_tree=actual_tree)

    legend_elements = [
        mpatches.Patch(facecolor='#e8f5e9', edgecolor='#006400', label='Correct Pick', linewidth=2),
        mpatches.Patch(facecolor='#ffebee', edgecolor='#cc0000', label='Wrong Pick', linewidth=2),
        mpatches.Patch(facecolor='#e3f2fd', edgecolor='#1565c0', label='Actual Winner', linewidth=2),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=16, frameon=True)

    safe_name = model_name.lower().replace(' ', '_')
    fname = f'bracket_{year}_{safe_name}.png'
    out_path = out_dir / fname
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    return {
        'model': model_name, 'year': year, 'accuracy': pct,
        'champion_pred': champ_pred, 'champion_actual': champ_actual,
        'stats': stats, 'path': str(out_path),
    }


def generate_prediction_bracket(year, model_name, model_cfg, bracket, region_names,
                                 out_dir, team_stats):
    """Generate prediction-only bracket (no actual results)."""
    predictor = create_predictor(model_cfg, team_stats, year)
    pred_tree = simulate_bracket(bracket, predictor)
    champ = pred_tree['Championship_winner']

    fig, ax = plt.subplots(1, 1, figsize=(20, 28))
    fig.patch.set_facecolor('white')
    title = f'{model_name} — {year} Prediction'
    draw_bracket(ax, pred_tree, title, region_names, compare_tree=None)

    safe_name = model_name.lower().replace(' ', '_')
    fname = f'bracket_{year}_{safe_name}.png'
    out_path = out_dir / fname
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    return {
        'model': model_name, 'year': year,
        'champion': champ['name'], 'champion_seed': champ['seed'],
        'path': str(out_path),
    }


# ── Main ──────────────────────────────────────────────────────────

def main():
    print("Loading Kaggle data...")
    seeds, tourney, teams, team_stats, id_to_name = load_kaggle_data()

    # For 2026 bracket, we need to map team names to TeamIDs
    name_to_id = {v: k for k, v in id_to_name.items()}
    for region in BRACKET_2026:
        for team in region:
            tid = name_to_id.get(team['name'])
            if tid is None:
                # Try fuzzy match
                for k, v in id_to_name.items():
                    if v.lower() == team['name'].lower():
                        tid = k
                        break
            team['team_id'] = tid
            if tid is None:
                print(f"  WARNING: No TeamID for {team['name']}")

    all_results = []

    # Historical years
    for year in HISTORICAL_YEARS:
        # Check if we have tourney data for this year
        year_games = tourney[(tourney.Season == year) & (tourney.DayNum >= 136)]
        if len(year_games) < 32:
            print(f"\n  Skipping {year} — only {len(year_games)} tournament games")
            continue

        out_dir = VIZ_DIR / str(year)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"  {year} (vs actual results)")
        print(f"{'='*60}")
        for model_name, model_cfg in MODELS.items():
            r = generate_historical_bracket(year, model_name, model_cfg, out_dir,
                                            seeds, tourney, team_stats, id_to_name)
            champ_match = 'Y' if r['champion_pred'].lower() == r['champion_actual'].lower() else ''
            print(f"  {model_name:<20s} {r['accuracy']:5.1f}%  "
                  f"Pred: {r['champion_pred']:<18s} Actual: {r['champion_actual']:<18s} {champ_match}")
            all_results.append(r)

    # Prediction years
    for year in PREDICTION_YEARS:
        out_dir = VIZ_DIR / str(year)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"  {year} (prediction only)")
        print(f"{'='*60}")
        for model_name, model_cfg in MODELS.items():
            r = generate_prediction_bracket(
                year, model_name, model_cfg,
                BRACKET_2026, REGION_NAMES_2026, out_dir, team_stats
            )
            print(f"  {model_name:<20s} Champion: ({r['champion_seed']}) {r['champion']}")
            all_results.append(r)

    # Summary tables
    historical = [r for r in all_results if 'accuracy' in r]
    years_with_data = sorted(set(r['year'] for r in historical))
    model_names = list(MODELS.keys())

    print(f"\n\n{'='*80}")
    print("ACCURACY SUMMARY BY YEAR AND MODEL")
    print(f"{'='*80}")

    print(f"{'Model':<20s}", end='')
    for year in years_with_data:
        print(f"  {year:>6}", end='')
    print(f"  {'Avg':>6s}")
    print('-' * 80)

    for mname in model_names:
        print(f"{mname:<20s}", end='')
        accs = []
        for year in years_with_data:
            match = [r for r in historical if r['model'] == mname and r['year'] == year]
            if match:
                acc = match[0]['accuracy']
                accs.append(acc)
                print(f"  {acc:5.1f}%", end='')
            else:
                print(f"  {'N/A':>6s}", end='')
        avg = np.mean(accs) if accs else 0
        print(f"  {avg:5.1f}%")

    print(f"{'='*80}")

    # 2026 predictions
    print(f"\n2026 PREDICTIONS:")
    print(f"{'-'*60}")
    pred_results = [r for r in all_results if r['year'] == 2026]
    for r in pred_results:
        print(f"  {r['model']:<20s} Champion: ({r['champion_seed']}) {r['champion']}")


if __name__ == '__main__':
    main()
