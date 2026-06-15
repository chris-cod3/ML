"""Show Final Four and Championship predictions for all years."""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Load model
model = joblib.load('models/logistic_regression_23f.pkl')
scaler = joblib.load('models/scaler_23f.pkl')
features = joblib.load('models/features_23.pkl')

data_dir = Path('data/raw')

def normalize_team_name(name):
    mappings = {
        'UConn Huskies': 'Connecticut',
        'NC State Wolfpack': 'North Carolina State',
        'Miami Hurricanes': 'Miami (FL)',
        'Loyola Chicago Ramblers': 'Loyola (IL)',
        'Loyola-Chicago Ramblers': 'Loyola (IL)',
        'UCLA Bruins': 'UCLA',
    }
    if name in mappings:
        return mappings[name]
    suffixes = [' Huskies', ' Wolfpack', ' Tar Heels', ' Blue Devils', ' Crimson Tide',
                ' Boilermakers', ' Wildcats', ' Tigers', ' Bulldogs', ' Bears',
                ' Cyclones', ' Fighting Illini', ' Volunteers', ' Cougars',
                ' Gamecocks', ' Ducks', ' Longhorns', ' Red Raiders', ' Flyers',
                ' Spartans', ' Golden Eagles', ' Ramblers', ' Jayhawks', ' Badgers',
                ' Orange', ' Sooners', ' Owls', ' Hurricanes', ' Aztecs', ' Bruins',
                ' Gators']
    result = name
    for suffix in suffixes:
        if result.endswith(suffix):
            result = result[:-len(suffix)]
            break
    return result.strip()

def get_team_stats(sr, team_name):
    norm = normalize_team_name(team_name).lower().strip()
    match = sr[sr['school_norm'] == norm]
    if not match.empty:
        return match.iloc[0]
    # Try partial match
    match = sr[sr['school_norm'].str.contains(norm.split()[0], na=False)]
    if len(match) == 1:
        return match.iloc[0]
    return None

def count_tourney_games_wins(tourney_df):
    game_counts = {}
    win_counts = {}
    for _, game in tourney_df.iterrows():
        if game['round'] == 'F4':
            continue
        home = game['home_team_name']
        away = game['away_team_name']
        game_counts[home] = game_counts.get(home, 0) + 1
        game_counts[away] = game_counts.get(away, 0) + 1
        if game['home_score'] > game['away_score']:
            win_counts[home] = win_counts.get(home, 0) + 1
        else:
            win_counts[away] = win_counts.get(away, 0) + 1
    return game_counts, win_counts

def adjust_stats(sr, tourney_df):
    game_counts, win_counts = count_tourney_games_wins(tourney_df)
    game_counts_norm = {normalize_team_name(k).lower(): v for k, v in game_counts.items()}
    win_counts_norm = {normalize_team_name(k).lower(): v for k, v in win_counts.items()}

    for idx, row in sr.iterrows():
        school_norm = str(row['School']).lower().strip()
        t_games = game_counts_norm.get(school_norm, 0)
        t_wins = win_counts_norm.get(school_norm, 0)
        if t_games > 0:
            orig_games = row.get('Overall_G', 0)
            orig_wins = row.get('Overall_W', 0)
            adj_games = orig_games - t_games
            adj_wins = orig_wins - t_wins
            adj_win_pct = adj_wins / adj_games if adj_games > 0 else 0
            sr.at[idx, 'Overall_G'] = adj_games
            sr.at[idx, 'Overall_W'] = adj_wins
            sr.at[idx, 'Overall_W-L%'] = adj_win_pct
    return sr

feature_map = {
    'diff_srs': 'Overall_SRS', 'diff_pts_for': 'Points_Tm.', 'diff_trb': 'Totals_TRB',
    'diff_sos': 'Overall_SOS', 'diff_pts_against': 'Points_Opp.', 'diff_ast': 'Totals_AST',
    'diff_win_pct': 'Overall_W-L%', 'diff_three_made': 'Totals_3P',
    'diff_ortg': 'School Advanced_ORtg', 'diff_blk': 'Totals_BLK', 'diff_stl': 'Totals_STL',
    'diff_tov_pct': 'School Advanced_TOV%', 'diff_pace': 'School Advanced_Pace',
    'diff_ts_pct': 'School Advanced_TS%', 'diff_tov': 'Totals_TOV', 'diff_ft_pct': 'Totals_FT%',
    'diff_fg_pct': 'Totals_FG%', 'diff_efg_pct': 'School Advanced_eFG%',
    'diff_three_pct': 'Totals_3P%', 'diff_blk_pct': 'School Advanced_BLK%',
    'diff_ftr': 'School Advanced_FTr', 'diff_trb_pct': 'School Advanced_TRB%',
}

results = []

for year in [2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024]:
    tourney = pd.read_csv(data_dir / f'espn_tournament_{year}.csv')
    sr = pd.read_csv(data_dir / f'sportsref_combined_{year}.csv')
    sr = adjust_stats(sr, tourney)
    sr['school_norm'] = sr['School'].str.lower().str.strip()

    f4_final = tourney[tourney['round'].isin(['R4', 'FINAL'])]

    for _, game in f4_final.iterrows():
        home = game['home_team_name']
        away = game['away_team_name']
        home_seed = int(game['home_seed'])
        away_seed = int(game['away_seed'])
        round_name = 'Final Four' if game['round'] == 'R4' else 'Championship'

        actual_winner = home if game['home_score'] > game['away_score'] else away

        home_stats = get_team_stats(sr, home)
        away_stats = get_team_stats(sr, away)

        if home_stats is None or away_stats is None:
            continue

        # Compute features
        feat_vals = []
        for feat in features:
            if feat == 'seed_diff':
                feat_vals.append(away_seed - home_seed)
            elif feat in feature_map:
                col = feature_map[feat]
                h = pd.to_numeric(home_stats.get(col, 0), errors='coerce') or 0
                a = pd.to_numeric(away_stats.get(col, 0), errors='coerce') or 0
                feat_vals.append(h - a)
            else:
                feat_vals.append(0)

        X = np.array(feat_vals).reshape(1, -1)
        X_scaled = scaler.transform(X)
        prob_home = model.predict_proba(X_scaled)[0, 1]

        if prob_home >= 0.5:
            predicted = home
            confidence = prob_home
        else:
            predicted = away
            confidence = 1 - prob_home

        correct = normalize_team_name(predicted).lower() == normalize_team_name(actual_winner).lower()

        results.append({
            'year': year,
            'round': round_name,
            'home': home,
            'away': away,
            'home_seed': home_seed,
            'away_seed': away_seed,
            'predicted': predicted,
            'confidence': confidence,
            'actual': actual_winner,
            'correct': correct
        })

# Print results
print("FINAL FOUR & CHAMPIONSHIP PREDICTIONS (Leakage-Free)")
print("="*90)

current_year = None
for r in results:
    if r['year'] != current_year:
        current_year = r['year']
        print()
        print(f"{current_year}")
        print("-"*90)

    mark = "Y" if r['correct'] else "N"
    matchup = f"({r['home_seed']}) {r['home'][:22]:22s} vs ({r['away_seed']}) {r['away'][:22]}"
    print(f"  {r['round']:12s}: {matchup}")
    print(f"               Predicted: {r['predicted'][:28]:28s} ({r['confidence']*100:.0f}%) [{mark}]")
    print(f"               Actual:    {r['actual']}")

# Summary
f4_results = [r for r in results if r['round'] == 'Final Four']
champ_results = [r for r in results if r['round'] == 'Championship']

f4_correct = sum(1 for r in f4_results if r['correct'])
champ_correct = sum(1 for r in champ_results if r['correct'])

print()
print("="*90)
print("SUMMARY")
print("="*90)
print(f"Final Four Games:   {f4_correct}/{len(f4_results)} correct ({100*f4_correct/len(f4_results):.0f}%)")
print(f"Championship Games: {champ_correct}/{len(champ_results)} correct ({100*champ_correct/len(champ_results):.0f}%)")

# List champions
print()
print("Champion Predictions by Year:")
for r in champ_results:
    mark = "Y" if r['correct'] else "N"
    print(f"  {r['year']}: Predicted {r['predicted'][:22]:22s} Actual: {r['actual'][:22]} [{mark}]")
