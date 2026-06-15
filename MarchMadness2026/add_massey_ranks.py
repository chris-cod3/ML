"""
Add Massey rankings to tournament_matchups.csv and pretourney stats.

Uses DayNum 133 (pre-tournament) rankings from systems with full coverage
of our training years (2014-2026 excl 2020).

Adds:
  - Composite rank (average across all 10 full-coverage systems)
  - Individual system ranks: PGH (Pugh), POM (KenPom), LMC (Logistic Markov Chain), MAS (Massey)
  - Corresponding diff_ columns in tournament_matchups.csv
  - Massey_*_Rank columns in pretourney CSVs
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
PROC_DIR = PROJECT_ROOT / 'data' / 'processed'

# ── 1. Load Massey data ──────────────────────────────────────────

massey = pd.read_csv(DATA_DIR / 'MMasseyOrdinals.csv')
teams_df = pd.read_csv(DATA_DIR / 'MTeams.csv')

# Pre-tournament rankings only
pre = massey[massey['RankingDayNum'] == 133].copy()

# Find systems with full coverage of our years
our_years = set(range(2014, 2027)) - {2020}
sys_years = pre.groupby('SystemName')['Season'].apply(set)
full_systems = [s for s, yrs in sys_years.items() if len(yrs & our_years) == len(our_years)]
print(f'Systems with full {len(our_years)}-year coverage: {full_systems}')

# Composite: average rank across all full-coverage systems
pre_good = pre[pre['SystemName'].isin(full_systems)]
composite = pre_good.groupby(['Season', 'TeamID'])['OrdinalRank'].mean().reset_index()
composite.rename(columns={'OrdinalRank': 'composite_rank'}, inplace=True)
print(f'Composite ranks computed for {len(composite)} team-seasons')

# Individual system ranks we use as features
INDIVIDUAL_SYSTEMS = ['PGH', 'POM', 'LMC', 'MAS']
individual_ranks = {}
for sys_name in INDIVIDUAL_SYSTEMS:
    sys_df = pre[pre['SystemName'] == sys_name][['Season', 'TeamID', 'OrdinalRank']].copy()
    sys_df.rename(columns={'OrdinalRank': f'{sys_name}_rank'}, inplace=True)
    individual_ranks[sys_name] = sys_df
    print(f'{sys_name} ranks: {len(sys_df)} team-seasons')

# ── 2. Build ESPN team name -> Massey TeamID mapping ─────────────

tn_to_id = dict(zip(teams_df['TeamName'], teams_df['TeamID']))

# Manual overrides for names that don't auto-match
MANUAL_MAP = {
    'App State Mountaineers': 'Appalachian St',
    'Cal State Bakersfield Roadrunners': 'CS Bakersfield',
    'Cal State Fullerton Titans': 'CS Fullerton',
    'Charleston Cougars': 'Col Charleston',
    "Gardner-Webb Runnin' Bulldogs": 'Gardner Webb',
    "Hawai'i Rainbow Warriors": 'Hawaii',
    'Little Rock Trojans': 'Ark Little Rock',
    'Loyola Chicago Ramblers': 'Loyola-Chicago',
    'McNeese Cowboys': 'McNeese St',
    'Miami Hurricanes': 'Miami FL',
    "Mount St. Mary's Mountaineers": "Mt St Mary's",
    'Ole Miss Rebels': 'Mississippi',
    "Saint Joseph's Hawks": "St Joseph's PA",
    "Saint Mary's Gaels": "St Mary's CA",
    "Saint Peter's Peacocks": "St Peter's",
    'Southern Jaguars': 'Southern Univ',
    "St. Bonaventure Bonnies": 'St Bonaventure',
    "St. John's Red Storm": "St John's",
    'Stephen F. Austin Lumberjacks': 'SF Austin',
    'UConn Huskies': 'Connecticut',
    'Fairleigh Dickinson Knights': 'F Dickinson',
    'Green Bay Phoenix': 'WI Green Bay',
    'Milwaukee Panthers': 'WI Milwaukee',
    'Omaha Mavericks': 'NE Omaha',
    'SIU Edwardsville Cougars': 'SIUE',
    'Saint Louis Billikens': 'St Louis',
    'UAlbany Great Danes': 'SUNY Albany',
    'East Tennessee State Buccaneers': 'ETSU',
    'Eastern Kentucky Colonels': 'E Kentucky',
    'Eastern Washington Eagles': 'E Washington',
    'George Washington Revolutionaries': 'G Washington',
    'Middle Tennessee Blue Raiders': 'MTSU',
    'Northern Kentucky Norse': 'N Kentucky',
    'Southeast Missouri State Redhawks': 'SE Missouri St',
    'Western Kentucky Hilltoppers': 'WKU',
    'Western Michigan Broncos': 'W Michigan',
}


def build_name_to_id(team_names):
    """Map ESPN-style team names to Massey TeamIDs."""
    name_to_id = {}
    for name in team_names:
        # Try manual map first
        if name in MANUAL_MAP:
            massey_name = MANUAL_MAP[name]
            if massey_name in tn_to_id:
                name_to_id[name] = tn_to_id[massey_name]
                continue
        # Auto-match: check if Massey TeamName is prefix of our name
        name_lower = name.lower()
        for _, row in teams_df.iterrows():
            if name_lower.startswith(row['TeamName'].lower()):
                name_to_id[name] = row['TeamID']
                break
    return name_to_id


# ── 3. Add to tournament_matchups.csv ────────────────────────────

matchups = pd.read_csv(PROC_DIR / 'tournament_matchups.csv')
all_teams = sorted(set(matchups['home_team'].unique()) | set(matchups['away_team'].unique()))
name_to_id = build_name_to_id(all_teams)

unmatched = [t for t in all_teams if t not in name_to_id]
print(f'Team name matching: {len(name_to_id)}/{len(all_teams)} matched')
if unmatched:
    print(f'  Unmatched: {unmatched}')

matchups['home_tid'] = matchups['home_team'].map(name_to_id)
matchups['away_tid'] = matchups['away_team'].map(name_to_id)

# Drop old rank columns if re-running
old_rank_cols = [c for c in matchups.columns if 'composite_rank' in c or 'rank_' in c.lower()
                 and c not in ('home_team', 'away_team')]
# Be more precise: drop columns that look like our rank outputs
for col in list(matchups.columns):
    if 'composite_rank' in col or col.startswith('diff_rank_') or col.startswith('home_rank_') or col.startswith('away_rank_'):
        matchups.drop(columns=[col], inplace=True)

# Merge composite ranks
matchups = matchups.merge(
    composite.rename(columns={'composite_rank': 'home_composite_rank'}),
    left_on=['season', 'home_tid'], right_on=['Season', 'TeamID'], how='left'
).drop(columns=['Season', 'TeamID'], errors='ignore')

matchups = matchups.merge(
    composite.rename(columns={'composite_rank': 'away_composite_rank'}),
    left_on=['season', 'away_tid'], right_on=['Season', 'TeamID'], how='left'
).drop(columns=['Season', 'TeamID'], errors='ignore')

matchups['diff_rank_composite'] = matchups['home_composite_rank'] - matchups['away_composite_rank']

# Merge individual system ranks
for sys_name, sys_df in individual_ranks.items():
    home_col = f'home_rank_{sys_name}'
    away_col = f'away_rank_{sys_name}'
    diff_col = f'diff_rank_{sys_name}'

    matchups = matchups.merge(
        sys_df.rename(columns={f'{sys_name}_rank': home_col}),
        left_on=['season', 'home_tid'], right_on=['Season', 'TeamID'], how='left'
    ).drop(columns=['Season', 'TeamID'], errors='ignore')

    matchups = matchups.merge(
        sys_df.rename(columns={f'{sys_name}_rank': away_col}),
        left_on=['season', 'away_tid'], right_on=['Season', 'TeamID'], how='left'
    ).drop(columns=['Season', 'TeamID'], errors='ignore')

    matchups[diff_col] = matchups[home_col] - matchups[away_col]
    missing_sys = matchups[diff_col].isna().sum()
    print(f'{sys_name} rank coverage: {len(matchups) - missing_sys}/{len(matchups)} games ({missing_sys} missing)')

missing = matchups['diff_rank_composite'].isna().sum()
print(f'Composite rank coverage: {len(matchups) - missing}/{len(matchups)} games ({missing} missing)')

# Drop temp columns, keep the rank columns
matchups.drop(columns=['home_tid', 'away_tid'], inplace=True)

matchups.to_csv(PROC_DIR / 'tournament_matchups.csv', index=False)
print(f'Updated {PROC_DIR / "tournament_matchups.csv"}')

# ── 4. Add to pretourney stats CSVs ─────────────────────────────

# We need SR school name -> Massey TeamID for the pretourney files
# SR names are like "Houston", "Duke", etc. (no mascots)
SR_TO_MASSEY = {
    'Connecticut': 'Connecticut',
    'Miami (FL)': 'Miami FL',
    'Saint Mary\'s': "St Mary's CA",
    'Saint Mary\'s (CA)': "St Mary's CA",
    "St. John's (NY)": "St John's",
    'Southern California': 'Southern California',  # auto-match should work
    'Louisiana State': 'Louisiana St',
    'Mississippi': 'Mississippi',
    'Texas Christian': 'TCU',
    'Brigham Young': 'BYU',
    'North Carolina State': 'NC State',
    'Virginia Commonwealth': 'VCU',
    'Southern Methodist': 'SMU',
    'Maryland-Baltimore County': 'UMBC',
    'Prairie View A&M': 'Prairie View',
    'Long Island University': 'Long Island',
    'California Baptist': 'Cal Baptist',
    'Queens (NC)': 'Queens NC',
    'East Tennessee State': 'ETSU',
    'Eastern Kentucky': 'E Kentucky',
    'Eastern Washington': 'E Washington',
    'George Washington': 'G Washington',
    'Middle Tennessee': 'MTSU',
    'Northern Kentucky': 'N Kentucky',
    'SIU Edwardsville': 'SIUE',
    'Saint Louis': 'St Louis',
    'Southeast Missouri State': 'SE Missouri St',
    'Western Kentucky': 'WKU',
    'Western Michigan': 'W Michigan',
    'Green Bay': 'WI Green Bay',
    'Milwaukee': 'WI Milwaukee',
    'Fairleigh Dickinson': 'F Dickinson',
    'Albany (NY)': 'SUNY Albany',
}


def sr_name_to_tid(sr_name):
    """Convert SR school name to Massey TeamID."""
    # Manual override
    if sr_name in SR_TO_MASSEY:
        massey_name = SR_TO_MASSEY[sr_name]
        if massey_name in tn_to_id:
            return tn_to_id[massey_name]
    # Direct match
    if sr_name in tn_to_id:
        return tn_to_id[sr_name]
    # Try lowercase prefix
    sr_lower = sr_name.lower()
    for _, row in teams_df.iterrows():
        if sr_lower.startswith(row['TeamName'].lower()) or row['TeamName'].lower().startswith(sr_lower):
            return row['TeamID']
    return None


for year in list(range(2014, 2027)):
    if year == 2020:
        continue
    stats_path = DATA_DIR / f'sportsref_pretourney_{year}.csv'
    if not stats_path.exists():
        stats_path = DATA_DIR / f'sportsref_combined_{year}.csv'
    if not stats_path.exists():
        print(f'  {year}: no stats file found, skipping')
        continue

    stats = pd.read_csv(stats_path)
    # Drop old Massey columns if re-running
    for col in [c for c in stats.columns if 'Massey' in c]:
        stats.drop(columns=[col], inplace=True)

    stats['_tid'] = stats['School'].apply(sr_name_to_tid)

    # Composite rank
    year_ranks = composite[composite['Season'] == year][['TeamID', 'composite_rank']]
    stats = stats.merge(year_ranks, left_on='_tid', right_on='TeamID', how='left')
    stats.rename(columns={'composite_rank': 'Massey_Composite_Rank'}, inplace=True)
    stats.drop(columns=['TeamID'], errors='ignore', inplace=True)

    # Individual system ranks
    for sys_name, sys_df in individual_ranks.items():
        yr_sys = sys_df[sys_df['Season'] == year][['TeamID', f'{sys_name}_rank']]
        stats = stats.merge(yr_sys, left_on='_tid', right_on='TeamID', how='left')
        stats.rename(columns={f'{sys_name}_rank': f'Massey_{sys_name}_Rank'}, inplace=True)
        stats.drop(columns=['TeamID'], errors='ignore', inplace=True)

    stats.drop(columns=['_tid'], errors='ignore', inplace=True)

    matched = stats['Massey_Composite_Rank'].notna().sum()
    print(f'  {year}: {matched}/{len(stats)} teams matched')
    stats.to_csv(stats_path, index=False)

print('\nDone! Massey composite ranks added to all files.')
