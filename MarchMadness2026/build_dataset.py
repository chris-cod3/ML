"""
Build tournament matchup dataset from Kaggle CSV files + Massey Ordinals.

Data sources:
  - MRegularSeasonDetailedResults.csv  (box scores 2003-2026, includes conf tourney)
  - MNCAATourneyDetailedResults.csv    (tourney results = labels)
  - MNCAATourneySeeds.csv              (seed numbers)
  - MMasseyOrdinals.csv                (ranking systems, DayNum 133)
  - MTeams.csv                         (TeamID lookup)

Output:
  - data/processed/kaggle_team_stats.csv   (per-team per-season aggregated stats)
  - data/processed/kaggle_tourney.csv      (tournament matchups with diff features)
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RAW = PROJECT_ROOT / 'data' / 'raw'
PROCESSED = PROJECT_ROOT / 'data' / 'processed'
PROCESSED.mkdir(parents=True, exist_ok=True)

# ── 1. Load raw data ─────────────────────────────────────────────

reg = pd.read_csv(RAW / 'MRegularSeasonDetailedResults.csv')
tourney = pd.read_csv(RAW / 'MNCAATourneyDetailedResults.csv')
seeds = pd.read_csv(RAW / 'MNCAATourneySeeds.csv')
massey = pd.read_csv(RAW / 'MMasseyOrdinals.csv')
teams = pd.read_csv(RAW / 'MTeams.csv')
conf_tourney = pd.read_csv(RAW / 'MConferenceTourneyGames.csv')

# Filter to 2003+ (detailed box scores start here)
reg = reg[reg.Season >= 2003].copy()
tourney = tourney[tourney.Season >= 2003].copy()
conf_tourney = conf_tourney[conf_tourney.Season >= 2003].copy()
seeds = seeds[seeds.Season >= 2003].copy()

print(f"Regular season games: {len(reg)} ({reg.Season.min()}-{reg.Season.max()})")
print(f"Tournament games: {len(tourney)} ({tourney.Season.min()}-{tourney.Season.max()})")
print(f"Seeds entries: {len(seeds)}")

# ── 2. Aggregate regular season stats per team per season ────────
# Each game has W/L perspective — we need to flip to team-neutral.
# A team's stats come from W columns when they won, L columns when they lost.
# Opponent stats come from the other side.

def build_team_season_stats(games_df):
    """
    From game-level W/L box scores, compute per-team per-season averages.
    Returns DataFrame indexed by (Season, TeamID).
    """
    rows = []

    # Winner perspective: team = W, opponent = L
    for _, g in games_df.iterrows():
        rows.append({
            'Season': g.Season, 'TeamID': g.WTeamID,
            'FGM': g.WFGM, 'FGA': g.WFGA,
            'FGM3': g.WFGM3, 'FGA3': g.WFGA3,
            'FTM': g.WFTM, 'FTA': g.WFTA,
            'OR': g.WOR, 'DR': g.WDR,
            'Ast': g.WAst, 'TO': g.WTO,
            'Stl': g.WStl, 'Blk': g.WBlk,
            'PF': g.WPF,
            'Pts': g.WScore,
            'OppFGA': g.LFGA, 'OppFGA3': g.LFGA3, 'OppFTA': g.LFTA,
            'OppOR': g.LOR, 'OppDR': g.LDR,
            'OppTO': g.LTO,
            'OppPts': g.LScore,
            'OppFGM': g.LFGM,
            'Win': 1,
        })

    # Loser perspective: team = L, opponent = W
    for _, g in games_df.iterrows():
        rows.append({
            'Season': g.Season, 'TeamID': g.LTeamID,
            'FGM': g.LFGM, 'FGA': g.LFGA,
            'FGM3': g.LFGM3, 'FGA3': g.LFGA3,
            'FTM': g.LFTM, 'FTA': g.LFTA,
            'OR': g.LOR, 'DR': g.LDR,
            'Ast': g.LAst, 'TO': g.LTO,
            'Stl': g.LStl, 'Blk': g.LBlk,
            'PF': g.LPF,
            'Pts': g.LScore,
            'OppFGA': g.WFGA, 'OppFGA3': g.WFGA3, 'OppFTA': g.WFTA,
            'OppOR': g.WOR, 'OppDR': g.WDR,
            'OppTO': g.WTO,
            'OppPts': g.WScore,
            'OppFGM': g.WFGM,
            'Win': 0,
        })

    df = pd.DataFrame(rows)

    # Aggregate: season averages per team
    agg = df.groupby(['Season', 'TeamID']).agg(
        Games=('Win', 'size'),
        Wins=('Win', 'sum'),
        FGM=('FGM', 'mean'),
        FGA=('FGA', 'mean'),
        FGM3=('FGM3', 'mean'),
        FGA3=('FGA3', 'mean'),
        FTM=('FTM', 'mean'),
        FTA=('FTA', 'mean'),
        OR=('OR', 'mean'),
        DR=('DR', 'mean'),
        Ast=('Ast', 'mean'),
        TO=('TO', 'mean'),
        Stl=('Stl', 'mean'),
        Blk=('Blk', 'mean'),
        PF=('PF', 'mean'),
        Pts=('Pts', 'mean'),
        OppFGA=('OppFGA', 'mean'),
        OppFGA3=('OppFGA3', 'mean'),
        OppFTA=('OppFTA', 'mean'),
        OppOR=('OppOR', 'mean'),
        OppDR=('OppDR', 'mean'),
        OppTO=('OppTO', 'mean'),
        OppPts=('OppPts', 'mean'),
        OppFGM=('OppFGM', 'mean'),
    ).reset_index()

    return agg


print("\nAggregating regular season stats...")
stats = build_team_season_stats(reg)
print(f"  Team-seasons: {len(stats)}")

# ── 3. Compute advanced stats ────────────────────────────────────

# Possessions estimate (Dean Oliver formula)
stats['Poss'] = stats['FGA'] - stats['OR'] + stats['TO'] + 0.475 * stats['FTA']
stats['OppPoss'] = stats['OppFGA'] - stats['OppOR'] + stats['OppTO'] + 0.475 * stats['OppFTA']

# Pace (avg possessions per game — already per-game since we averaged)
stats['Pace'] = (stats['Poss'] + stats['OppPoss']) / 2

# Offensive/Defensive rating (pts per 100 possessions)
stats['ORtg'] = np.where(stats['Poss'] > 0, stats['Pts'] / stats['Poss'] * 100, 0)
stats['DRtg'] = np.where(stats['OppPoss'] > 0, stats['OppPts'] / stats['OppPoss'] * 100, 0)

# Four Factors (Dean Oliver)
stats['EFG_pct'] = np.where(stats['FGA'] > 0, (stats['FGM'] + 0.5 * stats['FGM3']) / stats['FGA'], 0)
stats['TOV_pct'] = np.where(stats['Poss'] > 0, stats['TO'] / stats['Poss'], 0)
stats['FT_pct'] = np.where(stats['FTA'] > 0, stats['FTM'] / stats['FTA'], 0)
stats['ThreeP_rate'] = np.where(stats['FGA'] > 0, stats['FGA3'] / stats['FGA'], 0)

# Rebounding (true ORB% uses opponent's defensive rebounds)
stats['ORB_pct'] = np.where(
    (stats['OR'] + stats['OppDR']) > 0,
    stats['OR'] / (stats['OR'] + stats['OppDR']),
    0
)
stats['TRB_pct'] = np.where(
    (stats['OR'] + stats['DR'] + stats['OppOR'] + stats['OppDR']) > 0,
    (stats['OR'] + stats['DR']) / (stats['OR'] + stats['DR'] + stats['OppOR'] + stats['OppDR']),
    0
)

# Assist rate (assists per made field goal)
stats['Ast_rate'] = np.where(stats['FGM'] > 0, stats['Ast'] / stats['FGM'], 0)

# Steal rate (steals per opponent possession)
stats['Stl_rate'] = np.where(stats['OppPoss'] > 0, stats['Stl'] / stats['OppPoss'], 0)

# Block rate (blocks per opponent 2-point attempts)
opp_2pa = stats['OppFGA'] - stats['OppFGA3']
stats['Blk_rate'] = np.where(opp_2pa > 0, stats['Blk'] / opp_2pa, 0)

# Win percentage
stats['WinPct'] = stats['Wins'] / stats['Games']

# Points against (opponent PPG)
stats['PtsAgainst'] = stats['OppPts']

# Net Rating (ORtg - DRtg, points per 100 poss differential)
stats['NetRtg'] = stats['ORtg'] - stats['DRtg']

# Average scoring margin (points per game)
stats['AvgMargin'] = stats['Pts'] - stats['OppPts']

# Strength of Schedule (avg opponent win rate)
print("  Computing Strength of Schedule...")
game_rows = []
for _, g in reg.iterrows():
    game_rows.append({'Season': g.Season, 'TeamID': g.WTeamID, 'OppID': g.LTeamID})
    game_rows.append({'Season': g.Season, 'TeamID': g.LTeamID, 'OppID': g.WTeamID})
games_df = pd.DataFrame(game_rows)
# Get each team's win pct
team_wp = stats[['Season', 'TeamID', 'WinPct']].copy()
# Join opponent win pct to each game
games_df = games_df.merge(team_wp.rename(columns={'TeamID': 'OppID', 'WinPct': 'OppWinPct'}),
                          on=['Season', 'OppID'], how='left')
# Average opponent win pct per team-season = SOS
sos = games_df.groupby(['Season', 'TeamID'])['OppWinPct'].mean().reset_index()
sos = sos.rename(columns={'OppWinPct': 'SOS'})
stats = stats.merge(sos, on=['Season', 'TeamID'], how='left')
stats['SOS'] = stats['SOS'].fillna(0.5)
print(f"  SOS computed: {stats['SOS'].notna().sum()}/{len(stats)} matched")

print(f"  Advanced stats computed.")

# ── 3b. Variance / Consistency metrics (game-level std devs) ────

print("  Computing variance metrics...")

def build_game_level_stats(games_df):
    """Build per-game stats for each team (both wins and losses)."""
    rows = []
    for _, g in games_df.iterrows():
        # Winner perspective
        w_fga = g.WFGA if g.WFGA > 0 else 1
        w_poss = w_fga - g.WOR + g.WTO + 0.475 * g.WFTA
        w_poss = max(w_poss, 1)
        l_fga = g.LFGA if g.LFGA > 0 else 1
        l_poss = l_fga - g.LOR + g.LTO + 0.475 * g.LFTA
        l_poss = max(l_poss, 1)
        rows.append({
            'Season': g.Season, 'TeamID': g.WTeamID, 'DayNum': g.DayNum,
            'Pts': g.WScore, 'OppPts': g.LScore,
            'ORtg': g.WScore / w_poss * 100,
            'DRtg': g.LScore / l_poss * 100,
            'Margin': g.WScore - g.LScore,
            'Win': 1,
        })
        rows.append({
            'Season': g.Season, 'TeamID': g.LTeamID, 'DayNum': g.DayNum,
            'Pts': g.LScore, 'OppPts': g.WScore,
            'ORtg': g.LScore / l_poss * 100,
            'DRtg': g.WScore / w_poss * 100,
            'Margin': g.LScore - g.WScore,
            'Win': 0,
        })
    return pd.DataFrame(rows)

game_stats = build_game_level_stats(reg)

# Variance metrics: std dev across entire season
variance_agg = game_stats.groupby(['Season', 'TeamID']).agg(
    Score_std=('Pts', 'std'),
    ORtg_std=('ORtg', 'std'),
    DRtg_std=('DRtg', 'std'),
    Margin_std=('Margin', 'std'),
).reset_index()

stats = stats.merge(variance_agg, on=['Season', 'TeamID'], how='left')

# ── 3c. Late-season momentum (last 10 games) ───────────────────

print("  Computing momentum metrics...")

def last_n_stats(game_df, n=10):
    """Compute stats over last N games per team-season."""
    game_df = game_df.sort_values(['Season', 'TeamID', 'DayNum'])
    rows = []
    for (season, team_id), grp in game_df.groupby(['Season', 'TeamID']):
        last_n = grp.tail(n)
        rows.append({
            'Season': season,
            'TeamID': team_id,
            'Last10_WinPct': last_n['Win'].mean(),
            'Last10_ORtg': last_n['ORtg'].mean(),
        })
    return pd.DataFrame(rows)

momentum = last_n_stats(game_stats, n=10)
stats = stats.merge(momentum, on=['Season', 'TeamID'], how='left')

# ── 3d. Conference tournament performance ───────────────────────

print("  Computing conference tournament performance...")

# Get scores for conf tourney games by joining with detailed results
# conf_tourney has (Season, DayNum, WTeamID, LTeamID)
# reg has the detailed box scores including those same games
conf_games = conf_tourney.merge(
    reg[['Season', 'DayNum', 'WTeamID', 'LTeamID', 'WScore', 'LScore']],
    on=['Season', 'DayNum', 'WTeamID', 'LTeamID'],
    how='inner'
)

# Build per-team conf tourney stats
ct_rows = []
for _, g in conf_games.iterrows():
    ct_rows.append({
        'Season': g.Season, 'TeamID': g.WTeamID,
        'CT_Win': 1, 'CT_Margin': g.WScore - g.LScore,
    })
    ct_rows.append({
        'Season': g.Season, 'TeamID': g.LTeamID,
        'CT_Win': 0, 'CT_Margin': g.LScore - g.WScore,
    })

ct_df = pd.DataFrame(ct_rows)
ct_agg = ct_df.groupby(['Season', 'TeamID']).agg(
    CT_Wins=('CT_Win', 'sum'),
    CT_Games=('CT_Win', 'size'),
    CT_AvgMargin=('CT_Margin', 'mean'),
).reset_index()
ct_agg['CT_WinPct'] = ct_agg['CT_Wins'] / ct_agg['CT_Games']

stats = stats.merge(ct_agg[['Season', 'TeamID', 'CT_Wins', 'CT_WinPct', 'CT_AvgMargin']],
                    on=['Season', 'TeamID'], how='left')

# Fill NaN for teams that didn't play in conf tourney (unlikely for tourney teams but safe)
stats['CT_Wins'] = stats['CT_Wins'].fillna(0)
stats['CT_WinPct'] = stats['CT_WinPct'].fillna(0)
stats['CT_AvgMargin'] = stats['CT_AvgMargin'].fillna(0)

print(f"  Conf tourney games matched: {len(conf_games)}")

# ── 4. Add Massey rankings (DayNum 133) ──────────────────────────

# POM only — other systems are >0.93 correlated with POM
RANKING_SYSTEMS = ['POM', 'LMC']

massey_pre = massey[massey.RankingDayNum == 133].copy()

for sys_name in RANKING_SYSTEMS:
    sys_data = massey_pre[massey_pre.SystemName == sys_name][['Season', 'TeamID', 'OrdinalRank']]
    sys_data = sys_data.rename(columns={'OrdinalRank': f'Rank_{sys_name}'})
    stats = stats.merge(sys_data, on=['Season', 'TeamID'], how='left')
    coverage = stats[f'Rank_{sys_name}'].notna().sum()
    print(f"  {sys_name}: {coverage}/{len(stats)} team-seasons matched")

# ── 4b. Ranking volatility & trend (time series, day <= 133) ────
print("\nComputing ranking volatility and trend...")
massey_ts = massey[massey.RankingDayNum <= 133].copy()

for sys_name in RANKING_SYSTEMS:
    sys_ts = massey_ts[massey_ts.SystemName == sys_name].copy()

    # Volatility: std dev of ranking over the season
    vol = sys_ts.groupby(['Season', 'TeamID'])['OrdinalRank'].std().reset_index()
    vol = vol.rename(columns={'OrdinalRank': f'RankVol_{sys_name}'})

    # Trend: slope of ranking over time (negative slope = improving)
    def rank_slope(group):
        if len(group) < 3:
            return np.nan
        x = group['RankingDayNum'].values.astype(float)
        y = group['OrdinalRank'].values.astype(float)
        # Simple linear regression slope
        slope = np.polyfit(x, y, 1)[0]
        return slope

    trend = sys_ts.groupby(['Season', 'TeamID']).apply(rank_slope, include_groups=False).reset_index()
    trend.columns = ['Season', 'TeamID', f'RankTrend_{sys_name}']

    stats = stats.merge(vol, on=['Season', 'TeamID'], how='left')
    stats = stats.merge(trend, on=['Season', 'TeamID'], how='left')

    vol_cov = stats[f'RankVol_{sys_name}'].notna().sum()
    trend_cov = stats[f'RankTrend_{sys_name}'].notna().sum()
    print(f"  {sys_name} volatility: {vol_cov}/{len(stats)}, trend: {trend_cov}/{len(stats)}")

# Fill NaN for missing ranking time series
for sys_name in RANKING_SYSTEMS:
    stats[f'RankVol_{sys_name}'] = stats[f'RankVol_{sys_name}'].fillna(0)
    stats[f'RankTrend_{sys_name}'] = stats[f'RankTrend_{sys_name}'].fillna(0)

# ── 5. Parse seeds ───────────────────────────────────────────────

seeds['SeedNum'] = seeds['Seed'].str[1:3].astype(int)
seeds['Region'] = seeds['Seed'].str[0]
seed_lookup = seeds[['Season', 'TeamID', 'SeedNum', 'Region']].copy()

print(f"\nSeeds parsed: {len(seed_lookup)} entries")

# ── 6. Build tournament matchups ─────────────────────────────────

print("\nBuilding tournament matchups...")

matchup_rows = []
for _, g in tourney.iterrows():
    season = g.Season
    w_id = g.WTeamID
    l_id = g.LTeamID

    # Get seeds
    w_seed_row = seed_lookup[(seed_lookup.Season == season) & (seed_lookup.TeamID == w_id)]
    l_seed_row = seed_lookup[(seed_lookup.Season == season) & (seed_lookup.TeamID == l_id)]

    if len(w_seed_row) == 0 or len(l_seed_row) == 0:
        continue  # skip if seeds not found

    w_seed = w_seed_row.iloc[0].SeedNum
    l_seed = l_seed_row.iloc[0].SeedNum

    # Get regular season stats
    w_stats = stats[(stats.Season == season) & (stats.TeamID == w_id)]
    l_stats = stats[(stats.Season == season) & (stats.TeamID == l_id)]

    if len(w_stats) == 0 or len(l_stats) == 0:
        continue

    w_stats = w_stats.iloc[0]
    l_stats = l_stats.iloc[0]

    # Deterministic random assignment: hash(season, team_ids) for reproducibility
    # This gives ~50/50 split without run-to-run variance
    game_hash = hash((season, min(w_id, l_id), max(w_id, l_id)))
    if game_hash % 2 == 0:
        home_id, away_id = w_id, l_id
        home_seed, away_seed = w_seed, l_seed
        home_stats, away_stats = w_stats, l_stats
        home_win = 1
    else:
        home_id, away_id = l_id, w_id
        home_seed, away_seed = l_seed, w_seed
        home_stats, away_stats = l_stats, w_stats
        home_win = 0

    row = {
        'season': season,
        'home_team_id': home_id,
        'away_team_id': away_id,
        'home_seed': home_seed,
        'away_seed': away_seed,
        'home_win': home_win,
        'seed_diff': away_seed - home_seed,  # positive = home has better (lower) seed
    }

    # Compute differentials for all candidate features
    diff_cols = {
        'diff_efg_pct': ('EFG_pct', 'EFG_pct'),
        'diff_tov_pct': ('TOV_pct', 'TOV_pct'),
        'diff_ft_pct': ('FT_pct', 'FT_pct'),
        'diff_three_par': ('ThreeP_rate', 'ThreeP_rate'),
        'diff_orb_pct': ('ORB_pct', 'ORB_pct'),
        'diff_trb_pct': ('TRB_pct', 'TRB_pct'),
        'diff_ast_rate': ('Ast_rate', 'Ast_rate'),
        'diff_stl_rate': ('Stl_rate', 'Stl_rate'),
        'diff_blk_rate': ('Blk_rate', 'Blk_rate'),
        'diff_pace': ('Pace', 'Pace'),
        'diff_ortg': ('ORtg', 'ORtg'),
        'diff_drtg': ('DRtg', 'DRtg'),
        'diff_pts_against': ('PtsAgainst', 'PtsAgainst'),
        'diff_win_pct': ('WinPct', 'WinPct'),
        # Variance metrics
        'diff_score_std': ('Score_std', 'Score_std'),
        'diff_ortg_std': ('ORtg_std', 'ORtg_std'),
        'diff_drtg_std': ('DRtg_std', 'DRtg_std'),
        'diff_margin_std': ('Margin_std', 'Margin_std'),
        # Momentum metrics
        'diff_last10_winpct': ('Last10_WinPct', 'Last10_WinPct'),
        'diff_last10_ortg': ('Last10_ORtg', 'Last10_ORtg'),
        # Conference tournament performance
        'diff_ct_wins': ('CT_Wins', 'CT_Wins'),
        'diff_ct_winpct': ('CT_WinPct', 'CT_WinPct'),
        'diff_ct_margin': ('CT_AvgMargin', 'CT_AvgMargin'),
        # Net rating, margin, SOS
        'diff_net_rtg': ('NetRtg', 'NetRtg'),
        'diff_avg_margin': ('AvgMargin', 'AvgMargin'),
        'diff_sos': ('SOS', 'SOS'),
        # Ranking volatility & trend
        'diff_rank_vol_POM': ('RankVol_POM', 'RankVol_POM'),
        'diff_rank_vol_LMC': ('RankVol_LMC', 'RankVol_LMC'),
        'diff_rank_trend_POM': ('RankTrend_POM', 'RankTrend_POM'),
        'diff_rank_trend_LMC': ('RankTrend_LMC', 'RankTrend_LMC'),
    }

    for feat_name, (col_home, col_away) in diff_cols.items():
        row[feat_name] = home_stats[col_home] - away_stats[col_away]

    # Ranking differentials (lower rank = better, so away - home = positive when home is better)
    for sys_name in RANKING_SYSTEMS:
        col = f'Rank_{sys_name}'
        h_rank = home_stats[col] if pd.notna(home_stats[col]) else np.nan
        a_rank = away_stats[col] if pd.notna(away_stats[col]) else np.nan
        if pd.notna(h_rank) and pd.notna(a_rank):
            row[f'diff_rank_{sys_name}'] = a_rank - h_rank  # positive = home ranked better
        else:
            row[f'diff_rank_{sys_name}'] = np.nan

    matchup_rows.append(row)

matchups = pd.DataFrame(matchup_rows)
print(f"  Matchups built: {len(matchups)}")
print(f"  Seasons: {matchups.season.min()}-{matchups.season.max()}")
print(f"  Home wins: {matchups.home_win.mean():.1%} (deterministic hash, ~50% expected)")

# ── 7. Save outputs ──────────────────────────────────────────────

# Add TeamName from MTeams for bracket simulator lookups
team_names = teams[['TeamID', 'TeamName']].drop_duplicates()
stats = stats.merge(team_names, on='TeamID', how='left')

stats.to_csv(PROCESSED / 'kaggle_team_stats.csv', index=False)
matchups.to_csv(PROCESSED / 'kaggle_tourney.csv', index=False)

print(f"\nSaved:")
print(f"  {PROCESSED / 'kaggle_team_stats.csv'}")
print(f"  {PROCESSED / 'kaggle_tourney.csv'}")

# ── 8. Summary of available features ─────────────────────────────

candidate_features = [c for c in matchups.columns if c.startswith('diff_') or c == 'seed_diff']

print(f"\nCandidate features ({len(candidate_features)}):")
for f in candidate_features:
    non_null = matchups[f].notna().sum()
    print(f"  {f:<22s}  {non_null}/{len(matchups)} non-null")

print("\nDone! Next: run feature selection and retrain models.")
