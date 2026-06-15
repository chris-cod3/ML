"""Show a concrete example of how the leakage fix works."""
import pandas as pd

sr = pd.read_csv('data/raw/sportsref_combined_2024.csv')
tourney = pd.read_csv('data/raw/espn_tournament_2024.csv')

# === UConn: 2024 National Champion (6 tournament games, 6 wins) ===
uconn = sr[sr['School'] == 'Connecticut'].iloc[0]

print("=" * 70)
print("EXAMPLE: UConn 2024 (National Champion)")
print("=" * 70)
print()
print("ORIGINAL Sports Reference Stats (includes tournament games):")
print(f"  Games:   {int(uconn['Overall_G'])}")
print(f"  Wins:    {int(uconn['Overall_W'])}")
print(f"  Losses:  {int(uconn['Overall_L'])}")
print(f"  Win%:    {uconn['Overall_W-L%']:.4f} ({uconn['Overall_W-L%']*100:.1f}%)")
print()

# Find UConn's tournament games
mask_home = tourney['home_team_name'].str.contains('UConn')
mask_away = tourney['away_team_name'].str.contains('UConn')
uconn_games = tourney[(mask_home | mask_away) & (tourney['round'] != 'F4')]

print("UConn's tournament games:")
t_wins = 0
for _, g in uconn_games.iterrows():
    home_won = g['home_score'] > g['away_score']
    winner = g['home_team_name'] if home_won else g['away_team_name']
    is_uconn_win = 'UConn' in winner
    if is_uconn_win:
        t_wins += 1
    print(f"  {g['round']:6s}: {g['home_team_name'][:22]:22s} {int(g['home_score']):3d} - {int(g['away_score']):3d} {g['away_team_name'][:22]}")

t_games = len(uconn_games)
print(f"\nTournament: {t_games} games, {t_wins} wins")

print()
print("AFTER LEAKAGE FIX (subtract tournament games):")
adj_games = int(uconn['Overall_G']) - t_games
adj_wins = int(uconn['Overall_W']) - t_wins
adj_pct = adj_wins / adj_games
print(f"  Games:   {int(uconn['Overall_G'])} - {t_games} = {adj_games}")
print(f"  Wins:    {int(uconn['Overall_W'])} - {t_wins} = {adj_wins}")
print(f"  Win%:    {adj_pct:.4f} ({adj_pct*100:.1f}%)")
print(f"  Change:  {uconn['Overall_W-L%']*100:.1f}% -> {adj_pct*100:.1f}% (diff: {(uconn['Overall_W-L%'] - adj_pct)*100:.1f}%)")

# === Stetson: Lost in Round of 64 (1 tournament game, 0 wins) ===
print()
print("=" * 70)
print("EXAMPLE: Stetson 2024 (Lost Round of 64)")
print("=" * 70)
print()

stetson = sr[sr['School'] == 'Stetson'].iloc[0]

print("ORIGINAL Stats:")
print(f"  Games:   {int(stetson['Overall_G'])}")
print(f"  Wins:    {int(stetson['Overall_W'])}")
print(f"  Win%:    {stetson['Overall_W-L%']:.4f} ({stetson['Overall_W-L%']*100:.1f}%)")

mask_home = tourney['home_team_name'].str.contains('Stetson')
mask_away = tourney['away_team_name'].str.contains('Stetson')
stetson_games = tourney[(mask_home | mask_away) & (tourney['round'] != 'F4')]
s_games = len(stetson_games)
s_wins = 0
for _, g in stetson_games.iterrows():
    winner = g['home_team_name'] if g['home_score'] > g['away_score'] else g['away_team_name']
    if 'Stetson' in winner:
        s_wins += 1

print(f"\nTournament: {s_games} game(s), {s_wins} win(s)")

print()
print("AFTER LEAKAGE FIX:")
s_adj_games = int(stetson['Overall_G']) - s_games
s_adj_wins = int(stetson['Overall_W']) - s_wins
s_adj_pct = s_adj_wins / s_adj_games
print(f"  Games:   {int(stetson['Overall_G'])} - {s_games} = {s_adj_games}")
print(f"  Wins:    {int(stetson['Overall_W'])} - {s_wins} = {s_adj_wins}")
print(f"  Win%:    {s_adj_pct:.4f} ({s_adj_pct*100:.1f}%)")
print(f"  Change:  {stetson['Overall_W-L%']*100:.1f}% -> {s_adj_pct*100:.1f}% (diff: {(stetson['Overall_W-L%'] - s_adj_pct)*100:.1f}%)")

# === The key insight ===
print()
print("=" * 70)
print("THE KEY INSIGHT")
print("=" * 70)
print()
print("Win% differential (UConn - Stetson) used as model feature:")
orig_diff = uconn['Overall_W-L%'] - stetson['Overall_W-L%']
fix_diff = adj_pct - s_adj_pct
print(f"  Before fix: {orig_diff:.4f}")
print(f"  After fix:  {fix_diff:.4f}")
print(f"  Difference: {(orig_diff - fix_diff):.4f}")
print()
print("The leakage gave UConn an extra boost because they won 6")
print("tournament games while Stetson lost their only one.")
print()
print("BUT: Our fix ONLY adjusts win%, games, and wins.")
print("Other stats like SRS, points, rebounds, etc. still include")
print("tournament game contributions and we have NO way to subtract them.")
print()

# Show which features we actually use
import joblib
features = joblib.load('models/features_23.pkl')
print("=" * 70)
print("OUR 23 MODEL FEATURES - Which ones are fixed?")
print("=" * 70)
fixed = {'diff_win_pct'}
partially_leaked = {'diff_srs', 'diff_sos', 'diff_pts_for', 'diff_pts_against',
                     'diff_trb', 'diff_ast', 'diff_three_made', 'diff_blk',
                     'diff_stl', 'diff_tov', 'diff_ortg', 'diff_pace',
                     'diff_ts_pct', 'diff_efg_pct', 'diff_tov_pct', 'diff_fg_pct',
                     'diff_three_pct', 'diff_ft_pct', 'diff_blk_pct', 'diff_ftr',
                     'diff_trb_pct'}
clean = {'seed_diff'}

for f in features:
    if f in fixed:
        status = "FIXED (adjusted)"
    elif f in clean:
        status = "CLEAN (no leakage)"
    else:
        status = "STILL LEAKED (not adjusted)"
    print(f"  {f:20s} -> {status}")
