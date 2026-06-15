"""
2026 NCAA Tournament Bracket Predictions

Generates deterministic bracket predictions for all 3 models + ensemble.
No actual results to compare against — pure prediction mode.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.prediction.bracket_simulator import BracketSimulator

DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
MODELS_DIR = PROJECT_ROOT / 'models'
VIZ_DIR = PROJECT_ROOT / 'results' / 'visualizations'
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# ── 2026 Bracket Definition ──────────────────────────────────────
# First Four resolved (actual results):
#   Texas over NC State (11 West), Southern Methodist over Miami (OH) (11 Midwest)
#   Howard over Maryland-Baltimore County (16 Midwest), Prairie View A&M over Lehigh (16 South)
#
# F4 pairings: East vs South (semi 1), West vs Midwest (semi 2)
# Region order: [East, South, West, Midwest]

BRACKET_2026 = [
    # East Region (Duke #1 overall seed)
    [
        {'name': 'Duke', 'seed': 1},
        {'name': 'Connecticut', 'seed': 2},
        {'name': 'Michigan State', 'seed': 3},
        {'name': 'Kansas', 'seed': 4},
        {'name': "St. John's (NY)", 'seed': 5},
        {'name': 'Louisville', 'seed': 6},
        {'name': 'UCLA', 'seed': 7},
        {'name': 'Ohio State', 'seed': 8},
        {'name': 'Texas Christian', 'seed': 9},
        {'name': 'UCF', 'seed': 10},
        {'name': 'South Florida', 'seed': 11},
        {'name': 'Northern Iowa', 'seed': 12},
        {'name': 'California Baptist', 'seed': 13},
        {'name': 'North Dakota State', 'seed': 14},
        {'name': 'Furman', 'seed': 15},
        {'name': 'Siena', 'seed': 16},
    ],
    # South Region (Florida #4 overall seed)
    [
        {'name': 'Florida', 'seed': 1},
        {'name': 'Houston', 'seed': 2},
        {'name': 'Illinois', 'seed': 3},
        {'name': 'Nebraska', 'seed': 4},
        {'name': 'Vanderbilt', 'seed': 5},
        {'name': 'North Carolina', 'seed': 6},
        {'name': "Saint Mary's", 'seed': 7},
        {'name': 'Clemson', 'seed': 8},
        {'name': 'Iowa', 'seed': 9},
        {'name': 'Texas A&M', 'seed': 10},
        {'name': 'Virginia Commonwealth', 'seed': 11},
        {'name': 'McNeese', 'seed': 12},
        {'name': 'Troy', 'seed': 13},
        {'name': 'Pennsylvania', 'seed': 14},
        {'name': 'Idaho', 'seed': 15},
        {'name': 'Prairie View A&M', 'seed': 16},
    ],
    # West Region (Arizona #2 overall seed)
    [
        {'name': 'Arizona', 'seed': 1},
        {'name': 'Purdue', 'seed': 2},
        {'name': 'Gonzaga', 'seed': 3},
        {'name': 'Arkansas', 'seed': 4},
        {'name': 'Wisconsin', 'seed': 5},
        {'name': 'Brigham Young', 'seed': 6},
        {'name': 'Miami (FL)', 'seed': 7},
        {'name': 'Villanova', 'seed': 8},
        {'name': 'Utah State', 'seed': 9},
        {'name': 'Missouri', 'seed': 10},
        {'name': 'Texas', 'seed': 11},
        {'name': 'High Point', 'seed': 12},
        {'name': 'Hawaii', 'seed': 13},
        {'name': 'Kennesaw State', 'seed': 14},
        {'name': 'Queens (NC)', 'seed': 15},
        {'name': 'Long Island University', 'seed': 16},
    ],
    # Midwest Region (Michigan #3 overall seed)
    [
        {'name': 'Michigan', 'seed': 1},
        {'name': 'Iowa State', 'seed': 2},
        {'name': 'Virginia', 'seed': 3},
        {'name': 'Alabama', 'seed': 4},
        {'name': 'Texas Tech', 'seed': 5},
        {'name': 'Tennessee', 'seed': 6},
        {'name': 'Kentucky', 'seed': 7},
        {'name': 'Georgia', 'seed': 8},
        {'name': 'Saint Louis', 'seed': 9},
        {'name': 'Santa Clara', 'seed': 10},
        {'name': 'Southern Methodist', 'seed': 11},
        {'name': 'Akron', 'seed': 12},
        {'name': 'Hofstra', 'seed': 13},
        {'name': 'Wright State', 'seed': 14},
        {'name': 'Tennessee State', 'seed': 15},
        {'name': 'Howard', 'seed': 16},
    ],
]

REGION_NAMES = ['East', 'South', 'West', 'Midwest']

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
}


# ── Visualization helpers (from bracket_viz.ipynb) ────────────────

MATCHUP_ORDER = [0, 15, 7, 8, 4, 11, 3, 12, 5, 10, 2, 13, 6, 9, 1, 14]

def shorten(name):
    """Shorten team name for bracket display."""
    table = {
        'Connecticut': 'UConn', 'North Carolina': 'UNC',
        'Michigan State': 'Mich St', 'Mississippi State': 'Miss St',
        'San Diego State': 'SDSU', 'South Dakota State': 'SD State',
        'Florida Atlantic': 'FAU', 'Western Kentucky': 'W Kentucky',
        'Long Beach State': 'Long Beach', 'Iowa State': 'Iowa St',
        'Colorado State': 'Colorado St', 'South Carolina': 'S Carolina',
        'Utah State': 'Utah St', 'James Madison': 'James Mad.',
        'Grand Canyon': 'Gr Canyon', 'Texas Christian': 'TCU',
        'Brigham Young': 'BYU', 'Alabama-Birmingham': 'UAB',
        "Saint Mary's (CA)": "St Mary's", "Saint Mary's": "St Mary's",
        'Morehead State': 'Morehead St',
        "Saint Peter's": "St Peter's", 'Washington State': 'Wash St',
        'North Carolina State': 'NC State', 'Texas A&M': 'Texas A&M',
        'Virginia Commonwealth': 'VCU',
        'McNeese State': 'McNeese St',
        'Norfolk State': 'Norfolk St', 'Alabama State': 'Alabama St',
        'Robert Morris': 'Robert Mor.',
        "Mount St. Mary's": "Mt St Mary's", 'SIU Edwardsville': 'SIU-E',
        "St. John's (NY)": "St John's", 'UNC Wilmington': 'UNCW',
        'Mississippi': 'Ole Miss', 'Saint Francis Red': 'St Francis',
        # New for 2026
        'Southern Methodist': 'SMU',
        'Maryland-Baltimore County': 'UMBC',
        'Prairie View A&M': 'PV A&M',
        'California Baptist': 'Cal Baptist',
        'Long Island University': 'LIU',
        'Queens (NC)': 'Queens',
        'Saint Louis': 'St Louis',
        'North Dakota State': 'ND State',
        'Northern Iowa': 'N Iowa',
        'Kennesaw State': 'Kennesaw St',
        'Tennessee State': 'Tenn State',
        'South Florida': 'S Florida',
        'Wright State': 'Wright St',
        'High Point': 'High Point',
    }
    return table.get(name, name)


def get_r64_ordered(bracket):
    """Get R64 teams in standard matchup order per region."""
    r64 = {}
    for ri, region in enumerate(bracket):
        region_sorted = sorted(region, key=lambda x: x['seed'])
        ordered = []
        for i in range(0, 16, 2):
            ordered.append(region_sorted[MATCHUP_ORDER[i]])
            ordered.append(region_sorted[MATCHUP_ORDER[i + 1]])
        r64[ri] = ordered
    return r64


def build_pred_tree(results, bracket):
    """Build tree from simulator results."""
    tree = {'R64_teams': get_r64_ordered(bracket)}
    rounds_cfg = [('R64', 8), ('R32', 4), ('S16', 2), ('E8', 1)]
    for rname, gpr in rounds_cfg:
        key = rname + '_winners'
        tree[key] = {}
        games = results[rname]
        for ri in range(4):
            start = ri * gpr
            tree[key][ri] = [
                {'name': g['winner'], 'seed': g['winner_seed']}
                for g in games[start:start + gpr]
            ]
    tree['F4_winners'] = [
        {'name': g['winner'], 'seed': g['winner_seed']} for g in results['F4']
    ]
    tree['Championship_winner'] = {
        'name': results['Championship'][0]['winner'],
        'seed': results['Championship'][0]['winner_seed']
    }
    return tree


def draw_prediction_bracket(ax, tree, title, region_names):
    """
    Draw NCAA bracket prediction (no actual results comparison).
    All picks shown in dark blue-gray.
    """
    FS       = 14
    FS_HDR   = 14
    FS_REG   = 16
    FS_TITLE = 28
    PRED_CLR = '#1a237e'  # dark blue for predictions

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

    def put(x, y, team, c='#111111', bold=False, fs_override=None):
        ax.text(x, y, lbl(team), fontsize=fs_override or FS, color=c,
                fontweight='bold' if bold else 'normal',
                va='center', fontfamily='monospace', clip_on=False)

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
            put(RX[1], y, t, c=PRED_CLR)
            bracket_line(RX[0]+TW, yb+RH-0.5 - i*2,
                                   yb+RH-1.5 - i*2, RX[1]-0.05, y)

        w2 = tree.get('R32_winners', {}).get(ri, [])
        for i, t in enumerate(w2):
            y = yb + RH - 2 - i*4
            put(RX[2], y, t, c=PRED_CLR)
            if len(w1) > i*2+1:
                bracket_line(RX[1]+TW, yb+RH-1 - (i*2)*2,
                                       yb+RH-1 - (i*2+1)*2, RX[2]-0.05, y)

        w3 = tree.get('S16_winners', {}).get(ri, [])
        for i, t in enumerate(w3):
            y = yb + RH - 4 - i*8
            put(RX[3], y, t, c=PRED_CLR)
            if len(w2) > i*2+1:
                bracket_line(RX[2]+TW, yb+RH-2 - (i*2)*4,
                                       yb+RH-2 - (i*2+1)*4, RX[3]-0.05, y)

        w4 = tree.get('E8_winners', {}).get(ri, [])
        if w4:
            y = yb + RH/2
            put(RX[4], y, w4[0], c=PRED_CLR, bold=True)
            if len(w3) >= 2:
                bracket_line(RX[3]+TW, yb+RH-4, yb+RH-12, RX[4]-0.05, y)

    # Final Four
    f4 = tree.get('F4_winners', [])
    y_top = (ystarts[0] + RH/2 + ystarts[1] + RH/2) / 2
    y_bot = (ystarts[2] + RH/2 + ystarts[3] + RH/2) / 2

    for idx, yy, regions in [(0, y_top, [0,1]), (1, y_bot, [2,3])]:
        if idx < len(f4):
            put(RX[5], yy, f4[idx], c=PRED_CLR, bold=True)
            for pri in regions:
                if tree.get('E8_winners', {}).get(pri):
                    bracket_line(RX[4]+TW, ystarts[pri]+RH/2,
                                           ystarts[pri]+RH/2, RX[5]-0.05, yy)

    # Champion
    ch = tree.get('Championship_winner')
    if ch and len(f4) >= 2:
        yc = (y_top + y_bot) / 2
        ax.text(RX[5]+1.2, yc+1.8, 'PREDICTED CHAMPION', fontsize=16,
                fontweight='bold', ha='center', va='bottom', color='#37474f')
        put(RX[5]+0.2, yc, ch, c='#b71c1c', bold=True, fs_override=FS+3)
        bracket_line(RX[5]+TW, y_top, y_bot, RX[5]+0.15, yc)

    # Round headers
    hdr_y = max(ystarts) + RH + 1.5
    for i, h in enumerate(['R64', 'R32', 'Sweet 16', 'Elite 8', 'Final 4', 'Champ']):
        ax.text(RX[i]+0.6, hdr_y, h, fontsize=FS_HDR, ha='center',
                color='#546e7a', fontweight='bold')

    # Prediction label
    ax.text(RX[0], -2.8, 'PREDICTION (no results yet)',
            fontsize=18, fontfamily='monospace', fontweight='bold',
            color='#37474f', va='center', ha='left', clip_on=False)


# ── Main: Generate predictions ────────────────────────────────────

def main():
    stats_path = DATA_DIR / 'sportsref_pretourney_2026.csv'
    if not stats_path.exists():
        stats_path = DATA_DIR / 'sportsref_combined_2026.csv'
    print(f"Using stats: {stats_path}")

    results_summary = []

    for model_name, model_cfg in MODELS.items():
        print(f"\n--- {model_name} ---")

        sim = BracketSimulator(
            str(MODELS_DIR / model_cfg['model']),
            str(MODELS_DIR / model_cfg['scaler']),
            str(MODELS_DIR / model_cfg['features'])
        )
        sim.load_team_stats(str(stats_path))

        predicted = sim.generate_deterministic_bracket(BRACKET_2026)
        pred_tree = build_pred_tree(predicted, BRACKET_2026)

        champ = pred_tree['Championship_winner']
        print(f"  Champion: ({champ['seed']}) {champ['name']}")

        # Final Four
        f4 = pred_tree['F4_winners']
        e8 = pred_tree.get('E8_winners', {})
        f4_teams = [e8[ri][0] for ri in range(4)]
        f4_str = ', '.join(f"({t['seed']}) {t['name']}" for t in f4_teams)
        print(f"  Final Four: {f4_str}")

        # Draw bracket
        fig, ax = plt.subplots(1, 1, figsize=(20, 28))
        fig.patch.set_facecolor('white')

        title = f'{model_name} — 2026 Prediction'
        draw_prediction_bracket(ax, pred_tree, title, REGION_NAMES)

        safe_name = model_name.lower().replace(' ', '_')
        fname = f'bracket_2026_{safe_name}.png'
        out_path = VIZ_DIR / fname
        plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  Saved: {out_path}")

        results_summary.append({
            'model': model_name,
            'champion': champ['name'],
            'champion_seed': champ['seed'],
            'final_four': [e8[ri][0]['name'] for ri in range(4)],
        })

    # Summary
    print("\n" + "=" * 60)
    print("2026 NCAA TOURNAMENT PREDICTIONS")
    print("=" * 60)
    print(f"{'Model':<20s} {'Champion':<25s} {'Seed':>4s}")
    print("-" * 60)
    for r in results_summary:
        print(f"{r['model']:<20s} ({r['champion_seed']}) {r['champion']:<22s}")
    print("=" * 60)

    print("\nFinal Four picks:")
    for r in results_summary:
        f4 = ', '.join(r['final_four'])
        print(f"  {r['model']:<20s} {f4}")


if __name__ == '__main__':
    main()
