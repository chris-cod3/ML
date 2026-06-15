"""
Fix Data Leakage in Sports Reference Stats.

The Sports Reference stats include tournament games, which creates data leakage.
This module provides functions to estimate pre-tournament stats by subtracting
tournament game contributions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple


def load_tournament_games(year: int, data_dir: Path) -> pd.DataFrame:
    """Load tournament results and count games per team."""
    tourney_path = data_dir / f'espn_tournament_{year}.csv'
    if not tourney_path.exists():
        raise FileNotFoundError(f"Tournament data not found: {tourney_path}")

    return pd.read_csv(tourney_path)


def count_tournament_games(tourney_df: pd.DataFrame) -> Dict[str, int]:
    """
    Count how many tournament games each team played.

    Returns dict mapping team_name -> number of tournament games
    """
    game_counts = {}

    for _, game in tourney_df.iterrows():
        # Skip First Four games (they have 'F4' round but dates in March before main tournament)
        if game['round'] == 'F4':
            continue

        home = game['home_team_name']
        away = game['away_team_name']

        game_counts[home] = game_counts.get(home, 0) + 1
        game_counts[away] = game_counts.get(away, 0) + 1

    return game_counts


def count_tournament_wins(tourney_df: pd.DataFrame) -> Dict[str, int]:
    """
    Count how many tournament wins each team had.

    Returns dict mapping team_name -> number of tournament wins
    """
    win_counts = {}

    for _, game in tourney_df.iterrows():
        # Skip First Four games
        if game['round'] == 'F4':
            continue

        # Determine winner
        if game['home_score'] > game['away_score']:
            winner = game['home_team_name']
        else:
            winner = game['away_team_name']

        win_counts[winner] = win_counts.get(winner, 0) + 1

    return win_counts


def normalize_team_name(name: str) -> str:
    """Normalize ESPN team name to Sports Reference format."""
    # Direct mappings for known mismatches
    mappings = {
        'UConn Huskies': 'Connecticut',
        'NC State Wolfpack': 'North Carolina State',
        'Pitt Panthers': 'Pittsburgh',
        'UNC Tar Heels': 'North Carolina',
        'USC Trojans': 'Southern California',
        'LSU Tigers': 'Louisiana State',
        'SMU Mustangs': 'Southern Methodist',
        'UCF Knights': 'Central Florida',
        'UNLV Rebels': 'Nevada-Las Vegas',
        'VCU Rams': 'Virginia Commonwealth',
        'BYU Cougars': 'Brigham Young',
        'TCU Horned Frogs': 'Texas Christian',
        'UAB Blazers': 'Alabama-Birmingham',
        'UTEP Miners': 'Texas-El Paso',
    }

    if name in mappings:
        return mappings[name]

    # Remove common suffixes
    suffixes = [
        ' Huskies', ' Wolfpack', ' Tar Heels', ' Blue Devils', ' Crimson Tide',
        ' Boilermakers', ' Wildcats', ' Tigers', ' Bulldogs', ' Bears',
        ' Cyclones', ' Fighting Illini', ' Volunteers', ' Cougars',
        ' Gamecocks', ' Ducks', ' Longhorns', ' Red Raiders', ' Flyers',
        ' Spartans', ' Golden Eagles', ' Catamounts', ' Zips', ' Eagles',
        ' Hatters', ' Lancers', ' Cowboys', ' Dukes', ' Golden Grizzlies',
        ' Seahawks', ' Bison', ' Rams', ' Buffaloes', ' Broncos',
        ' Beach', ' Colonels', ' Gaels', ' Lopes', ' Lobos', ' Owls',
        ' Horned Frogs', ' Aggies', ' Raiders', ' Peacocks', ' Bluejays',
        ' Aztecs', ' Blazers', ' Cavaliers', ' Jayhawks', ' Mountaineers',
        ' Hoosiers', ' Hawkeyes', ' Badgers', ' Gophers', ' Buckeyes',
        ' Nittany Lions', ' Terrapins', ' Scarlet Knights', ' Hokies',
        ' Orange', ' Cardinals', ' Demon Deacons', ' Yellow Jackets',
        ' Seminoles', ' Hurricanes', ' Wolfpack', ' Commodores',
        ' Razorbacks', ' Rebels', ' Sooners', ' Cornhuskers', ' Jackrabbits',
        ' Musketeers', ' Friars', ' Hilltoppers', ' Shockers', ' Rockets',
        ' Thundering Herd', ' Mean Green', ' Roadrunners', ' Miners',
        ' Golden Hurricane', ' Bearcats', ' Red Storm', ' Hoyas',
        ' Pirates', ' Billikens', ' Explorers', ' Hawks', ' Bonnies',
        ' Bobcats', ' Wolf Pack',
    ]

    result = name
    for suffix in suffixes:
        if result.endswith(suffix):
            result = result[:-len(suffix)]
            break

    return result.strip()


def estimate_pre_tournament_stats(
    stats_df: pd.DataFrame,
    tourney_games: Dict[str, int],
    tourney_wins: Dict[str, int]
) -> pd.DataFrame:
    """
    Estimate pre-tournament stats by subtracting tournament contributions.

    This is an APPROXIMATION since we don't have game-by-game data.
    We adjust:
    - Games played (subtract tournament games)
    - Wins (subtract tournament wins)
    - Win percentage (recalculate)

    Note: We cannot accurately adjust SRS, points, rebounds, etc. without
    knowing the exact stats from each tournament game. Those are left as-is
    but will be slightly inflated for teams that went deep.
    """
    df = stats_df.copy()

    # Create normalized name column for matching
    df['_normalized'] = df['School'].apply(lambda x: normalize_team_name(x).lower().strip())

    # Create lookup for tournament data
    tourney_games_normalized = {
        normalize_team_name(k).lower().strip(): v
        for k, v in tourney_games.items()
    }
    tourney_wins_normalized = {
        normalize_team_name(k).lower().strip(): v
        for k, v in tourney_wins.items()
    }

    # Track adjustments
    adjustments = []

    for idx, row in df.iterrows():
        norm_name = row['_normalized']

        t_games = tourney_games_normalized.get(norm_name, 0)
        t_wins = tourney_wins_normalized.get(norm_name, 0)

        if t_games > 0:
            # Original values
            orig_games = row.get('Overall_G', row.get('G', 0))
            orig_wins = row.get('Overall_W', row.get('W', 0))

            # Adjusted values
            adj_games = orig_games - t_games
            adj_wins = orig_wins - t_wins
            adj_losses = adj_games - adj_wins
            adj_win_pct = adj_wins / adj_games if adj_games > 0 else 0

            # Update DataFrame
            if 'Overall_G' in df.columns:
                df.at[idx, 'Overall_G'] = adj_games
            if 'Overall_W' in df.columns:
                df.at[idx, 'Overall_W'] = adj_wins
            if 'Overall_L' in df.columns:
                df.at[idx, 'Overall_L'] = adj_losses
            if 'Overall_W-L%' in df.columns:
                df.at[idx, 'Overall_W-L%'] = adj_win_pct

            adjustments.append({
                'school': row['School'],
                'tourney_games': t_games,
                'tourney_wins': t_wins,
                'orig_games': orig_games,
                'adj_games': adj_games,
                'orig_win_pct': row.get('Overall_W-L%', 0),
                'adj_win_pct': adj_win_pct
            })

    # Remove temporary column
    df = df.drop(columns=['_normalized'])

    # Print adjustment summary
    if adjustments:
        print(f"\nAdjusted {len(adjustments)} teams:")
        adj_df = pd.DataFrame(adjustments).sort_values('tourney_games', ascending=False)
        print(adj_df.head(10).to_string(index=False))

    return df


def create_clean_stats(year: int, data_dir: Path, output_dir: Path = None) -> pd.DataFrame:
    """
    Create a clean stats file with tournament games removed.

    Args:
        year: Season year
        data_dir: Directory containing raw data files
        output_dir: Where to save cleaned stats (defaults to data_dir)

    Returns:
        DataFrame with pre-tournament stats
    """
    if output_dir is None:
        output_dir = data_dir

    print(f"\n{'='*60}")
    print(f"Creating pre-tournament stats for {year}")
    print(f"{'='*60}")

    # Load original stats
    stats_path = data_dir / f'sportsref_combined_{year}.csv'
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")

    stats_df = pd.read_csv(stats_path)
    print(f"Loaded {len(stats_df)} teams from {stats_path}")

    # Load tournament data
    tourney_df = load_tournament_games(year, data_dir)
    print(f"Loaded {len(tourney_df)} tournament games")

    # Count tournament games and wins per team
    tourney_games = count_tournament_games(tourney_df)
    tourney_wins = count_tournament_wins(tourney_df)

    print(f"\nTeams in tournament: {len(tourney_games)}")
    print(f"Champion games: {max(tourney_games.values())}")

    # Estimate pre-tournament stats
    clean_df = estimate_pre_tournament_stats(stats_df, tourney_games, tourney_wins)

    # Save cleaned stats
    output_path = output_dir / f'sportsref_combined_{year}_clean.csv'
    clean_df.to_csv(output_path, index=False)
    print(f"\nSaved cleaned stats to {output_path}")

    return clean_df


def create_all_clean_stats(years: list, data_dir: Path) -> None:
    """Create clean stats files for multiple years."""
    for year in years:
        try:
            create_clean_stats(year, data_dir)
        except FileNotFoundError as e:
            print(f"Skipping {year}: {e}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Fix data leakage in Sports Reference stats')
    parser.add_argument('--year', type=int, default=2024, help='Year to process')
    parser.add_argument('--all', action='store_true', help='Process all available years')
    parser.add_argument('--data-dir', type=str, default='data/raw', help='Data directory')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if args.all:
        years = [2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024]
        create_all_clean_stats(years, data_dir)
    else:
        create_clean_stats(args.year, data_dir)
