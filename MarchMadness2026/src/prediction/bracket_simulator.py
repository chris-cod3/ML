"""
NCAA Tournament Bracket Simulator

Simulates the entire tournament bracket using trained ML model to predict
game outcomes. Supports multiple simulation runs to generate win probabilities.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse


class BracketSimulator:
    """Simulates NCAA tournament brackets using trained ML model."""

    # Tournament structure: teams per round
    ROUNDS = ['R64', 'R32', 'S16', 'E8', 'F4', 'Championship']
    GAMES_PER_ROUND = [32, 16, 8, 4, 2, 1]

    # ESPN scoring system
    ESPN_POINTS = {
        'R64': 10,
        'R32': 20,
        'S16': 40,
        'E8': 80,
        'F4': 160,
        'Championship': 320
    }

    def __init__(self, model_path: str, scaler_path: str, features_path: str):
        """
        Initialize simulator with trained model.

        Args:
            model_path: Path to trained model .pkl file
            scaler_path: Path to scaler .pkl file
            features_path: Path to features list .pkl file
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.features = joblib.load(features_path)
        self.team_stats = None

        # Only scale features for models that were trained on scaled data
        # (logistic regression). Tree-based models (XGBoost, RF) were trained
        # on unscaled data and should not be scaled at inference time.
        model_type = type(self.model).__name__
        self.needs_scaling = model_type in ('LogisticRegression',)

    def load_team_stats(self, stats_path: str):
        """Load team season statistics."""
        self.team_stats = pd.read_csv(stats_path)
        # Support both old (School) and new (TeamName) column names
        name_col = 'TeamName' if 'TeamName' in self.team_stats.columns else 'School'
        self.team_stats['school_normalized'] = self.team_stats[name_col].str.lower().str.strip()
        print(f"Loaded stats for {len(self.team_stats)} teams")

    def get_team_stats(self, team_name: str) -> Optional[pd.Series]:
        """Get stats for a team by name (fuzzy matching)."""
        if self.team_stats is None:
            raise ValueError("Team stats not loaded. Call load_team_stats() first.")

        name_lower = team_name.lower().strip()

        # Try exact match first
        match = self.team_stats[self.team_stats['school_normalized'] == name_lower]
        if len(match) == 1:
            return match.iloc[0]

        # Try partial match
        match = self.team_stats[self.team_stats['school_normalized'].str.contains(name_lower, na=False)]
        if len(match) == 1:
            return match.iloc[0]

        # Try matching first word
        first_word = name_lower.split()[0] if ' ' in name_lower else name_lower
        match = self.team_stats[self.team_stats['school_normalized'].str.startswith(first_word)]
        if len(match) == 1:
            return match.iloc[0]

        return None

    def compute_features(self, team_a: pd.Series, team_b: pd.Series) -> np.ndarray:
        """
        Compute feature differentials for a matchup.

        Args:
            team_a: Stats for team A (home/higher seed)
            team_b: Stats for team B (away/lower seed)

        Returns:
            Feature array for model input
        """
        # Map feature names to column names in stats CSV
        # Supports both Kaggle (kaggle_team_stats.csv) and SportsRef (sportsref_pretourney_*.csv) formats
        feature_map = {
            'seed_diff': None,  # Computed separately
            # Kaggle box-score derived features
            'diff_pace': ('Pace', 'Pace'),
            'diff_ortg': ('ORtg', 'ORtg'),
            'diff_drtg': ('DRtg', 'DRtg'),
            'diff_efg_pct': ('EFG_pct', 'EFG_pct'),
            'diff_tov_pct': ('TOV_pct', 'TOV_pct'),
            'diff_ft_pct': ('FT_pct', 'FT_pct'),
            'diff_three_par': ('ThreeP_rate', 'ThreeP_rate'),
            'diff_orb_pct': ('ORB_pct', 'ORB_pct'),
            'diff_trb_pct': ('TRB_pct', 'TRB_pct'),
            'diff_ast_rate': ('Ast_rate', 'Ast_rate'),
            'diff_stl_rate': ('Stl_rate', 'Stl_rate'),
            'diff_blk_rate': ('Blk_rate', 'Blk_rate'),
            'diff_win_pct': ('WinPct', 'WinPct'),
            'diff_pts_against': ('PtsAgainst', 'PtsAgainst'),
            'diff_rank_POM': ('Rank_POM', 'Rank_POM'),
            # Legacy SportsRef mappings (kept for backward compatibility)
            'diff_srs': ('Overall_SRS', 'Overall_SRS'),
            'diff_rank_composite': ('Massey_Composite_Rank', 'Massey_Composite_Rank'),
            'diff_rank_PGH': ('Massey_PGH_Rank', 'Massey_PGH_Rank'),
            'diff_rank_LMC': ('Massey_LMC_Rank', 'Massey_LMC_Rank'),
            'diff_rank_MAS': ('Massey_MAS_Rank', 'Massey_MAS_Rank'),
        }

        features = []
        for feat in self.features:
            if feat == 'seed_diff':
                # Will be set by caller
                features.append(0)
            elif feat in feature_map and feature_map[feat] is not None:
                col_a, col_b = feature_map[feat]
                val_a = pd.to_numeric(team_a.get(col_a, 0), errors='coerce')
                val_b = pd.to_numeric(team_b.get(col_b, 0), errors='coerce')
                if pd.isna(val_a): val_a = 0
                if pd.isna(val_b): val_b = 0
                features.append(val_a - val_b)
            else:
                features.append(0)

        return np.array(features).reshape(1, -1)

    def predict_game(self, team_a: Dict, team_b: Dict, use_probability: bool = True) -> Tuple[Dict, float]:
        """
        Predict winner of a single game.

        Args:
            team_a: Dict with 'name', 'seed', and optionally 'stats'
            team_b: Dict with 'name', 'seed', and optionally 'stats'
            use_probability: If True, use probability for stochastic simulation

        Returns:
            Tuple of (winner dict, win probability)
        """
        # Get team stats
        stats_a = team_a.get('stats') or self.get_team_stats(team_a['name'])
        stats_b = team_b.get('stats') or self.get_team_stats(team_b['name'])

        if stats_a is None or stats_b is None:
            # Fallback to seed-based prediction
            if team_a['seed'] <= team_b['seed']:
                return team_a, 0.5 + (team_b['seed'] - team_a['seed']) * 0.03
            else:
                return team_b, 0.5 + (team_a['seed'] - team_b['seed']) * 0.03

        # Compute features (team_a as "home")
        features = self.compute_features(stats_a, stats_b)

        # Set seed differential
        seed_idx = self.features.index('seed_diff') if 'seed_diff' in self.features else -1
        if seed_idx >= 0:
            features[0, seed_idx] = team_b['seed'] - team_a['seed']

        # Scale features only for models trained on scaled data (LogReg)
        model_input = self.scaler.transform(features) if self.needs_scaling else features

        # Get win probability
        prob_a_wins = self.model.predict_proba(model_input)[0, 1]

        if use_probability:
            # Stochastic: sample from probability
            winner = team_a if np.random.random() < prob_a_wins else team_b
        else:
            # Deterministic: pick higher probability
            winner = team_a if prob_a_wins >= 0.5 else team_b

        win_prob = prob_a_wins if winner == team_a else (1 - prob_a_wins)

        return winner, win_prob

    def simulate_bracket(self, bracket: List[List[Dict]], use_probability: bool = True) -> Dict:
        """
        Simulate entire tournament bracket.

        Args:
            bracket: List of regions, each containing list of teams with 'name' and 'seed'
            use_probability: If True, use stochastic simulation

        Returns:
            Dict with results by round and champion
        """
        results = {round_name: [] for round_name in self.ROUNDS}

        # Flatten bracket into matchups (assumes standard 64-team bracket)
        # Each region has 16 teams seeded 1-16
        current_round_teams = []

        for region in bracket:
            # Sort by seed
            region_sorted = sorted(region, key=lambda x: x['seed'])
            current_round_teams.extend(region_sorted)

        # Standard bracket matchups: 1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15 per region
        matchup_order = [0, 15, 7, 8, 4, 11, 3, 12, 5, 10, 2, 13, 6, 9, 1, 14]

        for round_idx, round_name in enumerate(self.ROUNDS):
            round_winners = []
            n_games = self.GAMES_PER_ROUND[round_idx]

            if round_idx == 0:
                # First round: use matchup order within each region
                for region_idx in range(4):
                    region_teams = current_round_teams[region_idx * 16:(region_idx + 1) * 16]
                    for i in range(0, 16, 2):
                        idx_a = matchup_order[i]
                        idx_b = matchup_order[i + 1]
                        team_a = region_teams[idx_a]
                        team_b = region_teams[idx_b]

                        winner, prob = self.predict_game(team_a, team_b, use_probability)
                        results[round_name].append({
                            'team_a': team_a['name'],
                            'seed_a': team_a['seed'],
                            'team_b': team_b['name'],
                            'seed_b': team_b['seed'],
                            'winner': winner['name'],
                            'winner_seed': winner['seed'],
                            'probability': prob
                        })
                        round_winners.append(winner)
            else:
                # Subsequent rounds: winners play in order
                for i in range(0, len(current_round_teams), 2):
                    team_a = current_round_teams[i]
                    team_b = current_round_teams[i + 1]

                    winner, prob = self.predict_game(team_a, team_b, use_probability)
                    results[round_name].append({
                        'team_a': team_a['name'],
                        'seed_a': team_a['seed'],
                        'team_b': team_b['name'],
                        'seed_b': team_b['seed'],
                        'winner': winner['name'],
                        'winner_seed': winner['seed'],
                        'probability': prob
                    })
                    round_winners.append(winner)

            current_round_teams = round_winners

        results['champion'] = current_round_teams[0] if current_round_teams else None
        return results

    def run_simulations(self, bracket: List[List[Dict]], n_simulations: int = 1000) -> Dict:
        """
        Run multiple tournament simulations to get win probabilities.

        Args:
            bracket: Tournament bracket
            n_simulations: Number of simulations to run

        Returns:
            Dict with team win counts by round
        """
        all_teams = [team['name'] for region in bracket for team in region]

        # Track wins by round
        round_wins = {round_name: {team: 0 for team in all_teams} for round_name in self.ROUNDS}
        championships = {team: 0 for team in all_teams}

        for i in range(n_simulations):
            if (i + 1) % 100 == 0:
                print(f"Simulation {i + 1}/{n_simulations}")

            results = self.simulate_bracket(bracket, use_probability=True)

            for round_name in self.ROUNDS:
                for game in results[round_name]:
                    round_wins[round_name][game['winner']] += 1

            if results['champion']:
                championships[results['champion']['name']] += 1

        # Convert to probabilities
        round_probs = {
            round_name: {team: wins / n_simulations for team, wins in team_wins.items()}
            for round_name, team_wins in round_wins.items()
        }
        championship_probs = {team: wins / n_simulations for team, wins in championships.items()}

        return {
            'round_probabilities': round_probs,
            'championship_probabilities': championship_probs,
            'n_simulations': n_simulations
        }

    def generate_deterministic_bracket(self, bracket: List[List[Dict]]) -> Dict:
        """Generate single bracket using deterministic predictions (no randomness)."""
        return self.simulate_bracket(bracket, use_probability=False)

    def print_bracket(self, results: Dict):
        """Print bracket results in readable format."""
        print("\n" + "=" * 70)
        print("NCAA TOURNAMENT BRACKET PREDICTION")
        print("=" * 70)

        for round_name in self.ROUNDS:
            print(f"\n{round_name}")
            print("-" * 50)
            for game in results[round_name]:
                print(f"  ({game['seed_a']}) {game['team_a']:20s} vs ({game['seed_b']}) {game['team_b']:20s}")
                print(f"       Winner: ({game['winner_seed']}) {game['winner']} ({game['probability']:.1%})")

        print("\n" + "=" * 70)
        if results['champion']:
            print(f"CHAMPION: ({results['champion']['seed']}) {results['champion']['name']}")
        print("=" * 70)

    def print_championship_odds(self, sim_results: Dict, top_n: int = 20):
        """Print championship odds from simulations."""
        probs = sim_results['championship_probabilities']
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)

        print("\n" + "=" * 50)
        print(f"CHAMPIONSHIP ODDS ({sim_results['n_simulations']} simulations)")
        print("=" * 50)

        for i, (team, prob) in enumerate(sorted_probs[:top_n], 1):
            bar = "#" * int(prob * 50)
            print(f"{i:2d}. {team:25s} {prob:6.1%}  {bar}")


class EnsembleSimulator(BracketSimulator):
    """Ensemble simulator that averages probabilities from multiple models."""

    def __init__(self, model_configs: List[Dict[str, str]], stats_path: str = None):
        """
        Args:
            model_configs: List of dicts with 'model', 'scaler', 'features' paths.
            stats_path: Path to team stats CSV (optional, can call load_team_stats later).
        """
        self.simulators = []
        for cfg in model_configs:
            sim = BracketSimulator(cfg['model'], cfg['scaler'], cfg['features'])
            self.simulators.append(sim)

        # Use the first simulator's attributes for shared state
        self.features = self.simulators[0].features
        self.scaler = self.simulators[0].scaler
        self.model = self.simulators[0].model
        self.needs_scaling = self.simulators[0].needs_scaling
        self.team_stats = None

        if stats_path:
            self.load_team_stats(stats_path)

    def load_team_stats(self, stats_path: str):
        """Load stats into all sub-simulators."""
        super().load_team_stats(stats_path)
        for sim in self.simulators:
            sim.team_stats = self.team_stats
            sim.team_stats = self.team_stats

    def predict_game(self, team_a: Dict, team_b: Dict, use_probability: bool = True) -> Tuple[Dict, float]:
        """Average probabilities across all models, then pick winner."""
        probs = []
        for sim in self.simulators:
            stats_a = team_a.get('stats') or sim.get_team_stats(team_a['name'])
            stats_b = team_b.get('stats') or sim.get_team_stats(team_b['name'])

            if stats_a is None or stats_b is None:
                # Fallback
                if team_a['seed'] <= team_b['seed']:
                    probs.append(0.5 + (team_b['seed'] - team_a['seed']) * 0.03)
                else:
                    probs.append(0.5 - (team_a['seed'] - team_b['seed']) * 0.03)
                continue

            features = sim.compute_features(stats_a, stats_b)
            seed_idx = sim.features.index('seed_diff') if 'seed_diff' in sim.features else -1
            if seed_idx >= 0:
                features[0, seed_idx] = team_b['seed'] - team_a['seed']

            model_input = sim.scaler.transform(features) if sim.needs_scaling else features
            prob_a = sim.model.predict_proba(model_input)[0, 1]
            probs.append(prob_a)

        # Average probability across all models
        avg_prob = np.mean(probs)

        if use_probability:
            winner = team_a if np.random.random() < avg_prob else team_b
        else:
            winner = team_a if avg_prob >= 0.5 else team_b

        win_prob = avg_prob if winner == team_a else (1 - avg_prob)
        return winner, win_prob


def create_sample_bracket() -> List[List[Dict]]:
    """Create a sample 2024-style bracket for testing."""
    # 4 regions with 16 teams each
    regions = [
        # South Region
        [
            {'name': 'Houston', 'seed': 1},
            {'name': 'Auburn', 'seed': 2},
            {'name': 'Iowa State', 'seed': 3},
            {'name': 'Duke', 'seed': 4},
            {'name': 'Wisconsin', 'seed': 5},
            {'name': 'BYU', 'seed': 6},
            {'name': 'Texas', 'seed': 7},
            {'name': 'Florida', 'seed': 8},
            {'name': 'Michigan State', 'seed': 9},
            {'name': 'New Mexico', 'seed': 10},
            {'name': 'NC State', 'seed': 11},
            {'name': 'McNeese', 'seed': 12},
            {'name': 'Samford', 'seed': 13},
            {'name': 'Oakland', 'seed': 14},
            {'name': 'Western Kentucky', 'seed': 15},
            {'name': 'Longwood', 'seed': 16},
        ],
        # East Region
        [
            {'name': 'Connecticut', 'seed': 1},
            {'name': 'Iowa State', 'seed': 2},
            {'name': 'Illinois', 'seed': 3},
            {'name': 'Auburn', 'seed': 4},
            {'name': 'San Diego State', 'seed': 5},
            {'name': 'Creighton', 'seed': 6},
            {'name': 'Northwestern', 'seed': 7},
            {'name': 'FAU', 'seed': 8},
            {'name': 'Drake', 'seed': 9},
            {'name': 'Colorado', 'seed': 10},
            {'name': 'Oregon', 'seed': 11},
            {'name': 'Grand Canyon', 'seed': 12},
            {'name': 'Vermont', 'seed': 13},
            {'name': 'Morehead State', 'seed': 14},
            {'name': 'Long Beach State', 'seed': 15},
            {'name': 'Stetson', 'seed': 16},
        ],
        # Midwest Region
        [
            {'name': 'Purdue', 'seed': 1},
            {'name': 'Tennessee', 'seed': 2},
            {'name': 'Creighton', 'seed': 3},
            {'name': 'Kansas', 'seed': 4},
            {'name': 'Gonzaga', 'seed': 5},
            {'name': 'South Carolina', 'seed': 6},
            {'name': 'Texas', 'seed': 7},
            {'name': 'Utah State', 'seed': 8},
            {'name': 'TCU', 'seed': 9},
            {'name': 'Colorado State', 'seed': 10},
            {'name': 'NC State', 'seed': 11},
            {'name': 'James Madison', 'seed': 12},
            {'name': 'Yale', 'seed': 13},
            {'name': 'Morehead State', 'seed': 14},
            {'name': 'South Dakota State', 'seed': 15},
            {'name': 'Grambling', 'seed': 16},
        ],
        # West Region
        [
            {'name': 'North Carolina', 'seed': 1},
            {'name': 'Arizona', 'seed': 2},
            {'name': 'Baylor', 'seed': 3},
            {'name': 'Alabama', 'seed': 4},
            {'name': 'St Marys', 'seed': 5},
            {'name': 'Clemson', 'seed': 6},
            {'name': 'Dayton', 'seed': 7},
            {'name': 'Mississippi State', 'seed': 8},
            {'name': 'Michigan State', 'seed': 9},
            {'name': 'Nevada', 'seed': 10},
            {'name': 'New Mexico', 'seed': 11},
            {'name': 'Grand Canyon', 'seed': 12},
            {'name': 'Charleston', 'seed': 13},
            {'name': 'Colgate', 'seed': 14},
            {'name': 'Wagner', 'seed': 15},
            {'name': 'Howard', 'seed': 16},
        ],
    ]
    return regions


def main():
    parser = argparse.ArgumentParser(description='NCAA Tournament Bracket Simulator')
    parser.add_argument('--year', type=int, default=2025, help='Tournament year')
    parser.add_argument('--simulations', type=int, default=1000, help='Number of simulations')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic predictions')
    args = parser.parse_args()

    # Paths
    project_root = Path(__file__).parent.parent.parent
    model_path = project_root / 'models' / 'logistic_regression.pkl'
    scaler_path = project_root / 'models' / 'scaler.pkl'
    features_path = project_root / 'models' / 'features.pkl'
    # Use pretourney stats (leakage-free) if available
    stats_path = project_root / 'data' / 'raw' / f'sportsref_pretourney_{args.year}.csv'
    if not stats_path.exists():
        stats_path = project_root / 'data' / 'raw' / f'sportsref_combined_{args.year}.csv'
    if not stats_path.exists():
        print(f"Warning: Stats file not found for {args.year}")
        stats_path = project_root / 'data' / 'raw' / 'sportsref_pretourney_2024.csv'
        print(f"Using 2024 stats instead: {stats_path}")

    # Initialize simulator
    print("Loading model...")
    sim = BracketSimulator(str(model_path), str(scaler_path), str(features_path))

    print("Loading team stats...")
    sim.load_team_stats(str(stats_path))

    # Create bracket (would be loaded from file for real use)
    print("Creating bracket...")
    bracket = create_sample_bracket()

    if args.deterministic:
        print("\nGenerating deterministic bracket...")
        results = sim.generate_deterministic_bracket(bracket)
        sim.print_bracket(results)
    else:
        print(f"\nRunning {args.simulations} simulations...")
        sim_results = sim.run_simulations(bracket, n_simulations=args.simulations)
        sim.print_championship_odds(sim_results)

        # Also show deterministic bracket
        print("\n\nDeterministic Bracket Prediction:")
        results = sim.generate_deterministic_bracket(bracket)
        sim.print_bracket(results)


if __name__ == '__main__':
    main()
