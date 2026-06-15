"""
Statistical Leaders Visualization.
Generates leader charts for all major metrics by season.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.plotting import NCAAPlotter


class StatisticalLeadersVisualizer:
    """Generate statistical leader charts by season."""

    def __init__(self, data_path: str, output_dir: str = "results/visualizations/stat_leaders_by_season"):
        """
        Initialize visualizer.

        Args:
            data_path: Path to team profiles CSV
            output_dir: Output directory for charts
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plotter = NCAAPlotter(output_dir=str(self.output_dir))

        # Load data
        try:
            self.df = pd.read_csv(data_path)
            print(f"Loaded {len(self.df)} team-season records from {data_path}")
        except Exception as e:
            print(f"Error loading data: {e}")
            self.df = pd.DataFrame()

    def get_available_stats(self) -> List[str]:
        """Get list of available statistical columns."""
        # Exclude identifier columns
        exclude = ['season', 'team_id', 'team_name', 'conference', 'seed']
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        return [col for col in numeric_cols if col not in exclude]

    def generate_leader_chart(self, stat: str, season: int, n: int = 10) -> Optional[plt.Figure]:
        """
        Generate a leader chart for a specific stat and season.

        Args:
            stat: Statistic column name
            season: Season year
            n: Number of leaders to show

        Returns:
            matplotlib Figure or None if error
        """
        if stat not in self.df.columns:
            print(f"Warning: Statistic '{stat}' not found in data")
            return None

        # Filter season
        season_df = self.df[self.df['season'] == season].copy()

        if season_df.empty:
            print(f"Warning: No data for {season} season")
            return None

        # Generate filename
        filename = f"{stat}_leaders_{season}.png"
        save_path = self.output_dir / filename

        # Create plot
        stat_display = stat.replace('_', ' ').title()
        title = f"{stat_display} Leaders - {season} Season"

        try:
            fig = self.plotter.plot_top_n_leaders(
                df=season_df,
                stat_col=stat,
                n=n,
                title=title,
                save_path=str(save_path),
                show_tournament_seed=True
            )
            plt.close(fig)
            return fig

        except Exception as e:
            print(f"Error generating chart for {stat} ({season}): {e}")
            return None

    def generate_all_leader_charts(self, seasons: Optional[List[int]] = None,
                                   stats: Optional[List[str]] = None,
                                   n: int = 10) -> dict:
        """
        Generate leader charts for all stats and seasons.

        Args:
            seasons: List of seasons (all if None)
            stats: List of stats (all if None)
            n: Number of leaders per chart

        Returns:
            Dictionary with generation results
        """
        # Get seasons
        if seasons is None:
            seasons = sorted(self.df['season'].unique())

        # Get stats
        if stats is None:
            stats = self.get_key_statistics()

        print(f"\nGenerating leader charts for {len(stats)} stats across {len(seasons)} seasons...")
        print(f"Total charts to generate: {len(stats) * len(seasons)}")

        results = {
            'generated': 0,
            'failed': 0,
            'charts': []
        }

        for season in seasons:
            print(f"\n{season} Season:")
            for stat in stats:
                chart_file = self.generate_leader_chart(stat, season, n)
                if chart_file:
                    results['generated'] += 1
                    results['charts'].append(f"{stat}_{season}")
                else:
                    results['failed'] += 1

                # Print progress
                print(f"  ✓ {stat}", end='\r')

            print(f"  ✓ Completed {season} season ({len(stats)} charts)")

        print(f"\n{'='*60}")
        print(f"Summary:")
        print(f"  Successfully generated: {results['generated']} charts")
        print(f"  Failed: {results['failed']} charts")
        print(f"  Output directory: {self.output_dir}")
        print(f"{'='*60}\n")

        return results

    def get_key_statistics(self) -> List[str]:
        """Get list of key statistics to visualize."""
        key_stats = [
            # Scoring
            'points_per_game',
            'points_allowed',

            # Shooting
            'fg_pct',
            'fg3_pct',
            'ft_pct',
            'effective_fg_pct',

            # Advanced metrics
            'offensive_rating',
            'defensive_rating',
            'net_ranking',

            # Tempo and efficiency
            'tempo',
            'pace',
            'turnover_pct',
            'free_throw_rate',

            # Rebounding and assists
            'rebounds_per_game',
            'assists_per_game',
            'offensive_reb_pct',

            # Other
            'turnovers_per_game',
            'steals_per_game',
            'blocks_per_game',

            # Record
            'wins',
            'win_pct',
            'strength_schedule',
        ]

        # Return only stats that exist in the data
        return [stat for stat in key_stats if stat in self.df.columns]

    def generate_comparison_chart(self, stat: str, seasons: List[int],
                                 team_name: str) -> Optional[plt.Figure]:
        """
        Generate a chart comparing a team's stat across multiple seasons.

        Args:
            stat: Statistic to compare
            seasons: List of seasons
            team_name: Team name to track

        Returns:
            matplotlib Figure or None
        """
        # Filter data
        team_df = self.df[self.df['team_name'] == team_name].copy()
        team_df = team_df[team_df['season'].isin(seasons)]

        if team_df.empty:
            print(f"No data found for {team_name}")
            return None

        # Sort by season
        team_df = team_df.sort_values('season')

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(team_df['season'], team_df[stat], marker='o', linewidth=2,
               markersize=8, color='steelblue')

        # Labels
        stat_display = stat.replace('_', ' ').title()
        ax.set_title(f"{team_name} - {stat_display} Over Time",
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Season', fontsize=12)
        ax.set_ylabel(stat_display, fontsize=12)
        ax.grid(True, alpha=0.3)

        # Add value labels
        for _, row in team_df.iterrows():
            ax.text(row['season'], row[stat], f"{row[stat]:.1f}",
                   ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        # Save
        filename = f"{team_name.replace(' ', '_')}_{stat}_over_time.png"
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison chart to {save_path}")
        plt.close(fig)

        return fig


def main():
    """Command-line interface for statistical leaders visualizer."""
    import argparse

    parser = argparse.ArgumentParser(description='Generate NCAA statistical leader charts')
    parser.add_argument('--data', type=str, default='data/processed/team_profiles.csv',
                       help='Path to team profiles CSV')
    parser.add_argument('--seasons', type=str, default=None,
                       help='Seasons to generate (e.g., "2024" or "2015-2024")')
    parser.add_argument('--stats', type=str, nargs='+', default=None,
                       help='Specific stats to generate (all key stats if not specified)')
    parser.add_argument('--output', type=str, default='results/visualizations/stat_leaders_by_season',
                       help='Output directory')
    parser.add_argument('--top-n', type=int, default=10,
                       help='Number of leaders to show per chart')

    args = parser.parse_args()

    # Create visualizer
    visualizer = StatisticalLeadersVisualizer(
        data_path=args.data,
        output_dir=args.output
    )

    # Parse seasons
    seasons = None
    if args.seasons:
        if '-' in args.seasons:
            start, end = map(int, args.seasons.split('-'))
            seasons = list(range(start, end + 1))
        else:
            seasons = [int(args.seasons)]

    # Generate charts
    visualizer.generate_all_leader_charts(
        seasons=seasons,
        stats=args.stats,
        n=args.top_n
    )


if __name__ == '__main__':
    main()
