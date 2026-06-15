"""
Core visualization utilities for NCAA tournament prediction model.
Provides reusable plotting functions for statistical analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class NCAAPlotter:
    """Visualization utilities for NCAA basketball data."""

    def __init__(self, output_dir: str = "results/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Color palettes
        self.conference_colors = self._get_conference_colors()

    def _get_conference_colors(self) -> Dict[str, str]:
        """Get color mapping for major conferences."""
        return {
            'ACC': '#003366',
            'Big Ten': '#990000',
            'Big 12': '#CC0000',
            'SEC': '#000080',
            'Pac-12': '#8C1515',
            'Big East': '#003DA5',
            'American': '#0033A0',
            'Mountain West': '#003865',
            'Atlantic 10': '#00205B',
            'WCC': '#4B2E84',
            'Other': '#808080'
        }

    def plot_top_n_leaders(self, df: pd.DataFrame, stat_col: str,
                          n: int = 10, season: Optional[int] = None,
                          title: Optional[str] = None,
                          save_path: Optional[str] = None,
                          show_tournament_seed: bool = True) -> plt.Figure:
        """
        Plot top N teams for a specific statistic.

        Args:
            df: DataFrame with team statistics
            stat_col: Column name of statistic to plot
            n: Number of top teams to show
            season: Season year (for filtering and title)
            title: Plot title (auto-generated if None)
            save_path: Path to save figure
            show_tournament_seed: Annotate with tournament seeds

        Returns:
            matplotlib Figure object
        """
        # Filter by season if specified
        plot_df = df.copy()
        if season and 'season' in df.columns:
            plot_df = plot_df[plot_df['season'] == season]

        # Sort and get top N
        plot_df = plot_df.nlargest(n, stat_col)

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))

        # Get colors by conference
        colors = [self.conference_colors.get(conf, self.conference_colors['Other'])
                 for conf in plot_df['conference']] if 'conference' in plot_df.columns else 'steelblue'

        # Create horizontal bar chart
        y_pos = np.arange(len(plot_df))
        bars = ax.barh(y_pos, plot_df[stat_col], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

        # Customize
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_df['team_name'] if 'team_name' in plot_df.columns else plot_df.index)
        ax.invert_yaxis()  # Top team at top

        # Labels
        stat_display = stat_col.replace('_', ' ').title()
        if title is None:
            title = f"Top {n} Teams by {stat_display}"
            if season:
                title += f" ({season} Season)"

        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel(stat_display, fontsize=12, fontweight='bold')

        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, plot_df[stat_col])):
            width = bar.get_width()
            label_text = f'{value:.1f}' if isinstance(value, float) else str(value)

            # Add seed annotation if available
            if show_tournament_seed and 'seed' in plot_df.columns:
                seed = plot_df.iloc[i]['seed']
                if pd.notna(seed):
                    label_text += f'  (Seed: {int(seed)})'

            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f'  {label_text}',
                   ha='left', va='center', fontweight='bold', fontsize=9)

        # Add legend for conferences if applicable
        if 'conference' in plot_df.columns:
            unique_confs = plot_df['conference'].unique()
            legend_elements = [plt.Rectangle((0,0),1,1, facecolor=self.conference_colors.get(conf, '#808080'),
                                            edgecolor='black', label=conf)
                             for conf in unique_confs if conf in self.conference_colors]
            ax.legend(handles=legend_elements, loc='lower right', title='Conference',
                     frameon=True, fancybox=True, shadow=True)

        plt.tight_layout()

        # Save if requested
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")

        return fig

    def plot_distribution(self, df: pd.DataFrame, stat_col: str,
                         bins: int = 30, season: Optional[int] = None,
                         title: Optional[str] = None,
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot distribution of a statistic.

        Args:
            df: DataFrame with statistics
            stat_col: Column to plot
            bins: Number of histogram bins
            season: Filter by season
            title: Plot title
            save_path: Path to save figure

        Returns:
            matplotlib Figure object
        """
        # Filter by season if specified
        plot_df = df.copy()
        if season and 'season' in df.columns:
            plot_df = plot_df[plot_df['season'] == season]

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot histogram and KDE
        data = plot_df[stat_col].dropna()
        ax.hist(data, bins=bins, alpha=0.7, color='steelblue', edgecolor='black', density=True)

        # Add KDE
        from scipy import stats
        kde = stats.gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 200)
        ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')

        # Add mean and median lines
        mean_val = data.mean()
        median_val = data.median()
        ax.axvline(mean_val, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')

        # Labels
        stat_display = stat_col.replace('_', ' ').title()
        if title is None:
            title = f"Distribution of {stat_display}"
            if season:
                title += f" ({season} Season)"

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(stat_display, fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.legend(frameon=True, fancybox=True, shadow=True)

        plt.tight_layout()

        # Save if requested
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")

        return fig

    def plot_correlation_heatmap(self, df: pd.DataFrame,
                                columns: Optional[List[str]] = None,
                                title: str = "Feature Correlation Heatmap",
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot correlation heatmap of features.

        Args:
            df: DataFrame with features
            columns: List of columns to include (all numeric if None)
            title: Plot title
            save_path: Path to save figure

        Returns:
            matplotlib Figure object
        """
        # Select columns
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        # Compute correlation
        corr = df[columns].corr()

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 12))

        # Plot heatmap
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                   ax=ax)

        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout()

        # Save if requested
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")

        return fig

    def plot_scatter_with_trend(self, df: pd.DataFrame, x_col: str, y_col: str,
                               hue_col: Optional[str] = None,
                               title: Optional[str] = None,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot scatter plot with trend line.

        Args:
            df: DataFrame
            x_col: X-axis column
            y_col: Y-axis column
            hue_col: Column for color coding
            title: Plot title
            save_path: Path to save figure

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        # Scatter plot
        if hue_col:
            for category in df[hue_col].unique():
                mask = df[hue_col] == category
                ax.scatter(df.loc[mask, x_col], df.loc[mask, y_col],
                          label=category, alpha=0.6, s=50)
        else:
            ax.scatter(df[x_col], df[y_col], alpha=0.6, s=50, color='steelblue')

        # Add trend line
        z = np.polyfit(df[x_col].dropna(), df[y_col].dropna(), 1)
        p = np.poly1d(z)
        ax.plot(df[x_col], p(df[x_col]), "r--", linewidth=2, label='Trend')

        # Labels
        x_display = x_col.replace('_', ' ').title()
        y_display = y_col.replace('_', ' ').title()

        if title is None:
            title = f"{y_display} vs {x_display}"

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(x_display, fontsize=12)
        ax.set_ylabel(y_display, fontsize=12)

        if hue_col or True:
            ax.legend(frameon=True, fancybox=True, shadow=True)

        plt.tight_layout()

        # Save if requested
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")

        return fig

    def plot_time_series(self, df: pd.DataFrame, stat_col: str,
                        group_col: Optional[str] = None,
                        title: Optional[str] = None,
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot time series of a statistic across seasons.

        Args:
            df: DataFrame with season column
            stat_col: Statistic to plot
            group_col: Column to group by (e.g., 'conference')
            title: Plot title
            save_path: Path to save figure

        Returns:
            matplotlib Figure object
        """
        if 'season' not in df.columns:
            raise ValueError("DataFrame must have 'season' column for time series plot")

        fig, ax = plt.subplots(figsize=(14, 7))

        if group_col:
            # Plot separate lines for each group
            for group in df[group_col].unique():
                group_df = df[df[group_col] == group]
                season_avg = group_df.groupby('season')[stat_col].mean()
                ax.plot(season_avg.index, season_avg.values, marker='o', label=group, linewidth=2)
        else:
            # Plot overall average
            season_avg = df.groupby('season')[stat_col].mean()
            ax.plot(season_avg.index, season_avg.values, marker='o', color='steelblue', linewidth=2)

        # Labels
        stat_display = stat_col.replace('_', ' ').title()

        if title is None:
            title = f"{stat_display} Over Time"
            if group_col:
                title += f" by {group_col.title()}"

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Season', fontsize=12)
        ax.set_ylabel(f'Average {stat_display}', fontsize=12)
        ax.grid(True, alpha=0.3)

        if group_col:
            ax.legend(frameon=True, fancybox=True, shadow=True)

        plt.tight_layout()

        # Save if requested
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")

        return fig
