"""
ESPN API Data Collector.
Scrapes team stats, schedules, and scores from ESPN's public APIs.
"""

import requests
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
from tqdm import tqdm
import time


class ESPNCollector:
    """Collects NCAA basketball data from ESPN APIs."""

    BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"

    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_teams(self, year: int) -> pd.DataFrame:
        """Get list of all teams for a season."""
        url = f"{self.BASE_URL}/teams"
        params = {'season': year, 'limit': 400}

        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()

            teams = []
            sports_data = data.get('sports', [])
            if sports_data:
                leagues = sports_data[0].get('leagues', [])
                if leagues:
                    all_teams = leagues[0].get('teams', [])

                    for team_data in all_teams:
                        team = team_data.get('team', {})
                        teams.append({
                            'team_id': team.get('id'),
                            'team_name': team.get('displayName'),
                            'abbreviation': team.get('abbreviation'),
                            'season': year
                        })

            return pd.DataFrame(teams)

        except Exception as e:
            print(f"Error fetching teams for {year}: {e}")
            return pd.DataFrame()

    def get_team_stats(self, team_id: str, year: int) -> Optional[Dict]:
        """Get statistics for a specific team."""
        url = f"{self.BASE_URL}/teams/{team_id}/statistics"
        params = {'season': year}

        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()

            # Parse statistics
            stats = {
                'team_id': team_id,
                'season': year
            }

            # Extract stats from response
            results = data.get('results', {})
            stat_block = results.get('stats', {})
            categories = stat_block.get('categories', [])

            for category in categories:
                cat_name = category.get('name', '')
                cat_stats = category.get('stats', [])

                for stat in cat_stats:
                    stat_name = stat.get('name', '')
                    stat_value = stat.get('value')
                    stats[stat_name] = stat_value

            return stats

        except Exception as e:
            print(f"Error fetching stats for team {team_id} in {year}: {e}")
            return None

    def get_scoreboard(self, date: str) -> pd.DataFrame:
        """
        Get scoreboard for a specific date.

        Args:
            date: Date in YYYYMMDD format (e.g., '20240315')
        """
        url = f"{self.BASE_URL}/scoreboard"
        params = {'dates': date}

        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()

            games = []
            events = data.get('events', [])

            for event in events:
                game_id = event.get('id')
                competitions = event.get('competitions', [])

                if competitions:
                    comp = competitions[0]
                    competitors = comp.get('competitors', [])

                    if len(competitors) >= 2:
                        home_team = competitors[0] if competitors[0].get('homeAway') == 'home' else competitors[1]
                        away_team = competitors[1] if competitors[0].get('homeAway') == 'home' else competitors[0]

                        # Extract round info from notes
                        notes = comp.get('notes', [])
                        round_name = notes[0].get('headline', '') if notes else ''

                        games.append({
                            'game_id': game_id,
                            'date': date,
                            'home_team_id': home_team.get('id'),
                            'home_team_name': home_team.get('team', {}).get('displayName'),
                            'home_score': home_team.get('score'),
                            'home_seed': home_team.get('curatedRank', {}).get('current'),
                            'away_team_id': away_team.get('id'),
                            'away_team_name': away_team.get('team', {}).get('displayName'),
                            'away_score': away_team.get('score'),
                            'away_seed': away_team.get('curatedRank', {}).get('current'),
                            'round': round_name,
                            'tournament': comp.get('type', {}).get('abbreviation', ''),
                            'completed': comp.get('status', {}).get('type', {}).get('completed', False)
                        })

            return pd.DataFrame(games)

        except Exception as e:
            print(f"Error fetching scoreboard for {date}: {e}")
            return pd.DataFrame()

    def get_tournament_games(self, year: int, dates: List[str] = None) -> pd.DataFrame:
        """
        Get all NCAA tournament games across multiple days.

        Args:
            year: Tournament year (e.g., 2024)
            dates: List of dates in YYYYMMDD format. If None, scans
                   mid-March through early April to find tournament games.
        """
        if dates is None:
            # Generate dates covering the typical tournament window
            from datetime import date as dt_date, timedelta
            start = dt_date(year, 3, 14)
            end = dt_date(year, 4, 10)
            dates = []
            current = start
            while current <= end:
                dates.append(current.strftime('%Y%m%d'))
                current += timedelta(days=1)

        all_games = []
        for date in tqdm(dates, desc=f"Tournament {year}"):
            df = self.get_scoreboard(date)
            if not df.empty:
                # Filter to NCAA Championship games only (exclude conference tournaments)
                ncaa = df[
                    (df['tournament'] == 'TRNMNT') &
                    (df['round'].str.contains("Men's Basketball Championship", case=False, na=False))
                ]
                if not ncaa.empty:
                    all_games.append(ncaa)
            time.sleep(0.1)

        if not all_games:
            print(f"No tournament games found for {year}")
            return pd.DataFrame()

        combined = pd.concat(all_games, ignore_index=True)

        # Map ESPN round descriptions to short labels
        # Pre-2016 uses old naming: 1st ROUND=F4, 2nd ROUND=R64, 3rd ROUND=R32
        # 2016+ uses: First Four=F4, 1st Round=R64, 2nd Round=R32
        has_3rd_round = combined['round'].str.upper().str.contains('3RD ROUND', na=False).any()
        old_naming = has_3rd_round  # If "3rd Round" exists, it's the old convention

        def _map_round(desc):
            d = desc.upper()
            if 'FIRST FOUR' in d:
                return 'F4'
            elif '3RD ROUND' in d:
                return 'R32'
            elif '2ND ROUND' in d:
                return 'R64' if old_naming else 'R32'
            elif '1ST ROUND' in d:
                return 'F4' if old_naming else 'R64'
            elif 'SWEET 16' in d:
                return 'R16'
            elif 'ELITE 8' in d or 'ELITE EIGHT' in d:
                return 'R8'
            elif 'FINAL FOUR' in d or 'SEMIFINAL' in d:
                return 'R4'
            elif 'NATIONAL CHAMPIONSHIP' in d or d.endswith('FINAL'):
                return 'FINAL'
            return desc

        combined['round'] = combined['round'].apply(_map_round)

        # Drop columns we don't need and reorder so round is after date
        combined = combined.drop(columns=['tournament', 'completed', 'season'], errors='ignore')
        cols = list(combined.columns)
        cols.remove('round')
        date_idx = cols.index('date') + 1
        cols.insert(date_idx, 'round')
        combined = combined[cols]

        output_path = self.output_dir / f"espn_tournament_{year}.csv"
        combined.to_csv(output_path, index=False)
        print(f"Saved {len(combined)} tournament games to {output_path}")

        return combined

    def collect_season_teams(self, year: int) -> pd.DataFrame:
        """Collect all teams for a season."""
        print(f"Collecting teams for {year} season...")
        teams = self.get_teams(year)

        if not teams.empty:
            output_path = self.output_dir / f"teams_{year}.csv"
            teams.to_csv(output_path, index=False)
            print(f"Saved {len(teams)} teams to {output_path}")

        return teams

    def collect_season_stats(self, year: int) -> pd.DataFrame:
        """Collect stats for all teams in a season."""
        # First get teams
        teams = self.get_teams(year)

        if teams.empty:
            print(f"No teams found for {year}")
            return pd.DataFrame()

        print(f"Collecting stats for {len(teams)} teams in {year}...")
        all_stats = []

        for _, team in tqdm(teams.iterrows(), total=len(teams), desc=f"Stats {year}"):
            stats = self.get_team_stats(team['team_id'], year)
            if stats:
                stats['team_name'] = team['team_name']
                all_stats.append(stats)

            # Rate limiting
            time.sleep(0.1)

        df = pd.DataFrame(all_stats)

        if not df.empty:
            output_path = self.output_dir / f"espn_stats_{year}.csv"
            df.to_csv(output_path, index=False)
            print(f"Saved stats for {len(df)} teams to {output_path}")

        return df

    def collect_multiple_seasons(self, start_year: int, end_year: int) -> pd.DataFrame:
        """Collect data for multiple seasons."""
        all_stats = []

        for year in range(start_year, end_year + 1):
            stats = self.collect_season_stats(year)
            if not stats.empty:
                all_stats.append(stats)

            # Longer pause between seasons
            time.sleep(1)

        combined = pd.concat(all_stats, ignore_index=True) if all_stats else pd.DataFrame()

        if not combined.empty:
            output_path = self.output_dir / f"espn_stats_{start_year}_{end_year}.csv"
            combined.to_csv(output_path, index=False)
            print(f"\nTotal: {len(combined)} team-season records saved to {output_path}")

        return combined


def main():
    """Command-line interface for ESPN collector."""
    import argparse

    parser = argparse.ArgumentParser(description='Collect NCAA basketball data from ESPN')
    parser.add_argument('--years', type=str, default='2024',
                       help='Years to collect (e.g., "2024" or "2015-2024")')
    parser.add_argument('--output', type=str, default='data/raw',
                       help='Output directory')

    args = parser.parse_args()

    collector = ESPNCollector(output_dir=args.output)

    # Parse year range
    if '-' in args.years:
        start, end = map(int, args.years.split('-'))
        collector.collect_multiple_seasons(start, end)
    else:
        year = int(args.years)
        collector.collect_season_stats(year)


if __name__ == '__main__':
    main()
