"""
Sports-Reference Data Collector.
Scrapes comprehensive historical data directly from Sports-Reference HTML tables.
"""

import pandas as pd
from io import StringIO
from pathlib import Path
from tqdm import tqdm
import requests
import time
from typing import Optional


class SportsRefCollector:
    """Collects NCAA basketball data from Sports-Reference."""

    BASE_URL = "https://www.sports-reference.com/cbb/seasons/men"
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _fetch_table(self, url: str, table_id: str) -> Optional[pd.DataFrame]:
        """Fetch and parse an HTML table from Sports-Reference."""
        try:
            response = requests.get(url, headers=self.HEADERS, timeout=30)
            response.raise_for_status()

            dfs = pd.read_html(StringIO(response.text), attrs={'id': table_id})
            if not dfs:
                return None

            df = dfs[0]

            # Flatten multi-level column headers
            if isinstance(df.columns, pd.MultiIndex):
                new_cols = []
                for col in df.columns:
                    parts = [str(c) for c in col if 'Unnamed' not in str(c) and 'level' not in str(c)]
                    new_cols.append('_'.join(parts) if parts else str(col[-1]))
                df.columns = new_cols

            # Drop separator rows (where Rk == 'Rk' or is NaN)
            if 'Rk' in df.columns:
                df = df[df['Rk'] != 'Rk']
                df = df.dropna(subset=['Rk'])
                df = df.drop(columns=['Rk'])

            # Drop empty separator columns (all NaN)
            df = df.dropna(axis=1, how='all')
            # Also drop any columns that still have 'Unnamed' in the name
            df = df[[c for c in df.columns if 'Unnamed' not in str(c)]]

            # Clean school names (remove NCAA tournament markers like †)
            if 'School' in df.columns:
                df['School'] = df['School'].str.replace(r'[\u2020\u00a0]|NCAA', '', regex=True).str.strip()

            return df

        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None

    def collect_basic_stats(self, year: int) -> pd.DataFrame:
        """
        Collect basic school stats for a season.

        Args:
            year: Season year (e.g., 2024 for the 2023-24 season)
        """
        url = f"{self.BASE_URL}/{year}-school-stats.html"
        print(f"Collecting basic stats for {year}...")

        df = self._fetch_table(url, 'basic_school_stats')
        if df is None:
            print(f"No basic stats found for {year}")
            return pd.DataFrame()

        df['season'] = year

        output_path = self.output_dir / f"sportsref_basic_{year}.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} teams to {output_path}")

        return df

    def collect_advanced_stats(self, year: int) -> pd.DataFrame:
        """
        Collect advanced school stats for a season.

        Args:
            year: Season year (e.g., 2024 for the 2023-24 season)
        """
        url = f"{self.BASE_URL}/{year}-advanced-school-stats.html"
        print(f"Collecting advanced stats for {year}...")

        df = self._fetch_table(url, 'adv_school_stats')
        if df is None:
            print(f"No advanced stats found for {year}")
            return pd.DataFrame()

        df['season'] = year

        output_path = self.output_dir / f"sportsref_advanced_{year}.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} teams to {output_path}")

        return df

    def collect_season(self, year: int) -> dict:
        """
        Collect both basic and advanced stats for a season, merged into one DataFrame.

        Args:
            year: Season year

        Returns:
            Dictionary with 'basic', 'advanced', and 'combined' DataFrames.
        """
        basic = self.collect_basic_stats(year)
        time.sleep(2)
        advanced = self.collect_advanced_stats(year)

        result = {'basic': basic, 'advanced': advanced}

        if not basic.empty and not advanced.empty:
            # Advanced duplicates the record columns; keep only the new advanced columns
            adv_only_cols = [c for c in advanced.columns
                            if c not in basic.columns or c == 'School']
            advanced_slim = advanced[adv_only_cols]

            combined = basic.merge(advanced_slim, on='School', how='left')

            output_path = self.output_dir / f"sportsref_combined_{year}.csv"
            combined.to_csv(output_path, index=False)
            print(f"Saved {len(combined)} combined records to {output_path}")

            result['combined'] = combined

        return result

    def collect_multiple_seasons(self, start_year: int, end_year: int) -> dict:
        """
        Collect data for multiple seasons.

        Args:
            start_year: First season year
            end_year: Last season year

        Returns:
            Dictionary with 'basic', 'advanced', and 'combined' DataFrames.
        """
        all_basic = []
        all_advanced = []
        all_combined = []

        for year in range(start_year, end_year + 1):
            print(f"\n{'='*60}")
            print(f"Collecting data for {year} season")
            print(f"{'='*60}")

            data = self.collect_season(year)

            if not data.get('basic', pd.DataFrame()).empty:
                all_basic.append(data['basic'])
            if not data.get('advanced', pd.DataFrame()).empty:
                all_advanced.append(data['advanced'])
            if 'combined' in data and not data['combined'].empty:
                all_combined.append(data['combined'])

            # Rate limit between seasons
            time.sleep(3)

        result = {}

        if all_basic:
            combined_basic = pd.concat(all_basic, ignore_index=True)
            output_path = self.output_dir / f"sportsref_basic_{start_year}_{end_year}.csv"
            combined_basic.to_csv(output_path, index=False)
            print(f"\nTotal basic: {len(combined_basic)} team-seasons saved to {output_path}")
            result['basic'] = combined_basic

        if all_advanced:
            combined_adv = pd.concat(all_advanced, ignore_index=True)
            output_path = self.output_dir / f"sportsref_advanced_{start_year}_{end_year}.csv"
            combined_adv.to_csv(output_path, index=False)
            print(f"Total advanced: {len(combined_adv)} team-seasons saved to {output_path}")
            result['advanced'] = combined_adv

        if all_combined:
            combined_all = pd.concat(all_combined, ignore_index=True)
            output_path = self.output_dir / f"sportsref_combined_{start_year}_{end_year}.csv"
            combined_all.to_csv(output_path, index=False)
            print(f"Total combined: {len(combined_all)} team-seasons saved to {output_path}")
            result['combined'] = combined_all

        return result


def main():
    """Command-line interface for Sports-Reference collector."""
    import argparse

    parser = argparse.ArgumentParser(description='Collect NCAA basketball data from Sports-Reference')
    parser.add_argument('--years', type=str, default='2024',
                       help='Years to collect (e.g., "2024" or "2015-2024")')
    parser.add_argument('--output', type=str, default='data/raw',
                       help='Output directory')
    parser.add_argument('--basic-only', action='store_true',
                       help='Only collect basic stats (skip advanced)')

    args = parser.parse_args()

    collector = SportsRefCollector(output_dir=args.output)

    # Parse year range
    if '-' in args.years:
        start, end = map(int, args.years.split('-'))
        if args.basic_only:
            for year in range(start, end + 1):
                collector.collect_basic_stats(year)
                time.sleep(3)
        else:
            collector.collect_multiple_seasons(start, end)
    else:
        year = int(args.years)
        if args.basic_only:
            collector.collect_basic_stats(year)
        else:
            collector.collect_season(year)


if __name__ == '__main__':
    main()
