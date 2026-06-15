"""
Game Log Collector: Produces leakage-free pre-tournament stats.

Hybrid approach:
1. Barttorvik time machine - pre-tournament AdjOE/AdjDE/Pace snapshots
2. Sports Reference game logs - box score stats filtered to exclude NCAA tournament games

Output: sportsref_pretourney_{year}.csv with the same column format as sportsref_combined_{year}.csv
but with tournament game stats removed from all tournament teams.
"""

import pandas as pd
import numpy as np
import requests
import gzip
import json
import re
import time
import io
from pathlib import Path
from bs4 import BeautifulSoup
from typing import Optional

# Selection Sunday dates (day before we want the snapshot)
# We use these to fetch Barttorvik time machine data
SELECTION_SUNDAY_DATES = {
    2014: '20140316',
    2015: '20150315',
    2016: '20160313',
    2017: '20170312',
    2018: '20180311',
    2019: '20190317',
    2021: '20210314',
    2022: '20220313',
    2023: '20230312',
    2024: '20240317',
    2025: '20250316',
    2026: '20260315',
}

# ESPN name -> Sports Reference school name mapping
# Handles abbreviations, multi-word mascots, and naming differences
ESPN_TO_SR = {
    # Abbreviations and alternate names
    'UConn Huskies': 'Connecticut',
    'UConn': 'Connecticut',
    'UCF Knights': 'UCF',
    'UNLV Rebels': 'UNLV',
    'USC Trojans': 'Southern California',
    'LSU Tigers': 'Louisiana State',
    'SMU Mustangs': 'SMU',
    'VCU Rams': 'Virginia Commonwealth',
    'BYU Cougars': 'Brigham Young',
    'Ole Miss Rebels': 'Mississippi',
    'Pitt Panthers': 'Pittsburgh',
    'Miami Hurricanes': 'Miami (FL)',
    "Saint Mary's Gaels": "Saint Mary's (CA)",
    'NC State Wolfpack': 'NC State',
    'ETSU Buccaneers': 'East Tennessee State',
    'FDU Knights': 'Fairleigh Dickinson',
    'Fairleigh Dickinson Knights': 'Fairleigh Dickinson',
    'UAB Blazers': 'UAB',
    'UCSB Gauchos': 'UC Santa Barbara',
    'UNC Asheville Bulldogs': 'UNC Asheville',
    'UNC Wilmington Seahawks': 'UNC Wilmington',
    'UT Arlington Mavericks': 'UT Arlington',
    'App State Mountaineers': 'Appalachian State',

    # Multi-word mascots (would break simple last-word removal)
    'Alabama Crimson Tide': 'Alabama',
    'Duke Blue Devils': 'Duke',
    'Illinois Fighting Illini': 'Illinois',
    'Marquette Golden Eagles': 'Marquette',
    'Nevada Wolf Pack': 'Nevada',
    'North Carolina Tar Heels': 'North Carolina',
    'Oakland Golden Grizzlies': 'Oakland',
    'TCU Horned Frogs': 'TCU',
    'Texas Tech Red Raiders': 'Texas Tech',
    'Middle Tennessee Blue Raiders': 'Middle Tennessee',
    'Charleston Cougars': 'College of Charleston',
    'McNeese Cowboys': 'McNeese State',
    'Long Beach State Beach': 'Long Beach State',
    'South Dakota State Jackrabbits': 'South Dakota State',
    'North Dakota State Bison': 'North Dakota State',
    'Prairie View A&M Panthers': 'Prairie View',
    'North Carolina Central Eagles': 'North Carolina Central',
    'East Tennessee State Buccaneers': 'East Tennessee State',
    'Stephen F. Austin Lumberjacks': 'Stephen F. Austin',
    'Florida Gulf Coast Eagles': 'Florida Gulf Coast',
    'George Washington Revolutionaries': 'George Washington',

    # Multi-word school names with standard mascots
    'Texas A&M Aggies': 'Texas A&M',
    'Texas A&M-CC Islanders': 'Texas A&M-Corpus Christi',
    'Cal State Fullerton Titans': 'Cal State Fullerton',
    'South Florida Bulls': 'South Florida',
    'Western Kentucky Hilltoppers': 'Western Kentucky',
    'Loyola Chicago Ramblers': 'Loyola (IL)',
    'Loyola-Chicago Ramblers': 'Loyola (IL)',
    'Murray State Racers': 'Murray State',
    'Green Bay Phoenix': 'Green Bay',
    'Little Rock Trojans': 'Little Rock',
    'Northern Kentucky Norse': 'Northern Kentucky',
    'Jacksonville State Gamecocks': 'Jacksonville State',
    'Montana State Bobcats': 'Montana State',
    "Saint Peter's Peacocks": "Saint Peter's",
    'Gardner-Webb Runnin\' Bulldogs': 'Gardner-Webb',
    'UAlbany Great Danes': 'Albany (NY)',
    'Mount St. Mary\'s Mountaineers': "Mount St. Mary's",
    'Cal Poly Mustangs': 'Cal Poly',
    'UNC Greensboro Spartans': 'UNC Greensboro',
    'New Mexico State Aggies': 'New Mexico State',
    'Abilene Christian Wildcats': 'Abilene Christian',
    'UC Irvine Anteaters': 'UC Irvine',
    'UC Davis Aggies': 'UC Davis',
    'Kent State Golden Flashes': 'Kent State',

    # More multi-word mascots found across all years
    'Arizona State Sun Devils': 'Arizona State',
    'California Golden Bears': 'California',
    'Delaware Blue Hens': 'Delaware',
    'Georgia Tech Yellow Jackets': 'Georgia Tech',
    "Hawai'i Rainbow Warriors": "Hawaii",
    "Louisiana Ragin' Cajuns": 'Louisiana',
    'Marshall Thundering Herd': 'Marshall',
    'Minnesota Golden Gophers': 'Minnesota',
    'North Dakota Fighting Hawks': 'North Dakota',
    'North Texas Mean Green': 'North Texas',
    'Notre Dame Fighting Irish': 'Notre Dame',
    'Oral Roberts Golden Eagles': 'Oral Roberts',
    'Penn State Nittany Lions': 'Penn State',
    'Rutgers Scarlet Knights': 'Rutgers',
    "St. John's Red Storm": "St. John's (NY)",
    'Tulsa Golden Hurricane': 'Tulsa',
    'UMBC Retrievers': 'UMBC',
    'American University Eagles': 'American',
}

# Teams whose SR normalized name doesn't match their SR school stats page name
# (used for slug lookup when the standard slug extraction fails)
SR_NAME_TO_SLUG = {
    'UMBC': 'maryland-baltimore-county',
    'SMU': 'southern-methodist',
    'Fairleigh Dickinson': 'fairleigh-dickinson',
    'UCF': 'central-florida',
    'UNLV': 'nevada-las-vegas',
    'UAB': 'alabama-birmingham',
    'VCU': 'virginia-commonwealth',
    'UConn': 'connecticut',
    'LSU': 'louisiana-state',
    'BYU': 'brigham-young',
    "Saint Mary's (CA)": 'saint-marys-ca',
    "Saint Peter's": 'saint-peters',
    'Loyola (IL)': 'loyola-il',
    "St. John's (NY)": 'st-johns-ny',
    'Miami (FL)': 'miami-fl',
    'Albany (NY)': 'albany-ny',
    'NC State': 'north-carolina-state',
    'College of Charleston': 'college-of-charleston',
    'McNeese State': 'mcneese',
}


def normalize_espn_name(espn_name):
    """Convert ESPN full team name to Sports Reference school name."""
    if espn_name in ESPN_TO_SR:
        return ESPN_TO_SR[espn_name]
    # Remove mascot (last word)
    parts = espn_name.rsplit(' ', 1)
    if len(parts) == 2:
        return parts[0]
    return espn_name


class GameLogCollector:
    """Collects pre-tournament stats from Barttorvik + SR game logs."""

    SR_BASE = "https://www.sports-reference.com"
    BT_BASE = "https://barttorvik.com"
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    # NCAA tournament game types in SR game logs
    NCAAT_PREFIXES = ('ROUND-', 'NATIONAL-')

    # Barttorvik time machine column indices
    BT_COLS = {
        'team_name': 1,
        'record': 3,
        'adj_oe': 4,
        'adj_de': 6,
        'barthag': 8,
        'wins': 10,
        'losses': 11,
        'pace': 44,
    }

    def __init__(self, output_dir: str = "data/raw", cache_dir: str = "data/raw/gamelogs"):
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _fetch_with_cache(self, url: str, cache_path: Path, is_gzip: bool = False) -> str:
        """Fetch URL with local file caching and rate limiting."""
        # For gzip URLs, we cache as plain .json
        if is_gzip:
            json_cache = cache_path.with_suffix('.json')
            if json_cache.exists():
                return json_cache.read_text(encoding='utf-8')
        elif cache_path.exists():
            return cache_path.read_text(encoding='utf-8')

        time.sleep(3)
        response = requests.get(url, headers=self.HEADERS, timeout=30)

        if response.status_code == 429:
            print("    Rate limited, waiting 30s...")
            time.sleep(30)
            response = requests.get(url, headers=self.HEADERS, timeout=30)

        response.raise_for_status()

        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if is_gzip:
            # Try gzip decompression first, fall back to plain text
            # (server may return uncompressed despite .gz extension)
            try:
                decompressed = gzip.decompress(response.content).decode('utf-8')
            except (gzip.BadGzipFile, OSError):
                decompressed = response.content.decode('utf-8')
            # Cache as plain text for simplicity
            cache_path = cache_path.with_suffix('.json')
            cache_path.write_text(decompressed, encoding='utf-8')
            return decompressed
        else:
            cache_path.write_text(response.text, encoding='utf-8')
            return response.text

    # =========================================================================
    # Barttorvik time machine
    # =========================================================================

    def fetch_barttorvik_pretourney(self, year: int) -> dict:
        """
        Fetch pre-tournament ratings from Barttorvik time machine.

        Returns dict: {school_name_lower: {adj_oe, adj_de, barthag, pace, wins, losses}}
        """
        date_str = SELECTION_SUNDAY_DATES.get(year)
        if not date_str:
            print(f"  No Selection Sunday date for {year}, skipping Barttorvik")
            return {}

        url = f"{self.BT_BASE}/timemachine/team_results/{date_str}_team_results.json.gz"
        cache_path = self.cache_dir / "barttorvik" / f"{date_str}_team_results.json.gz"

        try:
            text = self._fetch_with_cache(url, cache_path, is_gzip=True)
            data = json.loads(text)
        except Exception as e:
            print(f"  Error fetching Barttorvik for {year}: {e}")
            return {}

        result = {}
        for team in data:
            name = str(team[self.BT_COLS['team_name']]).strip()
            result[name.lower()] = {
                'adj_oe': float(team[self.BT_COLS['adj_oe']]),
                'adj_de': float(team[self.BT_COLS['adj_de']]),
                'barthag': float(team[self.BT_COLS['barthag']]),
                'pace': float(team[self.BT_COLS['pace']]),
                'wins': float(team[self.BT_COLS['wins']]),
                'losses': float(team[self.BT_COLS['losses']]),
            }

        print(f"  Barttorvik: loaded {len(result)} teams for {date_str}")
        return result

    # =========================================================================
    # Sports Reference school slug extraction
    # =========================================================================

    def extract_school_slugs(self, year: int) -> dict:
        """
        Parse the season stats page to get school_name -> URL slug mapping.

        Returns: {'Connecticut': 'connecticut', 'Duke': 'duke', ...}
        """
        url = f"{self.SR_BASE}/cbb/seasons/men/{year}-school-stats.html"
        cache_path = self.cache_dir / "slugs" / f"school_stats_{year}.html"

        html = self._fetch_with_cache(url, cache_path)
        soup = BeautifulSoup(html, 'html.parser')

        slugs = {}
        # Find all links to school pages
        for link in soup.find_all('a', href=True):
            href = link['href']
            match = re.match(r'/cbb/schools/([^/]+)/men/\d+\.html', href)
            if match:
                slug = match.group(1)
                name = link.text.strip()
                # Clean NCAA markers
                name = re.sub(r'[\u2020\u00a0]|NCAA', '', name).strip()
                if name:
                    slugs[name] = slug

        print(f"  School slugs: found {len(slugs)} schools")
        return slugs

    # =========================================================================
    # SR game log parsing
    # =========================================================================

    def _parse_gamelog_html(self, html: str) -> pd.DataFrame:
        """Parse a SR game log page into a DataFrame."""
        # SR sometimes hides tables in HTML comments
        text = html
        comments = re.findall(r'<!--(.*?)-->', text, re.DOTALL)
        for comment in comments:
            if 'gamelog' in comment.lower() or 'sgl' in comment.lower():
                text = text.replace(f'<!--{comment}-->', comment)

        try:
            dfs = pd.read_html(io.StringIO(text))
        except ValueError:
            return pd.DataFrame()

        # Find the table with a 'Type' column (the game log)
        for df in dfs:
            # Flatten multi-level headers
            if isinstance(df.columns, pd.MultiIndex):
                new_cols = []
                for col in df.columns:
                    parts = [str(c) for c in col
                             if 'Unnamed' not in str(c) and 'level' not in str(c)]
                    new_cols.append('_'.join(parts) if parts else str(col[-1]))
                df.columns = new_cols

            col_strs = [str(c) for c in df.columns]
            if any('Type' in c for c in col_strs):
                # Clean up separator rows
                if 'Rk' in df.columns:
                    df = df[df['Rk'] != 'Rk']
                    df = df.dropna(subset=['Rk'])
                return df

        # Fallback: return largest table
        if dfs:
            return max(dfs, key=len)
        return pd.DataFrame()

    def fetch_basic_gamelog(self, slug: str, year: int) -> pd.DataFrame:
        """Fetch and parse basic game log for a team."""
        url = f"{self.SR_BASE}/cbb/schools/{slug}/men/{year}-gamelogs.html"
        cache_path = self.cache_dir / "basic" / f"{slug}_{year}.html"

        try:
            html = self._fetch_with_cache(url, cache_path)
            return self._parse_gamelog_html(html)
        except Exception as e:
            print(f"    Error fetching basic gamelog for {slug} {year}: {e}")
            return pd.DataFrame()

    def fetch_advanced_gamelog(self, slug: str, year: int) -> pd.DataFrame:
        """Fetch and parse advanced game log for a team."""
        url = f"{self.SR_BASE}/cbb/schools/{slug}/men/{year}-gamelogs-advanced.html"
        cache_path = self.cache_dir / "advanced" / f"{slug}_{year}.html"

        try:
            html = self._fetch_with_cache(url, cache_path)
            return self._parse_gamelog_html(html)
        except Exception as e:
            print(f"    Error fetching advanced gamelog for {slug} {year}: {e}")
            return pd.DataFrame()

    def _filter_pre_tournament(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out NCAA tournament games from a game log DataFrame."""
        if df.empty:
            return df

        # Find the Type column (might have different name due to header flattening)
        type_col = None
        for col in df.columns:
            if 'Type' in str(col):
                type_col = col
                break

        if type_col is None:
            print("    WARNING: No 'Type' column found, returning all games")
            return df

        # Keep only non-NCAA-tournament games
        mask = ~df[type_col].astype(str).str.startswith(self.NCAAT_PREFIXES)
        filtered = df[mask].copy()
        return filtered

    def compute_basic_stats(self, basic_df: pd.DataFrame) -> dict:
        """
        Compute pre-tournament basic stats from filtered game log.

        Returns dict with keys matching sportsref_combined column names.
        """
        if basic_df.empty:
            return {}

        pre = self._filter_pre_tournament(basic_df)
        if pre.empty:
            return {}

        def to_num(series):
            return pd.to_numeric(series, errors='coerce')

        def get_col(prefix, stat):
            """Find column by prefix_stat first, then bare stat as fallback."""
            prefixed = f'{prefix}_{stat}'
            for c in pre.columns:
                if str(c) == prefixed:
                    return str(c)
            # Fallback to bare name
            for c in pre.columns:
                if str(c) == stat:
                    return str(c)
            return None

        n_games = len(pre)
        n_wins = 0
        n_losses = 0

        # Find result column (Score_Rslt or Rslt)
        rslt_col = None
        for col in pre.columns:
            if 'Rslt' in str(col):
                rslt_col = str(col)
                break

        if rslt_col:
            for val in pre[rslt_col].astype(str):
                if val.startswith('W'):
                    n_wins += 1
                elif val.startswith('L'):
                    n_losses += 1
        else:
            n_wins = n_games // 2
            n_losses = n_games - n_wins

        win_pct = n_wins / n_games if n_games > 0 else 0

        stats = {}

        # Points: Score_Tm and Score_Opp
        tm_col = get_col('Score', 'Tm')
        opp_col = get_col('Score', 'Opp')
        if tm_col:
            stats['Points_Tm.'] = to_num(pre[tm_col]).sum()
        if opp_col:
            stats['Points_Opp.'] = to_num(pre[opp_col]).sum()

        # Team shooting stats: Team_FG, Team_FGA, etc.
        stat_names = ['FG', 'FGA', '3P', '3PA', 'FT', 'FTA', 'ORB', 'DRB', 'TRB',
                      'AST', 'STL', 'BLK', 'TOV', 'PF']

        for stat in stat_names:
            col = get_col('Team', stat)
            if col:
                stats[f'Totals_{stat}'] = to_num(pre[col]).sum()

        # Compute percentages from totals
        fg = stats.get('Totals_FG', 0)
        fga = stats.get('Totals_FGA', 0)
        stats['Totals_FG%'] = fg / fga if fga > 0 else 0

        tp = stats.get('Totals_3P', 0)
        tpa = stats.get('Totals_3PA', 0)
        stats['Totals_3P%'] = tp / tpa if tpa > 0 else 0

        ft = stats.get('Totals_FT', 0)
        fta = stats.get('Totals_FTA', 0)
        stats['Totals_FT%'] = ft / fta if fta > 0 else 0

        # Record
        stats['Overall_G'] = n_games
        stats['Overall_W'] = n_wins
        stats['Overall_L'] = n_losses
        stats['Overall_W-L%'] = win_pct

        return stats

    def compute_advanced_stats(self, advanced_df: pd.DataFrame) -> dict:
        """
        Compute pre-tournament advanced stats from filtered game log.

        Returns dict with keys matching sportsref_combined column names.
        """
        if advanced_df.empty:
            return {}

        pre = self._filter_pre_tournament(advanced_df)
        if pre.empty:
            return {}

        def to_num(series):
            return pd.to_numeric(series, errors='coerce')

        # Map advanced game log column names -> sportsref_combined column names
        # Actual game log columns after header flattening:
        #   Advanced_ORtg, Advanced_Pace, Advanced_FTr, Advanced_3PAr,
        #   Advanced_TS%, Advanced_TRB%, Advanced_AST%, Advanced_STL%, Advanced_BLK%,
        #   Offensive Four Factors_eFG%, Offensive Four Factors_TOV%,
        #   Offensive Four Factors_ORB%, Offensive Four Factors_FT/FGA
        col_mapping = {
            'Advanced_ORtg': 'School Advanced_ORtg',
            'Advanced_Pace': 'School Advanced_Pace',
            'Advanced_FTr': 'School Advanced_FTr',
            'Advanced_3PAr': 'School Advanced_3PAr',
            'Advanced_TS%': 'School Advanced_TS%',
            'Advanced_TRB%': 'School Advanced_TRB%',
            'Advanced_AST%': 'School Advanced_AST%',
            'Advanced_STL%': 'School Advanced_STL%',
            'Advanced_BLK%': 'School Advanced_BLK%',
            'Offensive Four Factors_eFG%': 'School Advanced_eFG%',
            'Offensive Four Factors_TOV%': 'School Advanced_TOV%',
            'Offensive Four Factors_ORB%': 'School Advanced_ORB%',
            'Offensive Four Factors_FT/FGA': 'School Advanced_FT/FGA',
        }

        stats = {}
        for src_name, dst_name in col_mapping.items():
            if src_name in pre.columns:
                vals = to_num(pre[src_name]).dropna()
                if len(vals) > 0:
                    stats[dst_name] = vals.mean()

        return stats

    # =========================================================================
    # Tournament team identification
    # =========================================================================

    def get_tournament_teams(self, year: int) -> set:
        """Get set of SR school names that played in the NCAA tournament."""
        tourney_path = self.output_dir / f'espn_tournament_{year}.csv'
        if not tourney_path.exists():
            print(f"  WARNING: No tournament file for {year}")
            return set()

        tourney = pd.read_csv(tourney_path)
        teams = set()

        for _, game in tourney.iterrows():
            # Skip First Four
            if game['round'] == 'F4':
                continue
            teams.add(normalize_espn_name(str(game['home_team_name'])))
            teams.add(normalize_espn_name(str(game['away_team_name'])))

        return teams

    # =========================================================================
    # Main orchestration
    # =========================================================================

    def collect_pretourney_stats(self, year: int) -> pd.DataFrame:
        """
        Produce a leakage-free version of sportsref_combined_{year}.csv.

        For each tournament team:
        - Box score stats from SR game logs (filtered, no NCAA tourney games)
        - AdjOE/AdjDE from Barttorvik time machine (pre-tournament snapshot)

        Non-tournament teams are left unchanged.
        """
        print(f"\n{'='*60}")
        print(f"Collecting pre-tournament stats for {year}")
        print(f"{'='*60}")

        # Load existing combined stats as base
        combined_path = self.output_dir / f'sportsref_combined_{year}.csv'
        if not combined_path.exists():
            print(f"  ERROR: {combined_path} not found")
            return pd.DataFrame()

        df = pd.read_csv(combined_path)
        print(f"  Loaded {len(df)} teams from {combined_path}")

        # Get Barttorvik pre-tournament ratings
        bt_data = self.fetch_barttorvik_pretourney(year)

        # Get school slugs
        slugs = self.extract_school_slugs(year)

        # Get tournament teams
        tourney_teams = self.get_tournament_teams(year)
        print(f"  Tournament teams: {len(tourney_teams)}")

        # Process each tournament team
        adjusted = 0
        failed = []

        for team_name in sorted(tourney_teams):
            # Find this team in the combined DataFrame
            team_idx = df.index[df['School'] == team_name].tolist()
            if not team_idx:
                # Try case-insensitive match
                team_idx = df.index[df['School'].str.lower() == team_name.lower()].tolist()
            if not team_idx:
                # Team not in combined CSV - create a new row
                print(f"  {team_name} not in combined CSV, creating new row")
                new_row = pd.Series(dtype='object')
                new_row['School'] = team_name
                new_row['season'] = year
                df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
                team_idx = [len(df) - 1]

            idx = team_idx[0]

            # Find the URL slug
            slug = slugs.get(team_name)
            if not slug:
                # Try case-insensitive
                for sname, sslug in slugs.items():
                    if sname.lower() == team_name.lower():
                        slug = sslug
                        break
            if not slug:
                # Try manual slug mapping
                slug = SR_NAME_TO_SLUG.get(team_name)
            if not slug:
                failed.append(f"{team_name} (no URL slug)")
                continue

            print(f"  Processing {team_name} ({slug})...")

            # Fetch and compute basic stats from game log
            basic_gl = self.fetch_basic_gamelog(slug, year)
            basic_stats = self.compute_basic_stats(basic_gl)

            # Fetch and compute advanced stats from game log
            adv_gl = self.fetch_advanced_gamelog(slug, year)
            adv_stats = self.compute_advanced_stats(adv_gl)

            if not basic_stats:
                failed.append(f"{team_name} (basic gamelog failed)")
                continue

            # Update the row with pre-tournament stats
            for col, val in basic_stats.items():
                if col in df.columns:
                    df.at[idx, col] = val

            for col, val in adv_stats.items():
                if col in df.columns:
                    df.at[idx, col] = val

            # Update SRS/SOS with Barttorvik AdjOE/AdjDE
            bt_entry = bt_data.get(team_name.lower())
            if bt_entry:
                df.at[idx, 'Overall_SRS'] = bt_entry['adj_oe'] - bt_entry['adj_de']
                df.at[idx, 'Overall_SOS'] = (bt_entry['adj_oe'] + bt_entry['adj_de']) / 2 - 100
            else:
                # Try fuzzy match on barttorvik data
                for bt_name, bt_entry in bt_data.items():
                    if team_name.lower() in bt_name or bt_name in team_name.lower():
                        df.at[idx, 'Overall_SRS'] = bt_entry['adj_oe'] - bt_entry['adj_de']
                        df.at[idx, 'Overall_SOS'] = (bt_entry['adj_oe'] + bt_entry['adj_de']) / 2 - 100
                        break

            adjusted += 1

        print(f"\n  Adjusted {adjusted}/{len(tourney_teams)} tournament teams")
        if failed:
            print(f"  Failed ({len(failed)}):")
            for f in failed:
                print(f"    - {f}")

        # Save
        output_path = self.output_dir / f'sportsref_pretourney_{year}.csv'
        df.to_csv(output_path, index=False)
        print(f"  Saved to {output_path}")

        return df

    def collect_multiple_years(self, years: list) -> None:
        """Collect pre-tournament stats for multiple years."""
        for year in years:
            try:
                self.collect_pretourney_stats(year)
            except Exception as e:
                print(f"  ERROR for {year}: {e}")
                import traceback
                traceback.print_exc()


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Collect leakage-free pre-tournament stats'
    )
    parser.add_argument('--years', type=str, default='2024',
                        help='Years to process (e.g., "2024" or "2014-2024")')
    parser.add_argument('--output', type=str, default='data/raw',
                        help='Output directory')
    parser.add_argument('--cache', type=str, default='data/raw/gamelogs',
                        help='Cache directory for downloaded HTML')

    args = parser.parse_args()

    collector = GameLogCollector(output_dir=args.output, cache_dir=args.cache)

    if '-' in args.years:
        start, end = map(int, args.years.split('-'))
        all_years = [y for y in range(start, end + 1) if y != 2020]
        collector.collect_multiple_years(all_years)
    else:
        year = int(args.years)
        collector.collect_pretourney_stats(year)


if __name__ == '__main__':
    main()
