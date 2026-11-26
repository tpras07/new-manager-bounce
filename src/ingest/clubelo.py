"""ClubElo client and club name canonicalization across FBref, Understat, and ClubElo."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from src.config import RAW_DATA_DIR

TEAM_NAME_ALIASES: dict[str, str] = {
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Bournemouth": "Bournemouth",
    "AFC Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton": "Brighton",
    "Brighton & Hove Albion": "Brighton",
    "Brighton and Hove Albion": "Brighton",
    "Burnley": "Burnley",
    "Cardiff": "Cardiff City",
    "Cardiff City": "Cardiff City",
    "Chelsea": "Chelsea",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Huddersfield": "Huddersfield Town",
    "Huddersfield Town": "Huddersfield Town",
    "Ipswich": "Ipswich Town",
    "Ipswich Town": "Ipswich Town",
    "Leeds": "Leeds United",
    "Leeds United": "Leeds United",
    "Leicester": "Leicester City",
    "Leicester City": "Leicester City",
    "Liverpool": "Liverpool",
    "Luton": "Luton Town",
    "Luton Town": "Luton Town",
    "Man City": "Manchester City",
    "Manchester City": "Manchester City",
    "Man United": "Manchester United",
    "Man Utd": "Manchester United",
    "Manchester United": "Manchester United",
    "Newcastle": "Newcastle United",
    "Newcastle Utd": "Newcastle United",
    "Newcastle United": "Newcastle United",
    "Norwich": "Norwich City",
    "Norwich City": "Norwich City",
    "Nott'm Forest": "Nottingham Forest",
    "Nottingham Forest": "Nottingham Forest",
    "Forest": "Nottingham Forest",
    "Sheffield Utd": "Sheffield United",
    "Sheffield United": "Sheffield United",
    "Southampton": "Southampton",
    "Stoke": "Stoke City",
    "Stoke City": "Stoke City",
    "Swansea": "Swansea City",
    "Swansea City": "Swansea City",
    "Tottenham": "Tottenham Hotspur",
    "Tottenham Hotspur": "Tottenham Hotspur",
    "Spurs": "Tottenham Hotspur",
    "Watford": "Watford",
    "West Brom": "West Bromwich Albion",
    "West Bromwich Albion": "West Bromwich Albion",
    "West Ham": "West Ham United",
    "West Ham United": "West Ham United",
    "Wolves": "Wolverhampton Wanderers",
    "Wolverhampton": "Wolverhampton Wanderers",
    "Wolverhampton Wanderers": "Wolverhampton Wanderers",
}

CLUBELO_SLUGS: dict[str, str] = {
    "Arsenal": "Arsenal",
    "Aston Villa": "AstonVilla",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton": "Brighton",
    "Burnley": "Burnley",
    "Cardiff City": "Cardiff",
    "Chelsea": "Chelsea",
    "Crystal Palace": "CrystalPalace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Huddersfield Town": "Huddersfield",
    "Ipswich Town": "Ipswich",
    "Leeds United": "Leeds",
    "Leicester City": "Leicester",
    "Liverpool": "Liverpool",
    "Luton Town": "Luton",
    "Manchester City": "ManCity",
    "Manchester United": "ManUnited",
    "Newcastle United": "Newcastle",
    "Norwich City": "Norwich",
    "Nottingham Forest": "Forest",
    "Sheffield United": "SheffieldUnited",
    "Southampton": "Southampton",
    "Stoke City": "Stoke",
    "Swansea City": "Swansea",
    "Tottenham Hotspur": "Tottenham",
    "Watford": "Watford",
    "West Bromwich Albion": "WestBrom",
    "West Ham United": "WestHam",
    "Wolverhampton Wanderers": "Wolves",
}


def canonicalize_team_name(name: str) -> str:
    """Map source-specific club names to canonical Premier League identifiers."""
    cleaned = name.strip()
    return TEAM_NAME_ALIASES.get(cleaned, cleaned)


class ClubEloClient:
    """Fetch and cache daily pre-match ClubElo ratings from api.clubelo.com."""

    BASE_URL = "http://api.clubelo.com"

    def __init__(self, cache_dir: Path = RAW_DATA_DIR / "clubelo", timeout: int = 30) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

    def fetch_club_history(self, team_name: str, force_refresh: bool = False) -> pd.DataFrame:
        """Fetch full daily Elo history for a canonical club and cache raw CSV."""
        canonical = canonicalize_team_name(team_name)
        slug = CLUBELO_SLUGS.get(canonical, canonical.replace(" ", ""))
        cache_path = self.cache_dir / f"{slug}.csv"

        if cache_path.exists() and not force_refresh:
            raw_text = cache_path.read_text(encoding="utf-8")
        else:
            resp = requests.get(f"{self.BASE_URL}/{slug}", timeout=self.timeout)
            resp.raise_for_status()
            raw_text = resp.text
            cache_path.write_text(raw_text, encoding="utf-8")

        df = pd.read_csv(StringIO(raw_text))
        df["From"] = pd.to_datetime(df["From"])
        df["To"] = pd.to_datetime(df["To"])
        df["team_id"] = canonical
        return df

    def lookup_pre_match_elo(
        self,
        team_name: str,
        match_date: str | pd.Timestamp,
        elo_history: pd.DataFrame | None = None,
    ) -> float:
        """Return pre-match Elo strictly prior to match_date (no post-match leakage)."""
        hist = elo_history if elo_history is not None else self.fetch_club_history(team_name)
        target_dt = pd.to_datetime(match_date) - pd.Timedelta(days=1)
        matched = hist[(hist["From"] <= target_dt) & (hist["To"] >= target_dt)]
        if matched.empty:
            prior = hist[hist["From"] <= target_dt].sort_values("From")
            if prior.empty:
                raise KeyError(f"No pre-match Elo found for {team_name} on {match_date}")
            return float(prior.iloc[-1]["Elo"])
        return float(matched.iloc[-1]["Elo"])
