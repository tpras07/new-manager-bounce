"""Managerial tenure registry, Transfermarkt scraper, and NewMinutes covariate builder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.config import RAW_DATA_DIR
from src.ingest.clubelo import canonicalize_team_name


class TransfermarktTenureClient:
    """Scrapes Transfermarkt managerial change tables and computes NewMinutes shares."""

    BASE_URL = "https://www.transfermarkt.com"

    def __init__(self, cache_dir: Path = RAW_DATA_DIR / "managers", timeout: int = 30) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
                )
            }
        )

    def fetch_season_changes(self, season: str, force_refresh: bool = False) -> pd.DataFrame:
        """Fetch mid-season EPL managerial changes for a given season string (e.g. '2022-23')."""
        start_year = season.split("-")[0]
        cache_path = self.cache_dir / f"tm_epl_{start_year}.html"
        if cache_path.exists() and not force_refresh:
            html = cache_path.read_text(encoding="utf-8")
        else:
            url = f"{self.BASE_URL}/premier-league/trainerwechsel/wettbewerb/GB1/saison_id/{start_year}"
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            html = resp.text
            cache_path.write_text(html, encoding="utf-8")
        return self.parse_trainerwechsel_html(html, season)

    @staticmethod
    def parse_trainerwechsel_html(html: str, season: str) -> pd.DataFrame:
        """Parse Transfermarkt trainerwechsel table into standardized manager event rows."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "lxml")
        table = soup.find("table", class_="items")
        if table is None:
            return pd.DataFrame()
        records: list[dict[str, Any]] = []
        for tr in table.select("tbody > tr"):
            cols = tr.find_all("td", recursive=False)
            if len(cols) < 5:
                continue
            club_link = cols[0].find("a", title=True)
            team_name = canonicalize_team_name(club_link["title"]) if club_link else cols[0].get_text(strip=True)
            records.append(
                {
                    "team_id": team_name,
                    "season": season,
                    "old_manager": cols[1].get_text(" ", strip=True),
                    "date_sack": pd.to_datetime(cols[2].get_text(strip=True), errors="coerce"),
                    "new_manager": cols[3].get_text(" ", strip=True),
                    "date_perm_first_match": pd.to_datetime(cols[4].get_text(strip=True), errors="coerce"),
                }
            )
        return pd.DataFrame(records)


def compute_new_minutes_share(
    player_match_minutes: pd.DataFrame,
    manager_events: pd.DataFrame,
) -> pd.DataFrame:
    """Compute match-level NewMinutes_{i,t} in [0, 1] for players registered after T_perm."""
    req = {"match_id", "team_id", "season", "match_date", "player_id", "registration_date", "minutes"}
    missing = req - set(player_match_minutes.columns)
    if missing:
        raise ValueError(f"player_match_minutes missing columns: {missing}")

    pm = player_match_minutes.copy()
    pm["match_date"] = pd.to_datetime(pm["match_date"])
    pm["registration_date"] = pd.to_datetime(pm["registration_date"])

    keys = ["team_id", "season"]
    appointments = manager_events[[*keys, "date_perm_first_match"]].copy()
    appointments["date_perm_first_match"] = pd.to_datetime(appointments["date_perm_first_match"])
    merged = pd.merge_asof(
        pm.sort_values("match_date"), appointments.sort_values("date_perm_first_match"),
        left_on="match_date", right_on="date_perm_first_match", by=keys,
    )
    is_post_reg = (merged["registration_date"] > merged["date_perm_first_match"]) & (
        merged["registration_date"] <= merged["match_date"]
    )
    merged["new_minutes"] = merged["minutes"].where(is_post_reg, 0.0)
    agg = (
        merged.groupby(["match_id", "team_id"], as_index=False)
        .agg(total_player_minutes=("minutes", "sum"), new_player_minutes=("new_minutes", "sum"))
    )
    agg["new_minutes_share"] = (
        agg["new_player_minutes"] / agg["total_player_minutes"].clip(lower=1.0)
    ).clip(0.0, 1.0)
    return agg[["match_id", "team_id", "new_minutes_share"]]
