"""Understat shot-level and minute-level game-state ingestor."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.config import RAW_DATA_DIR
from src.ingest.clubelo import canonicalize_team_name
from src.ingest.schemas import SHOTS_REQUIRED_COLS


class UnderstatClient:
    """Scrapes and caches Understat EPL match metadata and shot-level JSON payloads."""

    BASE_URL = "https://understat.com"

    def __init__(self, cache_dir: Path = RAW_DATA_DIR / "understat", timeout: int = 30) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; EPLBounceResearch/2.0)"})

    @staticmethod
    def _season_to_understat_year(season: str) -> str:
        return season.split("-")[0]

    @staticmethod
    def _extract_json_var(html: str, var_name: str) -> Any:
        pattern = rf"var\s+{re.escape(var_name)}\s*=\s*JSON\.parse\('([^']+)'\)"
        match = re.search(pattern, html)
        if not match:
            raise ValueError(f"Could not find Understat JSON variable '{var_name}'")
        decoded = match.group(1).encode("utf-8").decode("unicode_escape")
        return json.loads(decoded)

    def fetch_season_matches(self, season: str, force_refresh: bool = False) -> list[dict[str, Any]]:
        """Fetch league match index for an EPL season (e.g. '2022-23')."""
        year = self._season_to_understat_year(season)
        cache_path = self.cache_dir / f"epl_{year}_matches.json"
        if cache_path.exists() and not force_refresh:
            return json.loads(cache_path.read_text(encoding="utf-8"))

        url = f"{self.BASE_URL}/league/EPL/{year}"
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        data = self._extract_json_var(resp.text, "datesData")
        cache_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return data

    def fetch_match_shots(self, understat_match_id: str, force_refresh: bool = False) -> dict[str, Any]:
        """Fetch raw shot-level payload for a single Understat match ID."""
        cache_path = self.cache_dir / f"match_{understat_match_id}_shots.json"
        if cache_path.exists() and not force_refresh:
            return json.loads(cache_path.read_text(encoding="utf-8"))

        url = f"{self.BASE_URL}/match/{understat_match_id}"
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        shots_data = self._extract_json_var(resp.text, "shotsData")
        cache_path.write_text(json.dumps(shots_data, indent=2), encoding="utf-8")
        return shots_data

    @staticmethod
    def parse_match_shots(
        raw_shots: dict[str, list[dict[str, Any]]],
        match_id: str,
        season: str,
        home_team: str,
        away_team: str,
        red_cards: list[dict[str, Any]] | None = None,
    ) -> pd.DataFrame:
        """Convert Understat shot JSON into canonical shot rows with pre-shot score & man state."""
        home_c = canonicalize_team_name(home_team)
        away_c = canonicalize_team_name(away_team)
        rows: list[dict[str, Any]] = []

        for side_key, team_name in (("h", home_c), ("a", away_c)):
            for idx, shot in enumerate(raw_shots.get(side_key, [])):
                xg = float(np.clip(float(shot.get("xG", 0.0)), 0.0, 1.0))
                minute = float(shot.get("minute", 0.0))
                situation = str(shot.get("situation", "OpenPlay"))
                result = str(shot.get("result", ""))
                x_coord = float(shot.get("X", 0.85))
                y_coord = float(shot.get("Y", 0.5))
                # Understat normalized pitch (105m x 68m) distance to goal center (1.0, 0.5)
                dist_m = float(np.hypot((1.0 - x_coord) * 105.0, (0.5 - y_coord) * 68.0))
                rows.append(
                    {
                        "match_id": match_id,
                        "team_id": team_name,
                        "season": season,
                        "minute": minute,
                        "xg": xg,
                        "is_penalty": situation.lower() == "penalty",
                        "is_goal": result.lower() == "goal",
                        "is_own_goal": result.lower() == "owngoal",
                        "shot_distance": dist_m,
                        "_raw_idx": idx,
                    }
                )

        if not rows:
            return pd.DataFrame(columns=[*SHOTS_REQUIRED_COLS, "is_11v11"])

        df = pd.DataFrame(rows).sort_values(["minute", "_raw_idx"]).reset_index(drop=True)
        # Reconstruct running pre-shot score differential and red-card man differential
        red_events = sorted(red_cards or [], key=lambda e: float(e["minute"]))
        home_goals = 0
        away_goals = 0
        score_diffs = []
        man_diffs = []
        full_sides = []
        chain_ids = []
        last_team = None
        last_min = -99.0
        chain_counter = 0

        for _, row in df.iterrows():
            m = float(row["minute"])
            team = row["team_id"]
            home_reds = sum(1 for e in red_events if e["team_id"] == home_c and float(e["minute"]) < m)
            away_reds = sum(1 for e in red_events if e["team_id"] == away_c and float(e["minute"]) < m)

            full_sides.append(home_reds == away_reds == 0)
            if team == home_c:
                score_diffs.append(home_goals - away_goals)
                man_diffs.append((11 - home_reds) - (11 - away_reds))
            else:
                score_diffs.append(away_goals - home_goals)
                man_diffs.append((11 - away_reds) - (11 - home_reds))

            # Group rebound chains within 0.25 minutes (~15 seconds) by same team
            if team == last_team and (m - last_min) <= 0.25:
                chain_ids.append(f"{match_id}_{team}_{chain_counter}")
            else:
                chain_counter += 1
                chain_ids.append(f"{match_id}_{team}_{chain_counter}")
            last_team = None if row["is_goal"] or row["is_own_goal"] else team
            last_min = m

            if row["is_goal"]:
                if team == home_c:
                    home_goals += 1
                else:
                    away_goals += 1
            elif row["is_own_goal"]:
                if team == home_c:
                    away_goals += 1
                else:
                    home_goals += 1

        df["score_diff_pre_shot"] = score_diffs
        df["is_11v11"] = full_sides
        df["man_diff_pre_shot"] = man_diffs
        df["possession_chain_id"] = chain_ids
        return df.drop(columns=["_raw_idx", "is_own_goal"])


def compute_game_state_minutes(
    match_duration: float,
    home_team: str,
    away_team: str,
    goal_events: list[dict[str, Any]],
    red_card_events: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Integrate exact minute shares spent leading, trailing, tied, 11v11, and 11v11 |dG|<=1."""
    events: list[tuple[float, str, str]] = [(0.0, "start", ""), (float(match_duration), "end", "")]
    for g in goal_events:
        events.append((float(np.clip(g["minute"], 0.0, match_duration)), "goal", g["team_id"]))
    for r in red_card_events:
        events.append((float(np.clip(r["minute"], 0.0, match_duration)), "red", r["team_id"]))
    events.sort(key=lambda x: x[0])

    state = {
        home_team: {
            "min_11v11": 0.0,
            "min_11v11_comp": 0.0,
            "min_down_men": 0.0,
            "min_up_men": 0.0,
            "min_leading": 0.0,
            "min_trailing": 0.0,
            "min_tied": 0.0,
            "total_minutes": float(match_duration),
        },
        away_team: {
            "min_11v11": 0.0,
            "min_11v11_comp": 0.0,
            "min_down_men": 0.0,
            "min_up_men": 0.0,
            "min_leading": 0.0,
            "min_trailing": 0.0,
            "min_tied": 0.0,
            "total_minutes": float(match_duration),
        },
    }

    hg, ag, hr, ar = 0, 0, 0, 0
    prev_t = 0.0
    for t, kind, team in events:
        dt = max(0.0, t - prev_t)
        if dt > 0.0:
            is_11v11 = (hr == 0) and (ar == 0)
            diff_h = hg - ag
            man_diff_h = (11 - hr) - (11 - ar)

            for tm, sd, md in ((home_team, diff_h, man_diff_h), (away_team, -diff_h, -man_diff_h)):
                if is_11v11:
                    state[tm]["min_11v11"] += dt
                    if abs(sd) <= 1:
                        state[tm]["min_11v11_comp"] += dt
                if md < 0:
                    state[tm]["min_down_men"] += dt
                elif md > 0:
                    state[tm]["min_up_men"] += dt

                if sd > 0:
                    state[tm]["min_leading"] += dt
                elif sd < 0:
                    state[tm]["min_trailing"] += dt
                else:
                    state[tm]["min_tied"] += dt

        if kind == "goal":
            if team == home_team:
                hg += 1
            elif team == away_team:
                ag += 1
        elif kind == "red":
            if team == home_team:
                hr += 1
            elif team == away_team:
                ar += 1
        prev_t = t

    return state
