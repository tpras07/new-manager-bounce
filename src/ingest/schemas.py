"""Canonical Parquet schemas, validators, and season-partitioned I/O utilities."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

import pandas as pd
import polars as pl

from src.config import PROCESSED_DATA_DIR, TACTICAL_FEATURES

MATCHES_REQUIRED_COLS: tuple[str, ...] = (
    "match_id",
    "season",
    "matchweek",
    "date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "home_elo_pre",
    "away_elo_pre",
)

SHOTS_REQUIRED_COLS: tuple[str, ...] = (
    "match_id",
    "team_id",
    "season",
    "minute",
    "xg",
    "is_penalty",
    "is_goal",
    "score_diff_pre_shot",
    "man_diff_pre_shot",
    "possession_chain_id",
    "shot_distance",
)

TEAM_MATCH_STATS_REQUIRED_COLS: tuple[str, ...] = (
    "match_id",
    "team_id",
    "opponent_id",
    "season",
    "matchweek",
    "date",
    "is_home",
    "goals_for",
    "goals_against",
    "points",
    "elo_pre",
    "opp_elo_pre",
    "xp",
    "npxgd_11v11",
    "min_11v11",
    "min_11v11_comp",
    "min_down_men",
    "min_up_men",
    "min_leading",
    "min_trailing",
    "min_tied",
    "total_minutes",
    "new_minutes_share",
    *TACTICAL_FEATURES,
)

MANAGER_EVENTS_REQUIRED_COLS: tuple[str, ...] = (
    "event_id",
    "team_id",
    "season",
    "old_manager",
    "new_manager",
    "date_sack",
    "date_perm_first_match",
    "matchweek_sack",
    "matchweek_perm",
    "interim_match_count",
    "is_january_window",
    "is_repeat_season_sacking",
)


def _check_required_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def validate_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Validate matches table invariants and return a clean copy."""
    _check_required_columns(df, MATCHES_REQUIRED_COLS, "matches")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    if out["match_id"].duplicated().any():
        raise ValueError("matches contains duplicate match_id values")
    if (out["home_team"] == out["away_team"]).any():
        raise ValueError("matches contains rows where home_team == away_team")
    if ((out["home_goals"] < 0) | (out["away_goals"] < 0)).any():
        raise ValueError("matches contains negative goal counts")
    return out


def validate_shots(df: pd.DataFrame) -> pd.DataFrame:
    """Validate shot-level records."""
    _check_required_columns(df, SHOTS_REQUIRED_COLS, "shots")
    out = df.copy()
    if not out["xg"].between(0, 1).all():
        raise ValueError("shots contains xG values outside [0, 1]")
    if (out["minute"] < 0).any():
        raise ValueError("shots contains negative minutes")
    return out


def validate_team_match_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Validate team-match observations."""
    _check_required_columns(df, TEAM_MATCH_STATS_REQUIRED_COLS, "team_match_stats")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    if out.duplicated(subset=["match_id", "team_id"]).any():
        raise ValueError("team_match_stats contains duplicate (match_id, team_id)")
    if out.duplicated(["season", "team_id", "matchweek"]).any():
        raise ValueError("A team has multiple matches at the same matchweek")
    if (out["total_minutes"] <= 0).any():
        raise ValueError("Match duration must be positive")
    if ((out["xp"] < 0.0) | (out["xp"] > 3.0001)).any():
        raise ValueError("team_match_stats contains xP outside [0, 3]")
    if ((out["new_minutes_share"] < 0.0) | (out["new_minutes_share"] > 1.0)).any():
        raise ValueError("team_match_stats contains new_minutes_share outside [0, 1]")
    return out


def validate_manager_events(df: pd.DataFrame) -> pd.DataFrame:
    """Validate managerial sacking and appointment event records."""
    _check_required_columns(df, MANAGER_EVENTS_REQUIRED_COLS, "manager_events")
    out = df.copy()
    out["date_sack"] = pd.to_datetime(out["date_sack"])
    out["date_perm_first_match"] = pd.to_datetime(out["date_perm_first_match"])
    if (out["date_perm_first_match"] < out["date_sack"]).any():
        raise ValueError("First permanent match precedes the dismissal")
    if out["event_id"].duplicated().any():
        raise ValueError("manager_events contains duplicate event_id values")
    if (out["interim_match_count"] < 0).any():
        raise ValueError("manager_events contains negative interim_match_count")
    return out


def read_canonical_table(table_name: str, base_dir: Path = PROCESSED_DATA_DIR) -> pd.DataFrame:
    target = base_dir / f"{table_name}.parquet"
    if not target.exists():
        raise FileNotFoundError(f"Missing canonical table: {target}")
    frame = pl.read_parquet(target, hive_partitioning=True)
    return pd.DataFrame(frame.to_dict(as_series=False))


def save_canonical_tables(matches, team_match_stats, shots, manager_events,
                          base_dir: Path = PROCESSED_DATA_DIR) -> dict[str, Path]:
    """Stage the complete dataset before replacing any existing tables."""
    tables = {"matches": validate_matches(matches), "team_match_stats": validate_team_match_stats(team_match_stats),
              "shots": validate_shots(shots), "manager_events": validate_manager_events(manager_events)}
    base_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: base_dir / f"{name}.parquet" for name in tables}
    with TemporaryDirectory(dir=base_dir) as directory:
        stage = Path(directory)
        for name, data in tables.items():
            frame = pl.DataFrame(data.to_dict("list"))
            target = stage / paths[name].name
            if name == "manager_events" or data.empty:
                frame.write_parquet(target)
            else:
                frame.write_parquet(target, partition_by="season")
        installed = []
        backups = {}
        try:
            for name, target in paths.items():
                if target.exists():
                    backups[target] = target.rename(stage / f"{name}.old")
                (stage / target.name).rename(target)
                installed.append(target)
        except OSError:
            for target in installed:
                target.rename(stage / (target.stem + ".new"))
            for target, backup in backups.items():
                backup.rename(target)
            raise
    return paths


def load_canonical_tables(base_dir: Path = PROCESSED_DATA_DIR) -> dict[str, pd.DataFrame]:
    """Load and validate all four canonical tables."""
    return {
        "matches": validate_matches(read_canonical_table("matches", base_dir)),
        "team_match_stats": validate_team_match_stats(
            read_canonical_table("team_match_stats", base_dir)
        ),
        "shots": validate_shots(read_canonical_table("shots", base_dir)),
        "manager_events": validate_manager_events(
            read_canonical_table("manager_events", base_dir)
        ),
    }
