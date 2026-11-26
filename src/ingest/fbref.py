"""FBref match-level tactical aggregates ingestor and feature builder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from src.config import RAW_DATA_DIR, TACTICAL_FEATURES
from src.ingest.clubelo import canonicalize_team_name


class FBrefClient:
    """Fetches and caches FBref match logs and computes the 14 tactical feature inputs."""

    def __init__(self, cache_dir: Path = RAW_DATA_DIR / "fbref", timeout: int = 30) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

    def fetch_season_team_logs(self, season: str, force_refresh: bool = False) -> pd.DataFrame:
        """Read cached EPL match logs or fetch them through soccerdata."""
        cache_path = self.cache_dir / f"fbref_epl_{season}.parquet"
        if cache_path.exists() and not force_refresh:
            return pd.read_parquet(cache_path)

        try:
            import soccerdata as sd  # type: ignore

            fbref = sd.FBref(leagues="ENG-Premier League", seasons=season)
            passing = fbref.read_team_match_stats(stat_type="passing")
            possession = fbref.read_team_match_stats(stat_type="possession")
            defense = fbref.read_team_match_stats(stat_type="defense")
            misc = fbref.read_team_match_stats(stat_type="misc")
            merged = passing.join(possession, rsuffix="_poss").join(defense, rsuffix="_def").join(
                misc, rsuffix="_misc"
            )
            merged = merged.reset_index()
            merged.to_parquet(cache_path, index=False)
            return merged
        except ImportError as exc:
            raise RuntimeError(
                "Optional package 'soccerdata' is not installed and no cached FBref parquet exists at "
                f"{cache_path}. Install with `pip install .[scraping]` or use cached/synthetic data."
            ) from exc

    @staticmethod
    def parse_html_table(html: str, table_id: str) -> pd.DataFrame:
        """Parse an FBref HTML stats table into a flat DataFrame."""
        soup = BeautifulSoup(html, "lxml")
        table = soup.find("table", id=table_id)
        if table is None:
            raise ValueError(f"HTML table '{table_id}' not found")
        headers = [th.get("data-stat", th.get_text(strip=True)) for th in table.select("thead tr:last-child th")]
        rows: list[list[Any]] = []
        for tr in table.select("tbody tr"):
            if "thead" in (tr.get("class") or []):
                continue
            cells = [td.get_text(strip=True) for td in tr.find_all(["th", "td"])]
            if len(cells) == len(headers):
                rows.append(cells)
        return pd.DataFrame(rows, columns=headers)

    @staticmethod
    def compute_tactical_features_from_raw(df: pd.DataFrame) -> pd.DataFrame:
        """Derive the 14 canonical tactical features from raw FBref + Understat match columns."""
        out = df.copy()
        eps = 1e-6

        opp_passes_def_mid = (
            out["opp_passes_cmp_def_3rd"].astype(float) + out["opp_passes_cmp_mid_3rd"].astype(float)
        )
        team_tkl_int_att_mid = (
            out["tackles_int_att_3rd"].astype(float) + out["tackles_int_mid_3rd"].astype(float)
        )
        total_tkl_int = (
            out["tackles_int_def_3rd"].astype(float)
            + out["tackles_int_mid_3rd"].astype(float)
            + out["tackles_int_att_3rd"].astype(float)
        )

        out["ppda_proxy"] = opp_passes_def_mid / np.maximum(team_tkl_int_att_mid, 1.0)
        out["att_3rd_def_actions_per_opp_touch"] = out["tackles_int_att_3rd"].astype(float) / np.maximum(
            out["opp_touches_def_3rd"].astype(float), 1.0
        )
        out["mid_att_3rd_foul_rate"] = out["fouls_mid_att_3rd"].astype(float) / np.maximum(
            out["total_minutes"].astype(float) / 90.0, 0.5
        )
        out["long_pass_share"] = out["passes_long_att"].astype(float) / np.maximum(
            out["passes_att"].astype(float), 1.0
        )
        out["prog_pass_dist_per_pass"] = out["prog_pass_dist"].astype(float) / np.maximum(
            out["passes_att"].astype(float), 1.0
        )
        out["carries_to_passes_ratio"] = out["carries"].astype(float) / np.maximum(
            out["passes_att"].astype(float), 1.0
        )
        out["gk_launch_share"] = out["gk_launch_share"].astype(float)
        out["field_tilt"] = out["touches_att_3rd"].astype(float) / np.maximum(
            out["touches_att_3rd"].astype(float) + out["opp_touches_att_3rd"].astype(float), eps
        )
        out["possession_share"] = out["possession"].astype(float) / 100.0 if out["possession"].max() > 1.5 else out["possession"].astype(float)
        out["def_action_height"] = (
            1.0 * out["tackles_int_def_3rd"].astype(float)
            + 2.0 * out["tackles_int_mid_3rd"].astype(float)
            + 3.0 * out["tackles_int_att_3rd"].astype(float)
        ) / np.maximum(total_tkl_int, 1.0)
        out["crosses_per_att_3rd_touch"] = out["crosses"].astype(float) / np.maximum(
            out["touches_att_3rd"].astype(float), 1.0
        )
        out["avg_shot_dist"] = out["avg_shot_dist"].astype(float)
        out["through_balls_per_100_passes"] = (
            100.0 * out["through_balls"].astype(float) / np.maximum(out["passes_att"].astype(float), 1.0)
        )
        out["opp_avg_shot_dist"] = out["opp_avg_shot_dist"].astype(float)
        out["team_id"] = out["team_id"].map(canonicalize_team_name)
        for col in TACTICAL_FEATURES:
            if col not in out.columns:
                raise ValueError(f"Failed to derive tactical feature: {col}")
        return out
