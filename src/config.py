"""Project-wide configuration, paths, and econometric constants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

SEASONS: tuple[str, ...] = (
    "2017-18",
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
)

TACTICAL_PILLAR_MAP: dict[str, tuple[str, ...]] = {
    "Pressing": (
        "ppda_proxy",
        "att_3rd_def_actions_per_opp_touch",
        "mid_att_3rd_foul_rate",
    ),
    "Directness": (
        "long_pass_share",
        "prog_pass_dist_per_pass",
        "carries_to_passes_ratio",
        "gk_launch_share",
    ),
    "Territoriality": (
        "field_tilt",
        "possession_share",
        "def_action_height",
    ),
    "ChanceProfile": (
        "crosses_per_att_3rd_touch",
        "avg_shot_dist",
        "through_balls_per_100_passes",
        "opp_avg_shot_dist",
    ),
}

TACTICAL_FEATURES = tuple(feature for group in TACTICAL_PILLAR_MAP.values() for feature in group)
REVERSE_FEATURES = {"ppda_proxy", "carries_to_passes_ratio", "through_balls_per_100_passes"}

GAME_STATE_MINUTES = ("min_trailing", "min_leading", "min_down_men", "min_up_men")


@dataclass(frozen=True)
class PipelineConfig:
    """Match windows and donor eligibility for the appointment study."""

    seasons: tuple[str, ...] = SEASONS
    k_pre: int = 8
    k_post: int = 15
    min_k_pre_balanced: int = 6
    min_k_post_balanced: int = 10
    elo_caliper: float = 150.0
    n_neighbors: int = 5
    near_miss_n_placebos: int = 30
    near_miss_window: int = 8
    in_time_placebo_shift: int = 5
    sutva_contam_horizon: int = 5
    short_term_window: tuple[int, int] = (1, 5)
    medium_term_window: tuple[int, int] = (6, 15)
    random_seed: int = 42
    processed_dir: Path = PROCESSED_DATA_DIR
    reports_dir: Path = REPORTS_DIR

    def __post_init__(self) -> None:
        if self.n_neighbors < 1 or self.k_pre < 3 or self.k_post < 1:
            raise ValueError("Matching requires a positive donor count, three pre-matches and one post-match")
        if self.near_miss_window < 1 or self.in_time_placebo_shift < 1:
            raise ValueError("Placebo windows must be positive")
        a, b = self.short_term_window
        c, d = self.medium_term_window
        if not 1 <= a <= b < c <= d:
            raise ValueError("Early and later match windows must be positive, ordered and disjoint")
