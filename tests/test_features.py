"""Expected points, game state and fixture adjustment."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.residualize import MatchAdjustment
from src.features.xp_simulator import (
    collapse_rebound_chains,
    compute_11v11_npxgd,
    compute_match_xp_exact,
)
from src.ingest.understat import compute_game_state_minutes


def test_rebound_chain_caps_expected_goals_below_one() -> None:
    shots = pd.DataFrame(
        [
            {"match_id": "M1", "team_id": "Arsenal", "possession_chain_id": "C1", "xg": 0.6},
            {"match_id": "M1", "team_id": "Arsenal", "possession_chain_id": "C1", "xg": 0.7},
        ]
    )
    chains = collapse_rebound_chains(shots)
    assert len(chains) == 1
    expected = 1.0 - (1.0 - 0.6) * (1.0 - 0.7)
    assert np.isclose(chains.iloc[0]["p_chain"], expected)
    assert chains.iloc[0]["p_chain"] <= 1.0


def test_exact_points_for_empty_certain_and_independent_chances() -> None:
    assert compute_match_xp_exact([], []) == (1, 1, 0, 1, 0)
    assert compute_match_xp_exact([1], []) == (3, 0, 1, 0, 0)
    # Home has a 50% chance to score, away 25%. A draw occurs at 0-0 or 1-1.
    assert np.allclose(compute_match_xp_exact([.5], [.25]), (1.625, .875, .375, .5, .125))
    assert np.allclose(compute_match_xp_exact([.5, .5], [1]), (1.25, 1.25, .25, .5, .25))


def test_game_state_minutes_and_11v11_filter() -> None:
    gs = compute_game_state_minutes(
        90.0,
        "Arsenal",
        "Chelsea",
        goal_events=[{"team_id": "Arsenal", "minute": 30.0}, {"team_id": "Arsenal", "minute": 60.0}],
        red_card_events=[{"team_id": "Chelsea", "minute": 75.0}],
    )
    assert np.isclose(gs["Arsenal"]["min_tied"], 30.0)
    assert np.isclose(gs["Arsenal"]["min_leading"], 60.0)
    assert np.isclose(gs["Arsenal"]["min_11v11"], 75.0)
    # 0-30 (sd=0), 30-60 (sd=1), 60-75 (sd=2 -> excluded), 75-90 (11v10 -> excluded)
    assert np.isclose(gs["Arsenal"]["min_11v11_comp"], 60.0)

    tms = pd.DataFrame(
        [
            {"match_id": "M1", "team_id": "Arsenal", "opponent_id": "Chelsea", "min_11v11_comp": 60.0, "min_11v11": 75.0},
            {"match_id": "M1", "team_id": "Chelsea", "opponent_id": "Arsenal", "min_11v11_comp": 60.0, "min_11v11": 75.0},
        ]
    )
    shots = pd.DataFrame(
        [
            {"match_id": "M1", "team_id": "Arsenal", "xg": 0.30, "is_penalty": False, "man_diff_pre_shot": 0, "score_diff_pre_shot": 0},
            {"match_id": "M1", "team_id": "Arsenal", "xg": 0.78, "is_penalty": True, "man_diff_pre_shot": 0, "score_diff_pre_shot": 0},
            {"match_id": "M1", "team_id": "Arsenal", "xg": 0.40, "is_penalty": False, "man_diff_pre_shot": 0, "score_diff_pre_shot": 2},
            {"match_id": "M1", "team_id": "Chelsea", "xg": 0.10, "is_penalty": False, "man_diff_pre_shot": -1, "score_diff_pre_shot": -1},
        ]
    )
    shots["minute"] = [5, 40, 65, 80]
    # Equal numbers after a second red card also gives man_diff == 0.
    ten_a_side = shots.iloc[[0]].assign(minute=86, xg=0.8)
    npxgd = compute_11v11_npxgd(tms, pd.concat([shots, ten_a_side]))

    ars_val = float(npxgd.loc[npxgd["team_id"] == "Arsenal", "npxgd_11v11"].iloc[0])
    # Only the 0.30 shot qualifies, over 60 mins -> 0.30 / (60/90) = 0.45
    assert np.isclose(ars_val, 0.45)


def test_fixture_adjustment_respects_frozen_elo() -> None:
    rng = np.random.default_rng(7)
    n = 120
    elo_self = rng.normal(1700, 100, size=n)
    elo_opp = rng.normal(1700, 100, size=n)
    is_home = rng.integers(0, 2, size=n)
    xp = np.clip(1.3 + 0.25 * is_home + 0.4 * ((elo_self - elo_opp) / 100.0) + rng.normal(0, 0.2, n), 0, 3)

    from src.config import TACTICAL_FEATURES

    df = pd.DataFrame(
        {
            "match_id": [f"M{i}" for i in range(n)],
            "team_id": "T1",
            "opponent_id": "T2",
            "season": "2022-23",
            "matchweek": np.arange(n),
            "date": pd.date_range("2022-08-15", periods=n, freq="7D"),
            "is_home": is_home.astype(bool),
            "elo_pre": elo_self,
            "opp_elo_pre": elo_opp,
            "xp": xp,
            "npxgd_11v11": xp - 1.3,
            "min_trailing": 20.0,
            "min_leading": 20.0,
            "min_down_men": 0.0,
            "min_up_men": 0.0,
            "total_minutes": 90.0,
            "is_non_treated_season": True,
        }
    )
    for f in TACTICAL_FEATURES:
        df[f] = rng.normal(0, 1, size=n)

    model = MatchAdjustment.fit(df)
    df_frozen = df.copy()
    df_frozen["elo_frozen"] = 1850.0
    out_live = model.outcomes(df, df.elo_pre.to_numpy())
    out_froz = model.outcomes(df_frozen, df_frozen.elo_frozen.to_numpy())
    assert not np.allclose(out_live["xp_elo_pred"], out_froz["xp_elo_pred"])
