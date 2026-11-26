"""Expected points from shot chains and goal difference from competitive 11v11 play."""

from __future__ import annotations

import numpy as np
import pandas as pd


def collapse_rebound_chains(shots_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse same-possession rebound chains via p_chain = 1 - prod(1 - p_j)."""
    if shots_df.empty:
        return pd.DataFrame(columns=["match_id", "team_id", "possession_chain_id", "p_chain"])

    df = shots_df.copy()
    df["one_minus_xg"] = 1.0 - df["xg"].clip(0.0, 1.0)
    grouped = (
        df.groupby(["match_id", "team_id", "possession_chain_id"], as_index=False, sort=False)["one_minus_xg"]
        .prod()
    )
    grouped["p_chain"] = (1.0 - grouped["one_minus_xg"]).clip(0.0, 1.0)
    return grouped[["match_id", "team_id", "possession_chain_id", "p_chain"]]


def poisson_binomial_pmf(probs: np.ndarray) -> np.ndarray:
    """Compute exact goal-count PMF for independent Bernoulli probabilities."""
    pmf = np.array([1.0], dtype=float)
    for p in probs:
        pmf = np.convolve(pmf, [1.0 - p, p])
    return pmf


def compute_match_xp_exact(
    home_chain_probs: np.ndarray,
    away_chain_probs: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """Compute exact (xp_home, xp_away, p_home_win, p_draw, p_away_win) via Poisson-Binomial convolution."""
    h_pmf = poisson_binomial_pmf(home_chain_probs)
    a_pmf = poisson_binomial_pmf(away_chain_probs)
    joint = np.outer(h_pmf, a_pmf)
    p_draw = float(np.trace(joint))
    p_home_win = float(np.tril(joint, k=-1).sum())
    p_away_win = float(np.triu(joint, k=1).sum())
    xp_home = 3.0 * p_home_win + p_draw
    xp_away = 3.0 * p_away_win + p_draw
    return xp_home, xp_away, p_home_win, p_draw, p_away_win


def compute_11v11_npxgd(
    team_match_stats: pd.DataFrame,
    shots_df: pd.DataFrame,
    min_comp_floor: float = 15.0,
) -> pd.DataFrame:
    """Non-penalty xG difference per 90 close-score 11v11 minutes, with a 15-minute default floor."""
    # With permanent dismissals, total 11v11 minutes is also the time of the first red.
    # Older shot tables record only the difference in player counts, which is zero at 10v10.
    full_sides = shots_df.get("is_11v11")
    if full_sides is None or full_sides.isna().any():
        first_red = team_match_stats.groupby("match_id")["min_11v11"].first()
        legacy = shots_df["minute"] <= shots_df["match_id"].map(first_red)
        full_sides = legacy if full_sides is None else full_sides.fillna(legacy)
    comp_shots = shots_df[
        full_sides
        & (~shots_df["is_penalty"].astype(bool))
        & (shots_df["man_diff_pre_shot"] == 0)
        & (shots_df["score_diff_pre_shot"].abs() <= 1)
    ]
    agg = (
        comp_shots.groupby(["match_id", "team_id"], as_index=False)["xg"]
        .sum()
        .rename(columns={"xg": "npxg_11v11_for"})
    )
    out = team_match_stats[["match_id", "team_id", "opponent_id", "min_11v11_comp"]].merge(
        agg, on=["match_id", "team_id"], how="left"
    )
    out["npxg_11v11_for"] = out["npxg_11v11_for"].fillna(0.0)
    opp_agg = agg.rename(columns={"team_id": "opponent_id", "npxg_11v11_for": "npxg_11v11_against"})
    out = out.merge(opp_agg, on=["match_id", "opponent_id"], how="left")
    out["npxg_11v11_against"] = out["npxg_11v11_against"].fillna(0.0)
    denom = np.maximum(out["min_11v11_comp"].astype(float), min_comp_floor) / 90.0
    out["npxgd_11v11"] = (out["npxg_11v11_for"] - out["npxg_11v11_against"]) / denom
    return out[["match_id", "team_id", "npxgd_11v11"]]


def enrich_team_match_stats_with_xp(
    team_match_stats: pd.DataFrame,
    shots_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute exact xP and 11v11 npxGD for every (match_id, team_id) row."""
    chains = collapse_rebound_chains(shots_df)
    chain_map: dict[tuple[str, str], np.ndarray] = {
        (mid, tid): grp["p_chain"].to_numpy(dtype=float)
        for (mid, tid), grp in chains.groupby(["match_id", "team_id"], sort=False)
    }

    tms = team_match_stats.copy()
    home_rows = tms[tms["is_home"].astype(bool)][["match_id", "team_id", "opponent_id"]]

    xp_lookup: dict[tuple[str, str], float] = {}
    for mid, h_team, a_team in home_rows.itertuples(index=False, name=None):
        h_probs = chain_map.get((mid, h_team), np.empty(0, dtype=float))
        a_probs = chain_map.get((mid, a_team), np.empty(0, dtype=float))
        xp_h, xp_a, _, _, _ = compute_match_xp_exact(h_probs, a_probs)
        xp_lookup[(mid, h_team)] = xp_h
        xp_lookup[(mid, a_team)] = xp_a

    tms["xp"] = [
        xp_lookup[(mid, tid)] for mid, tid in zip(tms["match_id"], tms["team_id"])
    ]
    npxgd_df = compute_11v11_npxgd(tms, shots_df)
    tms = tms.drop(columns=["npxgd_11v11"], errors="ignore").merge(
        npxgd_df, on=["match_id", "team_id"], how="left"
    )
    tms["npxgd_11v11"] = tms["npxgd_11v11"].fillna(0.0)
    return tms
