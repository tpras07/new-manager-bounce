"""Changes in points per match relative to each club's pre-sacking average."""

import numpy as np
import pandas as pd

COMPONENTS = ["points_minus_xp", "fixture_shift", "comparison_form", "appointment_gap"]


def decompose(matched: pd.DataFrame) -> pd.DataFrame:
    pre = matched.loc[matched.event_k < 0].groupby("event_id")[["points", "xp", "xp_elo_pred", "comparison_xp_resid"]].mean()
    post = matched.loc[matched.event_k > 0].join(pre.add_suffix("_pre"), on="event_id")
    result = post[["event_id", "team_id", "season", "event_k"]].copy()
    result["raw_points_bounce"] = post.points - post.points_pre
    result["points_minus_xp"] = result.raw_points_bounce - (post.xp - post.xp_pre)
    result["fixture_shift"] = post.xp_elo_pred - post.xp_elo_pred_pre
    result["comparison_form"] = post.comparison_xp_resid - post.comparison_xp_resid_pre
    result["appointment_gap"] = post.gap
    if not np.allclose(result[COMPONENTS].sum(axis=1), result.raw_points_bounce, atol=1e-10, rtol=0):
        raise ValueError("Decomposition does not sum to the observed points change")
    return result


def summarize_window(decomposition: pd.DataFrame, window: tuple[int, int]) -> dict:
    rows = decomposition.loc[decomposition.event_k.between(*window)]
    # Give each appointment equal weight even when its final window is truncated.
    means = rows.groupby("event_id")[["raw_points_bounce", *COMPONENTS]].mean().mean()
    total = means["raw_points_bounce"]
    shares = {f"{name}_pct": 100 * means[name] / total if abs(total) > 1e-8 else None
              for name in COMPONENTS}
    return {**{name: float(value) if pd.notna(value) else None for name, value in means.items()}, **shares}
