"""Match appointments to comparable stretches of seasons without a manager change."""

import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.features.residualize import MatchAdjustment


def match_events(matches: pd.DataFrame, events: pd.DataFrame, models: dict[str, MatchAdjustment],
                 config: PipelineConfig, clean_donors: bool = False) -> pd.DataFrame:
    """Select donors by pre-window mean adjusted xP and non-penalty xG difference."""
    groups = {key: group.sort_values("date") for key, group in matches.groupby(["team_id", "season"])}
    controls = matches.loc[matches["is_non_treated_season"]]
    if clean_donors:
        controls = controls.loc[~controls["is_sutva_contaminated"]]
    keys = ["team_id", "season"]
    rows = []
    for event in events.itertuples(index=False):
        model = models[event.season]
        treated = groups[event.team_id, event.season]
        pre = treated.loc[treated.date < event.date_sack].tail(event.k_pre_used)
        post = treated.loc[treated.date >= event.date_perm_first_match].head(event.k_post_used)
        if len(pre) < 3 or post.empty:
            continue
        window = pd.concat([pre, post])
        horizons = np.r_[np.arange(-len(pre), 0), np.arange(1, len(post) + 1)]
        weeks = dict(zip(window.matchweek, horizons, strict=True))
        aligned = controls.loc[(controls.season <= event.season) & controls.matchweek.isin(weeks)]
        ratings = matches.loc[(matches.matchweek == event.mw_freeze) &
                              matches.is_non_treated_season, keys + ["elo_pre"]]
        ratings = ratings.loc[(ratings.elo_pre - event.elo_freeze).abs() <= config.elo_caliper]
        ratings = ratings.loc[~((ratings.team_id == event.team_id) & (ratings.season == event.season))]
        donors = aligned.merge(ratings.rename(columns={"elo_pre": "frozen_elo"}), on=keys,
                               validate="many_to_one")
        donors = donors.loc[donors.groupby(keys)["matchweek"].transform("size") == len(window)]
        if donors.empty:
            continue
        donors = model.outcomes(donors, donors.frozen_elo.to_numpy())
        donors["event_k"] = donors.matchweek.map(weeks)
        treated = model.outcomes(window, event.elo_freeze).assign(event_k=horizons)
        form = ["xp_resid", "npxgd_resid"]
        donor_form = donors.loc[donors.event_k < 0].groupby(keys)[form].mean()
        if len(donor_form) < config.n_neighbors:
            continue
        target = treated.loc[treated.event_k < 0, form].mean()
        # Scale by between-donor variation so both measures contribute on the same scale.
        scale = donor_form.std(ddof=0).replace(0, np.inf)
        distance = ((donor_form - target) / scale).pow(2).sum(axis=1)
        nearest = distance.sort_values(kind="stable").head(config.n_neighbors).index
        selected = donors.set_index(keys).loc[nearest]
        baseline = selected.groupby("event_k")["xp_resid"].mean()
        treated["comparison_xp_resid"] = treated.event_k.map(baseline)
        difference = treated.xp_resid - treated.comparison_xp_resid
        # Matching need not reproduce the pre-sacking level exactly. Compare changes.
        treated["gap"] = difference - difference.loc[treated.event_k < 0].mean()
        treated["event_id"] = event.event_id
        treated["n_eligible_donors"] = len(donor_form)
        treated["comparison_teams"] = ", ".join(f"{team} ({season})" for team, season in nearest)
        rows.append(treated)
    columns = ["event_id", "team_id", "season", "match_id", "event_k", "points", "xp",
               "xp_elo_pred", "xp_resid", "comparison_xp_resid", "gap",
               "n_eligible_donors", "comparison_teams", "new_minutes_share"]
    return pd.concat(rows, ignore_index=True)[columns] if rows else pd.DataFrame(columns=columns)


def summarize_gaps(matched: pd.DataFrame) -> pd.DataFrame:
    gaps = matched["gap"].astype(float).groupby(matched["event_k"])
    return pd.DataFrame({"n_events": gaps.count(), "mean_gap": gaps.mean(),
                         "q25": gaps.quantile(.25), "q75": gaps.quantile(.75)}).reset_index()
