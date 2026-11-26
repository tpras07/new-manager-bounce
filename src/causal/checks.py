"""Repeat the comparison at placebo dates and on restricted cohorts."""

import pandas as pd

from src.causal.matching import match_events, summarize_gaps
from src.config import PipelineConfig
from src.features.residualize import MatchAdjustment
from src.ingest.panel import PanelBundle


def placebo_event(group, index, event_id, config, post_length):
    """Start a hypothetical appointment at index within an uninterrupted manager's spell."""
    k_pre = min(index, config.k_pre)
    freeze = group.iloc[max(0, index - k_pre - 1)]
    first = group.iloc[index]
    return dict(event_id=event_id, team_id=first.team_id, season=first.season,
                date_sack=first.date, date_perm_first_match=first.date,
                k_pre_used=k_pre, k_post_used=min(len(group) - index, post_length),
                elo_freeze=freeze.elo_pre, mw_freeze=freeze.matchweek)


def near_misses(matches, config):
    events = []
    controls = matches.loc[matches.is_non_treated_season].sort_values("date")
    for (team, season), group in controls.groupby(["team_id", "season"]):
        score = (group.points + .5 * group.xp).rolling(config.near_miss_window).sum().shift()
        first = max(config.k_pre, config.near_miss_window)
        last = len(group) - config.min_k_post_balanced
        if last < first:
            continue
        eligible = score.iloc[first:last + 1]
        index = first + eligible.argmin()
        event = placebo_event(group, index, f"slump_{team}_{season}", config, config.k_post)
        events.append(dict(event, slump_score=eligible.min()))
    if not events:
        return pd.DataFrame()
    return pd.DataFrame(events).nsmallest(config.near_miss_n_placebos, "slump_score")


def comparison_checks(panel: PanelBundle, models: dict[str, MatchAdjustment], matched: pd.DataFrame,
                      config: PipelineConfig) -> dict[str, pd.DataFrame]:
    matches = panel.team_match_stats
    primary = panel.primary_balanced_events
    shifted = []
    caretakers = []
    for ev in panel.manager_events.itertuples(index=False):
        season = matches.loc[(matches.team_id == ev.team_id) & (matches.season == ev.season)].sort_values("date")
        pre = season.loc[season.date < ev.date_sack].tail(ev.n_pre_available)
        index = len(pre) - config.in_time_placebo_shift
        if ev.is_primary_balanced and index >= 3:
            shifted.append(placebo_event(pre, index, f"early_{ev.event_id}", config, config.in_time_placebo_shift))
        interim = season.loc[(season.date >= ev.date_sack) & (season.date < ev.date_perm_first_match)]
        if len(pre) >= 3 and not interim.empty:
            spell = pd.concat([pre, interim])
            caretakers.append(placebo_event(spell, len(pre), f"interim_{ev.event_id}", config, min(3, len(interim))))

    cohorts = {
        "slump": near_misses(matches, config),
        "in_time": pd.DataFrame(shifted),
        "caretaker": pd.DataFrame(caretakers),
    }
    checks = {name: summarize_gaps(match_events(matches, events, models, config))
              for name, events in cohorts.items()}
    checks["clean_donors"] = summarize_gaps(match_events(matches, primary, models, config, clean_donors=True))
    for name, flag in [("january", "is_january_window"), ("no_window", "is_no_window_hire")]:
        ids = primary.loc[primary[flag], "event_id"]
        checks[name] = summarize_gaps(matched.loc[matched.event_id.isin(ids)])
    short = matched.loc[matched.event_k.between(*config.short_term_window)]
    for key in ["team_id", "season"]:
        checks[f"leave_out_{key}"] = pd.DataFrame([
            {key: value, "mean_gap": short.loc[short[key] != value, "gap"].mean()}
            for value in short[key].unique()
        ])
    return checks
