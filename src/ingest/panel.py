"""Match windows around dismissals and permanent appointments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import PipelineConfig


@dataclass
class PanelBundle:
    """Structured event-study panel separating non-treated, primary balanced, unbalanced, and interim cohorts."""

    team_match_stats: pd.DataFrame
    manager_events: pd.DataFrame
    non_treated_team_seasons: set[tuple[str, str]]
    primary_balanced_events: pd.DataFrame
    unbalanced_events: pd.DataFrame
    interim_spells: pd.DataFrame
    event_windows: pd.DataFrame


def build_event_panel(
    team_match_stats: pd.DataFrame,
    manager_events: pd.DataFrame,
    config: PipelineConfig = PipelineConfig(),
) -> PanelBundle:
    """Count matches in each manager spell and align the first permanent match at k=1."""
    tms = team_match_stats.sort_values(["season", "team_id", "date"]).copy()
    events = manager_events.sort_values(["season", "team_id", "matchweek_perm"]).copy()

    # Seasons with any dismissal are ineligible as comparisons.
    all_ts = set(zip(tms["team_id"], tms["season"]))
    treated_ts = set(zip(events["team_id"], events["season"]))
    non_treated_ts = all_ts - treated_ts
    tms["is_non_treated_season"] = [
        (tm, sn) in non_treated_ts for tm, sn in zip(tms["team_id"], tms["season"])
    ]

    recent_appointments = {
        (event.team_id, event.season, week)
        for event in events.itertuples()
        for week in range(event.matchweek_perm, event.matchweek_perm + config.sutva_contam_horizon)
    }
    tms["is_sutva_contaminated"] = [
        key in recent_appointments for key in zip(tms.opponent_id, tms.season, tms.matchweek)
    ]

    ev_records = []
    window_rows = []
    interim_rows = []

    for _, ev in events.iterrows():
        tm = ev["team_id"]
        sn = ev["season"]
        ev_id = ev["event_id"]
        sub = tms[(tms["team_id"] == tm) & (tms["season"] == sn)].sort_values("date").reset_index(drop=True)
        if sub.empty:
            continue

        pre_matches = sub[sub["date"] < ev["date_sack"]].copy()
        interim_matches = sub[
            (sub["date"] >= ev["date_sack"]) & (sub["date"] < ev["date_perm_first_match"])
        ].copy()
        post_matches = sub[sub["date"] >= ev["date_perm_first_match"]].copy()

        # Respect repeat-sacking boundaries within the same season: stop post_matches before next sacking
        later_events = events[
            (events["team_id"] == tm)
            & (events["season"] == sn)
            & (events["date_sack"] > ev["date_perm_first_match"])
        ]
        if not later_events.empty:
            next_sack_dt = later_events["date_sack"].min()
            post_matches = post_matches[post_matches["date"] < next_sack_dt].copy()

        prior_events = events[
            (events["team_id"] == tm)
            & (events["season"] == sn)
            & (events["date_perm_first_match"] < ev["date_sack"])
        ]
        if not prior_events.empty:
            prev_perm_dt = prior_events["date_perm_first_match"].max()
            pre_matches = pre_matches[pre_matches["date"] >= prev_perm_dt].copy()

        n_pre = len(pre_matches)
        n_post = len(post_matches)

        is_balanced = (min(n_pre, config.k_pre) >= config.min_k_pre_balanced
                       and min(n_post, config.k_post) >= config.min_k_post_balanced)
        is_full_horizon = (n_pre >= config.k_pre) and (n_post >= config.k_post)
        is_unbalanced_late = (not is_balanced) and (n_pre >= config.min_k_pre_balanced) and (n_post >= 3)

        # Freeze Elo at t_freeze = t_0 - K_pre - 1 (or earliest available pre-window match if shorter)
        k_pre_used = min(n_pre, config.k_pre)
        pre_win = pre_matches.tail(k_pre_used).copy()
        if not pre_win.empty:
            freeze_idx = max(0, len(pre_matches) - k_pre_used - 1)
            elo_freeze = float(pre_matches.iloc[freeze_idx]["elo_pre"])
            mw_freeze = int(pre_matches.iloc[freeze_idx]["matchweek"])
        else:
            elo_freeze = float(post_matches.iloc[0]["elo_pre"]) if not post_matches.empty else np.nan
            mw_freeze = int(ev["matchweek_sack"])

        k_post_used = min(n_post, config.k_post)
        post_win = post_matches.head(k_post_used).copy()

        mean_new_min_post = float(post_win["new_minutes_share"].mean()) if not post_win.empty else 0.0
        is_no_window_hire = (not bool(ev["is_january_window"])) and (mean_new_min_post == 0.0)

        ev_row = ev.to_dict()
        ev_row.update(
            {
                "n_pre_available": n_pre,
                "n_post_available": n_post,
                "k_pre_used": k_pre_used,
                "k_post_used": k_post_used,
                "elo_freeze": elo_freeze,
                "mw_freeze": mw_freeze,
                "is_primary_balanced": is_balanced,
                "is_full_15_horizon": is_full_horizon,
                "is_secondary_unbalanced": is_unbalanced_late,
                "mean_new_minutes_post": mean_new_min_post,
                "is_no_window_hire": is_no_window_hire,
            }
        )
        ev_records.append(ev_row)

        # Tag event-time k = -k_pre_used ... -1 for pre-sack and k = +1 ... +k_post_used for post-perm
        for frame, phase, horizons, destination in (
            (pre_win, "pre", range(-len(pre_win), 0), window_rows),
            (post_win, "post", range(1, len(post_win) + 1), window_rows),
            (interim_matches, "interim", range(1, len(interim_matches) + 1), interim_rows),
        ):
            if not frame.empty:
                destination.append(frame.assign(event_id=ev_id, event_k=horizons, phase=phase,
                                                elo_freeze=elo_freeze))

    events_enriched = pd.DataFrame(ev_records)
    primary_balanced = (
        events_enriched[events_enriched["is_primary_balanced"]].reset_index(drop=True)
        if not events_enriched.empty
        else events_enriched
    )
    unbalanced = (
        events_enriched[events_enriched["is_secondary_unbalanced"]].reset_index(drop=True)
        if not events_enriched.empty
        else events_enriched
    )
    event_windows = pd.concat(window_rows, ignore_index=True) if window_rows else pd.DataFrame()
    interim_df = pd.concat(interim_rows, ignore_index=True) if interim_rows else pd.DataFrame()

    return PanelBundle(
        team_match_stats=tms,
        manager_events=events_enriched,
        non_treated_team_seasons=non_treated_ts,
        primary_balanced_events=primary_balanced,
        unbalanced_events=unbalanced,
        interim_spells=interim_df,
        event_windows=event_windows,
    )
