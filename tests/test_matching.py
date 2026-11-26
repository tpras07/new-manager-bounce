import numpy as np
import pandas as pd

from src.causal.decomposition import COMPONENTS, decompose, summarize_window
from src.causal.matching import match_events, summarize_gaps
from src.config import PipelineConfig, TACTICAL_PILLAR_MAP
from src.features.residualize import MatchAdjustment
from src.features.tactics import summarize_tactics


def test_nearest_pre_form_and_regime_bounds():
    # A has identical pre-form, B has better post-form, and C plays in a future season.
    matches = []
    for team, season, form in [("T", "2020", 1), ("A", "2020", 1), ("B", "2020", 0), ("C", "2021", 1)]:
        for week in range(1, 9):
            xp = form if week <= 4 else {"T": 2, "A": .5, "B": 2, "C": 2}[team]
            matches.append(dict(team_id=team, season=season, match_id=f"{team}{week}", matchweek=week,
                                date=pd.Timestamp(2020, 1, week), xp=xp, points=1 if week < 5 else 3,
                                elo_pre=1600., opp_elo_pre=1600., is_home=True, npxgd_11v11=xp,
                                is_non_treated_season=team != "T", new_minutes_share=0))
    matches = pd.DataFrame(matches)
    event = dict(event_id="E", team_id="T", season="2020", date_sack=pd.Timestamp(2020, 1, 5),
                 date_perm_first_match=pd.Timestamp(2020, 1, 6), k_pre_used=3, k_post_used=2,
                 mw_freeze=1, elo_freeze=1600.)
    # Fixed predictions make the hand calculation independent of fixture-model fitting.
    model = MatchAdjustment(np.zeros((4, 2)), np.empty(0), np.empty(0))
    matched = match_events(matches, pd.DataFrame([event]), {"2020": model}, PipelineConfig(n_neighbors=1))
    empty = match_events(matches, pd.DataFrame([event]), {"2020": model}, PipelineConfig(n_neighbors=3))
    assert summarize_gaps(empty).empty
    assert matched.match_id.tolist() == ["T2", "T3", "T4", "T6", "T7"]
    assert matched.comparison_teams.unique().tolist() == ["A (2020)"]
    assert matched.n_eligible_donors.eq(2).all()
    assert matched.gap.tolist() == [0, 0, 0, 1.5, 1.5]
    changes = decompose(matched)
    assert np.allclose(changes[COMPONENTS], [[1, 0, -.5, 1.5]] * 2)
    assert summarize_window(changes, (1, 2))["appointment_gap_pct"] == 75
    # A constant level difference before and after appointment is not a bounce.
    matches.loc[matches.team_id == "T", "xp"] += .1
    shifted = match_events(matches, pd.DataFrame([event]), {"2020": model}, PipelineConfig(n_neighbors=1))
    assert np.allclose(shifted.gap, matched.gap)
    assert summarize_window(changes.assign(raw_points_bounce=0), (1, 2))["appointment_gap_pct"] is None


def test_later_matches_do_not_contribute_to_early_tactical_change():
    weeks = [-2, -1, 1, 2, 3]
    matched = pd.DataFrame(dict(event_id="one", season="2020", team_id="A", match_id=weeks,
                                event_k=weeks, gap=[0, 0, .8, 1, -.5]))
    scores = matched[["match_id", "team_id"]].assign(**{name: [0, 0, 3, 1, 100]
                                                       for name in TACTICAL_PILLAR_MAP})
    row = summarize_tactics(matched, scores, PipelineConfig(short_term_window=(1, 2),
                                                            medium_term_window=(3, 4))).iloc[0]
    assert row.tactical_change == 4
    assert row.later_gap == -.5
    assert row.later_matches == 1
