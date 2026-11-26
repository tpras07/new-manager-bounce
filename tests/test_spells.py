import pandas as pd

from src.config import PipelineConfig
from src.ingest.panel import build_event_panel


def test_repeat_sacking_caps_the_previous_window_even_if_repeat_is_ineligible():
    dates = pd.date_range("2020-08-01", periods=24, freq="7D")
    matches = pd.DataFrame(dict(team_id="A", season="2020", matchweek=range(1, 25),
                                date=dates, elo_pre=1600., opponent_id="B", new_minutes_share=0))
    events = pd.DataFrame([
        dict(event_id=eid, team_id="A", season="2020", date_sack=dates[sack] + pd.Timedelta(days=1),
             date_perm_first_match=dates[start], matchweek_sack=sack + 1,
             matchweek_perm=start + 1, is_january_window=False)
        for eid, sack, start in [("first", 7, 9), ("repeat", 19, 21)]
    ])
    panel = build_event_panel(matches, events, PipelineConfig())
    eligible = panel.primary_balanced_events
    assert eligible.event_id.tolist() == ["first"]
    assert eligible.iloc[0].k_post_used == 11
    assert panel.event_windows.query("event_id == 'first' and phase == 'post'").date.max() == dates[19]
    assert panel.event_windows.query("event_id == 'repeat' and phase == 'pre'").date.min() >= dates[9]



def test_new_player_minutes_use_the_manager_in_charge_on_match_day():
    from src.ingest.managers import compute_new_minutes_share

    players = pd.DataFrame(dict(match_id=["M"] * 2, team_id="A", season="2020",
                                match_date="2020-12-01", player_id=["old", "new"],
                                registration_date=["2020-09-01", "2020-11-15"], minutes=[60, 30]))
    events = pd.DataFrame(dict(team_id="A", season="2020", event_id=["first", "second"],
                               date_perm_first_match=["2020-08-01", "2020-11-01"]))
    assert compute_new_minutes_share(players, events).iloc[0].new_minutes_share == 1 / 3
