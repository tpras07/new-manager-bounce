"""Generate simulated fixtures, shot chains, manager spells and tactical statistics."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from src.config import SEASONS, TACTICAL_FEATURES
from src.features.xp_simulator import enrich_team_match_stats_with_xp
from src.ingest.understat import compute_game_state_minutes

EPL_CLUB_POOL: list[tuple[str, float]] = [
    ("Manchester City", 1980.0),
    ("Liverpool", 1940.0),
    ("Arsenal", 1890.0),
    ("Chelsea", 1835.0),
    ("Tottenham Hotspur", 1810.0),
    ("Manchester United", 1815.0),
    ("Newcastle United", 1775.0),
    ("Aston Villa", 1750.0),
    ("Brighton", 1735.0),
    ("West Ham United", 1715.0),
    ("Crystal Palace", 1690.0),
    ("Brentford", 1685.0),
    ("Fulham", 1670.0),
    ("Bournemouth", 1655.0),
    ("Wolverhampton Wanderers", 1665.0),
    ("Everton", 1645.0),
    ("Nottingham Forest", 1630.0),
    ("Leicester City", 1660.0),
    ("Leeds United", 1625.0),
    ("Southampton", 1610.0),
    ("Burnley", 1595.0),
    ("Watford", 1585.0),
    ("Sheffield United", 1575.0),
    ("Norwich City", 1560.0),
    ("Ipswich Town", 1555.0),
    ("Luton Town", 1550.0),
]

# True 14x4 factor loading structure mapping [Pressing, Directness, Territoriality, ChanceProfile]
# onto the 14 standardized tactical features in the simulated matches.
TRUE_FACTOR_LOADINGS = np.array(
    [
        # Pressing, Directness, Territoriality, ChanceProfile
        [-0.90, 0.08, -0.15, 0.05],  # ppda_proxy (lower PPDA = higher pressing)
        [0.92, -0.05, 0.12, 0.04],   # att_3rd_def_actions_per_opp_touch
        [0.86, 0.10, -0.08, -0.05],  # mid_att_3rd_foul_rate
        [-0.08, 0.91, -0.18, 0.08],  # long_pass_share
        [0.05, 0.89, -0.10, 0.04],   # prog_pass_dist_per_pass
        [0.10, -0.84, 0.15, -0.06],  # carries_to_passes_ratio
        [-0.06, 0.88, -0.16, 0.05],  # gk_launch_share
        [0.18, -0.15, 0.91, 0.06],   # field_tilt
        [0.10, -0.22, 0.92, -0.04],  # possession_share
        [0.32, -0.10, 0.82, 0.05],   # def_action_height
        [-0.04, 0.12, 0.10, 0.89],   # crosses_per_att_3rd_touch
        [-0.05, 0.15, -0.08, 0.87],  # avg_shot_dist
        [0.12, 0.08, 0.14, -0.83],   # through_balls_per_100_passes
        [0.10, -0.10, 0.05, 0.86],   # opp_avg_shot_dist
    ],
    dtype=float,
)

# Illustrative appointments in the simulated seasons
FEATURED_REAL_EVENTS: dict[tuple[str, str], tuple[str, str, int, int, bool, np.ndarray, float]] = {
    # (season, team): (old_mgr, new_mgr, mw_sack, interim_cnt, jan_win, true_delta[4], true_persist)
    ("2020-21", "Chelsea"): (
        "Frank Lampard",
        "Thomas Tuchel",
        19,
        0,
        True,
        np.array([0.55, -0.95, 1.15, -0.60]),
        0.34,
    ),
    ("2022-23", "Aston Villa"): (
        "Steven Gerrard",
        "Unai Emery",
        11,
        2,
        False,
        np.array([0.85, -0.60, 0.90, -0.75]),
        0.38,
    ),
    ("2022-23", "Everton"): (
        "Frank Lampard",
        "Sean Dyche",
        20,
        0,
        False,
        np.array([-0.80, 1.25, -0.65, 0.70]),
        0.28,
    ),
    ("2023-24", "Crystal Palace"): (
        "Roy Hodgson",
        "Oliver Glasner",
        24,
        1,
        False,
        np.array([1.15, -0.45, 0.65, -0.50]),
        0.36,
    ),
}


FEATURE_SCALES = {
    "ppda_proxy": (11.5, 3.2, 3.5, 28.0),
    "att_3rd_def_actions_per_opp_touch": (0.045, 0.014, 0.01, 0.12),
    "mid_att_3rd_foul_rate": (6.2, 1.8, 1.0, 15.0),
    "long_pass_share": (0.14, 0.04, 0.04, 0.35),
    "prog_pass_dist_per_pass": (5.4, 0.85, 2.5, 9.5),
    "carries_to_passes_ratio": (0.68, 0.09, 0.35, 1.05),
    "gk_launch_share": (0.42, 0.16, 0.05, 0.95),
    "field_tilt": (0.50, 0.13, 0.12, 0.88),
    "possession_share": (0.50, 0.11, 0.18, 0.82),
    "def_action_height": (1.85, 0.22, 1.15, 2.65),
    "crosses_per_att_3rd_touch": (0.09, 0.03, 0.02, 0.25),
    "avg_shot_dist": (16.2, 2.1, 8.0, 26.0),
    "through_balls_per_100_passes": (0.45, 0.22, 0.0, 1.8),
    "opp_avg_shot_dist": (16.2, 2.1, 8.0, 26.0),
}


def _team_process_modifier(events: list[dict], mw: int, rng: np.random.Generator) -> tuple[float, float, float]:
    """Return (process_shift, finishing_luck_shift, new_min_share)."""
    proc = 0.0
    fin_shift = 0.0
    new_min = 0.0
    for idx, ev in enumerate(events):
        if idx + 1 < len(events) and mw > events[idx + 1]["matchweek_sack"]:
            continue
        mw_s = ev["matchweek_sack"]
        mw_p = ev["matchweek_perm"]
        if mw_s - 7 <= mw <= mw_s:
            # Ashenfelter dip: moderate process dip + severe bad finishing luck
            dist = mw_s - mw
            proc -= 0.18 * np.exp(-0.25 * (dist - 1))
            fin_shift -= 0.35
        elif mw_s < mw < mw_p:
            # Interim spell: brief emotional relief, noisy process
            proc += 0.08
        elif mw >= mw_p:
            k = mw - mw_p + 1
            # Short-term relief decays exponentially; true persistence endures
            relief = ev["_true_relief"] * np.exp(-0.35 * (k - 1))
            persist = ev["_true_persist"] * (1.0 - np.exp(-0.5 * k))
            proc += relief + persist
            if ev["is_january_window"] and k >= 1:
                new_min = float(np.clip(0.18 + 0.05 * rng.normal(), 0.05, 0.45))
    return proc, fin_shift, new_min

def _generate_double_round_robin(teams: list[str], rng: np.random.Generator) -> list[list[tuple[str, str]]]:
    """Generate a balanced 38-matchweek Premier League schedule for 20 teams."""
    n = len(teams)
    order = list(teams)
    rng.shuffle(order)
    first_half: list[list[tuple[str, str]]] = []
    for r in range(n - 1):
        round_pairs = []
        for i in range(n // 2):
            t1 = order[i]
            t2 = order[n - 1 - i]
            if (r + i) % 2 == 0:
                round_pairs.append((t1, t2))
            else:
                round_pairs.append((t2, t1))
        first_half.append(round_pairs)
        order = [order[0]] + [order[-1]] + order[1:-1]
    second_half = [[(away, home) for home, away in rnd] for rnd in first_half]
    return first_half + second_half


def generate_synthetic_epl_dataset(
    seed: int = 42,
    seasons: tuple[str, ...] = SEASONS,
) -> dict[str, pd.DataFrame]:
    """Generate the four study tables for the requested seasons."""
    rng = np.random.default_rng(seed)

    club_elo = {name: base for name, base in EPL_CLUB_POOL}
    all_matches: list[dict[str, Any]] = []
    all_shots: list[dict[str, Any]] = []
    all_tms: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []

    match_counter = 1
    event_counter = 1

    # Assign 20 active clubs per season with realistic relegation/promotion churn
    core_teams = [name for name, _ in EPL_CLUB_POOL[:17]]
    rotating_teams = [name for name, _ in EPL_CLUB_POOL[17:]]

    for s_idx, season in enumerate(seasons):
        start_year = int(season.split("-")[0])
        season_start = datetime(start_year, 8, 12)
        rot_slice = [rotating_teams[(s_idx * 2 + j) % len(rotating_teams)] for j in range(3)]
        season_teams = core_teams + rot_slice
        schedule = _generate_double_round_robin(season_teams, rng)

        # Select 4-5 treated clubs per season (including featured events when in that season)
        featured_in_season = [
            tm for (sn, tm) in FEATURED_REAL_EVENTS if sn == season and tm in season_teams
        ]
        other_candidates = [tm for tm in season_teams if tm not in featured_in_season]
        rng.shuffle(other_candidates)
        n_extra = 4 - len(featured_in_season)
        treated_teams_season = featured_in_season + other_candidates[:n_extra]

        # Build managerial event schedule and latent tactical trajectories for each team in season
        team_latent_z: dict[str, np.ndarray] = {}
        team_event_info: dict[str, list[dict[str, Any]]] = {}

        for tm in season_teams:
            base_elo = club_elo[tm]
            # Initial tactical identity correlated moderately with club strength
            z0 = np.array(
                [
                    (base_elo - 1720.0) / 220.0 + rng.normal(0.0, 0.45),
                    -(base_elo - 1720.0) / 220.0 + rng.normal(0.0, 0.45),
                    (base_elo - 1720.0) / 180.0 + rng.normal(0.0, 0.40),
                    rng.normal(0.0, 0.50),
                ]
            )
            z_path = np.zeros((39, 4), dtype=float)
            z_path[0] = z0

            # Natural within-manager random walk (Q ~ 0.025 * I)
            for mw in range(1, 39):
                z_path[mw] = z_path[mw - 1] + rng.normal(0.0, 0.14, size=4)

            if tm in treated_teams_season:
                if (season, tm) in FEATURED_REAL_EVENTS:
                    old_m, new_m, mw_sack, interim_cnt, jan_win, true_delta, true_persist = FEATURED_REAL_EVENTS[
                        (season, tm)
                    ]
                    is_repeat = False
                else:
                    # Mix of mid-season sackings (mw 9..22), late-season sackings (mw 28..32), and repeat sackings
                    is_late = (s_idx % 3 == 2) and (tm == treated_teams_season[-1])
                    mw_sack = int(rng.integers(28, 32)) if is_late else int(rng.integers(9, 22))
                    interim_cnt = int(rng.choice([0, 0, 1, 2]))
                    jan_win = bool(19 <= mw_sack <= 23 and rng.random() < 0.65)
                    is_repeat = bool(tm in ("Watford", "Chelsea", "Southampton") and rng.random() < 0.45 and not is_late and mw_sack <= 16)
                    # Vary genuine tactical overhaul vs low-drift "psychological relief" hire
                    if rng.random() < 0.55:
                        true_delta = rng.normal(0.0, 0.75, size=4)
                        # Relegation candidates sometimes do pragmatic consolidation
                        if base_elo < 1650.0 and rng.random() < 0.6:
                            true_delta = np.array([-0.75, 1.05, -0.50, 0.55]) + rng.normal(0.0, 0.25, size=4)
                    else:
                        true_delta = rng.normal(0.0, 0.15, size=4)  # minimal tactical change

                    drift_norm = float(np.linalg.norm(true_delta))
                    # True medium-term persistence scales with genuine structural tactical drift
                    true_persist = 0.02 + 0.18 * max(0.0, drift_norm - 0.3) + rng.normal(0.0, 0.05)
                    old_m = f"OldMgr_{tm[:4]}_{s_idx}"
                    new_m = f"NewMgr_{tm[:4]}_{s_idx}"

                mw_perm = mw_sack + interim_cnt + 1
                if mw_perm <= 38:
                    # Apply true structural break at mw_perm
                    z_path[mw_perm:] += true_delta

                    dt_sack = season_start + timedelta(days=(mw_sack - 1) * 7 + 3)
                    dt_perm = season_start + timedelta(days=(mw_perm - 1) * 7)

                    ev_dict = {
                        "event_id": f"EVT_{event_counter:03d}",
                        "team_id": tm,
                        "season": season,
                        "old_manager": old_m,
                        "new_manager": new_m,
                        "date_sack": pd.Timestamp(dt_sack),
                        "date_perm_first_match": pd.Timestamp(dt_perm),
                        "matchweek_sack": mw_sack,
                        "matchweek_perm": mw_perm,
                        "interim_match_count": interim_cnt,
                        "is_january_window": jan_win,
                        "is_repeat_season_sacking": False,
                        "_true_delta": true_delta,
                        "_true_persist": true_persist,
                        "_true_relief": float(rng.uniform(0.06, 0.22)),
                    }
                    event_counter += 1
                    team_event_info.setdefault(tm, []).append(ev_dict)
                    all_events.append({k: v for k, v in ev_dict.items() if not k.startswith("_")})

                    # Optional repeat-season sacking (e.g., Watford-style double sacking)
                    if (season, tm) not in FEATURED_REAL_EVENTS and is_repeat and mw_perm + 11 <= 36:
                        mw_sack2 = mw_perm + 10
                        mw_perm2 = mw_sack2 + 1
                        delta2 = rng.normal(0.0, 0.55, size=4)
                        z_path[mw_perm2:] += delta2
                        ev2 = {
                            "event_id": f"EVT_{event_counter:03d}",
                            "team_id": tm,
                            "season": season,
                            "old_manager": new_m,
                            "new_manager": f"Firefighter_{tm[:4]}_{s_idx}",
                            "date_sack": pd.Timestamp(season_start + timedelta(days=(mw_sack2 - 1) * 7 + 2)),
                            "date_perm_first_match": pd.Timestamp(season_start + timedelta(days=(mw_perm2 - 1) * 7)),
                            "matchweek_sack": mw_sack2,
                            "matchweek_perm": mw_perm2,
                            "interim_match_count": 0,
                            "is_january_window": False,
                            "is_repeat_season_sacking": True,
                            "_true_delta": delta2,
                            "_true_persist": 0.14,
                            "_true_relief": 0.10,
                        }
                        event_counter += 1
                        team_event_info[tm].append(ev2)
                        all_events.append({k: v for k, v in ev2.items() if not k.startswith("_")})

            team_latent_z[tm] = z_path

        # Simulate all 38 matchweeks
        for mw_idx, fixtures in enumerate(schedule, start=1):
            match_dt = pd.Timestamp(season_start + timedelta(days=(mw_idx - 1) * 7))

            for home_tm, away_tm in fixtures:
                match_id = f"{season}_MW{mw_idx:02d}_{match_counter:05d}"
                match_counter += 1

                h_elo_pre = club_elo[home_tm]
                a_elo_pre = club_elo[away_tm]

                h_proc, h_fin_bias, h_new_min = _team_process_modifier(team_event_info.get(home_tm, []), mw_idx, rng)
                a_proc, a_fin_bias, a_new_min = _team_process_modifier(team_event_info.get(away_tm, []), mw_idx, rng)

                # Expected non-penalty shot rates & average xG per shot from Elo + home adv + process
                elo_diff_h = (h_elo_pre - a_elo_pre) / 400.0
                base_h_xg = float(np.clip(1.45 + 0.22 + 0.75 * elo_diff_h + 0.85 * h_proc - 0.55 * a_proc, 0.35, 3.3))
                base_a_xg = float(np.clip(1.20 - 0.15 - 0.70 * elo_diff_h + 0.85 * a_proc - 0.55 * h_proc, 0.30, 3.0))

                # Possible red cards (~4% per team per match)
                red_events: list[dict[str, Any]] = []
                if rng.random() < 0.045:
                    red_events.append({"team_id": home_tm, "minute": float(rng.uniform(22.0, 82.0))})
                if rng.random() < 0.045:
                    red_events.append({"team_id": away_tm, "minute": float(rng.uniform(22.0, 82.0))})

                # Generate shot timestamps, xG, and goals for both teams
                raw_shot_specs: list[dict[str, Any]] = []
                for tm, target_xg, fin_bias, z_vec in (
                    (home_tm, base_h_xg, h_fin_bias, team_latent_z[home_tm][mw_idx]),
                    (away_tm, base_a_xg, a_fin_bias, team_latent_z[away_tm][mw_idx]),
                ):
                    n_shots = int(np.clip(rng.poisson(max(3, target_xg * 8.5)), 2, 26))
                    mean_shot_xg = target_xg / n_shots
                    chain_idx = 0
                    for _ in range(n_shots):
                        minute = float(np.clip(rng.uniform(1.0, 93.0), 1.0, 94.0))
                        is_pen = bool(rng.random() < 0.012)
                        xg = 0.78 if is_pen else float(np.clip(rng.beta(1.3, max(1.5, 1.3 / max(mean_shot_xg, 0.03) - 1.3)), 0.015, 0.85))
                        # Finishing variance shift (negative pre-sacking)
                        goal_prob = float(np.clip(xg * np.exp(0.35 * fin_bias), 0.005, 0.92))
                        is_goal = bool(rng.random() < goal_prob)
                        # Shot distance affected by ChanceProfile factor z_vec[3]
                        shot_dist = float(np.clip(16.5 + 2.2 * z_vec[3] - 8.5 * xg + rng.normal(0.0, 3.5), 4.0, 34.0))
                        chain_idx += 1
                        raw_shot_specs.append(
                            {
                                "match_id": match_id,
                                "team_id": tm,
                                "season": season,
                                "minute": minute,
                                "xg": xg,
                                "is_penalty": is_pen,
                                "is_goal": is_goal,
                                "shot_distance": shot_dist,
                                "possession_chain_id": f"{match_id}_{tm}_{chain_idx}",
                            }
                        )
                        # Occasional rebound shot in same possession chain
                        if (not is_goal) and (not is_pen) and (rng.random() < 0.08):
                            reb_xg = float(np.clip(rng.uniform(0.08, 0.35), 0.02, 0.65))
                            reb_goal = bool(rng.random() < float(np.clip(reb_xg * np.exp(0.35 * fin_bias), 0.01, 0.90)))
                            raw_shot_specs.append(
                                {
                                    "match_id": match_id,
                                    "team_id": tm,
                                    "season": season,
                                    "minute": min(94.5, minute + 0.08),
                                    "xg": reb_xg,
                                    "is_penalty": False,
                                    "is_goal": reb_goal,
                                    "shot_distance": float(np.clip(shot_dist - 3.5, 3.0, 22.0)),
                                    "possession_chain_id": f"{match_id}_{tm}_{chain_idx}",
                                }
                            )

                raw_shot_specs.sort(key=lambda s: s["minute"])
                goal_events = [{"team_id": s["team_id"], "minute": s["minute"]} for s in raw_shot_specs if s["is_goal"]]

                # Track pre-shot score differential and man differential
                hg_run, ag_run = 0, 0
                for s in raw_shot_specs:
                    m = s["minute"]
                    h_reds = sum(1 for r in red_events if r["team_id"] == home_tm and r["minute"] < m)
                    a_reds = sum(1 for r in red_events if r["team_id"] == away_tm and r["minute"] < m)
                    s["is_11v11"] = h_reds == a_reds == 0
                    if s["team_id"] == home_tm:
                        s["score_diff_pre_shot"] = hg_run - ag_run
                        s["man_diff_pre_shot"] = (11 - h_reds) - (11 - a_reds)
                    else:
                        s["score_diff_pre_shot"] = ag_run - hg_run
                        s["man_diff_pre_shot"] = (11 - a_reds) - (11 - h_reds)
                    if s["is_goal"]:
                        if s["team_id"] == home_tm:
                            hg_run += 1
                        else:
                            ag_run += 1
                all_shots.extend(raw_shot_specs)

                home_goals, away_goals = hg_run, ag_run
                match_dur = 95.0
                gs_mins = compute_game_state_minutes(match_dur, home_tm, away_tm, goal_events, red_events)

                all_matches.append(
                    {
                        "match_id": match_id,
                        "season": season,
                        "matchweek": mw_idx,
                        "date": match_dt,
                        "home_team": home_tm,
                        "away_team": away_tm,
                        "home_goals": home_goals,
                        "away_goals": away_goals,
                        "home_elo_pre": h_elo_pre,
                        "away_elo_pre": a_elo_pre,
                    }
                )

                # Compute provisional match-level tactical features from latent z + game state + Elo + noise
                shot_distances = {
                    team: np.mean([shot["shot_distance"] for shot in raw_shot_specs if shot["team_id"] == team])
                    for team in (home_tm, away_tm)
                }
                for tm, opp, is_home, gf, ga, elo_self, elo_opp, new_m in (
                    (home_tm, away_tm, True, home_goals, away_goals, h_elo_pre, a_elo_pre, h_new_min),
                    (away_tm, home_tm, False, away_goals, home_goals, a_elo_pre, h_elo_pre, a_new_min),
                ):
                    pts = 3 if gf > ga else (1 if gf == ga else 0)
                    mins = gs_mins[tm]
                    sh_tr = mins["min_trailing"] / mins["total_minutes"]
                    sh_ld = mins["min_leading"] / mins["total_minutes"]
                    sh_dn = mins["min_down_men"] / mins["total_minutes"]
                    sh_up = mins["min_up_men"] / mins["total_minutes"]
                    elo_d = (elo_self - elo_opp) / 100.0

                    z_vec = team_latent_z[tm][mw_idx]
                    # Standardized feature value = true loading * latent state + game-state/Elo bias + obs noise
                    feat_std = (
                        TRUE_FACTOR_LOADINGS @ z_vec
                        + 0.14 * (1.0 if is_home else -1.0)
                        + 0.18 * elo_d
                        + 0.45 * sh_tr
                        - 0.45 * sh_ld
                        - 0.70 * sh_dn
                        + 0.55 * sh_up
                        + rng.normal(0.0, 0.42, size=len(TACTICAL_FEATURES))
                    )
                    # Map standardized features to realistic soccer scales
                    row_dict: dict[str, Any] = {
                        "match_id": match_id,
                        "team_id": tm,
                        "opponent_id": opp,
                        "season": season,
                        "matchweek": mw_idx,
                        "date": match_dt,
                        "is_home": is_home,
                        "goals_for": gf,
                        "goals_against": ga,
                        "points": pts,
                        "elo_pre": elo_self,
                        "opp_elo_pre": elo_opp,
                        "min_11v11": mins["min_11v11"],
                        "min_11v11_comp": mins["min_11v11_comp"],
                        "min_down_men": mins["min_down_men"],
                        "min_up_men": mins["min_up_men"],
                        "min_leading": mins["min_leading"],
                        "min_trailing": mins["min_trailing"],
                        "min_tied": mins["min_tied"],
                        "total_minutes": mins["total_minutes"],
                        "new_minutes_share": new_m,
                    }
                    for f_idx, fname in enumerate(TACTICAL_FEATURES):
                        mu, sd, lo, hi = FEATURE_SCALES[fname]
                        val = float(np.clip(mu + sd * feat_std[f_idx], lo, hi))
                        if fname == "avg_shot_dist":
                            val = float(0.6 * val + 0.4 * shot_distances[tm])
                        elif fname == "opp_avg_shot_dist":
                            val = float(0.6 * val + 0.4 * shot_distances[opp])
                        row_dict[fname] = val
                    all_tms.append(row_dict)

                # Update Elo ratings post-match (K=20, home advantage +65)
                expected_h = 1.0 / (1.0 + 10.0 ** ((a_elo_pre - (h_elo_pre + 65.0)) / 400.0))
                actual_h = 1.0 if home_goals > away_goals else (0.5 if home_goals == away_goals else 0.0)
                elo_shift = 20.0 * (actual_h - expected_h)
                club_elo[home_tm] += elo_shift
                club_elo[away_tm] -= elo_shift

        # Mild summer mean reversion of ClubElo ratings between seasons
        for tm in club_elo:
            club_elo[tm] = 0.85 * club_elo[tm] + 0.15 * 1700.0

    matches_df = pd.DataFrame(all_matches)
    shots_df = pd.DataFrame(all_shots)
    tms_df = pd.DataFrame(all_tms)
    events_df = pd.DataFrame(all_events)

    tms_df = enrich_team_match_stats_with_xp(tms_df, shots_df)

    return {
        "matches": matches_df,
        "shots": shots_df,
        "team_match_stats": tms_df,
        "manager_events": events_df,
    }
