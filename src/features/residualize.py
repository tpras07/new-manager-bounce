"""Fit fixture and game-state adjustments on seasons without a manager change."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import GAME_STATE_MINUTES, REVERSE_FEATURES, TACTICAL_FEATURES, TACTICAL_PILLAR_MAP


def fixture_design(rows: pd.DataFrame, elo) -> np.ndarray:
    home = rows["is_home"].to_numpy(dtype=float)
    difference = (elo - rows["opp_elo_pre"].to_numpy()) / 100
    return np.column_stack([np.ones(len(rows)), home, difference, home * difference])


def tactical_design(rows: pd.DataFrame) -> np.ndarray:
    minutes = rows[list(GAME_STATE_MINUTES)]
    shares = minutes.div(rows["total_minutes"], axis=0)
    return np.column_stack([fixture_design(rows, rows["elo_pre"].to_numpy()), shares])


@dataclass
class MatchAdjustment:
    outcome_coef: np.ndarray
    tactical_coef: np.ndarray
    tactical_scale: np.ndarray

    @classmethod
    def fit(cls, matches: pd.DataFrame) -> "MatchAdjustment":
        controls = matches.loc[matches["is_non_treated_season"]]
        if controls.empty:
            raise ValueError("Fixture adjustment requires team-seasons without a manager change.")
        design = fixture_design(controls, controls["elo_pre"].to_numpy())
        outcomes = controls[["xp", "npxgd_11v11"]].to_numpy()
        coef = np.linalg.lstsq(design, outcomes, rcond=None)[0]
        design = tactical_design(controls)
        features = controls[list(TACTICAL_FEATURES)].to_numpy()
        tactical_coef = np.linalg.lstsq(design, features, rcond=None)[0]
        scale = (features - design @ tactical_coef).std(axis=0)
        return cls(coef, tactical_coef, np.where(scale > 1e-10, scale, np.inf))

    def outcomes(self, rows: pd.DataFrame, elo) -> pd.DataFrame:
        predicted = fixture_design(rows, elo) @ self.outcome_coef
        xp_pred = predicted[:, 0].clip(0, 3)
        return rows.assign(xp_elo_pred=xp_pred, xp_resid=rows["xp"] - xp_pred,
                           npxgd_resid=rows["npxgd_11v11"] - predicted[:, 1])

    def tactical_scores(self, rows: pd.DataFrame) -> pd.DataFrame:
        observed = rows[list(TACTICAL_FEATURES)].to_numpy()
        residuals = (observed - tactical_design(rows) @ self.tactical_coef) / self.tactical_scale
        scores = pd.DataFrame(residuals, columns=TACTICAL_FEATURES, index=rows.index)
        scores[list(REVERSE_FEATURES)] *= -1
        return rows.assign(**{name: scores[list(features)].mean(axis=1)
                              for name, features in TACTICAL_PILLAR_MAP.items()})
