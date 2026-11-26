"""Early tactical changes and subsequent adjusted performance, by appointment."""

import numpy as np
import pandas as pd

from src.config import PipelineConfig, TACTICAL_PILLAR_MAP


def summarize_tactics(matched: pd.DataFrame, scores: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    names = list(TACTICAL_PILLAR_MAP)
    rows = matched.merge(scores[["match_id", "team_id", *names]], on=["match_id", "team_id"],
                         validate="many_to_one")
    pre = rows.loc[rows.event_k < 0].groupby("event_id")[names].mean()
    early = rows.loc[rows.event_k.between(*config.short_term_window)].groupby("event_id")[names].mean()
    changes = (early - pre).add_prefix("delta_")
    changes["tactical_change"] = np.linalg.norm(changes.to_numpy(), axis=1)
    later = rows.loc[rows.event_k.between(*config.medium_term_window)].groupby("event_id")
    changes["later_gap"] = later["gap"].mean()
    changes["later_matches"] = later.size().reindex(changes.index, fill_value=0)
    metadata = rows.groupby("event_id")[["team_id", "season"]].first()
    return metadata.join(changes).reset_index()
