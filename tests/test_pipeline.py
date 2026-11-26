from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from src.causal.decomposition import COMPONENTS
from src.config import PipelineConfig
from src.ingest.schemas import load_canonical_tables, save_canonical_tables
from src.pipeline import run_full_pipeline


def test_generated_and_reloaded_study(tmp_path):
    cfg = PipelineConfig(seasons=("2019-20", "2020-21"), processed_dir=tmp_path / "data",
                         reports_dir=tmp_path / "reports", near_miss_n_placebos=4)
    study = run_full_pipeline(cfg)
    assert study.report_path.stat().st_size > 1000
    assert study.summary["matched_appointments"] > 0
    assert np.allclose(study.decomposition[COMPONENTS].sum(axis=1), study.decomposition.raw_points_bounce)
    assert (study.event_curve.q75 >= study.event_curve.q25).all()
    assert not study.checks["in_time"].empty
    assert not study.checks["slump"].empty
    assert study.checks["slump"].n_events.max() <= 4
    assert json.loads((cfg.reports_dir / "pipeline_summary.json").read_text()) == study.summary

    # Existing tables may still store Monte Carlo estimates. Loading them must use their shots.
    tables = load_canonical_tables(cfg.processed_dir)
    tables["team_match_stats"]["xp"] = 0.
    save_canonical_tables(**tables, base_dir=cfg.processed_dir)
    replay = run_full_pipeline(cfg, use_existing_parquet=True)
    pd.testing.assert_frame_equal(study.decomposition.reset_index(drop=True),
                                  replay.decomposition.reset_index(drop=True), check_dtype=False)

    with pytest.raises(FileNotFoundError):
        run_full_pipeline(replace(cfg, processed_dir=tmp_path / "missing"), use_existing_parquet=True)
