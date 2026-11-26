"""Run the matched appointment study and write its tables and charts."""

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.causal.checks import comparison_checks
from src.causal.decomposition import decompose, summarize_window
from src.causal.matching import match_events, summarize_gaps
from src.config import PipelineConfig
from src.features.residualize import MatchAdjustment
from src.features.tactics import summarize_tactics
from src.features.xp_simulator import enrich_team_match_stats_with_xp
from src.ingest.panel import PanelBundle, build_event_panel
from src.ingest.schemas import load_canonical_tables, save_canonical_tables
from src.ingest.synthetic_epl import generate_synthetic_epl_dataset
from src.viz.plots import write_report


@dataclass
class PipelineArtifacts:
    panel: PanelBundle
    matched: pd.DataFrame
    decomposition: pd.DataFrame
    event_curve: pd.DataFrame
    tactics: pd.DataFrame
    checks: dict[str, pd.DataFrame]
    summary: dict
    report_path: Path


def run_full_pipeline(config: PipelineConfig = PipelineConfig(),
                      use_existing_parquet: bool = False) -> PipelineArtifacts:
    if use_existing_parquet:
        tables = load_canonical_tables(config.processed_dir)
        # Bundled and older tables can contain simulated xP. Use their original shots.
        tables["team_match_stats"] = enrich_team_match_stats_with_xp(tables["team_match_stats"], tables["shots"])
        source = "Existing Parquet tables (the bundled dataset is synthetic)."
    else:
        tables = generate_synthetic_epl_dataset(seed=config.random_seed, seasons=config.seasons)
        save_canonical_tables(**tables, base_dir=config.processed_dir)
        source = f"Simulated seasons generated with seed {config.random_seed}."

    panel = build_event_panel(tables["team_match_stats"], tables["manager_events"], config)
    matches = panel.team_match_stats
    models = {season: MatchAdjustment.fit(matches.loc[matches.season <= season])
              for season in matches.season.unique()}
    matched = match_events(matches, panel.primary_balanced_events, models, config)
    if matched.empty:
        raise ValueError("No appointment has enough eligible pre- and post-matches and comparison teams.")
    decomposition = decompose(matched)
    curve = summarize_gaps(matched)
    checks = comparison_checks(panel, models, matched, config)
    scores = pd.concat([models[season].tactical_scores(group)
                        for season, group in matches.groupby("season")])
    tactics = summarize_tactics(matched, scores, config)
    correlation = tactics.tactical_change.corr(tactics.later_gap) if tactics.later_gap.count() >= 2 else None
    summary = {
        "source": source,
        "matches": len(tables["matches"]),
        "appointments": len(panel.manager_events),
        "eligible_appointments": len(panel.primary_balanced_events),
        "matched_appointments": matched.event_id.nunique(),
        "short_window": summarize_window(decomposition, config.short_term_window),
        "medium_window": summarize_window(decomposition, config.medium_term_window),
        "tactical_change_later_gap_correlation": float(correlation) if pd.notna(correlation) else None,
    }
    directory = config.reports_dir
    directory.mkdir(exist_ok=True, parents=True)
    report = write_report(decomposition, curve, checks, tactics, panel.manager_events,
                          directory / "case_studies.html", source)
    decomposition.to_csv(directory / "bounce_decomposition.csv", index=False)
    matched.to_csv(directory / "matched_events.csv", index=False)
    tactics.to_csv(directory / "tactical_changes.csv", index=False)
    curve.to_csv(directory / "event_curve.csv", index=False)
    (directory / "pipeline_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return PipelineArtifacts(panel, matched, decomposition, curve, tactics, checks, summary, report)
