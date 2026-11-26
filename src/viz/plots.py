"""An offline Plotly report with downloadable charts and a club-level breakdown."""

from pathlib import Path

import plotly.graph_objects as go
from plotly.io import to_html

from src.causal.decomposition import COMPONENTS
from src.config import TACTICAL_PILLAR_MAP

LABELS = ["Points minus expected points", "Fixture shift", "Comparison form", "Appointment gap"]


def breakdown(rows):
    figure = go.Figure()
    for column, label in zip(COMPONENTS, LABELS, strict=True):
        figure.add_bar(x=rows.event_k, y=rows[column], name=label)
    figure.add_scatter(x=rows.event_k, y=rows.raw_points_bounce, name="Points change",
                       mode="lines+markers", line_color="black")
    figure.update_layout(barmode="relative", xaxis_title="Matches since appointment",
                         yaxis_title="Change in points per match")
    return figure


def write_report(decomposition, curve, checks, tactics, events, path: Path, source: str) -> Path:
    mean = decomposition.groupby("event_k", as_index=False)[[*COMPONENTS, "raw_points_bounce"]].mean()
    overall = breakdown(mean)
    overall.update_layout(title="Where the points change comes from")

    study = go.Figure()
    study.add_scatter(x=curve.event_k, y=curve.q25, line_width=0, showlegend=False)
    study.add_scatter(x=curve.event_k, y=curve.q75, line_width=0, fill="tonexty",
                      name="Middle 50% of appointments")
    study.add_scatter(x=curve.event_k, y=curve.mean_gap, name="Appointments")
    for cohort, label in [("slump", "Slumps with manager retained"), ("in_time", "Before actual dismissal")]:
        data = checks[cohort]
        study.add_scatter(x=data.event_k, y=data.mean_gap, name=label)
    study.update_layout(title="Performance relative to matched clubs", xaxis_title="Match relative to appointment",
                        yaxis_title="Change in adjusted expected points gap")

    shifts = go.Figure()
    for name in TACTICAL_PILLAR_MAP:
        shifts.add_box(y=tactics[f"delta_{name}"], name=name, boxpoints="all")
    shifts.update_layout(title="Early tactical changes", yaxis_title="Change in standardized score")

    persistence = go.Figure(go.Scatter(x=tactics.tactical_change, y=tactics.later_gap,
                                      text=tactics.team_id + " " + tactics.season, mode="markers"))
    persistence.update_layout(title="Early tactical change and later form",
                              xaxis_title="Distance between pre-appointment and early tactical scores",
                              yaxis_title="Later change in adjusted expected points gap")

    cases = go.Figure()
    buttons = []
    names = events.set_index("event_id")
    for i, (eid, group) in enumerate(decomposition.groupby("event_id")):
        for trace in breakdown(group).data:
            trace.visible = i == 0
            cases.add_trace(trace)
        event = names.loc[eid]
        label = f"{event.team_id}, {event.season} ({event.new_manager})"
        buttons.append(dict(label=label, method="update", args=[{"visible": []}]))
    for i, button in enumerate(buttons):
        button["args"][0]["visible"] = [j // (len(COMPONENTS) + 1) == i for j in range(len(cases.data))]
    cases.update_layout(title="Individual appointments", barmode="relative",
                        xaxis_title="Matches since appointment", yaxis_title="Change in points per match",
                        updatemenus=[dict(buttons=buttons, y=1.15)])
    charts = [overall, study, shifts, persistence, cases]
    fragments = [to_html(fig, full_html=False, include_plotlyjs=(index == 0),
                         config={"toImageButtonOptions": {"format": "svg"}})
                 for index, fig in enumerate(charts)]
    path.write_text('<!doctype html><meta charset="utf-8"><title>Manager bounce</title>'
                    f'<h1>Manager bounce</h1><p>{source}</p>' + ''.join(fragments), encoding="utf-8")
    return path
