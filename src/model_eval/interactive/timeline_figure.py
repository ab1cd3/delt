# src/model_eval/interactive/timeline_figure.py

from __future__ import annotations

from typing import Optional
import pandas as pd
import plotly.graph_objs as go
import plotly.colors as pc

from model_eval.config import (
    ERROR_COLORS,
    METRIC_CONFIDENCE,
    METRIC_AREA,
    METRIC_MOVE_DIST,
    METRIC_MOVE_IOU,
)


def _metric_col_and_label(metric: str) -> tuple[str, str]:
    m = metric.lower()
    if m == METRIC_CONFIDENCE:
        return METRIC_CONFIDENCE, "Confidence"
    if m == METRIC_AREA:
        return METRIC_AREA, "Box area (pixels)"
    if m == METRIC_MOVE_DIST:
        return METRIC_MOVE_DIST, "Movement distance (pixels)"
    if m == METRIC_MOVE_IOU:
        return METRIC_MOVE_IOU, "Movement IoU vs previous frame"
    raise ValueError(f"Unknown metric: {metric}")


def build_timeline_figure(
        df_pred: pd.DataFrame,
        df_gt: Optional[pd.DataFrame] = None,
        metric: str = METRIC_CONFIDENCE,
        title: Optional[str] = None,
        show_gt_overlay: bool = True,
        frame_start: Optional[int] = None,
        frame_end: Optional[int] = None,
) -> go.Figure:
    """
    Build a Plotly figure for the interactive timeline.
        - PRED: bar plot colored by error_type (x=frame, y=metric).
        - GT (optional when metric != confidence):
            line plot (one per segment_id) using same metric.
    """
    metric_col, metric_label = _metric_col_and_label(metric)
    fig = go.Figure()

    # ---------------------------------------------------------------
    # PRED: bars
    # ---------------------------------------------------------------
    if df_pred is not None and not df_pred.empty:
        df_pred_plot = df_pred.copy()
        df_pred_plot["error_type"] = df_pred_plot["error_type"].str.lower()

        for err_type, color in ERROR_COLORS.items():
            df_e = df_pred_plot[df_pred_plot["error_type"] == err_type]
            if df_e.empty:
                continue

            fig.add_trace(
                go.Bar(
                    x=df_e["frame"],
                    y=df_e[metric_col],
                    name=f"PRED: {err_type.upper()}",
                    marker=dict(color=color),
                    opacity=0.7,
                    width=1,
                    hovertemplate=(
                        "Frame: %{x}<br>"
                        f"{metric_label}: " + "%{y:.3f}<br>"
                        f"Error: {err_type.upper()}<extra></extra>"
                    ),
                )
            )
    # ---------------------------------------------------------------
    # GT: lines
    # ---------------------------------------------------------------
    if (
        df_gt is not None
        and not df_gt.empty
        and metric_col != METRIC_CONFIDENCE
        and show_gt_overlay
    ):
        df_gt_plot = df_gt.copy()

        # Blue palette for GT lines (skip light tones)
        gt_palette = pc.sequential.Blues[3:]

        for i, (seg_id, g) in enumerate(df_gt_plot.groupby("segment_id")):
            color = gt_palette[i % len(gt_palette)]
            label = f"GT {seg_id}"
            fig.add_trace(
                go.Scatter(
                    x=g["frame"],
                    y=g[metric_col],
                    mode="lines",
                    name=label,
                    line=dict(color=color, width=2),
                    hovertemplate=(
                        "Frame: %{x}<br>"
                        f"{metric_label}: " + "%{y:.3f}<br>"
                        f"GT segment: {seg_id}<extra></extra>"
                    ),
                )
            )

    # ---------------------------------------------------------------
    # LAYOUT
    # ---------------------------------------------------------------
    if title is None:
        title = f"Timeline: {metric_label}"

    fig.update_layout(
        title=title,
        xaxis_title="Frame",
        yaxis_title=metric_label,
        barmode="overlay",
        showlegend=True,
        legend_title_text="ERRORS",
    )
    fig.update_xaxes(showgrid=True, zeroline=False)
    fig.update_yaxes(showgrid=True, zeroline=True, zerolinewidth=1)

    if frame_start is not None and frame_end is not None:
        fig.update_xaxes(range=[int(frame_start), int(frame_end)])

    return fig