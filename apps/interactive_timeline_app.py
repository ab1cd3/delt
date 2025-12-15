# apps/interactive_timeline_app.py

import streamlit as st

st.set_page_config(layout="wide")

from model_eval.interactive.timeline_data import (
    load_timeline_base_df,
    list_video_series,
    get_series_frame_bounds,
    get_events_for_series,
    get_event_frame_bounds,
    prepare_timeline_views,
)
from model_eval.interactive.timeline_figure import build_timeline_figure
from model_eval.config import (
    IOU_THRESHOLD,
    LATEST_PRED_VERSION,
    METRIC_CONFIDENCE,
    METRIC_AREA,
    METRIC_MOVE_DIST,
    METRIC_MOVE_IOU,
)


def main():
    """
    Run streamlit interactive plot:
        cmd: streamlit run apps/interactive_timeline_app.py
    """
    st.title(f"Turtle events – interactive timeline (IoU threshold={IOU_THRESHOLD})")

    # Top & bottom containers in the main area
    plot_container = st.container()
    slider_container = st.container()

    # ----------------------------------------------------
    # Sidebar controls
    # ----------------------------------------------------
    with st.sidebar:
        st.header("Controls")

        pred_choice = st.text_input(
            "Prediction version", value=LATEST_PRED_VERSION
        )

    # Load data ONCE
    @st.cache_data
    def load_df(pred_version: str):
        return load_timeline_base_df(iou_thresh=IOU_THRESHOLD, pred_version=pred_version)

    df = load_df(pred_choice)

    with st.sidebar:
        # Series selection (mandatory)
        series_list = list_video_series(df)
        series = st.selectbox("Video series", series_list)

        # Event dropdown for this series (optional)
        events_df = get_events_for_series(df, series)

        event_options = ["<None>"]

        if not events_df.empty:
            event_options.extend(
                events_df["segment_id"].astype(str).unique().tolist()
            )

        event_choice = st.selectbox("Event", event_options)
        event_segment_id = None if event_choice == "<None>" else event_choice

        # Metric selection (mandatory)
        metric = st.selectbox(
            "Metric",
            [METRIC_CONFIDENCE, METRIC_AREA, METRIC_MOVE_DIST, METRIC_MOVE_IOU],
            index=0,
        )

        # GT Overlay selection (optional) -> only for non-confidence metrics
        disabled = False if metric != METRIC_CONFIDENCE else True
        show_gt_overlay = st.checkbox(
            "Show GT overlay",
            value=False,
            disabled=disabled,
        )

    # --------------------------------------------------------------
    # Frame range slider (in bottom container, appears below plot)
    # --------------------------------------------------------------
    if event_segment_id is not None:
        f_min, f_max = get_event_frame_bounds(df, series, event_segment_id)
    else:
        f_min, f_max = get_series_frame_bounds(df, series)

    with slider_container:
        left, center, right = st.columns([0.04, 0.87, 0.09])
        with center:
            frame_start, frame_end = st.slider(
                "Frame range",
                min_value=int(f_min),
                max_value=int(f_max),
                value=(int(f_min), int(f_max)),
            )

    # ----------------------------------------------------
    # Prepare data
    # ----------------------------------------------------
    df_pred, df_gt = prepare_timeline_views(
        df,
        video_series=series,
        frame_start=frame_start,
        frame_end=frame_end,
        metric=metric,
        segment_id=event_segment_id,
    )

    # ----------------------------------------------------
    # Build figure
    # ----------------------------------------------------
    if event_segment_id is not None:
        title = f"EVENT: {series}/{event_segment_id} – {metric}"
    else:
        title = f"VIDEO SERIES: {series} – {metric}"

    fig = build_timeline_figure(
        df_pred=df_pred,
        df_gt=df_gt,
        metric=metric,
        title=title,
        show_gt_overlay=show_gt_overlay,
        frame_start=frame_start,
        frame_end=frame_end,
    )

    # Plot in the top container
    with plot_container:
        st.plotly_chart(fig, width="stretch")


if __name__ == "__main__":
    main()