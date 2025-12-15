from __future__ import annotations
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt

from model_eval.data.loaders import load_gt_pred_all_features
from model_eval.config import PRED_SOURCE, get_paths_for_version, ensure_dir, ERROR_COLORS

ERRORS = ["tp", "fp", "bg"]


def _error_counts(
    df: pd.DataFrame,
    label: str,
    column: str = "error_type",
    normalize: bool = False,
) -> pd.DataFrame:
    tmp = df[[column]].copy()
    tmp["_label"] = label
    tmp[column] = tmp[column].astype(str).str.lower()

    ct = pd.crosstab(
        index=tmp["_label"],
        columns=tmp[column],
        normalize="index" if normalize else False,
    )

    ct = ct.reindex(columns=ERRORS, fill_value=0.0 if normalize else 0)
    ct = ct.reset_index()
    ct["count_mode"] = "pct" if normalize else "abs"
    return ct


def plot_bytetrack_filtering(
        bt_version: str = "v0_bt",
        normalize_filtered: bool = True,
        save_path: Optional[Path] = None,
) -> Path:
    """
    Stacked bar plot with 3 horizontal bars error counts (tp/fp/bg):
      - Base version (subplot 1)
      - Bytetrack version (given) (subplot 1)
      - Filtered count (base - bytetrack) (normalized) (subplot 2)
    """
    base_version = bt_version.replace("_bt", "")

    df_base = load_gt_pred_all_features(pred_version=base_version)
    pred_base = df_base[df_base["source"] == PRED_SOURCE].copy()

    df_bt = load_gt_pred_all_features(pred_version=bt_version)
    pred_bt = df_bt[df_bt["source"] == PRED_SOURCE].copy()

    # Anti-join base - bt = filtered
    key_cols = ["video_series", "frame", "x_min", "y_min", "x_max", "y_max", "confidence"]
    for c in ["x_min", "y_min", "x_max", "y_max", "confidence"]:
        pred_base[c] = pred_base[c].round(2)
        pred_bt[c] = pred_bt[c].round(2)

    base_k = pred_base[key_cols + ["error_type"]].copy()
    bt_k = pred_bt[key_cols].copy()

    filtered = (
        base_k.merge(bt_k, on=key_cols, how="left", indicator=True)
              .query("_merge == 'left_only'")
              .drop(columns="_merge")
    )

    # Counts
    base_counts = _error_counts(pred_base, "YOLO (baseline)", normalize=False)
    bt_counts = _error_counts(pred_bt, "YOLO + ByteTrack", normalize=False)
    filtered_counts = _error_counts(filtered, "Filtered out by ByteTrack", normalize=normalize_filtered)

    print(f"Filtered count: {len(filtered)} | Base - ByteTrack: {len(pred_base) - len(pred_bt)}")

    # ----------------------------------------------------
    # Plot
    # ----------------------------------------------------
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw = {"height_ratios": [2, 1.5]})
    fig.suptitle(
        f"ByteTrack filtered {int(len(filtered)):,} prediction boxes", fontsize=17, fontweight="bold",
    )
    fig.subplots_adjust(top=0.88)
    combined_counts = pd.concat([bt_counts, base_counts], ignore_index=True)


    for ax, count_df in zip(axs, [combined_counts, filtered_counts]):
        mode = str(count_df["count_mode"].iloc[0])  # "abs" or "pct"

        # Values of error counts
        vals = count_df[ERRORS].copy()

        if mode == "pct":
            vals = vals * 100.0  # percent
            total_targets = [100.0] * len(vals)
            xlim = 110.0
        else:
            total_targets = vals.sum(axis=1).tolist()
            xlim = max(total_targets) * 1.15 if total_targets else 1.0

        # 1 horizontal bar per row
        for y, (_, row) in enumerate(vals.iterrows()):
            left = 0.0
            label = str(count_df.loc[count_df.index[y], "_label"])

            # Label above-left of bar
            ax.text(
                0.0, y + 0.3, label,
                ha="left", va="bottom",
                fontsize=14, fontweight="bold", color="black",
            )

            # Stacked segments
            for col in ERRORS:
                w = float(row[col])
                if w <= 0:
                    continue

                ax.barh(
                    y=y,
                    width=w,
                    left=left,
                    height=0.5,
                    color=ERROR_COLORS[col],
                    edgecolor="none",
                )

                # Centered label inside segment
                if mode == "pct":
                    seg_txt = f"{col.upper()}:\n\n{w:.0f}%"
                    show_txt = w >= 4
                    ax.set_ylim(-0.6, 0.5)
                else:
                    seg_txt = f"{col.upper()}:\n\n{int(round(w)):,}"
                    show_txt = w >= 40

                if show_txt:
                    ax.text(
                        left + w / 2.0,
                        y,
                        seg_txt,
                        ha="center",
                        va="center",
                        fontsize=12,
                        fontweight="bold",
                        color="white",
                    )

                left += w

            # total on the right
            total = total_targets[y]
            total_txt = "100%" if mode == "pct" else f"{int(total):,}"
            ax.text(
                total + (xlim * 0.01),
                y,
                total_txt,
                ha="left",
                va="center",
                fontsize=14,
                fontweight="bold",
                color="black",
            )

        # Formatting
        ax.set_yticks([])
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
        for s in ax.spines.values():
            s.set_visible(False)

    # Save
    paths = get_paths_for_version(bt_version)
    out_dir = paths.reports_experiments_dir
    ensure_dir(out_dir)

    if save_path is None:
        save_path = out_dir / f"bytetrack_filtering_{base_version}_vs_{bt_version}.png"
    else:
        save_path = Path(save_path)
        if save_path.is_dir():
            save_path = save_path / f"bytetrack_filtering_{base_version}_vs_{bt_version}.png"
        ensure_dir(save_path.parent)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"[PLOT] Saved -> {save_path}")
    return save_path