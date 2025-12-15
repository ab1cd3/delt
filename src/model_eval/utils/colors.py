# src/model_eval/utils/colors.py

GT_COLOR = "#4CAF50"    # green
PRED_COLOR  = "#4292C6" # blue

OVERLAY_GT_COLOR = "#00FF00" # bright green
OVERLAY_PRED_COLOR = "#0000FF" #bright blue

TP_COLOR = GT_COLOR    # green
FP_COLOR = "#F44336"   # red
FN_COLOR = "#FF9800"   # orange
BG_COLOR = "#C78CDB"   # purple (background/non-event segments)

ERROR_COLORS = {
    "tp": TP_COLOR,
    "fp": FP_COLOR,
    "fn": FN_COLOR,
    "bg": BG_COLOR,
}