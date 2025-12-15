# src/model_eval/utils/constants.py

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

GT_SOURCE = "GT"
PRED_SOURCE = "PRED"

# FIFTYONE
FIFTYONE_DATASET_NAME = "delt_turtle"
FIFTYONE_CLASS_NAME = "turtle"

# Analysis thresholds
IOU_THRESHOLD = 0.1
TP_THRESHOLD = 2

# Metrics
METRIC_CONFIDENCE = "confidence"
METRIC_AREA = "area_px"
METRIC_MOVE_DIST = "move_dist"
METRIC_MOVE_IOU = "move_iou"