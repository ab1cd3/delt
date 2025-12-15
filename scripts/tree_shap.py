# scripts/tree_shap.py

from model_eval.experiments.tree_shap_models import shap_all_tree_models_for_config
from model_eval.experiments.tree_shap_versions import shap_for_all_versions

def main():
    # SHAP for all versions: (v0, v0_bt) * (include_track_id: True/False)
    shap_for_all_versions("v0", "v0_bt")

    # SHAP for all models (base, not tuned)
    # Example using most common config in best ranking:
    # ByteTrack, no track ids, tp_count labeling with threshold 1
    shap_all_tree_models_for_config(
        pred_version="v0_bt",
        include_track_ids=False,
        label_mode="tp_count",
        tp_count_thresh=1,
        window_size=150,
        step_size=75,
        val_frac=0.3,
        save=True,
        show=False,
    )


if __name__ == "__main__":
    main()