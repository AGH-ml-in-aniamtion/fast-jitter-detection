import warnings
from functools import partial

import numpy as np
import torch

from analysis.dataset_analysis import evaluate_custom_jitter, mdcss_classify
from analysis.dataset_analyzer import DatasetConfigurator
from metrics.dataset_metrics import torch_mdcss
from publication_scripts.lafan1_custom_noise_redone_metrics import pca_pred, pca_eval, savgol_eval, \
    savgol_pred

warnings.filterwarnings("ignore")


def evaluate_model(dataset_analyzer: DatasetConfigurator, model_name='mdcss', verbose=False):
    highlight_frames = []
    jitter_std_vals = [
        0.02,
        0.0175,
        0.015,
        0.0125,
        0.01,
        0.0075,
        0.005,
    ]

    model_trainer = __MODEL_TO_TRAIN_MAPPING[model_name]

    # Training
    model, extra_kwargs = model_trainer(dataset_analyzer) if model_trainer is not None else (None, {})
    eval_func, pred_func = __MODEL_TO_EVAL_MAPPING[model_name]  # Partial stuff for non mdcss
    eval_func = partial(eval_func, model=model)

    print(f"Testing: {model_name}")
    print(f"---------------------------------------------")
    reps = 10
    for std_i, j_std in enumerate(jitter_std_vals):
        auc_roc_total = 0
        f1_total = 0
        tpr_total = 0
        precision_total = 0
        for i in range(reps):
            seed = i * 100
            torch.manual_seed(seed)
            np.random.seed(seed + 1)
            with torch.no_grad():
                auc_roc_result, f1_result, tpr_result, precision_result = evaluate_custom_jitter(
                    dataset_analyzer,
                    eval_func,
                    pred_func,
                    n=2,
                    custom_noise_windows=highlight_frames,
                    analyze_errors_and_warnings=True,
                    analyze_noise_windows=True,
                    jitter_std=j_std,
                    verbose=verbose,
                    run_smoke_test=(i == 0 and std_i == 0),
                    **extra_kwargs,
                )

                auc_roc_total += auc_roc_result
                f1_total += f1_result
                tpr_total += tpr_result
                precision_total += precision_result

        print(
            f"std: {j_std}, "
            f"AOC ROC: {auc_roc_total / reps}, "
            f"f1: {f1_total / reps}, "
            f"sensitivity (TPR): {tpr_total / reps}, "
            f"precision: {precision_total / reps}"
        )

    print(f"---------------------------------------------")


def main():
    ds_config = "data/datasets/publication_configs/lafan1_custom_jitter_whole_comparison.json"
    dataset_analyzer = DatasetConfigurator(dataset_config_path=ds_config)
    model_names = list(__MODEL_TO_TRAIN_MAPPING.keys())
    for model_name in model_names:
        evaluate_model(dataset_analyzer, model_name)


__MODEL_TO_TRAIN_MAPPING = {
    # 'mdcss': None,
    # 'pca': fit_to_train_pca,
    'savgol': None,
}

__MODEL_TO_EVAL_MAPPING = {
    'mdcss': (torch_mdcss, mdcss_classify),
    'pca': (pca_eval, pca_pred),
    'savgol': (savgol_eval, savgol_pred),
}

if __name__ == "__main__":
    main()

# Testing: mdcss
# ---------------------------------------------
# Smoke test detection result: 5 errors detected
# Expected following window: [3915, 3930] => detected
# Expected following window: [4630, 4645] => detected
# Expected following window: [5608, 5623] => detected
# Expected following window: [6414, 6429] => detected
# Expected following window: [6870, 6885] => detected
# Incorrect detections: 0
# All detections: [[3922]
#  [4638]
#  [5616]
#  [6421]
#  [6877]]
# std: 0.02, AOC ROC: 0.9997410142413481, f1: 0.831647126404266, sensitivity (TPR): 0.9550085588254656, precision: 0.7367325093034973
# std: 0.0175, AOC ROC: 0.9997356308494183, f1: 0.852030660056804, sensitivity (TPR): 0.927049180327869, precision: 0.7885856254420063
# std: 0.015, AOC ROC: 0.9997247706820296, f1: 0.8568452325529721, sensitivity (TPR): 0.8688524590163935, precision: 0.8458207697058521
# std: 0.0125, AOC ROC: 0.9996911492507493, f1: 0.823233363555859, sensitivity (TPR): 0.7565573770491805, precision: 0.9039365481105536
# std: 0.01, AOC ROC: 0.9995573545486796, f1: 0.6828457670517931, sensitivity (TPR): 0.5360655737704919, precision: 0.9417989770867665
# std: 0.0075, AOC ROC: 0.9991393843293984, f1: 0.3322378876639087, sensitivity (TPR): 0.20040983606557372, precision: 0.9782743373016579
# std: 0.005, AOC ROC: 0.9966505510127321, f1: 0.012205077152812346, sensitivity (TPR): 0.006147540983606558, precision: 0.9
# ---------------------------------------------


# Preparing data for PCA...
# Fitting PCA...
# PCA ready for evaluation
# Testing: pca
# ---------------------------------------------
# Smoke test detection result: 9 errors detected
# Expected following window: [3915, 3930] => detected
# Expected following window: [4630, 4645] => detected
# Expected following window: [5608, 5623] => detected
# Expected following window: [6414, 6429] => detected
# Expected following window: [6870, 6885] => detected
# Incorrect detections: 0
# All detections: [[3922]
#  [4638]
#  [5616]
#  [6421]
#  [6877]
#  [6878]
#  [6879]
#  [6880]
#  [6881]]
# std: 0.02, AOC ROC: 0.9998135413156307, f1: 0.7782653267407655, sensitivity (TPR): 0.996311475409836, precision: 0.6386436976428127
# std: 0.0175, AOC ROC: 0.9998134007271806, f1: 0.8235975124985915, sensitivity (TPR): 0.9901639344262294, precision: 0.7051233896520573
# std: 0.015, AOC ROC: 0.9998134063739927, f1: 0.8699273062749058, sensitivity (TPR): 0.972950819672131, precision: 0.7868704687164444
# std: 0.0125, AOC ROC: 0.9998125246919173, f1: 0.882698881182779, sensitivity (TPR): 0.8840163934426231, precision: 0.8824117434469473
# std: 0.01, AOC ROC: 0.9998075292768778, f1: 0.7228977057813459, sensitivity (TPR): 0.5860655737704918, precision: 0.945481887209303
# std: 0.0075, AOC ROC: 0.9997572093566447, f1: 0.2003967471451386, sensitivity (TPR): 0.11188524590163933, precision: 0.9738787748982652
# std: 0.005, AOC ROC: 0.998857695186965, f1: 0.0, sensitivity (TPR): 0.0, precision: 0.0
# ---------------------------------------------


# Testing: savgol
# ---------------------------------------------
# Smoke test detection result: 9 errors detected
# Expected following window: [3915, 3930] => detected
# Expected following window: [4630, 4645] => detected
# Expected following window: [5608, 5623] => detected
# Expected following window: [6414, 6429] => detected
# Expected following window: [6870, 6885] => detected
# Incorrect detections: 0
# All detections: [[3922]
#  [4638]
#  [5616]
#  [6421]
#  [6877]
#  [6878]
#  [6879]
#  [6880]
#  [6881]]
# std: 0.02, AOC ROC: 0.9998102939586463, f1: 0.7775420283557557, sensitivity (TPR): 0.9971311475409836, precision: 0.6373259292503572
# std: 0.0175, AOC ROC: 0.9998098546237466, f1: 0.8230749111942485, sensitivity (TPR): 0.9905737704918034, precision: 0.7042590086379287
# std: 0.015, AOC ROC: 0.9998098667312927, f1: 0.8679780288137199, sensitivity (TPR): 0.9696721311475409, precision: 0.7858650786314296
# std: 0.0125, AOC ROC: 0.9998096565655293, f1: 0.8835422281846729, sensitivity (TPR): 0.8844262295081966, precision: 0.8835941154900994
# std: 0.01, AOC ROC: 0.9998061785890631, f1: 0.7143743734250354, sensitivity (TPR): 0.5737704918032785, precision: 0.9495245569624569
# std: 0.0075, AOC ROC: 0.9997695835304032, f1: 0.18888769544641548, sensitivity (TPR): 0.10491803278688525, precision: 0.9689259834368531
# std: 0.005, AOC ROC: 0.9990320871726981, f1: 0.0008163265306122449, sensitivity (TPR): 0.0004098360655737705, precision: 0.1
# ---------------------------------------------
