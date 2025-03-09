from analysis.dataset_analysis import calculate_metrics_with_random_noise
from analysis.dataset_analyzer import DatasetConfigurator


if __name__ == "__main__":
    ds_configs = [
        "data/datasets/publication_configs/lafan1_custom_jitter_whole.json",
    ]

    highlight_frames = []
    jitter_std_vals = [0.02, 0.0175, 0.015, 0.0125, 0.01, 0.0075, 0.005]
    for j_std in jitter_std_vals:
        detection = 0
        for ds_conf in ds_configs:
            dataset_analyzer = DatasetConfigurator(dataset_config_path=ds_conf)

            for i in range(10):
                detection_rate, _, _ = calculate_metrics_with_random_noise(
                    dataset_analyzer,
                    n=2,
                    noise_windows=highlight_frames,
                    analyze_errors_and_warnings=True,
                    analyze_noise_windows=True,
                    jitter_std=j_std
                )
                detection += detection_rate

            print(f"std: {j_std}, detection rate: {detection / 10}")

# STD: 0.02S
# MDC metric (averaged over joints and then over frames): 1.3105781078338623
# MDCSS metric (averaged over joints and then over frames): 3.2358696460723877
# Error windows: 12577
# Warning windows: 13895
# Detected windows: 1605/1639
# std: 0.02, detection rate: 0.9825503355704699
#
# STD: 0.0175S
# MDC metric (averaged over joints and then over frames): 1.1528573036193848
# MDCSS metric (averaged over joints and then over frames): 2.8840553760528564
# Error windows: 12389
# Warning windows: 14976
# Detected windows: 1591/1639
# std: 0.0175, detection rate: 0.9749847467968273
#
# STD: 0.015S
# MDC metric (averaged over joints and then over frames): 0.9772948026657104
# MDCSS metric (averaged over joints and then over frames): 2.4775991439819336
# Error windows: 10951
# Warning windows: 15160
# Detected windows: 1570/1639
# std: 0.015, detection rate: 0.9533251982916411
#
# STD: 0.0125S
# MDC metric (averaged over joints and then over frames): 0.8064149618148804
# MDCSS metric (averaged over joints and then over frames): 2.0933988094329834
# Error windows: 8271
# Warning windows: 14922
# Detected windows: 1492/1639
# std: 0.0125, detection rate: 0.9040878584502747
#
# STD: 0.01S
# MDC metric (averaged over joints and then over frames): 0.65614914894104
# MDCSS metric (averaged over joints and then over frames): 1.7423830032348633
# Error windows: 4575
# Warning windows: 14025
# Detected windows: 1268/1639
# std: 0.01, detection rate: 0.7520439292251374
#
# STD: 0.0075S
# MDC metric (averaged over joints and then over frames): 0.48685213923454285
# MDCSS metric (averaged over joints and then over frames): 1.3526970148086548
# Error windows: 1165
# Warning windows: 11628
# Detected windows: 611/1639
# std: 0.0075, detection rate: 0.35790115924344107
#
# STD: 0.005S
# MDC metric (averaged over joints and then over frames): 0.3241826593875885
# MDCSS metric (averaged over joints and then over frames): 0.9740831851959229
# Error windows: 33
# Warning windows: 6160
# Detected windows: 22/1639
# std: 0.005, detection rate: 0.015070164734594266
