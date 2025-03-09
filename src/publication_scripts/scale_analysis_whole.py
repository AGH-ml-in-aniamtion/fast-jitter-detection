from analysis.dataset_analysis import calculate_metrics
from analysis.dataset_analyzer import DatasetConfigurator

if __name__ == "__main__":
    ds_configs = [
        "data/datasets/publication_configs/contemporary_scale1_whole.json",
        "data/datasets/publication_configs/contemporary_scale01_whole.json",
        "data/datasets/publication_configs/contemporary_scale2_whole.json",
    ]

    for ds_conf in ds_configs:
        dataset_analyzer = DatasetConfigurator(dataset_config_path=ds_conf)
        calculate_metrics(dataset_analyzer)

# Scale: x1
#   Metric: 0.40757808089256287
#   FFT: 8.108827590942383
#
# Scale: x0.1
#   Metric: 0.40757814049720764
#   FFT: 8.1088285446167
#
# Scale: x2
#   Metric: 0.40757808089256287
#   FFT: 8.108827590942383
