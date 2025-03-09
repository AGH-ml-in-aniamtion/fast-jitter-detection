from analysis.dataset_analysis import subsample_clip
from analysis.dataset_analyzer import DatasetConfigurator

if __name__ == "__main__":
    ds_conf = "data/datasets/publication_configs/contemporary_scale01.json"
    dataset_analyzer = DatasetConfigurator(dataset_config_path=ds_conf)
    subsample_clip(dataset_analyzer, 1, "data/datasets/bvh/Sophie_Afraid-01_scale01.bvh")
