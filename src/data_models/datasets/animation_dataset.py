import os.path
from abc import ABC
from dataclasses import dataclass
from glob import glob

from torch.utils.data import Dataset, DataLoader

from config.device_config import DeviceConfig
from data_models.dataset_metadata import DatasetMetadata
from data_models.experiment_config import ExperimentConfig


@dataclass
class AnimationDataset(ABC):
    dataset_metadata: DatasetMetadata
    device_config: DeviceConfig
    train_dataset: Dataset = None
    test_dataset: Dataset = None

    def get_train_data(self, experiment_config: ExperimentConfig):
        data_loader = DataLoader(
            self.train_dataset,
            batch_size=experiment_config.batch_size,
            shuffle=self.dataset_metadata.train_config.shuffle
        )
        return self.train_dataset, data_loader
    
    def get_test_data(self, experiment_config: ExperimentConfig):
        data_loader = DataLoader(
            self.test_dataset,
            batch_size=experiment_config.batch_size,
            shuffle=self.dataset_metadata.test_config.shuffle
        )
        return self.test_dataset, data_loader

    def _get_animation_files(self, extension: str, file_matcher: str, match_values: list[str]) -> list[str]:
        anim_files = []
        file_regex = f"*.{extension}"
        for file in glob(os.path.join(self.dataset_metadata.location, file_regex)):
            for match_val in match_values:
                no_matcher = file_matcher is None
                is_prefix_match = file_matcher == "prefix" and os.path.basename(file).startswith(match_val)
                is_suffix_match = file_matcher == "suffix" and os.path.basename(file).split(".")[0].endswith(match_val)
                if no_matcher or is_prefix_match or is_suffix_match:
                    anim_files.append(file)
        return anim_files
