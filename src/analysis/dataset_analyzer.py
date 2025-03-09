from config.device_config import DeviceConfig
from data_models.dataset_metadata import DatasetMetadata
from data_models.datasets.bvh_dataset import BvhDataset
from data_models.experiment_config import ExperimentConfig

# TODO: refactor
_INSTANCE = None


class DatasetConfigurator:
    def __init__(
            self,
            dataset_config_path: str = "data/datasets/amass_cyprus_contemporary_sophie_test.json", # meow
            experiment_config_path: str = "experiments_configs/dataset_analysis.json",
            **kwargs
    ):
        with open(dataset_config_path, encoding="utf-8") as fp:
            self.dataset_metadata: DatasetMetadata = DatasetMetadata.from_json(fp.read())
            if self.dataset_metadata.skeleton_joint_labels:
                self.dataset_metadata.skeleton_joint_labels = {
                    int(joint_idx_str): label
                    for joint_idx_str, label in self.dataset_metadata.skeleton_joint_labels.items()
                }

        with open(experiment_config_path, encoding="utf-8") as fp:
            self.experiment_config: ExperimentConfig = ExperimentConfig.from_json(fp.read())

        self.device_config = DeviceConfig()
        dataset = BvhDataset(self.dataset_metadata, self.device_config)
        # TODO: Proper dataset extraction
        self.train_ds, self.train_loader = dataset.get_train_data(self.experiment_config)
        self.test_ds, self.test_loader = dataset.get_test_data(self.experiment_config)

    def get_dataset_metadata(self) -> DatasetMetadata:
        return self.dataset_metadata

    @staticmethod
    def get_global():
        global _INSTANCE
        if _INSTANCE is None:
            _INSTANCE = DatasetConfigurator()
        return _INSTANCE
