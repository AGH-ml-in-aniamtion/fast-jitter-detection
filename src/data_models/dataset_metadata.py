from dataclasses import dataclass, field

import numpy as np
from dataclasses_json import dataclass_json

from data_models.downloadable import Downloadable


@dataclass_json
@dataclass
class DatasetSubconfig:
    window: int
    offset: int
    start_frame: int
    match_values: list[str]
    file_matcher: str = None  # prefix/suffix
    shuffle: bool = False


@dataclass_json
@dataclass
class PreprocessingConfig:
    start_centered: bool = False
    skeleton_scale: float = None  # FIXME: Is not being applied correctly at runtime for joint lengths
    # skeleton_scale_axis: str = None


@dataclass_json
@dataclass
class FeetInfo:
    left_idx: list[int] = field(default_factory=list)
    right_idx: list[int] = field(default_factory=list)


@dataclass_json
@dataclass
class DatasetMetadata(Downloadable):
    dataset_name: str
    related_publication: str
    feet_info: FeetInfo
    location: str
    format: str  # TODO: ENUM
    fps: int
    train_config: DatasetSubconfig
    test_config: DatasetSubconfig
    preprocessing_config: PreprocessingConfig = None
    skeleton_joint_labels: dict[int, str] = None

    def download(self):
        pass  # TODO

    def load_data(self) -> np.ndarray:
        pass  # TODO, also define format

    def get_skeleton_scale(self) -> float:
        if self.preprocessing_config and self.preprocessing_config.skeleton_scale is not None:
            return self.preprocessing_config.skeleton_scale
        return 1.

    def is_start_centering(self) -> bool:
        if self.preprocessing_config:
            return self.preprocessing_config.start_centered
        return False
