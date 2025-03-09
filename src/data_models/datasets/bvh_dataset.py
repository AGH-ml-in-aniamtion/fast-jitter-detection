import os
from dataclasses import dataclass
from pprint import pprint
from random import random

import numpy as np
import torch
from torch.utils.data import Dataset

from config.device_config import DeviceConfig
from data_models.dataset_metadata import DatasetMetadata, FeetInfo
from data_models.datasets.animation_dataset import AnimationDataset
from data_models.parsers import bvh_parser
from data_models.parsers.bvh_parser import Animation
from utils.numpy_utils import np_anim


class BvhDataset(AnimationDataset):

    def __init__(
            self,
            dataset_metadata: DatasetMetadata,
            device_config: DeviceConfig
    ):
        super(BvhDataset, self).__init__(dataset_metadata, device_config)

        skeleton_scale = dataset_metadata.get_skeleton_scale()

        self.train_dataset = _BvhDataSet(
            bvh_files_to_load=self._get_animation_files(
                extension="bvh",
                file_matcher=self.dataset_metadata.train_config.file_matcher,
                match_values=self.dataset_metadata.train_config.match_values
            ),
            feet_info=self.dataset_metadata.feet_info,
            window=self.dataset_metadata.train_config.window,
            offset=self.dataset_metadata.train_config.offset,
            start_frame=self.dataset_metadata.train_config.start_frame,
            device=self.device_config.device,
            dtype=self.device_config.dtype,
            skeleton_scale=skeleton_scale
        )

        self.test_dataset = _BvhDataSet(
            bvh_files_to_load=self._get_animation_files(
                extension="bvh",
                file_matcher=self.dataset_metadata.test_config.file_matcher,
                match_values=self.dataset_metadata.test_config.match_values
            ),
            feet_info=self.dataset_metadata.feet_info,
            window=self.dataset_metadata.test_config.window,
            offset=self.dataset_metadata.test_config.offset,
            start_frame=self.dataset_metadata.test_config.start_frame,
            device=self.device_config.device,
            dtype=self.device_config.dtype,
            skeleton_scale=skeleton_scale
        )

        print("{} clips in train dataset.".format(len(self.train_dataset)))
        print("{} clips in test dataset.".format(len(self.test_dataset)))


class ShuffledBvhDataset(BvhDataset):

    def __init__(
            self,
            dataset_metadata: DatasetMetadata,
            device_config: DeviceConfig
    ):
        super(ShuffledBvhDataset, self).__init__(dataset_metadata, device_config)

        skeleton_scale = dataset_metadata.get_skeleton_scale()

        self.train_dataset = _BvhDataSet(
            bvh_files_to_load=self._get_animation_files(
                extension="bvh",
                file_matcher=self.dataset_metadata.train_config.file_matcher,
                match_values=self.dataset_metadata.train_config.match_values
            ),
            feet_info=self.dataset_metadata.feet_info,
            window=self.dataset_metadata.train_config.window,
            offset=self.dataset_metadata.train_config.offset,
            start_frame=self.dataset_metadata.train_config.start_frame,
            device=self.device_config.device,
            dtype=self.device_config.dtype,
            skeleton_scale=skeleton_scale
        )

        self.test_dataset = _BvhDataSet(
            bvh_files_to_load=self._get_animation_files(
                extension="bvh",
                file_matcher=self.dataset_metadata.test_config.file_matcher,
                match_values=self.dataset_metadata.test_config.match_values
            ),
            feet_info=self.dataset_metadata.feet_info,
            window=self.dataset_metadata.test_config.window,
            offset=self.dataset_metadata.test_config.offset,
            start_frame=self.dataset_metadata.test_config.start_frame,
            device=self.device_config.device,
            dtype=self.device_config.dtype,
            skeleton_scale=skeleton_scale
        )

        print("{} clips in train dataset.".format(len(self.train_dataset)))
        print("{} clips in test dataset.".format(len(self.test_dataset)))


# TODO
# Credit to Motion In-betweening via Two-stage Transformers
class _BvhDataSet(Dataset):
    def __init__(self, bvh_files_to_load, feet_info: FeetInfo,
                 window=50, offset=1,
                 start_frame=0, device="cuda:0", dtype=torch.float32,
                 skeleton_scale: float = 1.,
                 **kwargs):
        """
        Bvh data set.

        Args:
            bvh_folder (str): Bvh folder path.
            actors (list of str): List of actors to be included in the dataset.
            window (int, optional): Length of window. Defaults to 50.
            offset (int, optional): Offset of window. Defaults to 1.
            start_frame (int, optional):
                Override the start frame of each bvh file. Defaults to 0.
            device (str, optional): Device. e.g. "cpu", "cuda:0".
                Defaults to "cpu".
            dtype: torch.float16, torch.float32, torch.float64 etc.
        """
        super(_BvhDataSet, self).__init__()
        self.bvh_files_to_load = bvh_files_to_load
        self.window = window
        self.offset = offset
        self.start_frame = start_frame
        self.device = device
        self.dtype = dtype
        self.skeleton_scale = skeleton_scale
        self.parents = []
        self.bone_lengths = []
        self.feet_info = feet_info
        self.sequence_names = []

        self.load_bvh_files()

    def _to_tensor(self, array):
        return torch.tensor(array, dtype=self.dtype, device=self.device)

    def load_bvh_files(self):
        self.positions = []
        self.rotations = []
        self.skeleton_joint_offsets = []
        self.foot_contact = []
        self.frames = []
        self.parents = []


        if not self.bvh_files_to_load:
            raise FileNotFoundError(
                f"No bvh files found in {self.bvh_files_to_load}."
            )

        self.bvh_files_to_load.sort()
        print("Processing files:")
        pprint(self.bvh_files_to_load)
        for bvh_path in self.bvh_files_to_load:
            anim = bvh_parser.load_bvh(bvh_path, start=self.start_frame)
            self.bone_lengths.append(self._calculate_bone_lengths(anim))

            # global joint rotation, position
            gr, gp = np_anim.fk(anim.rotations, anim.positions, anim.parents)

            # left, right foot contact
            cl, cr = np_anim.extract_feet_contacts(
                gp, self.feet_info.left_idx, self.feet_info.right_idx, vel_threshold=0.2)

            self.positions.append(self._to_tensor(anim.positions * self.skeleton_scale))
            self.rotations.append(self._to_tensor(anim.rotations))
            self.skeleton_joint_offsets.append(anim.offsets * self.skeleton_scale)
            self.foot_contact.append(self._to_tensor(
                np.concatenate([cl, cr], axis=-1)))
            self.frames.append(anim.positions.shape[0])
            self.parents.append(anim.parents)

            sequence_name = os.path.basename(bvh_path)
            self.sequence_names.append(sequence_name)

    def _calculate_bone_lengths(self, animation: Animation):
        parents = animation.parents
        offsets = animation.offsets
        bones_lengths = np.zeros(len(offsets))
        lengths_based_on_offsets = np.linalg.norm(offsets, axis=-1)

        # find parent chain
        parent_stack = []
        previous_parent_idx = -1
        for i, parent_idx in enumerate(parents):
            if i == len(parents) - 1:
                parent_stack.append((i, parent_idx))

            if previous_parent_idx > parent_idx or i == len(parents) - 1:
                # defuse stack
                for child_idx, _ in reversed(parent_stack[1:]):
                    bones_lengths[child_idx] = lengths_based_on_offsets[child_idx]
                parent_stack = []

            if i > 0:
                parent_stack.append((i, parent_idx))

            previous_parent_idx = parent_idx
        bones_lengths *= self.skeleton_scale
        return self._to_tensor(bones_lengths)

    def __len__(self):
        if self.window is None:
            return len(self.positions)
        count = 0
        for frame in self.frames:
            count += int(float(frame - self.window) / self.offset) + 1
        return count

    def __getitem__(self, idx):
        if self.window is None:
            return (
                self.positions[idx],
                self.rotations[idx],
                self.skeleton_joint_offsets[idx],
                self.foot_contact[idx],
                self.parents[idx],
                self.bone_lengths[idx],
                self.sequence_names[idx],
                idx
            )

        curr_idx = idx

        for i, frame in enumerate(self.frames):
            tmp_idx = curr_idx - \
                      int(float(frame - self.window) / self.offset) - 1

            if tmp_idx >= 0:
                curr_idx = tmp_idx
                continue

            start_idx = curr_idx * self.offset
            end_idx = start_idx + self.window


            positions = self.positions[i][start_idx:end_idx]
            rotations = self.rotations[i][start_idx:end_idx]
            joint_offsets = self.skeleton_joint_offsets[i]
            foot_contact = self.foot_contact[i][start_idx:end_idx]
            parents = self.parents[i]
            bone_lengths = self.bone_lengths[i]
            sequence_name = self.sequence_names[i]
            return (
                positions,
                rotations,
                joint_offsets,
                foot_contact,
                parents,
                bone_lengths,
                sequence_name,
                idx
            )


@dataclass
class JunctionConfig:
    parent: int
    children: list[int]

    def shuffle(self):
        random.shuffle(self.children)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.parent == other.parent and self.children == other.children

    def copy(self):
        return JunctionConfig(
            parent=self.parent,
            children=self.children.copy()
        )



def shuffle_animation(data, skel_shuffle_config: dict[int, JunctionConfig]):
    parents_ref = data.parents.copy()
    positions = data.positions.clone()
    rotations = data.rotations.clone()
    g_positions = data.global_positions.clone()
    g_rotations = data.global_rotations.clone()
    parents = data.parents.copy()

    children_mapping = __parents_to_children_map(parents_ref)
    joint_stack = [(-1, 0)]
    retargeted_joint = 0

    while joint_stack:
        (parent, cur_joint) = joint_stack.pop()
        if cur_joint in skel_shuffle_config:
            joint_stack.extend(reversed([(retargeted_joint, c) for c in skel_shuffle_config[cur_joint].children]))
        elif children_mapping[cur_joint]:
            joint_stack.append((retargeted_joint, children_mapping[cur_joint][0]))

        positions[:, retargeted_joint] = data.positions[:, cur_joint]
        rotations[:, retargeted_joint] = data.rotations[:, cur_joint]
        g_positions[:, retargeted_joint] = data.global_positions[:, cur_joint]
        g_rotations[:, retargeted_joint] = data.global_rotations[:, cur_joint]
        parents[retargeted_joint] = parent
        retargeted_joint += 1

    data.positions = positions
    data.rotations = rotations
    data.global_positions = g_positions
    data.global_rotations = g_rotations
    data.parents = parents

def __parents_to_children_map(parents):
    children_mapping = {j: [] for j in range(len(parents))}
    for j, p in enumerate(parents):
        if p not in children_mapping:
            continue  # skip root
        children_mapping[p].append(j)
    return children_mapping

# FIXME: maybe useful?
def generate_shuffle_config(parents, shuffle_variant_count: int, shuffle_depth: int = None) -> list[
    dict[int, JunctionConfig]
]:
    shuffle_configs = []
    regular_hierarchy_variant = {}
    for par in range(len(parents)):
        children = [i for i, x in enumerate(parents) if x == par]
        if len(children) > 1:
            regular_hierarchy_variant[par] = (
                JunctionConfig(
                    parent=par,
                    children=children
                )
            )

    for variant in range(shuffle_variant_count):
        new_variant = {
            parent: joint_junction.copy()
            for parent, joint_junction in regular_hierarchy_variant.items()
        }
        while new_variant == regular_hierarchy_variant:
            for joint_junction in new_variant.values():
                joint_junction.shuffle()
        shuffle_configs.append(new_variant)
    return shuffle_configs
