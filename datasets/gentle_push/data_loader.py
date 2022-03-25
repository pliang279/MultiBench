"""Implements dataloaders for Gentle Push tasks.

Sourced from https://github.com/brentyi/multimodalfilter/blob/master/crossmodal/tasks/_push.py 
"""


import argparse
import sys
from typing import Any, Dict, List, NamedTuple, Tuple

from PIL.Image import NONE

import fannypack
import fannypack as fp
import numpy as np
import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from robustness.visual_robust import add_visual_noise
from robustness.timeseries_robust import add_timeseries_noise

dataset_urls = {
    # Mujoco URLs
    "gentle_push_10.hdf5": "https://drive.google.com/file/d/1qmBCfsAGu8eew-CQFmV1svodl9VJa6fX/view?usp=sharing",
    "gentle_push_100.hdf5": "https://drive.google.com/file/d/1PmqQy5myNXSei56upMy3mXKu5Lk7Fr_g/view?usp=sharing",
    "gentle_push_300.hdf5": "https://drive.google.com/file/d/18dr1z0N__yFiP_DAKxy-Hs9Vy_AsaW6Q/view?usp=sharing",
    "gentle_push_1000.hdf5": "https://drive.google.com/file/d/1JTgmq1KPRK9HYi8BgvljKg5MPqT_N4cR/view?usp=sharing",
    # Real data (kloss_dataset=True)
    "kloss_train0.hdf5": "https://drive.google.com/file/d/1nk4BO0rcVTKw22vYq6biewiwAFUPevM1/view?usp=sharing",
    "kloss_train1.hdf5": "https://drive.google.com/file/d/1gBWoB2PCrgYlLjuDJQm6BFAf_xwMqRxa/view?usp=sharing",
    "kloss_train2.hdf5": "https://drive.google.com/file/d/15W2zj52bSITxIRVRi7ajehAmz14RU33M/view?usp=sharing",
    "kloss_train3.hdf5": "https://drive.google.com/file/d/1WhRFu4SDlIYKnLYLyDdgOQYjP20JOTLE/view?usp=sharing",
    "kloss_train4.hdf5": "https://drive.google.com/file/d/1-ur_hzyBvd1_QCLTamaO8eWJ7rXii7y4/view?usp=sharing",
    "kloss_train5.hdf5": "https://drive.google.com/file/d/1ni8vEy4c1cmCKP2ZlWfXqLo7a4sdRFwe/view?usp=sharing",
    "kloss_val.hdf5": "https://drive.google.com/file/d/1-CRocf7I4mTLBp7Tjo7-D-QvkwcGZkNo/view?usp=sharing",
}


class TrajectoryNumpy(NamedTuple):
    """Named tuple containing states, observations, and controls represented in NumPy."""

    states: Any
    observations: Any
    controls: Any


class PushTask():
    """Dataset definition and model registry for pushing task."""

    @classmethod
    def add_dataset_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add dataset options to an argument parser.

        Args:
            parser (argparse.ArgumentParser): Parser to add arguments to.
        """
        parser.add_argument("--no_vision", action="store_true")
        parser.add_argument("--no_proprioception", action="store_true")
        parser.add_argument("--no_haptics", action="store_true")
        parser.add_argument("--image_blackout_ratio", type=float, default=0.0)
        parser.add_argument("--sequential_image_rate", type=int, default=1)
        parser.add_argument("--kloss_dataset", action="store_true")

    @classmethod
    def get_dataset_args(cls, args: argparse.Namespace) -> Dict[str, Any]:
        """Get a dataset_args dictionary from a parsed set of arguments.

        Args:
            args (argparse.Namespace): Parsed arguments.
        """
        dataset_args = {
            "use_vision": not args.no_vision,
            "use_proprioception": not args.no_proprioception,
            "use_haptics": not args.no_haptics,
            "image_blackout_ratio": args.image_blackout_ratio,
            "sequential_image_rate": args.sequential_image_rate,
            "kloss_dataset": args.kloss_dataset,
        }
        return dataset_args

    @classmethod
    def get_dataloader(
        cls, subsequence_length: int, modalities=None, batch_size=32, drop_last=True, **dataset_args
    ):
        """Get dataloaders for gentle-push dataset.

        Args:
            subsequence_length (int): Length of subsequences for each sample
            modalities (list, optional): List of strings defining which modalities to use. Defaults to None. Pick from ['gripper_pos', 'gripper_sensors', 'image', 'control'].
            batch_size (int, optional): Batch size. Defaults to 32.
            drop_last (bool, optional): Whether to drop last datasample or not. Defaults to True.

        Returns:
            tuple: Tuple of training dataloader, validation dataloader, and test dataloader
        """
        # Load trajectories into memory
        train_trajectories = cls.get_train_trajectories(**dataset_args)
        val_trajectories = cls.get_eval_trajectories(**dataset_args)
        test_trajectories = cls.get_test_trajectories(
            modalities, **dataset_args)
        train_loader = DataLoader(
            SubsequenceDataset(train_trajectories,
                               subsequence_length, modalities),
            batch_size=batch_size,
            shuffle=True,
            drop_last=drop_last,
        )
        val_loader = DataLoader(
            SubsequenceDataset(
                val_trajectories, subsequence_length, modalities),
            batch_size=batch_size,
            shuffle=True,
        )
        test_loader = dict()
        for modality, trajectories in test_trajectories.items():
            test_loader[modality] = []
            for traj in trajectories:
                test_loader[modality].append(DataLoader(
                    SubsequenceDataset(traj, subsequence_length, modalities),
                    batch_size=batch_size,
                    shuffle=False,
                )
                )
        return train_loader, val_loader, test_loader

    @classmethod
    def get_train_trajectories(
        cls, **dataset_args
    ) -> List[TrajectoryNumpy]:
        """Get training trajectories."""
        kloss_dataset = (
            dataset_args["kloss_dataset"] if "kloss_dataset" in dataset_args else False
        )
        if kloss_dataset:
            return _load_trajectories(
                "kloss_train0.hdf5",
                "kloss_train1.hdf5",
                "kloss_train2.hdf5",
                "kloss_train3.hdf5",
                "kloss_train4.hdf5",
                "kloss_train5.hdf5",
                **dataset_args,
            )
        else:
            return _load_trajectories("gentle_push_1000.hdf5", **dataset_args)

    @classmethod
    def get_eval_trajectories(
        cls, **dataset_args
    ) -> List[TrajectoryNumpy]:
        """Get evaluation trajectories."""
        kloss_dataset = (
            dataset_args["kloss_dataset"] if "kloss_dataset" in dataset_args else False
        )
        if kloss_dataset:
            return _load_trajectories(("kloss_val.hdf5", 50), **dataset_args)
        else:
            return _load_trajectories("gentle_push_10.hdf5", **dataset_args)

    @classmethod
    def get_test_trajectories(
        cls, modalities, **dataset_args
    ):
        """Get test trajectories."""
        kloss_dataset = (
            dataset_args["kloss_dataset"] if "kloss_dataset" in dataset_args else False
        )
        if kloss_dataset:
            raise Exception('No test dataset for kloss')
        else:
            trajectories = dict()
            if modalities == None or 'image' in modalities:
                trajectories['image'] = []
                for i in range(10):
                    trajectories['image'].append(_load_trajectories(
                        "gentle_push_300.hdf5", visual_noise=i/10, **dataset_args))
            if modalities == None or 'gripper_pos' in modalities:
                trajectories['proprio'] = []
                for i in range(10):
                    trajectories['proprio'].append(_load_trajectories(
                        "gentle_push_300.hdf5", prop_noise=i/10, **dataset_args))
            if modalities == None or 'gripper_sensors' in modalities:
                trajectories['haptics'] = []
                for i in range(10):
                    trajectories['haptics'].append(_load_trajectories(
                        "gentle_push_300.hdf5", haptics_noise=i/10, **dataset_args))
            if modalities == None or 'controls' in modalities:
                trajectories['controls'] = []
                for i in range(10):
                    trajectories['controls'].append(_load_trajectories(
                        "gentle_push_300.hdf5", controls_noise=i/10, **dataset_args))
            if modalities == None:
                trajectories['multimodal'] = []
                for i in range(10):
                    trajectories['multimodal'].append(_load_trajectories(
                        "gentle_push_300.hdf5", multimodal_noise=i/10, **dataset_args))
            return trajectories


def _load_trajectories(
    *input_files,
    use_vision: bool = True,
    use_proprioception: bool = True,
    use_haptics: bool = True,
    vision_interval: int = 10,
    image_blackout_ratio: float = 0.0,
    sequential_image_rate: int = 1,
    start_timestep: int = 0,
    kloss_dataset: bool = False,
    visual_noise: int = 0,
    prop_noise: int = 0,
    haptics_noise: int = 0,
    controls_noise: int = 0,
    multimodal_noise: int = 0,
) -> List[TrajectoryNumpy]:
    """Loads a list of trajectories from a set of input files, where each trajectory is
    a tuple containing...
        states: an (T, state_dim) array of state vectors
        observations: a key->(T, *) dict of observations
        controls: an (T, control_dim) array of control vectors

    Each input can either be a string or a (string, int) tuple, where int indicates the
    maximum number of trajectories to import.


    Args:
        *input_files: Trajectory inputs. Should be members of
            `crossmodal.push_data.dataset_urls.keys()`.

    Keyword Args:
        use_vision (bool, optional): Set to False to zero out camera inputs.
        vision_interval (int, optional): # of times each camera image is duplicated. For
            emulating a slow image rate.
        use_proprioception (bool, optional): Set to False to zero out kinematics data.
        use_haptics (bool, optional): Set to False to zero out F/T sensors.
        image_blackout_ratio (float, optional): Dropout probabiliity for camera inputs.
            0.0 = no dropout, 1.0 = all images dropped out.
        sequential_image_rate (int, optional): If value is `N`, we only send 1 image
            frame ever `N` timesteps. All others are zeroed out.
        start_timestep (int, optional): If value is `N`, we skip the first `N` timesteps
            of each trajectory.

    Returns:
        List[TrajectoryNumpy]: list of trajectories.
    """
    trajectories = []

    assert 1 > image_blackout_ratio >= 0
    assert image_blackout_ratio == 0 or sequential_image_rate == 1

    for name in input_files:
        max_trajectory_count = sys.maxsize
        if type(name) == tuple:
            name, max_trajectory_count = name
        assert type(max_trajectory_count) == int

        # Load trajectories file into memory, all at once
        with fannypack.data.TrajectoriesFile(
            fannypack.data.cached_drive_file(name, dataset_urls[name])
        ) as f:
            raw_trajectories = list(f)

        # Iterate over each trajectory
        for raw_trajectory_index, raw_trajectory in enumerate(raw_trajectories):
            if raw_trajectory_index >= max_trajectory_count:
                break

            if kloss_dataset:
                timesteps = len(raw_trajectory["pos"])
            else:
                timesteps = len(raw_trajectory["object-state"])

            # State is just (x, y)
            state_dim = 2
            states = np.full((timesteps, state_dim), np.nan, dtype=np.float32)

            if kloss_dataset:
                states[:, 0] = raw_trajectory["pos"][:, 0]
                states[:, 1] = raw_trajectory["pos"][:, 2]
            else:
                states[:, :2] = raw_trajectory["Cylinder0_pos"][:, :2]  # x, y

            # Pull out observations
            # This is currently consisted of:
            # > gripper_pos: end effector position
            # > gripper_sensors: F/T, contact sensors
            # > image: camera image

            observations = {}

            if kloss_dataset:
                observations["gripper_pos"] = raw_trajectory["tip"]
            else:
                observations["gripper_pos"] = raw_trajectory["eef_pos"]
            if prop_noise != 0:
                observations["gripper_pos"] = add_timeseries_noise(
                    [observations["gripper_pos"]], noise_level=prop_noise, struct_drop=False)[0]
            assert observations["gripper_pos"].shape == (timesteps, 3)

            if kloss_dataset:
                observations["gripper_sensors"] = np.zeros(
                    (timesteps, 7), dtype=np.float32)
                observations["gripper_sensors"][:,
                                                :3] = raw_trajectory["force"]
                observations["gripper_sensors"][:,
                                                6] = raw_trajectory["contact"]
            else:
                observations["gripper_sensors"] = np.concatenate(
                    (
                        raw_trajectory["force"],
                        raw_trajectory["contact"][:, np.newaxis],
                    ),
                    axis=1,
                )
            if haptics_noise != 0:
                observations["gripper_sensors"] = add_timeseries_noise(
                    [observations["gripper_sensors"]], noise_level=haptics_noise, struct_drop=False)[0]
            assert observations["gripper_sensors"].shape[1] == 7

            # Zero out proprioception or haptics if unused
            if not use_proprioception:
                observations["gripper_pos"][:] = 0
            if not use_haptics:
                observations["gripper_sensors"][:] = 0

            # Get image
            if kloss_dataset:
                observations["image"] = np.mean(
                    raw_trajectory["image"], axis=-1)
            else:
                observations["image"] = raw_trajectory["image"].copy()
            if visual_noise != 0:
                observations["image"] = np.array(add_visual_noise(
                    observations["image"], noise_level=visual_noise))
            assert observations["image"].shape == (timesteps, 32, 32)

            # Mask image observations based on dataset args
            image_mask: np.ndarray
            if not use_vision:
                # Use a zero mask
                image_mask = np.zeros((timesteps, 1, 1), dtype=np.float32)
            elif image_blackout_ratio == 0.0:
                # Apply sequential rate
                image_mask = np.zeros((timesteps, 1, 1), dtype=np.float32)
                image_mask[::sequential_image_rate, 0, 0] = 1.0
            else:
                # Apply blackout rate
                image_mask = (
                    (np.random.uniform(size=(timesteps,)) > image_blackout_ratio)
                    .astype(np.float32)
                    .reshape((timesteps, 1, 1))
                )
            observations["image"] *= image_mask

            # Pull out controls
            # This is currently consisted of:
            # > previous end effector position
            # > end effector position delta
            # > binary contact reading
            if kloss_dataset:
                eef_positions = raw_trajectory["tip"]
            else:
                eef_positions = raw_trajectory["eef_pos"]
            eef_positions_shifted = np.roll(eef_positions, shift=1, axis=0)
            eef_positions_shifted[0] = eef_positions[0]

            # Force controls to be of type float32
            # NOTE: In numpy 1.20+, this can be done
            # through dtype kwarg to concatenate
            controls = np.empty((timesteps, 7), dtype=np.float32)
            np.concatenate(
                [
                    eef_positions_shifted,
                    eef_positions - eef_positions_shifted,
                    raw_trajectory["contact"][
                        :, np.newaxis
                    ],  # "contact" key same for both kloss and normal dataset
                ],
                axis=1,
                out=controls
            )
            if controls_noise != 0:
                controls = add_timeseries_noise(
                    [controls], noise_level=controls_noise, struct_drop=False)[0]

            if multimodal_noise != 0:
                tmp = add_timeseries_noise([observations["image"], observations["gripper_pos"],
                                            observations["gripper_sensors"], controls], noise_level=multimodal_noise, rand_drop=False)
                observations["image"] = tmp[0]
                observations["gripper_pos"] = tmp[1]
                observations["gripper_sensors"] = tmp[2]
                controls = tmp[3]

            # Normalize data
            if kloss_dataset:
                observations["gripper_pos"] -= np.array(
                    [[-0.00360131, 0.0, 0.00022349]],
                    dtype=np.float32,
                )
                observations["gripper_pos"] /= np.array(
                    [[0.07005621, 1.0, 0.06883541]],
                    dtype=np.float32,
                )
                observations["gripper_sensors"] -= np.array(
                    [
                        [
                            3.04424347e-02,
                            1.61328610e-02,
                            -2.47517393e-04,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            6.25842857e-01,
                        ]
                    ],
                    dtype=np.float32,
                )
                observations["gripper_sensors"] /= np.array(
                    [[2.09539968, 2.0681382, 0.00373115, 1.0, 1.0, 1.0, 0.48390451]],
                    dtype=np.float32,
                )
                states -= np.array(
                    [[-0.00279736, -0.00027878]],
                    dtype=np.float32,
                )
                states /= np.array(
                    [[0.06409658, 0.06649422]],
                    dtype=np.float32,
                )
                controls -= np.array(
                    [
                        [
                            -3.55868486e-03,
                            0.00000000e00,
                            2.34369027e-04,
                            -4.26185595e-05,
                            0.00000000e00,
                            -1.08724583e-05,
                            6.25842857e-01,
                        ]
                    ],
                    dtype=np.float32,
                )
                controls /= np.array(
                    [
                        [
                            0.0693582,
                            1.0,
                            0.06810329,
                            0.01176415,
                            1.0,
                            0.0115694,
                            0.48390451,
                        ]
                    ],
                    dtype=np.float32,
                )

            else:
                observations["gripper_pos"] -= np.array(
                    [[0.46806443, -0.0017836, 0.88028437]],
                    dtype=np.float32,
                )
                observations["gripper_pos"] /= np.array(
                    [[0.02410769, 0.02341035, 0.04018243]],
                    dtype=np.float32,
                )
                observations["gripper_sensors"] -= np.array(
                    [
                        [
                            4.9182904e-01,
                            4.5039989e-02,
                            -3.2791464e00,
                            -3.3874984e-03,
                            1.1552566e-02,
                            -8.4817986e-04,
                            2.1303751e-01,
                        ]
                    ],
                    dtype=np.float32,
                )
                observations["gripper_sensors"] /= np.array(
                    [
                        [
                            1.6152629,
                            1.666905,
                            1.9186896,
                            0.14219016,
                            0.14232528,
                            0.01675198,
                            0.40950698,
                        ]
                    ],
                    dtype=np.float32,
                )
                states -= np.array(
                    [[0.4970164, -0.00916641]],
                    dtype=np.float32,
                )
                states /= np.array(
                    [[0.0572766, 0.06118315]],
                    dtype=np.float32,
                )
                controls -= np.array(
                    [
                        [
                            4.6594709e-01,
                            -2.5247163e-03,
                            8.8094306e-01,
                            1.2939950e-04,
                            -5.4364675e-05,
                            -6.1112235e-04,
                            2.2041667e-01,
                        ]
                    ],
                    dtype=np.float32,
                )
                controls /= np.array(
                    [
                        [
                            0.02239027,
                            0.02356066,
                            0.0405312,
                            0.00054858,
                            0.0005754,
                            0.00046352,
                            0.41451886,
                        ]
                    ],
                    dtype=np.float32,
                )

            trajectories.append(
                TrajectoryNumpy(
                    states[start_timestep:],
                    fannypack.utils.SliceWrapper(observations)[
                        start_timestep:],
                    controls[start_timestep:],
                )
            )

            # Reduce memory usage
            raw_trajectories[raw_trajectory_index] = None
            del raw_trajectory

    # Uncomment this line to generate the lines required to normalize data
    

    return trajectories


# From https://github.com/stanford-iprl-lab/torchfilter:

def split_trajectories(
    trajectories: List[TrajectoryNumpy], subsequence_length: int, modalities=None
):
    """
    Helper for splitting a list of trajectories into a list of overlapping
    subsequences. For each trajectory, assuming a subsequence length of 10, this function
    includes in its output overlapping subsequences corresponding to
    timesteps...
    ```
        [0:10], [10:20], [20:30], ...
    ```
    as well as...
    ```
        [5:15], [15:25], [25:30], ...
    ```
    Args:
        trajectories (List[torchfilter.base.TrajectoryNumpy]): List of trajectories.
        subsequence_length (int): # of timesteps per subsequence.
    Returns:
        List[torchfilter.base.TrajectoryNumpy]: List of subsequences.
    """

    subsequences = []

    for traj in trajectories:
        # Chop up each trajectory into overlapping subsequences
        trajectory_length = len(traj.states)
        assert len(fp.utils.SliceWrapper(
            traj.observations)) == trajectory_length
        assert len(fp.utils.SliceWrapper(traj.controls)) == trajectory_length

        # We iterate over two offsets to generate overlapping subsequences
        for offset in (0, subsequence_length // 2):

            def split_fn(x: np.ndarray) -> np.ndarray:
                """Helper: splits arrays of shape `(T, ...)` into `(sections,
                subsequence_length, ...)`, where `sections = orig_length //
                subsequence_length`."""
                # Offset our starting point
                x = x[offset:]

                # Make sure our array is evenly divisible
                sections = len(x) // subsequence_length
                new_length = sections * subsequence_length
                x = x[:new_length]

                # Split & return
                return np.split(x, sections)

            for s, o, c in zip(
                # States are always raw arrays
                split_fn(traj.states),
                # Observations and controls can be dictionaries, so we have to jump
                # through some hoops
                fp.utils.SliceWrapper(
                    fp.utils.SliceWrapper(traj.observations).map(split_fn)
                ),
                fp.utils.SliceWrapper(
                    fp.utils.SliceWrapper(traj.controls).map(split_fn)
                ),
            ):
                # Add to subsequences
                if modalities is None:
                    subsequences.append(
                        (
                            o['gripper_pos'],
                            o['gripper_sensors'],
                            o['image'],
                            c,
                            s,
                        )
                    )
                else:
                    mods = []
                    for m in modalities:
                        if m == 'gripper_pos': 
                            mods.append(o['gripper_pos'])
                        elif m == 'gripper_sensors':
                            mods.append(o['gripper_sensors'])
                        elif m == 'image':
                            mods.append(o['image'])
                        elif m == 'control':
                            mods.append(c)
                    mods.append(s)
                    subsequences.append(tuple(mods))
    return subsequences


class SubsequenceDataset(Dataset):
    """A data preprocessor for producing training subsequences from
    a list of trajectories. Thin wrapper around `torchfilter.data.split_trajectories()`.
    
    Args:
        trajectories (list): list of trajectories, where each is a tuple of
            `(states, observations, controls)`. Each tuple member should be
            either a numpy array or dict of numpy arrays with shape `(T, ...)`.
        subsequence_length (int): # of timesteps per subsequence.
    """

    def __init__(
        self, trajectories: List[TrajectoryNumpy], subsequence_length: int, modalities=None
    ):
        """Initialize SubsequenceDataset object.

        Args:
            trajectories (List[TrajectoryNumpy]): List of trajectories
            subsequence_length (int): Length to sample each subsequence
            modalities (list, optional): List of strings of modalities to choose. Defaults to None. Choose from ['gripper_pos', 'gripper_sensors', 'image', 'control'].
        """
        # Split trajectory into overlapping subsequences
        self.subsequences = split_trajectories(
            trajectories, subsequence_length, modalities
        )

    def __getitem__(self, index: int):
        """Get a subsequence from our dataset.
        Args:
            index (int): Subsequence number in our dataset.
        Returns:
            tuple: `(states, observations, controls)` tuple that contains
            data for a single subsequence. Each tuple member should be either a
            numpy array or dict of numpy arrays with shape
            `(subsequence_length, ...)`.
        """
        return self.subsequences[index]

    def __len__(self) -> int:
        """Total number of subsequences in the dataset.
        Returns:
            int: Length of dataset.
        """
        return len(self.subsequences)
