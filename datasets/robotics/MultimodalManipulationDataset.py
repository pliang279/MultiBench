"""Implements dataset for MultiModal Manipulation Task."""

import h5py
import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset
from robustness.visual_robust import add_visual_noise
from robustness.timeseries_robust import add_timeseries_noise


class MultimodalManipulationDataset(Dataset):
    """Multimodal Manipulation dataset."""

    def __init__(
        self,
        filename_list,
        transform=None,
        episode_length=50,
        training_type="selfsupervised",
        n_time_steps=1,
        action_dim=4,
        pairing_tolerance=0.06,
        filedirprefix=""
    ):
        """Initialize dataset.

        Args:
            filename_list (str): List of files to get data from
            transform (fn, optional): Optional function to transform data. Defaults to None.
            episode_length (int, optional): Length of each episode. Defaults to 50.
            training_type (str, optional): Type of training. Defaults to "selfsupervised".
            n_time_steps (int, optional): Number of time steps. Defaults to 1.
            action_dim (int, optional): Action dimension. Defaults to 4.
            pairing_tolerance (float, optional): Pairing tolerance. Defaults to 0.06.
            filedirprefix (str, optional): File directory prefix (unused). Defaults to "".
        """
        #self.dataset_path = [(filedirprefix + ff) for ff in filename_list]
        self.dataset_path = filename_list
        self.transform = transform
        self.episode_length = episode_length
        self.training_type = training_type
        self.n_time_steps = n_time_steps
        self.dataset = {}
        self.action_dim = action_dim
        self.pairing_tolerance = pairing_tolerance

        self._config_checks()
        self._init_paired_filenames()

    def __len__(self):
        """Get number of items in dataset."""
        return len(self.dataset_path) * (self.episode_length - self.n_time_steps)

    def __getitem__(self, idx):
        """Get item in dataset at index idx."""
        list_index = idx // (self.episode_length - self.n_time_steps)
        dataset_index = idx % (self.episode_length - self.n_time_steps)
        filename = self.dataset_path[list_index][:-8]

        file_number, filename = self._parse_filename(filename)

        unpaired_filename, unpaired_idx = self.paired_filenames[(
            list_index, dataset_index)]

        if dataset_index >= self.episode_length - self.n_time_steps - 1:
            dataset_index = np.random.randint(
                self.episode_length - self.n_time_steps - 1
            )

        sample = self._get_single(
            self.dataset_path[list_index],
            list_index,
            unpaired_filename,
            dataset_index,
            unpaired_idx,
        )
        return sample

    def _get_single(
        self, dataset_name, list_index, unpaired_filename, dataset_index, unpaired_idx
    ):

        dataset = h5py.File(dataset_name, "r", swmr=True, libver="latest")
        unpaired_dataset = h5py.File(
            unpaired_filename, "r", swmr=True, libver="latest")

        if self.training_type == "selfsupervised":

            image = dataset["image"][dataset_index]
            depth = dataset["depth_data"][dataset_index]
            proprio = dataset["proprio"][dataset_index][:8]
            force = dataset["ee_forces_continuous"][dataset_index]

            if image.shape[0] == 3:
                image = np.transpose(image, (2, 1, 0))

            if depth.ndim == 2:
                depth = depth.reshape((128, 128, 1))

            flow = np.array(dataset["optical_flow"][dataset_index])
            flow_mask = np.expand_dims(
                np.where(
                    flow.sum(axis=2) == 0,
                    np.zeros_like(flow.sum(axis=2)),
                    np.ones_like(flow.sum(axis=2)),
                ),
                2,
            )

            unpaired_image = image
            unpaired_depth = depth
            unpaired_proprio = unpaired_dataset["proprio"][unpaired_idx][:8]
            unpaired_force = unpaired_dataset["ee_forces_continuous"][unpaired_idx]

            sample = {
                "image": image,
                "depth": depth,
                "flow": flow,
                "flow_mask": flow_mask,
                "action": dataset["action"][dataset_index + 1],
                "force": force,
                "proprio": proprio,
                "ee_yaw_next": dataset["proprio"][dataset_index + 1][:self.action_dim],
                "contact_next": np.array(
                    [dataset["contact"][dataset_index + 1].sum() > 0]
                ).astype(np.float),
                "unpaired_image": unpaired_image,
                "unpaired_force": unpaired_force,
                "unpaired_proprio": unpaired_proprio,
                "unpaired_depth": unpaired_depth,
            }

        dataset.close()
        unpaired_dataset.close()

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _init_paired_filenames(self):
        """
        Precalculates the paired filenames.
        Imposes a distance tolerance between paired images
        """
        tolerance = self.pairing_tolerance

        all_combos = set()

        self.paired_filenames = {}
        for list_index in tqdm(range(len(self.dataset_path)), desc="pairing_files"):
            filename = self.dataset_path[list_index]
            file_number, _ = self._parse_filename(filename[:-8])

            dataset = h5py.File(filename, "r", swmr=True, libver="latest")

            for idx in range(self.episode_length - self.n_time_steps):

                proprio_dist = None
                while proprio_dist is None or proprio_dist < tolerance:
                    # Get a random idx, file that is not the same as current
                    unpaired_dataset_idx = np.random.randint(self.__len__())
                    unpaired_filename, unpaired_idx, _ = self._idx_to_filename_idx(
                        unpaired_dataset_idx)

                    while unpaired_filename == filename:
                        unpaired_dataset_idx = np.random.randint(
                            self.__len__())
                        unpaired_filename, unpaired_idx, _ = self._idx_to_filename_idx(
                            unpaired_dataset_idx)

                    with h5py.File(unpaired_filename, "r", swmr=True, libver="latest") as unpaired_dataset:
                        proprio_dist = np.linalg.norm(
                            dataset['proprio'][idx][:3] - unpaired_dataset['proprio'][unpaired_idx][:3])

                self.paired_filenames[(list_index, idx)] = (
                    unpaired_filename, unpaired_idx)
                all_combos.add((unpaired_filename, unpaired_idx))

            dataset.close()

    def _idx_to_filename_idx(self, idx):
        """
        Utility function for finding info about a dataset index

        Args:
            idx (int): Dataset index

        Returns:
            filename (string): Filename associated with dataset index
            dataset_index (int): Index of data within that file
            list_index (int): Index of data in filename list
        """
        list_index = idx // (self.episode_length - self.n_time_steps)
        dataset_index = idx % (self.episode_length - self.n_time_steps)
        filename = self.dataset_path[list_index]
        return filename, dataset_index, list_index

    def _parse_filename(self, filename):
        """ Parses the filename to get the file number and filename"""
        if filename[-2] == "_":
            file_number = int(filename[-1])
            filename = filename[:-1]
        else:
            file_number = int(filename[-2:])
            filename = filename[:-2]

        return file_number, filename

    def _config_checks(self):
        if self.training_type != "selfsupervised":
            raise ValueError(
                "Training type not supported: {}".format(self.training_type)
            )


class MultimodalManipulationDataset_robust(Dataset):
    """Multimodal Manipulation dataset."""

    def __init__(
        self,
        filename_list,
        transform=None,
        episode_length=50,
        training_type="selfsupervised",
        n_time_steps=1,
        action_dim=4,
        pairing_tolerance=0.06,
        filedirprefix="",
        noise_level=0,
        image_noise=False,
        force_noise=False,
        prop_noise=False
    ):
        """
        Args:
            hdf5_file (handle): h5py handle of the hdf5 file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #self.dataset_path = [(filedirprefix + ff) for ff in filename_list]
        self.dataset_path = filename_list
        self.transform = transform
        self.episode_length = episode_length
        self.training_type = training_type
        self.n_time_steps = n_time_steps
        self.dataset = {}
        self.action_dim = action_dim
        self.pairing_tolerance = pairing_tolerance
        self.noise_level = noise_level
        self.image_noise = image_noise
        self.force_noise = force_noise
        self.prop_noise = prop_noise

        self._config_checks()
        self._init_paired_filenames()

    def __len__(self):
        """Get number of items in dataset."""
        return len(self.dataset_path) * (self.episode_length - self.n_time_steps)

    def __getitem__(self, idx):
        """Get item in dataset at index idx."""
        
        list_index = idx // (self.episode_length - self.n_time_steps)
        dataset_index = idx % (self.episode_length - self.n_time_steps)
        filename = self.dataset_path[list_index][:-8]

        file_number, filename = self._parse_filename(filename)

        unpaired_filename, unpaired_idx = self.paired_filenames[(
            list_index, dataset_index)]

        if dataset_index >= self.episode_length - self.n_time_steps - 1:
            dataset_index = np.random.randint(
                self.episode_length - self.n_time_steps - 1
            )

        sample = self._get_single(
            self.dataset_path[list_index],
            list_index,
            unpaired_filename,
            dataset_index,
            unpaired_idx,
        )
        return sample

    def _get_single(
        self, dataset_name, list_index, unpaired_filename, dataset_index, unpaired_idx
    ):

        dataset = h5py.File(dataset_name, "r", swmr=True, libver="latest")
        unpaired_dataset = h5py.File(
            unpaired_filename, "r", swmr=True, libver="latest")

        if self.training_type == "selfsupervised":

            image = dataset["image"][dataset_index]
            if self.image_noise:
                image = add_visual_noise(
                    [image], noise_level=self.noise_level)[0]
            depth = dataset["depth_data"][dataset_index]
            proprio = dataset["proprio"][dataset_index][:8]
            if self.prop_noise:
                proprio = add_timeseries_noise(
                    [proprio], noise_level=self.noise_level)[0]
            force = dataset["ee_forces_continuous"][dataset_index]
            if self.force_noise:
                force = add_timeseries_noise(
                    [force], noise_level=self.noise_level)[0]

            if image.shape[0] == 3:
                image = np.transpose(image, (2, 1, 0))

            if depth.ndim == 2:
                depth = depth.reshape((128, 128, 1))

            flow = np.array(dataset["optical_flow"][dataset_index])
            flow_mask = np.expand_dims(
                np.where(
                    flow.sum(axis=2) == 0,
                    np.zeros_like(flow.sum(axis=2)),
                    np.ones_like(flow.sum(axis=2)),
                ),
                2,
            )

            unpaired_image = image
            unpaired_depth = depth
            unpaired_proprio = unpaired_dataset["proprio"][unpaired_idx][:8]
            unpaired_force = unpaired_dataset["ee_forces_continuous"][unpaired_idx]

            sample = {
                "image": image,
                "depth": depth,
                "flow": flow,
                "flow_mask": flow_mask,
                "action": dataset["action"][dataset_index + 1],
                "force": force,
                "proprio": proprio,
                "ee_yaw_next": dataset["proprio"][dataset_index + 1][:self.action_dim],
                "contact_next": np.array(
                    [dataset["contact"][dataset_index + 1].sum() > 0]
                ).astype(np.float),
                "unpaired_image": unpaired_image,
                "unpaired_force": unpaired_force,
                "unpaired_proprio": unpaired_proprio,
                "unpaired_depth": unpaired_depth,
            }

        dataset.close()
        unpaired_dataset.close()

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _init_paired_filenames(self):
        """
        Precalculates the paired filenames.
        Imposes a distance tolerance between paired images
        """
        tolerance = self.pairing_tolerance

        all_combos = set()

        self.paired_filenames = {}
        for list_index in tqdm(range(len(self.dataset_path)), desc="pairing_files"):
            filename = self.dataset_path[list_index]
            file_number, _ = self._parse_filename(filename[:-8])

            dataset = h5py.File(filename, "r", swmr=True, libver="latest")

            for idx in range(self.episode_length - self.n_time_steps):

                proprio_dist = None
                while proprio_dist is None or proprio_dist < tolerance:
                    # Get a random idx, file that is not the same as current
                    unpaired_dataset_idx = np.random.randint(self.__len__())
                    unpaired_filename, unpaired_idx, _ = self._idx_to_filename_idx(
                        unpaired_dataset_idx)

                    while unpaired_filename == filename:
                        unpaired_dataset_idx = np.random.randint(
                            self.__len__())
                        unpaired_filename, unpaired_idx, _ = self._idx_to_filename_idx(
                            unpaired_dataset_idx)

                    with h5py.File(unpaired_filename, "r", swmr=True, libver="latest") as unpaired_dataset:
                        proprio_dist = np.linalg.norm(
                            dataset['proprio'][idx][:3] - unpaired_dataset['proprio'][unpaired_idx][:3])

                self.paired_filenames[(list_index, idx)] = (
                    unpaired_filename, unpaired_idx)
                all_combos.add((unpaired_filename, unpaired_idx))

            dataset.close()

    def _idx_to_filename_idx(self, idx):
        """
        Utility function for finding info about a dataset index

        Args:
            idx (int): Dataset index

        Returns:
            filename (string): Filename associated with dataset index
            dataset_index (int): Index of data within that file
            list_index (int): Index of data in filename list
        """
        list_index = idx // (self.episode_length - self.n_time_steps)
        dataset_index = idx % (self.episode_length - self.n_time_steps)
        filename = self.dataset_path[list_index]
        return filename, dataset_index, list_index

    def _parse_filename(self, filename):
        """ Parses the filename to get the file number and filename"""
        if filename[-2] == "_":
            file_number = int(filename[-1])
            filename = filename[:-1]
        else:
            file_number = int(filename[-2:])
            filename = filename[:-2]

        return file_number, filename

    def _config_checks(self):
        if self.training_type != "selfsupervised":
            raise ValueError(
                "Training type not supported: {}".format(self.training_type)
            )
