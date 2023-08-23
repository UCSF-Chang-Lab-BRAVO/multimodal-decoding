"""
This script contains function that handle creating the torch datasets for the ECoG data and associated data
"""

# todo support for variable length io, batched processing, etc

import collections.abc as container_abcs
import itertools
import os
import random
import re

import numpy as np
import torch
import torchvision as tv
from torch._six import string_classes

int_classes = int
np_str_obj_array_pattern = re.compile(r'[SaUO]')
from neural_synthesis import utils, transforms


class TrialDataset(torch.utils.data.Dataset):
    """Dataset for single trial data from ECoG and associated behavior."""

    def __init__(self, filename, data_types, root_dir, subject,
                 cpu_transforms=None, torch_transforms=None, sample_weighting=None, signal_length=500,
                 data_fraction=1.0, window_length=500, variable_length_inputs=False, num_electrodes=128,
                 single_point_prediction=False, config=None, max_seq_len=None, aud_sr=16000, expression_variant_encoder=False):
        """ Dataset class for trial aligned sequential data.

        Each sample comprises {'data1': tensor,
                               'data2': tensor,
                               ... ,
                               (optional) 'seq_lengths': 1D tensor,
                               (optional) 'sample_weighting': 1D tensor}
        Sequential data is assumed to be [time x features] (2D only)

        This requires that the list of block information is already processed and in the synthesis data directory

        """

        # load params
        self.data_types = data_types
        self.root_dir = root_dir
        self.subject = subject
        self.cpu_transforms = cpu_transforms
        self.torch_transforms = torch_transforms
        self.sample_weighting = sample_weighting
        self.cache = dict()
        self.window_length = window_length
        self.signal_length = signal_length
        self.data_fraction = data_fraction
        self.variable_length_inputs = variable_length_inputs
        self.num_electrodes = num_electrodes
        self.single_point_prediction = single_point_prediction
        self.cls = False
        self.labels = []
        self.idx_to_class = {}
        self.class_to_idx = {}
        self.max_seq_len = max_seq_len
        self.aud_sr = aud_sr
        self.expression_variant_encoder = expression_variant_encoder

        # define amount of windows to use and corresponding trials for each window
        self.file_list = utils.files_to_list(filename, os.path.join(self.root_dir, 'synthesis'))
        self.num_windows = self.signal_length - self.window_length + 1
        if self.num_windows > 0:
            self.file_list = list(
                itertools.chain.from_iterable(itertools.repeat(x, self.num_windows) for x in self.file_list))

        # classifier
        for data_type in data_types:
            if data_type[:3] == 'lab':
                self.cls = True
        if self.cls:
            for filename in self.file_list:
                filename_base = filename.split('.')[0]
                utterance_id = int(filename_base.split('_')[-1])
                self.labels.append(utterance_id)
            self.labels = np.sort(list(set(self.labels))).tolist()
            print("Number of classes: ", len(self.labels))
            for idx, label in enumerate(self.labels):
                self.idx_to_class[idx] = label
                self.class_to_idx[label] = idx
            checkpoint_dir = os.path.join(config['root_dir'], 'torch_models', config['experiment_name'],
                                          config['run_name'])
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            np.save(os.path.join(checkpoint_dir, 'idx_to_class'), self.idx_to_class)
            np.save(os.path.join(checkpoint_dir, 'class_to_idx'), self.class_to_idx)
            np.save(os.path.join(checkpoint_dir, 'labels'), self.labels)

        # define amount of data to use
        self.n_total_samples = len(self.file_list)
        self.n_samples_to_use = int(np.floor(self.n_total_samples * self.data_fraction))
        if self.data_fraction < 1.0:
            self.file_list = self.file_list[:self.n_samples_to_use]

        print("Number of trials used: ", int(np.floor(self.n_samples_to_use // self.num_windows)))
        print("Number of windows used: ", self.n_samples_to_use)
        print("Number of windows per sample: ", self.num_windows)
        print("Data types used: ", self.data_types)
        print("\n")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):

        # get filename for this particular trial
        if torch.is_tensor(idx):
            idx = idx.tolist()
        filename = self.file_list[idx].split('.')[0]
        utterance_id = int(filename.split('_')[-1])
        try:
            block_id = int(filename.split('_')[-2])
        except:
            pass

        # load the trial data, including retrieval of label if desired
        data = dict()
        for i, data_type in enumerate(self.data_types):

            # trial loading slightly different depending on whether input/output is ECoG, X, or label.
            if data_type[:3] in ['hgr', 'hga', 'lfp', 'raw', 'com']:  # ECoG data
                filepath = os.path.join(self.root_dir, 'data', self.subject, data_type,
                                        filename + self._get_suffix(data_type))
                data[data_type] = np.load(filepath)
            elif data_type[:3] == 'lab':  # label id
                data[data_type] = np.array(self.class_to_idx[utterance_id])
            elif data_type[:3] == 'blk':  # block id
                data[data_type] = block_id
            elif data_type[:3] == 'pho' or data_type[:3] == 'seq':
                continue
            else:
                if utterance_id in [5009, 5010, 5011, 5012, 5013, 5014, 5015, 5016, 5017] and self.expression_variant_encoder:
                    print("expression variant encoder")
                    expression_directory = os.path.join(self.root_dir, 'data', self.subject, 'spg_expression_100')
                    valid_expression_files = []
                    for expression_filename in os.listdir(expression_directory):
                        if expression_filename.endswith(".npy"):
                            valid_expression_files.append(expression_filename)
                    random_expression_variant_fname = random.choice(valid_expression_files)
                    filepath = os.path.join(expression_directory, random_expression_variant_fname)
                else:
                    fname = str(utterance_id)
                    filepath = os.path.join(self.root_dir, 'data', self.subject, data_type,
                                            fname + self._get_suffix(data_type))

                data[data_type] = np.load(filepath)
            if data_type[:3] == 'wav':
                data[data_type] = data[data_type][:int(self.signal_length / 200 * self.aud_sr)]  # todo no hardcode
            if data_type[:3] == 'mel':
                data[data_type] = data[data_type][:, :self.signal_length]
            elif len(np.shape(data[data_type])) == 2 and not self.variable_length_inputs:
                data[data_type] = data[data_type][:self.signal_length, :]
            if self.num_windows > 1 and len(np.shape(data[data_type])) == 2:
                slice_idx = idx % self.num_windows
                data[data_type] = data[data_type][slice_idx:slice_idx + self.window_length, :]
                if self.single_point_prediction and data_type[:3] in ['ema', 'spg', 'syn']:
                    data[data_type] = np.expand_dims(data[data_type][-1, :], 0)

        # apply cpu transforms to the data
        if self.cpu_transforms:
            data = self.cpu_transforms(data)
        # if self.cpu_transforms:
        #     for data_type in self.data_types:
        #         if data_type[:3] not in ['lab', 'blk']:
        #             data[data_type] = self.cpu_transforms(data[data_type])

        # apply torch transforms
        if self.torch_transforms is not None:
            for data_type in self.data_types:
                if data_type in self.torch_transforms:
                    # transform neural feature streams independently
                    for n in range(data[data_type].shape[1] // self.num_electrodes):
                        transformed_neural_features = self.torch_transforms[data_type](
                            data[data_type][:, n * self.num_electrodes:(n + 1) * self.num_electrodes])
                        data[data_type][:,
                        n * self.num_electrodes:(n + 1) * self.num_electrodes] = transformed_neural_features
        return data

    def _get_suffix(self, data_type):
        return '.' + data_type[:3] + '.npy'


def get_dataloaders(
        train_filename,
        test_filename,
        input_types,
        output_types,
        train_batch_size,
        test_batch_size,
        root_dir,
        subject,
        torch_transforms=None,
        signal_length=500,
        shuffle_training_data=True,
        shuffle_test_data=True,
        train_data_fraction=1.0,
        test_data_fraction=1.0,
        window_length=500,
        use_test_set=False,
        num_train_workers=4,
        num_test_workers=1,
        pin_memory=False,
        variable_length_inputs=False,
        num_electrodes=128,
        single_time_point_prediction_train=False,
        single_time_point_prediction_test=False,
        config=None,
        max_seq_len=None,
        expression_variant_encoder=False
):
    """Constructs Torch Datasets and DataLoaders for training and test data."""

    # get data types and transformers
    data_types = tuple(set(input_types + output_types))
    cpu_transforms = tv.transforms.Compose([transforms.ToTensor()])
    test_transforms = None

    print("data loader num elecs: ", num_electrodes)

    if isinstance(torch_transforms, str):
        transform_names = torch_transforms
        torch_transforms = dict()
        test_transforms = dict()
        for data_type in data_types:
            if data_type[:3] in ['hgr', 'hga', 'lfp', 'raw', 'com']:
                if 'spec_augment' in transform_names:
                    torch_transforms[data_type] = transforms.SpecAugment()

    # create datasets
    train_dataset = TrialDataset(train_filename, data_types, root_dir, subject,
                                 cpu_transforms=cpu_transforms,
                                 torch_transforms=torch_transforms,
                                 sample_weighting=None, signal_length=signal_length,
                                 data_fraction=train_data_fraction, window_length=window_length,
                                 variable_length_inputs=variable_length_inputs, num_electrodes=num_electrodes,
                                 single_point_prediction=single_time_point_prediction_train, config=config,
                                 max_seq_len=max_seq_len, expression_variant_encoder=expression_variant_encoder)
    test_dataset = TrialDataset(test_filename, data_types, root_dir, subject, cpu_transforms=cpu_transforms,
                                torch_transforms=test_transforms, sample_weighting=None,
                                data_fraction=test_data_fraction, signal_length=signal_length,
                                window_length=window_length,
                                variable_length_inputs=variable_length_inputs, num_electrodes=num_electrodes,
                                single_point_prediction=single_time_point_prediction_test, config=config,
                                max_seq_len=max_seq_len, expression_variant_encoder=expression_variant_encoder)

    # wrap datasets in dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=train_batch_size,
                                                   shuffle=shuffle_training_data,
                                                   num_workers=num_train_workers,
                                                   pin_memory=pin_memory,
                                                   collate_fn=default_collate)
    if use_test_set:
        test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                      batch_size=test_batch_size,
                                                      shuffle=shuffle_test_data,
                                                      num_workers=num_test_workers,
                                                      pin_memory=pin_memory,
                                                      collate_fn=default_collate)
    else:
        test_dataloader = None
    return train_dataloader, test_dataloader


def default_collate(batch):
    """
    Puts the data into tensors with outer dimension test_batch size.
    See https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py for details.
    """

    default_collate_err_msg_format = (
        "default_collate: test_batch must contain tensors, numpy arrays, numbers, "
        "dicts or lists; found {}")

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        full_d = dict()
        for key in elem:
            if key == 'txt':
                full_d[key] = torch.cat([d[key] for d in batch], 0)
            else:
                full_d[key] = default_collate([d[key] for d in batch])
        return full_d
        # return {key: default_collate([d[key] for d in test_batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in test_batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of test_batch should be of equal size')
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def default_convert(data):
    """ Converts NumPy array data field into tensors """
    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        return data
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        # array of string classes and object
        if elem_type.__name__ == 'ndarray' \
                and np_str_obj_array_pattern.search(data.dtype.str) is not None:
            return data
        return torch.as_tensor(data)
    elif isinstance(data, container_abcs.Mapping):
        return {key: default_convert(data[key]) for key in data}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return elem_type(*(default_convert(d) for d in data))
    elif isinstance(data, container_abcs.Sequence) and not isinstance(data, string_classes):
        return [default_convert(d) for d in data]
    else:
        return data
