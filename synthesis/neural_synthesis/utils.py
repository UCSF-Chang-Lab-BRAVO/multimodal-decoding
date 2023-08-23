"""
Contains various utility functions used in the repo
"""
import glob
import logging
import math
import os
import random
import sys

import git
import h5py
import librosa
import numpy as np
import pysptk
import pyworld
import scipy
import torch
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def files_to_list(filename, directory):
    """
    Takes a text file of filename and makes a list of filename

    Args:
        filename: str, filename of file contain more filename
        directory: str, dir where this file is located
    Returns:
        files: list, list of files in the directory/filename file
    """
    filepath = os.path.join(directory, filename)
    with open(filepath, encoding='utf-8') as f:
        files = f.readlines()
    files = [f.rstrip().split('.')[0] for f in files]
    return files


def torch_correlation(x, y, dtw=False):
    """PyTorch version of column_correlate (located in utils.py). This is apparently very slow."""

    def _pearsonr(x, y, length):
        length = torch.tensor(length, dtype=torch.float32)
        numerator = length * torch.sum(x * y) - (torch.sum(x) * torch.sum(y))
        denominator = torch.sqrt((length * torch.sum(x.square()) - torch.sum(x).square()) * (
                length * torch.sum(y.square()) - torch.sum(y).square()))
        return numerator / denominator

    err = 1e-9
    x = x + err * torch.randn(x.shape).to(x.device)
    y = y + err * torch.randn(y.shape).to(y.device)

    if dtw:
        C, C_hat = x, y
        distance, path = fastdtw(C, C_hat, dist=euclidean)
        distance/= (len(C) + len(C_hat))
        pathx = list(map(lambda l: l[0], path))
        pathy = list(map(lambda l: l[1], path))
        x, y = C[pathx], C_hat[pathy]

    return torch.stack(
        [_pearsonr(x[:, i], y[:, i], y.shape[0]) for i in range(y.shape[-1])])


def column_correlate(y, yp):
    """Column-wise correlations for 2D matrices of shape (time, features).

    Args:
        y: np.array, 2D matrix.
        yp: np.array, 2D matrix with same shape as y.

    Returns:
        np.array, pearson's r for each feature (i.e. normalized covariance)
    """
    corr = np.zeros((y.shape[1]))
    for j in range(y.shape[1]):
        corr[j] = scipy.stats.pearsonr(y[:, j], yp[:, j])[0]
    return corr


def accuracy(x, y):
    """Computes portion elements between two 1D vectors that are the same."""
    y = torch.squeeze(y.to(torch.long))
    correct = (x == y)
    accuracy = correct.sum().float() / y.size(0)
    return accuracy, correct


def cosine_similarity(x, y, labels):
    """computes cosine similarity between x, y and the latent feature labels (i.e. correct coordinates)"""
    # Repeat vectors to obtain shape [test_batch, latent_features, classes]
    labels = labels.unsqueeze(0).repeat(x.size(0), 1, 1)
    y = y.unsqueeze(2).repeat(1, 1, labels.size(2))
    x = x.unsqueeze(2).repeat(1, 1, labels.size(2))
    cs_x = torch.nn.functional.cosine_similarity(x, labels, dim=1)
    cs_y = torch.nn.functional.cosine_similarity(y, labels, dim=1)
    return cs_x.argmax(dim=1), cs_y.argmax(dim=1)


def euclidean_distance(x, y, labels):
    """computes euclidean distance between x, y and the latent feature labels (i.e. correct coordinates)"""
    # Repeat vectors to obtain shape [test_batch, classes, latent features]
    labels = labels.unsqueeze(0).repeat(x.size(0), 1, 1).transpose(1, 2)
    y = y.unsqueeze(1)
    x = x.unsqueeze(1)
    cs_x = torch.cdist(x, labels)
    cs_y = torch.cdist(y, labels)
    return cs_x.argmin(dim=2).squeeze(), cs_y.argmin(dim=2).squeeze()


class Checkpointer(object):
    """Returns checkpoint path for saving a model."""

    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir

    def __call__(self, ckpt_number):
        return os.path.join(self.checkpoint_dir,
                            'model_' + str(ckpt_number) + '.pt')


def repackage_data(batch, data_types, device, noise=False):
    """Unpacks sample dictionary for given data types and moves to device.

    Args:
        batch: dict, current data test_batch
        data_types: list, data types (i.e. keys) in data test_batch
        device: str, device to move data to
    Returns:
        tuple, data in test_batch but in tuple form and moved to specified device (ordered by original test_batch keys)
    """
    if noise:
        return tuple(torch.randn(batch[data_type].shape).to(device) for data_type in data_types)
    else:
        return tuple(batch[data_type].to(device) for data_type in data_types)


def sequence_mask(lengths, maxlen=None, dtype=torch.bool):
    """ Computes sequence mask to limit sequence length

    Args:
        lengths: lengths of the input sequence
        maxlen: max length a sequence should be
        dtype: datatype of mask
    Returns:
        mask: mask to use to mask out features beyond maxlen
    """
    if maxlen is None:
        maxlen = lengths.max()
    mask = ~(torch.ones((len(lengths), maxlen)).to(lengths.device).cumsum(
        dim=1).t() > lengths).t()
    mask.type(torch.uint8).type(dtype)
    return mask


def fix_format(x):
    """ converts an object into float64 continuous array"""
    return np.ascontiguousarray(x).astype('float64')


def obj_to_tuple(obj, obj_type):
    """Turns object into a tuple of size one if it is not already a tuple."""
    if isinstance(obj, obj_type):
        obj = (obj,)
    return obj


def tuple_to_obj(obj):
    """Removes tuple wrapping of length 1 tuple."""
    if isinstance(obj, tuple) and len(obj) == 1:
        obj = obj[0]
    # return just the object, no tuple
    return obj


def get_axis_shapes(list_of_inputs, axis=0):
    """returns list of shapes for an axis in a list of inputs (assuming each input has the axis in question)"""
    return [x.shape[axis] for x in list_of_inputs]


def receptive_field_size(total_layers, max_dilation, kernel_size,
                         dilation=lambda x: 2 ** x):
    """Compute receptive field size
    Args:
        total_layers: int, total layers
        max_dilation: int, maximum dilation
        kernel_size: int, kernel size
        dilation: lambda, lambda to compute dilation factor. ``lambda x : 1``
          to disable dilated convolution.
    Returns:
        int: receptive field size in sample
    """
    layers_per_cycle = math.floor(math.log2(max_dilation)) + 1
    print(layers_per_cycle)
    assert total_layers % layers_per_cycle == 0
    dilations = [dilation(i % layers_per_cycle) for i in range(total_layers)]
    return (kernel_size - 1) * sum(dilations) + 1


def keys_to_vals(lookup_table, vecs):
    """ maps a list of keys to a list of vals"""
    new_list = []
    for v in vecs:
        new_v = lookup_table[v]
        new_list.append(new_v)
    return new_list


def make_data_folder(subject, name, root_dir):
    """ generate a data folder given data_dir, subject name, and filename """
    folder_name = os.path.join(root_dir, 'data', subject, str(name))
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name


def make_test_block_list(full_file, test_file, blocks, root_dir):
    """Makes test file list given block numbers."""
    path = os.path.join(root_dir, 'synthesis')
    full_list = np.loadtxt(os.path.join(path, full_file), str)
    test_list = []
    for filename in full_list:
        block = int(filename.split('_')[1][1:])
        if block in blocks:
            test_list.append(filename)
    np.savetxt(os.path.join(path, test_file), np.array(test_list), fmt='%s')


def make_test_trial_list(full_file, test_file, n_reps, root_dir, exclude_file=None, word_list_filter=[]):
    """Makes randomized test file list given number of reps per trial."""
    path = os.path.join(root_dir, 'synthesis')
    test_list = []
    if n_reps > 0:
        full_list = list(np.loadtxt(os.path.join(path, full_file), str))
        exclude_list = []
        if exclude_file is not None:
            exclude_list = list(np.loadtxt(os.path.join(path, exclude_file), str))

        random.shuffle(full_list)
        reps = {}
        saved_reps = {}
        for filename in full_list:
            if filename not in exclude_list:
                class_id = int(filename.split('.')[0].split('_')[-1])
                if class_id not in word_list_filter:
                    if class_id in reps:
                        reps[class_id] += 1
                    else:
                        reps[class_id] = 1
                    if reps[class_id] <= n_reps:
                        test_list.append(filename)
                        if class_id in saved_reps:
                            saved_reps[class_id] += 1
                        else:
                            saved_reps[class_id] = 1
        print("\n")
        print("reps: ", n_reps)
        print("len kept: ", len(test_list))
        print("len excluded: ", len(exclude_list))
        for key, val in saved_reps.items():
            print(key, val)
    np.savetxt(os.path.join(path, test_file), np.array(test_list), fmt='%s')


def remove_test_files_from_train_files(train_file, test_file, new_train_file, root_dir):
    """Remove any test filename from a text file with training filename"""
    path = os.path.join(root_dir, 'synthesis')
    train_list = np.loadtxt(os.path.join(path, train_file), str)
    test_list = np.loadtxt(os.path.join(path, test_file), str)
    new_train_list = []
    for filename in train_list:
        if [filename] not in test_list:
            new_train_list.append(filename)
    np.savetxt(os.path.join(path, new_train_file), np.array(new_train_list), fmt='%s')


def list_files_to_txt(data_type, filename, root_dir, subject='bravo1', must_include_list=None):
    """Prints filename to txt file."""
    data_path = os.path.join(root_dir, 'data', subject, data_type, '*.npy')
    list_path = os.path.join(root_dir, 'synthesis', filename)
    file_list = [os.path.basename(path) for path in glob.glob(data_path)]
    file_list = sorted(file_list)
    if must_include_list is not None:
        base_must_include_list = []
        for file in must_include_list:
            base_must_include_list.append(os.path.basename(file))
        filtered_file_list = []
        for file in file_list:
            if file in base_must_include_list:
                filtered_file_list.append(file)
        file_list = filtered_file_list
    np.savetxt(list_path, np.array(file_list), fmt='%s')


def remove_conflicting_trials(input_dirs, full_file, subject, root_dir):
    """ ensure training only occurs on data for which we have all needed in / out data """

    # get the trials for which we have all needed inputs and outputs
    intersection = []
    for i, input_dir in enumerate(input_dirs):
        curr_set = []
        for file in os.listdir(os.path.join(root_dir, 'data', subject, input_dir)):
            # first input dir should be ECoG
            if i == 0:
                trial = file[:-8]
            else:
                import pdb
                pdb.set_trace()
            curr_set.append(trial)
        if i == 0:
            intersection = curr_set
        else:
            intersection = list(set(intersection) & set(curr_set))

    # modify the original full file list
    file_path = os.path.join(root_dir, 'synthesis', full_file)
    file = open(file_path, "r")
    list_of_lines = file.readlines()
    list_of_new_lines = []
    for line in list_of_lines:
        trial = line[:-9]
        if trial in intersection:
            list_of_new_lines.append(line)
    file = open(file_path, "w")
    file.writelines(list_of_new_lines)
    file.close()


def get_git_root(path):
    git_repo = git.Repo(path, search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return git_root


def get_config_file(filename):
    repo_dir = get_git_root(os.path.dirname(os.path.realpath(__file__)))
    config_file = os.path.join(repo_dir, 'neural_synthesis', 'config', filename)
    return config_file


def read_hdf5(hdf5_name, hdf5_path):
    """Read hdf5 dataset.
    Args:
        hdf5_name (str): Filename of hdf5 file.
        hdf5_path (str): Dataset name in hdf5 file.
    Return:
        any: Dataset values.
    """
    if not os.path.exists(hdf5_name):
        logging.error(f"There is no such a hdf5 file ({hdf5_name}).")
        sys.exit(1)

    hdf5_file = h5py.File(hdf5_name, "r")

    if hdf5_path not in hdf5_file:
        logging.error(f"There is no such a data in hdf5 file. ({hdf5_path})")
        sys.exit(1)

    hdf5_data = hdf5_file[hdf5_path][()]
    hdf5_file.close()

    return hdf5_data


def merge_yaml_configs(experiment_yaml_config, default_yaml_config):
    """ merges two yaml config files """
    if isinstance(experiment_yaml_config, dict) and isinstance(default_yaml_config, dict):
        for k, v in default_yaml_config.items():
            if k not in experiment_yaml_config:
                experiment_yaml_config[k] = v
            else:
                experiment_yaml_config[k] = merge_yaml_configs(experiment_yaml_config[k], v)
    return experiment_yaml_config

def get_label_error_rate(r, h):

    """
    Given two list of strings how many word error rate(insert, delete or substitution).
    adapted from Kiettiphong Manovisut
    """
    d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint16)
    d = d.reshape((len(r) + 1, len(h) + 1))
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
    result = float(d[len(r)][len(h)]) / len(r) * 100
    return result

def normalize_volume(audio):
    """ Normalize an audio waveform to be between 0 and 1 """
    rms = librosa.feature.rms(audio)
    max_rms = rms.max() + 0.01
    target_rms = 0.2
    audio = audio * (target_rms/max_rms)
    max_val = np.abs(audio).max()
    if max_val > 1.0:
        audio = audio / max_val
    return audio

def mcd_calc(C, C_hat):
    """ Computes MCD between ground truth and target MFCCs. First computes DTW aligned MFCCs

    Consistent with Anumanchipalli et al. 2019 Nature, we use MC 0 < d < 25 with k = 10 / log10
    """

    # ignore first MFCC
    K = 10 / np.log(10)
    C = C[:, 1:25]
    C_hat = C_hat[:, 1:25]

    # compute alignment
    distance, path = fastdtw(C, C_hat, dist=euclidean)
    distance/= (len(C) + len(C_hat))
    pathx = list(map(lambda l: l[0], path))
    pathy = list(map(lambda l: l[1], path))
    C, C_hat = C[pathx], C_hat[pathy]
    frames = C_hat.shape[0]

    # compute MCD
    z = C_hat - C
    s = np.sqrt((z * z).sum(-1)).sum()
    MCD_value = K * float(s) / float(frames)
    return MCD_value

def wav2mcep_numpy(wav, sr, alpha=0.65, fft_size=512, mcep_size=25):
    """ Given a waveform, extract the MCEP features """

    # Use WORLD vocoder to extract spectral envelope
    _, sp, _ = pyworld.wav2world(wav.astype(np.double), fs=sr,frame_period=5.0, fft_size=fft_size)

    # Extract MCEP features
    mgc = pysptk.sptk.mcep(sp, order=mcep_size, alpha=alpha, maxiter=0,
                           etype=1, eps=1.0E-8, min_det=0.0, itype=3)
    return mgc

def compute_mcd(sample, y, sr_desired=16000):
    """ Computes MCD between target waveform and predicted waveform """

    # equalize lengths
    if len(sample) < len(y):
        y = y[:len(sample)]
    else:
        sample = sample[:len(y)]

    # normalize volume
    y = normalize_volume(y)
    sample = normalize_volume(sample)

    # compute MCD
    mfcc_y_ = wav2mcep_numpy(sample, sr_desired)
    mfcc_y = wav2mcep_numpy(y, sr_desired)

    mcd = mcd_calc(mfcc_y, mfcc_y_)
    return mcd