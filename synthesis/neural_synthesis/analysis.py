import neural_synthesis as ns
import numpy as np
import torch


def get_correlations(preds, targs, mean=False, dtw=False):
    """ Get per dimension performance correlations. Inputs of shape [ B x T x D ]"""
    n_features = targs.shape[-1]
    preds = preds.reshape(-1, n_features)
    targs = targs.reshape(-1, n_features)
    corrs = ns.utils.torch_correlation(targs, preds, dtw=dtw)
    if mean:
        corrs = torch.nanmean(corrs)
    return corrs


def get_bootstrapped_accuracies(trial_accuracies, num_repeats=None, median=True):
    """
    Computes accuracies using a bootstrapping method.
    """

    trial_accuracies = np.asarray(trial_accuracies)
    num_trials = len(trial_accuracies)

    if num_repeats is None:
        num_repeats = num_trials

    bootstrapped_accuracies = np.empty(shape=[num_repeats], dtype=float)

    for i in range(num_repeats):
        np.random.seed(i)
        cur_trial_samples = np.random.choice(trial_accuracies, size=num_trials, replace=True)
        if median:
            bootstrapped_accuracies[i] = np.median(cur_trial_samples)
        else:
            bootstrapped_accuracies[i] = np.mean(cur_trial_samples)

    return bootstrapped_accuracies


def compute_bootstrapped_performance(*args, **kwargs):
    """
    Returns a 2-element tuple containing the mean and st. dev.
    """

    bootstrapped_accuracies = get_bootstrapped_accuracies(*args, **kwargs)
    return np.mean(bootstrapped_accuracies), np.std(bootstrapped_accuracies)
