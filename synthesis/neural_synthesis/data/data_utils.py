import numpy as np
import torchaudio
from neural_synthesis.utils import *
from scipy.signal import resample


def wave2spec(wave, n_fft=512, wave_fr=16000, spec_fr=200, noise_db=-50, max_db=22.5, to_db=True, power=2):
    """ convert audio waveform to spectrogram """
    if to_db:
        return (torchaudio.transforms.AmplitudeToDB()(
            torchaudio.transforms.Spectrogram(n_fft * 2 - 1, win_length=n_fft * 2 - 1,
                                              hop_length=int(wave_fr / spec_fr), power=power)(wave)).clamp(
            min=noise_db).transpose(-2, -1) - noise_db) / (max_db - noise_db) * 2 - 1
    else:
        return torchaudio.transforms.Spectrogram(n_fft * 2 - 1, win_length=n_fft * 2 - 1,
                                                 hop_length=int(wave_fr / spec_fr), power=power)(wave).transpose(-2, -1)


def pad_end_of_sequence(sequence, n_samples, zero_pad=False, pad_idx=None, beginning=False, truncate=False):
    """ pads end of sequence (T x D) to length n_samples """
    if np.shape(sequence)[0] < n_samples:
        if zero_pad:
            pad_val = np.zeros((np.shape(sequence)[1],))
        elif pad_idx is not None:
            pad_val = np.full_like((np.shape(sequence)[1],), pad_idx)
        else:
            pad_val = sequence[-1, :]
        n_additions = n_samples - np.shape(sequence)[0]
        pad_val = np.repeat(np.expand_dims(pad_val, axis=0), n_additions, axis=0)
        if beginning:
            sequence = np.concatenate((pad_val, sequence), axis=0)
        else:
            sequence = np.concatenate((sequence, pad_val), axis=0)
    else:
        if np.shape(sequence)[0] > n_samples and not beginning and truncate:
            sequence = sequence[:n_samples, :]
    return sequence


def upsample(array, fs, fs_new):
    """ upsample sequence (T x D) """
    ratio = fs_new / fs
    num_samples = round(len(array) * ratio)
    return resample(array, num_samples)
