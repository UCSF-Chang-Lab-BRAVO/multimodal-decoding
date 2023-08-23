import copy
import torch
import numpy as np
from torch.utils.data import Dataset
import numpy as np
import random

class basics2s(Dataset):
    def __init__(self, X, lens, Y, inds, transform=None):
        
        self.X = X
        self.Y = Y
        self.lens = lens
        self.transform = transform
        self.inds = inds
        
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = copy.deepcopy(self.X[idx])
        if not self.transform is None:
            return (self.transform(sample), self.lens[idx], self.Y[idx], self.inds[idx])
        else:
            return (sample, self.lens[idx], self.Y[idx], self.inds[idx])
        
        
class hybridloader(Dataset): 
    def __init__(self, X, lens, Y, inds, Y_ctc, ctc_lens, transform=None):
        self.X = X
        self.Y = Y
        self.lens= lens
        self.transform = transform
        self.inds = inds
        self.Y_ctc = Y_ctc
        self.ctc_lens = ctc_lens
        
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = copy.deepcopy(self.X[idx])
        if not self.transform is None:
            return (self.transform(sample), self.lens[idx], self.Y[idx], self.inds[idx], self.Y_ctc[idx], self.ctc_lens[idx])
        else:
            return (sample, self.lens[idx], self.Y[idx], self.inds[idx], self.Y_ctc[idx], self.ctc_lens[idx])
        
    
class CTCDataset(Dataset):
    def __init__(self, X, Y, lens, outlens, inds, transform=None, y_transforms=None):
        
        self.X = X
        self.Y = Y
        self.lens = lens
        self.outlens = outlens
        self.inds = inds
        self.transform = transform
        
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = copy.deepcopy(self.X[idx])
        if not self.transform is None:
            return (self.transform(sample), self.Y[idx], self.lens[idx], self.outlens[idx],  self.inds[idx])
        else:
            return (sample, self.Y[idx], self.lens[idx], self.outlens[idx], self.inds[idx])
        
        
class CTCDataset_Wordct(Dataset):
    def __init__(self, X, Y, lens, outlens, inds, wordct, transform=None, y_transforms=None):
        
        self.X = X
        self.Y = Y
        self.lens = lens
        self.outlens = outlens
        self.inds = inds
        self.transform = transform
        self.wordct = wordct
        
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = copy.deepcopy(self.X[idx])
        if not self.transform is None:
            return (self.transform(sample), self.Y[idx], self.lens[idx], self.outlens[idx],  self.inds[idx], self.wordct[idx])
        else:
            return (sample, self.Y[idx], self.lens[idx], self.outlens[idx], self.inds[idx], self.wordct[idx])
        
class Jitter(object):
    """
    randomly select the default window from the original window
    scale the amt of jitter by jitter amt
    validation: just return the default window. 
    """
    def __init__(self, original_window, default_window, jitter_amt, sr=200, decimation=6, validate=False):
        self.original_window = original_window
        self.default_window = default_window
        self.jitter_scale = jitter_amt
        
        default_samples = np.asarray(default_window) - self.original_window[0]
        default_samples = np.asarray(default_samples)*sr/decimation
        
        default_samples[0] = int(default_samples[0])
        default_samples[1] = int(default_samples[1])
        
        self.default_samples = default_samples
        self.validate = validate
        
        self.winsize = int(default_samples[1] - default_samples[0])+1
        self.max_start = int(int((original_window[1] - original_window[0])*sr/decimation) - self.winsize)
        
        
    def __call__(self, sample):
        if self.validate: 
            return sample[int(self.default_samples[0]):int(self.default_samples[1])+1, :]
        else: 
            start = np.random.randint(0, self.max_start)
            scaled_start = np.abs(start-self.default_samples[0])
            scaled_start = int(scaled_start*self.jitter_scale)
            scaled_start = int(scaled_start*np.sign(start-self.default_samples[0]) + self.default_samples[0])
            return sample[scaled_start:scaled_start+self.winsize]
        
        
class Blackout(object):
    """
    The blackout augmentation.
    """
    def __init__(self, blackout_max_length=0.3, blackout_prob=0.5):
        
        self.bomax = blackout_max_length
        self.bprob = blackout_prob
        
        
    def __call__(self, sample):
      
        blackout_times = int(np.random.uniform(0, 1)*sample.shape[0]*self.bomax)
        start = np.random.randint(0, sample.shape[0]-sample.shape[0]*self.bomax)
        if random.uniform(0, 1) < self.bprob: 
            sample[start:(start+blackout_times), :] = 0
        return sample
    
class ChannelBlackout(object):
    """
    Randomly blackout a channel. 
    """
    def __init__(self, blackout_chans_max=20, blackout_prob=0.2):
        self.bcm = blackout_chans_max
        self.bp = blackout_prob
    def __call__(self, sample):
        if random.uniform(0, 1) < self.bp:
            inds = np.arange(sample.shape[-1])
            np.random.shuffle(inds)
            boi = inds[:self.bcm]
            sample[:, bcm] = 0

def normalize(x, axis=-1, order=2):
    """Normalizes a Numpy array.
    Args:
        x: Numpy array to normalize.
        axis: axis along which to normalize.
        
        order: Normalization order (e.g. `order=2` for L2 norm).
    Returns:
        A normalized copy of the array.
    """
    l2 = np.atleast_1d(np.linalg.norm(x, order, axis))
    l2[l2 == 0] = 1
    return x / np.expand_dims(l2, axis)


class Normalize(object):
    def __init__(self, axis):
        """
        Does normalization func
        """
        self.axis = axis
        
    def __call__(self, sample):
        sample_ = normalize(sample, axis=self.axis)
        return sample_
    
class AdditiveNoise(object):
    def __init__(self, sigma):
        """
        Just adds white noise.
        """
        self.sigma = sigma
        
    def __call__(self, sample):
        sample_ = sample + self.sigma*np.random.randn(*sample.shape)
        return sample_
        
class ScaleAugment(object):
    def __init__(self, low_range, up_range):
        self.up_range = up_range # e.g. .8
        self.low_range = low_range
        print('scale', self.low_range, self.up_range)
#         assert self.up_range >= self.low_range
    def __call__(self, sample):
        multiplier = np.random.uniform(self.low_range, self.up_range)
        return sample*multiplier

class LevelChannelNoise(object):
    def __init__(self, sigma, channels=128):
        """
        Sigma: the noise std. 
        """
        self.sigma= sigma
        self.channels = channels
        
    def __call__(self, sample):
        sample += self.sigma*np.random.randn(1,sample.shape[-1]) # Add uniform noise across the whole channel. 
        return sample
    
    