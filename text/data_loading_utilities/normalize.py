import numpy as np

# This code is copied from Keras code. 
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
def minmax_scaling(X): 
    chanmins = np.min(np.min(X, axis=0), axis=0)
    chanmax = np.max(np.max(X, axis=0), axis=0)
    X = X-chanmins
    X = X/(chanmax-chanmins)
    print('zero 1', chanmins, chanmax)
    return X

def pertrial_minmax(X): 
    chanmins = np.min(X, axis=1, keepdims=True)
    chanmax = np.max(X, axis=1, keepdims=True)
    X = X - chanmins
    X = X/ (chanmax -chanmins)
    return X

def rezscore(X): 
    chanmeans = np.mean(np.mean(X, axis=0), axis=0)
    chanstd = np.mean(np.std(X, axis=1), axis=0)
    print('cm, cst', chanmeans, chanstd)
    X = X - chanmeans
    X = X- chanstd
    return X