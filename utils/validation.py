import numpy as np
from six import string_types
from collections.abc import Sequence
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.utils import DataConversionWarning


def check_array(arr, warn_on_dtype=True, ensure_all_finite=True,
                ensure_min_features=1, ensure_min_samples=1, ensure_2d=True,
                copy=False, order=None, dtype='numeric'):
    
    dtype_numeric = isinstance(dtype, string_types) and dtype == "numeric"
    dtype_orig = getattr(arr, "dtype", None)
    if not hasattr(dtype_orig, "kind"):
        dtype_orig = None
    
    if dtype_numeric:
        if dtype_orig is not None and dtype_orig.kind == "O":
            dtype = np.float64
        else:
            dtype = None

    if isinstance(dtype, (list, tuple)):
        if dtype_orig is not None and dtype_orig in dtype:
            # no dtype conversion required
            dtype = None
        else:
            dtype = dtype[0]

    arr = np.asarray(arr, order=order, dtype=dtype)

    if ensure_2d:
        if arr.ndim == 1:
            raise ValueError(                    
                "Expected 2D array, got 1D array instead:\narray={}.\n"
                "Reshape your data either using array.reshape(-1, 1) if "
                "your data has a single feature or array.reshape(1, -1) "
                "if it contains a single sample.".format(arr)
            )
        arr = np.atleast_2d(arr)
        arr = np.asarray(arr, dtype=dtype, order=order)

    if ensure_all_finite:
        assert_all_finite(arr)
    
    arr_shape = np.shape(arr)

    if ensure_min_samples > 0:
        n_samples = arr_shape[0]
        if n_samples < ensure_min_samples:
            raise ValueError("Found array with %d sample(s) (shape=%s) while a"
                             " minimum of %d is required."
                             % (n_samples, arr_shape, ensure_min_samples))

    if ensure_min_features > 0 and arr.ndim == 2:
        n_features = arr.shape[1]
        if n_features < ensure_min_features:
            raise ValueError(
                            "Found array with %d feature(s) (shape=%s) while a"
                            " minimum of %d is required."
                            % (n_features, arr_shape, ensure_min_features)
            )
    
    # if warn_on_dtype and dtype_orig is not None and arr.dtype != dtype_orig:
    #     msg = ("Data with input dtype %s was converted to %s."
    #            % (dtype_orig, arr.dtype))
    #     warnings.warn(msg, DataConversionWarning)

    # TODO: Fix nasty error caused by commented code: TypeError: issubclass() arg 2 must be a class, a tuple of classes, or a union
    return arr


def num_samples(x):
    """Return number of samples in array-like x."""
    if hasattr(x, 'fit') and callable(x.fit):
        # Don't get num_samples from an ensembles length!
        raise TypeError('Expected sequence or array-like, got '
                        'estimator %s' % x)
    if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
        if hasattr(x, '__array__'):
            x = np.asarray(x)
        else:
            raise TypeError("Expected sequence or array-like, got %s" %
                            type(x))
    if hasattr(x, 'shape'):
        if len(x.shape) == 0:
            raise TypeError("Singleton array %r cannot be considered"
                            " a valid collection." % x)
        return x.shape[0]
    else:
        return len(x)


def assert_1d(arr, warn=False):
    
    arr_shape = np.shape(arr)
    if len(arr_shape) == 1:
        return np.ravel(arr)
    if len(arr_shape) == 2 and arr_shape[1] == 1:
        # warnings.warn("A column-vector was passed when 1d "
        #               "array was expected. Please change shape of y "
        #               "You can use np.ravel()", DataConversionWarning)
        # TODO: Fix nasty error caused by commented code: TypeError: issubclass() arg 2 must be a class, a tuple of classes, or a union

        return np.ravel(arr)
    
    raise ValueError("Provided array has invalid shape {0}".format(arr_shape))


def assert_all_finite(arr):
    arr = np.asanyarray(arr)

    if (arr.dtype.char in np.typecodes["AllFloat"] 
            and not np.isfinite(arr.sum())
            and not np.isfinite(arr).all()):
        
        raise ValueError("Input contains NaN, infinity"
                         " or a value too large for %r." % arr.dtype)


def check_consistent_length(*arrays):
    lengths = [num_samples(X) for X in arrays if X is not None]
    if len(np.unique(lengths)) > 1:
        raise ValueError("Found input variables with inconsistent numbers of"
                         " samples: %r" % [int(ln) for ln in lengths])
    

def check_X_y(X, y, warn_on_dtype=True, y_numeric=False,
              ensure_all_finite=True,
              ensure_min_features=1, ensure_min_samples=1,
              ensure_2d=True, copy=False, order=None, dtype='numeric'):
    
    X = check_array(X, warn_on_dtype, ensure_all_finite,
                    ensure_min_features, ensure_min_samples,
                    ensure_2d, copy, order, dtype)
    
    y = assert_1d(y, warn=True)
    assert_all_finite(y)

    if y_numeric and y.dtype.kind == 'O':
        y = y.astype(np.float64)

    check_consistent_length(X, y)

    return X, y


def check_classification_target(y):
    target_type = determine_target_type(y)
    if target_type in ['binary', 'multiclass']:
        return target_type
    return 'unknown'


def determine_target_type(y):
    valid = ((isinstance(y, Sequence) or hasattr(y, "__array__")) 
             and not isinstance(y, string_types))
    
    if not valid:
        raise ValueError('Expected array-like (array or non-string sequence), '
                         'got %r' % y)

    try:
        y = np.asarray(y)
    except ValueError:
        # Known to fail in numpy 1.3 for array of arrays
        return 'unknown'
    
    if y.ndim > 2 or (y.dtype == object and len(y) and
                      not isinstance(y[0], string_types)):
        return 'unknown'
    
    if y.ndim == 2 and y.shape[1] == 0:
        return 'unknown'
    
    if y.ndim == 2 and y.shape[1] > 1:
        suffix = '-multilabel'
    else:
        suffix = ''

    if y.dtype.kind == 'f' and np.any(y != y.astype(int)):
        return "continious" + suffix
    if (len(np.unique(y)) > 2) and (y.ndim >= 2 and len(y[0]) > 1):
        return 'multiclass' + suffix
    else:
        return 'binary' + suffix
    

def check_classification_X_y(X, y, return_target=False):
    X, y = check_X_y(X, y)

    target_type = check_classification_target(y)

    if target_type == 'unknown':
        raise ValueError("Couldn't determine classification target type")
    if not return_target:
        return X, y 
    else:
        return X, y, target_type
