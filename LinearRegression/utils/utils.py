import sys
from functools import wraps

import warnings
import unittest
import numpy as np


def ignore_warnings(obj=None, category="Warning"):
    if callable(obj):
        return _IgnoreWarnings(category=category)(obj)
    else:
        return _IgnoreWarnings(category=category)
    

def clean_warning_registry():
    """Safe way to reset warnings."""
    warnings.resetwarnings()
    reg = "__warningregistry__"
    for mod_name, mod in list(sys.modules.items()):
        if 'six.moves' in mod_name:
            continue
        if hasattr(mod, reg):
            getattr(mod, reg).clear()


class _IgnoreWarnings(object):

    def __init__(self, category):
        self._record = True
        self._module = sys.modules['warnings']
        self._entered = False
        self.log = []
        self.category = category

    def __call__(self, fn):
        """Decorator to catch and hide warnings without visual nesting."""
        @wraps(fn)
        def wrapper(*args, **kwargs):
            # very important to avoid uncontrolled state propagation
            clean_warning_registry()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", self.category)
                return fn(*args, **kwargs)

        return wrapper


def assert_raise_message(exceptions, message, function, *args, **kwargs):
    """Helper function to test error messages in exceptions.

    Parameters
    ----------
    exceptions : exception or tuple of exception
        Name of the estimator

    function : callable
        Calable object to raise error

    *args : the positional arguments to `function`.

    **kw : the keyword arguments to `function`
    """
    try:
        function(*args, **kwargs)
    except exceptions as e:
        error_message = str(e)
        if message not in error_message:
            raise AssertionError("Error message does not include the expected"
                                 " string: %r. Observed error message: %r" %
                                 (message, error_message))
    else:
        # concatenate exception names
        if isinstance(exceptions, tuple):
            names = " or ".join(e.__name__ for e in exceptions)
        else:
            names = exceptions.__name__

        raise AssertionError("%s not raised by %s" %
                             (names, function.__name__))


_dummy = unittest.TestCase('__init__')
assert_equal = _dummy.assertEqual
assert_not_equal = _dummy.assertNotEqual
assert_true = _dummy.assertTrue
assert_false = _dummy.assertFalse
assert_raises = _dummy.assertRaises
SkipTest = unittest.case.SkipTest
assert_dict_equal = _dummy.assertDictEqual
assert_in = _dummy.assertIn
assert_not_in = _dummy.assertNotIn
assert_less = _dummy.assertLess
assert_greater = _dummy.assertGreater
assert_less_equal = _dummy.assertLessEqual
assert_greater_equal = _dummy.assertGreaterEqual

try:
    assert_raises_regex = _dummy.assertRaisesRegex
except AttributeError:
    # Python 2.7
    assert_raises_regex = _dummy.assertRaisesRegexp

assert_raises_regexp = assert_raises_regex


class DataConversionWarning(UserWarning):
    """Warning used to notify implicit data conversions happening in the code.
    This warning occurs when some input data needs to be converted or
    interpreted in a way that may not match the user's expectations."""
    
    pass


# Preprocessing 
def preprocess_data(X, y, intercept, normalize=False, weights=None):
    if intercept:
        X_mean = np.average(X, axis=0, weights=weights)
        y_mean = np.average(y, axis=0, weights=weights)
        
        X_centered = X - X_mean if intercept else X
        y_centered = y - y_mean if intercept else y
        
        if normalize:
            l2_norm = np.linalg.norm(X_centered, axis=0)
            l2_norm[l2_norm == 0] = 1  # Avoid division by zero
            X_scaled = X_centered / l2_norm
        else:
            l2_norm = np.ones(X_centered.shape[1])
            X_scaled = X_centered

        return X_scaled, y_centered, X_mean, y_mean, l2_norm
    
    else:
        X_mean = np.zeros(X.shape[1])
        y_mean = 0
        l2_norm = np.ones(X.shape[1])

        return X, y, X_mean, y_mean, l2_norm


def assert_warns(warning_category, func, *args, **kwargs):
    clean_warning_registry()
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        res = func(*args, **kwargs)
        if hasattr(np, 'VisibleDeprecationWarning'):
            # Filter out numpy-specific warnings in numpy >= 1.9
            w = [e for e in w
                 if e.category is not np.VisibleDeprecationWarning]
        if len(w) == 0:
            raise ValueError("No warnings were raised by function")
        found = any(warning.category is warning_category for warning in w)
        if not found:
            raise ValueError(
                "Warning of category %s wasn't raised" % warning_category)
        return res
    

def assert_warn_message(warning_category, message, func, *args, **kwargs):
    clean_warning_registry()
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        res = func(*args, **kwargs)
        
        if hasattr(np, 'VisibleDeprecationWarning'):
            # Filter out numpy-specific warnings in numpy >= 1.9
            w = [e for e in w
                 if e.category is not np.VisibleDeprecationWarning]
        
        if len(w) == 0:
            raise ValueError("No warnings were raised by function")
        
        found = False
        
        for warning in w:
            if hasattr(warning, "category") and \
                    warning.category == warning_category and \
                    str(warning.message) == message:
                
                found = True
                continue
        
        if not found:
            raise ValueError(
                "Warning of category %s with message %s wasn't raised"
                % (warning_category, message))   
            
        return res