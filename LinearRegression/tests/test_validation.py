import numpy as np
from numpy.testing import assert_array_equal
from itertools import product

from LinearRegression.utils.utils import assert_true, assert_false, assert_equal
from LinearRegression.utils.utils import assert_raises, assert_raise_message
from LinearRegression.utils.utils import ignore_warnings
from LinearRegression.utils.utils import SkipTest
from LinearRegression.utils.utils import ignore_warnings
from LinearRegression.utils.validation import check_array, check_X_y

@ignore_warnings
def test_check_array():
    # ensure_2d=False
    X_array = check_array([0, 1, 2], ensure_2d=False)
    assert_equal(X_array.ndim, 1)
    # ensure_2d=True
    assert_raise_message(ValueError, 'Expected 2D array, got 1D array instead',
                         check_array, [0, 1, 2], ensure_2d=True)
    # ensure_all_finite=False
    # nan
    X_array = check_array([0, 1, np.nan], ensure_2d=False,
                          ensure_all_finite=False)
    assert_array_equal(X_array, np.array([0, 1, np.nan]))
    # ensure_all_finite=True
    # nan
    assert_raise_message(ValueError, "Input contains NaN, infinity"
                         " or a value too large for %r." %
                         np.array([0, 1, np.nan]).dtype,
                         check_array, [0, 1, np.nan], ensure_2d=False,
                         ensure_all_finite=True)
    # ensure_all_finite=False
    # inf
    X_array = check_array([0, 1, np.inf], ensure_2d=False,
                          ensure_all_finite=False)
    assert_array_equal(X_array, np.array([0, 1, np.inf]))
    # ensure_all_finite=True
    # inf
    assert_raise_message(ValueError, "Input contains NaN, infinity"
                         " or a value too large for %r." %
                         np.array([0, 1, np.inf]).dtype,
                         check_array, [0, 1, np.inf], ensure_2d=False,
                         ensure_all_finite=True)
    # ensure_min_features = 4 raise error
    assert_raise_message(ValueError, 
                         "Found array with %d feature(s) (shape=%s) while a"
                         " minimum of %d is required."
                         % (np.array([[0, 1, 2], [0, 1, 2]]).shape[1],
                             np.array([[0, 1, 2], [0, 1, 2]]).shape, 4),
                         check_array, [[0, 1, 2], [23, 5, 23]],
                         ensure_min_features=4)
    
    # ensure_min_samples=4 raise error
    assert_raise_message(ValueError, 
                         "Found array with %d sample(s) (shape=%s) while a"
                         " minimum of %d is required."
                         % (np.array([[0, 1, 2], [0, 1, 2]]).shape[0],
                             np.array([[0, 1, 2], [0, 1, 2]]).shape, 4),
                         check_array, [[0, 1, 2], [0, 1, 2]],
                         ensure_min_samples=4)

    # dtype, order, copy checks
    x_C = np.arange(6).reshape(2, 3).copy("C")
    x_F = x_C.copy('F')
    x_INT = x_C.astype(np.int64) 
    x_FLOAT = x_C.astype(np.float64)
    xs = [x_C, x_F, x_FLOAT, x_INT] 
    copy = [True, False]
    dtypes = [np.float32, np.float64, np.int64, np.int32, None]
    orders = ["C", "F", None]
    for x, copy, dtype, order in product(xs, copy, dtypes, orders):
        x_checked = check_array(x, dtype=dtype, order=order, copy=copy)
        if dtype is not None:
            assert_equal(x_checked.dtype, dtype)
        else:
            assert_equal(x_checked.dtype, x.dtype)
        if order == 'C':
            assert_true(x_checked.flags['C_CONTIGUOUS'])
            assert_false(x_checked.flags['F_CONTIGUOUS'])
        elif order == 'F':
            assert_true(x_checked.flags['F_CONTIGUOUS'])
            assert_false(x_checked.flags['C_CONTIGUOUS'])


