__all__ = [
    'from_numpy_to_shared_array',
    'from_shared_array_to_numpy',
]

import numpy as np
import multiprocessing
from multiprocessing import Array, RawArray


def from_numpy_to_shared_array(a, return_numpy=False, raw=False):
    """Allocates a multiprocessing.Array and populates with the content of the passed array.
    """

    # Create the shared memory
    size = int(np.prod(a.shape))
    if raw:
        a_shared = RawArray(np.ctypeslib.as_ctypes_type(a.dtype), size)
    else:
        a_shared = Array(np.ctypeslib.as_ctypes_type(a.dtype), size)

    # Add shape and dtype
    a_shared.shape = a.shape
    a_shared.dtype = a.dtype

    # Copy the content of the NumPy array
    # We first map the shared memory to a Numpy arrays for ease
    a_np = from_shared_array_to_numpy(a_shared)
    np.copyto(a_np, a)

    if return_numpy:
        return a_shared, a_np
    else:
        return a_shared


def from_shared_array_to_numpy(a):
    """Creates a NumPy representation of a multiprocessing.Array
        NOTE: It assumes that the passed object has been create using `from_numpy_to_shared_array`
    """
    if isinstance(a, multiprocessing.sharedctypes.SynchronizedArray):
        return np.frombuffer(a.get_obj(), dtype=a.dtype).reshape(a.shape)
    else:  # RawArray
        return np.frombuffer(a, dtype=a.dtype).reshape(a.shape)
