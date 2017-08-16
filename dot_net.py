from System import Int32, Array, Single
from System.Runtime.InteropServices import GCHandle, GCHandleType
import ctypes
import numpy as np


def net2numpy(net_array):
    src_hndl = GCHandle.Alloc(net_array, GCHandleType.Pinned)
    try:
        src_ptr = src_hndl.AddrOfPinnedObject().ToInt64()
        bufType = ctypes.c_float * len(net_array)
        cbuf = bufType.from_address(src_ptr)
        np_array = np.frombuffer(cbuf, dtype=cbuf._type_)
    finally:
        if src_hndl.IsAllocated: src_hndl.Free()

    return np_array.reshape(
        net_array.GetLength(0),
        net_array.GetLength(1))


def numpy2net(np_array):
    # TODO: set reference instead of copy
    x = Array.CreateInstance(Single, np_array.shape[0], np_array.shape[1])
    for i1 in range(np_array.shape[0]):
        for i2 in range(np_array.shape[1]):
            x[i1, i2] = np_array[i1, i2]
    return x
