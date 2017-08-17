import sys
import clr
# same path is set as working directory
sys.path.append(r"G:\Projects\ML\gate-net\GatesNet\GatesNet\bin\Debug")
clr.AddReference("GatesNet")

from System import Int32, Array, Single
from System.Runtime.InteropServices import GCHandle, GCHandleType
from GatesNet.Gates import GArray

import ctypes
import numpy as np
from gates.gate import *

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


class FromDotNet(Gate):
    # prev is GateNet
    def __init__(self, prev):
        super().__init__(size=prev.Size)
        self.prev = prev

    def forward(self, value):
        net_input = numpy2net(value)
        gpu_input = GArray(net_input)
        gpu_output = self.prev.Forward(gpu_input)
        net_output = gpu_output.Data
        self.value = net2numpy(net_output)
        return self.value

    def backward(self, gValue, optimizer):
        net_input = numpy2net(gValue)
        gpu_input = GArray(net_input)
        self.prev.Backward(gpu_input, None)