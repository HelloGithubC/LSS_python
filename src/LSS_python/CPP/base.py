import ctypes

import numpy as np


class c_float_complex(ctypes.Structure):
    _fields_ = [("real", ctypes.c_float), ("imag", ctypes.c_float)]

    @property
    def value(self):
        return self.real + 1j * self.imag


class c_double_complex(ctypes.Structure):
    _fields_ = [("real", ctypes.c_double), ("imag", ctypes.c_double)]

    @property
    def value(self):
        return self.real + 1j * self.imag


def normalize_mesh_inputs(pos, boxsize, ngrids, field, weights=None, values=None):
    dtype = pos.dtype
    if dtype not in (np.float32, np.float64):
        print(f"Warning: pos is not float32 or float64 ({dtype}). Now try converting to float32")
        try:
            pos = pos.astype(np.float32)
        except Exception as exc:
            raise ValueError(
                "Positions must be float32 or float64, or other types that can be cast to float32"
            ) from exc
        dtype = pos.dtype

    pos = np.ascontiguousarray(pos)

    if weights is not None:
        if weights.shape[0] != pos.shape[0]:
            raise ValueError(
                f"Weights must have the same number of particles as positions ({weights.shape[0]:d} != {pos.shape[0]:d})"
            )
        weights = np.ascontiguousarray(weights, dtype=dtype)

    if values is not None:
        if values.shape[0] != pos.shape[0]:
            raise ValueError(
                f"Values must have the same number of particles as positions ({values.shape[0]:d} != {pos.shape[0]:d})"
            )
        values = np.ascontiguousarray(values, dtype=dtype)

    try:
        len(boxsize)
    except TypeError:
        boxsize_array = np.array([boxsize] * 3, dtype=dtype)
    else:
        boxsize_array = np.array(boxsize, dtype=dtype)
    boxsize_array = np.ascontiguousarray(boxsize_array)

    try:
        len(ngrids)
    except TypeError:
        ngrids_array = np.array([ngrids] * 3, dtype=np.uint64)
    else:
        ngrids_array = np.array(ngrids, dtype=np.uint64)
    ngrids_array = np.ascontiguousarray(ngrids_array)

    if field.shape != tuple(ngrids_array):
        raise ValueError(f"Output field must have the same shape as ngrids ({field.shape} != {tuple(ngrids_array)})")
    if field.dtype != dtype:
        raise ValueError(f"Output field must have the same dtype as positions ({field.dtype} != {dtype})")
    if not field.flags.c_contiguous:
        raise ValueError("Output field must be C-contiguous")

    return pos, boxsize_array, ngrids_array, field, weights, values, dtype
