import os
import numpy as np


def convert_data(data, add_weight=False):
    if isinstance(data, np.ndarray) and data.dtype.names is None:
        if data.ndim != 2 or data.shape[1] not in (3, 4):
            raise ValueError("Plain array must be 2-D with 3 or 4 columns.")
        if data.shape[1] == 3 and add_weight:
            return np.column_stack([data, np.ones(data.shape[0])])
        return data

    if isinstance(data, np.ndarray) and data.dtype.names is not None:
        fields = data.dtype.names
    elif isinstance(data, dict):
        fields = list(data.keys())
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

    cols = []
    if 'Pos' in fields or 'pos' in fields:
        key = 'Pos' if 'Pos' in fields else 'pos'
        pos = np.asarray(data[key])
        if pos.ndim == 2 and pos.shape[1] == 3:
            cols = [pos[:, 0], pos[:, 1], pos[:, 2]]
        else:
            raise ValueError(f"'{key}' must be a 2-D array with 3 columns.")
    elif all(k in fields for k in ('X', 'Y', 'Z')):
        cols = [np.asarray(data['X']), np.asarray(data['Y']), np.asarray(data['Z'])]
    else:
        raise KeyError("Data must contain 'Pos'/'pos' or 'X','Y','Z' fields.")

    has_weight = 'Weight' in fields or 'weight' in fields
    if has_weight:
        wkey = 'Weight' if 'Weight' in fields else 'weight'
        cols.append(np.asarray(data[wkey]))

    result = np.column_stack(cols)

    if result.shape[1] == 3 and add_weight:
        result = np.column_stack([result, np.ones(result.shape[0])])

    return result


def checkfile(filename, force=False, comm=None, root=0):
    if comm is None:
        if os.path.exists(filename) and not force:
            return True
        else:
            return False
    else:
        rank = comm.rank
        if rank == root:
            if os.path.exists(filename) and not force:
                judge = True
            else:
                judge = False
        else:
            judge = None
        judge = comm.bcast(judge, root=root)
        return judge