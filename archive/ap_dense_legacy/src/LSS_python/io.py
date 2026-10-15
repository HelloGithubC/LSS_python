import os, joblib
import numpy as np

def read_data(filename, general_binary_loader="joblib"):
    """
    根据文件类型读取数据
    
    Parameters:
    -----------
    filename : str
        文件路径
    general_binary_loader : str
        一般二进制文件的加载方式，默认为 "joblib"
    
    Returns:
    --------
    data : 各种类型
        读取的数据
    
    Raises:
    -------
    ValueError
        当文件类型不支持或加载失败时抛出
    """
    # 检查文件是否存在
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    
    # 使用 file 命令判断文件类型
    type_source = os.popen(f"file {filename}")
    file_type_info = type_source.read()
    type_source.close()
    
    # 判断文件类型并读取
    # 1. ASCII 文件
    if "ASCII" in file_type_info or "text" in file_type_info:
        try:
            data = np.loadtxt(filename)
            return data
        except Exception as e:
            raise ValueError(f"Failed to read ASCII file '{filename}': {e}")
    
    # 2. NPY/NPZ 文件
    elif filename.endswith('.npy') or filename.endswith('.npz') or "NumPy" in file_type_info:
        try:
            data = np.load(filename, allow_pickle=True)
            # 如果是 npz 文件，转换为字典形式
            if isinstance(data, np.lib.npyio.NpzFile):
                return dict(data)
            return data
        except Exception as e:
            raise ValueError(f"Failed to read numpy file '{filename}': {e}")
    
    # 3. 一般二进制文件
    elif "data" in file_type_info or "binary" in file_type_info or "executable" in file_type_info:
        if general_binary_loader == "joblib":
            try:
                data = joblib.load(filename)
                return data
            except Exception as e:
                raise ValueError(
                    f"Failed to read binary file '{filename}' with joblib. "
                    f"Error: {e}. "
                    f"This file type is not supported yet."
                )
        ### Add new binary loaders here
        else:
            raise ValueError(
                f"Unsupported binary loader: '{general_binary_loader}'. "
                f"Currently only 'joblib' is supported."
            )
    
    # 4. 其他未知类型，尝试作为二进制文件加载
    else:
        try:
            data = joblib.load(filename)
            return data
        except Exception as e:
            raise ValueError(
                f"Unsupported file type for '{filename}'. "
                f"File type info: {file_type_info.strip()}. "
                f"Currently supported types: ASCII text files, .npy/.npz files, "
                f"and binary files loadable by joblib."
            )


def convert_data(data, add_weight=False):
    if isinstance(data, np.ndarray) and data.dtype.names is None:
        if data.ndim != 2 or data.shape[1] not in (3, 4):
            raise ValueError("Plain array must be 2-D with 3 or 4 columns.")
        if data.shape[1] == 3 and add_weight:
            return np.column_stack([data, np.ones(data.shape[0], dtype=data.dtype)])
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
        result = np.column_stack([result, np.ones(result.shape[0], dtype=result.dtype)])

    return result

def unify_data_type(*args, data_names=None, verbose=False):
    """
    对数据列表中所有数据进行精度调整，统一为最高精度
    
    支持两种调用方式：
    - unify_data_type(data1, data2, data3) - 直接传入多个数组
    - unify_data_type([data1, data2, data3]) - 传入列表或元组
    
    Parameters:
    -----------
    *args : numpy.ndarray or list/tuple of numpy.ndarray
        数据数组，可以单独传入多个数组，也可以传入一个列表或元组
    data_names : list of str, optional
        数据名称列表，用于打印警告信息，默认为 None
    verbose : bool, optional
        是否打印警告信息，默认为 False
    
    Returns:
    --------
    data_list : list of numpy.ndarray
        精度统一后的数据列表
    
    Raises:
    -------
    TypeError
        当输入不是 numpy 数组时抛出
    ValueError
        当没有提供任何数据时抛出
    """
    # 处理参数：支持列表/元组传入或直接传入多个参数
    if len(args) == 0:
        raise ValueError("No data provided")
    
    # 如果只有一个参数且是列表或元组，则使用该列表/元组
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        data_list = list(args[0])
    else:
        # 否则将所有参数作为数据列表
        data_list = list(args)
    
    if len(data_list) == 0:
        return data_list

    if verbose:
        if data_names is not None:
            if len(data_list) != len(data_names):
                raise ValueError(f"Length of data_list and data_names must be equal, got {len(data_list)} and {len(data_names)}")
    
    # 检查所有元素是否为 numpy 数组
    for i, data in enumerate(data_list):
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Element {i} in data_list is not a numpy array, got {type(data)}")
    
    # 找出最高精度的数据类型
    higher_dtype = data_list[0].dtype
    higher_index = 0
    
    for i, data in enumerate(data_list):
        # 对于结构化数组，比较第一个字段的精度
        if data.dtype.names is not None:
            raise TypeError(f"Structured array is not supported yet.")
        else:
            # 普通数组
            current_itemsize = data.dtype.itemsize
            higher_itemsize = higher_dtype.itemsize if higher_dtype.names is None else higher_dtype[0].itemsize
        
        # 更新最高精度
        if current_itemsize > higher_itemsize:
            higher_dtype = data.dtype
            higher_index = i
    
    # 统一所有数据的精度
    result_list = []
    for i, data in enumerate(data_list):
        if data.dtype != higher_dtype:
            if verbose:
                if data_names is not None:
                    data_name = data_names[i]
                else:
                    data_name = f"data_list[{i}]"
                print(f"Warning: Converting {data_name} from {data.dtype} to {higher_dtype}")
            result_list.append(data.astype(higher_dtype))
        else:
            result_list.append(data)
    
    return result_list


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