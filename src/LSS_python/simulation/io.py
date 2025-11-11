import re, ast
import os, sys
import numpy as np 

## Some structures
gadget2_header_dtype = np.dtype([
    ('np', np.int32, 6),
    ("mass", np.float64, 6),
    ("time", np.float64),
    ("redshift", np.float64),
    ("flag_sfr", np.int32),
    ("flag_feedback", np.int32),
    ("np_total", np.uint32, 6),
    ("flag_cooling", np.int32),
    ("num_files", np.int32),
    ("boxsize", np.float64),
    ("omega0", np.float64),
    ("omegaLambda", np.float64),
    ("hubble_param", np.float64),
    ("flag_stellarage", np.int32),
    ("flag_metals", np.int32),
    ("np_total_highword", np.uint32, 6),
    ("flag_entropy_instead_u", np.int32),
    ("fill", np.bytes_, 60)
    ], align=True)

## Some assistant functions
def analyze_float_format(s):
    """
    更精确地分析浮点数字符串的格式
    """
    value = float(s)
    
    # 科学计数法模式
    sci_pattern_E = r'^([-+]?)(\d+)(?:\.(\d+))?[E]([-+]?\d+)$'
    sci_pattern_e = r'^([-+]?)(\d+)(?:\.(\d+))?[e]([-+]?\d+)$'
    # 普通小数模式
    decimal_pattern = r'^([-+]?)(\d*)(?:\.(\d+))?$'
    
    sci_match_e = re.match(sci_pattern_e, s)
    sci_match_E = re.match(sci_pattern_E, s)
    decimal_match = re.match(decimal_pattern, s)
    
    if sci_match_e or sci_match_E:
        # 科学计数法
        if sci_match_e:
            sign, int_part, frac_part, exp = sci_match_e.groups()
            format_str = "e"
        elif sci_match_E:
            sign, int_part, frac_part, exp = sci_match_E.groups()
            format_str = "E"
        if frac_part:
            precision = len(frac_part)
            format_spec = f".{precision}{format_str}"
        else:
            format_spec = f".0{format_str}"
    elif decimal_match:
        # 普通小数
        sign, int_part, frac_part = decimal_match.groups()
        if frac_part:
            precision = len(frac_part)
            format_spec = f".{precision}f"
        else:
            format_spec = ".0f"
    else:
        format_spec = ".6f"  # 默认格式
    
    return value, format_spec

## Read and write parameter files (Only support lua format if the simulation supports, or txt format)
def read_params_lua(filename: str):
    params = {}

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            # 去掉注释部分: "--"及之后
            line = line.split("--")[0].strip()
            if not line:
                continue

            # 只保留 key=value 格式
            if "=" not in line:
                continue

            key, value = map(str.strip, line.split("=", 1))

            # 若 value 是花括号形式 {a, b, c}，转为 list
            if re.match(r"^\{.*\}$", value):
                # 去掉花括号并按逗号分
                inner = value.strip("{}").strip()
                # 转换数字 & 保留字符串
                list_vals = []
                for v in inner.split(","):
                    v = v.strip()
                    try:
                        # 尝试转数字 (int/float)
                        if "." in v:
                            list_vals.append(float(v))
                        else:
                            list_vals.append(int(v))
                    except ValueError:
                        list_vals.append(v)
                params[key] = list(list_vals)
            else:
                # 转换常规 value (数字/true/false/字符串)
                v = value.strip()
                # 布尔
                if v.lower() == "true":
                    params[key] = True
                elif v.lower() == "false":
                    params[key] = False
                else:
                    try:
                        eval_value = ast.literal_eval(v)
                        if isinstance(eval_value, float) or isinstance(eval_value, int):
                            if isinstance(eval_value, float):
                                params[key] = analyze_float_format(v)
                            else:
                                params[key] = eval_value
                        if isinstance(eval_value, str):
                            params[key] = eval_value.strip()
                    except Exception:
                        params[key] = (v, "var")

    return params

def write_params_lua(params: dict, filename: str):
    def format_value(v):
        # tuple -> {a, b, c}
        if isinstance(v, list):
            items = []
            for x in v:
                if isinstance(x, (int, float)):
                    items.append(str(x))
                else:
                    items.append(f'"{x}"')
            return "{ " + ", ".join(items) + " }"
        elif isinstance(v, tuple):
            value = v[0]
            format_str = v[1]
            if format_str == "var":
                return str(value)
            return format(value, format_str)
        # boolean -> true / false
        elif isinstance(v, bool):
            return "true" if v else "false"

        # number -> keep numeric format
        elif isinstance(v, (int, float)):
            return str(v)

        # string -> quote
        elif isinstance(v, str):
            return f'"{v}"'
        else:
            raise ValueError(f"Unsupported value type: {type(v)}")

    with open(filename, "w", encoding="utf-8") as f:
        for key, value in params.items():
            f.write(f"{key} = {format_value(value)}\n")

def read_params_txt(filepath, comment_char="%", sep=None):
    params = {}

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 去掉注释
            if comment_char in line:
                line = line.split(comment_char, 1)[0].strip()

            if not line:
                continue

            parts = line.split(sep)
            if len(parts) >= 2:
                key = parts[0].strip()
                value = " ".join(parts[1:]).strip()

                v = value.strip()
                # 布尔
                if v.lower() == "true":
                    params[key] = True
                elif v.lower() == "false":
                    params[key] = False
                else:
                    try:
                        eval_value = ast.literal_eval(v)
                        if isinstance(eval_value, float) or isinstance(eval_value, int):
                            if isinstance(eval_value, float):
                                params[key] = analyze_float_format(v)
                            else:
                                params[key] = eval_value
                        else:
                            params[key] = v.strip('"')
                    except Exception:
                        params[key] = v
            # 其他情况忽略（比如纯注释行）

    return params

def write_params_txt(params, filepath, sep=None, str_no_quote=False):
    """
    sep (str): separator between key and value. Default is None, which means use eight spaces
    """
    if sep is None:
        sep = " " * 8
    def format_value(v):
        # boolean -> true / false
        if isinstance(v, tuple):
            value = v[0]
            format_str = v[1]
            return format(value, format_str)
        # boolean -> true / false
        elif isinstance(v, bool):
            return "true" if v else "false"

        # number -> keep numeric format
        elif isinstance(v, (int, float)):
            return str(v)

        # string -> quote
        elif str_no_quote:
            return v
        else:
            return f'"{v}"'
    with open(filepath, "w", encoding="utf-8") as f:
        for key, value in params.items():
            f.write(f"{key}{sep}{format_value(value)}\n")

## Load default parameters
def get_default_params(param_name):
    """ Get default parameters for a simulation
    Args:
        param_name (str): The param name. Support cola_halo, mg_cola_C, mg_cola_CPP, rockstar
    """
    supported_sims = ["cola_halo", "mg_cola_C", "mg_cola_CPP", "rockstar"]
    if param_name not in supported_sims:
        raise ValueError(f"Simulation {param_name} is not supported")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_params_file = {
        "cola_halo": (os.path.join(current_dir, "paras", "cola_halo.lua"), "lua"),
        "mg_cola_C": (os.path.join(current_dir, "paras", "mg_picola_c.txt"), "txt"),
        "mg_cola_CPP": (os.path.join(current_dir, "paras", "mg_picola_cpp.lua"), "lua"),
        "rockstar": (os.path.join(current_dir, "paras", "rockstar.cfg"), "txt"),
    }
    seq_dict = {
        "rockstar": "="
    }
    comment_dict = {
        "rockstar": "#", 
        "mg_cola_C": "%", 
        "mg_cola_CPP": "%"
    }
    default_filename, default_fileformat = default_params_file[param_name]
    if default_fileformat == "lua":
        return read_params_lua(default_filename)
    elif default_fileformat == "txt":
        if param_name in seq_dict:
            sep = seq_dict[param_name]
        else:
            sep = None
        if param_name in comment_dict:
            comment_char = comment_dict[param_name]
        else:
            comment_char = None
        return read_params_txt(default_filename, comment_char=comment_char, sep=sep)
    else:
        raise ValueError(f"File format {default_fileformat} is not supported")
    

def read_gadget2(filename=None, filenames_list=None, use_long_int=False):
    if filename is None and filenames_list is None:
        raise ValueError("Either filename or filenames_list must be provided")
    if filename is not None:
        filenames_list = [filename, ]
    
    headers_array = np.empty(len(filenames_list), dtype=gadget2_header_dtype)
    pos_list = []
    vel_list = []
    id_list = []

    for i, filename in enumerate(filenames_list):
        f = open(filename, "rb")
        header_flag_1 = np.fromfile(f, dtype=np.int32, count=1)[0]
        headers_array[i] = np.fromfile(f, dtype=gadget2_header_dtype, count=1)[0]
        header_flag_2 = np.fromfile(f, dtype=np.int32, count=1)[0]
        if header_flag_1 != header_flag_2:
            raise ValueError(f"file {filename} is not a gadget2 snapshot(header_flag not consistent)")
        npar = headers_array[i]["np"][1]
        pos_flag_1 = np.fromfile(f, dtype=np.int32, count=1)[0]
        pos_temp = np.fromfile(f, dtype=np.float32, count=npar*3).reshape(npar, 3)
        pos_flag_2 = np.fromfile(f, dtype=np.int32, count=1)[0]
        if pos_flag_1 != pos_flag_2:
            raise ValueError(f"file {filename} is not a gadget2 snapshot(pos_flag not consistent)")
        vel_flag_1 = np.fromfile(f, dtype=np.int32, count=1)[0]
        vel_temp = np.fromfile(f, dtype=np.float32, count=npar*3).reshape(npar, 3)
        vel_flag_2 = np.fromfile(f, dtype=np.int32, count=1)[0]
        if vel_flag_1 != vel_flag_2:
            raise ValueError(f"file {filename} is not a gadget2 snapshot(vel_flag not consistent)")
        id_flag_1 = np.fromfile(f, dtype=np.int32, count=1)[0]
        id_temp = np.fromfile(f, dtype=np.uint64 if use_long_int else np.uint32, count=npar)
        id_flag_2 = np.fromfile(f, dtype=np.int32, count=1)[0]
        if id_flag_1 != id_flag_2:
            raise ValueError(f"file {filename} is not a gadget2 snapshot(id_flag not consistent)")
        
        pos_list.append(pos_temp)
        vel_list.append(vel_temp)
        id_list.append(id_temp.astype(np.int64))
    
    pos_array = np.concatenate(pos_list, axis=0)
    vel_array = np.concatenate(vel_list, axis=0)
    id_array = np.concatenate(id_list, axis=0)
    if headers_array.shape[0] == 1:
        headers_array = headers_array[0]
    
    return {
        "pos": pos_array,
        "vel": vel_array,
        "id": id_array,
        "header": headers_array,
    }

def read_rockstar(filename, need_elements = ["pos", "vel", "Rvir", "Mvir"], num_density=None, boxsize=1000.0):
    """
    Args:
        num_density (float): If set, will select a subset of particles with density \approx num_density. Default is None. The result ndarray must include "pos" and "Mvir".
    """
    supported_elements = ["pos", "vel", "Rvir", "Mvir"]
    if not set(need_elements).issubset(supported_elements):
        raise ValueError(f"Elements {need_elements} are not supported")
    source = np.loadtxt(filename, usecols=[8, 9, 10, 11, 12, 13, 2, 5], dtype=np.float32)

    if num_density is not None:
        if source.shape[0] < boxsize ** 3 * num_density:
            raise ValueError(f"num_density {num_density} is too large. Now only {source.shape[0] / boxsize ** 3:.3E}. The number of halo must be smaller than {boxsize ** 3 * num_density:.3E}.")
        else:
            if "pos" not in need_elements:
                need_elements.append("pos")
            if "Mvir" not in need_elements:
                need_elements.append("Mvir")

    dtype_list = []
    if "pos" in need_elements:
        dtype_list.append(("pos", np.float32, 3))
    if "vel" in need_elements:
        dtype_list.append(("vel", np.float32, 3))
    if "Rvir" in need_elements:
        dtype_list.append(("Rvir", np.float32))
    if "Mvir" in need_elements:
        dtype_list.append(("Mvir", np.float32))
    npy_dtype = np.dtype(dtype_list)

    result_npy = np.empty(shape=(len(source),), dtype=npy_dtype)
    for need_element in need_elements:
        if need_element == "pos":
            result_npy["pos"] = source[:, :3]
        if need_element == "vel":
            result_npy["vel"] = source[:, 3:6]
        if need_element == "Rvir":
            result_npy["Rvir"] = source[:, 6]
        if need_element == "Mvir":
            result_npy["Mvir"] = source[:, 7]

    if num_density is not None:
        result_npy.sort(order="Mvir")
        result_npy = result_npy[::-1]
        min_num = int(boxsize ** 3 * num_density)
        base_mass = result_npy["Mvir"][min_num - 1]
        iter_mass = result_npy["Mvir"][min_num]
        while iter_mass >= base_mass:
            min_num += 1
            if min_num >= len(result_npy):
                break
            iter_mass = result_npy["Mvir"][min_num]
        result_npy = result_npy[:min_num] 

    return result_npy
