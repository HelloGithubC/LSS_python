import numpy as np
import os 
from astropy.cosmology import Planck13
import colibri.cosmology as cc 

def create_PS(omega_m0, w0, kmax, verbose=False, params=None, method="CAMB", kmin=1e-3, k_point=1500):
    """ A simple function to create linear PS in z=0.0 for simulation
    Args:
        params: dict, cosmology parameters
            H0: Hubble parameter
            Ob0: baryon density
            wa: w0wa cosmology
            ns: primordial index
            sigma8: sigma8
    Return:
    dict:
      kh: kh
      pk: pk
      As: As_target (only be valid when using CAMB)
    """
    if params is None:
        cosmo = Planck13
        params = {
            "H0": cosmo.H0.value,
            "Ob0": cosmo.Ob0,
            "wa": 0.0,
            "ns": 0.96, 
            "sigma8": 0.8288
        }
        print("Waring: using Planck13 cosmology as default")

    H0 = params.get("H0", 67.7)
    h = H0 / 100.0
    Om0 = omega_m0
    Ob0 = params.get("Ob0", 0.0)
    ns = params.get("ns", 0.97)
    sigma8 = params.get("sigma8", 0.8)
    wa = params.get("wa", 0.0)

    if verbose:
        print("Cosmology parameters:")
        print(f"h = {h:.4f}, Om0 = {Om0:.5f}, Ob0 = {Ob0:.5f}, w0 = {w0:.2f}, ns = {ns:.5f}, sigma8 = {sigma8:.4f}")
        print("Method: ")
        print(method)

    kh = np.logspace(np.log10(kmin), np.log10(kmax), k_point)
    if method == "CAMB":

        time = 10

        As_init = 2.1e-9
        cosmo_need = cc.cosmo(
            h=h,
            Omega_m=Om0,
            Omega_b=Ob0,
            w0=w0,
            wa=0.0,
            ns=ns,
            As=As_init, 
        )
        _, Pk_camb = cosmo_need.camb_Pk(z=0, k=kh, nonlinear=False)
        s8_current = cosmo_need.compute_sigma_8(kh, Pk_camb[0])[0]
        As_target = As_init

        if verbose:
            print(f"As = {As_target:.5e}, sigma8 = {s8_current:.4f}")

        while abs(s8_current - sigma8) > 1e-4 or time <= 0:
            As_target = As_target * (sigma8 / s8_current) ** 2
            cosmo_need = cc.cosmo(
                h=h,
                Omega_m=Om0,
                Omega_b=Ob0,
                w0=w0,
                wa=0.0,
                ns=ns,
                As=As_target, 
            )
            _, Pk_camb = cosmo_need.camb_Pk(z=0, k=kh, nonlinear=False)
            s8_current = cosmo_need.compute_sigma_8(kh, Pk_camb[0])[0]
            if verbose:
                print(f"As = {As_target:.5e}, sigma8 = {s8_current:.4f}")
            time -= 1

        if time<=0:
            print("Warning: failed to converge to sigma8")
        else:
            pk = Pk_camb[0]
        
    elif method == "CLASSY":
        cosmo_need = cc.cosmo(
            h=h,
            Omega_m=Om0,
            Omega_b=Ob0,
            w0=w0,
            wa=wa,
            ns=ns,
            As=None, 
            sigma_8=sigma8
        )
        kh = np.logspace(np.log10(kmin), np.log10(kmax), k_point)
        z = 0.0
        _, pkz = cosmo_need.class_Pk(z=z, k=kh, nonlinear=False)
        pk = pkz[0]
        As_target = None 
    else:
        raise ValueError("method is not supported")
    

    # pars.set_matter_power(redshifts=[0.0], kmax=kmax, nonlinear=False)
    # PK = get_matter_power_interpolator(pars, nonlinear=False, kmax=kmax, hubble_units=True, k_hunit=True, zmax=zmax)
    
    # kh = np.logspace(-3, np.log10(kmax), 1500)
    # pk = PK.P(0.0, kh)

    return {
        "kh": kh, 
        "pk": pk, 
        "As": As_target
    }

## Read and write parameter files (Only support lua format if the simulation supports, or txt format)
def read_params_lua(filename: str):
    import ast,re 
    from .io import analyze_float_format
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
    from .io import analyze_float_format
    import ast
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