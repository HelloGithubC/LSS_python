import numpy as np 

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
 ])

def read_snapshot(filename, only_read_header=False):
    if isinstance(filename, str):
        filename_list = [filename]
    elif isinstance(filename, list) or isinstance(filename, tuple) or isinstance(filename, np.ndarray):
        filename_list = filename
    else:
        raise ValueError("filename must be a string, list, tuple or numpy array")
    header_list = []
    position_list = []
    vel_list = []
    for filename in filename_list:
        f = open(filename, "rb")
        header_flag_1 = np.fromfile(f, dtype=np.int32, count=1)
        if header_flag_1[0] != gadget2_header_dtype.itemsize:
            raise ValueError("file {} is not a gadget2 snapshot(header_flag wrong)".format(filename))
        header = np.fromfile(f, dtype=gadget2_header_dtype, count=1)
        npar = header[0]["np"][1] # dark matter
        header_flag_2 = np.fromfile(f, dtype=np.int32, count=1)
        if header_flag_2[0] != gadget2_header_dtype.itemsize:
            raise ValueError("file {} is not a gadget2 snapshot(header_flag not consistent)".format(filename))
        header_list.append(header[0])
        
        if not only_read_header:
            position_flag_1 = np.fromfile(f, dtype=np.int32, count=1)
            # if position_flag_1[0] != npar*3*4: # 4 == sizeof(float32)
            #     raise ValueError("file {} is not a gadget2 snapshot(position_flag wrong)".format(filename))
            position = np.fromfile(f, dtype=np.float32, count=npar*3).reshape(npar, 3)
            position_flag_2 = np.fromfile(f, dtype=np.int32, count=1)
            if position_flag_2[0] != position_flag_1[0]:
                raise ValueError("file {} is not a gadget2 snapshot(position_flag not consistent)".format(filename))

            vel_flag_1 = np.fromfile(f, dtype=np.int32, count=1)
            # if vel_flag_1[0] != npar*3*4: # 4 == sizeof(float32)
            #     raise ValueError("file {} is not a gadget2 snapshot(vel_flag wrong)".format(filename))
            vel = np.fromfile(f, dtype=np.float32, count=npar*3).reshape(npar, 3)
            vel_flag_2 = np.fromfile(f, dtype=np.int32, count=1)
            if vel_flag_1[0] != vel_flag_2[0]:
                raise ValueError("file {} is not a gadget2 snapshot(vel_flag not consistent)".format(filename))

            position_list.append(position)
            vel_list.append(vel)
        f.close()
    if only_read_header:
        position_array = None 
        vel_array = None
    else:
        position_array = np.concatenate(position_list, axis=0)
        vel_array = np.concatenate(vel_list, axis=0)
    return {
        "position": position_array,
        "vel": vel_array,
        "header": header_list,
    }
