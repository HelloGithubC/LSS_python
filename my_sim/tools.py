import numpy as np

from .cola import read_snapshot

# About 2-point statistics
def intergrate_fftpower(fftpower, kmin=-1, kmax=-1, mu_min=-1, mu_max=-1, integrate="k", norm=False):
    power = fftpower.power 
    k_array = np.nanmean(power["k"], axis=1)
    mu_array = np.nanmean(power["mu"], axis=0)
    if kmin<0:
        k_min_index = 0
    else:
        k_min_index = np.where(k_array >= kmin)[0][0]
    if kmax<0:
        k_max_index = len(k_array)
    else:
        k_max_index = np.where(k_array > kmax)[0][0]

    if mu_min<0:
        mu_min_index = 0
    else:
        mu_min_index = np.where(mu_array >= mu_min)[0][0]
    if mu_max<0:
        mu_max_index = len(mu_array)
    else:
        mu_max_index = np.where(mu_array > mu_max)[0][0]

    Pkmu_select = np.real(power["Pkmu"][k_min_index:k_max_index, mu_min_index:mu_max_index]) - fftpower.attrs["shotnoise"]
    if integrate == "k":
        Pkmu_integrate = np.nanmean(Pkmu_select, axis=0)
        if norm:
            Pkmu_integrate = Pkmu_integrate / np.nanmean(Pkmu_integrate)
        return mu_array[mu_min_index: mu_max_index], Pkmu_integrate
    elif integrate == "mu":
        Pkmu_integrate = np.nanmean(Pkmu_select, axis=1)
        if norm:
            Pkmu_integrate = Pkmu_integrate / np.nanmean(Pkmu_integrate)
        return k_array[k_min_index: k_max_index], Pkmu_integrate
    else:
        raise ValueError("integrate must be k or mu")

def integrate_tpCF(tpCF_result, smin, smax, mumax, integrate="k", norm=False): 
    """
    tpCF_result: dict. result of cal_tpCF_from_pairs
    integrate: "mu" or "s"

    Returns:
    s, s**2*xis_s: integrate="mu"
    mu, xismu: integrate="s"
    """

    if "DDwpairs" in tpCF_result:
        with_weight = True 
    else:
        with_weight = False
    
    s_array = (tpCF_result["sedges"][:-1] + tpCF_result["sedges"][1:]) / 2.0
    smin_index = np.where(s_array > smin)[0][0]
    smax_index = np.where(s_array < smax)[0][-1]
    s = s_array[smin_index:smax_index+1]

    mu_array = (tpCF_result["muedges"][:-1] + tpCF_result["muedges"][1:]) / 2.0
    mumax_index = np.where(mu_array < mumax)[0][-1]
    mu = mu_array[:mumax_index+1]

    if with_weight:
        DD_need = tpCF_result["DDwpairs"][smin_index:smax_index+1, :mumax_index+1]
        DR_need = tpCF_result["DRwpairs"][smin_index:smax_index+1, :mumax_index+1]
        RR_need = tpCF_result["RRwpairs"][smin_index:smax_index+1, :mumax_index+1]
    else:
        DD_need = tpCF_result["DDnpairs"][smin_index:smax_index+1, :mumax_index+1]
        DR_need = tpCF_result["DRnpairs"][smin_index:smax_index+1, :mumax_index+1]
        RR_need = tpCF_result["RRnpairs"][smin_index:smax_index+1, :mumax_index+1]
    

    dd1d2 = DD_need / tpCF_result["norm_d1d2"]
    dd1r2 = DR_need / tpCF_result["norm_d1r2"]
    dr1d2 = DR_need / tpCF_result["norm_r1d2"]
    dr1r2 = RR_need / tpCF_result["norm_r1r2"]

    dr1r2_remove_0 = np.copy(dr1r2)
    dr1r2_remove_0[dr1r2_remove_0 == 0] = 1e-15

    if integrate == "mu":
        xis_s = (dd1d2.sum(axis=1) - dr1d2.sum(axis=1) - dd1r2.sum(axis=1) + dr1r2.sum(axis=1)) / dr1r2_remove_0.sum(axis=1)
        # xis_s = np.mean(xis_source, axis=1)
        if norm:
            mean_xis_s = np.mean(xis_s)
            if mean_xis_s == 0:
                print("Warning: mean_xis_s is 0, set to 1e-15")
                xis_s = xis_s / 1e-15
            else:
                xis_s = xis_s / mean_xis_s
        return s, xis_s * s**2 
    elif integrate == "s":
        xis_source = (dd1d2 - dr1d2 - dd1r2 + dr1r2) / dr1r2_remove_0
        xismu = np.mean(xis_source, axis=0)
        if norm:
            xismu = xismu / np.mean(xismu)
        return mu, xismu
    else:
        raise ValueError("integrate must be 'mu' or 's'")
    
# About data conversion
def convert_cola_snapshot_to_structed_npy(cola_snapshot_filename):
    cola_snapshot = read_snapshot(cola_snapshot_filename)
    num = cola_snapshot["header"][0]["np"][1] # header is a list and the second element of np is the number of dark matter particles
    npy_dtype = np.dtype(
        [("Position", np.float32, 3), 
         ("Vel", np.float32, 3)]
    )
    npy_array = np.empty(num, dtype=npy_dtype)
    npy_array["Position"] = cola_snapshot["position"]
    npy_array["Vel"] = cola_snapshot["vel"]
    return npy_array

def convert_cola_halo_to_structed_npy(cola_halo_filename):
    cola_halo = np.loadtxt(cola_halo_filename)
    npy_dtype = np.dtype(
        [("Position", np.float32, 3), 
         ("Vel", np.float32, 3)]
    )
    npy_array = np.empty(cola_halo.shape[0], dtype=npy_dtype)
    npy_array["Position"] = cola_halo[:, 1:4]
    npy_array["Vel"] = cola_halo[:, 4:7]
    return npy_array

def convert_rockstar_to_structured_npy(rockstar_filename):
    npy_dtype = np.dtype([
        ("ID", np.int32), 
        ("Pos", np.float64, 3),
        ("Vel", np.float64, 3),
        ("Mvir", np.float64),
        ("Rvir", np.float64),
    ])

    dtype_cols_dict = {
        "ID": 0,
        "Pos": slice(8, 11),
        "Vel": slice(11, 14),
        "Mvir": 2,
        "Rvir": 5
    }

    source = np.loadtxt(rockstar_filename, dtype=np.float64)
    data_npy = np.empty(shape=(len(source),), dtype=npy_dtype)

    for key, col in dtype_cols_dict.items():
        data_npy[key] = source[:,col]
    return data_npy

# Asistant functions
def Hz(omegam, w, h, z):
    return (
        100
        * h
        * np.sqrt(
            omegam * (1.0 + z) ** 3.0 + (1.0 - omegam) * (1.0 + z) ** (3.0 * (1 + w))
        )
    )

def Hz_CPL(omegam, w0, wa, h, z):
    return (
        100
        * h
        * np.sqrt(
            omegam * (1.0 + z) ** 3.0
            + (1.0 - omegam)
            * np.exp(3.0 * ((1.0 + w0 + wa) * (np.log(1.0 + z)) - wa * z / (1.0 + z)))
        )
    )

def Hz_Mpc_to_h(omegam, w, z):
    return 100 * np.sqrt(
        omegam * (1.0 + z) ** 3.0 + (1.0 - omegam) * (1.0 + z) ** (3.0 * (1 + w))
    )

def comov_r(omegam, w, z, is_CPL=False, wa=0.5):
    h = 0.667
    CONST_C = 299792.458
    if is_CPL:
        w0 = w
        wa = wa
        x = np.linspace(0, z, 1000)
        y = np.empty(x.shape[0])
        for i, xx in enumerate(x):
            y[i] = 1.0 / Hz_CPL(omegam, w0, wa, h, xx)

    else:
        x = np.linspace(0, z, 1000)
        y = np.empty(x.shape[0])
        for i, xx in enumerate(x):
            y[i] = 1.0 / Hz(omegam, w, h, xx)
    return CONST_C * np.trapezoid(y, x) * h

def DA(omegam, w, z, is_CPL=False, wa=0.0):
    return comov_r(omegam, w, z, is_CPL, wa) / (1.0 + z)

def add_rsd(data_npy, BoxSize, redshift, omega_m, w0=-1.0, wa=0.0, h=0.7, pos_col=2):
    """
    Add RSD to data_npy.
    data_npy: structured numpy array, including position(Pos, Position, position, pos) and 
            velocity(Vel, Velocity, velocity, vel). The precision will be limit to float32
    """
    from redshift_space_library import pos_redshift_space
    if "Pos" in data_npy.dtype.names:
        pos_name = "Pos"
    elif "Position" in data_npy.dtype.names:
        pos_name = "Position"
    elif "position" in data_npy.dtype.names:
        pos_name = "position"
    elif "pos" in data_npy.dtype.names:
        pos_name = "pos"
    else:
        raise ValueError("data_npy does not include position")
    
    if "Vel" in data_npy.dtype.names:
        vel_name = "Vel"
    elif "Velocity" in data_npy.dtype.names:
        vel_name = "Velocity"
    elif "velocity" in data_npy.dtype.names:
        vel_name = "velocity"
    elif "vel" in data_npy.dtype.names:
        vel_name = "vel"
    else:
        raise ValueError("data_npy does not include velocity")
    
    Hubble = Hz_CPL(omegam=omega_m, w0=w0, wa=wa, h=h, z=redshift)
    pos = np.copy(data_npy[pos_name]).astype(np.float32)
    vel = np.copy(data_npy[vel_name]).astype(np.float32)
    pos_redshift_space(pos, vel, BoxSize, Hubble, redshift, axis=pos_col)
    data_npy[pos_name] = pos