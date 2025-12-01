import numpy as np 

from .base import comov_dist, VOLUME_FULL

def cal_full_volume(z_end, omega_m, w=-1.0, z_start=None, z_point=1000):
    if z_start is None:
        r_start = 0.0 
    else:
        r_start = comov_dist(z_start, omega_m, w, z_start=0.0, z_point=z_point)
    
    r_end = comov_dist(z_end, omega_m, w, z_start=0.0, z_point=z_point)

    return 4.0/3.0*np.pi*(r_end**3 - r_start**3)

def cal_survey_volume(V_angle, z_end, omega_m, w=-1.0, z_start=None, z_point=1000):
    V_full = cal_full_volume(z_end, omega_m, w, z_start=z_start, z_point=z_point)
    return V_angle / VOLUME_FULL * V_full