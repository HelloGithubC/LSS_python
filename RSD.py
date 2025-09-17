import numpy as np 
from numba import njit, prange, set_num_threads

from .base import comov_dist_jit, CONST_C

@njit(parallel=True)
def add_rsd(pos, vel, redshift, boxsize, omega_m, w=-1.0, axis=2, num_threads=1):
    set_num_threads(num_threads)
    for i in prange(len(pos)):
        delta_z =  vel[i,axis] * (1.0 + redshift) / CONST_C
        delta_r = comov_dist_jit(redshift + delta_z, omega_m, w, redshift, 20)
        pos[i,axis] += delta_r
        if pos[i,axis] > boxsize:
            pos[i,axis] -= boxsize
        elif pos[i,axis] < 0.0:
            pos[i,axis] += boxsize
        else:
            pass