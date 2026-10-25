"""Random-catalog helpers for marked-correlation workflows.

Density and gradient calculations live in :mod:`LSS_python.marks`.
"""

import numpy as np

from LSS_python.base import Hz, DA


def create_random(omega_mf, w_f, omega_mm, w_m, redshift, boxsize_source, npar, return_boxsize_new=False):
    Hz_f, Hz_m = Hz(redshift, omega_mf, w_f), Hz(redshift, omega_mm, w_m)
    DA_f, DA_m = DA(redshift, omega_mf, w_f), DA(redshift, omega_mm, w_m)
    convert_factor = np.array([DA_m / DA_f, DA_m / DA_f, Hz_f / Hz_m])
    boxsize = boxsize_source * convert_factor
    random = np.random.uniform(0, boxsize, size=(int(npar * np.prod(boxsize)), 3)).astype(np.float32)
    if return_boxsize_new:
        return random, boxsize 
    else:
        return random
