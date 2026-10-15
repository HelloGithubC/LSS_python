import numpy as np 

from .base import comov_dist, VOLUME_FULL


def cal_full_volume_r(r_start, r_end):
    """Full-sky comoving volume of a spherical shell.

    Parameters
    ----------
    r_start : float
        Inner comoving radius (Mpc/h).  Use 0.0 for a sphere.
    r_end : float
        Outer comoving radius (Mpc/h).

    Returns
    -------
    float
        Volume = 4π/3 × (r_end³ − r_start³)  (Mpc/h)³
    """
    return 4.0 / 3.0 * np.pi * (r_end**3 - r_start**3)


def cal_survey_volume_r(V_angle, r_start, r_end):
    """Survey volume for a sky patch given comoving radii.

    Parameters
    ----------
    V_angle : float
        Solid angle of the sky patch in square degrees.
    r_start : float
        Inner comoving radius (Mpc/h).  Use 0.0 to start from the origin.
    r_end : float
        Outer comoving radius (Mpc/h).

    Returns
    -------
    float
        Volume = (V_angle / 41252.96) × 4π/3 × (r_end³ − r_start³)  (Mpc/h)³
    """
    return V_angle / VOLUME_FULL * cal_full_volume_r(r_start, r_end)


def cal_full_volume_z(z_end, omega_m, w=-1.0, z_start=None, z_point=1000):
    """Full-sky comoving volume, with redshift inputs.

    Converts redshifts to comoving distances and delegates to
    :func:`cal_full_volume_r`.
    """
    if z_start is None:
        r_start = 0.0
    else:
        r_start = comov_dist(z_start, omega_m, w, z_start=0.0, z_point=z_point)

    r_end = comov_dist(z_end, omega_m, w, z_start=0.0, z_point=z_point)
    return cal_full_volume_r(r_start, r_end)


def cal_survey_volume_z(V_angle, z_end, omega_m, w=-1.0, z_start=None, z_point=1000):
    """Survey volume for a sky patch, with redshift inputs.

    Converts redshifts to comoving distances and delegates to
    :func:`cal_survey_volume_r`.
    """
    V_full = cal_full_volume_z(z_end, omega_m, w, z_start=z_start, z_point=z_point)
    return V_angle / VOLUME_FULL * V_full


# Backward-compatible aliases (original names without suffix)
cal_full_volume = cal_full_volume_z
cal_survey_volume = cal_survey_volume_z