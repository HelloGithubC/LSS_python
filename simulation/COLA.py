import os 

import numpy as np
from astropy.cosmology import Planck13
import camb 
from camb import CAMBparams
import colibri.cosmology as cc 

def create_PS(omega_m0, w0, kmax, sigma8_kmax=10.0, verbose=False, params=None, method="CAMB", kmin=1e-3, k_point=1500):
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