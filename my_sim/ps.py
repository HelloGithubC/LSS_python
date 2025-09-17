import numpy as np

def get_sigma8_0_camb(h, om, ob, As, ns, kmax=10.0):
    import camb 
    params = camb.CAMBparams()
    params.NonLinear = camb.model.NonLinear_none
    params.set_cosmology(H0=h*100, ombh2=ob*h*h, omch2=om*h*h)
    params.InitPower.set_params(ns=ns, As=As)
    params.set_matter_power(redshifts=[0.0], kmax=kmax)
    results = camb.get_results(params)
    sigma8_now = results.get_sigma8_0()
    return sigma8_now
def create_PS_camb(cosmo_params, z, kminh, kmaxh, npoints, As_max_times=20, process=False, return_result=False):
    """ Create a matter power spectrum using CAMB
    Args:
        cosmo_params: dictionary with the cosmological parameters
            'h': Hubble parameter(H/100.0)
            'om': Omega matter(Not Omh**2)
            'ob': Omega baryon
            'As': scalar amplitude of the primordial spectrum
            'ns': spectral index of the primordial spectrum
            'sigma8': sigma8. If set, sigma8 is used to set As. If As is set at the same time, the value of As is used as initial value.
        z: redshift
        kminh: minimum k in h/Mpc
        kmaxh: maximum k in h/Mpc
        npoints: number of points in the power spectrum
    """
    import camb 
    h = cosmo_params['h']
    om = cosmo_params['om']
    ob = cosmo_params['ob']
    As = cosmo_params.get('As', None)
    ns = cosmo_params["ns"]
    sigma8 = cosmo_params.get('sigma8', None)
    if As is None and sigma8 is None:
        raise ValueError("As and sigma8 are both not set")
    if sigma8 is not None:
        if As is None:
            if process:
                print("sigma8 is set, As is not set, use sigma8 to set As")
            As_upper = 3e-9
        else:
            if process:
                print("sigma8 is set, As is set, use As as initial value")
            As_upper = As

        As_lower = 1e-10
        for i in range(As_max_times):
            print(As_lower, As_upper)
            As_mid = 0.5 * (As_lower + As_upper)
            params = camb.CAMBparams()
            params.NonLinear = camb.model.NonLinear_none
            params.set_cosmology(H0=h*100, ombh2=ob*h*h, omch2=om*h*h)
            params.InitPower.set_params(ns=ns, As=As_mid)
            params.set_matter_power(redshifts=[z], kmax=10.0)
            results = camb.get_results(params)
            sigma8_now = results.get_sigma8_0()

            diff = sigma8_now - sigma8
            if abs(diff) < 1e-4:
                As = As_mid
                sigma8 = sigma8_now
                break

            if diff < 0:
                As_lower = As_mid
            else:
                As_upper = As_mid
            
        if process:
            if i >= As_max_times:
                print(f"Warning: As is not converged. The iteration times is {i:d}")
            print(f"As is set to {As:.3e} and sigma8 is set to {sigma8:.5f}")
    
    params = camb.CAMBparams()
    params.set_cosmology(H0=100.0*h, ombh2=ob*h**2, omch2=om*h**2)
    params.set_matter_power(redshifts=[z], kmax=kmaxh * h)
    params.InitPower.set_params(As=As, ns=ns)
    result = camb.get_results(params)

    k, z, pk = result.get_matter_power_spectrum(minkh=kminh, maxkh=kmaxh, npoints=npoints)
    
    if return_result:
        return k, pk[0], result
    else:
        return k, pk[0]

def create_PS_class(cosmo_params, z, kminh, kmaxh, npoints, process=False, return_result=False):
    """ Create a matter power spectrum using CAMB
    Args:
        cosmo_params: dictionary with the cosmological parameters
            'h': Hubble parameter(H/100.0)
            'om': Omega matter(Not Omh**2)
            'ob': Omega baryon
            'As': scalar amplitude of the primordial spectrum
            'ns': spectral index of the primordial spectrum
            'sigma8': sigma8. If set, sigma8 is used to set As. If As is set at the same time, the value of As is used as initial value.
        z: redshift
        kminh: minimum k in h/Mpc
        kmaxh: maximum k in h/Mpc
        npoints: number of points in the power spectrum
    """
    from classy import Class
    h = cosmo_params['h']
    om = cosmo_params['om']
    ob = cosmo_params['ob']
    As = cosmo_params.get('As', None)
    ns = cosmo_params["ns"]
    sigma8 = cosmo_params.get('sigma8', None)
    if As is None and sigma8 is None:
        raise ValueError("As and sigma8 are both not set")
    if sigma8 is not None:
        As = None 
    
    params = {
        'h': h,
        'omega_b': ob * h**2,
        'omega_cdm': om * h**2,
        # 'A_s': As_list[i],
        'n_s': ns,
        'output': 'mPk',
        'P_k_max_h/Mpc': 10.0,
        'z_pk': f'{z:.3f}',
        "sigma8": sigma8
        
    }
 
    # 运行CLASS
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()
    
    result = cosmo
    As = cosmo.get_current_derived_parameters(["A_s"])["A_s"]
    sigma8 = cosmo.sigma8()
    if process:
        print(f"As is set to {As:.3e} and sigma8 is set to {sigma8:.5f}")
    
    k_vals = np.logspace(np.log10(kminh), np.log10(kmaxh), npoints)  # h/Mpc
    kh_vals = k_vals * h # change to Mpc to calculate power spectrum
    pk_temp = [cosmo.pk_lin(k, z) for k in kh_vals]

    if return_result:
        return k_vals, np.array(pk_temp) * h**3, result
    else:
        return k_vals, np.array(pk_temp) * h**3
    
def create_PS(cosmo_params, z, kminh, kmaxh, npoints, method="CAMB", As_max_times=20, process=False, return_result=False, cosmology_model=None):
    if cosmology_model is not None:
        cosmo_params = cosmology_model.get_cosmo_params()
    if method == "CAMB" or method == "camb":
        return create_PS_camb(cosmo_params, z, kminh, kmaxh, npoints, As_max_times, process, return_result)
    elif method == "CLASS" or method == "class":
        return create_PS_class(cosmo_params, z, kminh, kmaxh, npoints, process, return_result)
    else:
        raise ValueError("method is not supported")

