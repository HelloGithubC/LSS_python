import os 
import numpy as np 
import emcee 
from emcee.backends import HDFBackend
from multiprocessing import Pool

def ln_prior(X, X_range):
    for x, x_range in zip(X, X_range):
        if x < x_range[0] or x > x_range[1]:
            return -np.inf
    return 0.0

def ln_prob(X, X_range, chi2_core):
    ln_prior_value = ln_prior(X, X_range)
    if np.isinf(ln_prior_value):
        return ln_prior_value
    else:
        return ln_prior_value - 0.5 * chi2_core(X)

def get_xismus_convert(source_xismu_dict, asistant_xismus_dict, snaps_pair, weipows_str_list, parameters, redshift_dict, fiducial_parameters=(0.3071, -1.0), mapping_s=(3, 70), **args):
    snap1, snap2 = snaps_pair
    omega_mm, w_m = parameters
    omega_mf, w_f = fiducial_parameters 
    new_xismus_dict = {
        weipow_str: {} 
        for weipow_str in weipows_str_list
    }

    smin = args.get("smin", 6.0)
    smax = args.get("smax", 40.0)
    mupack = args.get("mupack", 6)
    mumax = args.get("mumax", 0.97)

    for snap in snaps_pair:
        for weipow_str in weipows_str_list:
            xismu_source_temp = source_xismu_dict[weipow_str][snap]
            new_xismu = xismu_source_temp.cosmo_conv_DenseToSparse(
                omstd = omega_mf,
                wstd = w_f,
                omwrong = omega_mm,
                wwrong = w_m,
                redshift = redshift_dict[snap],
                smin_mapping=mapping_s[0],
                smax_mapping=mapping_s[1],
                assistant_xismu = asistant_xismus_dict[weipow_str]
            )
            new_xismus_dict[weipow_str][snap] = new_xismu
    xismus_diff_weipows_dict = {}
    for weipow_str in weipows_str_list:
        mu_temp, xis_mu_temp_1 = new_xismus_dict[weipow_str][snap1].integrate_tpcf(smin=smin, smax=smax, mupack=mupack, mumax=mumax, is_norm=True, intximu=True, quick_return=True)
        mu_temp, xis_mu_temp_2 = new_xismus_dict[weipow_str][snap2].integrate_tpcf(smin=smin, smax=smax, mupack=mupack,  mumax=mumax, is_norm=True, intximu=True, quick_return=True)
        xismus_diff_weipows_dict[weipow_str] = xis_mu_temp_1 - xis_mu_temp_2
    return xismus_diff_weipows_dict
def get_xismus_diff_concatenate_mcmc(xismus_diff_weipows_dict, weipows_str_list):
    xismus_diff_weipows_list = []
    for weipow_str in weipows_str_list:
        xismus_diff_weipows_list.append(xismus_diff_weipows_dict[weipow_str])
    return np.concatenate(xismus_diff_weipows_list)


def get_chain(
    backend_filename, is_CPL=False, remove_exception=False, thin=5, discard_min=500, return_loglikes=False, flatten=True
,no_discard=False):
    dim = 3 if is_CPL else 2
    exception_factor = 2
    if not os.path.exists(backend_filename):
        raise FileNotFoundError(f"{backend_filename} not exists")
    backend = HDFBackend(backend_filename, read_only=True)
    autocorr_time = int(np.max(backend.get_autocorr_time(tol=0)))

    discard = max(discard_min, autocorr_time) if not no_discard else 0
    if remove_exception:
        chains = backend.get_chain(
            thin=thin, discard=discard, flat=False
        )
        except_index = []
        for i in range(chains.shape[1]):
            chain_temp = np.delete(chains, i, axis=1)
            mean_values = np.mean(chain_temp.reshape(-1,2), axis=0)
            std_values = np.std(chain_temp.reshape(-1,2), axis=0)
            mean = np.mean(chains[:, i], axis=0)
            if (mean > (mean_values + exception_factor * std_values)).any() or (
                mean < (mean_values - exception_factor * std_values)
            ).any():
                except_index.append(i)
        chain = np.delete(chains, except_index, axis=1)

    else:
        chain = backend.get_chain(
            thin=thin, discard=discard, flat=False
        )
    if flatten:
        chain = chain.reshape(-1, dim)
    if return_loglikes:
        loglikes = backend.get_log_prob(thin=thin, discard=max(discard_min, autocorr_time), flat=False)
        if remove_exception:
            loglikes = np.delete(loglikes, except_index, axis=1)
        if flatten:
            loglikes = loglikes.ravel()
        return chain, loglikes
    else:
        return chain
    
def is_converged_GR(chains, converge_factor=1.05, parameters=None, return_W_B_R=False): # shape = (n_chain, n_length, n_paramters)
    n_length, n_chain, n_paramters = chains.shape
    if parameters is None:
        parameters = [f"param{i+1:d}" for i in range(n_paramters)]
    R_dict = {
        parameter: None 
        for parameter in parameters
    }
    if return_W_B_R:
        B_dict = {
            parameter: None 
            for parameter in parameters
        }
        W_dict = {
            parameter: None 
            for parameter in parameters
        }
    for i in range(n_paramters):
        parameter = parameters[i]
        mean_chains = [np.mean(chains[:, j, i]) for j in range(n_chain)]
        mean_all = np.mean(mean_chains)
        B = n_length / (n_chain - 1) * np.sum((mean_chains - mean_all)**2)
        W = np.mean([np.var(chains[:,j, i], ddof=1) for j in range(n_chain)])
        if return_W_B_R:
            W_dict[parameter] = (n_length - 1) / n_length * W
            B_dict[parameter] = 1 / n_length * B
        R_dict[parameter] = ((n_length - 1) / n_length * W + 1 / n_length * B) / W
    if return_W_B_R:
        return R_dict, W_dict, B_dict
    else:
        return np.max(list(R_dict.values())) < converge_factor
    
def is_converged(backend, converge_factor=None, method="GR", verbose=False): # method: GR, tau
    if method == "GR":
        chains = backend.get_chain()
        R_dict, W_dict, B_dict = is_converged_GR(chains, converge_factor, return_W_B_R=True)
        converage = (np.max(list(R_dict.values())) < converge_factor)
        if verbose:
            result_dict = {
                "converged": converage,
                "R": list(R_dict.values()),
                "W": list(W_dict.values()),
                "B": list(B_dict.values())
            }
            return result_dict
        else:
            return converage
    elif method == "tau":
        tau = backend.get_autocorr_time(tol=0)
        converged = np.all(tau * converge_factor < backend.iteration)
        if verbose:
            result_dict = {
                "converged": converged,
                "tau": tau,
                "converge_factor": backend.iteration / converge_factor
            }
            return result_dict
        else:
            return converged
    else:
        raise NotImplementedError

def run_mcmc_core_emcee(nwalkers, ndim, init_state, lnprob, args, moves, backend, pool, max_iterator, use_converge_factor, converge_factor, converge_method, detail, progress_kwargs, output):
    sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            lnprob,
            args=args,
            moves=moves,
            backend=backend,
            pool=pool,
        )
    old_tau = np.inf
    for sample in sampler.sample(
        init_state,
        iterations=max_iterator,
        progress=True,
        progress_kwargs=progress_kwargs,
    ):
        if use_converge_factor:
            if sampler.iteration % 100:
                continue
            
            old_tau = np.inf
            result_dict = is_converged(sampler, converge_factor=converge_factor, method=converge_method, verbose=True)
            if detail and not sampler.iteration % 1000:
                if converge_method == "tau":
                    tau = result_dict["tau"]
                    print(f"{sampler.iteration:d}: tau, {tau:.5f}", file=output)
                elif converge_method == "GR":
                    print(f"{sampler.iteration:d}: R, {result_dict['R']}", file=output)
                else:
                    pass
            converged = result_dict["converged"]
            if converge_method == "tau":
                converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
                old_tau = tau
            if converged:
                print(f"Converged at {sampler.iteration:d}", file=output)
                if converge_method == "tau":
                    print(
                        f"Tau: {np.mean(tau):.5f}; converge_factor_now: {np.max(sampler.iteration / tau):.1f}",
                        file=output,
                    )
                    print(
                        f"Tau difference: {np.min(np.abs(old_tau - tau) / tau):.5f}",
                        file=output,
                    )
                elif converge_method == "GR":
                    print(f"R: {result_dict['R']}", file=output)
                else:
                    pass
                break

    if not use_converge_factor:
        result_dict = is_converged(sampler, converge_factor=converge_factor, method=converge_method, verbose=True)
        print(f"Finished at {max_iterator:d}", file=output)
        if converge_method == "tau":
            tau = result_dict["tau"]
            print(
                f"Tau: {np.mean(tau):.5f}; converge_factor_now: {np.max(max_iterator / tau):.1f}",
                file=output,
            )
        elif converge_method == "GR":
            print(f"R: {result_dict['R']}", file=output)
        else:
            pass
        converged = True
    
    if not converged:
        print(f"Not Converged at {max_iterator:d}", file=output)
        if converge_method == "tau":
            print(
                f"Tau: {np.mean(tau):.5f}; converge_factor_now: {np.max(max_iterator / tau):.1f}",
                file=output,
            )
            print(
                f"Tau difference: {np.min(np.abs(old_tau - tau) / tau):.5f}",
                file=output,
            )
        elif converge_method == "GR":
            print(f"R: {result_dict['R']}", file=output)
        else:
            pass
    return converged

def run_mcmc_main_emcee(
    init_state,
    max_iterator,
    nwalkers,
    ndim,
    lnprob,
    args,
    converge_method="tau",
    converge_factor=50,
    moves=None,
    backend=None,
    detail=False,
    desc_str=None,
    output=None,
    force_no_pool=False
):
    """To run mcmc
    Args:
    init_state: ndarray or state, meaning starting mcmc or continuing mcmc
    max_iterator: int, a maximum step to stop mcmc. If converted before it, stop.
    nwalkers: int, the number of chains.
    lnprob: fun, the function of lnprob.
    args: tunble, the args for lnprob
    backend: if None, just store in memory; if HDFBackend class, will store in specific file.
    detail: if True, print tau in progress(once per 1000 steps)
    output: redirect output.

    Returns:
    autocorr: ndarray, storing autocorr.


    """
    if converge_factor <= 0:
        use_converge_factor = False 
        print("converge_factor is negative or zero, so it will not be used.", file=output)
    else:
        use_converge_factor = True

    progress_kwargs = {"desc": desc_str} if desc_str else {"desc": "Running MCMC"}

    converged = True
    if force_no_pool:
        converged = run_mcmc_core_emcee(
            nwalkers = nwalkers,
            ndim = ndim,
            init_state = init_state,
            lnprob = lnprob,
            args = args,
            moves = moves,
            backend = backend,
            pool = None,
            max_iterator = max_iterator,
            use_converge_factor = use_converge_factor,
            converge_method=converge_method,
            converge_factor = converge_factor,
            detail = detail,
            progress_kwargs = progress_kwargs,
            output = output,
        )
    else:
        with Pool(processes=int(nwalkers / 2)) as pool:
            converged = run_mcmc_core_emcee(
                nwalkers = nwalkers,
                ndim = ndim,
                init_state = init_state,
                lnprob = lnprob,
                args = args,
                moves = moves,
                backend = backend,
                pool = pool,
                max_iterator = max_iterator,
                use_converge_factor = use_converge_factor,
                converge_method=converge_method,
                converge_factor = converge_factor,
                detail = detail,
                progress_kwargs = progress_kwargs,
                output = output,
            )
    return converged