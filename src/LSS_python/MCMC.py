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


def get_chain(
    backend_filename, remove_exception=False, thin=5, discard_min=500, return_loglikes=False, flatten=True
,no_discard=False):
    exception_factor = 3
    if not os.path.exists(backend_filename):
        raise FileNotFoundError(f"{backend_filename} not exists")
    backend = HDFBackend(backend_filename, read_only=True)
    autocorr_time = int(np.max(backend.get_autocorr_time(tol=0)))

    discard = max(discard_min, autocorr_time) if not no_discard else 0
    if remove_exception:
        chains = backend.get_chain(
            thin=thin, discard=discard, flat=False
        )
        ndim = chains.shape[-1]
        except_index = []
        for i in range(chains.shape[1]):
            chain_temp = np.delete(chains, i, axis=1)
            mean_values = np.mean(chain_temp.reshape(-1,ndim), axis=0)
            std_values = np.std(chain_temp.reshape(-1,ndim), axis=0)
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
        ndim = chain.shape[-1]
    if flatten:
        chain = chain.reshape(-1, ndim)
    if return_loglikes:
        loglikes = backend.get_log_prob(thin=thin, discard=max(discard_min, autocorr_time), flat=False)
        if remove_exception:
            loglikes = np.delete(loglikes, except_index, axis=1)
        if flatten:
            loglikes = loglikes.ravel()
        return chain, loglikes
    else:
        return chain
    
def get_MCSample(
    backend_filename, remove_exception=False, thin=5, discard_min=500, return_loglikes=False, flatten=True, names=["x1", "x2"], labels=[r"\Omega_m", r"w"]
):
    from getdist import MCSamples
    if return_loglikes:
        chain, loglikes = get_chain(
            backend_filename, remove_exception=remove_exception, thin=thin, discard_min=discard_min, return_loglikes=return_loglikes, flatten=flatten
        )
        return MCSamples(samples=chain, names=names, labels=labels, loglikes=loglikes)
    else:
        chain = get_chain(
            backend_filename, remove_exception=remove_exception, thin=thin, discard_min=discard_min, return_loglikes=return_loglikes, flatten=flatten
        )
        return MCSamples(samples=chain, names=names, labels=labels)
    
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
                "converge_factor": backend.iteration / tau
            }
            return result_dict
        else:
            return converged
    else:
        raise NotImplementedError
    
def get_params(sampler, bestfit=False):
    names_source = sampler.getParamNames().names
    names = [name.name for name in names_source]
    means = sampler.getMeans()
    stds = np.sqrt(sampler.getVars())
    cov = sampler.getCov()
    output_dict = {
        "mean":{}, 
        "cov": None, 
        "std": {}, 
    }
    if bestfit:
        bestfit_index = np.argmax(sampler.loglikes)
        bestfit_values = sampler.samples[bestfit_index]
        output_dict["bestfit"] = {}
    for i, name in enumerate(names):
        output_dict["mean"][name] = means[i]
        output_dict["cov"] = cov 
        output_dict["std"][name] = stds[i]
        if bestfit:
            output_dict["bestfit"][name] = bestfit_values[i]
    return output_dict 

def get_area(sampler, params=["x1", "x2"], level=0.68):
    density_2d = sampler.get2DDensity(params[0], params[1])

    dx = density_2d.x[1] - density_2d.x[0]
    dy = density_2d.y[1] - density_2d.y[0]
    pixel_area = dx * dy

    level = density_2d.getContourLevels([level,])[0]
    return np.sum(density_2d.P > level) * pixel_area


def run_mcmc_core_emcee(nwalkers, ndim, init_state, lnprob, args, moves, backend, pool, max_iterator, use_converge_factor, converge_factor, verbose, progress_kwargs, output):
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
            
            result_dict = is_converged(sampler, converge_factor=converge_factor, method="tau", verbose=True)
            if verbose and not sampler.iteration % 1000:
                tau = result_dict["tau"]
                print(f"{sampler.iteration:d}: tau, {tau}; old tau: {old_tau}", file=output)
            converged = result_dict["converged"]
            tau = result_dict["tau"]
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            if converged:
                print(f"Converged at {sampler.iteration:d}", file=output)
                print(
                    f"Tau: {np.max(tau):.5f}; converge_factor_now: {np.min(sampler.iteration / tau):.1f}",
                    file=output,
                )
                print(
                    f"Tau difference: {np.max(np.abs(old_tau - tau) / tau):.5f}",
                    file=output,
                )
                break
            else:
                old_tau = tau

    if not use_converge_factor:
        result_dict = is_converged(sampler, converge_factor=converge_factor, method="tau", verbose=True)
        print(f"Finished at {max_iterator:d}", file=output)
        tau = result_dict["tau"]
        print(
            f"Tau: {np.max(tau):.5f}; converge_factor_now: {np.min(max_iterator / tau):.1f}",
            file=output,
        )
        converged = True
    
    if not converged:
        print(f"Not Converged at {max_iterator:d}", file=output)
        print(
            f"Tau: {np.max(tau):.5f}; converge_factor_now: {np.min(max_iterator / tau):.1f}",
            file=output,
        )
        print(
            f"Tau difference: {np.max(np.abs(old_tau - tau) / tau):.5f}",
            file=output,
        )
    return converged

def run_mcmc_main_emcee(
    init_state,
    max_iterator,
    nwalkers,
    ndim,
    lnprob,
    args,
    converge_factor=50,
    moves=None,
    backend=None,
    verbose=False,
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
    verbose: if True, print tau in progress(once per 1000 steps)
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
            converge_factor = converge_factor,
            verbose = verbose,
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
                converge_factor = converge_factor,
                verbose = verbose,
                progress_kwargs = progress_kwargs,
                output = output,
            )
    return converged


def check_convergence_cobaya(chains, converge_factor=1.05, parameters=None, verbose=False):
    """
    Check convergence for cobaya MCMC chains using Gelman-Rubin diagnostic.
    
    Args:
        chains: ndarray with shape (n_length, n_chains, n_parameters) or list of chains
        converge_factor: float, threshold for R-1 statistic (default 1.05)
        parameters: list of parameter names (optional)
        verbose: bool, if True return verboseed convergence information
    
    Returns:
        If verbose=False: bool indicating convergence
        If verbose=True: dict with convergence verboses
    """
    # Convert list of chains to array if needed
    if isinstance(chains, list):
        chains = np.array(chains)
    
    # Ensure shape is (n_length, n_chains, n_parameters)
    if chains.ndim != 3:
        raise ValueError(f"chains should be 3D array, got shape {chains.shape}")
    
    n_length, n_chains, n_parameters = chains.shape
    
    if parameters is None:
        parameters = [f"param{i+1:d}" for i in range(n_parameters)]
    
    R_dict = {}
    W_dict = {}
    B_dict = {}
    
    for i in range(n_parameters):
        parameter = parameters[i]
        
        # Calculate mean for each chain
        mean_chains = [np.mean(chains[:, j, i]) for j in range(n_chains)]
        mean_all = np.mean(mean_chains)
        
        # Between-chain variance
        B = n_length / (n_chains - 1) * np.sum((mean_chains - mean_all)**2)
        
        # Within-chain variance
        W = np.mean([np.var(chains[:, j, i], ddof=1) for j in range(n_chains)])
        
        # Pooled variance estimate
        var_plus = (n_length - 1) / n_length * W + 1 / n_length * B
        
        # R-1 statistic
        R_dict[parameter] = var_plus / W if W > 0 else np.inf
        W_dict[parameter] = W
        B_dict[parameter] = B
    
    max_R = np.max(list(R_dict.values()))
    converged = max_R < converge_factor
    
    if verbose:
        return {
            "converged": converged,
            "max_R": max_R,
            "R_dict": R_dict,
            "W_dict": W_dict,
            "B_dict": B_dict
        }
    else:
        return converged


def run_mcmc_main_cobaya(info, resume=False, force=False):
    """
    Run cobaya MCMC with simplified interface. All configuration should be provided in info dict.

    Args:
        info: dict, cobaya configuration dictionary (must contain params, likelihood, and optionally output, sampler, etc.)
        resume: bool, resume from previous run
        force: bool, force overwrite existing output

    Returns:
        dict with 'converged' bool, 'info', and 'sampler'
    """
    from cobaya.run import run

    # Handle output filename: if not resuming/forcing, avoid overwriting existing output
    if not resume and not force:
        output = info.get('output')
        if output is not None:
            import os
            # Find the next available suffix
            counter = 0
            while True:
                candidate = output if counter == 0 else f"{output}_{counter}"
                candidate_dir = os.path.dirname(candidate) or '.'
                candidate_base = os.path.basename(candidate)

                # Check if any file with this prefix exists in the directory
                if candidate_dir == '.':
                    exists = any(
                        f.startswith(candidate_base)
                        for f in os.listdir('.')
                        if os.path.isfile(f)
                    )
                else:
                    exists = any(
                        f.startswith(candidate_base)
                        for f in os.listdir(candidate_dir)
                        if os.path.isfile(os.path.join(candidate_dir, f))
                    )

                if not exists:
                    info['output'] = candidate
                    break
                counter += 1

    # Run cobaya
    updated_info, sampler = run(info, resume=resume, force=force)

    # Get convergence status from sampler
    converged = getattr(sampler, 'converged', False)

    return {
        'converged': converged,
        'info': updated_info,
        'sampler': sampler
    }