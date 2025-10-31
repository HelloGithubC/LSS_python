import os 
import numpy as np 
import emcee 
from emcee.backends import HDFBackend
from multiprocessing import Pool

def get_chain(
    backend_filename, is_CPL=False, remove_exception=False, thin=5, discard_min=500, return_loglikes=False, flatten=True
):
    dim = 3 if is_CPL else 2
    exception_factor = 2
    if not os.path.exists(backend_filename):
        raise FileNotFoundError(f"{backend_filename} not exists")
    backend = HDFBackend(backend_filename, read_only=True)
    autocorr_time = int(np.max(backend.get_autocorr_time(tol=0)))
    if remove_exception:
        chains = backend.get_chain(
            thin=thin, discard=max(discard_min, autocorr_time), flat=False
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
            thin=thin, discard=max(discard_min, autocorr_time), flat=False
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

    
def run_mcmc_core_emcee(nwalkers, ndim, init_state, lnprob, args, moves, backend, pool, max_iterator, use_converge_factor, converge_factor, detail, progress_kwargs, output):
    sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            lnprob,
            args=args,
            moves=moves,
            backend=backend,
            pool=pool,
        )
    autocorr = np.empty((max_iterator))
    indent = 0
    i = 1
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
            tau = sampler.get_autocorr_time(tol=0)
            autocorr[indent] = np.mean(tau)
            indent += 1

            if detail and not sampler.iteration % 1000:
                print(f"{i:d}: tau, {tau:.5f}", file=output)
                i += 1
            converged = np.all(tau * converge_factor < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            if converged:
                print(f"Converged at {sampler.iteration:d}", file=output)
                print(
                    f"Tau: {np.mean(tau):.5f}; converge_factor_now: {np.max(sampler.iteration / tau):.1f}",
                    file=output,
                )
                print(
                    f"Tau difference: {np.min(np.abs(old_tau - tau) / tau):.5f}",
                    file=output,
                )
                break
            else:
                old_tau = tau
        else:
            if sampler.iteration >= max_iterator:
                print(f"Finished at {max_iterator:d}", file=output)
                print(
                    f"Tau: {np.mean(tau):.5f}; converge_factor_now: {np.max(max_iterator / tau):.1f}",
                    file=output,
                )
                print(
                    f"Tau difference: {np.min(np.abs(old_tau - tau) / tau):.5f}",
                    file=output,
                )
                autocorr = 0.0
                converged = True
                break 
    
    if not converged:
        print(f"Not Converged at {max_iterator:d}", file=output)
        print(
            f"Tau: {np.mean(tau):.5f}; converge_factor_now: {np.max(max_iterator / tau):.1f}",
            file=output,
        )
        print(
            f"Tau difference: {np.min(np.abs(old_tau - tau) / tau):.5f}",
            file=output,
        )
    return autocorr, converged

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
        autocorr, converged = run_mcmc_core_emcee(
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
            detail = detail,
            progress_kwargs = progress_kwargs,
            output = output,
        )
    else:
        with Pool(processes=int(nwalkers / 2)) as pool:
            autocorr, converged = run_mcmc_core_emcee(
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
                detail = detail,
                progress_kwargs = progress_kwargs,
                output = output,
            )
    return autocorr, converged