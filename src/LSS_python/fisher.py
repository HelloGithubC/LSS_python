import numpy as np
from scipy.stats import chi2, norm

from LSS_python.AP import tpcf_convert_main


def _legacy_cal_jacobian(func, best_fit, delta=None, args=()):
    """
    Compute the Jacobian matrix of a model function via central finite differences.

    The Jacobian is defined as J[k, i] = ∂μ_k / ∂θ_i, where μ = func(θ, *args)
    is the model prediction. Each partial derivative is approximated by
    the central difference:

        ∂f/∂θ_i ≈ [f(θ + δ_i e_i, *args) - f(θ - δ_i e_i, *args)] / (2 δ_i)

    Parameters
    ----------
    func : callable
        Model function whose first argument is the parameter vector
        (list, tuple or ndarray, same length as `best_fit`), followed by
        any additional arguments collected in `args`.
        Example: for 2 parameters, func(params, a) with params = [x1, x2].
    best_fit : array_like
        Parameter values at which to evaluate the Jacobian.
    delta : float or array_like, optional
        Finite difference step size. If None, automatically determined
        using optimal step size δ = ε^(1/3) * max(|θ|, 1) for each parameter,
        where ε ≈ 2.22e-16 is machine epsilon. This gives 4th order
        accuracy for central differences. Can also be a single float
        applied to all parameters, or an array matching best_fit length.
    args : tuple, optional
        Extra positional arguments passed to `func` after the parameter
        vector, in the same way as `scipy.integrate.quad`.

    Returns
    -------
    jacobian : ndarray, shape (n_output, n_params)
        Jacobian matrix. n_output is the model output dimension (1 for
        scalar models) and n_params is the number of parameters.

    Notes
    -----
    The automatic step size is based on the optimal step for numerical
    differentiation to minimize truncation and round-off errors:

    δ_optimal ≈ ε^(1/3) * max(|x|, 1)

    where ε is machine epsilon. This balances the truncation error
    (∝ δ²) and round-off error (∝ 1/δ) in central differences.
    """
    # Convert inputs to numpy arrays
    best_fit = np.atleast_1d(best_fit)
    n_params = len(best_fit)

    # Determine step sizes for each parameter
    if delta is None:
        # Automatic step size: δ = ε^(1/3) * max(|θ|, 1)
        # ε^(1/3) ≈ 6.05e-6 for double precision
        machine_eps = np.finfo(float).eps
        delta_factor = machine_eps ** (1/3)
        delta = delta_factor * np.maximum(np.abs(best_fit), 1.0)
    elif np.isscalar(delta):
        delta = np.full(n_params, delta)
    else:
        delta = np.atleast_1d(delta)
        if len(delta) != n_params:
            raise ValueError(f"delta length {len(delta)} does not match "
                            f"number of parameters {n_params}")

    # Evaluate function at best fit point to determine output dimension
    f0 = func(best_fit, *args)

    # Determine if model output is scalar or vector
    if hasattr(f0, '__len__') and not isinstance(f0, (float, int)):
        f0 = np.asarray(f0)
        n_output = f0.shape[0] if f0.ndim > 0 else 1
    else:
        n_output = 1

    # Initialize Jacobian matrix (n_output x n_params)
    jacobian = np.zeros((n_output, n_params))

    # Compute gradient for each parameter using central differences
    for i in range(n_params):
        # Forward point
        theta_plus = best_fit.copy()
        theta_plus[i] += delta[i]
        f_plus = func(theta_plus, *args)

        # Backward point
        theta_minus = best_fit.copy()
        theta_minus[i] -= delta[i]
        f_minus = func(theta_minus, *args)

        # Central difference
        if n_output == 1:
            jacobian[0, i] = (f_plus - f_minus) / (2 * delta[i])
        else:
            jacobian[:, i] = (np.asarray(f_plus) - np.asarray(f_minus)) / (2 * delta[i])

    return jacobian


def _legacy_cal_parameter_bias(func, target, initial_value, cov_matrix, delta=None,
                       tol=1e-3, max_iter=10, return_details=False, args=(),
                       adaptive_delta=False):
    """
    Compute the parameter bias induced by a biased (distorted) statistics.

    Suppose the observed statistics is a fixed target vector `target` y
    (e.g. a data vector distorted by systematic effects), while the reference
    (unbiased) parameter point is theta_0 = `initial_value`. The parameter
    bias is the shift theta_final - theta_0 that would be obtained by fitting
    y, i.e. by minimizing

        chi^2(theta) = (y - mu(theta))^T C^{-1} (y - mu(theta)),

    where mu = func(theta, *args) is the model prediction.

    The first-order correction, evaluated at theta_0, is

        delta_theta = F^{-1} b,
        F = J^T C^{-1} J,   b = J^T C^{-1} (y - mu(theta_0)),

    where J = dmu/dtheta is the Jacobian at theta_0. When the bias is large
    or the model is strongly nonlinear around theta_0, the first-order
    correction may be insufficient. This function therefore supports an
    iterative (Gauss-Newton) refinement:

        J_k = cal_jacobian(func, theta_k),
        F_k = J_k^T C^{-1} J_k,
        delta_theta_k = F_k^{-1} J_k^T C^{-1} (y - mu(theta_k)),
        theta_{k+1} = theta_k + delta_theta_k,

    iterated until |delta_theta_k, i| < tol * sigma_i for all parameters,
    where sigma_i = sqrt((F_k^{-1})_ii) is the 1-sigma uncertainty. The
    first iteration reproduces the purely first-order correction, so
    ``max_iter=1`` gives the linearized result only. This is an actual
    "shooting" procedure: the iteration finds the parameter point whose
    model prediction hits the target within tolerance.

    Parameters
    ----------
    func : callable
        Model function whose first argument is the parameter vector
        (list, tuple or ndarray, same length as `initial_value`), followed by
        any additional arguments collected in `args`, in the same way as
        :func:`cal_jacobian` and :func:`cal_Fisher_matrix`.
    target : array_like, shape (n_output,)
        Biased statistics (data vector) y to hit. Must have the same shape
        as the model output.
    initial_value : array_like
        Reference (unbiased) parameter values theta_0. The returned bias is
        the shift away from this point, and the iteration starts here.
    cov_matrix : ndarray, shape (n_output, n_output)
        Covariance matrix of the model outputs.
    delta : float or array_like, optional
        Finite difference step size, passed to :func:`cal_jacobian`. Either
        a scalar applied to all parameters, or a sequence matching
        ``len(initial_value)`` giving one step per parameter.
        With ``adaptive_delta=True`` the first iteration uses the automatic step
        (``delta=None``) or the user ``delta`` as-is, and later iterations
        shrink it down (never up), bounded below by 1% of the first step.
    adaptive_delta : bool, default False
        Enable feedback adjustment of the finite-difference steps based on
        the size of the previous Gauss-Newton update: each parameter's step
        becomes ``clip(10 * |update|, 0.01 * delta_first, delta_first)``.
        Near convergence the update is small while the fallback step
        (proportional to max(|theta|, 1)) can be large enough that the
        Jacobian truncation error dominates, making the update oscillate
        across the target; shrinking the step with the update avoids this.
        Only effective from the second iteration onward.
    tol : float, default 1e-3
        Convergence tolerance in units of the parameter 1-sigma error:
        iteration stops when |delta_theta_i| < tol * sigma_i for all i.
    max_iter : int, default 10
        Maximum number of Gauss-Newton iterations. Set to 1 to obtain the
        purely first-order correction.
    return_details : bool, default False
        If True, return a dictionary with the final parameter vector, the
        number of iterations performed, whether convergence was reached,
        and the list of per-iteration parameter updates.
    args : tuple, optional
        Extra positional arguments passed to `func` after the parameter
        vector.

    Returns
    -------
    parameter_bias : ndarray, shape (n_params,)
        Final estimate of the parameter bias delta_theta = theta_final - theta_0.
        With max_iter=1 this is the first-order correction.
    details : dict, optional
        Only returned if return_details=True. Keys:
        - 'theta_final' : ndarray - final parameter vector
        - 'n_iter' : int - number of iterations performed
        - 'converged' : bool - whether the tolerance was met
        - 'updates' : list of ndarray - parameter updates per iteration
        - 'sigmas' : list of ndarray - parameter 1-sigma uncertainties
          sqrt(diag(F^{-1})) at each iteration, used in the convergence test
        - 'sigma_updates' : list of ndarray - per-iteration |update|/sigma,
          i.e. the update magnitude in units of the parameter 1-sigma error;
          convergence requires all elements < tol

    Notes
    -----
    Equivalently, y = mu(theta_0) + delta_mu defines a statistics bias
    delta_mu in the model-output space; the first-iteration result then
    reduces to the standard result that a bias delta_mu propagates into a
    parameter bias F^{-1} J^T C^{-1} delta_mu at the fiducial point. The
    iteration simply re-expands about the updated point, which is exactly
    the Gauss-Newton method for the generalized least-squares problem above.
    """
    theta = np.atleast_1d(np.asarray(initial_value, dtype=np.float64))
    y = np.atleast_1d(np.asarray(target, dtype=np.float64))

    cov_matrix = np.atleast_2d(np.asarray(cov_matrix, dtype=np.float64))
    try:
        cov_inv = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        raise ValueError("Covariance matrix is singular, cannot compute inverse")

    updates = []
    sigmas = []
    converged = False
    n_iter = 0
    theta_final = theta.copy()

    # Adaptive finite-difference steps (enabled for delta=None always, and
    # for explicit delta when adaptive_delta=True): iteration 1 uses the baseline
    # step (cal_jacobian's automatic delta_0 = eps^(1/3)*max(|theta|,1), or
    # the user-provided delta as-is); later iterations shrink with the
    # Gauss-Newton update: delta_k = clip(10*|update|, 0.01*delta_first,
    # delta_first). Near convergence the update is small while the baseline
    # step (proportional to max(|theta|,1), can be large far from theta=1)
    # is big enough that truncation error in the Jacobian dominates the
    # local curvature, causing the update to oscillate across the target.
    # The 0.01 lower bound keeps round-off noise (proportional to eps/delta)
    # from dominating. Never grows above the first-iteration step.
    adaptive_delta_enabled = adaptive_delta or delta is None
    delta_first = None
    delta_current = None
    update = None

    for iteration in range(max_iter):
        n_iter = iteration + 1

        if adaptive_delta_enabled:
            if delta_current is None:
                if delta is None:
                    delta_current = (
                        np.finfo(float).eps ** (1.0 / 3.0)
                        * np.maximum(np.abs(theta_final), 1.0)
                    )
                else:
                    delta_array = np.atleast_1d(np.asarray(delta, dtype=np.float64))
                    if len(delta_array) == 1:
                        delta_array = np.full(theta.shape, delta_array[0])
                    elif len(delta_array) != len(theta):
                        raise ValueError(
                            f"delta length {len(delta_array)} does not match "
                            f"number of parameters {len(theta)}"
                        )
                    delta_current = delta_array
                delta_first = np.copy(delta_current)
            else:
                delta_current = np.clip(
                    10.0 * np.abs(update), 0.01 * delta_first, delta_first
                )
        else:
            delta_current = delta

        jacobian = cal_jacobian(func, theta_final, delta=delta_current, args=args)
        if np.allclose(jacobian, 0.0):
            raise ValueError(
                f"Jacobian is all zero at iteration {n_iter} "
                f"(within numerical tolerance). The model output "
                f"does not respond to finite-difference steps of size "
                f"{delta!r} around theta = {theta_final.tolist()}; the "
                f"model may be returning a constant (e.g. an early-return "
                f"path in the AP conversion), or delta is too small. "
                f"Provide an explicit larger delta, or skip the bias "
                f"computation."
            )
        fisher = jacobian.T @ cov_inv @ jacobian
        try:
            fisher_inv = np.linalg.inv(fisher)
        except np.linalg.LinAlgError:
            raise ValueError(
                f"Fisher matrix is singular at iteration {n_iter}; "
                f"parameters may be degenerate"
            )

        residual = y - np.atleast_1d(np.asarray(func(theta_final, *args), dtype=np.float64))
        update = fisher_inv @ (jacobian.T @ cov_inv @ residual)
        updates.append(update)
        theta_final = theta_final + update

        sigma = np.sqrt(np.diag(fisher_inv))
        sigmas.append(sigma)
        if np.all(np.abs(update) < tol * sigma):
            converged = True
            break

    parameter_bias = theta_final - theta
    if return_details:
        sigma_updates = [
            np.abs(update) / sigma
            for update, sigma in zip(updates, sigmas)
        ]
        return parameter_bias, {
            'theta_final': theta_final,
            'n_iter': n_iter,
            'converged': converged,
            'updates': updates,
            'sigmas': sigmas,
            'sigma_updates': sigma_updates,
        }
    return parameter_bias


class _RejectedParameterPoint(ValueError):
    """Internal marker used when a validator rejects a trial point."""


def _as_parameter_vector(values, name="parameter values"):
    """Return a non-empty, finite one-dimensional parameter vector."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def _as_model_vector(value, label="model output"):
    """Normalize a scalar/vector model result and reject malformed values."""
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1:
        raise ValueError(f"{label} must be a scalar or one-dimensional vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains NaN or infinite values")
    return array


def _call_parameter_validator(parameter_validator, point):
    """Call a user validator without embedding project-specific constraints."""
    if parameter_validator is None:
        return
    accepted = parameter_validator(np.asarray(point, dtype=np.float64).copy())
    try:
        accepted = bool(accepted)
    except (TypeError, ValueError) as error:
        raise ValueError("parameter_validator must return a scalar boolean") from error
    if not accepted:
        raise _RejectedParameterPoint(
            "parameter_validator rejected point "
            f"{np.asarray(point, dtype=float).tolist()}"
        )


def _normalise_difference_steps(delta, n_params, reference):
    """Return validated central-difference steps."""
    if delta is None:
        steps = np.finfo(float).eps ** (1.0 / 3.0) * np.maximum(np.abs(reference), 1.0)
    elif np.isscalar(delta):
        steps = np.full(n_params, float(delta), dtype=np.float64)
    else:
        steps = np.asarray(delta, dtype=np.float64)
    if steps.shape != (n_params,):
        raise ValueError(
            f"delta must be one scalar or a sequence of length {n_params}; got shape {steps.shape}"
        )
    if not np.all(np.isfinite(steps)) or np.any(steps <= 0.0):
        raise ValueError("delta must contain only finite positive values")
    return steps


def _prepare_precision(cov_matrix, covariance_inverse, n_output):
    """Validate covariance/precision input and return a symmetric precision matrix."""
    if (cov_matrix is None) == (covariance_inverse is None):
        raise ValueError("provide exactly one of cov_matrix or covariance_inverse")
    matrix = np.asarray(
        cov_matrix if covariance_inverse is None else covariance_inverse,
        dtype=np.float64,
    )
    if matrix.ndim == 0:
        matrix = matrix.reshape(1, 1)
    if matrix.ndim != 2 or matrix.shape != (n_output, n_output):
        label = "cov_matrix" if covariance_inverse is None else "covariance_inverse"
        raise ValueError(
            f"{label} must be a square ({n_output}, {n_output}) matrix; got {matrix.shape}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("covariance/precision matrix contains NaN or infinite values")
    if not np.allclose(matrix, matrix.T, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("covariance/precision matrix must be symmetric")
    eigenvalues = np.linalg.eigvalsh(matrix)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(np.min(eigenvalues)) <= np.finfo(float).eps * scale:
        label = "covariance" if covariance_inverse is None else "precision"
        raise ValueError(f"{label} matrix must be positive definite")
    if covariance_inverse is None:
        try:
            matrix = np.linalg.solve(matrix, np.eye(n_output, dtype=np.float64))
        except np.linalg.LinAlgError as error:
            raise ValueError("Covariance matrix is singular, cannot compute inverse") from error
    return 0.5 * (matrix + matrix.T)


def _fisher_diagnostics(jacobian, precision, iteration=None):
    """Validate a Jacobian/Fisher matrix and return its inverse and sigmas."""
    jacobian = np.asarray(jacobian, dtype=np.float64)
    if jacobian.ndim != 2 or not np.all(np.isfinite(jacobian)):
        raise ValueError("Jacobian must be a finite two-dimensional matrix")
    column_norms = np.linalg.norm(jacobian, axis=0)
    if np.any(column_norms == 0.0):
        suffix = "" if iteration is None else f" at iteration {iteration}"
        raise ValueError(f"Jacobian contains an exactly zero parameter column{suffix}")
    raw_fisher = jacobian.T @ precision @ jacobian
    fisher = 0.5 * (raw_fisher + raw_fisher.T)
    if not np.all(np.isfinite(fisher)):
        raise ValueError("Fisher matrix contains NaN or infinite values")
    eigenvalues = np.linalg.eigvalsh(fisher)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(np.min(eigenvalues)) <= np.finfo(float).eps * scale:
        suffix = "" if iteration is None else f" at iteration {iteration}"
        raise ValueError(f"Fisher matrix is singular or not positive definite{suffix}")
    condition_number = float(np.max(eigenvalues) / np.min(eigenvalues))
    if condition_number > 1.0e12:
        suffix = "" if iteration is None else f" at iteration {iteration}"
        raise ValueError(
            f"Fisher matrix is too ill-conditioned{suffix} "
            f"(condition number {condition_number:.3e})"
        )
    try:
        covariance = np.linalg.solve(fisher, np.eye(fisher.shape[0], dtype=np.float64))
    except np.linalg.LinAlgError as error:
        raise ValueError("Cannot invert Fisher matrix") from error
    covariance = 0.5 * (covariance + covariance.T)
    sigmas = np.sqrt(np.diag(covariance))
    if not np.all(np.isfinite(sigmas)) or np.any(sigmas <= 0.0):
        raise ValueError("Fisher parameter uncertainties are not finite and positive")
    return fisher, covariance, sigmas


def _evaluate_model(func, point, args, expected_size=None):
    """Evaluate a model and enforce a stable output dimension."""
    result = _as_model_vector(func(point, *args))
    if expected_size is not None and result.size != expected_size:
        raise ValueError(
            f"model output has length {result.size}, expected {expected_size}"
        )
    return result


def cal_jacobian(func, best_fit, delta=None, args=(), parameter_validator=None):
    """Compute a checked central finite-difference Jacobian."""
    point = _as_parameter_vector(best_fit, "best_fit")
    _call_parameter_validator(parameter_validator, point)
    steps = _normalise_difference_steps(delta, point.size, point)
    f0 = _evaluate_model(func, point, args)
    jacobian = np.empty((f0.size, point.size), dtype=np.float64)
    for index, step in enumerate(steps):
        plus = point.copy()
        minus = point.copy()
        plus[index] += step
        minus[index] -= step
        _call_parameter_validator(parameter_validator, plus)
        _call_parameter_validator(parameter_validator, minus)
        f_plus = _evaluate_model(func, plus, args, f0.size)
        f_minus = _evaluate_model(func, minus, args, f0.size)
        jacobian[:, index] = (f_plus - f_minus) / (2.0 * step)
    if not np.all(np.isfinite(jacobian)):
        raise ValueError("Jacobian contains NaN or infinite values")
    return jacobian


def _jacobian_with_stability(func, point, steps, args, parameter_validator, stability_tolerance):
    """Compute a Jacobian and optionally compare it with a half-step estimate."""
    jacobian = cal_jacobian(
        func, point, delta=steps, args=args, parameter_validator=parameter_validator
    )
    if stability_tolerance is None:
        return jacobian, np.full(steps.shape, np.nan, dtype=np.float64)
    if not np.isfinite(stability_tolerance) or stability_tolerance <= 0.0:
        raise ValueError("stability_tolerance must be positive or None")
    half_jacobian = cal_jacobian(
        func,
        point,
        delta=steps / 2.0,
        args=args,
        parameter_validator=parameter_validator,
    )
    denominator = np.maximum(np.linalg.norm(half_jacobian, axis=0), np.finfo(float).tiny)
    relative_change = np.linalg.norm(jacobian - half_jacobian, axis=0) / denominator
    if not np.all(np.isfinite(relative_change)):
        raise ValueError("Finite-difference stability diagnostic is non-finite")
    if np.any(relative_change > stability_tolerance):
        raise ValueError(
            "Finite-difference Jacobian is unstable when the step is halved: "
            f"relative changes {relative_change.tolist()} exceed "
            f"stability_tolerance={stability_tolerance:g}"
        )
    return jacobian, relative_change


def _chi_square(residual, precision):
    """Evaluate a non-negative quadratic form with round-off tolerance."""
    residual = np.asarray(residual, dtype=np.float64)
    value = float(residual @ precision @ residual)
    if not np.isfinite(value):
        raise ValueError("chi-square is NaN or infinite")
    if value < -1.0e-8:
        raise ValueError(f"chi-square is negative: {value}")
    return max(value, 0.0)


def _joint_sigma(displacement, fisher):
    """Return a displacement length in the local Fisher metric."""
    displacement = np.asarray(displacement, dtype=np.float64)
    value = float(displacement @ fisher @ displacement)
    if not np.isfinite(value):
        raise ValueError("Fisher-metric displacement is non-finite")
    return float(np.sqrt(max(value, 0.0)))


def cal_parameter_bias(
    func,
    target,
    initial_value,
    cov_matrix=None,
    delta=None,
    tol=1e-3,
    max_iter=10,
    return_details=False,
    args=(),
    adaptive_delta=False,
    covariance_inverse=None,
    parameter_validator=None,
    step_control="none",
    stability_tolerance=None,
):
    """Estimate a parameter bias with first-order or iterative Fisher updates.

    ``max_iter=1`` returns the complete first-order Fisher update.  For more
    iterations, ``step_control='none'`` accepts full Gauss--Newton updates,
    while ``step_control='backtracking'`` selects the first validator-valid
    candidate that does not increase exact chi-square.  A validator is an
    optional callback supplied by the calling project for physical domains.
    """
    if not isinstance(max_iter, (int, np.integer)) or isinstance(max_iter, bool) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be positive and finite")
    if step_control not in ("none", "backtracking"):
        raise ValueError("step_control must be 'none' or 'backtracking'")
    theta0 = _as_parameter_vector(initial_value, "initial_value")
    target_vector = _as_model_vector(target, "target")
    _call_parameter_validator(parameter_validator, theta0)
    model0 = _evaluate_model(func, theta0, args)
    if model0.shape != target_vector.shape:
        raise ValueError(
            f"target shape {target_vector.shape} does not match model output shape {model0.shape}"
        )
    precision = _prepare_precision(cov_matrix, covariance_inverse, target_vector.size)
    base_steps = _normalise_difference_steps(delta, theta0.size, theta0)
    adaptive = bool(adaptive_delta or delta is None)

    current = theta0.copy()
    current_model = model0
    current_chi2 = _chi_square(target_vector - current_model, precision)
    initial_chi2 = current_chi2
    previous_update = None
    updates = []
    proposed_updates = []
    sigmas = []
    sigma_updates = []
    proposed_sigma_updates = []
    joint_sigma_updates = []
    proposed_joint_sigma_updates = []
    fisher_matrices = []
    parameter_covariances = []
    steps_used = []
    derivative_changes = []
    current_chi2s = []
    candidate_chi2s = []
    candidate_points = []
    line_search_factors = []
    last_unaccepted_update = None
    last_unaccepted_candidate = None
    last_unaccepted_candidate_chi2 = np.nan
    converged = False
    termination_reason = "max_iter_reached"
    n_iter = 0

    for iteration in range(int(max_iter)):
        n_iter = iteration + 1
        if adaptive and previous_update is not None:
            steps = np.clip(10.0 * np.abs(previous_update), 0.01 * base_steps, base_steps)
        else:
            steps = base_steps.copy()
        steps_used.append(steps.copy())
        jacobian, relative_change = _jacobian_with_stability(
            func, current, steps, args, parameter_validator, stability_tolerance
        )
        derivative_changes.append(relative_change.copy())
        fisher, parameter_covariance, sigma = _fisher_diagnostics(jacobian, precision, n_iter)
        fisher_matrices.append(fisher.copy())
        parameter_covariances.append(parameter_covariance.copy())
        sigmas.append(sigma.copy())
        residual = target_vector - current_model
        proposed_update = np.linalg.solve(fisher, jacobian.T @ precision @ residual)
        if not np.all(np.isfinite(proposed_update)):
            raise ValueError(f"Fisher update contains NaN or infinite values at iteration {n_iter}")
        proposed_update = np.asarray(proposed_update, dtype=np.float64)
        proposed_updates.append(proposed_update.copy())
        proposed_sigma_updates.append(np.abs(proposed_update) / sigma)
        proposed_joint_sigma_updates.append(_joint_sigma(proposed_update, fisher))
        current_chi2s.append(float(current_chi2))

        accepted_update = None
        accepted_point = None
        accepted_model = None
        accepted_chi2 = np.nan
        accepted_factor = np.nan
        full_point = current + proposed_update

        if int(max_iter) == 1 or step_control == "none":
            _call_parameter_validator(parameter_validator, full_point)
            full_model = _evaluate_model(func, full_point, args, target_vector.size)
            accepted_update = proposed_update
            accepted_point = full_point
            accepted_model = full_model
            accepted_chi2 = _chi_square(target_vector - full_model, precision)
            accepted_factor = 1.0
        else:
            for factor in 0.5 ** np.arange(0, 21, dtype=np.float64):
                trial_point = current + factor * proposed_update
                try:
                    _call_parameter_validator(parameter_validator, trial_point)
                except _RejectedParameterPoint:
                    continue
                trial_model = _evaluate_model(func, trial_point, args, target_vector.size)
                trial_chi2 = _chi_square(target_vector - trial_model, precision)
                if trial_chi2 <= current_chi2:
                    accepted_update = factor * proposed_update
                    accepted_point = trial_point
                    accepted_model = trial_model
                    accepted_chi2 = trial_chi2
                    accepted_factor = float(factor)
                    break
            if accepted_update is None:
                last_unaccepted_update = proposed_update.copy()
                last_unaccepted_candidate = full_point.copy()
                try:
                    _call_parameter_validator(parameter_validator, full_point)
                    full_model = _evaluate_model(func, full_point, args, target_vector.size)
                    last_unaccepted_candidate_chi2 = _chi_square(
                        target_vector - full_model, precision
                    )
                except _RejectedParameterPoint:
                    last_unaccepted_candidate_chi2 = np.nan
                candidate_chi2s.append(float(last_unaccepted_candidate_chi2))
                candidate_points.append(full_point.copy())
                termination_reason = "no_acceptable_step"
                break

        if accepted_update is None or accepted_point is None or accepted_model is None:
            raise RuntimeError("Internal Fisher update bookkeeping error")
        updates.append(accepted_update.copy())
        sigma_updates.append(np.abs(accepted_update) / sigma)
        joint_sigma_updates.append(_joint_sigma(accepted_update, fisher))
        line_search_factors.append(float(accepted_factor))
        candidate_chi2s.append(float(accepted_chi2))
        candidate_points.append(accepted_point.copy())
        current = accepted_point
        current_model = accepted_model
        current_chi2 = float(accepted_chi2)
        previous_update = accepted_update.copy()
        if np.all(np.abs(accepted_update) <= tol * sigma):
            converged = True
            termination_reason = "update_below_tolerance"
            break

    if updates:
        final_steps = (
            np.clip(10.0 * np.abs(previous_update), 0.01 * base_steps, base_steps)
            if adaptive and previous_update is not None
            else base_steps.copy()
        )
        try:
            final_jacobian, final_relative_change = _jacobian_with_stability(
                func, current, final_steps, args, parameter_validator, stability_tolerance
            )
            final_fisher, final_covariance, final_sigmas = _fisher_diagnostics(
                final_jacobian, precision, "final"
            )
        except ValueError:
            final_fisher = fisher_matrices[-1]
            final_covariance = parameter_covariances[-1]
            final_sigmas = sigmas[-1]
            final_relative_change = derivative_changes[-1]
    else:
        final_fisher = fisher_matrices[-1]
        final_covariance = parameter_covariances[-1]
        final_sigmas = sigmas[-1]
        final_relative_change = derivative_changes[-1]

    parameter_bias = current - theta0
    if not return_details:
        return parameter_bias
    return parameter_bias, {
        "theta_final": current.copy(),
        "n_iter": n_iter,
        "iterations": len(updates),
        "converged": converged,
        "termination_reason": termination_reason,
        "initial_chi2": float(initial_chi2),
        "chi2": float(current_chi2),
        "current_chi2s": current_chi2s,
        "candidate_chi2s": candidate_chi2s,
        "candidate_points": candidate_points,
        "updates": updates,
        "proposed_updates": proposed_updates,
        "sigmas": sigmas,
        "parameter_sigmas": final_sigmas.copy(),
        "sigma_updates": sigma_updates,
        "proposed_sigma_updates": proposed_sigma_updates,
        "joint_sigma_updates": joint_sigma_updates,
        "proposed_joint_sigma_updates": proposed_joint_sigma_updates,
        "line_search_factors": line_search_factors,
        "finite_difference_steps": steps_used,
        "derivative_relative_changes": derivative_changes,
        "final_derivative_relative_changes": final_relative_change,
        "fisher_matrix": final_fisher,
        "fisher_matrices": fisher_matrices,
        "parameter_covariance": final_covariance,
        "parameter_covariances": parameter_covariances,
        "parameter_bias_in_sigma": parameter_bias / final_sigmas,
        "parameter_bias_joint_sigma": _joint_sigma(parameter_bias, final_fisher),
        "last_unaccepted_update": last_unaccepted_update,
        "last_unaccepted_candidate": last_unaccepted_candidate,
        "last_unaccepted_candidate_chi2": last_unaccepted_candidate_chi2,
    }


def cal_Fisher_matrix(func, best_fit, cov_matrix=None, delta=None, computed_jac=None, return_jac=False,
                      args=(), covariance_inverse=None):
    """
    Calculate Fisher matrix from covariance matrix and model function.

    The Fisher information matrix is computed from the Jacobian of the
    model function with respect to parameters:

    F = (∂μ/∂θ)^T * C^{-1} * (∂μ/∂θ)

    where μ = func(θ, *args) is the model prediction, C is the covariance
    matrix, and ∂μ/∂θ is the Jacobian matrix.

    Parameters
    ----------
    func : callable
        Model function whose first argument is the parameter vector
        (list, tuple or ndarray, same length as `best_fit`), followed by
        any additional arguments collected in `args`.
        Example: for 2 parameters, func(params, a) with params = [x1, x2].
    best_fit : array_like
        Best-fit parameter values.
    cov_matrix : ndarray
        Covariance matrix of the model outputs.
    delta : float or array_like, optional
        Finite difference step size, passed to :func:`cal_jacobian`.
        See :func:`cal_jacobian` for details.
    computed_jac : ndarray, optional
        Precomputed Jacobian matrix of shape (n_output, n_params).
        If given, `func` is not evaluated and `delta` and `args` are ignored.
    args : tuple, optional
        Extra positional arguments passed to `func` after the parameter
        vector, in the same way as `scipy.integrate.quad`.

    Returns
    -------
    ndarray
        Fisher information matrix with shape (n_params, n_params).
    """
    if computed_jac is None:
        jacobian = cal_jacobian(func, best_fit, delta=delta, args=args)
    else:
        jacobian = np.asarray(computed_jac, dtype=np.float64)
    if jacobian.ndim != 2 or not np.all(np.isfinite(jacobian)):
        raise ValueError("computed_jac must be a finite two-dimensional matrix")
    precision = _prepare_precision(cov_matrix, covariance_inverse, jacobian.shape[0])
    fisher, _, _ = _fisher_diagnostics(jacobian, precision)
    if return_jac:
        return fisher, jacobian
    return fisher


def cal_Fisher_matrix_from_precomputed(parameter_points, function_values, cov_matrix, return_jac=False,
                                       best_fit=None, delta=None):
    """
    Calculate Fisher matrix from precomputed parameter-function value pairs.

    This function is useful when function evaluations are expensive and have been
    precomputed at ±delta perturbations for each parameter. It uses the structured
    format where the meaning of each point is explicit from its position in the array.

    The Fisher information matrix is computed as:
    F = (∂μ/∂θ)^T * C^{-1} * (∂μ/∂θ)

    Parameters
    ----------
    parameter_points : ndarray, shape (n_params, 2, n_params)
        Parameter values at which the model function was evaluated.
        parameter_points[i, 0] = parameter vector with parameter i perturbed upward
        parameter_points[i, 1] = parameter vector with parameter i perturbed downward
    
    function_values : ndarray, shape (n_params, 2, n_output)
        Model function values at the corresponding parameter_points.
        function_values[i, 0] = f(parameter_points[i, 0])
        function_values[i, 1] = f(parameter_points[i, 1])
    
    cov_matrix : ndarray, shape (n_output, n_output)
        Covariance matrix of the model outputs.
    
    return_jac : bool, default False
        If True, return both Fisher matrix and Jacobian matrix.

    best_fit : array_like, optional
        Best-fit parameter values. Must be provided together with delta.
        When provided, delta is computed as:
            delta_i = parameter_points[i, 0, i] - best_fit[i]
        This is useful when perturbation points are not symmetric around the
        center, or when you want to explicitly specify the center point.
    
    delta : float or array_like, optional
        Finite difference step sizes. Must be provided together with best_fit.
        When provided together with best_fit, delta is used directly.
        Can be a single float applied to all parameters, or an array matching
        the number of parameters.

    Returns
    -------
    fisher : ndarray, shape (n_params, n_params)
        Fisher information matrix.
    
    jacobian : ndarray, shape (n_output, n_params), optional
        Jacobian matrix. Only returned if return_jac=True.

    Examples
    --------
    >>> best_fit = [1.0, 2.0]
    >>> delta = [0.01, 0.02]
    >>> 
    >>> # Build structured arrays: (n_params, 2, ...)
    >>> # For each parameter i: [plus_point, minus_point]
    >>> n_params = len(best_fit)
    >>> parameter_points = np.zeros((n_params, 2, n_params))
    >>> for i in range(n_params):
    ...     parameter_points[i, 0] = best_fit.copy()
    ...     parameter_points[i, 0, i] += delta[i]  # plus point
    ...     parameter_points[i, 1] = best_fit.copy()
    ...     parameter_points[i, 1, i] -= delta[i]  # minus point
    >>> 
    >>> # Evaluate model at all points
    >>> function_values = np.array([
    ...     [expensive_model(*parameter_points[i, 0]),
    ...      expensive_model(*parameter_points[i, 1])]
    ...     for i in range(n_params)
    ... ])
    >>> 
    >>> # Compute Fisher matrix (delta inferred from parameter_points)
    >>> fisher = cal_Fisher_matrix_from_precomputed(
    ...     parameter_points, function_values, cov_matrix
    ... )
    >>> 
    >>> # Compute Fisher matrix (delta computed from best_fit)
    >>> fisher = cal_Fisher_matrix_from_precomputed(
    ...     parameter_points, function_values, cov_matrix,
    ...     best_fit=[1.0, 2.0]
    ... )

    Notes
    -----
    The function computes derivatives using central differences:
        ∂f/∂θ_i ≈ [f(θ + δ_i e_i) - f(θ - δ_i e_i)] / (2 δ_i)
    
    Two modes of operation:
    1. If both `best_fit` and `delta` are provided, delta is used directly.
    2. If neither is provided, delta is inferred from the perturbation points:
           delta_i = (parameter_points[i, 0, i] - parameter_points[i, 1, i]) / 2
    
    Providing only one of `best_fit` or `delta` will raise a ValueError.
    """
    # Convert inputs to numpy arrays
    parameter_points = np.asarray(parameter_points, dtype=np.float64)
    function_values = np.asarray(function_values, dtype=np.float64)
    
    # Validate shapes
    if parameter_points.ndim != 3:
        raise ValueError(
            f"parameter_points must be 3D with shape (n_params, 2, n_params), "
            f"got {parameter_points.ndim}D"
        )
    n_params = parameter_points.shape[0]
    if parameter_points.shape[1] != 2:
        raise ValueError(
            f"parameter_points.shape[1] must be 2 (for plus/minus), "
            f"got {parameter_points.shape[1]}"
        )
    if parameter_points.shape[2] != n_params:
        raise ValueError(
            f"parameter_points.shape[2] must equal n_params ({n_params}), "
            f"got {parameter_points.shape[2]}"
        )
    
    if function_values.shape[:2] != (n_params, 2):
        raise ValueError(
            f"function_values must have shape (n_params, 2, n_output), "
            f"got {function_values.shape}"
        )
    n_output = function_values.shape[2] if function_values.ndim == 3 else 1
    if function_values.ndim == 2:
        function_values = function_values.reshape(n_params, 2, 1)
    
    # Compute delta for each parameter
    # Two modes: (1) both best_fit and delta provided, (2) neither provided
    if (best_fit is None) != (delta is None):
        raise ValueError(
            "best_fit and delta must be both provided or both omitted. "
            "Got best_fit={} and delta={}.".format(
                "provided" if best_fit is not None else "None",
                "provided" if delta is not None else "None"
            )
        )
    
    if best_fit is not None and delta is not None:
        # Both provided: use delta directly, validate best_fit length
        best_fit = np.atleast_1d(np.asarray(best_fit, dtype=np.float64))
        if len(best_fit) != n_params:
            raise ValueError(
                f"best_fit length {len(best_fit)} does not match "
                f"number of parameters {n_params}"
            )
        if np.isscalar(delta):
            delta_arr = np.full(n_params, float(delta), dtype=np.float64)
        else:
            delta_arr = np.atleast_1d(np.asarray(delta, dtype=np.float64))
            if len(delta_arr) != n_params:
                raise ValueError(
                    f"delta length {len(delta_arr)} does not match "
                    f"number of parameters {n_params}"
                )
    else:
        # Neither provided: infer delta from perturbation points
        # delta_i = (parameter_points[i, 0, i] - parameter_points[i, 1, i]) / 2
        delta_arr = np.zeros(n_params, dtype=np.float64)
        for i in range(n_params):
            delta_arr[i] = (parameter_points[i, 0, i] - parameter_points[i, 1, i]) / 2.0

    # Validate delta values
    for i in range(n_params):
        if np.abs(delta_arr[i]) < 1e-15:
            raise ValueError(
                f"delta for parameter {i} is too small: {delta_arr[i]}. "
                f"Check that parameter_points[{i}, 0] and parameter_points[{i}, 1] "
                f"differ in the {i}-th component, or provide best_fit/delta explicitly."
            )
    
    # Compute inverse covariance matrix
    cov_matrix = np.atleast_2d(cov_matrix)
    if cov_matrix.shape != (n_output, n_output):
        raise ValueError(f"cov_matrix shape {cov_matrix.shape} does not match "
                        f"model output dimension {n_output}")
    try:
        cov_inv = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        raise ValueError("Covariance matrix is singular, cannot compute inverse")
    
    # Compute Jacobian from structured arrays using central differences
    # jacobian[:, i] = (function_values[i, 0] - function_values[i, 1]) / (2 * delta[i])
    jacobian = np.zeros((n_output, n_params), dtype=np.float64)
    for i in range(n_params):
        f_plus = function_values[i, 0]   # shape: (n_output,)
        f_minus = function_values[i, 1]  # shape: (n_output,)
        jacobian[:, i] = (f_plus - f_minus) / (2 * delta_arr[i])
    
    # Compute Fisher matrix: F = J^T * C^{-1} * J
    fisher = jacobian.T @ cov_inv @ jacobian
    
    if return_jac:
        return fisher, jacobian
    return fisher


def get_fisher_from_tpcf(xismu_source_dict, xismu_assis_dict, snap_ids, redshift_dict, delta, best_fit, cov_matrix, return_details=False, **kwargs):
    """
    Compute Fisher matrix from two-point correlation function with AP parameters.

    Evaluates the model at ±delta perturbations for each parameter and computes
    the Fisher information matrix. Returns both the standardized data and the
    Fisher matrix.

    Returns
    -------
    fisher : ndarray
        Fisher information matrix.
    parameter_points : ndarray, shape (n_params, 2, n_params)
        Standardized parameter points.
    function_values : ndarray, shape (n_params, 2, n_output)
        Function values at the standardized points.
    """
    from .tpcf import get_diff_array
    best_fit = np.atleast_1d(np.asarray(best_fit, dtype=np.float64))
    n_params = len(best_fit)

    if n_params != 2 and n_params != 3:
        raise ValueError(
            f"Number of parameters must be 2 or 3, got {n_params}"
        )

    # Process delta
    if np.isscalar(delta):
        delta = np.full(n_params, float(delta))
    else:
        delta = np.atleast_1d(np.asarray(delta, dtype=np.float64))

    # Build structured arrays: (n_params, 2, ...)
    # For each parameter i: [plus_point, minus_point]
    parameter_points = np.zeros((n_params, 2, n_params), dtype=np.float64)
    function_values_list = []

    for i in range(n_params):
        # Plus point: best_fit + delta[i] * e_i
        plus_point = best_fit.copy()
        plus_point[i] += delta[i]
        parameter_points[i, 0] = plus_point

        # Minus point: best_fit - delta[i] * e_i
        minus_point = best_fit.copy()
        minus_point[i] -= delta[i]
        parameter_points[i, 1] = minus_point

        # Evaluate model at plus point
        if n_params == 2:
            omega_mm, w_m = plus_point
            wa_m = 0.0
        else:
            omega_mm, w_m, wa_m = plus_point
        xismu_dict_temp = {}
        for snap_id in snap_ids:
            xismu_temp = tpcf_convert_main(
                xismu_source_dict[snap_id], best_fit[0], best_fit[1],
                omega_mm, w_m, redshift_dict[snap_id],
                assis_xismu=xismu_assis_dict[snap_id],
                wa_f=0.0, wa_m=wa_m
            )
            xismu_dict_temp[snap_id] = xismu_temp
        f_plus = get_diff_array(xismu_dict_temp, snap_ids, **kwargs)
        function_values_list.append(f_plus)

        # Evaluate model at minus point
        if n_params == 2:
            omega_mm, w_m = minus_point
            wa_m = 0.0
        else:
            omega_mm, w_m, w_am = minus_point
        xismu_dict_temp = {}
        for snap_id in snap_ids:
            xismu_temp = tpcf_convert_main(
                xismu_source_dict[snap_id], best_fit[0], best_fit[1],
                omega_mm, w_m, redshift_dict[snap_id],
                assis_xismu=xismu_assis_dict[snap_id],
                wa_f=0.0, wa_m=wa_m
            )
            xismu_dict_temp[snap_id] = xismu_temp
        f_minus = get_diff_array(xismu_dict_temp, snap_ids, **kwargs)
        function_values_list.append(f_minus)

    # Reshape function_values to (n_params, 2, n_output)
    function_values = np.array(function_values_list).reshape(n_params, 2, -1)

    fisher = cal_Fisher_matrix_from_precomputed(
        parameter_points, function_values, cov_matrix
    )

    if return_details:
        return fisher, parameter_points, function_values
    else:
        return fisher



def _compute_ellipse_params_from_fisher(fisher, confidence_level=0.683):
    """
    Compute ellipse parameters from Fisher matrix (internal helper function).

    Parameters
    ----------
    fisher : ndarray
        Fisher information matrix, must be 2x2.
    confidence_level : float, default 0.683
        Confidence level for the ellipse (0 < confidence_level < 1).

    Returns
    -------
    dict
        Dictionary containing:
        - 'semi_minor' : float - semi-minor axis length (smaller)
        - 'semi_major' : float - semi-major axis length (larger)
        - 'angle_rad' : float - rotation angle in radians
        - 'angle_deg' : float - rotation angle in degrees
        - 'eigenvals' : ndarray - eigenvalues [λ_small, λ_large]
        - 'eigenvecs' : ndarray - eigenvectors as columns
        - 'delta_chi2' : float - chi-squared critical value

    Raises
    ------
    ValueError
        If fisher is not a 2x2 matrix or not positive definite.
    """
    # Validate fisher matrix dimensions
    fisher = np.atleast_2d(fisher)
    if fisher.shape != (2, 2):
        raise ValueError(f"Fisher matrix must be 2x2 for ellipse computation, "
                        f"got shape {fisher.shape}")

    # Validate confidence level
    if not (0 < confidence_level < 1):
        raise ValueError(f"confidence_level must be between 0 and 1, got {confidence_level}")

    # Chi-squared critical value for 2 degrees of freedom
    delta_chi2 = chi2.ppf(confidence_level, df=2)

    # Eigen-decomposition of Fisher matrix
    # Fisher matrix F is the inverse of parameter covariance matrix C: F = C^{-1}
    # The error ellipse satisfies: (θ - θ₀)^T · F · (θ - θ₀) = Δχ²
    #
    # Eigen-decomposition: F = Q · Λ · Q^T
    #   where Λ = diag(λ₁, λ₂) with λ₁ ≥ λ₂ > 0 (eigenvalues)
    #   Q is orthogonal matrix (rotation) from eigenvectors
    #
    # In eigenvector coordinates: λ₁ u₁² + λ₂ u₂² = Δχ²
    #   → u₁²/(Δχ²/λ₁) + u₂²/(Δχ²/λ₂) = 1
    #   → semi-axis lengths: a = sqrt(Δχ²/λ₁), b = sqrt(Δχ²/λ₂)
    #
    # Note: λ₁ is the larger eigenvalue → a is the semi-minor (smaller error)
    #       λ₂ is the smaller eigenvalue → b is the semi-major (larger error)

    eigenvals, eigenvecs = np.linalg.eigh(fisher)
    # eigh returns sorted ascending: λ₂ ≤ λ₁
    lambda_small, lambda_large = eigenvals  # λ₂ (small), λ₁ (large)
    # eigenvectors are columns: v₁ (for λ₁), v₂ (for λ₂)

    # Check for positive definiteness (both eigenvalues > 0)
    if lambda_small <= 0:
        raise ValueError(f"Fisher matrix is not positive definite. "
                        f"Eigenvalues: {eigenvals}. "
                        f"The Fisher matrix must be positive definite for ellipse computation. "
                        f"This may indicate parameter degeneracy or a poorly constrained model.")

    # Semi-axis lengths (before rotation)
    # a = sqrt(Δχ² / λ_large)  (semi-minor, smaller)
    # b = sqrt(Δχ² / λ_small)  (semi-major, larger)
    semi_minor = np.sqrt(delta_chi2 / lambda_large)
    semi_major = np.sqrt(delta_chi2 / lambda_small)

    # Angle of the ellipse (rotation from eigenvector of larger eigenvalue)
    # The eigenvector corresponding to the larger eigenvalue (λ₁) gives
    # the direction of the semi-minor axis (smaller uncertainty).
    v_semi_minor = eigenvecs[:, 1]  # eigenvector for λ₁ (larger eigenvalue) -> semi-minor axis
    angle_rad = np.arctan2(v_semi_minor[1], v_semi_minor[0])
    angle_deg = np.degrees(angle_rad)

    return {
        'semi_minor': semi_minor,
        'semi_major': semi_major,
        'angle_rad': angle_rad,
        'angle_deg': angle_deg,
        'eigenvals': eigenvals,
        'eigenvecs': eigenvecs,
        'delta_chi2': delta_chi2,
        'v_semi_minor': v_semi_minor,  # eigenvector for λ₁ (larger eigenvalue)
        'v_semi_major': eigenvecs[:, 0]  # eigenvector for λ₂ (smaller eigenvalue)
    }


def cal_ellipse_from_fisher(fisher, confidence_level=0.683, full_output=False):
    """
    Calculate ellipse area and parameters from Fisher matrix.

    Parameters
    ----------
    fisher : ndarray
        Fisher information matrix, must be 2x2.
    confidence_level : float, default 0.683
        Confidence level for the ellipse (0 < confidence_level < 1).
        For 1\\sigma Gaussian: 0.683, for 2\\sigma: 0.954, for 3\\sigma: 0.997.
    full_output : bool, default False
        If True, return additional ellipse parameters.

    Returns
    -------
    area : float
        Area of the error ellipse (π * a * b).
    params : dict, optional
        Only returned if full_output=True. Dictionary containing:
        - 'semi_minor' : float - semi-minor axis length (smaller)
        - 'semi_major' : float - semi-major axis length (larger)
        - 'minor_axis_slope' : float - slope of the semi-minor axis
        - 'major_axis_slope' : float - slope of the semi-major axis
        - 'angle_rad' : float - rotation angle in radians
        - 'angle_deg' : float - rotation angle in degrees
        - 'eigenvals' : ndarray - eigenvalues [λ_small, λ_large]
        - 'eigenvecs' : ndarray - eigenvectors as columns
        - 'delta_chi2' : float - chi-squared critical value

    Raises
    ------
    ValueError
        If fisher is not a 2x2 matrix or not positive definite.

    Notes
    -----
    The ellipse area is calculated as: Area = π * a * b
    where a is the semi-minor axis and b is the semi-major axis.

    The axis slopes are calculated from the eigenvectors of the Fisher matrix:
    - The semi-minor axis aligns with the eigenvector of the larger eigenvalue
    - The semi-major axis aligns with the eigenvector of the smaller eigenvalue
    """
    # Compute ellipse parameters using helper function
    params = _compute_ellipse_params_from_fisher(fisher, confidence_level)

    # Calculate area: π * a * b
    area = np.pi * params['semi_minor'] * params['semi_major']

    if full_output:
        # Calculate axis slopes from eigenvectors
        # eigenvectors are columns: v_semi_major (for λ_small), v_semi_minor (for λ_large)
        # semi-minor axis aligns with v_semi_minor (eigenvector for larger eigenvalue)
        # semi-major axis aligns with v_semi_major (eigenvector for smaller eigenvalue)
        v_semi_minor = params['eigenvecs'][:, 1]  # eigenvector for λ_large (semi-minor)
        v_semi_major = params['eigenvecs'][:, 0]  # eigenvector for λ_small (semi-major)

        # Slope = y/x (be careful with vertical lines where x ≈ 0)
        # For near-vertical lines, slope approaches infinity
        minor_axis_slope = v_semi_minor[1] / v_semi_minor[0] if np.abs(v_semi_minor[0]) > 1e-10 else np.inf
        major_axis_slope = v_semi_major[1] / v_semi_major[0] if np.abs(v_semi_major[0]) > 1e-10 else np.inf

        # Add slopes and vectors to params
        params['minor_axis_slope'] = minor_axis_slope
        params['major_axis_slope'] = major_axis_slope
        params['v_semi_minor'] = v_semi_minor
        params['v_semi_major'] = v_semi_major

        return area, params
    else:
        return area


def get_sigma_point_from_fisher(fisher, sigma=1.0, center=None):
    """
    Get points along degenerate and non-degenerate directions from Fisher matrix at specified sigma.

    This function computes the ellipse parameters from the Fisher matrix and returns
    the coordinates of points along the semi-minor (non-degenerate, smaller uncertainty)
    and semi-major (degenerate, larger uncertainty) directions at a given sigma confidence
    level. The sigma parameter corresponds to the number of standard deviations in a
    one-dimensional Gaussian distribution:
        - sigma=1 → 68.3% confidence
        - sigma=2 → 95.4% confidence
        - sigma=3 → 99.7% confidence

    This ensures consistency with standard statistical practice where sigma refers to
    the standard deviation of a 1D normal distribution.

    Parameters
    ----------
    fisher : ndarray
        Fisher information matrix, must be 2x2.
    sigma : float, default 1.0
        Confidence level in sigma units, corresponding to the number of standard
        deviations for a 1D Gaussian. The confidence level is computed as:
            CL = Φ(sigma) - Φ(-sigma)
        where Φ is the standard normal cumulative distribution function.
        For example:
        - sigma=1.0 → CL=0.683 (68.3%)
        - sigma=2.0 → CL=0.954 (95.4%)
        - sigma=3.0 → CL=0.997 (99.7%)
    center : array-like, optional
        Center coordinates (x0, y0) of the ellipse. If None, defaults to (0, 0).

    Returns
    -------
    dict
        Dictionary containing:
        - 'non_degenerate_points' : ndarray of shape (2, 2)
            Points along the non-degenerate direction (semi-minor axis, smaller uncertainty).
            Row 0: positive direction (+sigma), Row 1: negative direction (-sigma).
        - 'degenerate_points' : ndarray of shape (2, 2)
            Points along the degenerate direction (semi-major axis, larger uncertainty).
            Row 0: positive direction (+sigma), Row 1: negative direction (-sigma).
        - 'ellipse_params' : dict
            Ellipse parameters computed at the corresponding confidence level.

    Notes
    -----
    The Fisher matrix F is the inverse of the covariance matrix C: F = C^{-1}.
    For a 2D Gaussian, the error ellipse satisfies:
        (θ - θ₀)^T · F · (θ - θ₀) = Δχ²

    The confidence level is determined by the one-dimensional Gaussian probability:
        CL = P(|Z| < sigma) where Z ~ N(0, 1)

    This CL is then used to compute the chi-squared critical value Δχ² = χ²_{2, CL}
    for the 2-degree-of-freedom chi-squared distribution.

    In the eigenvector basis (u along semi-minor, v along semi-major):
        λ_small * u² + λ_large * v² = Δχ²

    The semi-axis lengths are:
        a (semi-minor) = sqrt(Δχ² / λ_large) = sigma * sqrt(χ²_{2,0.683} / λ_large)
        b (semi-major) = sqrt(Δχ² / λ_small) = sigma * sqrt(χ²_{2,0.683} / λ_small)

    Examples
    --------
    >>> fisher = np.array([[1.0, 0.3], [0.3, 0.5]])
    >>> result = get_sigma_point_from_fisher(fisher, sigma=1.0, center=[0.5, -0.2])
    >>> result['non_degenerate_points']  # points along the tighter constraint direction
    >>> result['degenerate_points']      # points along the more degenerate direction
    """
    # Compute eigen-decomposition
    fisher = np.atleast_2d(fisher)
    if fisher.shape != (2, 2):
        raise ValueError(f"Fisher matrix must be 2x2, got shape {fisher.shape}")

    # Convert sigma (1D normal distribution) to confidence level
    # sigma=1 -> CL=0.683, sigma=2 -> CL=0.954, sigma=3 -> CL=0.997
    confidence_level = norm.cdf(sigma) - norm.cdf(-sigma)

    # Use _compute_ellipse_params_from_fisher directly with the correct confidence level
    ellipse_params = _compute_ellipse_params_from_fisher(fisher, confidence_level=confidence_level)

    # Extract parameters (already scaled correctly)
    semi_minor = ellipse_params['semi_minor']
    semi_major = ellipse_params['semi_major']
    angle_rad = ellipse_params['angle_rad']
    eigenvals = ellipse_params['eigenvals']
    eigenvecs = ellipse_params['eigenvecs']
    delta_chi2 = ellipse_params['delta_chi2']

    # Set center
    if center is None:
        center = np.array([0.0, 0.0])
    else:
        center = np.asarray(center, dtype=float)

    # Rotation matrix (from principal axes to data axes)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    R = np.array([[cos_theta, -sin_theta],
                  [sin_theta,  cos_theta]])

    # Points in principal axes coordinates
    # The rotation angle is the semi-minor axis angle
    # In principal axes coordinates (before rotation):
    # - semi-minor axis (smaller uncertainty) is along x-direction (angle θ)
    # - semi-major axis (larger uncertainty) is along y-direction (angle θ + 90°)
    points_semi_minor_principal = np.array([[ semi_minor, 0.0],
                                             [-semi_minor, 0.0]])
    points_semi_major_principal = np.array([[0.0,  semi_major],
                                             [0.0, -semi_major]])

    # Rotate and translate to data coordinates
    points_semi_minor = points_semi_minor_principal @ R.T + center
    points_semi_major = points_semi_major_principal @ R.T + center

    # Build ellipse params dict (similar to _compute_ellipse_params_from_fisher)
    ellipse_params = {
        'semi_minor': semi_minor,
        'semi_major': semi_major,
        'angle_rad': angle_rad,
        'angle_deg': np.degrees(angle_rad),
        'eigenvals': eigenvals,
        'eigenvecs': eigenvecs,
        'delta_chi2': delta_chi2,
        'v_semi_minor': eigenvecs[:, 1],  # eigenvector for λ₁ (larger eigenvalue)
        'v_semi_major': eigenvecs[:, 0]   # eigenvector for λ₂ (smaller eigenvalue)
    }

    return {
        'non_degenerate_points': points_semi_minor,  # along semi-minor axis (smaller uncertainty)
        'degenerate_points': points_semi_major,      # along semi-major axis (larger uncertainty)
        'ellipse_params': ellipse_params
    }
