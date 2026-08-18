import numpy as np
from scipy.stats import chi2, norm

from LSS_python.AP import tpcf_convert_main


def cal_Fisher_matrix(func, best_fit, cov_matrix, delta=None, computed_jac=None, return_jac=False):
    """
    Calculate Fisher matrix from covariance matrix and model function.

    The Fisher information matrix is computed using the finite difference
    approximation of the gradient of the model function with respect to
    parameters:

    F = (∂μ/∂θ)^T * C^{-1} * (∂μ/∂θ)

    where μ = func(θ) is the model prediction, C is the covariance matrix,
    and ∂μ/∂θ is the Jacobian matrix.

    Parameters
    ----------
    func : callable
        Model function that takes parameters as individual arguments.
        Example: for 2 parameters, func(x1, x2).
    best_fit : array_like
        Best-fit parameter values.
    cov_matrix : ndarray
        Covariance matrix of the parameters.
    delta : float or array_like, optional
        Finite difference step size. If None, automatically determined
        using optimal step size δ = ε^(1/3) * max(|θ|, 1) for each parameter,
        where ε ≈ 2.22e-16 is machine epsilon. This gives 4th order
        accuracy for central differences. Can also be a single float
        applied to all parameters, or an array matching best_fit length.

    Returns
    -------
    ndarray
        Fisher information matrix with shape (n_params, n_params).

    Notes
    -----
    The automatic step size is based on the optimal step for numerical
    differentiation to minimize truncation and round-off errors:

    δ_optimal ≈ ε^(1/3) * max(|x|, 1)

    where ε is machine epsilon. This balances the truncation error
    (∝ δ²) and round-off error (∝ 1/δ) in central differences.
    """
    if computed_jac is None:
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

        # Compute inverse covariance matrix
        try:
            cov_inv = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            raise ValueError("Covariance matrix is singular, cannot compute inverse")

        # Compute Jacobian: ∂μ/∂θ for each parameter
        # We use central difference: (f(x+δ) - f(x-δ)) / (2δ)
        jacobian = np.zeros(n_params)

        # Evaluate function at best fit point to determine output dimension
        f0 = func(*best_fit)

        # Determine if model output is scalar or vector
        if hasattr(f0, '__len__') and not isinstance(f0, (float, int)):
            f0 = np.asarray(f0)
            n_output = f0.shape[0] if f0.ndim > 0 else 1
        else:
            f0 = float(f0)
            n_output = 1

        # Validate cov_matrix shape
        cov_matrix = np.atleast_2d(cov_matrix)
        if cov_matrix.shape != (n_output, n_output):
            raise ValueError(f"cov_matrix shape {cov_matrix.shape} does not match "
                            f"model output dimension {n_output}")

        # Initialize Jacobian matrix (n_output x n_params)
        jacobian = np.zeros((n_output, n_params))

        # Compute gradient for each parameter using central differences
        for i in range(n_params):
            # Forward point
            theta_plus = best_fit.copy()
            theta_plus[i] += delta[i]
            f_plus = func(*theta_plus)

            # Backward point
            theta_minus = best_fit.copy()
            theta_minus[i] -= delta[i]
            f_minus = func(*theta_minus)

            # Central difference
            if n_output == 1:
                jacobian[0, i] = (f_plus - f_minus) / (2 * delta[i])
            else:
                jacobian[:, i] = (np.asarray(f_plus) - np.asarray(f_minus)) / (2 * delta[i])

    # Compute Fisher matrix: F = J^T * C^{-1} * J
    # jacobian shape: (n_output, n_params)
    # cov_inv shape: (n_params, n_params)
    # Result: (n_params, n_output) @ (n_params, n_params) @ (n_output, n_params)
    #        = (n_params, n_params) @ (n_output, n_params) -> need to transpose jacobian
    else:
        jacobian = computed_jac
        cov_inv = np.linalg.inv(cov_matrix)
    fisher = jacobian.T @ cov_inv @ jacobian

    if return_jac:
        return fisher, jacobian
    else:
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
