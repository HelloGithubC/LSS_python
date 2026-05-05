"""Data compression module for cosmological MCMC fitting.

This module provides a unified interface for three data compression algorithms:
PCA (Principal Component Analysis), KL (Karhunen-Loève transform), and MOPED
(MOPED Algorithm for Parameter Estimation and Data Compression).
"""

import numpy as np
from pathlib import Path
from typing import List
from scipy import linalg
from scipy.linalg import eigh
from sklearn.decomposition import PCA
import joblib

from LSS_python.cov import cal_Fisher_matrix_from_precomputed


class Compressor:
    """Data compression using various algorithms (PCA, KL, MOPED).

    The compression method is fixed at initialization. All required parameters
    for training must be provided to the fit() method. The class does not infer
    any parameters automatically - users must provide all necessary information.

    Parameters
    ----------
    method : str
        Compression method. Options: 'pca', 'kl', 'moped'.

    Attributes
    ----------
    method : str
        The compression method used.
    is_fitted : bool
        Whether the compressor has been fitted.
    n_components : int
        Number of components after compression.

    Examples
    --------
    >>> # PCA compression
    >>> compressor = Compressor(method='pca')
    >>> compressor.fit(X_signal_train, n_components=10)
    >>> X_compressed = compressor.transform(X_test)

    >>> # KL compression with signal and noise arrays
    >>> compressor = Compressor(method='kl')
    >>> compressor.fit(X_signal=signal_train, X_noise=noise_train, snr_threshold=1.0)
    >>> X_compressed = compressor.transform(X_test)

    >>> # MOPED compression with precomputed data (central differences)
    >>> compressor = Compressor(method='moped')
    >>> compressor.fit(
    ...     precomputed_data=data,
    ...     delta=d,
    ...     cov_matrix=C
    ... )
    >>> X_compressed = compressor.transform(X_test)

    >>> # MOPED compression with Fisher matrix calculation
    >>> compressor = Compressor(method='moped')
    >>> compressor.fit(
    ...     precomputed_data=data,
    ...     delta=d,
    ...     cov_matrix=C,
    ...     best_fit=theta_best
    ... )
    >>> X_compressed = compressor.transform(X_test)
    >>> print(compressor.get_fisher_matrix())  # Access the computed Fisher matrix
    """

    def __init__(self, method: str):
        """Initialize the Compressor.

        Parameters
        ----------
        method : str
            Compression method. Options: 'pca', 'kl', 'moped'.

        Raises
        ------
        ValueError
            If method is not one of the valid options.
        """
        valid_methods = ('pca', 'kl', 'moped')
        if method not in valid_methods:
            raise ValueError(
                f"Invalid method '{method}'. Must be one of {valid_methods}."
            )

        self.method = method
        self.is_fitted = False
        self.n_components = None

        # Internal state for each method
        self._pca = None  # sklearn PCA instance
        self._mean = None  # Mean vector for centering
        self._projection = None  # Projection matrix (for KL and MOPED)
        self._eigenvalues_all = None # eigenvalues (for KL)
        self._moped_coefficients = None  # MOPED compression coefficients
        self._fisher_matrix = None  # Fisher information matrix (for MOPED)

    def get_fisher_matrix(self):
        """Get the Fisher information matrix.

        This method is only available for MOPED compression. An error is
        raised if the compressor uses a different method.

        Returns
        -------
        fisher_matrix : ndarray or None
            Fisher information matrix with shape (n_params, n_params) if
            computed during fitting (only available for MOPED method with
            best_fit parameter provided), otherwise None.

        Raises
        ------
        RuntimeError
            If the compressor method is not 'moped'.
        """
        if self.method != 'moped':
            raise RuntimeError(
                f"Fisher matrix is only available for MOPED compression, "
                f"but the compressor uses method '{self.method}'."
            )
        return self._fisher_matrix

    def fit(self, X_signal=None, *, n_components=None, pca_ratio_sum_max=None, X_noise=None, snr_threshold=None,
            fiducial_array=None, precomputed_data=None, delta=None, cov_matrix=None,
            best_fit=None):
        """Fit the compression model.

        Parameters
        ----------
        X_signal : ndarray, shape (n_samples, n_features), optional
            Training signal data. Required for PCA and KL methods.
            For MOPED, this parameter is ignored.

        # PCA-specific parameters
        n_components : int, optional
            Number of components to keep. Required for PCA if X_noise is not
            provided. If None and X_noise is provided, automatically determine
            components using std_signal > std_noise criterion.
        pca_ratio_sum_max : float, optional
            Maximum cumulative explained variance ratio (between 0 and 1).
            When provided, n_components is automatically determined to retain
            all components whose cumulative explained variance ratio is less
            than or equal to this value. If n_components is also provided,
            a warning is issued and n_components is ignored.
        X_noise : ndarray, shape (n_samples_var, n_features), optional
            Noise/variance data for automatic component selection. Only used
            when n_components is None.

        # KL-specific parameters
        snr_threshold : float, optional
            Signal-to-noise ratio threshold. Required for KL compression.
            Keep all modes with SNR >= threshold.

        # MOPED-specific parameters
        fiducial_array : ndarray, shape (n_features,), optional
            Fiducial array. Optional for MOPED compression; if provided,
            used only for dimension tracking, not for centering.
        precomputed_data : dict, optional
            Precomputed function values for derivative calculation. Required
            for MOPED compression. Format:
            {param_index: {'plus': f(theta + delta_i), 'minus': f(theta - delta_i)}}
        delta : float or array_like, optional
            Finite difference step size(s) for each parameter. Required for
            MOPED compression.
        cov_matrix : ndarray, shape (n_features, n_features), optional
            Covariance matrix. Required for MOPED compression.
        best_fit : array_like, optional
            Best-fit parameter values. Optional for MOPED. If provided, the
            Fisher information matrix is computed and stored in
            `self._fisher_matrix`.

        Returns
        -------
        self : Compressor
            Fitted compressor.

        Raises
        ------
        ValueError
            If required parameters are missing for the specified method.

        Examples
        --------
        >>> # PCA compression with explicit n_components
        >>> compressor = Compressor(method='pca')
        >>> compressor.fit(X_signal_train, n_components=10)

        >>> # PCA compression with automatic component selection
        >>> compressor = Compressor(method='pca')
        >>> compressor.fit(X_signal_train, X_noise=noise_train)

        >>> # PCA compression with cumulative variance threshold
        >>> compressor = Compressor(method='pca')
        >>> compressor.fit(X_signal_train, pca_ratio_sum_max=0.95)

        >>> # KL compression
        >>> compressor = Compressor(method='kl')
        >>> compressor.fit(X_signal_train, X_noise=noise_train, snr_threshold=1.0)

        >>> # MOPED compression
        >>> compressor = Compressor(method='moped')
        >>> compressor.fit(
        ...     precomputed_data=data,
        ...     delta=d,
        ...     cov_matrix=C
        ... )

        >>> # MOPED compression with Fisher matrix
        >>> compressor = Compressor(method='moped')
        >>> compressor.fit(
        ...     precomputed_data=data,
        ...     delta=d,
        ...     cov_matrix=C,
        ...     best_fit=theta_best
        ... )
    >>> print(compressor.get_fisher_matrix())
    """
        import warnings

        # Handle pca_ratio_sum_max: if provided, n_components will be ignored with a warning
        if pca_ratio_sum_max is not None and n_components is not None:
            warnings.warn(
                f"Both n_components ({n_components}) and pca_ratio_sum_max ({pca_ratio_sum_max}) "
                f"are provided. n_components will be ignored; pca_ratio_sum_max takes precedence.",
                UserWarning
            )
            n_components = None  # Override to let pca_ratio_sum_max control

        if self.method == 'moped':
            # For MOPED, X_signal is ignored; fiducial_array is optional
            # (provided for compatibility but not used for centering)
            if fiducial_array is not None:
                self._mean = np.asarray(fiducial_array, dtype=np.float64)
                if self._mean.ndim != 1:
                    raise ValueError(
                        f"fiducial_array must be 1D, got shape {self._mean.shape}"
                    )
            else:
                # Use a placeholder mean for dimension tracking; MOPED does not center
                self._mean = None
            # X_signal is ignored for MOPED
            X_signal = None
        else:
            # PCA and KL require X_signal
            if X_signal is None:
                raise ValueError(
                    f"X_signal is required for {self.method.upper()} compression"
                )
            X_signal = np.asarray(X_signal, dtype=np.float64)
            if X_signal.ndim != 2:
                raise ValueError(f"X_signal must be 2D, got shape {X_signal.shape}")
            n_samples, n_features = X_signal.shape
            if n_samples == 0:
                raise ValueError("X_signal must have at least one sample")
            self._mean = np.mean(X_signal, axis=0)

        # Dispatch to method-specific fit
        if self.method == 'pca':
            self._fit_pca(X_signal, n_components=n_components, pca_ratio_sum_max=pca_ratio_sum_max, X_noise=X_noise)
        elif self.method == 'kl':
            self._fit_kl(X_signal, X_noise=X_noise, snr_threshold=snr_threshold)
        elif self.method == 'moped':
            self._fit_moped(
                X_signal,
                precomputed_data=precomputed_data,
                delta=delta,
                cov_matrix=cov_matrix,
                best_fit=best_fit,
            )

        self.is_fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply compression to data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features) or (n_features,)
            Data to compress. Accepts both 1D vectors and 2D arrays.

        Returns
        -------
        X_compressed : ndarray, shape (n_samples, n_components) or (n_components,)
            Compressed data. Shape matches input: if input is 1D, output is 1D;
            if input is 2D, output is 2D.

        Raises
        ------
        RuntimeError
            If transform is called before fit.
        ValueError
            If input data shape doesn't match expected dimensions.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "This Compressor instance is not fitted yet. Call fit() first."
            )

        X = np.asarray(X, dtype=np.float64)

        # Handle 1D input: convert to 2D and track for output
        if X.ndim == 1:
            X = X.reshape(1, -1)
            return_1d = True
        elif X.ndim == 2:
            return_1d = False
        else:
            raise ValueError(f"X must be 1D or 2D, got {X.ndim}D with shape {X.shape}")

        n_features_expected = len(self._mean)
        if X.shape[1] != n_features_expected:
            raise ValueError(
                f"X has {X.shape[1]} features, but fitted with {n_features_expected} features"
            )

        # Apply method-specific transform
        if self.method == 'pca':
            # PCA: center the data first
            X_centered = X - self._mean
            X_compressed = self._transform_pca(X_centered)
        elif self.method == 'kl':
            # KL: center the data first
            X_centered = X - self._mean
            X_compressed = self._transform_kl(X_centered)
        elif self.method == 'moped':
            # MOPED: no centering, apply directly to data
            X_compressed = self._transform_moped(X)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # If input was 1D, squeeze output back to 1D
        if return_1d:
            X_compressed = X_compressed.squeeze(axis=0)

        return X_compressed

    def fit_transform(self, X_signal: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Fit and transform in one step.

        Parameters
        ----------
        X_signal : ndarray, shape (n_samples, n_features) or (n_features,)
            Training signal data. Accepts both 1D vectors and 2D arrays.
        *args, **kwargs
            Method-specific parameters.

        Returns
        -------
        X_compressed : ndarray, shape (n_samples, n_components) or (n_components,)
            Compressed data. Shape matches input: if input is 1D, output is 1D;
            if input is 2D, output is 2D.
        """
        return self.fit(X_signal, *args, **kwargs).transform(X_signal)

    # =========================================================================
    # PCA implementation
    # =========================================================================

    def _fit_pca(self, X_signal, n_components=None, pca_ratio_sum_max=None, X_noise=None):
        """Fit PCA compression.

        Parameters
        ----------
        X_signal : ndarray, shape (n_samples, n_features)
            Training signal data.
        n_components : int, optional
            Number of components to keep. If None and X_noise is provided,
            automatically determine the number of components using the criterion
            std_signal > std_noise. If an integer is provided, use that exact
            number of components from standard PCA (X_noise is ignored).
        pca_ratio_sum_max : float, optional
            Maximum cumulative explained variance ratio (between 0 and 1).
            When provided, n_components is automatically determined to retain
            all components whose cumulative explained variance ratio is less
            than or equal to this value. Takes precedence over n_components.
        X_noise : ndarray, shape (n_samples_var, n_features), optional
            Noise/variance data containing cosmological variance information.
            The first dimension can differ from X_signal, but the second dimension
            (n_features) must match. Only used when n_components is None.
            If provided and n_components is None, the compression will
            automatically determine components by:
            1. Fitting a temporary PCA on signal data X_signal
            2. Transforming both X_signal and X_noise to the PCA space
            3. Computing the covariance matrices separately for signal and noise data
            4. Retaining principal components where std_signal > std_noise

        Notes
        -----
        When n_components is explicitly provided (and pca_ratio_sum_max is None),
        X_noise is ignored and standard PCA is performed, selecting the top
        n_components by explained variance.

        When pca_ratio_sum_max is provided, it takes precedence and n_components
        is ignored (with a warning issued in the fit method).

        When n_components is None and X_noise is provided, the algorithm
        separately computes the covariance matrices for signal and noise data
        in the PCA space, then retains components where the signal standard
        deviation exceeds the noise standard deviation (std_signal > std_noise).
        """
        # If pca_ratio_sum_max is specified, use cumulative variance ratio to determine n_components
        if pca_ratio_sum_max is not None:
            if not (0 < pca_ratio_sum_max <= 1):
                raise ValueError(
                    f"pca_ratio_sum_max must be in (0, 1], got {pca_ratio_sum_max}"
                )
            # Fit PCA with all components first to get explained variance ratios
            self._pca = PCA(n_components=None)
            self._pca.fit(X_signal)
            # Get cumulative explained variance ratio
            cum_ratio = np.cumsum(self._pca.explained_variance_ratio_)
            # Find the number of components where cumulative ratio <= pca_ratio_sum_max
            # Use searchsorted to find the first index where cum_ratio > pca_ratio_sum_max
            n_selected = np.searchsorted(cum_ratio, pca_ratio_sum_max, side='right')
            if n_selected == 0:
                raise ValueError(
                    f"pca_ratio_sum_max={pca_ratio_sum_max} is too small; "
                    f"even the first component exceeds this threshold "
                    f"(first component ratio: {cum_ratio[0]:.6f})"
                )
            # Retrain PCA with the selected number of components
            self._pca = PCA(n_components=n_selected)
            self._pca.fit(X_signal)
            self.n_components = n_selected
            return

        # If n_components is specified, ignore X_noise and use standard PCA
        if n_components is not None:
            if not isinstance(n_components, int) or n_components <= 0:
                raise ValueError(
                    f"n_components must be a positive integer or None, "
                    f"got {n_components} of type {type(n_components).__name__}"
                )
            self._pca = PCA(n_components=n_components)
            self._pca.fit(X_signal)
            self.n_components = self._pca.n_components
            return

        # n_components is None: use automatic selection if X_noise provided
        # Fit PCA on signal data (keep all components for flexibility)
        self._pca = PCA(n_components=None)
        self._pca.fit(X_signal)

        if X_noise is not None:
            # Validate X_noise shape
            X_noise = np.asarray(X_noise, dtype=np.float64)
            if X_noise.ndim != 2:
                raise ValueError(f"X_noise must be 2D, got shape {X_noise.shape}")
            if X_noise.shape[1] != X_signal.shape[1]:
                raise ValueError(
                    f"X_noise has {X_noise.shape[1]} features, "
                    f"but X_signal has {X_signal.shape[1]} features"
                )

            # Transform both datasets to the PCA space
            signal_transformed = self._pca.transform(X_signal)
            noise_transformed = self._pca.transform(X_noise)

            # Compute covariance matrices separately
            cov_signal = np.cov(signal_transformed, rowvar=False)
            cov_noise = np.cov(noise_transformed, rowvar=False)

            # Extract diagonal elements (variances) and take square root (std)
            std_signal = np.sqrt(np.diag(cov_signal))
            std_noise = np.sqrt(np.diag(cov_noise))

            # Select components where std_signal > std_noise
            mask = std_signal > std_noise
            selected_indices = np.where(mask)[0]

            if len(selected_indices) == 0:
                raise ValueError(
                    "No components satisfy std_signal > std_noise. "
                    "All signal standard deviations are smaller than noise "
                    "standard deviations. Consider adjusting the data or providing "
                    "an explicit n_components value."
                )

            # Retrain PCA with the selected number of components
            n_selected = len(selected_indices)
            self._pca = PCA(n_components=n_selected)
            self._pca.fit(X_signal)
            self.n_components = n_selected
        else:
            # Standard PCA without noise data
            self.n_components = self._pca.n_components

    def _transform_pca(self, X):
        """Transform using fitted PCA."""
        return self._pca.transform(X)

    # =========================================================================
    # KL (Karhunen-Loève) implementation
    # =========================================================================

    def _fit_kl(self, X_signal, X_noise=None, snr_threshold=None):
        """Fit KL compression using generalized eigenvalue decomposition.

        Solves the generalized eigenvalue problem:
            C_signal * v = λ * C_noise * v
        where λ represents the signal-to-noise ratio (SNR) of each mode.

        Parameters
        ----------
        X_signal : ndarray, shape (n_samples, n_features)
            Signal training data (used to compute mean for centering and cov_signal).
        X_noise : ndarray, shape (n_samples_var, n_features), optional
            Noise/variance data. If provided, cov_noise is computed from this array.
            The first dimension can differ from X_signal, but the second dimension
            (n_features) must match. If None, an identity matrix is used (no noise).
        snr_threshold : float, required
            Signal-to-noise ratio threshold. Keep all modes with SNR >= threshold.

        Raises
        ------
        ValueError
            If any required parameter is missing or shapes mismatch.
        """
        # Validate required parameters
        if snr_threshold is None:
            raise ValueError("snr_threshold is required for KL compression")

        # Compute cov_signal from X_signal
        cov_signal = np.cov(X_signal, rowvar=False)

        # Compute cov_noise from X_noise or use identity
        if X_noise is not None:
            X_noise = np.asarray(X_noise, dtype=np.float64)
            if X_noise.ndim != 2:
                raise ValueError(f"X_noise must be 2D, got shape {X_noise.shape}")
            if X_noise.shape[1] != X_signal.shape[1]:
                raise ValueError(
                    f"X_noise has {X_noise.shape[1]} features, "
                    f"but X_signal has {X_signal.shape[1]} features"
                )
            cov_noise = np.cov(X_noise, rowvar=False)
        else:
            # Use identity matrix (no noise weighting)
            n_features = X_signal.shape[1]
            cov_noise = np.eye(n_features)

        n_features = len(self._mean)
        if cov_signal.shape != (n_features, n_features):
            raise ValueError(
                f"cov_signal shape {cov_signal.shape} doesn't match "
                f"expected ({n_features}, {n_features})"
            )
        if cov_noise.shape != (n_features, n_features):
            raise ValueError(
                f"cov_noise shape {cov_noise.shape} doesn't match "
                f"expected ({n_features}, {n_features})"
            )

        # Solve generalized eigenvalue problem: C_signal * v = λ * C_noise * v
        # Use eigh for symmetric positive definite matrices
        # eigenvalues are sorted in ascending order
        eigenvalues, eigenvectors = eigh(cov_signal, cov_noise)

        # Compute SNR for each mode (generalized eigenvalues)
        # Sort by descending SNR
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Select modes with SNR >= threshold
        mask = eigenvalues >= snr_threshold
        n_selected = np.sum(mask)

        if n_selected == 0:
            raise ValueError(
                f"No modes with SNR >= {snr_threshold}. "
                f"Maximum SNR is {np.max(eigenvalues):.3f}"
            )

        # Store projection matrix (each column is an eigenvector)
        self._projection = eigenvectors[:, mask]
        self.n_components = n_selected
        self._eigenvalues_all = eigenvalues

    def _transform_kl(self, X_centered):
        """Transform using fitted KL projection.

        Parameters
        ----------
        X_centered : ndarray, shape (n_samples, n_features)
            Centered data (X_signal - mean).

        Returns
        -------
        X_compressed : ndarray, shape (n_samples, n_components)
            Compressed data.
        """
        return X_centered @ self._projection

    # =========================================================================
    # MOPED implementation
    # =========================================================================

    def _fit_moped(self, X_signal, precomputed_data, delta, cov_matrix, best_fit=None, **kwargs):
        """Fit MOPED compression.

        MOPED (MOPED Algorithm for Parameter Estimation and Data Compression)
        compresses data to match the number of parameters while preserving
        Fisher information.

        The compression coefficients are computed as:
            b_i = (∂μ/∂θ_i)^T @ C^{-1} / sqrt((∂μ/∂θ_i)^T @ C^{-1} @ (∂μ/∂θ_i))

        Note: X_signal parameter is ignored for MOPED. The `fiducial_array`
        parameter (provided in fit()) is optional and only used for dimension
        tracking; it is NOT used for centering the data during transformation.

        Parameters
        ----------
        X_signal : None
            Ignored. Included for API consistency with other methods.
        precomputed_data : dict, required
            Precomputed function values for derivative calculation using central differences.
            Format: {param_index: {'plus': f(theta + delta_i), 'minus': f(theta - delta_i)}}
            Each parameter must have both 'plus' and 'minus' entries.
        delta : float or array_like, required
            Finite difference step size(s) for each parameter.
        cov_matrix : ndarray, shape (n_features, n_features), required
            Covariance matrix. Must be provided explicitly.
        best_fit : array_like, optional
            Best-fit parameter values. If provided, the Fisher information matrix
            is computed using `cal_Fisher_matrix_from_precomputed` and stored in
            `self._fisher_matrix`.

        Raises
        ------
        ValueError
            If any required parameter is missing or if precomputed_data
            does not contain both 'plus' and 'minus' for any parameter.
        """
        # Validate required parameters
        if precomputed_data is None:
            raise ValueError("precomputed_data is required for MOPED compression")
        if delta is None:
            raise ValueError("delta is required for MOPED compression")
        if cov_matrix is None:
            raise ValueError("cov_matrix is required for MOPED compression")

        # Determine number of parameters from precomputed_data
        n_parameters = len(precomputed_data)
        if n_parameters == 0:
            raise ValueError("precomputed_data must contain at least one parameter")

        # Process delta
        if np.isscalar(delta):
            delta = np.full(n_parameters, delta, dtype=np.float64)
        else:
            delta = np.asarray(delta, dtype=np.float64)
            if len(delta) != n_parameters:
                raise ValueError(
                    f"delta length {len(delta)} doesn't match "
                    f"number of parameters {n_parameters}"
                )

        # Determine n_features from cov_matrix (since fiducial_array is optional)
        cov_matrix = np.asarray(cov_matrix, dtype=np.float64)
        n_features = cov_matrix.shape[0]
        # Set mean for dimension tracking if not already set
        if self._mean is None:
            self._mean = np.zeros(n_features)

        # Validate covariance matrix shape
        if cov_matrix.shape != (n_features, n_features):
            raise ValueError(
                f"cov_matrix shape {cov_matrix.shape} doesn't match "
                f"expected ({n_features}, {n_features})"
            )

        # Compute covariance inverse
        try:
            cov_inv = linalg.inv(cov_matrix)
        except linalg.LinAlgError:
            # Use pseudo-inverse if singular
            cov_inv = linalg.pinv(cov_matrix)

        # Compute derivatives from precomputed data using central differences
        derivatives = np.zeros((n_features, n_parameters))
        for i in range(n_parameters):
            if i not in precomputed_data:
                raise ValueError(
                    f"precomputed_data missing entry for parameter index {i}"
                )

            param_data = precomputed_data[i]
            if 'plus' not in param_data or 'minus' not in param_data:
                raise ValueError(
                    f"precomputed_data for parameter {i} must contain both "
                    f"'plus' and 'minus' entries for central difference"
                )

            f_plus = np.asarray(param_data['plus'], dtype=np.float64)
            f_minus = np.asarray(param_data['minus'], dtype=np.float64)
            derivatives[:, i] = (f_plus - f_minus) / (2.0 * delta[i])

        # Compute MOPED compression coefficients using equation (13):
        # b_m = [C^{-1} μ,m - Σ_{q=1}^{m-1} (μ,m^T b_q) b_q] / sqrt[μ,m^T C^{-1} μ,m - Σ_{q=1}^{m-1} (μ,m^T b_q)^2]
        self._moped_coefficients = np.zeros((n_features, n_parameters))
        for i in range(n_parameters):
            db_dtheta = derivatives[:, i]  # μ,m = ∂μ/∂θ_m

            # Compute sum of squared projections for denominator
            sum_sq_projections = 0.0
            for j in range(i):
                b_j = self._moped_coefficients[:, j]
                # Projection coefficient: μ,m^T b_q
                proj_coeff = b_j @ db_dtheta
                sum_sq_projections += proj_coeff ** 2

            # Original squared norm: μ,m^T C^{-1} μ,m
            original_sq_norm = db_dtheta @ cov_inv @ db_dtheta

            # Denominator: sqrt(original_sq_norm - sum_sq_projections)
            denominator = np.sqrt(original_sq_norm - sum_sq_projections)
            if denominator < 1e-30:
                raise ValueError(
                    f"Derivative for parameter {i} is too small "
                    f"(norm < 1e-30). Check precomputed data."
                )

            # Numerator: C^{-1} [μ,m - Σ b_q (μ,m^T b_q)]
            numerator = cov_inv @ db_dtheta
            for j in range(i):
                b_j = self._moped_coefficients[:, j]
                proj_coeff = b_j @ db_dtheta
                numerator -= proj_coeff * b_j

            self._moped_coefficients[:, i] = numerator / denominator

        self.n_components = n_parameters

        # Compute and store Fisher information matrix if best_fit is provided
        if best_fit is not None:
            self._fisher_matrix = cal_Fisher_matrix_from_precomputed(
                precomputed_data, best_fit, delta, cov_matrix
            )

    def _transform_moped(self, X):
        """Transform using fitted MOPED coefficients.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data to compress. No centering is applied.

        Returns
        -------
        X_compressed : ndarray, shape (n_samples, n_parameters)
            Compressed data.
        """
        return X @ self._moped_coefficients

    # =========================================================================
    # Save and Load functionality
    # =========================================================================

    def save(self, filepath: str):
        """Save the fitted Compressor to a file.

        Parameters
        ----------
        filepath : str
            Path to the file where the compressor will be saved.
            Recommended extension: .pkl or .joblib

        Raises
        ------
        RuntimeError
            If the compressor has not been fitted yet.

        Examples
        --------
        >>> compressor = Compressor(method='pca')
        >>> compressor.fit(X_train, n_components=10)
        >>> compressor.save('compressor.pkl')
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Cannot save an unfitted Compressor. Call fit() first."
            )

        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: str) -> 'Compressor':
        """Load a fitted Compressor from a file.

        Parameters
        ----------
        filepath : str
            Path to the file from which to load the compressor.

        Returns
        -------
        compressor : Compressor
            The loaded and fitted Compressor instance.

        Examples
        --------
        >>> compressor = Compressor.load('compressor.pkl')
        >>> X_compressed = compressor.transform(X_test)
        """
        return joblib.load(filepath)

class CompressorSeries:
    """Container for a series of compressors."""

    def __init__(self, compressors: List[Compressor]):
        if not isinstance(compressors, list) or len(compressors) == 0:
            raise ValueError("compressors must be a non-empty list of Compressor instances")

        for i, compressor in enumerate(compressors):
            if not isinstance(compressor, Compressor):
                raise TypeError(
                    f"compressors[{i}] must be a Compressor instance, got "
                    f"{type(compressor).__name__}"
                )
            if not compressor.is_fitted:
                raise RuntimeError(
                    f"compressors[{i}] is not fitted. Fit each Compressor before "
                    "building a CompressorSeries."
                )
            if compressor._mean is None:
                raise RuntimeError(
                    f"compressors[{i}] has no fitted input feature definition (_mean is None)."
                )
            if compressor.n_components is None:
                raise RuntimeError(
                    f"compressors[{i}] has no fitted output dimension (n_components is None)."
                )

        for i in range(len(compressors) - 1):
            output_dim = compressors[i].n_components
            next_input_dim = len(compressors[i + 1]._mean)
            if output_dim != next_input_dim:
                raise ValueError(
                    f"Dimension mismatch between compressors[{i}] and compressors[{i + 1}]: "
                    f"output_dim={output_dim} != next_input_dim={next_input_dim}"
                )

        self.compressors = compressors

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply each compressor's transform sequentially."""
        X_transformed = X
        for compressor in self.compressors:
            X_transformed = compressor.transform(X_transformed)
        return X_transformed

    def save(self, filepath: str):
        """Save the compressor series to a file via joblib."""
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'CompressorSeries':
        """Load a compressor series from a file via joblib."""
        loaded = joblib.load(filepath)
        if not isinstance(loaded, cls):
            raise TypeError(
                f"Loaded object is {type(loaded).__name__}, expected {cls.__name__}"
            )
        return loaded
