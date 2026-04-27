"""Data compression module for cosmological MCMC fitting.

This module provides a unified interface for three data compression algorithms:
PCA (Principal Component Analysis), KL (Karhunen-Loève transform), and MOPED
(MOPED Algorithm for Parameter Estimation and Data Compression).
"""

import numpy as np
from pathlib import Path
from scipy import linalg
from scipy.linalg import eigh
from sklearn.decomposition import PCA
import joblib


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
    >>> compressor.fit(precomputed_data=data, delta=d, cov_matrix=C,
    ...                fiducial_array=mu_fid)
    >>> X_compressed = compressor.transform(X_test)
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

    def fit(self, X_signal, *args, **kwargs) -> 'Compressor':
        """Fit the compression model.

        Parameters
        ----------
        X_signal : ndarray, shape (n_samples, n_features), optional
            Training signal data. Required for PCA and KL methods.
            For MOPED, this parameter is ignored; use `fiducial_array` instead.
        *args, **kwargs
            Method-specific parameters. See individual method documentation.

        Returns
        -------
        self : Compressor
            Fitted compressor.

        Raises
        ------
        ValueError
            If required parameters are missing for the specified method.
        """
        if self.method == 'moped':
            # For MOPED, X_signal is ignored; require fiducial_array instead
            if 'fiducial_array' not in kwargs or kwargs['fiducial_array'] is None:
                raise ValueError(
                    "For MOPED compression, 'fiducial_array' parameter is required. "
                    "This provides the theoretical model prediction used for centering."
                )
            self._mean = np.asarray(kwargs['fiducial_array'], dtype=np.float64)
            if self._mean.ndim != 1:
                raise ValueError(
                    f"fiducial_array must be 1D, got shape {self._mean.shape}"
                )
            # Remove fiducial_array from kwargs to avoid passing to _fit_moped
            kwargs = {k: v for k, v in kwargs.items() if k != 'fiducial_array'}
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
            self._fit_pca(X_signal, **kwargs)
        elif self.method == 'kl':
            self._fit_kl(X_signal, **kwargs)
        elif self.method == 'moped':
            self._fit_moped(X_signal, **kwargs)

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

        # Center the data
        X_centered = X - self._mean

        # Apply method-specific transform
        if self.method == 'pca':
            X_compressed = self._transform_pca(X)
        elif self.method == 'kl':
            X_compressed = self._transform_kl(X_centered)
        elif self.method == 'moped':
            X_compressed = self._transform_moped(X_centered)

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

    def _fit_pca(self, X_signal, n_components=None, X_noise=None):
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
        When n_components is explicitly provided, X_noise is ignored and
        standard PCA is performed, selecting the top n_components by explained variance.

        When n_components is None and X_noise is provided, the algorithm
        separately computes the covariance matrices for signal and noise data
        in the PCA space, then retains components where the signal standard
        deviation exceeds the noise standard deviation (std_signal > std_noise).
        """
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

    def _fit_moped(self, X_signal, precomputed_data, delta, cov_matrix, **kwargs):
        """Fit MOPED compression.

        MOPED (MOPED Algorithm for Parameter Estimation and Data Compression)
        compresses data to match the number of parameters while preserving
        Fisher information.

        The compression coefficients are computed as:
            b_i = (∂μ/∂θ_i)^T @ C^{-1} / sqrt((∂μ/∂θ_i)^T @ C^{-1} @ (∂μ/∂θ_i))

        Note: X_signal parameter is ignored for MOPED. The mean vector is
        provided via `fiducial_array` in the fit() method.

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

        n_features = len(self._mean)

        # Validate and use provided covariance matrix
        cov_matrix = np.asarray(cov_matrix, dtype=np.float64)
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

    def _transform_moped(self, X_centered):
        """Transform using fitted MOPED coefficients.

        Parameters
        ----------
        X_centered : ndarray, shape (n_samples, n_features)
            Centered data (X_signal - fiducial_array).

        Returns
        -------
        X_compressed : ndarray, shape (n_samples, n_parameters)
            Compressed data.
        """
        return X_centered @ self._moped_coefficients

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
