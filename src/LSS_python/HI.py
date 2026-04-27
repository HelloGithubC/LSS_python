import numpy as np
from numba import njit, prange

from .base import yntra


# ============================================================================
# Core HI Mass Calculation Functions (JIT-compiled)
# ============================================================================

@njit(cache=True)
def _M_HI_halo_scalar(Mh, z, h, alpha_grid, m0_grid, mmin_grid, mhard_grid, redshift_grid, grid_size):
    """
    Calculate HI mass for a single halo (internal JIT-compiled version).
    
    Parameters
    ----------
    Mh : float64
        Halo mass in h^{-1} M_sun (comoving units)
    z : float64
        Redshift
    h : float64
        Hubble parameter (used for cosmological calculations, not for unit conversion)
    alpha_grid, m0_grid, mmin_grid, mhard_grid : ndarray
        Interpolation grids for parameters (in h^{-1} M_sun units)
    redshift_grid : ndarray
        Redshift grid points
    grid_size : int
        Size of interpolation grids
    
    Returns
    -------
    float64
        HI mass in h^{-1} M_sun (comoving units)
    
    Notes
    -----
    The formula used is:
        M_HI = M00 * (Mh / Mmin0)^alpha0 * exp(-(Mmin0 / Mh)^0.35)
    
    If Mh < Mhard0, the HI mass is set to 0.
    
    All masses (input, parameters, and output) are in h^{-1} M_sun units
    to maintain internal consistency within the h-dependent framework.
    """
    # Interpolate parameters to given redshift
    alpha0 = yntra(z, redshift_grid, alpha_grid, grid_size)
    M00 = yntra(z, redshift_grid, m0_grid, grid_size)
    Mmin0 = yntra(z, redshift_grid, mmin_grid, grid_size)
    Mhard0 = yntra(z, redshift_grid, mhard_grid, grid_size)
    
    # Apply hard cutoff
    if Mh < Mhard0:
        return 0.0
    
    # Calculate HI mass using the parameterized formula
    M_HI = M00 * (Mh / Mmin0)**alpha0 * np.exp(-(Mmin0 / Mh)**0.35)
    
    return M_HI


@njit(cache=True)
def _M_HI_halo_vectorized(Mh_array, z, h, alpha_grid, m0_grid, mmin_grid, mhard_grid, redshift_grid, grid_size):
    """
    Calculate HI mass for an array of halo masses at a single redshift (internal version).
    
    Parameters
    ----------
    Mh_array : ndarray
        Array of halo masses in h^{-1} M_sun (comoving units)
    z : float
        Redshift (same for all masses)
    h : float
        Hubble parameter
    alpha_grid, m0_grid, mmin_grid, mhard_grid : ndarray
        Interpolation grids for parameters
    redshift_grid : ndarray
        Redshift grid points
    grid_size : int
        Size of interpolation grids
    
    Returns
    -------
    ndarray
        Array of HI masses in h^{-1} M_sun (comoving units)
    """
    n = Mh_array.shape[0]
    result = np.empty(n, dtype=np.float64)
    
    for i in range(n):
        result[i] = _M_HI_halo_scalar(Mh_array[i], z, h, alpha_grid, m0_grid, mmin_grid, mhard_grid, redshift_grid, grid_size)
    
    return result


@njit(cache=True, parallel=True)
def _M_HI_halo_parallel(Mh_array, z_array, h, alpha_grid, m0_grid, mmin_grid, mhard_grid, redshift_grid, grid_size):
    """
    Calculate HI mass for arrays of halo masses and redshifts (parallel version).
    
    Parameters
    ----------
    Mh_array : ndarray
        Array of halo masses in h^{-1} M_sun (comoving units)
    z_array : ndarray
        Array of redshifts (same length as Mh_array)
    h : float
        Hubble parameter
    alpha_grid, m0_grid, mmin_grid, mhard_grid : ndarray
        Interpolation grids for parameters
    redshift_grid : ndarray
        Redshift grid points
    grid_size : int
        Size of interpolation grids
    
    Returns
    -------
    ndarray
        Array of HI masses in h^{-1} M_sun (comoving units)
    """
    n = Mh_array.shape[0]
    result = np.empty(n, dtype=np.float64)
    
    for i in prange(n):
        result[i] = _M_HI_halo_scalar(Mh_array[i], z_array[i], h, alpha_grid, m0_grid, mmin_grid, mhard_grid, redshift_grid, grid_size)
    
    return result


# ============================================================================
# Main HI Mass Calculator Class
# ============================================================================

class HI_Mass_Calculator:
    """
    A class for calculating HI mass from halo mass and redshift.
    
    This class encapsulates the cosmological parameters and interpolation grids
    needed for HI mass calculations, providing both scalar and vectorized 
    computation with optional parallel processing.
    
    Parameters
    ----------
    h : float, optional
        Hubble parameter (default: 0.6766, Planck 2018)
    Omegam : float, optional
        Matter density parameter (default: 0.3111, Planck 2018)
    Omegab : float, optional
        Baryon density parameter (default: 0.02242 / h^2, Planck 2018)
    Omegak : float, optional
        Curvature density parameter (default: 0.0)
    OmegaLambda : float, optional
        Dark energy density parameter (default: 0.6889, Planck 2018)
    ns : float, optional
        Scalar spectral index (default: 0.9665, Planck 2018)
    sigma8 : float, optional
        RMS mass fluctuation in 8 h^{-1} Mpc spheres (default: 0.8102, Planck 2018)
    
    Attributes
    ----------
    h, Omegam, Omegab, Omegak, OmegaLambda, ns, sigma8 : float
        Cosmological parameters
    rhoc, rhom, rhob : float
        Critical, matter, and baryon densities in M_sun/Mpc^3
    redshift_grid : ndarray
        Redshift grid points for interpolation
    alpha_grid, m0_grid, mmin_grid, mhard_grid : ndarray
        Interpolation grids for HI mass parameters
    grid_size : int
        Size of interpolation grids
    
    Examples
    --------
    >>> # Create calculator with default parameters
    >>> calc = HI_Mass_Calculator()
    >>>
    >>> # Calculate HI mass for a single halo
    >>> M_HI = calc.M_HI_halo(1e12, 0.0)
    >>>
    >>> # Calculate for array of halos
    >>> masses = np.array([1e10, 1e11, 1e12, 1e13])
    >>> M_HI = calc.M_HI_halo(masses, 0.5)
    >>>
    >>> # Use parallel computation
    >>> masses = np.logspace(8, 14, 100000)
    >>> M_HI = calc.M_HI_halo(masses, 0.5, parallel=True, nthreads=4)
    """
    
    def __init__(self, h=0.6766, Omegam=0.3111, Omegab=None, 
                 Omegak=0.0, OmegaLambda=0.6889, ns=0.9665, sigma8=0.8102):
        """
        Initialize the HI mass calculator with cosmological parameters.
        
        Parameters
        ----------
        h : float, optional
            Hubble parameter (default: 0.6766)
        Omegam : float, optional
            Matter density parameter (default: 0.3111)
        Omegab : float, optional
            Baryon density parameter (default: 0.02242 / h^2)
        Omegak : float, optional
            Curvature density parameter (default: 0.0)
        OmegaLambda : float, optional
            Dark energy density parameter (default: 0.6889)
        ns : float, optional
            Scalar spectral index (default: 0.9665)
        sigma8 : float, optional
            RMS mass fluctuation (default: 0.8102)
        """
        # Store cosmological parameters
        self.h = h
        self.Omegam = Omegam
        self.Omegab = Omegab if Omegab is not None else 0.02242 / h**2
        self.Omegak = Omegak
        self.OmegaLambda = OmegaLambda
        self.ns = ns
        self.sigma8 = sigma8
        
        # Calculate derived densities
        self.rhoc = 2.7752e11 * h**2  # Critical density in M_sun/Mpc^3
        self.rhom = self.rhoc * Omegam  # Matter density
        self.rhob = self.rhoc * self.Omegab  # Baryon density
        
        # Set up interpolation grids (fixed from empirical fits)
        self.redshift_grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        self.alpha_grid = np.array([0.24, 0.53, 0.60, 0.76, 0.79, 0.74], dtype=np.float64)
        self.m0_grid = np.array([4.3e10, 1.5e10, 1.3e10, 2.9e9, 1.4e9, 1.9e9], dtype=np.float64)
        self.mmin_grid = np.array([2.0e12, 6.0e11, 3.6e11, 6.7e10, 2.1e10, 2.0e10], dtype=np.float64)
        self.mhard_grid = np.array([1.5e10, 6.9e9, 3.1e9, 9.9e8, 3.9e8, 2.7e8], dtype=np.float64)
        self.grid_size = 6
    
    def M_HI_halo(self, Mh, z, parallel=False, nthreads=None):
        """
        Calculate HI mass from halo mass and redshift.
        
        This is the main interface function that automatically handles both
        scalar and array inputs.
        
        Parameters
        ----------
        Mh : float or ndarray
            Halo mass(es) in h^{-1} M_sun (comoving units). Can be:
            - A single float (scalar)
            - A numpy array of masses
        z : float or ndarray
            Redshift. Can be:
            - A single float (applied to all masses if Mh is array)
            - An array of redshifts (same length as Mh)
        parallel : bool, optional
            Whether to use parallel computation. Default is False.
            Only effective for array inputs with size > 1.
        nthreads : int, optional
            Number of threads to use for parallel computation.
            If None, uses Numba's default (all available cores).
            Only effective when parallel=True.
        
        Returns
        -------
        float or ndarray
            HI mass(es) in h^{-1} M_sun (comoving units). Returns scalar if inputs are scalar,
            otherwise returns numpy array.
        
        Examples
        --------
        >>> calc = HI_Mass_Calculator()
        >>> # Single halo
        >>> M_HI = calc.M_HI_halo(1e12, 0.0)
        >>> 
        >>> # Array of halos at same redshift
        >>> masses = np.array([1e10, 1e11, 1e12, 1e13])
        >>> M_HI = calc.M_HI_halo(masses, 0.5)
        >>> 
        >>> # Array of halos with different redshifts
        >>> masses = np.array([1e12, 1e12, 1e12])
        >>> redshifts = np.array([0.0, 0.5, 1.0])
        >>> M_HI = calc.M_HI_halo(masses, redshifts)
        >>> 
        >>> # Parallel computation with 4 threads
        >>> masses = np.logspace(8, 14, 100000)
        >>> M_HI = calc.M_HI_halo(masses, 0.5, parallel=True, nthreads=4)
        """
        # Set number of threads if specified
        if parallel and nthreads is not None:
            import numba
            numba.set_num_threads(nthreads)
        
        # Convert inputs to numpy arrays for uniform handling
        Mh_arr = np.asarray(Mh, dtype=np.float64)
        z_arr = np.asarray(z, dtype=np.float64)
        
        # Handle scalar case
        if Mh_arr.ndim == 0:
            return _M_HI_halo_scalar(float(Mh_arr), float(z_arr), self.h,
                                    self.alpha_grid, self.m0_grid, self.mmin_grid, 
                                    self.mhard_grid, self.redshift_grid, self.grid_size)
        
        # Handle array case
        if z_arr.ndim == 0:
            # Same redshift for all masses
            z_broadcast = np.full(Mh_arr.shape, float(z_arr), dtype=np.float64)
        else:
            # Array of redshifts
            if Mh_arr.shape != z_arr.shape:
                raise ValueError(f"Mh and z must have same shape, got {Mh_arr.shape} and {z_arr.shape}")
            z_broadcast = z_arr
        
        # Choose computation mode based on parallel parameter
        if parallel and Mh_arr.size > 1:
            # Use parallel version
            return _M_HI_halo_parallel(Mh_arr.flatten(), z_broadcast.flatten(), self.h,
                                      self.alpha_grid, self.m0_grid, self.mmin_grid,
                                      self.mhard_grid, self.redshift_grid, self.grid_size).reshape(Mh_arr.shape)
        else:
            # Use simple vectorized version (or scalar)
            if Mh_arr.ndim == 0:
                # Scalar case already handled above, but for completeness
                return _M_HI_halo_scalar(float(Mh_arr), float(z_broadcast), self.h,
                                        self.alpha_grid, self.m0_grid, self.mmin_grid,
                                        self.mhard_grid, self.redshift_grid, self.grid_size)
            else:
                # Array case with constant redshift
                return _M_HI_halo_vectorized(Mh_arr.flatten(), float(z_broadcast.flat[0]), self.h,
                                            self.alpha_grid, self.m0_grid, self.mmin_grid,
                                            self.mhard_grid, self.redshift_grid, self.grid_size).reshape(Mh_arr.shape)
    
    def Mhard0(self, z):
        """
        Calculate the minimum halo mass (hard cutoff) at redshift z.
        
        Parameters
        ----------
        z : float
            Redshift
        
        Returns
        -------
        float
            Minimum halo mass in h^{-1} M_sun (comoving units)
        
        Notes
        -----
        This corresponds to the Mhard0 function in 21cm.f.
        """
        return float(yntra(z, self.redshift_grid, self.mhard_grid, self.grid_size))


# ============================================================================
# Convenience wrapper function (for backward compatibility)
# ============================================================================

def M_HI_halo(Mh, z, parallel=False, nthreads=None):
    """
    Calculate HI mass from halo mass and redshift (convenience wrapper).
    
    This function creates a default HI_Mass_Calculator instance and calls
    its M_HI_halo method. For most use cases, consider creating a calculator
    instance directly for better performance when making multiple calls.
    
    Parameters
    ----------
    Mh : float or ndarray
        Halo mass(es) in h^{-1} M_sun (comoving units)
    z : float or ndarray
        Redshift
    parallel : bool, optional
        Whether to use parallel computation. Default is False.
    nthreads : int, optional
        Number of threads for parallel computation.
    
    Returns
    -------
    float or ndarray
        HI mass(es) in h^{-1} M_sun (comoving units)
    
    See Also
    --------
    HI_Mass_Calculator.M_HI_halo : Direct method with configurable parameters
    """
    calc = HI_Mass_Calculator()
    return calc.M_HI_halo(Mh, z, parallel=parallel, nthreads=nthreads)