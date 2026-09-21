import numpy as np
import os
from .JIT.fftpower import deal_ps_3d_multithreads, deal_ps_3d_single, cal_ps_from_numba

def deal_ps_3d_from_mesh(mesh, mesh_kernel=None, ps_3d=None, nthreads=1, device_id=-1, c_api=True, pybind=True):
    """
    Calculate 3D power spectrum from mesh's complex field.
    
    Args:
        mesh: Input mesh object with complex_field attribute
        mesh_kernel: Optional kernel mesh
        ps_3d: Optional pre-allocated array for output. If provided, it will be 
               COMPLETELY OVERWRITTEN with new results. Use np.empty() for best 
               performance (no initialization needed). If None, a new array will 
               be created automatically. Useful for memory reuse in repeated calls.
        nthreads: Number of threads for parallel computation
        device_id: GPU device ID (-1 for CPU)
        c_api: Whether to use C API
        pybind: Whether to use pybind11
    
    Returns:
        ps_3d: 3D power spectrum array (same object if ps_3d was provided)
    """
    complex_field = mesh.complex_field if device_id < 0 else mesh.complex_field_gpu
    if mesh_kernel is not None:
        # Extract real part of kernel (kernel should be real-valued, stored in complex array)
        ps_3d_kernel = mesh_kernel.complex_field.real if device_id < 0 else mesh_kernel.complex_field_gpu.real
    else:
        ps_3d_kernel = None
    return deal_ps_3d(complex_field, ps_3d=ps_3d, ps_3d_kernel=ps_3d_kernel, ps_3d_factor=np.prod(mesh.attrs["BoxSize"]), shotnoise=mesh.attrs["shotnoise"], nthreads=nthreads, device_id=device_id, c_api=c_api, pybind=pybind)

def deal_ps_3d(complex_field, ps_3d=None, ps_3d_kernel=None, ps_3d_factor=1.0, shotnoise=0.0, nthreads=1, device_id=-1, c_api=True, pybind=True):
    """
    Calculate 3D power spectrum from complex field.
    
    Args:
        complex_field: Input complex field array (complex64 or complex128)
        ps_3d: Optional pre-allocated array for output. If provided, it will be 
               COMPLETELY OVERWRITTEN with new results. Use np.empty() for best 
               performance (no initialization needed). If None, a new array will 
               be created automatically. Useful for memory reuse in repeated calls.
        ps_3d_kernel: Optional kernel array (must match complex_field precision)
        ps_3d_factor: Factor for power spectrum normalization
        shotnoise: Shot noise to subtract
        nthreads: Number of threads for parallel computation
        device_id: GPU device ID (-1 for CPU)
        c_api: Whether to use C API
        pybind: Whether to use pybind11
    
    Returns:
        ps_3d: 3D power spectrum array (same object if ps_3d was provided)
    
    Example:
        # Memory reuse for repeated calculations
        ps_3d = np.empty(complex_field.shape, dtype=np.float32)
        for i in range(100):
            ps_3d = deal_ps_3d(complex_field, ps_3d=ps_3d, ...)
    """
    if device_id >= 0:
        import cupy as cp
        from .cuda.fftpower import deal_ps_3d_from_cuda
        with cp.cuda.Device(device_id):
            ps_3d = deal_ps_3d_from_cuda(complex_field_gpu=complex_field, ps_3d_gpu=ps_3d, ps_kernel_3d_gpu=ps_3d_kernel, ps_3d_factor=ps_3d_factor, shotnoise=shotnoise)
    else:
        if c_api:
            if not pybind:
                from .CPP.fftpower import deal_ps_3d_c_api
                ps_3d = deal_ps_3d_c_api(complex_field, ps_3d, ps_3d_kernel, ps_3d_factor, shotnoise, nthreads)
            else:
                from .CPP.fftpower_pybind import deal_ps_3d_pybind
                ps_3d = deal_ps_3d_pybind(complex_field, ps_3d, ps_3d_kernel, ps_3d_factor, shotnoise, nthreads)
        else:
            # Determine expected dtype for ps_3d
            expected_dtype = np.float32 if complex_field.dtype == np.complex64 else np.float64
            
            # Handle ps_3d parameter for Numba backend
            if ps_3d is None:
                ps_3d = np.empty(complex_field.shape, dtype=expected_dtype)
            else:
                if not isinstance(ps_3d, np.ndarray):
                    raise TypeError(f"ps_3d must be a numpy.ndarray or None, got {type(ps_3d)}")
                if ps_3d.dtype != expected_dtype:
                    raise TypeError(f"ps_3d dtype must be {expected_dtype} for complex_field dtype {complex_field.dtype}, got {ps_3d.dtype}")
                if ps_3d.shape != complex_field.shape:
                    raise ValueError(f"ps_3d shape {ps_3d.shape} must match complex_field shape {complex_field.shape}")
            
            if nthreads > 1:
                ps_3d = deal_ps_3d_multithreads(complex_field, ps_3d, ps_3d_kernel, ps_3d_factor, shotnoise, nthreads)
            else:
                ps_3d = deal_ps_3d_single(complex_field, ps_3d, ps_3d_kernel, ps_3d_factor, shotnoise)
    return ps_3d

def cal_ps_2d_from_mesh(mesh, mesh_kernel=None, nthreads=1, c_api=True, dk=-1):
    if dk is not None and dk < 0:
        dk = None
        print("Warning (cal_ps_2d_from_mesh): dk < 0, dk is set to None (auto).")
    ps_factor = np.prod(mesh.attrs["BoxSize"])
    shotnoise = mesh.attrs["shotnoise"]
    if c_api:
        from .CPP.fftpower_pybind import cal_ps_2d_from_mesh as cal_ps_2d_from_mesh_cpp
        return cal_ps_2d_from_mesh_cpp(mesh, mesh_kernel, ps_factor, shotnoise, nthreads, dk=dk)
    else:
        from .JIT.fftpower import cal_ps_2d_from_mesh as cal_ps_2d_from_mesh_numba
        return cal_ps_2d_from_mesh_numba(mesh, mesh_kernel, ps_factor, shotnoise, nthreads, dk=dk)

class FFTPower:
    def __init__(self, Nmesh, BoxSize):
        if isinstance(Nmesh, int) or isinstance(Nmesh, float):
            self.Nmesh = np.array([Nmesh, Nmesh, Nmesh], dtype=np.int32)
        else:
            self.Nmesh = np.array(Nmesh, dtype=np.int32)
        if isinstance(BoxSize, float) or isinstance(BoxSize, int):
            self.BoxSize = np.array([BoxSize, BoxSize, BoxSize], dtype=float)
        else:
            self.BoxSize = np.array(BoxSize, dtype=float)

        self.attrs = {
            "Nmesh": self.Nmesh,
            "BoxSize": self.BoxSize,
        }
        self.power = None
        self.removed_shotnoise = False
        self.is_run_ps_3d = False

    @classmethod
    def from_fftpower_2d(cls, fftpower_2d, kmin, kmax, dk, Nmu=None, mode="2d", k_logarithmic=False, nthreads=1, c_api=True):
        """
        Create an FFTPower object from an FFTPower2D object.

        This constructs an FFTPower instance by extracting Nmesh and BoxSize
        from the given FFTPower2D and computing the power spectrum via
        ``cal_pkmu_from_ps_2d``.

        Args:
            fftpower_2d: FFTPower2D instance (must have called ``cal_ps_2d_from_mesh`` first).
            kmin, kmax, dk: k-space binning parameters (passed to ``cal_pkmu_from_ps_2d``).
            Nmu: Number of mu bins.
            mode: Power spectrum mode ("2d" or "1d").
            k_logarithmic: Whether k bins are logarithmic.
            nthreads: Number of threads for parallel computation.
            c_api: Whether to use C API.

        Returns:
            FFTPower instance with the power spectrum computed.
        """
        self = cls(fftpower_2d.attrs["Nmesh"], fftpower_2d.attrs["BoxSize"])
        self.removed_shotnoise = fftpower_2d.removed_shotnoise
        self.attrs["shotnoise"] = 0.0 if fftpower_2d.removed_shotnoise else fftpower_2d.attrs["shotnoise"]
        self.cal_pkmu_from_ps_2d(
            fftpower_2d.ps_2d, fftpower_2d.k_2d, fftpower_2d.modes_2d,
            kmin=kmin, kmax=kmax, dk=dk, Nmu=Nmu,
            mode=mode, k_logarithmic=k_logarithmic,
            nthreads=nthreads, c_api=c_api,
        )
        if self.power is not None:
            self.power["modes"] = fftpower_2d.modes_2d
        return self

    def cal_ps_from_mesh(self, mesh, kmin, kmax, dk, Nmu=None,
    mode="1d", k_logarithmic=False, mesh_kernel=None, compensated=True, force_create_complex_field=False, nthreads=1, device_id=-1, c_api=True, pybind=True, ps_3d=None):
        """
        Calculate power spectrum from a mesh.
        
        Args:
            mesh: Input mesh object
            kmin, kmax, dk: k-space binning parameters
            Nmu: Number of mu bins (required for mode="2d")
            mode: Power spectrum mode ("1d" or "2d")
            k_logarithmic: Whether k bins are logarithmic
            mesh_kernel: If set, its complex_field.real will be used to multiply the ps_3d
            compensated: Only used when mesh.complex_field is None
            force_create_complex_field: Force recreation of complex field
            nthreads: Number of threads for parallel computation
            device_id: GPU device ID (-1 for CPU)
            c_api: Whether to use C API
            pybind: Whether to use pybind11
            ps_3d: Optional pre-allocated array for 3D power spectrum. If provided, it will be 
                   COMPLETELY OVERWRITTEN with new results. Use np.empty() for best performance 
                   (no initialization needed). If None, a new array will be created automatically.
                   Useful for memory reuse in repeated calls to avoid memory bloat.
        
        Returns:
            power: Power spectrum result dictionary
        
        Example:
            # Memory reuse for repeated calculations
            ps_3d = np.empty(mesh.complex_field.shape, dtype=np.float32)
            for i in range(100):
                power = fftpower.cal_ps_from_mesh(mesh, kmin, kmax, dk, ps_3d=ps_3d)
                # Process power spectrum...
        """
        shotnoise = mesh.attrs["shotnoise"]
        if device_id >= 0:
            import cupy as cp
            if mesh.complex_field_gpu is None or force_create_complex_field:
                mesh.r2c(compensated=compensated, nthreads=nthreads, device_id=device_id, c_api=c_api)
            complex_field = mesh.complex_field_gpu
            boxsize_prod = cp.prod(mesh.attrs["BoxSize"], dtype=cp.float32)
        else:
            if mesh.complex_field is None or force_create_complex_field:
                mesh.r2c(compensated=compensated, nthreads=nthreads, device_id=device_id, c_api=c_api)
            complex_field = mesh.complex_field
            boxsize_prod = np.prod(mesh.attrs["BoxSize"], dtype=np.float32)
                      
        if complex_field is None:
            raise ValueError("mesh.complex_field(_gpu) is None. Please check if you have run the r2c or converted it to correct device.")
        
        if mesh_kernel is not None:
            # Extract real part of kernel (kernel should be real-valued, stored in complex array)
            ps_3d_kernel = mesh_kernel.complex_field_gpu.real if device_id >= 0 else mesh_kernel.complex_field.real
        else:
            ps_3d_kernel = None
            
        ps_3d = deal_ps_3d(complex_field, ps_3d=ps_3d, ps_3d_kernel=ps_3d_kernel, ps_3d_factor=float(boxsize_prod), shotnoise=shotnoise, nthreads=nthreads, c_api=c_api, pybind=pybind)
        self.removed_shotnoise = True # Avoid shotnoise being removed twice
        self.attrs["shotnoise"] = shotnoise

        return self.cal_ps_from_3d(ps_3d, kmin, kmax, dk, Nmu=Nmu, mode=mode, k_logarithmic=k_logarithmic, nthreads=nthreads, device_id=device_id, c_api=c_api, pybind=pybind)

    def cal_ps_from_3d(
        self, ps_3d,
        kmin, kmax, dk, Nmu=None,
        mode="1d", k_logarithmic=False, shotnoise=0.0,
        nthreads=1, device_id=-1, c_api=True, pybind=True
    ):
        self.removed_shotnoise = True
        if device_id >= 0:
            use_gpu = True 
        else:
            use_gpu = False 
        self.attrs["kmin"] = kmin
        self.attrs["kmax"] = kmax
        if dk < 0:
            dk = 2 * np.pi / self.attrs["BoxSize"]
        self.attrs["dk"] = dk
        k_array = np.arange(kmin, kmax, dk)
        self.attrs["Nk"] = len(k_array) - 1
        self.attrs["mode"] = mode
        if mode == "2d":
            if not isinstance(Nmu, int):
                raise ValueError("Nmu must be an integer")
            else:
                self.attrs["Nmu"] = Nmu
                mu_array = np.linspace(0, 1, Nmu + 1, endpoint=True)
        else:
            self.attrs["Nmu"] = 1
            mu_array = np.array([0.0, 1.0])

        if "shotnoise" not in self.attrs:
            self.attrs["shotnoise"] = shotnoise

        k_x_array = np.fft.fftfreq(self.Nmesh[0], d=self.BoxSize[0] / self.Nmesh[0]) * 2.0 * np.pi
        k_y_array = np.fft.fftfreq(self.Nmesh[1], d=self.BoxSize[1] / self.Nmesh[1]) * 2.0 * np.pi
        k_z_array = np.fft.rfftfreq(self.Nmesh[2], d=self.BoxSize[2] / self.Nmesh[2]) * 2.0 * np.pi

        if use_gpu:
            import cupy as cp
            from .cuda.fftpower import cal_ps_from_cuda
            with cp.cuda.Device(device_id):
                k_x_array_gpu = cp.asarray(k_x_array, dtype=cp.float64)
                k_y_array_gpu = cp.asarray(k_y_array, dtype=cp.float64)
                k_z_array_gpu = cp.asarray(k_z_array, dtype=cp.float64)
                k_array_gpu = cp.asarray(k_array, dtype=cp.float64)
                mu_array_gpu = cp.asarray(mu_array, dtype=cp.float64)
                power_k, power_mu, power, power_modes = cal_ps_from_cuda(
                    ps_3d,
                    [k_x_array_gpu, k_y_array_gpu, k_z_array_gpu],
                    k_array_gpu,
                    mu_array_gpu
                )
                power_k = cp.asnumpy(power_k)
                power_mu = cp.asnumpy(power_mu)
                power = cp.asnumpy(power)
                power_modes = cp.asnumpy(power_modes)
        else:
            if c_api:
                if not pybind:
                    from .CPP.fftpower import cal_ps_c_api
                    power_k, power_mu, power, power_modes = cal_ps_c_api(
                        ps_3d,
                        [k_x_array, k_y_array, k_z_array],
                        k_array,
                        mu_array,
                        k_logarithmic=k_logarithmic,
                        nthreads=nthreads,
                    )
                else:
                    from .CPP.fftpower_pybind import cal_ps_pybind
                    power_k, power_mu, power, power_modes = cal_ps_pybind(
                        ps_3d,
                        [k_x_array, k_y_array, k_z_array],
                        k_array,
                        mu_array,
                        k_logarithmic=k_logarithmic,
                        nthreads=nthreads,
                    )
            else:
                power_k, power_mu, power, power_modes = cal_ps_from_numba(
                    ps_3d,
                    [k_x_array, k_y_array, k_z_array],
                    k_array,
                    mu_array,
                    k_logarithmic=k_logarithmic,
                    nthreads=nthreads,
                )
        
        if power_mu is None:
            raise ValueError("power_mu is None. Please check if you have run the r2c or converted it to correct device.")
        masked_index = power_modes == 0
        need_index = np.logical_not(masked_index)
        power_k[masked_index] = np.nan 
        power_mu[masked_index] = np.nan
        power[masked_index] = np.nan
        power_k[need_index] = power_k[need_index] / power_modes[need_index]
        if mode == "2d":
            power_mu[need_index] = power_mu[need_index] / power_modes[need_index]
        power[need_index] = power[need_index] / power_modes[need_index]
        if mode == "2d":
            self.power = {"k": power_k, "mu": power_mu, "Pkmu": power, "modes": power_modes}
        else:
            self.power = {"k": power_k.ravel(), "Pk": power.ravel(), "modes": power_modes.ravel()}
        self.attrs["Nmu"] = Nmu
        self.attrs["kmin"] = kmin
        self.attrs["kmax"] = kmax
        self.attrs["dk"] = dk
        return self.power
    
    def cal_pkmu_from_ps_2d(
        self, ps_2d, k_2d, modes_2d,
        kmin, kmax, dk, Nmu=None,
        mode="2d", k_logarithmic=False,
        nthreads=1, c_api=True
    ):
        """
        Calculate power spectrum from pre-computed 2D power spectrum.
        
        Args:
            ps_2d: 2D power spectrum array
            k_2d: 2D k-space coordinates array with shape (k_perp_bin, k_parallel_bin, 2)
            kmin: Minimum k value
            kmax: Maximum k value
            dk: k bin width
            Nmu: Number of mu bins
            mode: Power spectrum mode ("2d" or "1d")
            k_logarithmic: Whether k bins are logarithmic
            nthreads: Number of threads to use
            c_api: If True, use C++ backend; if False, use JIT backend
            
        Returns:
            Power spectrum in 2D (k-mu) format
        """
        self.attrs["kmin"] = kmin
        self.attrs["kmax"] = kmax
        if dk < 0:
            dk = 2 * np.pi / self.attrs["BoxSize"]
        self.attrs["dk"] = dk
        
        k_array = np.arange(kmin, kmax, dk)
        self.attrs["Nk"] = len(k_array) - 1
        self.attrs["mode"] = mode
        
        if mode == "2d":
            if not isinstance(Nmu, int):
                raise ValueError("Nmu must be an integer")
            else:
                self.attrs["Nmu"] = Nmu
                mu_array = np.linspace(0, 1, Nmu + 1, endpoint=True)
        else:
            self.attrs["Nmu"] = 1
            mu_array = np.array([0.0, 1.0])

        if c_api:
            from .CPP.fftpower_pybind import cal_pkmu_from_ps_2d as cal_pkmu_from_ps_2d_cpp
            power_k, power_mu, power, power_modes = cal_pkmu_from_ps_2d_cpp(
                ps_2d, k_2d, modes_2d, k_array, mu_array, k_logarithmic, nthreads
            )
        else:
            from .JIT.fftpower import cal_pkmu_from_ps_2d as cal_pkmu_from_ps_2d_jit
            power_k, power_mu, power, power_modes = cal_pkmu_from_ps_2d_jit(
                ps_2d, k_2d, modes_2d, k_array, mu_array, k_logarithmic, nthreads
            )
        
        # Handle NaN values and averaging
        masked_index = power_modes == 0
        need_index = np.logical_not(masked_index)
        
        power_k[masked_index] = np.nan
        power[masked_index] = np.nan
        power_k[need_index] = power_k[need_index] / power_modes[need_index]
        power[need_index] = power[need_index] / power_modes[need_index]
        
        if mode == "2d" and power_mu is not None:
            power_mu[masked_index] = np.nan
            power_mu[need_index] = power_mu[need_index] / power_modes[need_index]
            self.power = {"k": power_k, "mu": power_mu, "Pkmu": power, "modes": power_modes}
        else:
            self.power = {"k": power_k.ravel(), "Pk": power.ravel(), "modes": power_modes.ravel()}
        
        return self.power
    
    def intergrate_fftpower(self, k_min=-1.0, k_max=-1.0, mu_min=-1.0, mu_max=-1.0, integrate="k", use_fix_mu=False, norm=False, bin_pack=1, remove_last_bin=False, use_modes=False, with_k2=False):
        from .base import packarray
        if with_k2 and use_modes:
            raise ValueError("with_k2 and use_modes cannot be True at the same time")
        if not isinstance(bin_pack, int):
            bin_pack = int(bin_pack)
        if bin_pack > 1:
            do_pack = True 
        else:
            do_pack = False
        power = self.power 
        if power is None:
            raise ValueError("power is None")
        k_array = np.nanmean(power["k"], axis=1)
        if use_fix_mu:
            mu_array_edges = np.linspace(0.0, 1.0, self.attrs["Nmu"]+1)
            mu_array = (mu_array_edges[:-1] + mu_array_edges[1:]) / 2.0
        else:
            mu_array = np.nanmean(power["mu"], axis=0)
        
        if k_min <= k_array[0] or k_min < 0.0:
            k_min_index = 0
        else:
            k_min_index_source = np.where(k_array >= k_min)[0]
            if len(k_min_index_source) == 0:
                raise ValueError(f"k_min({k_min:.2f}) is too large")
            else:
                k_min_index = k_min_index_source[0]
        if k_max >= k_array[-1] or k_max < 0.0:
            k_max_index = len(k_array)
        else:
            k_max_index_source = np.where(k_array >= k_max)[0]
            if len(k_max_index_source) == 0:
                raise ValueError("k_max({k_max:.2f}) is too small")
            else:
                k_max_index = k_max_index_source[0]

        if mu_min <= mu_array[0] or mu_min < 0:
            mu_min_index = 0
        else:
            mu_min_index_source = np.where(mu_array >= mu_min)[0]
            if len(mu_min_index_source) == 0:
                raise ValueError(f"mu_min({mu_min:.2f}) is too large")
            else:
                mu_min_index = mu_min_index_source[0]
        if mu_max >= mu_array[-1] or mu_max < 0:
            mu_max_index = len(mu_array)
        else:
            mu_max_index_source = np.where(mu_array >= mu_max)[0]
            if len(mu_max_index_source) == 0:
                raise ValueError("mu_max({mu_max:.2f}) is too small")
            else:
                mu_max_index = mu_max_index_source[0]

        Pkmu_select = power["Pkmu"][k_min_index:k_max_index, mu_min_index:mu_max_index]
        modes_select = power["modes"][k_min_index:k_max_index, mu_min_index:mu_max_index]
        mu_need = mu_array[mu_min_index: mu_max_index]
        k_need = k_array[k_min_index:k_max_index]
        if integrate == "k":
            # Apply k^2 weighting before integration
            if with_k2:
                Pkmu_select = Pkmu_select * k_need[:, np.newaxis] ** 2        
            if use_modes:
                # Weighted average based on mode counts, handling NaN values in Pkmu_select
                weights = np.where(modes_select > 0, modes_select, 0.0)
                weight_sum = np.sum(weights, axis=0)
                # Avoid division by zero
                safe_weight_sum = np.where(weight_sum > 0, weight_sum, 1.0)
                # Use nansum to handle NaN values in Pkmu_select
                Pkmu_integrate = np.nansum(Pkmu_select * weights, axis=0) / safe_weight_sum
                # Set NaN where no valid modes
                Pkmu_integrate = np.where(weight_sum > 0, Pkmu_integrate, np.nan)
            else:
                # Equal-weight average (simple mean), handling NaN values
                count = np.sum(~np.isnan(Pkmu_select), axis=0)
                Pkmu_integrate = np.nansum(Pkmu_select, axis=0) / np.where(count > 0, count, 1.0)
                # Set NaN where no valid data
                Pkmu_integrate = np.where(count > 0, Pkmu_integrate, np.nan)
            # Apply normalization before remove_last_bin to ensure it's computed on full data
            if norm:
                norm_factor = np.nanmean(Pkmu_integrate)
                if norm_factor > 0:
                    Pkmu_integrate = Pkmu_integrate / norm_factor
            # Apply remove_last_bin after normalization
                if remove_last_bin:
                    Pkmu_integrate = Pkmu_integrate[:-1]
                    mu_need = mu_need[:-1]
            # Apply bin_pack to 1D arrays after weighted average
            if do_pack:
                mu_need = packarray(mu_need, bin_pack=bin_pack, axis=0)
                Pkmu_integrate = packarray(Pkmu_integrate, bin_pack=bin_pack, axis=0)
            return mu_need, Pkmu_integrate
        elif integrate == "mu":
            # Apply k^2 weighting before integration
            if with_k2:
                Pkmu_select = Pkmu_select * k_need[:, np.newaxis] ** 2
            
            if use_modes:
                # Weighted average based on mode counts, handling NaN values in Pkmu_select
                weights = np.where(modes_select > 0, modes_select, 0.0)
                weight_sum = np.sum(weights, axis=1)
                # Avoid division by zero
                safe_weight_sum = np.where(weight_sum > 0, weight_sum, 1.0)
                # Use nansum to handle NaN values in Pkmu_select
                Pkmu_integrate = np.nansum(Pkmu_select * weights, axis=1) / safe_weight_sum
                # Set NaN where no valid modes
                Pkmu_integrate = np.where(weight_sum > 0, Pkmu_integrate, np.nan)
            else:
                # Equal-weight average (simple mean), handling NaN values
                count = np.sum(~np.isnan(Pkmu_select), axis=1)
                Pkmu_integrate = np.nansum(Pkmu_select, axis=1) / np.where(count > 0, count, 1.0)
                # Set NaN where no valid data
                Pkmu_integrate = np.where(count > 0, Pkmu_integrate, np.nan)
            # Apply normalization before remove_last_bin to ensure it's computed on full data
            if norm:
                norm_factor = np.nanmean(Pkmu_integrate)
                if norm_factor > 0:
                    Pkmu_integrate = Pkmu_integrate / norm_factor
            # Note: remove_last_bin is not applicable for mu integration in current implementation
            # Apply bin_pack to 1D arrays after weighted average
            if do_pack:
                k_need = packarray(k_need, bin_pack=bin_pack, axis=0)
                Pkmu_integrate = packarray(Pkmu_integrate, bin_pack=bin_pack, axis=0)
            return k_need, Pkmu_integrate
        else:
            raise ValueError("integrate must be k or mu")
    
    def save(self, filename):
        import joblib

        save_dict = {
            "power": self.power,
            "attrs": self.attrs,
        }
        
        dir_part = os.path.dirname(filename)
        if not os.path.exists(dir_part):
            os.makedirs(dir_part)
        joblib.dump(save_dict, filename)

    @classmethod
    def load(cls, filename):
        import joblib

        load_dict = joblib.load(filename)
        self = FFTPower(
            load_dict["attrs"]["Nmesh"],
            load_dict["attrs"]["BoxSize"],
        )
        self.power = load_dict["power"]
        self.attrs = load_dict["attrs"]
        return self

class FFTPower2D:
    def __init__(self, Nmesh, BoxSize):
        self.k_2d = None
        self.ps_2d = None
        self.modes_2d = None
        # The edges are retained because k_2d contains only mode-weighted cell
        # centres and is insufficient for a conservative AP remapping.
        self.kperp_edges = None
        self.kparallel_edges = None
        self.removed_shotnoise = False
        self.attrs = {
            "shotnoise": 0.0,
            "Nmesh": Nmesh,
            "BoxSize": BoxSize,
        }

    def cal_ps_2d_from_mesh(
        self, mesh, mesh_kernel=None, nthreads=1, device_id=-1, c_api=True, dk:int|None=-1,
        compensated=True, force_create_complex_field=False
    ):
        if device_id >= 0:
            raise ValueError("FFTPower2D currently does not support GPU mode for cal_ps_2d_from_mesh.")
        if mesh.complex_field is None or force_create_complex_field:
            mesh.r2c(compensated=compensated, nthreads=nthreads, device_id=device_id, c_api=c_api)
        self.k_2d, self.ps_2d, self.modes_2d = cal_ps_2d_from_mesh(
            mesh,
            mesh_kernel=mesh_kernel,
            nthreads=nthreads,
            c_api=c_api,
            dk=dk
        )
        # Keep the exact binning used by both the Python and C++ backends.
        # The public wrapper changes a negative dk to None, which means the
        # fundamental mode in the line-of-sight direction.
        boxsize = np.asarray(mesh.attrs["BoxSize"], dtype=float)
        nmesh = np.asarray(mesh.attrs["Nmesh"], dtype=int)
        if dk is None or dk < 0:
            dk_2d = 2.0 * np.pi / boxsize[2]
        else:
            dk_2d = float(dk)
        kx = np.fft.fftfreq(nmesh[0], d=boxsize[0] / nmesh[0]) * 2.0 * np.pi
        ky = np.fft.fftfreq(nmesh[1], d=boxsize[1] / nmesh[1]) * 2.0 * np.pi
        kz = np.fft.rfftfreq(nmesh[2], d=boxsize[2] / nmesh[2]) * 2.0 * np.pi
        kperp_max = np.max(np.sqrt(kx**2 + ky**2))
        self.kperp_edges = np.arange(0.0, kperp_max + dk_2d, dk_2d)
        self.kparallel_edges = np.arange(kz[0], kz[-1] + dk_2d, dk_2d)
        self.removed_shotnoise = True
        self.attrs["shotnoise"] = mesh.attrs["shotnoise"]
    
    def cal_pkmu_from_ps_2d(
        self,
        kmin, kmax, dk, Nmu=None,
        mode="2d", k_logarithmic=False,
        nthreads=1, c_api=True
    ):
        if self.k_2d is None or self.ps_2d is None:
            raise ValueError("cal_ps_2d_from_mesh should be called first.")
        fftpower = FFTPower(self.attrs["Nmesh"], self.attrs["BoxSize"])
        fftpower.removed_shotnoise = self.removed_shotnoise
        fftpower.attrs["shotnoise"] = 0.0 if self.removed_shotnoise else self.attrs["shotnoise"]
        power_temp = fftpower.cal_pkmu_from_ps_2d(
            self.ps_2d, self.k_2d, self.modes_2d,
            kmin=kmin, kmax=kmax, dk=dk, Nmu=Nmu,
            mode=mode, k_logarithmic=k_logarithmic,
            nthreads=nthreads, c_api=c_api
        )
        if fftpower.power is None:
            raise ValueError("cal_pkmu_from_ps_2d failed.")
        fftpower.power["modes_2d"] = self.modes_2d
        return fftpower

    def cal_pkmu_from_ps_2d_ap(
        self,
        omega_mf, w_f, omega_mm, w_m, redshift,
        kmin, kmax, dk, Nmu=None,
        mode="2d", k_logarithmic=False,
        mesh_done_norm=True, subcell_n=4, nthreads=1, c_api=True,
        wa_f=None, wa_m=None,
    ):
        """Apply an AP remapping while conservatively rebinning a 2D spectrum.

        The AP factors follow :func:`LSS_python.AP.ps_convert_main`:
        ``alpha_perp = DA_m / DA_f`` and ``alpha_parallel = Hz_f / Hz_m``.
        Supplying both ``wa_f`` and ``wa_m`` selects the CPL ``w0wa``
        background when evaluating those factors.
        ``subcell_n=1`` reproduces the existing centre-coordinate assignment.
        For larger values, each source (k_perp, k_parallel) cell is represented
        by a regular ``subcell_n`` by ``subcell_n`` midpoint quadrature.  The
        fractional mode weights are proportional to k_perp, as appropriate for
        the cylindrical Fourier-space measure.  The returned ``modes`` are
        therefore floating-point effective mode weights when ``subcell_n > 1``.

        This method avoids reconstructing a 3D power array, but cannot recover
        the exact locations of individual modes that were averaged in ps_2d.
        """
        if self.k_2d is None or self.ps_2d is None or self.modes_2d is None:
            raise ValueError("cal_ps_2d_from_mesh should be called first.")
        if self.kperp_edges is None or self.kparallel_edges is None:
            raise ValueError(
                "kperp_edges and kparallel_edges are unavailable. Recreate the "
                "FFTPower2D object with cal_ps_2d_from_mesh before AP rebinning."
            )
        from .base import DA, Hz, Hz_w0wa

        if (wa_f is None) != (wa_m is None):
            raise ValueError("wa_f and wa_m must be provided together.")
        if wa_f is None:
            hz_f = Hz(redshift, omega_mf, w_f)
            hz_m = Hz(redshift, omega_mm, w_m)
            da_f = DA(redshift, omega_mf, w_f)
            da_m = DA(redshift, omega_mm, w_m)
        else:
            hz_f = Hz_w0wa(redshift, omega_mf, w_f, wa_f)
            hz_m = Hz_w0wa(redshift, omega_mm, w_m, wa_m)
            da_f = DA(redshift, omega_mf, w_f, wa_f)
            da_m = DA(redshift, omega_mm, w_m, wa_m)
        alpha_perp = da_m / da_f
        alpha_parallel = hz_f / hz_m
        if not np.isfinite(alpha_perp) or not np.isfinite(alpha_parallel):
            raise ValueError("AP conversion factors must be finite.")
        if alpha_perp <= 0.0 or alpha_parallel <= 0.0:
            raise ValueError("AP conversion factors must be positive.")
        if isinstance(subcell_n, (bool, np.bool_)) or not isinstance(
            subcell_n, (int, np.integer)
        ) or subcell_n < 1:
            raise ValueError("subcell_n must be a positive integer.")
        if mode not in ("1d", "2d"):
            raise ValueError("mode must be '1d' or '2d'.")
        if k_logarithmic:
            raise NotImplementedError("k_logarithmic=True is not supported for AP rebinning.")
        if dk < 0:
            dk = 2.0 * np.pi / np.asarray(self.attrs["BoxSize"], dtype=float)[2]
            dk /= alpha_parallel
        if not np.isscalar(dk) or dk <= 0.0:
            raise ValueError("dk must be a positive scalar.")

        k_edge = np.arange(kmin, kmax, dk, dtype=float)
        if k_edge.size < 2:
            raise ValueError("kmin, kmax, and dk must define at least one k bin.")
        if mode == "2d":
            if not isinstance(Nmu, (int, np.integer)) or Nmu < 1:
                raise ValueError("Nmu must be a positive integer when mode='2d'.")
            mu_edge = np.linspace(0.0, 1.0, Nmu + 1)
        else:
            Nmu = 1
            mu_edge = np.array([0.0, 1.0])

        if not isinstance(nthreads, (int, np.integer)) or nthreads != 1:
            raise ValueError("cal_pkmu_from_ps_2d_ap currently supports nthreads=1 only")
        if not isinstance(c_api, (bool, np.bool_)):
            raise TypeError("c_api must be a boolean")

        n_k = k_edge.size - 1
        if c_api:
            from .CPP.fftpower_pybind import cal_pkmu_from_ps_2d_ap as cal_pkmu_from_ps_2d_ap_cpp

            jacobian = alpha_perp**2 * alpha_parallel
            amplitude = jacobian if mesh_done_norm else 1.0 / jacobian
            power, k_output, mu_output, weight_sum = cal_pkmu_from_ps_2d_ap_cpp(
                self.ps_2d, self.k_2d, self.modes_2d,
                self.kperp_edges, self.kparallel_edges, k_edge, mu_edge,
                alpha_perp, alpha_parallel, amplitude, subcell_n,
            )
            boxsize = np.asarray(self.attrs["BoxSize"], dtype=float)
            boxsize_ap = boxsize * np.array([alpha_perp, alpha_perp, alpha_parallel])
            fftpower = FFTPower(self.attrs["Nmesh"], boxsize_ap)
            fftpower.removed_shotnoise = self.removed_shotnoise
            fftpower.attrs["shotnoise"] = 0.0 if self.removed_shotnoise else self.attrs["shotnoise"]
            fftpower.attrs.update({
                "mesh_done_norm": mesh_done_norm, "kmin": kmin, "kmax": kmax,
                "dk": dk, "Nk": n_k, "Nmu": Nmu, "mode": mode,
                "alpha_perp": alpha_perp, "alpha_parallel": alpha_parallel,
                "subcell_n": subcell_n,
            })
            shape = (n_k, Nmu)
            if mode == "2d":
                fftpower.power = {"k": k_output.reshape(shape), "mu": mu_output.reshape(shape),
                                  "Pkmu": power.reshape(shape), "modes": weight_sum.reshape(shape),
                                  "modes_2d": self.modes_2d}
            else:
                fftpower.power = {"k": k_output, "Pk": power, "modes": weight_sum,
                                  "modes_2d": self.modes_2d}
            return fftpower

        n_target = n_k * Nmu
        power_sum = np.zeros(n_target, dtype=float)
        weight_sum = np.zeros(n_target, dtype=float)
        k_sum = np.zeros(n_target, dtype=float)
        mu_sum = np.zeros(n_target, dtype=float)
        valid = np.isfinite(self.ps_2d) & (self.modes_2d > 0)
        source_power = self.ps_2d
        source_modes = self.modes_2d.astype(float, copy=False)
        jacobian = alpha_perp**2 * alpha_parallel
        amplitude = jacobian if mesh_done_norm else 1.0 / jacobian

        def accumulate(kperp_source, kparallel_source, fractional_weight):
            kperp = kperp_source / alpha_perp
            kparallel = kparallel_source / alpha_parallel
            k = np.sqrt(kperp**2 + kparallel**2)
            k_index = np.searchsorted(k_edge, k, side="right") - 1
            # Match the established convention that the final right edge is
            # included in the last bin.
            k_index[k_index == n_k] = n_k - 1
            in_range = valid & (k >= k_edge[0]) & (k <= k_edge[-1])
            if mode == "2d":
                mu = np.divide(kparallel, k, out=np.zeros_like(k), where=k > 0.0)
                mu_index = np.searchsorted(mu_edge, mu, side="right") - 1
                mu_index[mu_index == Nmu] = Nmu - 1
                in_range &= (mu >= mu_edge[0]) & (mu <= mu_edge[-1])
            else:
                mu = np.zeros_like(k)
                mu_index = np.zeros_like(k_index)

            target_index = k_index * Nmu + mu_index
            weights = source_modes * fractional_weight
            select = in_range.ravel()
            indices = target_index.ravel()[select]
            selected_weights = weights.ravel()[select]
            power_sum[:] += np.bincount(
                indices, weights=(selected_weights * source_power.ravel()[select] * amplitude), minlength=n_target
            )
            weight_sum[:] += np.bincount(indices, weights=selected_weights, minlength=n_target)
            k_sum[:] += np.bincount(indices, weights=selected_weights * k.ravel()[select], minlength=n_target)
            mu_sum[:] += np.bincount(indices, weights=selected_weights * mu.ravel()[select], minlength=n_target)

        if subcell_n == 1:
            # This path deliberately uses the stored mode-weighted centres so
            # it is a numerical control matching the previous implementation.
            accumulate(self.k_2d[..., 0], self.k_2d[..., 1], np.ones_like(source_modes))
        else:
            perp_lo, perp_hi = self.kperp_edges[:-1], self.kperp_edges[1:]
            parallel_lo, parallel_hi = self.kparallel_edges[:-1], self.kparallel_edges[1:]
            if (perp_lo.size, parallel_lo.size) != self.ps_2d.shape:
                raise ValueError("Stored 2D bin edges are inconsistent with ps_2d.")
            midpoint = (np.arange(subcell_n, dtype=float) + 0.5) / subcell_n
            perp_nodes = perp_lo[:, None] + (perp_hi - perp_lo)[:, None] * midpoint
            # Sum over both the perpendicular and parallel subcells. The
            # latter contributes a factor subcell_n to the normalization.
            perp_fraction = perp_nodes / (subcell_n * np.sum(perp_nodes, axis=1, keepdims=True))
            parallel_nodes = parallel_lo[:, None] + (parallel_hi - parallel_lo)[:, None] * midpoint
            for i_perp in range(subcell_n):
                kperp_source = perp_nodes[:, i_perp, None]
                fractional_weight = perp_fraction[:, i_perp, None]
                for i_parallel in range(subcell_n):
                    accumulate(kperp_source, parallel_nodes[:, i_parallel][None, :], fractional_weight)

        nonzero = weight_sum > 0.0
        power = np.full(n_target, np.nan, dtype=float)
        k_output = np.full(n_target, np.nan, dtype=float)
        mu_output = np.full(n_target, np.nan, dtype=float)
        power[nonzero] = power_sum[nonzero] / weight_sum[nonzero]
        k_output[nonzero] = k_sum[nonzero] / weight_sum[nonzero]
        mu_output[nonzero] = mu_sum[nonzero] / weight_sum[nonzero]

        boxsize = np.asarray(self.attrs["BoxSize"], dtype=float)
        boxsize_ap = boxsize * np.array([alpha_perp, alpha_perp, alpha_parallel])
        fftpower = FFTPower(self.attrs["Nmesh"], boxsize_ap)
        fftpower.removed_shotnoise = self.removed_shotnoise
        fftpower.attrs["shotnoise"] = 0.0 if self.removed_shotnoise else self.attrs["shotnoise"]
        fftpower.attrs.update({
            "mesh_done_norm": mesh_done_norm,
            "kmin": kmin,
            "kmax": kmax,
            "dk": dk,
            "Nk": n_k,
            "Nmu": Nmu,
            "mode": mode,
            "alpha_perp": alpha_perp,
            "alpha_parallel": alpha_parallel,
            "subcell_n": subcell_n,
        })
        shape = (n_k, Nmu)
        if mode == "2d":
            fftpower.power = {
                "k": k_output.reshape(shape),
                "mu": mu_output.reshape(shape),
                "Pkmu": power.reshape(shape),
                "modes": weight_sum.reshape(shape),
                "modes_2d": self.modes_2d,
            }
        else:
            fftpower.power = {
                "k": k_output,
                "Pk": power,
                "modes": weight_sum,
                "modes_2d": self.modes_2d,
            }
        return fftpower

    def save(self, filename):
        import joblib

        save_dict = {
            "k_2d": self.k_2d,
            "ps_2d": self.ps_2d,
            "modes_2d": self.modes_2d,
            "kperp_edges": self.kperp_edges,
            "kparallel_edges": self.kparallel_edges,
            "attrs": self.attrs,
            "removed_shotnoise": self.removed_shotnoise,
        }
        
        dir_part = os.path.dirname(filename)
        if not os.path.exists(dir_part):
            os.makedirs(dir_part)
        joblib.dump(save_dict, filename)

    @classmethod
    def load(cls, filename, mmap_mode=None):
        """
        Load a saved FFTPower2D object from a joblib file.

        Parameters
        ----------
        filename : str
            Path to the saved file.
        mmap_mode : str or None, optional
            joblib mmap mode (e.g. ``"r"``).  When given, the stored numpy
            arrays are memory-mapped read-only from disk instead of being
            fully deserialized into RAM; the returned object must then be
            treated as read-only.  Default: None (fully load into RAM).
        """
        import joblib

        load_dict = joblib.load(filename, mmap_mode=mmap_mode)
        self = cls(
            load_dict["attrs"]["Nmesh"],
            load_dict["attrs"]["BoxSize"],
        )
        self.k_2d = load_dict["k_2d"]
        self.ps_2d = load_dict["ps_2d"]
        self.modes_2d = load_dict["modes_2d"]
        self.kperp_edges = load_dict.get("kperp_edges")
        self.kparallel_edges = load_dict.get("kparallel_edges")
        self.attrs = load_dict["attrs"]
        self.removed_shotnoise = load_dict.get("removed_shotnoise", False)
        return self
    
    @classmethod
    def save_list(cls, filename, iterable):
        """
        Save multiple FFTPpower2D objects to a single file.
        
        Args:
            filename: Output file path
            iterable: An iterable of FFTPpower2D objects
        """
        import joblib
        
        save_list = []
        for obj in iterable:
            save_dict = {
                "k_2d": obj.k_2d,
                "ps_2d": obj.ps_2d,
                "modes_2d": obj.modes_2d,
                "kperp_edges": obj.kperp_edges,
                "kparallel_edges": obj.kparallel_edges,
                "attrs": obj.attrs,
                "removed_shotnoise": obj.removed_shotnoise,
            }
            save_list.append(save_dict)
        
        dir_part = os.path.dirname(filename)
        if not os.path.exists(dir_part):
            os.makedirs(dir_part)
        joblib.dump(save_list, filename)
    
    @classmethod
    def load_list(cls, filename, mmap_mode=None):
        """
        Load multiple FFTPpower2D objects from a file.

        Args:
            filename: Input file path
            mmap_mode: Optional joblib mmap mode (e.g. ``"r"``).  When given,
                the stored numpy arrays are memory-mapped read-only from disk
                instead of being fully deserialized into RAM; the returned
                objects must then be treated as read-only.  Default: None
                (fully load into RAM).

        Returns:
            A list of FFTPpower2D objects
        """
        import joblib

        load_list = joblib.load(filename, mmap_mode=mmap_mode)
        obj_list = []
        for load_dict in load_list:
            obj = cls(
                load_dict["attrs"]["Nmesh"],
                load_dict["attrs"]["BoxSize"],
            )
            obj.k_2d = load_dict["k_2d"]
            obj.ps_2d = load_dict["ps_2d"]
            obj.modes_2d = load_dict["modes_2d"]
            obj.kperp_edges = load_dict.get("kperp_edges")
            obj.kparallel_edges = load_dict.get("kparallel_edges")
            obj.attrs = load_dict["attrs"]
            obj.removed_shotnoise = load_dict.get("removed_shotnoise", False)
            obj_list.append(obj)
        return obj_list

def get_diff_main(fftpowers_2d_dict, snap_ids, Nmu, k_min=0.3, k_max=0.8, dk=0.02, shift=0, return_mu=False, integrate_func=None, integrate_kwargs=None, **kwargs):
    """Compute the difference of integrated power spectra between two snapshots.

    For each snapshot in ``snap_ids``, the corresponding ``FFTPower2D`` (or
    ``FFTPower``) objects are optionally converted to P(k, mu) via
    ``cal_pkmu_from_ps_2d``, then integrated over k or mu using either the
    built-in ``intergrate_fftpower`` method or a user-supplied
    ``integrate_func``.  The result is the rolled difference between the two
    snapshot groups.

    Parameters
    ----------
    fftpowers_2d_dict : dict
        Dictionary mapping snapshot IDs to power spectrum objects.
        Values can be a single ``FFTPower2D`` / ``FFTPower`` instance, or a
        list / tuple / ndarray of them.
    snap_ids : list
        A list of exactly two snapshot IDs.  The power spectra of the first
        ID are rolled by ``shift`` before the difference is taken.
    Nmu : int
        Number of mu bins used in ``cal_pkmu_from_ps_2d``.
    k_min : float, optional
        Minimum k for integration (default: 0.3).
    k_max : float, optional
        Maximum k for integration (default: 0.8).
    dk : float, optional
        k bin width for ``cal_pkmu_from_ps_2d`` conversion (default: 0.02).
    shift : int, optional
        Number of bins to roll the first snapshot's P(mu) array along the
        sample axis before subtraction (default: 0).  Set to 0 automatically
        when only one object is present for a snapshot.
    return_mu : bool, optional
        If True, prepend the mu values as the first row of the returned array.
        With a custom ``integrate_func``, the function must return
        ``(mu, integrated_value)`` when this option is enabled.
    integrate_func : callable, optional
        Custom integration function.  Its first argument must be an
        ``FFTPower2D`` instance.  When ``None`` (default), the built-in
        ``intergrate_fftpower`` method is used.
    integrate_kwargs : dict, optional
        Additional keyword arguments passed to ``integrate_func``.  Required
        when ``integrate_func`` is not ``None``.
    **kwargs : optional
        Additional keyword arguments:

        - ``kmin`` (float): Minimum k for ``cal_pkmu_from_ps_2d`` conversion
          (default: 0.1).
        - ``kmax`` (float): Maximum k for ``cal_pkmu_from_ps_2d`` conversion
          (default: 2.1).
        - ``mu_min`` (float): Minimum mu for integration (default: -1.0,
          meaning use all bins).
        - ``mu_max`` (float): Maximum mu for integration (default: -1.0,
          meaning use all bins).
        - ``with_modes`` (bool): Whether to weight by mode counts
          (default: False).
        - ``with_k2`` (bool): Whether to apply k^2 weighting before
          integration (default: False).
        - ``remove_last_bin`` (bool): Whether to drop the last mu bin after
          integration (default: True).

    Returns
    -------
    ndarray
        Difference array.  If ``return_mu`` is False and each snapshot has a
        single object, returns a 1-D array P(mu).  Otherwise the result is a
        2-D array.  When ``return_mu`` is True, the first row contains the mu
        values and the remaining rows contain the difference array.
    """
    kmin = kwargs.get("kmin", 0.1)
    kmax = kwargs.get("kmax", 2.1)

    mu_min = kwargs.get("mu_min", -1.0)
    mu_max = kwargs.get("mu_max", -1.0)

    with_modes = kwargs.get("with_modes", False)
    with_k2 = kwargs.get("with_k2", False)
    remove_last_bin = kwargs.get("remove_last_bin", True)

    Pmu_array_list = []
    mu_temp = None
    for snap_index, snap_id in enumerate(snap_ids):
        fftpower_2d_list = fftpowers_2d_dict[snap_id]
        if isinstance(fftpower_2d_list, FFTPower2D) or isinstance(fftpower_2d_list, FFTPower):
            fftpower_2d_list = [fftpower_2d_list, ]
        elif isinstance(fftpower_2d_list, list) or isinstance(fftpower_2d_list, tuple) or (fftpower_2d_list, np.ndarray):
            pass 
        else:
            raise ValueError(f"fftpower_2d_list must be a list (tuple) or a dict, but got {type(fftpower_2d_list)}")
        
        if len(fftpower_2d_list) == 1:
            shift = 0
        
        Pmu_list = []
        for fftpower_2d in fftpower_2d_list:
            if isinstance(fftpower_2d, FFTPower2D):
                fftpower_temp = fftpower_2d.cal_pkmu_from_ps_2d(kmin=kmin, kmax=kmax, dk=dk, Nmu=Nmu, mode="2d", nthreads=1, c_api=True)
            else:
                fftpower_temp = fftpower_2d
            if integrate_func is None:
                mu_current, Pmu_temp = fftpower_temp.intergrate_fftpower(k_min=k_min, k_max=k_max, mu_min=mu_min, mu_max=mu_max, use_fix_mu=True, norm=True, use_modes=with_modes, with_k2=with_k2, remove_last_bin=remove_last_bin)
            else:
                if integrate_kwargs is None or not isinstance(integrate_kwargs, dict):
                    raise ValueError(f"integrate_kwargs must be a dict, but got {type(integrate_kwargs)}")
                integrate_result = integrate_func(fftpower_temp, **integrate_kwargs)
                if return_mu:
                    if not isinstance(integrate_result, tuple) or len(integrate_result) != 2:
                        raise ValueError(
                            "integrate_func must return (mu, integrated_value) when return_mu=True"
                        )
                    mu_current, Pmu_temp = integrate_result
                else:
                    Pmu_temp = integrate_result
            if return_mu and snap_index == 0:
                mu_temp = mu_current
            Pmu_list.append(Pmu_temp)
        Pmu_array_list.append(np.array(Pmu_list))
    Pmu_array = np.roll(Pmu_array_list[0], shift, axis=0) - np.array(Pmu_array_list[1])
    if return_mu:
        result_array = Pmu_array[0] if len(Pmu_array) == 1 else Pmu_array
        return np.vstack((mu_temp, result_array))
    if len(Pmu_array) == 1:
        return Pmu_array[0]
    else:
        return Pmu_array
