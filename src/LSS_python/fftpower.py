import numpy as np
import os
from .JIT.fftpower import deal_ps_3d_multithreads, deal_ps_3d_single, cal_ps_from_numba

def deal_ps_3d_from_mesh(mesh, mesh_kernel=None, inplace=True, nthreads=1, device_id=-1, c_api=True, pybind=True):
    complex_field = mesh.complex_field if device_id < 0 else mesh.complex_field_gpu
    if mesh_kernel is not None:
        ps_3d_kernel = mesh_kernel.complex_field if device_id < 0 else mesh_kernel.complex_field_gpu
    else:
        ps_3d_kernel = None
    return deal_ps_3d(complex_field, ps_3d_kernel, np.prod(mesh.attrs["BoxSize"]), mesh.attrs["shotnoise"], inplace, nthreads, device_id, c_api, pybind)

def deal_ps_3d(complex_field, ps_3d_kernel=None, ps_3d_factor=1.0, shotnoise=0.0, inplace=False, nthreads=1, device_id=-1, c_api=True, pybind=True):
    if inplace:
        ps_3d = complex_field
    if device_id >= 0:
        import cupy as cp
        from .cuda.fftpower import deal_ps_3d_from_cuda
        if not inplace:
            ps_3d = cp.copy(complex_field)
        with cp.cuda.Device(device_id):
            deal_ps_3d_from_cuda(ps_3d, ps_3d_kernel, ps_3d_factor, shotnoise)
    else:
        if not inplace:
            ps_3d = np.copy(complex_field)
        if c_api:
            if not pybind:
                from .CPP.fftpower import deal_ps_3d_c_api
                deal_ps_3d_c_api(ps_3d, ps_3d_kernel, ps_3d_factor, shotnoise, nthreads)
            else:
                from .CPP.fftpower_pybind import deal_ps_3d_pybind
                deal_ps_3d_pybind(ps_3d, ps_3d_kernel, ps_3d_factor, shotnoise, nthreads)
        else:
            if nthreads > 1:
                deal_ps_3d_multithreads(ps_3d, ps_3d_kernel, ps_3d_factor, shotnoise, nthreads)
            else:
                deal_ps_3d_single(ps_3d, ps_3d_kernel, ps_3d_factor, shotnoise)
    if not inplace:
        return ps_3d
    else:
        return complex_field

def cal_ps_2d_from_mesh(mesh, mesh_kernel=None, k_arrays=None, nthreads=1, c_api=True, dk=None):
    ps_factor = np.prod(mesh.attrs["BoxSize"])
    shotnoise = mesh.attrs["shotnoise"]
    if c_api:
        from .CPP.fftpower_pybind import cal_ps_2d_from_mesh as cal_ps_2d_from_mesh_cpp
        return cal_ps_2d_from_mesh_cpp(mesh, mesh_kernel, k_arrays, ps_factor, shotnoise, nthreads, dk=dk)
    else:
        from .JIT.fftpower import cal_ps_2d_from_mesh as cal_ps_2d_from_mesh_numba
        return cal_ps_2d_from_mesh_numba(mesh, mesh_kernel, k_arrays, ps_factor, shotnoise, nthreads, dk=dk)

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

    def cal_ps_from_mesh(self, mesh, kmin, kmax, dk, Nmu=None, k_arrays=None,
    mode="1d", k_logarithmic=False, ps_3d_inplace=True, mesh_kernel=None, compensated=True, force_create_complex_field=False, nthreads=1, device_id=-1, c_api=True, pybind=True):
        """
        Calculate power spectrum from a mesh.
            Args:
                compensated: only used when mesh.complex_field is None.
                mesh_kernel: If set, its complex_field will be used to muliple the ps_3d.
        """
        shotnoise = mesh.attrs["shotnoise"]
        if device_id >= 0:
            import cupy as cp
            if mesh.complex_field_gpu is None or force_create_complex_field:
                mesh.r2c(compensated=compensated, nthreads=nthreads, device_id=device_id, c_api=c_api)
            ps_3d_gpu = mesh.complex_field_gpu
            ps_3d_need = ps_3d_gpu
            boxsize_prod = cp.prod(mesh.attrs["BoxSize"], dtype=cp.float32)
        else:
            if mesh.complex_field is None or force_create_complex_field:
                mesh.r2c(compensated=compensated, nthreads=nthreads, device_id=device_id, c_api=c_api)
            ps_3d = mesh.complex_field
            ps_3d_need = ps_3d
            boxsize_prod = np.prod(mesh.attrs["BoxSize"], dtype=np.float32)
                      
        if ps_3d_need is None:
            raise ValueError("mesh.complex_field(_gpu) is None. Please check if you have run the r2c or converted it to correct device.")
        
        if mesh_kernel is not None:
            ps_3d_kernel_need = mesh_kernel.complex_field_gpu if device_id >= 0 else mesh_kernel.complex_field
        else:
            ps_3d_kernel_need = None
            
        ps_3d_need = deal_ps_3d(ps_3d_need, ps_3d_kernel=ps_3d_kernel_need, ps_3d_factor=float(boxsize_prod), shotnoise=shotnoise, inplace=ps_3d_inplace, nthreads=nthreads, c_api=c_api, pybind=pybind)
        self.removed_shotnoise = True # Avoid shotnoise being removed twice
        self.attrs["shotnoise"] = shotnoise

        return self.cal_ps_from_3d(ps_3d_need, kmin, kmax, dk, Nmu=Nmu, k_arrays=k_arrays, mode=mode, k_logarithmic=k_logarithmic, nthreads=nthreads, c_api=c_api)

    def cal_ps_from_3d(
        self, ps_3d,
        kmin, kmax, dk, Nmu=None, k_arrays=None,
        mode="1d", k_logarithmic=False, shotnoise=0.0,
        nthreads=1, device_id=-1, c_api=True, pybind=True
    ):
        self.removed_shotnoise = True
        if device_id >= 0:
            import cupy as cp
            from .cuda.fftpower import cal_ps_from_cuda
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

        if k_arrays is None:
            k_x_array = np.fft.fftfreq(self.Nmesh[0], d=self.BoxSize[0] / self.Nmesh[0]) * 2.0 * np.pi
            k_y_array = np.fft.fftfreq(self.Nmesh[1], d=self.BoxSize[1] / self.Nmesh[1]) * 2.0 * np.pi
            k_z_array = np.fft.rfftfreq(self.Nmesh[2], d=self.BoxSize[2] / self.Nmesh[2]) * 2.0 * np.pi
        else:
            k_x_array, k_y_array, k_z_array = k_arrays

        if use_gpu:
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
    
    def intergrate_fftpower(self, k_min=-1.0, k_max=-1.0, mu_min=-1.0, mu_max=-1.0, integrate="k", use_fit_mu=False, norm=False, bin_pack=1, remove_last_bin=False):
        from .base import packarray
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
        if use_fit_mu:
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

        Pkmu_select = np.real(power["Pkmu"][k_min_index:k_max_index, mu_min_index:mu_max_index])
        mu_need = mu_array[mu_min_index: mu_max_index]
        k_need = k_array[k_min_index:k_max_index]
        if integrate == "k":
            Pkmu_integrate = np.nanmean(Pkmu_select, axis=0)
            # Apply normalization before remove_last_bin to ensure it's computed on full data
            if norm:
                Pkmu_integrate = Pkmu_integrate / np.nanmean(Pkmu_integrate)
            # Apply remove_last_bin after normalization
                if remove_last_bin:
                    Pkmu_integrate = Pkmu_integrate[:-1]
                    mu_need = mu_need[:-1]
            # Apply bin_pack to 1D arrays after nanmean
            if do_pack:
                mu_need = packarray(mu_need, bin_pack=bin_pack, axis=0)
                Pkmu_integrate = packarray(Pkmu_integrate, bin_pack=bin_pack, axis=0)
            return mu_need, Pkmu_integrate
        elif integrate == "mu":
            Pkmu_integrate = np.nanmean(Pkmu_select, axis=1)
            # Apply normalization before remove_last_bin to ensure it's computed on full data
            if norm:
                Pkmu_integrate = Pkmu_integrate / np.nanmean(Pkmu_integrate)
            # Note: remove_last_bin is not applicable for mu integration in current implementation
            # Apply bin_pack to 1D arrays after nanmean
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

class FFTPower2D(FFTPower):
    def __init__(self, Nmesh, BoxSize):
        self.k_2d = None
        self.ps_2d = None
        self.modes_2d = None
        self.removed_shotnoise = False
        self.attrs = {
            "shotnoise": 0.0,
            "Nmesh": Nmesh,
            "BoxSize": BoxSize,
        }

    def cal_ps_2d_from_mesh(
        self, mesh, mesh_kernel=None, k_arrays=None, nthreads=1, device_id=-1, c_api=True, dk=None,
        compensated=True, force_create_complex_field=False
    ):
        if device_id >= 0:
            raise ValueError("FFTPower2D currently does not support GPU mode for cal_ps_2d_from_mesh.")
        if mesh.complex_field is None or force_create_complex_field:
            mesh.r2c(compensated=compensated, nthreads=nthreads, device_id=device_id, c_api=c_api)
        self.k_2d, self.ps_2d, self.modes_2d = cal_ps_2d_from_mesh(
            mesh,
            mesh_kernel=mesh_kernel,
            k_arrays=k_arrays,
            nthreads=nthreads,
            c_api=c_api,
            dk=dk
        )
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

    def save(self, filename):
        import joblib

        save_dict = {
            "k_2d": self.k_2d,
            "ps_2d": self.ps_2d,
            "modes_2d": self.modes_2d,
            "attrs": self.attrs,
            "removed_shotnoise": self.removed_shotnoise,
        }
        
        dir_part = os.path.dirname(filename)
        if not os.path.exists(dir_part):
            os.makedirs(dir_part)
        joblib.dump(save_dict, filename)

    @classmethod
    def load(cls, filename):
        import joblib

        load_dict = joblib.load(filename)
        self = cls(
            load_dict["attrs"]["Nmesh"],
            load_dict["attrs"]["BoxSize"],
        )
        self.k_2d = load_dict["k_2d"]
        self.ps_2d = load_dict["ps_2d"]
        self.modes_2d = load_dict["modes_2d"]
        self.attrs = load_dict["attrs"]
        self.removed_shotnoise = load_dict.get("removed_shotnoise", False)
        return self