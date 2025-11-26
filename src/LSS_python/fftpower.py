import numpy as np
from .JIT.fftpower import deal_ps_3d_multithreads, deal_ps_3d_single, cal_ps_from_numba

def deal_ps_3d(complex_field, ps_3d_kernel=None, ps_3d_factor=1.0, shotnoise=0.0, inplace=True, nthreads=1, device_id=-1, c_api=False):
    if not inplace:
        ps_3d = np.copy(complex_field)
    else:
        ps_3d = complex_field
    if device_id >= 0:
        import cupy as cp
        from .cuda.fftpower import deal_ps_3d_from_cuda
        with cp.cuda.Device(device_id):
            deal_ps_3d_from_cuda(ps_3d, ps_3d_kernel, ps_3d_factor, shotnoise)
    else:
        if c_api:
            from .CPP.fftpower import deal_ps_3d_c_api
            deal_ps_3d_c_api(ps_3d, ps_3d_kernel, ps_3d_factor, shotnoise, nthreads)
        else:
            if nthreads > 1:
                deal_ps_3d_multithreads(ps_3d, ps_3d_kernel, ps_3d_factor, shotnoise, nthreads)
            else:
                deal_ps_3d_single(ps_3d, ps_3d_kernel, ps_3d_factor, shotnoise)
    if not inplace:
        return ps_3d
    else:
        return complex_field

class FFTPower:
    def __init__(self, Nmesh, BoxSize, shotnoise=0.0):
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
            "shotnoise": shotnoise,
        }
        self.power = None
        self.removed_shotnoise = False

    def cal_ps_from_mesh(self, mesh, kmin, kmax, dk, Nmu=None, k_arrays=None,
    mode="1d", k_logarithmic=False, ps_3d_inplace=True, nthreads=1, device_id=-1, c_api=False, test_mode=False):
        shotnoise = mesh.attrs["shotnoise"]
        if device_id >= 0:
            import cupy as cp
            if ps_3d_inplace:
                ps_3d_gpu = mesh.complex_field_gpu
            else:
                ps_3d_gpu = cp.copy(mesh.complex_field_gpu)
            ps_3d_need = ps_3d_gpu
            boxsize_prod = cp.prod(self.BoxSize, dtype=cp.float32)
        else:
            if ps_3d_inplace:
                ps_3d = mesh.complex_field
            else:
                ps_3d = np.copy(mesh.complex_field)
            ps_3d_need = ps_3d
            boxsize_prod = np.prod(self.BoxSize, dtype=np.float32)
        
        if ps_3d_need is None:
            raise ValueError("mesh.complex_field(_gpu) is None. Please check if you have run the r2c or converted it to correct device.")
        
        deal_ps_3d(ps_3d_need, ps_3d_kernel=None, ps_3d_factor=boxsize_prod, shotnoise=shotnoise, nthreads=nthreads, c_api=c_api)
        self.removed_shotnoise = True # Avoid shotnoise being removed twice

        return self.cal_ps_from_3d(ps_3d, kmin, kmax, dk, Nmu=Nmu, k_arrays=k_arrays, mode=mode, k_logarithmic=k_logarithmic, nthreads=nthreads, c_api=c_api, test_mode=test_mode)

    def cal_ps_from_3d(
        self, ps_3d,
        kmin, kmax, dk, Nmu=None, k_arrays=None,
        mode="1d", k_logarithmic=False, nthreads=1, device_id=-1, c_api=False, test_mode=False
    ):
        if ps_3d is None:
            raise ValueError("mesh.complex_field(_gpu) is None. Please check if you have run the r2c or converted it to correct device.")
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

        if k_arrays is None:
            k_x_array = (
                np.fft.fftfreq(self.Nmesh[0], d=1.0)
                * 2.0
                * np.pi
                * self.Nmesh[0]
                / self.BoxSize[0]
            )
            k_y_array = (
                np.fft.fftfreq(self.Nmesh[1], d=1.0)
                * 2.0
                * np.pi
                * self.Nmesh[1]
                / self.BoxSize[1]
            )
            k_z_array = (
                np.fft.fftfreq(self.Nmesh[2], d=1.0)
                * 2.0
                * np.pi
                * self.Nmesh[2]
                / self.BoxSize[2]
            )[: ps_3d.shape[2]]
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
                power_k, power_mu, power, power_modes = cal_ps_from_numba(
                    ps_3d,
                    [k_x_array, k_y_array, k_z_array],
                    k_array,
                    mu_array,
                    k_logarithmic=k_logarithmic,
                    nthreads=nthreads,
                )
        
        if not self.removed_shotnoise:
            power -= self.attrs["shotnoise"]
        if test_mode:
            self.test_mode = True
            if mode == "2d":
                self.power_test = {"k": np.copy(power_k), "mu": np.copy(power_mu), "Pkmu": np.copy(np.real(power)), "Pkmu_C": np.copy(power), "modes": np.copy(power_modes)}
            else:
                self.power_test = {"k": np.copy(power_k).ravel(), "Pk": np.copy(np.real(power)).ravel(), "Pk_C": np.copy(power).ravel(), "modes": np.copy(power_modes).ravel()}
        else:
            self.test_mode = False

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
    
    def intergrate_fftpower(self, kmin=-1, kmax=-1, mu_min=-1, mu_max=-1, integrate="k", easy_mu_array=False, norm=False):
        power = self.power 
        k_array = np.nanmean(power["k"], axis=1)
        if easy_mu_array:
            mu_array_edges = np.linspace(0.0, 1.0, self.attrs["Nmu"]+1)
            mu_array = (mu_array_edges[:-1] + mu_array_edges[1:]) / 2.0
        else:
            mu_array = np.nanmean(power["mu"], axis=0)
        
        if kmin <= k_array[0]:
            k_min_index = 0
        else:
            k_min_index_source = np.where(k_array >= kmin)[0]
            if len(k_min_index_source) == 0:
                raise ValueError(f"kmin({kmin:.2f}) is too large")
            else:
                k_min_index = k_min_index_source[0]
        if kmax >= k_array[-1]:
            k_max_index = len(k_array)
        else:
            k_max_index_source = np.where(k_array >= kmax)[0]
            if len(k_max_index_source) == 0:
                raise ValueError("kmax({kmax:.2f}) is too small")
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
        if integrate == "k":
            Pkmu_integrate = np.nanmean(Pkmu_select, axis=0)
            if norm:
                Pkmu_integrate = Pkmu_integrate / np.nanmean(Pkmu_integrate)
            return mu_array[mu_min_index: mu_max_index], Pkmu_integrate
        elif integrate == "mu":
            Pkmu_integrate = np.nanmean(Pkmu_select, axis=1)
            if norm:
                Pkmu_integrate = Pkmu_integrate / np.nanmean(Pkmu_integrate)
            return k_array[k_min_index: k_max_index], Pkmu_integrate
        else:
            raise ValueError("integrate must be k or mu")
    
    def save(self, filename):
        import joblib

        save_dict = {
            "power": self.power,
            "attrs": self.attrs,
        }
        if self.test_mode:
            save_dict["power_test"] = self.power_test
        joblib.dump(save_dict, filename)

    @classmethod
    def load(cls, filename):
        import joblib

        load_dict = joblib.load(filename)
        self = FFTPower(
            load_dict["attrs"]["Nmesh"],
            load_dict["attrs"]["BoxSize"],
            load_dict["attrs"]["shotnoise"],
        )
        self.power = load_dict["power"]
        self.attrs = load_dict["attrs"]
        return self

