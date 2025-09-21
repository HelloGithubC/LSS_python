import numpy as np 
from numba import njit, set_num_threads, prange, get_thread_id

@njit(parallel=True)
def deal_ps_3d_multithreads(ps_3d, ps_3d_kernel=None, ps_3d_factor=1.0, shotnoise=0.0, nthreads=1):
    set_num_threads(nthreads)
    for ix in prange(ps_3d.shape[0]):
        for iy in prange(ps_3d.shape[1]):
            for iz in prange(ps_3d.shape[2]):
                kernel_element = ps_3d_kernel[ix, iy, iz] if ps_3d_kernel is not None else 1.0
                ps_3d[ix, iy, iz] *= kernel_element
                ps_3d[ix, iy, iz] = ps_3d[ix, iy, iz] * np.conj(ps_3d[ix, iy, iz]) * ps_3d_factor - shotnoise
    return

@njit
def deal_ps_3d_single(ps_3d, ps_3d_kernel=None, ps_3d_factor=1.0, shotnoise=0.0):
    for ix in range(ps_3d.shape[0]):
        for iy in range(ps_3d.shape[1]):
            for iz in range(ps_3d.shape[2]):
                kernel_element = ps_3d_kernel[ix, iy, iz] if ps_3d_kernel is not None else 1.0
                ps_3d[ix, iy, iz] *= kernel_element
                ps_3d[ix, iy, iz] = ps_3d[ix, iy, iz] * np.conj(ps_3d[ix, iy, iz]) * ps_3d_factor - shotnoise
    return

def deal_ps_3d(ps_3d, ps_3d_kernel=None, ps_3d_factor=1.0, shotnoise=0.0, nthreads=1, device_id=-1):
    if device_id >= 0:
        import cupy as cp
        from .cuda.fftpower import deal_ps_3d_from_cuda
        with cp.cuda.Device(device_id):
            deal_ps_3d_from_cuda(ps_3d, ps_3d_kernel, ps_3d_factor, shotnoise)
    else:
        if nthreads > 1:
            deal_ps_3d_multithreads(ps_3d, ps_3d_kernel, ps_3d_factor, shotnoise, nthreads)
        else:
            deal_ps_3d_single(ps_3d, ps_3d_kernel, ps_3d_factor, shotnoise)

@njit(parallel=True)
def run_core(ps_3d, k_arrays_list, k_array, mu_array, linear=True, nthreads=1):
    set_num_threads(nthreads)
    kx_array, ky_array, kz_array = k_arrays_list
    kbin = k_array.shape[0] - 1 
    mubin = mu_array.shape[0] - 1
    k_diff = k_array[1] - k_array[0]
    if mubin <= 1:
        use_mu = False
        mu_diff = 1.0
    else:
        use_mu = True
        mu_diff = mu_array[1] - mu_array[0]
    
    Pkmu_threads = np.zeros((nthreads, kbin, mubin), dtype=np.complex128)
    count_threads = np.zeros((nthreads, kbin, mubin), dtype=np.uint32)
    k_mesh_threads = np.zeros((nthreads, kbin, mubin), dtype=np.float64)
    mu_mesh_threads = np.zeros((nthreads, kbin, mubin), dtype=np.float64)

    kx_array = np.abs(kx_array)
    ky_array = np.abs(ky_array)
    for ix in prange(len(kx_array)):
        kx = kx_array[ix]
        for iy in prange(len(ky_array)):
            ky = ky_array[iy]
            for iz in prange(len(kz_array)):
                kz = kz_array[iz]
                if (kx <= k_array[0] / 2.0 and  ky <= k_array[0] / 2.0 and kz <= k_array[0] / 2.0) or \
                (kx >= k_array[-1] and ky >= k_array[-1] and kz >= k_array[-1]):
                    continue
                mode = 1 if iz == 0 else 2
                k = np.sqrt(kx**2 + ky**2 + kz**2)
                if k < k_array[0] or k > k_array[-1]:
                    continue
                if linear:
                    k_i = int((k - k_array[0]) / k_diff)
                else:
                    k_i = int(np.digitize(k, k_array) - 1)
                if k_i == kbin:
                    k_i -= 1 

                if use_mu:
                    mu = kz / k 
                    if mu < mu_array[0] or mu > mu_array[-1]:
                        continue
                    if linear:
                        mu_i = int((mu - mu_array[0]) / mu_diff)
                    else:
                        mu_i = int(np.digitize(mu, mu_array) - 1)
                    if mu_i == mubin:
                        mu_i -= 1
                else:
                    mu = 0.0
                    mu_i = 0 
                thread_id = get_thread_id()
                Pkmu_threads[thread_id, k_i, mu_i] += ps_3d[ix, iy, iz] * mode
                k_mesh_threads[thread_id, k_i, mu_i] += k * mode
                mu_mesh_threads[thread_id, k_i, mu_i] += mu * mode
                count_threads[thread_id, k_i, mu_i] += mode
    Pkmu = np.sum(Pkmu_threads, axis=0)
    k_mesh = np.sum(k_mesh_threads, axis=0)
    mu_mesh = np.sum(mu_mesh_threads, axis=0)
    count = np.sum(count_threads, axis=0)
    return k_mesh, mu_mesh, Pkmu, count 

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

    def run(
        self, ps_3d,
        kmin, kmax, dk, Nmu=None, k_arrays=None,
        mode="1d", linear=True, nthreads=1, device_id=-1
    ):
        if device_id >= 0:
            import cupy as cp
            from .cuda.fftpower import run_fftpower_from_cuda
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
                power_k, power_mu, power, power_modes = run_fftpower_from_cuda(
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
            power_k, power_mu, power, power_modes = run_core(
                ps_3d,
                [k_x_array, k_y_array, k_z_array],
                k_array,
                mu_array,
                linear=linear,
                nthreads=nthreads,
            )
        
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
    
    def save(self, filename):
        import joblib

        save_dict = {
            "power": self.power,
            "attrs": self.attrs,
        }
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

