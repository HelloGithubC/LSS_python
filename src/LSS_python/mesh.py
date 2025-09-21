import os, joblib 
import warnings

import numpy as np 
from numba import njit, prange, get_thread_id, set_num_threads

@njit 
def run_cic_single(pos, weight, field, BoxSize_array, Nmesh_array):
    if weight is None:
        use_weight = False
    else:
        use_weight = True
    nparticle = pos.shape[0]
    pos_i = np.zeros(3, dtype=np.uint32)
    for i in range(nparticle):
        sub_BoxSize = BoxSize_array / Nmesh_array
        pos_temp = np.copy(pos[i])
        
        diff_ratio_temp = np.zeros((3,2), dtype=np.float64)
        for j in range(3):
            if pos_temp[j] < 0.0 or pos_temp[j] > BoxSize_array[j]:
                continue
            pos_i[j] = np.uint32(pos_temp[j] / sub_BoxSize[j])
            if pos_i[j] == Nmesh_array[j]:
                pos_i[j] = 0
                pos_temp[j] = 0.0 
            
            diff_ratio_temp[j,0] = (pos_temp[j] - pos_i[j] * sub_BoxSize[j]) / sub_BoxSize[j]
            diff_ratio_temp[j,1] = 1.0 - diff_ratio_temp[j,0]

        x_i, y_i, z_i = pos_i 
        x_i_next = x_i + 1 if x_i < Nmesh_array[0] - 1 else 0
        y_i_next = y_i + 1 if y_i < Nmesh_array[1] - 1 else 0
        z_i_next = z_i + 1 if z_i < Nmesh_array[2] - 1 else 0

        weight_temp = weight[i] if use_weight else 1.0

        field[x_i, y_i, z_i] += (diff_ratio_temp[0,1] * diff_ratio_temp[1,1] * diff_ratio_temp[2,1]) * weight_temp
        field[x_i, y_i, z_i_next] += (diff_ratio_temp[0,1] * diff_ratio_temp[1,1] * diff_ratio_temp[2,0]) * weight_temp
        field[x_i, y_i_next, z_i] += (diff_ratio_temp[0,1] * diff_ratio_temp[1,0] * diff_ratio_temp[2,1]) * weight_temp
        field[x_i, y_i_next, z_i_next] += (diff_ratio_temp[0,1] * diff_ratio_temp[1,0] * diff_ratio_temp[2,0]) * weight_temp

        field[x_i_next, y_i, z_i] += (diff_ratio_temp[0,0] * diff_ratio_temp[1,1] * diff_ratio_temp[2,1]) * weight_temp
        field[x_i_next, y_i, z_i_next] += (diff_ratio_temp[0,0] * diff_ratio_temp[1,1] * diff_ratio_temp[2,0]) * weight_temp
        field[x_i_next, y_i_next, z_i] += (diff_ratio_temp[0,0] * diff_ratio_temp[1,0] * diff_ratio_temp[2,1]) * weight_temp
        field[x_i_next, y_i_next, z_i_next] += (diff_ratio_temp[0,0] * diff_ratio_temp[1,0] * diff_ratio_temp[2,0]) * weight_temp

@njit(parallel=True)
def run_cic_multithreads(pos, weight, field, BoxSize_array, Nmesh_array,nthreads=2):
    if weight is None:
        use_weight = False
    else:
        use_weight = True
    NDIM = 3
    if nthreads < 2:
        raise ValueError("nthreads must be greater than 1. Or you can call run_cic_single")
    else:
        if field.shape[2] < nthreads:
            raise ValueError("nthreads must be less than Nmesh. Or you can call run_cic_single")
        nparticle = pos.shape[0]
        batch_size = int(nparticle / nthreads)
        rest_particle = nparticle - batch_size * nthreads
        set_num_threads(nthreads)
        sub_BoxSize = BoxSize_array / Nmesh_array
        for _ in prange(nthreads):
            thread_id = get_thread_id()
            if thread_id < rest_particle:
                index_start = thread_id * (batch_size + 1)
                index_end = index_start + batch_size + 1
            else:
                index_start = thread_id * batch_size + rest_particle
                index_end = index_start + batch_size
            for i in range(len(pos)):
                pos_temp = np.copy(pos[i])
                for j in range(NDIM):
                    if pos_temp[j] < 0.0 or pos_temp[j] > BoxSize_array[j]:
                        continue
                z_i = np.int64(pos_temp[2] / sub_BoxSize[2])
                if z_i == Nmesh_array[2]:
                    z_i = 0
                    pos_temp[2] = 0.0
                z_i_next = z_i + 1 if z_i < Nmesh_array[2] - 1 else 0
                if (z_i < index_start or z_i >= index_end) and (z_i_next < index_start or z_i_next >= index_end):
                    continue
                else:
                    diff_ratio_temp = np.zeros((NDIM,2), dtype=np.float64)
                    x_i = np.int64(pos_temp[0] / sub_BoxSize[0])
                    if x_i == Nmesh_array[0]:
                        x_i = 0
                        pos_temp[0] = 0.0
                    x_i_next = x_i + 1 if x_i < Nmesh_array[0] - 1 else 0
                    y_i = np.int64(pos_temp[1] / sub_BoxSize[1])
                    if y_i == Nmesh_array[1]:
                        y_i = 0 
                        pos_temp[1] = 0.0
                    y_i_next = y_i + 1 if y_i < Nmesh_array[1] - 1 else 0

                    pos_i = np.array([x_i, y_i, z_i], dtype=np.int64)

                    for j in range(NDIM):
                        diff_ratio_temp[j, 0] = (pos_temp[j] - pos_i[j] * sub_BoxSize[j]) / sub_BoxSize[j]
                        diff_ratio_temp[j, 1] = 1.0 - diff_ratio_temp[j, 0]

                    weight_temp = weight[i] if use_weight else 1.0

                    if index_start - 1 <= z_i < index_end - 1:
                        field[x_i, y_i, z_i] += (diff_ratio_temp[0,1] * diff_ratio_temp[1,1] * diff_ratio_temp[2,1]) * weight_temp
                        field[x_i, y_i_next, z_i] += (diff_ratio_temp[0,1] * diff_ratio_temp[1,0] * diff_ratio_temp[2,1]) * weight_temp
                        field[x_i_next, y_i, z_i] += (diff_ratio_temp[0,0] * diff_ratio_temp[1,1] * diff_ratio_temp[2,1]) * weight_temp
                        field[x_i_next, y_i_next, z_i] += (diff_ratio_temp[0,0] * diff_ratio_temp[1,0] * diff_ratio_temp[2,1]) * weight_temp

                        field[x_i, y_i, z_i_next] += (diff_ratio_temp[0,1] * diff_ratio_temp[1,1] * diff_ratio_temp[2,0]) * weight_temp
                        field[x_i, y_i_next, z_i_next] += (diff_ratio_temp[0,1] * diff_ratio_temp[1,0] * diff_ratio_temp[2,0]) * weight_temp
                        field[x_i_next, y_i, z_i_next] += (diff_ratio_temp[0,0] * diff_ratio_temp[1,1] * diff_ratio_temp[2,0]) * weight_temp
                        field[x_i_next, y_i_next, z_i_next] += (diff_ratio_temp[0,0] * diff_ratio_temp[1,0] * diff_ratio_temp[2,0]) * weight_temp
                    elif z_i == index_end - 1:
                        field[x_i, y_i, z_i] += (diff_ratio_temp[0,1] * diff_ratio_temp[1,1] * diff_ratio_temp[2,1]) * weight_temp
                        field[x_i, y_i_next, z_i] += (diff_ratio_temp[0,1] * diff_ratio_temp[1,0] * diff_ratio_temp[2,1]) * weight_temp
                        field[x_i_next, y_i, z_i] += (diff_ratio_temp[0,0] * diff_ratio_temp[1,1] * diff_ratio_temp[2,1]) * weight_temp
                        field[x_i_next, y_i_next, z_i] += (diff_ratio_temp[0,0] * diff_ratio_temp[1,0] * diff_ratio_temp[2,1]) * weight_temp
                    else:
                        continue 

@njit(parallel=True)
def do_compensated(complex_field, k_arrays, ntreads=1):
    set_num_threads(ntreads)
    k_x_array, k_y_array, k_z_array = k_arrays
    for i in prange(complex_field.shape[0]):
        k_x = k_x_array[i]
        w_x = np.sqrt(1.0 - 2.0/3.0 * np.sin(k_x/2.0)**2)
        for j in range(complex_field.shape[1]):
            k_y = k_y_array[j]
            w_y = np.sqrt(1.0 - 2.0/3.0 * np.sin(k_y/2.0)**2)
            for k in range(complex_field.shape[2]):
                k_z = k_z_array[k]
                w_z = np.sqrt(1.0 - 2.0/3.0 * np.sin(k_z/2.0)**2)
                complex_field[i, j, k] /= w_x * w_y * w_z
                

class Mesh:
    def __init__(self, Nmesh, BoxSize):
        if isinstance(BoxSize, float):
            self.BoxSize = np.array([BoxSize, BoxSize, BoxSize])
        else:
            self.BoxSize = np.array(BoxSize)
        if isinstance(Nmesh, int) or isinstance(Nmesh, float):
            self.Nmesh = np.array([Nmesh, Nmesh, Nmesh], dtype=int)
        else:
            self.Nmesh = np.array(Nmesh, dtype=int)

        self.attrs = {
            "Nmesh": self.Nmesh, 
            "BoxSize": self.BoxSize,
        }

        self.real_field = None 
        self.real_field_gpu = None 
        self.complex_field = None
        self.complex_field_gpu = None

    def run_cic(self, pos, weight, nthreads=1, device_id=-1, is_norm=False, field_extern=None):
        """ The function to run CIC
        pos: ndarray, shape=(nparticle, 3); or list of ndarray
        weight: ndarray, shape=(nparticle,); or list of ndarray

        """
        if device_id >= 0:
            import cupy as cp 
            from .cuda.mesh import run_cic_from_cuda
            use_gpu = True 
            if field_extern is None:
                with cp.cuda.Device(device_id):
                    self.real_field_gpu = cp.zeros((self.Nmesh[0], self.Nmesh[1], self.Nmesh[2]), dtype=cp.float32)
            else:
                self.real_field_gpu = field_extern
        else:
            use_gpu = False 
            if field_extern is None:
                self.real_field = np.zeros((self.Nmesh[0], self.Nmesh[1], self.Nmesh[2]), dtype=np.float32)
            else:
                self.real_field = field_extern
            
        if isinstance(pos, np.ndarray):
            pos_list = [pos,]
        elif isinstance(pos, list) or isinstance(pos, tuple):
            pos_list = list(pos)
        else:
            raise ValueError("pos must be a ndarray or list of ndarray")
        if weight is not None:
            if isinstance(weight, np.ndarray):
                weight_list = [weight,]
            elif isinstance(weight, list) or isinstance(weight, tuple):
                weight_list = list(weight)
            else:
                raise ValueError("weight must be a ndarray or list of ndarray")
            use_weight = True
        else:
            use_weight = False
        
        N_total = 0 
        W_total = 0.0 
        W2_total = 0.0 

        for i, pos_e in enumerate(pos_list):
            if pos_e.dtype != np.float32:
                pos_e = pos_e.astype(np.float32)
            if use_weight:
                weight_e = weight_list[i]
                if weight_e.dtype != np.float32:
                    weight_e = weight_e.astype(np.float32)
            else:
                weight_e = None

            N_total += pos_e.shape[0]
            W_total += np.sum(weight_e) if weight_e is not None else N_total
            W2_total += np.sum(weight_e**2) if weight_e is not None else N_total

            if use_gpu:
                with cp.cuda.Device(device_id):
                    pos_gpu = cp.asarray(pos_e, dtype=cp.float32)
                    if use_weight:
                        weight_gpu = cp.asarray(weight_e, dtype=cp.float32)
                    else:
                        weight_gpu = None
                    run_cic_from_cuda(pos_gpu, weight_gpu, self.real_field_gpu, cp.asarray(self.BoxSize, dtype=cp.float64), cp.asarray(self.Nmesh, dtype=cp.int32))
            else:
                if nthreads == 1:
                    run_cic_single(pos_e, weight_e, self.real_field, self.BoxSize, self.Nmesh)
                else:
                    run_cic_multithreads(pos_e, weight_e, self.real_field, self.BoxSize, self.Nmesh, nthreads)
        self.attrs["N"] = N_total 
        self.attrs["W"] = W_total
        self.attrs["W2"] = W2_total
        self.attrs["num_per_cell"] = W_total / np.prod(self.Nmesh)
        self.attrs["shotnoise"] = np.prod(self.BoxSize) * W2_total / W_total**2
        if is_norm:
            if use_gpu:
                self.real_field_gpu /= self.attrs["num_per_cell"]
            else:
                self.real_field /= self.attrs["num_per_cell"]

    def r2c(self, compensated=False, k_arrays=None, device_id=-1, nthreads=1):
        if device_id >= 0:
            from cupyx.scipy.fft import rfftn
            import cupy as cp
            if self.real_field_gpu is None:
                raise ValueError('No real field to convert to complex field.')
            else:
                with cp.cuda.Device(device_id):
                    self.complex_field_gpu = rfftn(self.real_field_gpu) / cp.prod(cp.asarray(self.Nmesh, dtype=cp.float32))
        else:
            from scipy.fft import rfftn
            if self.real_field is None:
                raise ValueError('No real field to convert to complex field.')
            else:
                self.complex_field = rfftn(self.real_field) / self.Nmesh.astype(np.float32).prod()
        if compensated:
            self.do_compensated(k_arrays, device_id, nthreads)

    def c2r(self, device_id=-1):
        if device_id >= 0:
            from cupyx.scipy.fft import ifftn 
            import cupy as cp 
            if self.complex_field_gpu is None:
                raise ValueError('No complex field to convert to real field.')
            else:
                with cp.cuda.Device(device_id):
                    self.real_field_gpu = ifftn(self.complex_field_gpu) * cp.prod(self.Nmesh.astype(cp.float32))
        else:
            from scipy.fft import ifftn 
            self.real_field = ifftn(self.complex_field) * self.Nmesh.astype(np.float32).prod()

    def do_compensated(self, k_arrays, device_id=-1, nthreads=1):
        if device_id >= 0:
            import cupy as cp 
            from .cuda.mesh import do_compensated_from_cuda
            use_gpu = True 
            k_z_length = self.complex_field_gpu.shape[2]
        else:
            use_gpu = False 
            k_z_length = self.complex_field.shape[2]
        
        if k_arrays is None:
            k_arrays = [
            np.fft.fftfreq(n=self.Nmesh[0], d=1.0).astype(np.float32)
            * 2.0
            * np.pi,
            np.fft.fftfreq(n=self.Nmesh[1], d=1.0).astype(np.float32)
            * 2.0
            * np.pi,
            np.fft.fftfreq(n=self.Nmesh[2], d=1.0).astype(np.float32)[
                : k_z_length
            ]
            * 2.0
            * np.pi,
        ]
        if use_gpu:
            with cp.cuda.Device(device_id):
                k_arrays_gpu = [cp.asarray(k_array, dtype=cp.float32) for k_array in k_arrays]
                do_compensated_from_cuda(self.complex_field_gpu, k_arrays_gpu)
        else:
            do_compensated(self.complex_field, k_arrays, nthreads)

    def save(self, output_dir, mode="all"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if mode == "all":
            modes = ("real", "complex")
        elif mode == "real":
            modes = ("real", )
        elif mode == "complex":
            modes = ("complex", )
        else:
            raise ValueError("mode must be 'all', 'real' or 'complex'.")
        
        self.attrs["modes"] = modes
        joblib.dump(self.attrs, os.path.join(output_dir, "attrs.pkl"))
        
        for mode in modes:
            if mode == "real":
                if self.real_field is None and self.real_field_gpu is None:
                    raise ValueError("No real field to save.")
                elif self.real_field is not None:
                    np.save(os.path.join(output_dir, "real_field.npy"), self.real_field)
                else:
                    import cupy as cp
                    np.save(os.path.join(output_dir, "real_field.npy"), cp.asnumpy(self.real_field_gpu))
            else:
                if self.complex_field is None and self.complex_field_gpu is None:
                    raise ValueError("No complex field to save.")
                elif self.complex_field is not None:
                    np.save(os.path.join(output_dir, "complex_field.npy"), self.complex_field)
                else:
                    import cupy as cp
                    np.save(os.path.join(output_dir, "complex_field.npy"), cp.asnumpy(self.real_field_gpu))

    @classmethod
    def load(cls, input_dir):
        attrs_filename = os.path.join(input_dir, "attrs_dict.pkl")
        real_field_filename = os.path.join(input_dir, "real_field.npy")
        complex_field_filename = os.path.join(input_dir, "complex_field.npy")
        
        if not os.path.exists(attrs_filename) and not os.path.exists(real_field_filename) and not os.path.exists(complex_field_filename):
            raise FileNotFoundError("No attrs.pkl, real_field.npy or complex_field.npy found.")
        else:
            self = Mesh(Nmesh=512, BoxSize=1000.0) # Arbitrary values to initialize the object

            if os.path.exists(real_field_filename):
                self.real_field = np.load(real_field_filename)
            else:
                self.real_field = None
            if os.path.exists(complex_field_filename):
                self.complex_field = np.load(complex_field_filename)
            else:
                self.complex_field = None
                
            if os.path.exists(attrs_filename):
                attrs = joblib.load(attrs_filename)
                self.attrs.update(attrs)
            else:
                self.attrs["BoxSize"] = None 
                self.BoxSize = None 
                if self.real_field is not None:
                    self.attrs["Nmesh"] = self.real_field.shape
                    self.Nmesh = self.real_field.shape
                else:
                    self.attrs["Nmesh"] = None
                    self.Nmesh = None
                raise warnings.warn("No attrs.pkl found. The attrs will only contrain the Nmesh, if exists real field.")
                
        return self 
    
    def update_attrs(self, new_attrs):
        self.attrs.update(new_attrs)
        if "BoxSize" in new_attrs:
            self.BoxSize = new_attrs["BoxSize"]
        if "Nmesh" in new_attrs:
            self.Nmesh = new_attrs["Nmesh"]
        
    def transform_field(self, source_device_id, target_device_id):
        if source_device_id < 0 and target_device_id < 0:
            raise ValueError("Both source_device_id and target_device_id are negative.")
        import cupy as cp
        if source_device_id < 0 and target_device_id >= 0:
            with cp.cuda.Device(target_device_id):
                if self.real_field is not None:
                    self.real_field_gpu = cp.asarray(self.real_field)
                if self.complex_field is not None:
                    self.complex_field_gpu = cp.asarray(self.complex_field)
        elif source_device_id >= 0 and target_device_id >= 0:
            with cp.cuda.Device(target_device_id):
                if self.real_field_gpu is not None:
                    self.real_field_gpu = cp.array(self.real_field_gpu)
                if self.complex_field_gpu is not None:
                    self.complex_field_gpu = cp.array(self.complex_field_gpu)
        else:
            with cp.cuda.Device(source_device_id):
                if self.real_field_gpu is not None:
                    self.real_field = cp.asnumpy(self.real_field_gpu)
                if self.complex_field_gpu is not None:
                    self.complex_field = cp.asnumpy(self.complex_field_gpu)

