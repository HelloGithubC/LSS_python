import os, joblib 
import warnings

import numpy as np 

from .JIT.mesh import to_mesh_numba, do_compensation_from_numba, do_interlacing_from_numba

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
            "resampler": None,
            "compensated":None, 
            "interlaced": None
        }

        self.real_field = None 
        self.real_field_gpu = None 
        self.real_field_inverse = None
        self.real_field_inverse_gpu = None
        self.complex_field = None
        self.complex_field_gpu = None

    def to_mesh(self, pos, resampler="CIC", interlaced=False, weights=None, values=None, nthreads=1, device_id=-1, is_norm=False, field_extern=None, c_api=False) -> None:
        """ The function to run CIC
        pos: ndarray, shape=(nparticle, 3); or list of ndarray
        resampler: str, the resampler to use, can be "NGP", "CIC", "TSC , "PCS"
        interlaced: bool. If True, will do shift and set field as self.real_field_shift default.
        weight: ndarray, shape=(nparticle,); or list of ndarray
        values: ndarray, shape=(nparticle,); or list of ndarray

        nthreads: int, the number of threads to use. Only be valid when using C/C++ API
        device_id: int, the id of GPU device(if >= 0). Advice to set pos, weights and values as float32 type, because only the float32 array will be copied to GPU. (float64 will be converted to float32)

        c_api: bool, whether to use C/C++ API. Only support CPU
        """
        self.attrs["resampler"] = resampler
        self.attrs["interlaced"] = interlaced
        self.attrs["is_norm"] = is_norm
            
        if isinstance(pos, np.ndarray):
            pos_list = [pos,]
        elif isinstance(pos, list) or isinstance(pos, tuple):
            pos_list = list(pos)
        else:
            raise ValueError("pos must be a ndarray or list of ndarray")
        if weights is not None:
            if isinstance(weights, np.ndarray):
                weights_list = [weights,]
            elif isinstance(weights, list) or isinstance(weights, tuple):
                weights_list = list(weights)
            else:
                raise ValueError("weight must be a ndarray or list of ndarray")
        else:
            weights_list = [None,] * len(pos_list)

        if values is not None:
            if isinstance(values, np.ndarray):
                values_list = [values,]
            elif isinstance(values, list) or isinstance(values, tuple):
                values_list = list(values)
            else:
                raise ValueError("values must be a ndarray or list of ndarray")
        else:
            values_list = [None, ] * len(pos_list)

        field_dtype = pos_list[0].dtype

        if device_id >= 0:
            if c_api:
                raise ValueError("c_api is not valid when using GPU")
            import cupy as cp 
            from .cuda.mesh import to_mesh_from_cuda
            use_gpu = True 
            if field_extern is None:
                with cp.cuda.Device(device_id):
                    self.real_field_gpu = cp.zeros((self.Nmesh[0], self.Nmesh[1], self.Nmesh[2]), dtype=cp.float32)
            else:
                self.real_field_gpu = field_extern
            if interlaced:
                self.real1_gpu = cp.zeros((self.Nmesh[0], self.Nmesh[1], self.Nmesh[2]), dtype=cp.float32)
                self.real2_gpu = cp.zeros((self.Nmesh[0], self.Nmesh[1], self.Nmesh[2]), dtype=cp.float32)
        else:
            use_gpu = False 
            if field_extern is None:
                self.real_field = np.zeros((self.Nmesh[0], self.Nmesh[1], self.Nmesh[2]), dtype=field_dtype)
            else:
                self.real_field = field_extern
            if interlaced:
                self.real1 = np.zeros((self.Nmesh[0], self.Nmesh[1], self.Nmesh[2]), dtype=field_dtype)
                self.real2 = np.zeros((self.Nmesh[0], self.Nmesh[1], self.Nmesh[2]), dtype=field_dtype)
        
        N_total = 0 
        W_total = 0.0 
        W2_total = 0.0

        for i, pos_e in enumerate(pos_list):
            if pos_e.dtype != np.float32 and pos_e.dtype != np.float64:
                print(f"Waring:  pos is not float32 or float64 ({pos_e.dtype}). Now try converting to float32")
                pos_e = pos_e.astype(np.float32)
            weight_e = weights_list[i]
            if weight_e is not None:
                if weight_e.dtype != pos_e.dtype:
                    weight_e = weight_e.astype(pos_e.dtype)
            value_e = values_list[i]
            if value_e is not None:
                if value_e.dtype != pos_e.dtype:
                    value_e = value_e.astype(pos_e.dtype)
            
            N_total += pos_e.shape[0]
            if weight_e is not None or value_e is not None:
                if weight_e is None:
                    weight_temp = 1.0 
                    value_temp = value_e
                if value_e is None:
                    value_temp = 1.0
                    weight_temp = weight_e
                W_total += np.sum(weight_temp * value_temp)
                W2_total += np.suum(weight_temp**2 * value_temp**2)
            else:
                W_total = N_total 
                W2_total = N_total

            if use_gpu:
                with cp.cuda.Device(device_id):
                    pos_gpu = cp.asarray(pos_e, dtype=cp.float32)
                    if weight_e is not None:
                        weight_gpu = cp.asarray(weight_e, dtype=cp.float32)
                    else:
                        weight_gpu = None
                    if value_e is not None:
                        value_gpu = cp.asarray(value_e, dtype=cp.float32)
                    else:
                        value_gpu = None
                    if interlaced:
                        to_mesh_from_cuda(pos_gpu, weight_gpu, value_gpu, self.real1_gpu, cp.asarray(self.BoxSize, dtype=cp.float64), cp.asarray(self.Nmesh, dtype=cp.uint32), resampler=resampler, shift=0.0)
                        to_mesh_from_cuda(pos_gpu, weight_gpu, value_gpu, self.real2_gpu, cp.asarray(self.BoxSize, dtype=cp.float64), cp.asarray(self.Nmesh, dtype=cp.uint32), resampler=resampler, shift=0.5)
                    else:
                        to_mesh_from_cuda(pos_gpu, weight_gpu, value_gpu, self.real_field_gpu, cp.asarray(self.BoxSize, dtype=cp.float64), cp.asarray(self.Nmesh, dtype=cp.uint32), resampler=resampler, shift=0.0)
            else:
                if not c_api:
                    if interlaced:
                        to_mesh_numba(pos_e, weight_e, value_e, self.real1, self.BoxSize, self.Nmesh, resampler=resampler, shift=0.0)
                        to_mesh_numba(pos_e, weight_e, value_e, self.real2, self.BoxSize, self.Nmesh, resampler=resampler, shift=0.5)
                    else:
                        to_mesh_numba(pos_e, weight_e, value_e, self.real_field, self.BoxSize, self.Nmesh, resampler=resampler, shift=0.0)
                else:
                    from .CPP.mesh import to_mesh_c_api
                    if interlaced:
                        to_mesh_c_api(pos_e, self.BoxSize, self.Nmesh, self.real1, weight_e, value_e, resampler=resampler, shift=0.0, nthreads=nthreads)
                        to_mesh_c_api(pos_e, self.BoxSize, self.Nmesh, self.real2, weight_e, value_e, resampler=resampler, shift=0.5, nthreads=nthreads)
                    else:
                          to_mesh_c_api(pos_e, self.BoxSize, self.Nmesh, self.real_field, weight_e, value_e, resampler=resampler, shift=0.0, nthreads=nthreads)

        self.attrs["N"] = N_total 
        self.attrs["W"] = W_total
        self.attrs["W2"] = W2_total
        self.attrs["num_per_cell"] = (W_total / np.prod(self.Nmesh)).astype(field_dtype)
        self.attrs["shotnoise"] = (np.prod(self.BoxSize) * W2_total / W_total**2).astype(field_dtype)
        if is_norm:
            if use_gpu:
                if interlaced:
                    self.real_field_gpu[...] = (self.real1_gpu + self.real2_gpu) / 2.0 / self.attrs["num_per_cell"]
                else:
                    self.real_field_gpu /= self.attrs["num_per_cell"]
            else:
                if interlaced:
                    self.real_field[...] = (self.real1 + self.real2) / 2.0 / self.attrs["num_per_cell"]
                else:
                    self.real_field /= self.attrs["num_per_cell"]
        else:
            self.attrs["shotnoise"] *= self.attrs["num_per_cell"] ** 2
            if use_gpu:
                if interlaced:
                    self.real_field_gpu[...] = (self.real1_gpu + self.real2_gpu) / 2.0
            else:
                if interlaced:
                    self.real_field[...] = (self.real1 + self.real2) / 2.0

    def r2c(self, compensated=False, k_arrays_interlace=None, device_id=-1, nthreads=1, c_api=False) -> None:
        self.attrs["compensated"] = compensated
        if device_id >= 0:
            from cupyx.scipy.fft import rfftn
            import cupy as cp
            from .cuda.mesh import do_interlacing_from_cuda
            if self.real_field_gpu is None:
                raise ValueError('No real field to convert to complex field.')
            else:
                with cp.cuda.Device(device_id):
                    if self.attrs["interlaced"]:
                        complex1_gpu = rfftn(self.real1_gpu, norm="forward")
                        complex2_gpu = rfftn(self.real2_gpu, norm="forward")
                        if k_arrays_interlace is not None:
                            k_arrays_interlace_gpu = [cp.asarray(k_arrays_interlace[i], dtype=cp.float32) for i in range(len(k_arrays_interlace))]
                        do_interlacing_from_cuda(complex1_gpu, complex2_gpu, cp.asarray(self.BoxSize, dtype=cp.float32), cp.asarray(self.Nmesh, dtype=cp.int32), k_arrays_interlace_gpu)
                        if self.attrs["is_norm"]:
                            self.complex_field_gpu = self.complex_field_gpu / self.attrs["num_per_cell"]
                        else:
                            self.complex_field_gpu = complex1_gpu 
                    else:
                        self.complex_field_gpu = rfftn(self.real_field_gpu, norm="forward")
        else:
            from scipy.fft import rfftn
            if self.real_field is None:
                raise ValueError('No real field to convert to complex field.')
            else:
                if self.attrs["interlaced"]:
                    complex1 = rfftn(self.real1, workers=nthreads, norm="forward")
                    complex2 = rfftn(self.real2, workers=nthreads, norm="forward") 
                    if c_api:
                        from .CPP.mesh import do_interlacing_c_api
                        do_interlacing_c_api(complex1, complex2, self.BoxSize, self.Nmesh, k_arrays_interlace, nthreads) 
                    else:
                        do_interlacing_from_numba(complex1, complex2, self.BoxSize, self.Nmesh, k_arrays_interlace)
                    if self.attrs["is_norm"]:
                        self.complex_field = complex1 / self.attrs["num_per_cell"]
                    else:
                        self.complex_field = complex1
                else:
                    self.complex_field = rfftn(self.real_field, workers=nthreads, norm="forward")
        if compensated:
            self.do_compensation(device_id, nthreads, c_api)

    def c2r(self, device_id=-1):
        if device_id >= 0:
            from cupyx.scipy.fft import irfftn 
            import cupy as cp 
            if self.complex_field_gpu is None:
                raise ValueError('No complex field to convert to real field.')
            else:
                with cp.cuda.Device(device_id):
                    self.real_field_inverse_gpu = irfftn(self.complex_field_gpu, norm="forward")
        else:
            from scipy.fft import irfftn 
            self.real_field_inverse = irfftn(self.complex_field, norm="forward")

    def do_compensation(self, device_id=-1, nthreads=1, c_api=False):
        if device_id >= 0:
            import cupy as cp 
            from .cuda.mesh import do_compensation_from_cuda
            use_gpu = True 
            k_z_length = self.complex_field_gpu.shape[2]
        else:
            use_gpu = False 
            k_z_length = self.complex_field.shape[2]
        
        k_arrays = [
            np.fft.fftfreq(n=self.Nmesh[0], d=1.0)
            * 2.0
            * np.pi,
            np.fft.fftfreq(n=self.Nmesh[1], d=1.0)
            * 2.0
            * np.pi,
            np.fft.fftfreq(n=self.Nmesh[2], d=1.0)[
                : k_z_length
            ]
            * 2.0
            * np.pi,
        ]
        if use_gpu:
            if self.complex_field_gpu is None:
                raise ValueError('No complex field to do compensated summation.')
            with cp.cuda.Device(device_id):
                k_arrays_gpu = [cp.asarray(k_array, dtype=cp.float32) for k_array in k_arrays]
                do_compensation_from_cuda(self.complex_field_gpu, k_arrays_gpu, resampler=self.attrs["resampler"])
        else:
            if self.complex_field is None:
                raise ValueError('No complex field to do compensated summation.')
            if c_api:
                from .CPP.mesh import do_compensation_c_api
                do_compensation_c_api(self.complex_field, k_arrays, resampler=self.attrs["resampler"], interlaced=self.attrs["interlaced"], nthreads=nthreads)
            else:
                do_compensation_from_numba(self.complex_field, k_arrays, resampler=self.attrs["resampler"], interlace=self.attrs["interlaced"], nthreads=nthreads)

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
        joblib.dump(self.attrs, os.path.join(output_dir, "attrs_dict.pkl"))
        
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
    def load(cls, input_dir, only_real=False, only_complex=False):
        attrs_filename = os.path.join(input_dir, "attrs_dict.pkl")
        real_field_filename = os.path.join(input_dir, "real_field.npy")
        complex_field_filename = os.path.join(input_dir, "complex_field.npy")
        
        if not os.path.exists(attrs_filename) and not os.path.exists(real_field_filename) and not os.path.exists(complex_field_filename):
            raise FileNotFoundError("No attrs.pkl, real_field.npy or complex_field.npy found.")
        else:
            self = Mesh(Nmesh=512, BoxSize=1000.0) # Arbitrary values to initialize the object

            if os.path.exists(real_field_filename) and not only_complex:
                self.real_field = np.load(real_field_filename)
            else:
                self.real_field = None
            if os.path.exists(complex_field_filename) and not only_real:
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
                warnings.warn("No attrs.pkl found. The attrs will only contrain the Nmesh, if exists real field.")
            if only_complex:
                self.attrs["modes"] = ("complex", )
            elif only_real:
                self.attrs["modes"] = ("real", )
            else:
                self.attrs["modes"] = ("real", "complex")
                
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

