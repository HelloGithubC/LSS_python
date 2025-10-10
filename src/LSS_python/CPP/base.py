import ctypes

class c_float_complex(ctypes.Structure):
    _fields_ = [("real", ctypes.c_float), ("imag", ctypes.c_float)]
    @property
    def value(self):
        return self.real + 1j * self.imag
    
class c_double_complex(ctypes.Structure):
    _fields_ = [("real", ctypes.c_double), ("imag", ctypes.c_double)]
    @property
    def value(self):
        return self.real + 1j * self.imag