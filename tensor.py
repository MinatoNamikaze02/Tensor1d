import ctypes

# Load the shared library
lib = ctypes.CDLL('./libtensor.so')  # Replace with the actual path to your compiled .so file

class Tensor1d:
    def __init__(self, size_or_data=None, c_tensor=None):
        if c_tensor is not None:
            if not lib.verify_tensor(c_tensor):
                raise ValueError("Invalid C tensor pointer")
            self.tensor = c_tensor
            self._as_parameter_ = self.tensor  # For ctypes compatibility
        elif isinstance(size_or_data, int):
            self.tensor = lib.init_tensor(size_or_data, 1)
        elif isinstance(size_or_data, (list, range)):
            self.tensor = lib.init_tensor(len(size_or_data), 1)
            for i, val in enumerate(size_or_data):
                lib.set_item(self.tensor, ctypes.c_float(val), i)
        else:
            raise TypeError("Invalid initialization input")

        if not lib.verify_tensor(self.tensor):
            raise RuntimeError("Failed to create valid tensor")

    def __getitem__(self, key):
        if isinstance(key, slice):
            start = key.start if key.start is not None else 0
            stop = self.tensor.size if key.stop is None else key.stop
            step = 1 if key.step is None else key.step
            slice_ptr = lib.get_slice(self.tensor, start, stop, step)
            if not slice_ptr:
                raise IndexError("Invalid slice parameters")
            
            slice_size = lib.get_size(slice_ptr)
            buffer = (ctypes.c_float * slice_size)()
            lib.get_tensor_data(slice_ptr, buffer, slice_size)
            return Tensor1d(size_or_data=list(buffer))
        else:
            if key < 0:
                key += lib.get_size(self.tensor)
            if key < 0 or key >= lib.get_size(self.tensor):
                raise IndexError("Index out of range")
            return lib.get_item(self.tensor, key)

    def __setitem__(self, index, value):
        lib.set_item(self.tensor, ctypes.c_float(value), index)
        return lib.tensor_to_string(self.tensor).decode('utf-8')

    def __add__(self, other):
        if isinstance(other, (int, float)):
            c_tensor = lib.tensor_scalar_add(self.tensor, float(other))
        elif isinstance(other, Tensor1d):
            c_tensor = lib.add_tensor_to_tensor(self.tensor, other.tensor)
        else:
            raise TypeError("Invalid type for addition")
        if not c_tensor:
            raise RuntimeError("Failed to add tensors")
        return Tensor1d(c_tensor=c_tensor)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        result = lib.tensor_to_string(self.tensor)
        if not result:
            return "Invalid Tensor"
        return result.decode('utf-8')

    def __del__(self):
        if hasattr(self, 'tensor') and lib.verify_tensor(self.tensor):
            lib.free_tensor(self.tensor)

    @staticmethod
    def range_init(size):
        c_tensor = lib.tensor_arange(size)
        return Tensor1d(c_tensor=c_tensor)


lib.init_tensor.argtypes = [ctypes.c_int, ctypes.c_int]
lib.init_tensor.restype = ctypes.c_void_p

lib.get_tensor_data.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
lib.get_tensor_data.restype = None

lib.tensor_arange.argtypes = [ctypes.c_int]
lib.tensor_arange.restype = ctypes.c_void_p

lib.set_item.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_int]
lib.set_item.restype = None

lib.append_data.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.append_data.restype = None

lib.add_tensor_to_tensor.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
lib.add_tensor_to_tensor.restype = ctypes.c_void_p

lib.set_item.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_int]
lib.set_item.restype = None

lib.get_size.argtypes = [ctypes.c_void_p]
lib.get_size.restype = ctypes.c_int

lib.get_item_as_tensor.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.get_item_as_tensor.restype = ctypes.c_void_p

lib.get_item.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.get_item.restype = ctypes.c_float

lib.get_slice.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.get_slice.restype = ctypes.c_void_p

lib.free_tensor.argtypes = [ctypes.c_void_p]
lib.free_tensor.restype = None

lib.add_tensor_broadcast.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
lib.add_tensor_broadcast.restype = ctypes.c_void_p

lib.tensor_scalar_add.argtypes = [ctypes.c_void_p, ctypes.c_float]
lib.tensor_scalar_add.restype = ctypes.c_void_p

lib.print_tensor.argtypes = [ctypes.c_void_p]
lib.print_tensor.restype = None

lib.tensor_to_string.argtypes = [ctypes.c_void_p]
lib.tensor_to_string.restype = ctypes.c_char_p

lib.verify_tensor.argtypes = [ctypes.c_void_p]
lib.verify_tensor.restype = ctypes.c_bool

lib.get_tensor_data.argtypes = [ctypes.c_void_p, 
                               ctypes.POINTER(ctypes.c_float), 
                               ctypes.c_int]
lib.get_tensor_data.restype = None

if __name__ == "__main__":
    t = Tensor1d.range_init(20)
    print(t[3])
    t[-1] = 100.0
    print(t[-1]) 
    print(t)
    print(t[5:15:2])
    print(t[5:15:2][2:7])
    t = t + 10.0 
    print(t)
    t2 = Tensor1d.range_init(20)
    print(t2)
    t3 = t + t2
    t4 = (t + (Tensor1d.range_init(20) + 250))[2:10]
    print(t4)
