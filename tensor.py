import ctypes

# Load the shared library
lib = ctypes.CDLL('./libtensor.so')  # Replace with the actual path to your compiled .so file

class Tensor1d:
    def __init__(self, size_or_data=None, c_tensor=None, _prev=None):
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

        def _backward():
            pass

        self._backward = _backward
        self._inputs = []

        if _prev is not None:
            self._prev = _prev

        if not lib.verify_tensor(self.tensor):
            raise RuntimeError("Failed to create valid tensor")

    def __getitem__(self, key):
        if isinstance(key, slice):
            start = key.start if key.start is not None else 0
            stop = -1 if key.stop is None else key.stop
            step = 1 if key.step is None else key.step
            slice_ptr = lib.get_slice(self.tensor, start, stop, step)
            if not slice_ptr:
                raise IndexError("Invalid slice parameters")
            
            slice_size = lib.get_size(slice_ptr)
            buffer = (ctypes.c_float * slice_size)()
            lib.get_tensor_data(slice_ptr, buffer, slice_size)
            result = Tensor1d(size_or_data=list(buffer))
            if hasattr(self, '_backward'):
                result._backward = self._backward
            return result
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
            other = Tensor1d(size_or_data=[other] * lib.get_size(self.tensor))
            c_tensor = lib.add_tensor_to_tensor(self.tensor, other.tensor)
        elif isinstance(other, Tensor1d):
            c_tensor = lib.add_tensor_to_tensor(self.tensor, other.tensor)
        else:
            raise TypeError("Invalid type for addition")
        if not c_tensor:
            raise RuntimeError("Failed to add tensors")
        result = Tensor1d(c_tensor=c_tensor)
        def _backward():
            lib.backward(self.tensor, other.tensor, result.tensor)
        result._backward = _backward
        result._inputs = [self, other]

        return result
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor1d(size_or_data=[other] * lib.get_size(self.tensor))
            c_tensor = lib.mul_tensor_to_tensor(self.tensor, other.tensor)
        elif isinstance(other, Tensor1d):
            c_tensor = lib.mul_tensor_to_tensor(self.tensor, other.tensor)
        else:
            raise TypeError("Invalid type for multiplication")
        if not c_tensor:
            raise RuntimeError("Failed to multiply tensors")
        result = Tensor1d(c_tensor=c_tensor)
        def _backward():
            lib.backward(self.tensor, other.tensor, result.tensor)
        result._backward = _backward
        result._inputs = [self, other]       
        return result

    def backward(self):    
        if not hasattr(self, '_backward'):
            raise RuntimeError("This tensor does not have a backward function. Ensure it is part of a computational graph.")
        
        visited = set()
        queue = [self]

        while queue:
            tensor = queue.pop()
            if tensor in visited:
                continue
            visited.add(tensor)
            tensor._backward()
            print("Backward has been called")
            queue.extend(tensor._inputs)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __radd__(self, other):
        return self.__add__(other)

    def __repr__(self):
        return self.__str__()
    
    def __neg__(self):
        return self * -1
    
    def __len__(self):
        return lib.get_size(self.tensor)

    def __str__(self):
        result = lib.tensor_to_string(self.tensor)
        if not result:
            return "Invalid Tensor"
        return result.decode('utf-8')

    def __del__(self):
        if hasattr(self, 'tensor') and lib.verify_tensor(self.tensor):
            lib.free_tensor(self.tensor)

    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)
    
    def get_grad(self):
        c_tensor = lib.get_grad(self.tensor)
        if not c_tensor:
            raise RuntimeError("Failed to get the gradient of the tensor")
        return Tensor1d(c_tensor=c_tensor)

    @staticmethod
    def range_init(size):
        c_tensor = lib.tensor_arange(size)
        return Tensor1d(c_tensor=c_tensor)
    
    def sum(self):
        c_tensor = lib.tensor_sum(self.tensor) 
        if not c_tensor:
            raise RuntimeError("Failed to compute the sum of the tensor")
        result = Tensor1d(c_tensor=c_tensor)
        
        def _backward():
            grad = 1.0
            for i in range(lib.get_size(self.tensor)):
                lib.add_grad(self.tensor, i, grad)

        result._backward = _backward
        result._inputs = [self]
        return result
    
    @staticmethod
    def stack(tensors):
        if not tensors:
            raise ValueError("No tensors to stack")
        
        tensor_sizes = [lib.get_size(tensor.tensor) for tensor in tensors]
        if len(set(tensor_sizes)) != 1:
            raise ValueError("All tensors must have the same size to stack")

        num_tensors = len(tensors)
        tensor_size = num_tensors * tensor_sizes[0]
        stacked_tensor = lib.init_tensor(tensor_size, 1)
        
        for i, tensor in enumerate(tensors):
            value = lib.get_item(tensor.tensor, 0)
            lib.set_item(stacked_tensor, ctypes.c_float(value), i)
        
        result = Tensor1d(c_tensor=stacked_tensor)
        
        def _backward():
            for i, tensor in enumerate(tensors):
                grad = lib.get_item(result.tensor, 0)
                lib.add_grad(tensor.tensor, i, grad)
        
        result._backward = _backward
        result._inputs = tensors
        
        return result


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

lib.mul_tensor_to_tensor.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
lib.mul_tensor_to_tensor.restype = ctypes.c_void_p

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

lib.tensor_scalar_mul.argtypes = [ctypes.c_void_p, ctypes.c_float]
lib.tensor_scalar_mul.restype = ctypes.c_void_p

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

lib.backward_add.argtypes = [ctypes.c_void_p, ctypes.c_void_p,  ctypes.c_int, ctypes.c_void_p]
lib.backward_add.restype = ctypes.c_void_p

lib.backward_mul.argtypes = [ctypes.c_void_p, ctypes.c_void_p,  ctypes.c_int, ctypes.c_void_p]
lib.backward_mul.restype = ctypes.c_void_p

lib.backward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
lib.backward.restype = ctypes.c_void_p

lib.zero_grad.argtypes = [ctypes.c_void_p]
lib.zero_grad.restype = None

lib.tensor_sum.argtypes = [ctypes.c_void_p]
lib.tensor_sum.restype = ctypes.c_void_p

lib.add_grad.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_float]
lib.add_grad.restype = None

lib.get_grad.argtypes = [ctypes.c_void_p]
lib.get_grad.restype = ctypes.c_void_p