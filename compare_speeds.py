import time
import numpy as np
from tensor import Tensor1d

def compare_speeds():
    size = 10**6

    start = time.time()
    tensor = Tensor1d.arange(size)
    end = time.time()
    print(f"Tensor1d initialization: {end - start:.6f} seconds")

    start = time.time()
    py_list = list(range(size))
    end = time.time()
    print(f"Python list initialization: {end - start:.6f} seconds")

    start = time.time()
    np_array = np.arange(size)
    end = time.time()
    print(f"NumPy array initialization: {end - start:.6f} seconds")

    scalar = 10
    start = time.time()
    _ = tensor + scalar
    end = time.time()
    print(f"Tensor1d scalar addition: {end - start:.6f} seconds")

    start = time.time()
    _ = [x + scalar for x in py_list]
    end = time.time()
    print(f"Python list scalar addition: {end - start:.6f} seconds")

    start = time.time()
    _ = np_array + scalar
    end = time.time()
    print(f"NumPy array scalar addition: {end - start:.6f} seconds")

    slice_start, slice_end, slice_step = 100, 1000, 2
    start = time.time()
    _ = tensor[slice_start:slice_end:slice_step]
    end = time.time()
    print(f"Tensor1d slicing: {end - start:.6f} seconds")

    start = time.time()
    _ = py_list[slice_start:slice_end:slice_step]
    end = time.time()
    print(f"Python list slicing: {end - start:.6f} seconds")

    start = time.time()
    _ = np_array[slice_start:slice_end:slice_step]
    end = time.time()
    print(f"NumPy array slicing: {end - start:.6f} seconds")

if __name__ == "__main__":
    compare_speeds()
