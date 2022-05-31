import numpy as np
from cuda import cuda, nvrtc






class gpu_array:
    
    def __init__(self,size,dtype,h_array=None):
        if h_array is not None:
            self.h_array = h_array
            err, self.d_class = cuda.cuMemAlloc(self.h_array.nbytes)
            err, cuda.cuMemcpyHtoD(self.d_class, self.h_array.ctypes.data, self.h_array.nbytes)
            
        else: 
            self.h_array = np.empty(size,dtype=dtype)
            err, self.d_class = cuda.cuMemAlloc(self.h_array.nbytes)
            
            
        self.d_array = np.array([int(self.d_class)], dtype=np.uint64)
    def get(self,free=True):
        err, = cuda.cuMemcpyDtoH(self.h_array.ctypes.data, self.d_class, self.h_array.nbytes)
        if free:
            err, = cuda.cuMemFree(self.d_class)
            
            
        return self.h_array

        
        
    
def empty(size,dtype):
    array = gpu_array(size,dtype)
    return array

def to_gpu(array):
    array = gpu_array(array.size,array.dtype,h_array=array)
    return array