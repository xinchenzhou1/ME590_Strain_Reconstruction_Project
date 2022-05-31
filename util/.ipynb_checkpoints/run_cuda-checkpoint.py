import numpy as np
from cuda import cuda

def get_pointer(arg):
    str_arg =str(type(arg))
    
    if str_arg == "<class 'numpy.ndarray'>":
        # print('numpy')
        out = arg.ctypes.data
        
    elif str_arg == "<class 'gpuarray.gpu_array'>":
        # print('gpu')
        out = arg.d_array.ctypes.data
        
    else:
        # print('tex')
        out = id(arg)
    return out

def run_cuda_function(cuda_fun,args,grid_dim,block_dim):


    args = np.array([get_pointer(arg) for arg in args],dtype=np.uint64)
    # print(args)
    err, = cuda.cuLaunchKernel(
       cuda_fun,
       grid_dim[0],  # grid x dim
       grid_dim[1],  # grid y dim
       grid_dim[2],  # grid z dim
       block_dim[0],  # block x dim
       block_dim[1],  # block y dim
       block_dim[2],  # block z dim
       0,  # dynamic shared memory
       0,  # stream
       args.ctypes.data,  # kernel arguments
       0,  # extra (ignore)
    )    
    return err