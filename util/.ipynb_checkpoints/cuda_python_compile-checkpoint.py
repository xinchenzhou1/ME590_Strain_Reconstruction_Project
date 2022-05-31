from cuda import cuda, nvrtc
import numpy as np
def create_module(filename):
    with open(filename, 'r') as cudaSrc:
        src = cudaSrc.read()

    err, prog = nvrtc.nvrtcCreateProgram(str.encode(src), filename[:-3].encode(), 0,[],[])
    
    opts = []
    err, = nvrtc.nvrtcCompileProgram(prog, len(opts),opts)
    err, ptxSize = nvrtc.nvrtcGetPTXSize(prog)
    ptx = b" " * ptxSize
    err, = nvrtc.nvrtcGetPTX(prog, ptx)
    err, = cuda.cuInit(0)

    err, cuDevice = cuda.cuDeviceGet(0)
    err, context = cuda.cuCtxCreate(0, cuDevice)
    ptx = np.char.array(ptx)
    err, module = cuda.cuModuleLoadData(ptx.ctypes.data)
    
    return module