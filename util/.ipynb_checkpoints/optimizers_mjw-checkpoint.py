# import pycuda.driver as cuda
import numpy as np
from cuda import cuda, nvrtc
import time
import gpuarray
from util.config import Config
from util.run_cuda import run_cuda_function


def CrossEntropyMethod(recon, x, y,
                       XD, YD, OffsetD, MaskD, TrueMaskD, scoreD, S_gpu,
                       NumD=10000, numCut=100, cov=1e-6 * np.eye(9), MaxIter=50, mean=np.eye(3), BlockSize=256,
                       debug=False):
    if not recon.ImLoaded:
        recon.loadIm()
    if not recon.GsLoaded:
        recon.loadGs()
    x = np.array(x).astype(np.float32)
    y = np.array(y).astype(np.float32)
    
    for ii in range(MaxIter):
        #np.random.seed(ii)
        S = np.random.multivariate_normal(
            np.zeros(9), cov, size=(NumD)).reshape((NumD, 3, 3), order='C') + np.tile(mean, (NumD, 1, 1))
        
        Sr = S.ravel().astype(np.float32)
        err, = cuda.cuMemcpyHtoD( S_gpu,Sr.ctypes.data, Sr.nbytes)


        # Sim Strain #######################################################################################        

        
        args = [XD, YD, OffsetD, MaskD, TrueMaskD,
                              x, y, recon.afDetInfoD, np.array([int(S_gpu)], dtype=np.uint64),
                              recon.whichOmegaD, np.array(NumD).astype(np.int32), np.array(recon.NumG).astype(np.int32),
                              np.array(recon.Cfg.energy).astype(np.float32), np.array(recon.Cfg.window[2]).astype(np.int32), recon.LimD, 
                              np.array(5).astype(np.int32), np.array(recon.Cfg.omgInterval).astype(np.float32),
                              recon.tG]
        
        
        err = run_cuda_function(recon.sim_strain_func,args,(NumD,1,1),(recon.NumG,1,1))

        # Hit Fun ##################################################################################################

        args = [scoreD,XD, YD, OffsetD, MaskD, TrueMaskD,
                       recon.MaxIntD, np.array(recon.NumG).astype(np.int32), np.array(NumD).astype(np.int32), recon.windowD,recon.tcExp]
            
        
        err = run_cuda_function(recon.hit_func,args,(int(NumD/BlockSize+1),1,1),(BlockSize,1,1))
        
        
        score = scoreD.get(free=False)

        
        args = np.argpartition(score, -numCut)[-numCut:]
        cov = np.cov(S[args].reshape((numCut, 9), order='C').T)
        mean = np.mean(S[args], axis=0)
        
        
        if debug:
            print(np.max(score))
        if np.trace(np.absolute(cov)) < 1e-8:
            
            break

    
    return cov, mean, np.max(score[args])


def ChangeOneVoxel_KL(recon, x, y, mean, realMapsLogD, falseMapsD,
                      XD, YD, OffsetD, MaskD, TrueMaskD, diffD, S_gpu,
                      NumD=10000, numCut=50, cov=1e-6 * np.eye(9), epsilon=1e-6, MaxIter=3, BlockSize=256, debug=False):
    if not recon.GsLoaded:
        recon.loadGs()
    # remove the original hit
    
    S = mean
    Sr = S.ravel().astype(np.float32)
    err, = cuda.cuMemcpyHtoD( S_gpu,Sr.ctypes.data, Sr.nbytes)
    
    
    
    x = np.array(x).astype(np.float32)
    y = np.array(y).astype(np.float32)

    
    # Sim Strain ####################################################################################
    args = [XD, YD, OffsetD, MaskD, TrueMaskD,
                          x, y, recon.afDetInfoD, np.array([int(S_gpu)], dtype=np.uint64),
                          recon.whichOmegaD, np.array(1).astype(np.int32), np.array(recon.NumG).astype(np.int32),
                          np.array(recon.Cfg.energy).astype(np.float32), np.array(recon.Cfg.window[2]).astype(np.int32), recon.LimD, 
                          np.array(5).astype(np.int32), np.array(recon.Cfg.omgInterval).astype(np.float32),
                          recon.tG]


          
          
          
    err = run_cuda_function(recon.sim_strain_func,args,(1,1,1),(recon.NumG,1,1))


    # OneFun ###################################################################################
    
    args = [XD, YD, OffsetD, MaskD, TrueMaskD,
                   falseMapsD, np.array(recon.NumG).astype(np.int32),
                   np.array(epsilon).astype(np.float32), np.array(-1).astype(np.int32),recon.windowD]
    
    
    err = run_cuda_function(recon.One_func,args,(1,1,1),(recon.NumG,1,1))
   
    
    # find a better distortion matrix
    
    for ii in range(MaxIter):
        
        S = np.empty((NumD, 3, 3), dtype=np.float32)
        S[0, :, :] = mean
        #np.random.seed(42)
        S[1:, :, :] = np.random.multivariate_normal(
            mean.ravel(), cov, size=(NumD - 1)).reshape((NumD - 1, 3, 3), order='C')
        Sr = S.ravel().astype(np.float32)
        err, = cuda.cuMemcpyHtoD( S_gpu,Sr.ctypes.data, Sr.nbytes)
        
        # Sim Strain #####################################################################
        args = [XD, YD, OffsetD, MaskD, TrueMaskD,
                              x, y, recon.afDetInfoD, np.array([int(S_gpu)], dtype=np.uint64),
                              recon.whichOmegaD, np.array(NumD).astype(np.int32), np.array(recon.NumG).astype(np.int32),
                              np.array(recon.Cfg.energy).astype(np.float32), np.array(recon.Cfg.window[2]).astype(np.int32), recon.LimD, 
                              np.array(5).astype(np.int32), np.array(recon.Cfg.omgInterval).astype(np.float32),
                              recon.tG]

        
        err = run_cuda_function(recon.sim_strain_func,args,(NumD,1,1),(recon.NumG,1,1))
        # print(OffsetD.get())
        # KL ###################################################################################
        args = [diffD,XD, YD, OffsetD, MaskD, TrueMaskD,
                           realMapsLogD, falseMapsD,
                           np.array(recon.NumG).astype(np.int32), np.array(NumD).astype(np.int32),recon.windowD]
        
        err = run_cuda_function(recon.KL_diff_func,args,(int(NumD / BlockSize + 1),1,1),(BlockSize,1,1))       
        diffH = diffD.get(free=False)
        
        args = np.argpartition(diffH, numCut)[:numCut]
        
        cov = np.cov(S[args].reshape((numCut, 9), order='C').T)
        mean = np.mean(S[args], axis=0)
        
        
        if ii == 0:
            diff_init = diffH[0]
        if debug:
            print(np.min(diffH), diffH[0])
    # add the new hit
    S = mean
    Sr = S.ravel().astype(np.float32)
    err, = cuda.cuMemcpyHtoD( S_gpu,Sr.ctypes.data, Sr.nbytes)
    

    
    #Sim Strain #######################################################################
    args = [XD, YD, OffsetD, MaskD, TrueMaskD,
                          x, y, recon.afDetInfoD, np.array([int(S_gpu)], dtype=np.uint64),
                          recon.whichOmegaD, np.array(1).astype(np.int32), np.array(recon.NumG).astype(np.int32),
                          np.array(recon.Cfg.energy).astype(np.float32), np.array(recon.Cfg.window[2]).astype(np.int32), recon.LimD, 
                          np.array(5).astype(np.int32), np.array(recon.Cfg.omgInterval).astype(np.float32),
                          recon.tG]
    
    err = run_cuda_function(recon.sim_strain_func,args,(1,1,1),(recon.NumG,1,1))
    
    # KL ####################################################################################
    args = [diffD,XD, YD, OffsetD, MaskD, TrueMaskD,
                       realMapsLogD, falseMapsD,
                       np.array(recon.NumG).astype(np.int32), np.array(1).astype(np.int32),recon.windowD]

    err = run_cuda_function(recon.KL_diff_func,args,(int(NumD / BlockSize + 1),1,1),(BlockSize,1,1))   
    diffH = diffD.get(free=False)
    
    #One Fun ######################################################################################
    args = [XD, YD, OffsetD, MaskD, TrueMaskD,
                   falseMapsD, np.array(recon.NumG).astype(np.int32), 
                   np.array(epsilon).astype(np.float32), np.array(+1).astype(np.int32),recon.windowD]
    err = run_cuda_function(recon.One_func,args,(1,1,1),(recon.NumG,1,1))
    
    return cov, mean, diffH[0] - diff_init


