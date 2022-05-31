from cuda import cuda, nvrtc
import numpy as np
import util.RotRep as Rot
import util.Simulation as Gsim
from util.config import Config
import h5py
import time
import gpuarray
from collections import Counter
from scipy.sparse import coo_matrix
from scipy.ndimage import gaussian_filter
from util.run_cuda import run_cuda_function
from util.cuda_python_compile import create_module


class Initializer:

    def __init__(self, Cfg,grain):
        
        self.grain = grain
        self.module = create_module('strain_device_mjw_3.cu')
        
               
        self.sim_strain_func = cuda.cuModuleGetFunction(self.module, b'Simulate_for_Strain')[1]
        self.sim_pos_func = cuda.cuModuleGetFunction(self.module, b'Simulate_for_Pos')[1]
        self.KL_total_func = cuda.cuModuleGetFunction(self.module, b'KL_total')[1]
        self.KL_diff_func = cuda.cuModuleGetFunction(self.module, b'KL_diff')[1]
        self.One_func = cuda.cuModuleGetFunction(self.module, b'ChangeOne')[1]
        self.hit_func = cuda.cuModuleGetFunction(self.module, b'Hit_Score')[1]
        self.Cfg = Cfg
        self.mode = Cfg.mode
        self.ImLoaded = False
        self.GsLoaded = False
        self.GsGenerated = False
        self.windowD = gpuarray.to_gpu(np.array(self.Cfg.window).astype(np.int32))
        # Det parameters
        self.Det = Gsim.Detector(psizeJ=Cfg.pixelSize / 1000.0,
                                 psizeK=Cfg.pixelSize / 1000.0,
                                 J=Cfg.JCenter,
                                 K=Cfg.KCenter,
                                 trans=np.array([Cfg.Ldistance, 0, 0]),
                                 tilt=Rot.EulerZXZ2Mat(np.array(Cfg.tilt) / 180.0 * np.pi))
        afDetInfoH = np.concatenate(
            [[Cfg.JPixelNum, Cfg.KPixelNum, Cfg.pixelSize / 1000.0, Cfg.pixelSize / 1000.0],
             self.Det.CoordOrigin,
             self.Det.Norm,
             self.Det.Jvector,
             self.Det.Kvector]).astype(np.float32)
        self.afDetInfoD = gpuarray.to_gpu(afDetInfoH)
        # sample parameters 
        
        self.sample = Gsim.CrystalStr()
        self.sample.PrimA = Cfg.lattice[0] * np.array(Cfg.basis[0])
        self.sample.PrimB = Cfg.lattice[1] * np.array(Cfg.basis[1])
        self.sample.PrimC = Cfg.lattice[2] * np.array(Cfg.basis[2])
        Atoms = Cfg.atoms
        for ii in range(len(Atoms)):
            self.sample.addAtom(list(map(eval, Atoms[ii][0:3])), Atoms[ii][3])
        self.sample.getRecipVec()
        self.sample.getGs(Cfg.maxQ)

        if self.mode == 'rec':
            f = h5py.File(self.grain.peakFile, 'r')
            # Lim for window position
            self.LimH = np.array(f['limits']).astype(np.int32)
            self.LimD = gpuarray.to_gpu(self.LimH)
            # whichOmega for choosing between omega1 and omega2
            self.whichOmega = np.array(f['whichOmega']).astype(np.int32)
            self.whichOmegaD = gpuarray.to_gpu(self.whichOmega)
            # MaxInt for normalize the weight of each spot 
            # (because different spots have very different intensity but we want them equal weighted)
            self.MaxInt = np.array(f['MaxInt'], dtype=np.float32)
            self.MaxIntD = gpuarray.to_gpu(self.MaxInt)
            self.Gs = np.array(f['Gs'], dtype=np.float32)
            self.NumG = len(self.Gs)
            print('Number of Peaks:',self.NumG)
            self.orienM = np.array(f['OrienM'])
            self.avg_distortion = np.array(f['avg_distortion'])
            self.GsGenerated = True

    # transfer the ExpImgs and all Gs to texture memory
    def loadIm(self):
        f = h5py.File(self.grain.peakFile, 'r')
        AllIm = np.zeros(shape=(self.Cfg.window[1], self.Cfg.window[0], self.NumG * self.Cfg.window[2]), dtype=np.uint32, order='F')
        for ii in range(self.NumG):
            tmp = np.array(f['Imgs']['Im_%03d'%ii])
            AllIm[:tmp.shape[0], :tmp.shape[1], ii * self.Cfg.window[2]:(ii + 1) * self.Cfg.window[2]] = tmp

        self.ImLoaded = True
        Im = np.array(AllIm).astype(np.uint32)
        self.tcExp = gpuarray.to_gpu(Im.ravel())

    def loadGs(self):
        if not self.GsGenerated:
            raise RuntimeError('Gs are not generated yet')

        self.tG = gpuarray.to_gpu(np.array(np.transpose(self.Gs).astype(np.float32),order='F'))
        
        
        
        self.GsLoaded = True
  
        
    def generateGs(self, pos, orien, avg_distortion):
        self.pos = np.array(pos)
        self.orien = np.array(orien)
        self.orienM = Rot.EulerZXZ2Mat(self.orien / 180.0 * np.pi)
        self.avg_distortion = avg_distortion

        Ps, self.Gs, Info = Gsim.GetProjectedVertex(self.Det,
                                                    self.sample, self.avg_distortion.dot(self.orienM),
                                                    self.Cfg.etalimit / 180.0 * np.pi,
                                                    self.pos, getPeaksInfo=True,
                                                    omegaL=self.Cfg.omgRange[0],
                                                    omegaU=self.Cfg.omgRange[1], energy=self.Cfg.energy)
        self.NumG = len(self.Gs)
        Lims = []
        dx = 150
        dy = 80
        for ii in range(self.NumG):
            omegid = int((self.Cfg.omgRange[2] - Ps[ii, 2]) / self.Cfg.omgInterval) - 22  # becuase store 45 frames
            if omegid < 0:
                omegid += int(self.Cfg.omgRange[2] / self.Cfg.omgInterval)
            elif omegid >= int(self.Cfg.omgRange[2] / self.Cfg.omgInterval):
                omegid -= int(self.Cfg.omgRange[2] / self.Cfg.omgInterval)
            x1 = int(2047 - Ps[ii, 0] - dx)
            y1 = int(Ps[ii, 1] - dy)
            x2 = x1 + 2 * dx
            y2 = y1 + 2 * dy
            # ignore camera boundary limit, I'm just lazy, will correct it later
            Lims.append((x1, x2, y1, y2, omegid))
        self.LimH = np.array(Lims, dtype=np.int32)
        self.LimD = gpuarray.to_gpu(self.LimH)
        # whichOmega for choosing between omega1 and omega2
        self.whichOmega = np.zeros(len(Lims), dtype=np.int32)
        for ii in range(len(Lims)):
            if Info[ii]['WhichOmega'] == 'b':
                self.whichOmega[ii] = 2
            else:
                self.whichOmega[ii] = 1
        self.whichOmegaD = gpuarray.to_gpu(self.whichOmega)
        self.GsGenerated = True
    def MoveDet(self, dJ=0, dK=0, dD=0, dT=np.eye(3)):
        self.Det.Move(dJ, dK, np.array([dD, 0, 0]), dT)
        afDetInfoH = np.concatenate(
            [[self.Cfg.JPixelNum, self.Cfg.KPixelNum,
              self.Cfg.pixelSize / 1000.0, self.Cfg.pixelSize / 1000.0],
             self.Det.CoordOrigin,
             self.Det.Norm,
             self.Det.Jvector,
             self.Det.Kvector]).astype(np.float32)
        self.afDetInfoD = gpuarray.to_gpu(afDetInfoH)

    def ResetDet(self):
        self.Det.Reset()
        afDetInfoH = np.concatenate(
            [[self.Cfg.JPixelNum, self.Cfg.KPixelNum,
              self.Cfg.pixelSize / 1000.0, self.Cfg.pixelSize / 1000.0],
             self.Det.CoordOrigin,
             self.Det.Norm,
             self.Det.Jvector,
             self.Det.Kvector]).astype(np.float32)
        self.afDetInfoD = gpuarray.to_gpu(afDetInfoH)
    def sim_pos_wrapper(self, xs, ys, ss):
        NumD = len(xs)
        if self.GsLoaded == False:
            self.loadGs()
        XD = gpuarray.empty(self.NumG * NumD, dtype=np.int32)
        YD = gpuarray.empty(self.NumG * NumD, dtype=np.int32)
        OffsetD = gpuarray.empty(self.NumG * NumD, dtype=np.int32)
        MaskD = gpuarray.empty(self.NumG * NumD, dtype=np.bool_)
        TrueMaskD = gpuarray.empty(self.NumG * NumD, dtype=np.bool_)
        
        xsD = gpuarray.to_gpu(xs.astype(np.float32))
        ysD = gpuarray.to_gpu(ys.astype(np.float32))
        ssD = gpuarray.to_gpu(ss.ravel(order='C').astype(np.float32))

        args = [XD, YD, OffsetD, MaskD, TrueMaskD,
                              xsD, ysD, self.afDetInfoD, ssD,
                              self.whichOmegaD, np.array(NumD).astype(np.int32), np.array(self.NumG).astype(np.int32),
                              np.array(self.Cfg.energy).astype(np.float32), np.array(self.Cfg.window[2]).astype(np.int32), self.LimD, 
                              np.array(5).astype(np.int32), np.array(self.Cfg.omgInterval).astype(np.float32),
                              self.tG]
        
        err = run_cuda_function(self.sim_pos_func,args,(NumD,1,1),(self.NumG,1,1))
        
        xtmp = XD.get().reshape((-1, self.NumG))
        
        ytmp = YD.get().reshape((-1, self.NumG))
        otmp = OffsetD.get().reshape((-1, self.NumG))
        maskH = MaskD.get().reshape(-1, self.NumG)       

        xsD.get()
        ysD.get()
        ssD.get()
        TrueMaskD.get()
        # diffD.get()
        
        return xtmp, ytmp, otmp, maskH

    def simMap(self, tmpxx, tmpyy, AllMaxS, blur=False, dtype=np.uint32):
        if self.GsLoaded == False:
            self.loadGs()
        xtmp, ytmp, otmp, maskH = self.sim_pos_wrapper(tmpxx, tmpyy, AllMaxS)
        
        res = np.zeros(shape=(self.Cfg.window[1], self.Cfg.window[0], self.NumG * self.Cfg.window[2]), dtype=dtype)
        
        
        for ii in range(self.NumG):
            tmpMask = maskH[:, ii]
            tmpX = xtmp[tmpMask, ii]
            tmpY = ytmp[tmpMask, ii]
            tmpO = otmp[tmpMask, ii]
            myMaps = np.zeros((self.Cfg.window[2], self.LimH[ii][3] - self.LimH[ii][2], self.LimH[ii][1] - self.LimH[ii][0]),
                              dtype=dtype)
            
            for jj in range(self.Cfg.window[2]):
                idx = np.where(tmpO == jj)[0]
                if len(idx) == 0:
                    myMaps[jj] = 0
                    continue
                myCounter = Counter(zip(tmpX[idx], tmpY[idx]))
                val = list(myCounter.values())
                xx, yy = zip(*(myCounter.keys()))

                tmp = coo_matrix((val, (yy, xx)),
                                 shape=(
                                 self.LimH[ii][3] - self.LimH[ii][2], self.LimH[ii][1] - self.LimH[ii][0])).toarray()
                
                if blur:
                    myMaps[jj] = gaussian_filter(tmp, sigma=1, mode='nearest', truncate=4)
                else:
                    myMaps[jj] = tmp
                
            myMaps = np.moveaxis(myMaps, 0, 2)

            res[:myMaps.shape[0], :myMaps.shape[1], ii * self.Cfg.window[2]:(ii + 1) * self.Cfg.window[2]] = myMaps
        return res,[xtmp,ytmp,otmp,maskH]

        
        
