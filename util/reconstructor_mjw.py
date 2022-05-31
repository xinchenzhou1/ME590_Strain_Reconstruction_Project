# coding: utf-8
# import pycuda.gpuarray as gpuarray
# import pycuda.driver as cuda
from cuda import cuda, nvrtc
import numpy as np
from scipy.linalg import polar
from util.MicFileTool import read_mic_file
import util.RotRep as Rot
from util.initializer_mjw import Initializer
import h5py
import os
import time

import util.optimizers_mjw as optimizers
import gpuarray


class Reconstructor:

    def __init__(self, Cfg,grain):
        self.grain = grain
        self.Cfg = Cfg
        self.peakFile = h5py.File(grain.peakFile, 'r')
        self.recon = Initializer(Cfg,grain)
        self.outFN = grain.recFile
        self.micFN = grain.micFile

    def GetGrids(self):
        Sample = h5py.File(self.micFN,'r')
        
        GIDLayer = Sample["GrainID"][:].astype(int)
        
        if ("Xcoordinate" in Sample.keys()) and ("Ycoordinate" in Sample.keys()):
            xv = Sample["Xcoordinate"][:]
            yv = Sample["Ycoordinate"][:]
        else:      
            len1 = GIDLayer.shape[1]
            len2 = GIDLayer.shape[0]
            orig = Sample["origin"][:]
            step = Sample["stepSize"][:]
            tmpx = np.arange(orig[0], step[0] * len1 + orig[0], step[0])
            tmpy = np.arange(orig[1], step[1] * len2 + orig[1], step[1])
            xv, yv = np.meshgrid(tmpx, tmpy)
            
        idx = np.where(GIDLayer == self.grain.grainID)
        x = xv[idx]
        y = yv[idx]
        Sample.close()
        return x, y

    def ReconGridsPhase1(self, tmpxx, tmpyy, NumD=10000, numCut=10):
        # allocate gpu memory
        XD = gpuarray.empty(self.recon.NumG * NumD, dtype=np.int32)
        YD = gpuarray.empty(self.recon.NumG * NumD, dtype=np.int32)
        OffsetD = gpuarray.empty(self.recon.NumG * NumD, dtype=np.int32)
        MaskD = gpuarray.empty(self.recon.NumG * NumD, dtype=np.bool_)
        TrueMaskD = gpuarray.empty(self.recon.NumG * NumD, dtype=np.bool_)
        scoreD = gpuarray.empty(NumD, dtype=np.float32)
        S_gpu = cuda.cuMemAlloc(np.empty(NumD * 9 * 4,dtype=np.float32).nbytes)[1]

        AllMaxScore = []
        AllMaxS = []
        
        for ii in range(len(tmpxx)):

            t = optimizers.CrossEntropyMethod(self.recon, tmpxx[ii], tmpyy[ii],
                                              XD, YD, OffsetD, MaskD, TrueMaskD, scoreD, S_gpu,
                                              NumD=NumD, numCut=numCut)
            
            end = time.time()

            if ii % 1000 == 0:
                print(ii)
                start = time.time()
            AllMaxScore.append(t[2])
            AllMaxS.append(t[1])
        AllMaxS = np.array(AllMaxS)
        AllMaxScore = np.array(AllMaxScore)
        err, = cuda.cuMemFree(S_gpu)
        
        
        XD.get()
        YD.get()
        OffsetD.get()
        MaskD.get()
        TrueMaskD.get()
        scoreD.get()
        
        
        return AllMaxScore, AllMaxS

    def SimPhase1Result(self, tmpxx, tmpyy, AllMaxS, epsilon=1e-6):
        falseMaps = self.recon.simMap(tmpxx, tmpyy, AllMaxS, blur=False, dtype=np.uint32)[0]
        
        realMaps = np.zeros(shape=(self.Cfg.window[1], self.Cfg.window[0], self.recon.NumG * self.Cfg.window[2]), dtype=np.uint32)
        for ii in range(self.recon.NumG):
            tmp = np.array(self.peakFile['Imgs']['Im_%03d'%ii])
            realMaps[:tmp.shape[0], :tmp.shape[1], ii * self.Cfg.window[2]:(ii + 1) * self.Cfg.window[2]] = tmp

        self.falseMapsD = gpuarray.to_gpu((falseMaps.ravel() + epsilon).astype(np.float32))
        self.realMapsLogD = gpuarray.to_gpu(np.log(realMaps.ravel() + epsilon).astype(np.float32))
        self.realMapsD = gpuarray.to_gpu((realMaps.ravel() + epsilon).astype(np.float32))

        
        return

    def KL_eachG(self):
        KLdivergences = np.empty(self.recon.NumG)
        for ii in range(self.recon.NumG):
            KLD = gpuarray.empty(self.Cfg.window[0] * self.Cfg.window[1] * self.Cfg.window[2],dtype=np.float32)
            
            self.recon.KL_total_func(KLD, self.realMapsLogD, self.falseMapsD,
                                     np.int32(ii), np.int32(self.recon.NumG), np.int32(self.Cfg.window[2]),
                                     block=(self.Cfg.window[2], 1, 1), grid=(self.Cfg.window[0] * self.Cfg.window[1], 1))
            KLH = KLD.get()
            KLdivergences[ii] = np.sum(KLH)
        return KLdivergences


    def ReconGridsPhase2(self, tmpxx, tmpyy, AllMaxS,
                         NumD=10000, numCut=50, iterN=10, shuffle=False):
       # allocate gpu memory
        
        XD = gpuarray.empty(self.recon.NumG * NumD, dtype=np.int32)
        YD = gpuarray.empty(self.recon.NumG * NumD, dtype=np.int32)
        OffsetD = gpuarray.empty(self.recon.NumG * NumD, dtype=np.int32)
        MaskD = gpuarray.empty(self.recon.NumG * NumD, dtype=np.bool_)
        TrueMaskD = gpuarray.empty(self.recon.NumG * NumD, dtype=np.bool_)
        diffD = gpuarray.empty(NumD, dtype=np.float32)
        S_gpu = cuda.cuMemAlloc(np.empty(NumD * 9 * 4,dtype=np.float32).nbytes)[1]
        history = [0]
        acc = 0
        for jj in range(iterN):
            print("{0:d}/{1:d}, loss={2:}".format(jj + 1, iterN, acc))
            if shuffle:
                #np.random.seed(42)
                order = np.random.permutation(len(tmpxx))
            else:
                order = np.arange(len(tmpxx))
                
            for ii in order:
                
                tmp = optimizers.ChangeOneVoxel_KL(self.recon,
                                                   tmpxx[ii], tmpyy[ii], AllMaxS[ii], self.realMapsLogD,
                                                   self.falseMapsD,
                                                   XD, YD, OffsetD, MaskD, TrueMaskD, diffD, S_gpu,
                                                   NumD=NumD, numCut=numCut, cov=1e-6 * np.eye(9), MaxIter=3,
                                                   debug=False)
                
                AllMaxS[ii] = tmp[1]

                acc += tmp[2]
                history.append(acc)
        err, = cuda.cuMemFree(S_gpu)
        
        XD.get()
        YD.get()
        OffsetD.get()
        MaskD.get()
        TrueMaskD.get()
        diffD.get()
        return AllMaxS, np.array(history)

    def Transform2RealS(self, AllMaxS):
        # convert it from reciprocal space to real space
        S = np.array(AllMaxS) + (self.recon.avg_distortion - np.eye(3))
        realS = np.empty(AllMaxS.shape)
        realO = np.empty(AllMaxS.shape)
        for ii in range(len(realS)):
            t = np.linalg.inv(S[ii].T).dot(self.recon.orienM)
            realO[ii], realS[ii] = polar(t, 'left')
        return realO, realS

    def run(self):
        exists = os.path.isfile(self.outFN)
        if exists:
            f = h5py.File(self.outFN, 'r+')
            x = f["x"][:]
            y = f["y"][:]
            AllMaxS = f["Phase1_S"][:]
            self.SimPhase1Result(x, y, AllMaxS)
            AllMaxS, history = self.ReconGridsPhase2(x, y, AllMaxS)
            tmp = f["Phase2_S"]
            tmp[...] = AllMaxS
            tmp = f["Phase2_history"]
            del tmp
            KLd = self.KL_eachG()
            tmp = f["final_KLdivergence"]
            del tmp
            f.create_dataset("final_KLdivegence", data=KLd)
            f.create_dataset('Phase2_history', data=history)

            realO, realS = self.Transform2RealS(AllMaxS)
            tmp = f["realS"]
            tmp[...] = realS
            tmp = f["realO"]
            tmp[...] = realO
            f.close()
        else:
            with h5py.File(self.outFN, 'w') as f:
                x, y = self.GetGrids()
                f.create_dataset("x", data=x)
                f.create_dataset("y", data=y)

                AllMaxScore, AllMaxS = self.ReconGridsPhase1(x, y)
                f.create_dataset("Phase1_Conf", data=AllMaxScore)
                f.create_dataset("Phase1_S", data=AllMaxS)

                self.SimPhase1Result(x, y, AllMaxS)
                AllMaxS, history = self.ReconGridsPhase2(x, y, AllMaxS)
                f.create_dataset("Phase2_S", data=AllMaxS)
                KLd=self.KL_eachG()
                f.create_dataset("final_KLdivergence",data=KLd)
                f.create_dataset('Phase2_history', data=history)

                realO, realS = self.Transform2RealS(AllMaxS)
                f.create_dataset("realS", data=realS)
                f.create_dataset("realO", data=realO)
    def unload_module(self):
        err, = cuda.cuModuleUnload(self.recon.module)
        self.recon.tG.get()
        self.recon.tcExp.get()
        self.recon.LimD.get()
        self.recon.windowD.get()
        self.recon.whichOmegaD.get()
        self.recon.afDetInfoD.get()
        self.recon.MaxIntD.get()
        
        self.falseMapsD.get()
        self.realMapsLogD.get()
        self.realMapsD.get()
        return