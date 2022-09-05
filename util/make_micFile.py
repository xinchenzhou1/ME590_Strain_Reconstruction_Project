import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.signal import convolve2d
import util.Simulation as Gsim
import util.RotRep as Rot
from scipy import ndimage
from scipy import optimize
from scipy.ndimage import center_of_mass
from scipy.ndimage.measurements import label,find_objects
import json
import h5py
import yaml
from scipy.interpolate import griddata
from util.MicFileTool import MicFile
from util.config import Config
import shutil
import os


def find_grains(Cfg,conf_tol,match_tol,voxel_size =0.002):
    if Cfg.hexomapFile[-3:] == 'npy':
        a = np.load(Cfg.hexomapFile)

        grid_x = a[:,:,0]
        grid_y = a[:,:,1]
        grid_c = a[:,:,6]
        grid_e1 = a[:,:,3]
        grid_e2 = a[:,:,4]
        grid_e3 = a[:,:,5]
    else:
        a=MicFile(Cfg.hexomapFile)
        grid_x,grid_y=np.meshgrid(np.arange(a.snp[:,0].min(),a.snp[:,0].max(),voxel_size),np.arange(a.snp[:,1].min(),a.snp[:,1].max(),voxel_size))
        grid_c = griddata(a.snp[:,0:2],a.snp[:,9],(grid_x,grid_y),method='nearest')
        grid_e1 = griddata(a.snp[:,0:2],a.snp[:,6],(grid_x,grid_y),method='nearest')
        grid_e2 = griddata(a.snp[:,0:2],a.snp[:,7],(grid_x,grid_y),method='nearest')
        grid_e3 = griddata(a.snp[:,0:2],a.snp[:,8],(grid_x,grid_y),method='nearest')
        
    eulers = np.stack([grid_e1,grid_e2,grid_e3]) 
    right = np.abs(np.diff(eulers,axis=1,append=0)).mean(axis=0)
    up = np.abs(np.diff(eulers,axis=2,append=0)).mean(axis=0)
    
    misorientation = np.max(np.stack([right,up]),axis=0)
 
    
        
    
    g = np.where(misorientation>conf_tol,0,1)

    labels,num_features = label(g)

    ll = np.float32(labels.copy())
    ll[ll==0] = np.nan


    GrainDict={}
    for l in np.sort(np.unique(labels))[1:]:

        com =center_of_mass(g,labels,l)
        com = (int(com[0]),int(com[1]))
        GrainDict[l] = (grid_e1[com],grid_e2[com],grid_e3[com])
    GrainIDMap=np.zeros(grid_c.shape,dtype=int)
    for grainID in GrainDict:

        (e1,e2,e3)=GrainDict[grainID]
        tmp = grid_c > match_tol
        tmp*=np.absolute(grid_e1 - e1)<1
        tmp*=np.absolute(grid_e2 - e2)<1
        tmp*=np.absolute(grid_e3 - e3)<1
        GrainIDMap[tmp]= grainID 
    newGrainIDMap = GrainIDMap.copy()
    for i,ggg in enumerate(np.unique(GrainIDMap)):
        newGrainIDMap[newGrainIDMap==ggg] = i
        
    GrainIDMap = newGrainIDMap
    print(grid_x.min(),grid_y.min())
    with h5py.File(Cfg.micFile,'w') as f:
        ds=f.create_dataset("origin", data = np.array([grid_x.min(),grid_y.min()]))
        ds.attrs[u'units'] = u'mm'
        ds=f.create_dataset("stepSize", data = np.array([voxel_size,voxel_size]))
        ds.attrs[u'units'] = u'mm'
        f.create_dataset("Xcoordinate", data = grid_x)
        f.create_dataset("Ycoordinate", data = grid_y)
        f.create_dataset("Confidence", data = grid_c)
        f.create_dataset("Ph1", data = grid_e1)
        f.create_dataset("Psi", data = grid_e2)
        f.create_dataset("Ph2", data = grid_e3)
        f.create_dataset("GrainID", data = GrainIDMap)

    gg = np.float32(GrainIDMap.copy())
    gg[gg==0] = np.nan
    fig,ax = plt.subplots(ncols=4,figsize=(20,7))
    ax[0].imshow(grid_c,origin='lower')
    ax[0].set_title('Confidence Map')
    ax[1].imshow(g,origin='lower')
    ax[1].set_title('Thresholded and Binarized Misorientation Map')
    ax[2].imshow(ll,origin='lower')
    ax[2].set_title('Blob Finder Results')
    ax[3].imshow(gg,origin='lower')
    ax[3].set_title('Microstructure after Orientation Matching')
    plt.show()
    print('Number of Grains:',GrainIDMap.max())
    
    
    grain_Ids = np.unique(GrainIDMap)[1:]
    
    

    grain_posi = []

    for i in grain_Ids:
        with open(Cfg.grainTemp) as f:
            data = yaml.safe_load(f)
        i = int(i)
        locations = np.where(GrainIDMap==i,1,0)

        com_ind = np.int32(np.round(center_of_mass(locations)))

        grain_pos = np.round(np.array([ grid_x[com_ind[0],com_ind[1]],grid_y[com_ind[0],com_ind[1]],0]),4)
        grain_posi.append(grain_pos)
        euler = np.array([grid_e1[locations==1].mean(),grid_e2[locations==1].mean(),grid_e3[locations==1].mean()])


        data['grainID'] = i
        data['peakFile'] = 'Peak_Files/grain_%03d/Peaks_%03d.hdf5'%(i,i)
        data['recFile'] = 'Rec_Files/grain_%03d/Rec_%03d.hdf5'%(i,i)
        data['pixHitFile'] = 'Rec_Files/grain_%03d/pixHit_%03d.hdf5'%(i,i)
        data['grainPos'] = [float(g) for g in grain_pos]
        data['euler'] = [float(e) for e in euler]

        with open(f'Config_Files/Grain_Files/Grain_%03d.yml'%i, 'w') as file:

            documents = yaml.dump(data, file)
    
    
    
    
    return GrainIDMap.max()


