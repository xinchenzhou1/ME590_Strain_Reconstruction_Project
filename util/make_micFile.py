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

#Cfg - config file, conf_tol - Blob finding conf tolerance, match_tol - orientation tolerance
def find_grains(Cfg,conf_tol,match_tol,voxel_size =0.002):
    #checks config files' hexomapFile data type
    if Cfg.hexomapFile[-3:] == 'npy':
        a = np.load(Cfg.hexomapFile)

        grid_x = a[:,:,0]
        grid_y = a[:,:,1]
        grid_c = a[:,:,6]
        grid_e1 = a[:,:,3]
        grid_e2 = a[:,:,4]
        grid_e3 = a[:,:,5]
    else:
        #Constructor for class MicFile that takes hexomap file other than that ending in .npy
        #and instantiate variable a, where a contains outputs from standard 
        #nf-HEDM, the voxelized orientations on a cross section of the sample
        a=MicFile(Cfg.hexomapFile)
        grid_x,grid_y=np.meshgrid(np.arange(a.snp[:,0].min(),a.snp[:,0].max(),voxel_size),np.arange(a.snp[:,1].min(),a.snp[:,1].max(),voxel_size))
        #Makes a meshgrid with each grid the same size as the voxels.
        grid_c = griddata(a.snp[:,0:2],a.snp[:,9],(grid_x,grid_y),method='nearest')
        #Interpolating confidence into the meshgrid created above.
        #a.snp[:, 0:2] extracts column 0 and 1 for every row
        grid_e1 = griddata(a.snp[:,0:2],a.snp[:,6],(grid_x,grid_y),method='nearest')
        grid_e2 = griddata(a.snp[:,0:2],a.snp[:,7],(grid_x,grid_y),method='nearest')
        grid_e3 = griddata(a.snp[:,0:2],a.snp[:,8],(grid_x,grid_y),method='nearest')
        
    eulers = np.stack([grid_e1,grid_e2,grid_e3]) 
    right = np.abs(np.diff(eulers,axis=1,append=0)).mean(axis=0)
    up = np.abs(np.diff(eulers,axis=2,append=0)).mean(axis=0)
    
    misorientation = np.max(np.stack([right,up]),axis=0)
 
    
        
    
    g = np.where(misorientation>conf_tol,0,1)
    #If the difference between the two sets of euler angles is 
    #less than 1 degree for each angle, the voxel is added to that grain.
    
    #2D array g storing either 0 or 1 - when misorientation is greater than
    #confidence tolerance, set the voxel to 0. Otherwise voxel is set to 1.
    #This may be used to create thresholded and binarized misorientation map.
    #All the 1s would be the potential grain boundaries.

    labels,num_features = label(g)
    print ("Number of features found using blob finder: ", num_features)
    # Lowering the threshold results in identifying more features, vice versa
    # label() uses a four way search to connect neighbours of each non-zero values in input and count as features
    #lables is the labeled 2D array where 1s are considered as objects and 0s are 
    #considered background. 
    #num_features sums up the number of "objects"

    ll = np.float32(labels.copy())
    ll[ll==0] = np.nan
    #sets the background to nan - this is used to plot the blob finder results plot


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
    # (419, 456) array contains grain coordinates on this grid after misorientation matching
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
    # iterate each grain from 1 to 56
    # open grain_template.yml file under Config_Files to write each grain file

    for i in grain_Ids:
        with open(Cfg.grainTemp) as f:
            data = yaml.safe_load(f)
        i = int(i)
        # Return elements chosen from 1 or 0 depending on if elements in GrainIDMap equals to i
        locations = np.where(GrainIDMap==i,1,0)
        #computes center of mass of this grain in the grid based on the locations variable (somewhere in between 419, 456 grid)
        com_ind = np.int32(np.round(center_of_mass(locations)))
        # converts com into the x and y grain position based on grid size
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


