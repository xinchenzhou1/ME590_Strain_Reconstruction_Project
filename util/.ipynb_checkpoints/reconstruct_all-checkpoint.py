import numpy as np
from util.config import Config
from util.reconstructor_mjw import Reconstructor
import shutil
import os
import matplotlib.pyplot as plt
import h5py
import time

def reconstruct(Cfg_file,grain_file,NumD = 1000,num_it=10,conf=0.5,plot=False):


    c = Config(Cfg_file)
    g = Config(grain_file)
    
    if f'grain_%03d'%g.grainID in os.listdir('Rec_Files'):
        shutil.rmtree('Rec_Files/grain_%03d/'%g.grainID)
        
    os.mkdir('Rec_Files/grain_%03d/'%g.grainID)
    r = Reconstructor(c,g)
    with h5py.File(g.micFile,'r') as f:
        GrainIDMap = f["GrainID"][:].astype(int)
        grid_x = f["Xcoordinate"][:]
        grid_y = f["Ycoordinate"][:]
        grid_c = f["Confidence"][:]

    mask = np.where(GrainIDMap==g.grainID)
    x = grid_x[mask]
    y = grid_y[mask]
    con = grid_c[mask]

    threshold=conf
    if plot:
        fig, ax = plt.subplots()
        cax=ax.scatter(x[con>threshold],y[con>threshold],c=con[con>threshold],s=5)
        cbar = fig.colorbar(cax)

        plt.show()

    tmpx=x[con>threshold]
    tmpy=y[con>threshold]

    print('Phase 1')
    Phase1_Conf, Phase1_S = r.ReconGridsPhase1(tmpx,tmpy,NumD=NumD)
    
    r.SimPhase1Result(tmpx,tmpy,Phase1_S)
    tmpS=Phase1_S.copy()
    print('Phase 2')
    Phase2_S, history = r.ReconGridsPhase2(tmpx,tmpy,tmpS,iterN=num_it,NumD=NumD)
    if plot:
        plt.plot(history)
        plt.show()
    realO,realS = r.Transform2RealS(Phase2_S)

    with h5py.File(g.recFile, 'w') as f:
        f.create_dataset("x", data=tmpx)
        f.create_dataset("y", data=tmpy)

        f.create_dataset("Phase1_Conf", data=Phase1_Conf)
        f.create_dataset("Phase1_S", data=Phase1_S)

        f.create_dataset("Phase2_S", data=Phase2_S)

        f.create_dataset('Phase2_history', data=history)

        f.create_dataset("realS", data=realS)
        f.create_dataset("realO", data=realO)
        
        
        
    res,pixels = r.recon.simMap(tmpx,tmpy,Phase2_S)
    peak_dict = make_grain_dict(pixels,tmpx,tmpy,dict_type='peak')
    with h5py.File(g.pixHitFile, 'w') as f:
        for pk in peak_dict:
            grp = f.create_group(pk)
            for pix in peak_dict[pk]:
                pix_grp = grp.create_group(pix)
                pix_grp.create_dataset("Voxels",data = np.stack(peak_dict[pk][pix]))
    r.unload_module()
    
    
    
    return
def read_hdf(file):
    c = Config(file)
    with h5py.File(c.recFile, 'r') as f:
        tmpx = np.array(f['x'])
        tmpy = np.array(f['y'])
        realS = np.array(f['realS'])
    return tmpx,tmpy,realS

def make_grain_dict(pixels,tmpx,tmpy,dict_type='voxel'):
    xx,yy,oo,mask = pixels
    if dict_type == 'voxel':

        grain_dict = {}

        for i in range(xx.shape[0]):   
            g = {}
            for j in range(xx.shape[1]):
                g['Peak_%03d'%j] = np.stack([xx[i,j],yy[i,j],oo[i,j]])

            grain_dict[f'{np.round(tmpx[i],3)},{np.round(tmpy[i],3)}'] = g

        # print(grain_dict['Voxel_-0.166_0.022'])


    if dict_type == 'peak':
        grain_dict = {}

        for i in range(xx.shape[1]):  
            p = {}
            mask_p = mask[:,i]
            det_x = xx[mask_p,i]
            det_y = yy[mask_p,i]
            det_o = oo[mask_p,i]
            vox_x = tmpx[mask_p]
            vox_y = tmpy[mask_p]
            for x,y,o,sx,sy in zip(det_x,det_y,det_o,vox_x,vox_y):
                sx = np.round(sx,3)
                sy = np.round(sy,3)
                if f'{x},{y},{o}' in p.keys():
                    p[f'{x},{y},{o}'].append( [sx,sy])
                else:
                    p[f'{x},{y},{o}'] =  [[sx,sy]]


            grain_dict['Peak_%03d'%i] = p

    return grain_dict

def make_confidence_map(Cfg,grain):
    peak_dict = h5py.File(grain.pixHitFile)
    limits = h5py.File(grain.peakFile,'r')['limits']
    omegas = np.int16(Cfg.omgRange[1]/Cfg.omgInterval)
    data = np.zeros((Cfg.JPixelNum,Cfg.KPixelNum,omegas),dtype=int)
    alll = []
    for l,p in zip(limits,peak_dict):
        for pixel in peak_dict[p]:
            x,y,z = list(eval(pixel))
            intensity = len(peak_dict[p][pixel])
            data[x+l[0],y+l[2],z+l[4]] = intensity
            alll.append(z+l[4])
    return data
def plot_strain(grain_Ids,marker_size=0.1,min_max=None):
    print(len(grain_Ids),'grains')
    tmpx,tmpy,realS = [],[],[]
    for Id in grain_Ids:
        tx,ty,S = read_hdf('Config_Files/Grain_Files/Grain_%03d.yml'%Id)
        tmpx.append(tx)
        tmpy.append(ty)
        realS.append(S)
    tmpx = np.concatenate(tmpx)
    tmpy = np.concatenate(tmpy)
    realS = np.concatenate(realS)
    print(realS.shape)
    
    comp=[2,2]
    inds = [[0,0],[1,1],[2,2],[1,2],[0,2],[0,1]]
    labels =[r'$\varepsilon_{xx}$',r'$\varepsilon_{yy}$',r'$\varepsilon_{zz}$',r'$\varepsilon_{yz}$',r'$\varepsilon_{xz}$',r'$\varepsilon_{xy}$']

    fig, axs = plt.subplots(nrows=2,ncols=3,figsize=(18,10))
    for ax,ind,label in zip(axs.ravel(),inds,labels):
        s = realS[:,ind[0],ind[1]]-1
        if min_max == None:
            vmin = s.mean()-np.std(s)*3
            vmax = s.mean()+np.std(s)*3
            
        else:
            vmin,vmax = min_max
        
        cax=ax.scatter(tmpx,tmpy,c=s,
                       s=marker_size,cmap='jet',
                       vmin=vmin,vmax=vmax
                      
                      )
        cbar = fig.colorbar(cax,ax=ax)
        ax.set_title(label)


    return fig