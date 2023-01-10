import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.signal import convolve2d
import util.Simulation as Gsim
import util.RotRep as Rot
from scipy import ndimage
from scipy import optimize
from scipy.ndimage import center_of_mass
from scipy.ndimage.measurements import label, find_objects
import json
import h5py
import yaml
from scipy.interpolate import griddata
from util.MicFileTool import MicFile
from util.config import Config
import shutil
import os

# extract window around the Bragg peak on an omega frame


def fetch(ii, pks, fn, offset=0, dx=100, dy=50, verbo=False, more=False, pnx=2048, pny=2048, omega_step=0.05, start_num=0):
    num = int(180/omega_step)
    omegid = int((180-pks[ii, 2])*(1/omega_step))+offset

    if omegid < 0:
        omegid += num
    if omegid >= num:
        omegid -= num

    I = plt.imread(fn+'{0:06d}.tif'.format(omegid+start_num))

    x1 = int((pny-1-pks[ii, 0])-dx)
    y1 = int(pks[ii, 1]-dy)
    if verbo:
        print('y=', pks[ii, 1])
        print('x=', pks[ii, 0])
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = x1+2*dx
    y2 = y1+2*dy
    x2 = min(x2, pnx)
    y2 = min(y2, pny)
    if more:
        return I[y1:y2, x1:x2], (x1, x2, y1, y2, omegid)
    return I[y1:y2, x1:x2]


def getCenter2(Im, Omeg, dx=15, dy=7, do=2):
    Py, Px = ndimage.measurements.maximum_position(Im[Omeg])
    labels = np.zeros(Im.shape, dtype=int)
    x_window = np.array([Px-dx, Px+dy])
    y_window = np.array([Py-dy, Py+dy])
    x_window[x_window < 0] = 0
    x_window[x_window > Im.shape[2]] = Im.shape[2]
    y_window[y_window < 0] = 0
    y_window[y_window > Im.shape[1]] = Im.shape[1]
    o_window = np.array([Omeg-do, Omeg+do])
    o_window[o_window < 0] = 0
    o_window[o_window > Im.shape[0]] = Im.shape[0]

    # print(o_window,labels.shape)
    labels[o_window[0]:o_window[1], y_window[0]           :y_window[1], x_window[0]:x_window[1]] = 1

    co, cy, cx = center_of_mass(Im, labels=labels, index=1)
    return Py, Px, cy, cx, co


def fetch_images(Cfg, grain, path, start_num=0):

    if f'grain_%03d' % grain.grainID in os.listdir(path):
        shutil.rmtree(path+'grain_%03d/' % grain.grainID)

    os.mkdir(path+'grain_%03d/' % grain.grainID)
    os.mkdir(path+'grain_%03d/RawImgData/' % grain.grainID)
    os.mkdir(path+'grain_%03d/FilteredImgData/' % grain.grainID)
    # Cfg = Config_Files/Config_template.yml
    Det1 = Gsim.Detector(config=Cfg)
    crystal_str = Gsim.CrystalStr(config=Cfg)
    crystal_str.getRecipVec()
    crystal_str.getGs(Cfg.maxQ)
    # convert euler angle of grain into radians and then compute the Euler ZXZ rotation matrix
    o_mat = Rot.EulerZXZ2Mat(np.array(grain.euler)/180.0*np.pi)
    pks, Gs, Info = Gsim.GetProjectedVertex(Det1, crystal_str, o_mat, Cfg.etalimit/180*np.pi,
                                            grain.grainPos, getPeaksInfo=True,
                                            omegaL=Cfg.omgRange[0], omegaU=Cfg.omgRange[1], energy=Cfg.energy)

    dx, dy = Cfg.window[0]//2, Cfg.window[1]//2
    window = Cfg.window
    raw_data = Cfg.dataFile

    rng_low = window[2]//2
    if window[2] % 2 == 0:
        rng_high = rng_low
    else:
        rng_high = rng_low + 1
    for ii in range(len(pks)):
        allpks = []
        alllims = []
        totoffset = 0
        # f,axis=plt.subplots(9,5)

        for offset in range(totoffset-rng_low, totoffset+rng_high):
            Im, limits = fetch(ii, pks, raw_data, offset, dx=dx, dy=dy,
                               more=True, omega_step=Cfg.omgInterval, start_num=start_num)

            # ax.imshow(Im,vmin=0,vmax=30)
            allpks.append(Im)
            alllims.append(limits)

        # f.subplots_adjust(wspace=0,hspace=0)
        # f.savefig(path+'Ps_bf/%03d.png'%ii,dpi=200,bbox_inches='tight')
        # plt.close(f)
        allpks = np.array(allpks)
        alllims = np.array(alllims)
        np.save(path+'grain_%03d/RawImgData/Im_%03d' %
                (grain.grainID, ii), allpks)
        np.save(path+'grain_%03d/RawImgData/limit_%03d' %
                (grain.grainID, ii), alllims)
    return


def process_images(grain, path, window, flucThresh):
    Nfile = len([f for f in os.listdir(path+'grain_%03d/RawImgData' %
                                       grain.grainID) if f[:2] == 'Im'])

    Im = []

    for ii in range(Nfile):
        Im.append(np.load(path+'grain_%03d/RawImgData/Im_%03d.npy' %
                          (grain.grainID, ii)))
        Im[ii] = Im[ii]-np.median(Im[ii], axis=0)  # substract the median
        mask = Im[ii] > flucThresh
        Im[ii] = mask*Im[ii]  # make all pixel that below the fluctuation to be zero

    mykernel = np.array([[1, 1, 1], [1, -1, 1], [1, 1, 1]])
    # remove hot spot (whose value is higher than the sum of 8 neighbors)
    for ii in range(Nfile):
        for jj in range(window[2]):
            mask = convolve2d(Im[ii][jj], mykernel, mode='same') > 0
            Im[ii][jj] *= mask

    mykernel2 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])/16.0
    # Smoothing
    for ii in range(Nfile):
        for jj in range(window[2]):
            Im[ii][jj] = convolve2d(Im[ii][jj], mykernel2, mode='same')

    for ii in range(Nfile):
        np.save(path+'grain_%03d/FilteredImgData/Im_%03d.npy' %
                (grain.grainID, ii), Im[ii].astype('uint16'))
    return


def GetVertex(Det1, Gs, Omegas, orien, etalimit, grainpos, bIdx=True, omegaL=-90, omegaU=90, energy=50):
    Peaks = []
    rotatedG = orien.dot(Gs.T).T
    for ii in range(len(rotatedG)):
        g1 = rotatedG[ii]
        res = Gsim.frankie_angles_from_g(g1, verbo=False, energy=energy)

        if Omegas[ii] == 1:
            omega = res['omega_a']/180.0*np.pi
            newgrainx = np.cos(omega)*grainpos[0]-np.sin(omega)*grainpos[1]
            newgrainy = np.cos(omega)*grainpos[1]+np.sin(omega)*grainpos[0]
            idx = Det1.IntersectionIdx(np.array([newgrainx, newgrainy, 0]), res['2Theta'], res['eta'], bIdx, checkBoundary=False
                                       )

            Peaks.append([idx[0], idx[1], res['omega_a']])

        else:
            omega = res['omega_b']/180.0*np.pi
            newgrainx = np.cos(omega)*grainpos[0]-np.sin(omega)*grainpos[1]
            newgrainy = np.cos(omega)*grainpos[1]+np.sin(omega)*grainpos[0]
            idx = Det1.IntersectionIdx(np.array([newgrainx, newgrainy, 0]), res['2Theta'], -res['eta'], bIdx, checkBoundary=False
                                       )
            Peaks.append([idx[0], idx[1], res['omega_b']])

    Peaks = np.array(Peaks)
    return Peaks


def optimize_detector(centers_of_mass, Cfg, grain, cutoff=[60, 30, 10]):
    """
    Optimize detector parameters and grain position for the calibration dataset

    inputs:
        centers_of_mass: array containing the centers of mass for all the peaks in the dataset (numpy array,shape=(num_peaks,3))
        Cfg: Experiment Config object
        grain: Grain Config object
        cutoff: sequence of distances between simulated and experimental peak above which the peak will be thrown out. For [60,30,10],
                the procedure is as follows: Run optimization, remove all peaks which are further than 60 pixels from their simulated peak,
                run optimization again, remove peaks further than 30 pixels, run optimization again, remove all peaks further than 10 pixels,
                run optimation again.

    outputs:
        x : list containing optimzation parameters
        oldPs: Simulated peak positions before optimization
        newPs: Simulated peak positions after optimization
        absCOM: Experimental peak positions
        goodidx: indices of 'good' peaks which were not thrown out



    """

    goodidx = np.arange(len(centers_of_mass))

    Det1 = Gsim.Detector(config=Cfg)

    crystal_str = Gsim.CrystalStr(config=Cfg)

    crystal_str.getRecipVec()
    crystal_str.getGs(Cfg.maxQ)

    o_mat = Rot.EulerZXZ2Mat(np.array(grain.euler)/180.0*np.pi)
    pks, Gs, Info = Gsim.GetProjectedVertex(Det1, crystal_str, o_mat, Cfg.etalimit/180*np.pi,
                                            np.array(grain.grainPos), getPeaksInfo=True,
                                            omegaL=Cfg.omgRange[0], omegaU=Cfg.omgRange[1], energy=Cfg.energy)

    pars = {'J': 0, 'K': 0, 'L': 0, 'tilt': (
        0, 0, 0), 'x': 0, 'y': 0, 'distortion': ((0, 0, 0), (0, 0, 0), (0, 0, 0))}
    DetDefault = Gsim.Detector(
        psizeJ=Cfg.pixelSize*1e-3, psizeK=Cfg.pixelSize*1e-3)

    def SimP(x):

        DetDefault.Reset()
        pars['J'] = x[0] + Cfg.JCenter
        pars['K'] = x[1] + Cfg.KCenter
        pars['L'] = x[2]*10**(-3) + Cfg.Ldistance
        pars['tilt'] = Rot.EulerZXZ2Mat((x[3:6]+np.array(Cfg.tilt))/180*np.pi)
        pars['x'] = x[6]*10**(-3)+grain.grainPos[0]
        pars['y'] = x[7]*10**(-3)+grain.grainPos[1]
        pars['distortion'] = x[-9:].reshape((3, 3))*10**(-3)+np.eye(3)
        DetDefault.Move(pars['J'], pars['K'], np.array(
            [pars['L'], 0, 0]), pars['tilt'])
        pos = np.array([pars['x'], pars['y'], 0])
        Ps = GetVertex(DetDefault,
                       good_Gs,
                       whichOmega,
                       pars['distortion'],
                       Cfg.etalimit/180*np.pi,
                       pos,
                       bIdx=False,
                       omegaL=Cfg.omgRange[0], omegaU=Cfg.omgRange[1], energy=Cfg.energy)

        return Ps

    def CostFunc(x):
        Ps = SimP(x)
        weights = np.array((1, 5, 100))
        tmp = np.sum(((Ps-absCOM)*weights)**2, axis=0)
        return np.sum(tmp)

    tolerance = cutoff
    tolerance.append(tolerance[-1])
    for tol in tolerance:
        imgN = len(goodidx)

        LimH = np.empty((imgN, 5), dtype=np.int32)
        good_Gs = Gs[goodidx]
        whichOmega = np.empty(imgN, dtype=np.int32)

        for ii in range(imgN):
            limit = np.load(
                'Calibration_Files/grain_%03d/RawImgData/limit_%03d.npy' % (grain.grainID, goodidx[ii]))
            LimH[ii, :] = limit[0]

            if Info[goodidx[ii]]['WhichOmega'] == 'b':
                whichOmega[ii] = 2
            else:
                whichOmega[ii] = 1

        absCOM = np.empty(centers_of_mass.shape)
        for ii in range(len(absCOM)):
            absCOM[ii, 1] = LimH[ii, 2]+centers_of_mass[ii, 2]
            absCOM[ii, 0] = Cfg.JPixelNum-1 - \
                (LimH[ii, 0]+centers_of_mass[ii, 1])
            absCOM[ii, 2] = (LimH[ii, 4]+centers_of_mass[ii, 0])
            if absCOM[ii, 2] >= Cfg.omgRange[1]/Cfg.omgInterval:
                absCOM[ii, 2] -= Cfg.omgRange[1]/Cfg.omgInterval
            absCOM[ii, 2] = 180-absCOM[ii, 2]*Cfg.omgInterval

        res = optimize.minimize(CostFunc, np.zeros(17), bounds=[(-5, 5), (-5, 2), (-100, 50)]+3*[(-0.3, 3)]+2*[(-10, 20)]+9*[(-5, 10)]
                                )
        newPs = SimP(res['x'])
        oldPs = SimP(np.zeros(15))
        dists = np.absolute(np.linalg.norm(newPs-absCOM, axis=1))
        inds = np.where(dists < tol)
        goodidx = goodidx[inds]
        centers_of_mass = centers_of_mass[inds]
        print(np.linalg.det(res['x'][-9:].reshape((3, 3))))

    x = list([float(xx) for xx in res['x']])
    return x, oldPs, newPs, absCOM, goodidx


def data_prep(Cfg, grain, path, flucThresh=4, start_num=0):

    fetch_images(Cfg, grain, path, start_num)

    process_images(grain, path, Cfg.window, flucThresh)

    centers_of_mass = []
    Nfile = len([f for f in os.listdir(path+'grain_%03d/RawImgData' %
                                       grain.grainID) if f[:2] == 'Im'])
    goodidx = np.arange(Nfile)

    for idx in goodidx:
        tmp = np.load(path+'grain_%03d/FilteredImgData/Im_%03d.npy' %
                      (grain.grainID, idx))
        Omeg = np.argmax(tmp[Cfg.window[2]//3:2*Cfg.window[2]//3,
                             Cfg.window[1]//3:2*Cfg.window[1]//3,
                             Cfg.window[0]//3:2*Cfg.window[0]//3].sum(axis=(1, 2)))+Cfg.window[2]//3
        Py, Px, cy, cx, co = getCenter2(tmp, Omeg, dx=50, dy=50, do=5)
        print(Omeg)
        plt.imshow(tmp[Omeg])
        plt.scatter(cx, cy)
        plt.show()
        center = np.array([co, cx, cy])
        centers_of_mass.append(center)
    centers_of_mass = np.stack(centers_of_mass)
    return centers_of_mass


def write_config_file(x):

    with open(f'Config_Files/Config_template.yml') as f:
        data = yaml.safe_load(f)

    with open('Config_Files/Config.yml', 'w') as file:
        data['JCenter'] += float(x[0])
        data['KCenter'] += float(x[1])
        data['Ldistance'] += float(x[2]*1e-3)
        data['tilt'] = [float(a) for a in list(np.array(data['tilt'])+x[3:6])]
        documents = yaml.dump(data, file)
#     for g in range(1,Num_Grains+1):
#         with open('Config_Files/Grain_Files/Grain_%03d.yml'%g, 'r') as file:
#             data = yaml.safe_load(file)


#         data['grainPos'] = [float(a) for a in list(np.array(data['grainPos'])+np.array([x[6],x[7],0])*1e-3)]
#         with open('Config_Files/Grain_Files/Grain_%03d.yml'%g, 'w') as file:
#             documents = yaml.dump(data, file)
    return


def optimize_distortion(Cfg, grain, centers_of_mass, path):

    Det1 = Gsim.Detector(config=Cfg)
    crystal_str = Gsim.CrystalStr(config=Cfg)
    crystal_str.getRecipVec()
    crystal_str.getGs(Cfg.maxQ)

    o_mat = Rot.EulerZXZ2Mat(np.array(grain.euler)/180.0*np.pi)
    pks, Gs, Info = Gsim.GetProjectedVertex(Det1, crystal_str, o_mat, Cfg.etalimit/180*np.pi,
                                            grain.grainPos, getPeaksInfo=True,
                                            omegaL=Cfg.omgRange[0], omegaU=Cfg.omgRange[1], energy=Cfg.energy)

    imgN = len(centers_of_mass)

    LimH = np.empty((imgN, 5), dtype=np.int32)
    good_Gs = Gs

    whichOmega = np.empty(imgN, dtype=np.int32)

    for ii in range(imgN):
        limit = np.load(path+'grain_%03d/RawImgData/limit_%03d.npy' %
                        (grain.grainID, ii))
        LimH[ii, :] = limit[0]

        if Info[ii]['WhichOmega'] == 'b':
            whichOmega[ii] = 2
        else:
            whichOmega[ii] = 1

    absCOM = np.empty(centers_of_mass.shape)
    for ii in range(len(absCOM)):
        absCOM[ii, 1] = LimH[ii, 2]+centers_of_mass[ii, 2]
        absCOM[ii, 0] = Cfg.JPixelNum-1-(LimH[ii, 0]+centers_of_mass[ii, 1])
        absCOM[ii, 2] = (LimH[ii, 4]+centers_of_mass[ii, 0])
        if absCOM[ii, 2] >= Cfg.omgRange[1]/Cfg.omgInterval:
            absCOM[ii, 2] -= Cfg.omgRange[1]/Cfg.omgInterval
        absCOM[ii, 2] = 180-absCOM[ii, 2]*Cfg.omgInterval
    pars = {'J': 0, 'K': 0, 'L': 0, 'tilt': (
        0, 0, 0), 'x': 0, 'y': 0, 'distortion': ((0, 0, 0), (0, 0, 0), (0, 0, 0))}
    DetDefault = Gsim.Detector(
        psizeJ=Cfg.pixelSize*1e-3, psizeK=Cfg.pixelSize*1e-3)

    def SimP(x):

        DetDefault.Reset()
        pars['J'] = Cfg.JCenter
        pars['K'] = Cfg.KCenter
        pars['L'] = Cfg.Ldistance
        pars['tilt'] = Rot.EulerZXZ2Mat(np.array(Cfg.tilt)/180*np.pi)
        pars['x'] = grain.grainPos[0]
        pars['y'] = grain.grainPos[1]
        pars['distortion'] = x.reshape((3, 3))*10**(-3)+np.eye(3)
        DetDefault.Move(pars['J'], pars['K'], np.array(
            [pars['L'], 0, 0]), pars['tilt'])
        pos = np.array([pars['x'], pars['y'], 0])
        Ps = GetVertex(DetDefault,
                       good_Gs,
                       whichOmega,
                       pars['distortion'],
                       Cfg.etalimit/180*np.pi,
                       pos,
                       bIdx=False,
                       omegaL=Cfg.omgRange[0], omegaU=Cfg.omgRange[1], energy=Cfg.energy)

        return Ps

    def CostFunc(x):
        Ps = SimP(x)
        weights = np.array((1, 5, 100))
        tmp = np.sum(((Ps-absCOM)*weights)**2, axis=0)
        return np.sum(tmp)

    res = optimize.minimize(CostFunc, np.zeros(9), bounds=9*[(-5, 10)]
                            )
    newPs = SimP(res['x'])
    oldPs = SimP(np.zeros(9))
    dists = np.absolute(np.linalg.norm(newPs-absCOM, axis=1))

    x = list([float(xx) for xx in res['x']])

    return x, newPs, absCOM, good_Gs, Info


def write_peak_file(Cfg, grain, centers_of_mass, path, cutoff=10, opt=False):

    if opt:
        x, pks, absCOM, Gs, Info = optimize_distortion(
            Cfg, grain, centers_of_mass, path)
    else:
        Det1 = Gsim.Detector(config=Cfg)
        crystal_str = Gsim.CrystalStr(config=Cfg)
        crystal_str.getRecipVec()
        crystal_str.getGs(Cfg.maxQ)

        o_mat = Rot.EulerZXZ2Mat(np.array(grain.euler)/180.0*np.pi)
        pks, Gs, Info = Gsim.GetProjectedVertex(Det1, crystal_str, o_mat, Cfg.etalimit/180*np.pi,
                                                grain.grainPos, getPeaksInfo=True,
                                                omegaL=Cfg.omgRange[0], omegaU=Cfg.omgRange[1], energy=Cfg.energy)

        imgN = len(centers_of_mass)

        LimH = np.empty((imgN, 5), dtype=np.int32)
        good_Gs = Gs
        whichOmega = np.empty(imgN, dtype=np.int32)

        for ii in range(imgN):
            limit = np.load(
                path+'grain_%03d/RawImgData/limit_%03d.npy' % (grain.grainID, ii))
            LimH[ii, :] = limit[0]

            if Info[ii]['WhichOmega'] == 'b':
                whichOmega[ii] = 2
            else:
                whichOmega[ii] = 1

        absCOM = np.empty(centers_of_mass.shape)
        for ii in range(len(absCOM)):
            absCOM[ii, 1] = LimH[ii, 2]+centers_of_mass[ii, 2]
            absCOM[ii, 0] = Cfg.JPixelNum-1 - \
                (LimH[ii, 0]+centers_of_mass[ii, 1])
            absCOM[ii, 2] = (LimH[ii, 4]+centers_of_mass[ii, 0])
            if absCOM[ii, 2] >= Cfg.omgRange[1]/Cfg.omgInterval:
                absCOM[ii, 2] -= Cfg.omgRange[1]/Cfg.omgInterval
            absCOM[ii, 2] = 180-absCOM[ii, 2]*Cfg.omgInterval

    print('total peaks:', len(pks))
    dists = np.absolute(np.linalg.norm(pks-absCOM, axis=1))
    inds = np.where(dists < cutoff)
    goodidx = inds[0]
    print('peaks within cutoff distance:', len(goodidx))

    centers_of_mass = centers_of_mass[goodidx]

    fig, ax = plt.subplots(1, 2, figsize=(12, 3))
    pks = pks[goodidx]
    absCOM = absCOM[goodidx]
    ax[0].hist(pks[:, 2]-absCOM[:, 2],
               bins=np.arange(-0.2, 0.35, 0.05), alpha=0.5)
    ax[0].set_xlabel(r'$\Omega$ difference $(^\circ)$', fontsize=20)
    ax[1].scatter(pks[:, 0]-absCOM[:, 0], pks[:, 1]-absCOM[:, 1])
    ax[1].set_xlabel('horizontal difference (pixels)', fontsize=20)
    ax[1].set_ylabel('vertical difference (pixels)', fontsize=20)
    ax[0].tick_params(axis='both', which='major', labelsize=20)
    ax[1].tick_params(axis='both', which='major', labelsize=20)
    plt.savefig('calibration.png', dpi=100, bbox_inches='tight')
    plt.show()

    imgN = len(goodidx)
    peakMap = np.zeros(
        (Cfg.window[1], Cfg.window[0], Cfg.window[2]*imgN), dtype=np.uint16)
    LimH = np.empty((imgN, 5), dtype=np.int32)
    Gs_good = Gs[goodidx]
    whichOmega = np.empty(imgN, dtype=np.int32)

    avg_distortion = np.eye(3)  # x.reshape((3,3))
    MaxInt = np.empty(imgN, dtype=np.float32)
    g = grain.grainID
    for ii in range(imgN):

        limit = np.load(
            path+'grain_%03d/RawImgData/limit_%03d.npy' % (g, goodidx[ii]))
        img = np.load(path+'grain_%03d/FilteredImgData/Im_%03d.npy' %
                      (g, goodidx[ii]))
        peakMap[:img.shape[1], :img.shape[2], ii * Cfg.window[2]
            :(ii + 1) * Cfg.window[2]] = np.moveaxis(img, 0, -1)
        LimH[ii, :] = limit[0]
        MaxInt[ii] = np.max(img)
        if Info[goodidx[ii]]['WhichOmega'] == 'b':
            whichOmega[ii] = 2
        else:
            whichOmega[ii] = 1

    orien = Rot.EulerZXZ2Mat(np.array(grain.euler)/180.0*np.pi)

    with h5py.File(grain.peakFile, 'w') as f:
        f.create_dataset("limits", data=LimH)
        f.create_dataset("Gs", data=Gs_good)
        f.create_dataset("whichOmega", data=whichOmega)
        f.create_dataset("Pos", data=grain.grainPos)
        f.create_dataset("OrienM", data=orien)
        f.create_dataset("avg_distortion", data=avg_distortion)
        f.create_dataset("MaxInt", data=MaxInt)
        f.create_dataset("Goodidx", data=goodidx)

        grp = f.create_group('Imgs')
        for ii in range(imgN):
            grp.create_dataset(
                'Im_%03d' % ii, data=peakMap[:, :, ii * Cfg.window[2]:(ii + 1) * Cfg.window[2]])
    return
