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

# Cfg - config file, conf_tol - Blob finding conf tolerance, match_tol - orientation tolerance


def find_grains(Cfg, confidenceTol, misOrienMatchingTol, voxelSize=0.002):
    # checks config files' hexomapFile data type, use directly if file extension is in .npy
    if Cfg.hexomapFile[-3:] == 'npy':
        a = np.load(Cfg.hexomapFile)
        gridX = a[:, :, 0]
        gridY = a[:, :, 1]
        gridConfidence = a[:, :, 6]
        gridEuler1 = a[:, :, 3]
        gridEuler2 = a[:, :, 4]
        gridEuler3 = a[:, :, 5]
    else:
        # Constructor for class MicFile that takes hexomap file other than that ending in .npy
        # and instantiate variable a, where a contains outputs from standard
        # nf-HEDM, the voxelized orientations on a cross section of the sample
        a = MicFile(Cfg.hexomapFile)
        gridX, gridY = np.meshgrid(np.arange(a.snp[:, 0].min(), a.snp[:, 0].max(
        ), voxelSize), np.arange(a.snp[:, 1].min(), a.snp[:, 1].max(), voxelSize))
        # Makes a meshgrid with each grid the same size as the voxels.
        gridConfidence = griddata(
            a.snp[:, 0:2], a.snp[:, 9], (gridX, gridY), method='nearest')
        # Interpolating confidence into the meshgrid created above.
        # a.snp[:, 0:2] extracts column 0 and 1 for every row
        gridEuler1 = griddata(
            a.snp[:, 0:2], a.snp[:, 6], (gridX, gridY), method='nearest')
        gridEuler2 = griddata(
            a.snp[:, 0:2], a.snp[:, 7], (gridX, gridY), method='nearest')
        gridEuler3 = griddata(
            a.snp[:, 0:2], a.snp[:, 8], (gridX, gridY), method='nearest')

    eulerStacked = np.stack([gridEuler1, gridEuler2, gridEuler3])
    rightEuler = np.abs(np.diff(eulerStacked, axis=1, append=0)).mean(axis=0)
    upEuler = np.abs(np.diff(eulerStacked, axis=2, append=0)).mean(axis=0)
    # difference between the two sets of euler angles
    misorientationMap = np.max(np.stack([rightEuler, upEuler]), axis=0)

    # Thresholded and Binarized Misorientation Map
    blobFindingArr = np.where(misorientationMap > confidenceTol, 0, 1)
    # If the difference between the two sets of euler angles is
    # less than 1 degree for each angle, the voxel is added to that grain.

    # 2D array bolbFindingArr storing either 0 or 1 - when misorientation is greater than
    # confidence tolerance, set the voxel to 0. Otherwise voxel is set to 1.
    # This may be used to create thresholded and binarized misorientation map.
    # All the 1s would be the potential grain boundaries.

    blobFindingLabels, numFeatures = label(blobFindingArr)
    print("Number of features found using blob finder: ", numFeatures)
    # Lowering the threshold results in identifying more features, vice versa
    # label() uses a four way search to connect neighbours of each non-zero values in input and count as features
    # lables is the labeled 2D array where 1s are considered as objects and 0s are
    # considered background.
    # num_features sums up the number of "objects"

    blobFindingRes = np.float32(blobFindingLabels.copy())
    blobFindingRes[blobFindingRes == 0] = np.nan
    # sets the background to nan - this is used to plot the blob finder results plot

    # dict to store center of mass of 3 euler angles for each grain
    # key -> bolbFindingLabelIdx: int, value -> center of mass of euler grid found: tuple
    grainDict = {}
    for blobFindingLabelIdx in np.sort(np.unique(blobFindingLabels))[1:]:
        # finding coordinates of center of mass for each blob Finder labels
        com = center_of_mass(
            blobFindingArr, blobFindingLabels, blobFindingLabelIdx)
        com = (int(com[0]), int(com[1]))
        grainDict[blobFindingLabelIdx] = (
            gridEuler1[com], gridEuler2[com], gridEuler3[com])

    # (419, 456) array of zeros
    grainIDMap = np.zeros(gridConfidence.shape, dtype=int)

    for grainID in grainDict:
        (comEuler1, comEuler2, comEuler3) = grainDict[grainID]
        tmp = gridConfidence > misOrienMatchingTol
        tmp *= np.absolute(gridEuler1 - comEuler1) < 1
        tmp *= np.absolute(gridEuler2 - comEuler2) < 1
        tmp *= np.absolute(gridEuler3 - comEuler3) < 1
        grainIDMap[tmp] = grainID
    newGrainIDMap = grainIDMap.copy()
    for i, ggg in enumerate(np.unique(grainIDMap)):
        newGrainIDMap[newGrainIDMap == ggg] = i

    grainIDMap = newGrainIDMap
    print(gridX.min(), gridY.min())
    with h5py.File(Cfg.micFile, 'w') as f:
        ds = f.create_dataset(
            "origin", data=np.array([gridX.min(), gridY.min()]))
        ds.attrs[u'units'] = u'mm'
        ds = f.create_dataset(
            "stepSize", data=np.array([voxelSize, voxelSize]))
        ds.attrs[u'units'] = u'mm'
        f.create_dataset("Xcoordinate", data=gridX)
        f.create_dataset("Ycoordinate", data=gridY)
        f.create_dataset("Confidence", data=gridConfidence)
        f.create_dataset("Ph1", data=gridEuler1)
        f.create_dataset("Psi", data=gridEuler2)
        f.create_dataset("Ph2", data=gridEuler3)
        f.create_dataset("GrainID", data=grainIDMap)
    # (419, 456) array contains grain coordinates on this grid after misorientation matching
    gg = np.float32(grainIDMap.copy())
    gg[gg == 0] = np.nan
    fig, ax = plt.subplots(ncols=4, figsize=(20, 7))
    ax[0].imshow(gridConfidence, origin='lower')
    ax[0].set_title('Confidence Map')
    ax[1].imshow(blobFindingArr, origin='lower')
    ax[1].set_title('Thresholded and Binarized Misorientation Map')
    ax[2].imshow(blobFindingRes, origin='lower')
    ax[2].set_title('Blob Finder Results')
    ax[3].imshow(gg, origin='lower')
    ax[3].set_title('Microstructure after Orientation Matching')
    plt.show()
    print('Number of Grains:', grainIDMap.max())

    grain_Ids = np.unique(grainIDMap)[1:]

    grain_posi = []
    # iterate each grain from 1 to 56
    # open grain_template.yml file under Config_Files to write each grain file

    for i in grain_Ids:
        with open(Cfg.grainTemp) as f:
            data = yaml.safe_load(f)
        i = int(i)
        # Return elements chosen from 1 or 0 depending on if elements in grainIDMap equals to i
        locations = np.where(grainIDMap == i, 1, 0)
        # computes center of mass of this grain in the grid based on the locations variable (somewhere in between 419, 456 grid)
        com_ind = np.int32(np.round(center_of_mass(locations)))
        # converts com into the x and y grain position based on grid size
        grain_pos = np.round(
            np.array([gridX[com_ind[0], com_ind[1]], gridY[com_ind[0], com_ind[1]], 0]), 4)
        grain_posi.append(grain_pos)
        euler = np.array([gridEuler1[locations == 1].mean(
        ), gridEuler2[locations == 1].mean(), gridEuler3[locations == 1].mean()])

        data['grainID'] = i
        data['peakFile'] = 'Peak_Files/grain_%03d/Peaks_%03d.hdf5' % (i, i)
        data['recFile'] = 'Rec_Files/grain_%03d/Rec_%03d.hdf5' % (i, i)
        data['pixHitFile'] = 'Rec_Files/grain_%03d/pixHit_%03d.hdf5' % (i, i)
        data['grainPos'] = [float(g) for g in grain_pos]
        data['euler'] = [float(e) for e in euler]

        with open(f'Config_Files/Grain_Files/Grain_%03d.yml' % i, 'w') as file:

            documents = yaml.dump(data, file)

    return grainIDMap.max()
