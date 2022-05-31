# StrainRecon-cuda_python


This software is an intra-granular strain tensor reconstruction toolkit for near-field high-energy X-ray diffraction microscopy ([nf-HEDM](https://www.andrew.cmu.edu/user/suter/3dxdm/3dxdm.html)). This readme file contains information about its [Usage](#usage), its [Dependencies](#dependencies) on other libraries, the [Structure](#package-structure) of the package, the [Formats](#file-formats) of its input and output files. This is the implementation of https://github.com/Yufeng-shen/StrainRecon using cuda-python.

## Dependencies
|            | 
| ------------- |
| python = 3|
| jupyterlab     | 
| cuda-python     | 
| numpy      | 
| pyaml      | 
| matplotlib |
| scipy     |
| h5py       |

These dependencies are all contained in __nf_strn_recon_env.yml__.
The conda environment can be created using the following command: "conda env create -f environment.yml"
## Usage



### Step 1: Peak File Generation

- Case 1: Synthetic Bragg peak patterns
    1. Fill out __Config_Files/Config.yml__ with the detector and crystal parameters from the hexomap configuration files (This could be changed to write automatically from hexomap files in the future)
    2.Run at leas the first cell of __Calibration.ipynb__ to generate the microstructure file. If additional calibration required, run the rest of the cells (Instructions in the notebook)
    3. Run __Make_PeakFiles.ipynb__


### Step 2: Reconstruction

1. Reconstruct all grains with __Reconstruct_all.ipynb__ 

  
## File formats
In the whole reconstruction procedure, there are five kinds of files:

- Configure file: File in the folder ConfigFiles/. It contains the information for simulation or reconstruction, e.g. geometry parameters, sample information, file paths, etc. Its format is described in the templates.

- Grain file: Contains reconstruction information specific to each grain. Can be seen in the template

- [Peak file](#peakfile-format): Files in the folders RealPeaks/ and SimResult/. They store the Bragg peak patterns from a single grain. They can be the output of simulation or extract from experimental images.

- [Microstructure file](#micfile-format): Files in the folder micFile/. They are the input for both reconstruction and simulation. They store the grain ID map, orientation map, and strain map (only for simulation).

- [Reconstruction file](#recfile-format): Files in the folder recFile/. They contain the reconstructed strain values.

### peakFile format
It is a hdf5 file, which stores the Bragg peak patterns in windows along with other information about the experiment. As of now, the window size is fixed as (&Delta;J=300, &Delta;K=160, &Delta;&Omega;=45), the units are number of pixels, number of pixels, and number of frames. The datasets are: (assuming there are N peaks recorded)

- "/Gs": shape of (N,3). The corresponding reciprocal vectors before distortion. 

- "/MaxInt": shape of (N). The maximum intensities of peaks.

- "/OrienM": shape of (3,3). The average orientation of the grain.

- "/Pos": shape of (3). The center of mass of the grain.

- "/avg_distortion": shape of (3,3). The strain already considered in "/Gs".

- "/limits": shape of (N,5). The pixel coordinates of the window and the &Omega; indices of the first frame.

- "/whichOmega": shape of (N). Indicate is the first or second Bragg peak of that reciprocal vectors.

- "/Imgs/Im0": shape of (160,300,45). The diffraction pattern of the first Bragg peak.

...

An example of peak file (partial) is in the folder RealPeaks/. It is only used for demonstrating the format, so we removed the datasets "/Imgs/Im1", "/Imgs/Im2"... "/Imgs/Im91" to decrease the file size. You can also use the SimDemo.py script to generate a peak file.

### micFile format
It is a hdf5 file, which contains following datasets: (assuming the mesh on sample cross section  has N<sub>x</sub> columns and N<sub>y</sub> rows)

- "/origin": shape of (2). The position (x,y) of the bottom left corner of the mesh, in the unit of millimeter.

- "/stepSize": shape of (2). The step size (&delta;x,&delta;y) of the mesh.

- "/Confidence": shape of (N<sub>y</sub>,N<sub>x</sub>). The confidence from I9 reconstruction.

- "/Ph1": shape of (N<sub>y</sub>,N<sub>x</sub>). The first Euler angles on the voxels.

- "/Psi": shape of (N<sub>y</sub>,N<sub>x</sub>). The second Euler angles on the voxels.

- "/Ph2": shape of (N<sub>y</sub>,N<sub>x</sub>). The third Euler angles on the voxels.

- "/GrainID": shape of (N<sub>y</sub>,N<sub>x</sub>). The grain IDs on the voxels.

We also recommend store the following two datasets:

- "/Xcoordinate": shape of (N<sub>y</sub>,N<sub>x</sub>). The x coordinates of the voxels, in the unit of millimeter.

- "/Ycoordinate": shape of (N<sub>y</sub>,N<sub>x</sub>). The y coordinates of the voxels, in the unit of millimeter.

To be used for simulation, following datasets for the elastic strain components are also needed:

- "/E11": shape of (N<sub>y</sub>,N<sub>x</sub>).

- "/E12": shape of (N<sub>y</sub>,N<sub>x</sub>).

- "/E13": shape of (N<sub>y</sub>,N<sub>x</sub>).

- "/E22": shape of (N<sub>y</sub>,N<sub>x</sub>).

- "/E23": shape of (N<sub>y</sub>,N<sub>x</sub>).

- "/E33": shape of (N<sub>y</sub>,N<sub>x</sub>).

### recFile format
It is a hdf5 file, which contains five important datasets: (assuming there are N voxels in the grain)

- "/Phase2_S": shape of (N,3,3). The reconstructed distortion matrix in reciprocal space on voxels. (matrix __D__ in my thesis, p.g. 56)

- "/realS": shape of (N,3,3). The reconstructed strain tensor on voxels. (matrix __V__ in my thesis, p.g. 55)

- "/realO": shape of (N,3,3). The reconstructed orientation matrices on voxels. (matrix __R__ in my thesis, p.g. 55)

- "/x": shape of (N). The x coordinates of voxels, in the unit of millimeter.

- "/y": shape of (N). The y coordinates of voxels, in the unit of millimeter.

Other information about the reconstruction may also installed.

## Package Structure
- Basics
    - strain_device_mjw_3.cu : CUDA kernel functions.
    - util/config.py : wrapper for configuration files.
    - util/initializer_mjw.py : constructs GPU related functions and objects, along with the Detector and Material objects.
    - util/reconstructor_mjw.py : reconstructs the intra-granular strain tensor from Bragg peak patterns.
    - util/optimizers_mjw.py : the minimization algorithms used in the reconstruction.
    - util/reconstruct_all.py : file containing functions to run the whole reconstruction routine for all grains
    
- Scripts
    - Calibration.ipynb : calibrates the geometry parameters in the experimental setup.
    - Make_PeakFiles.ipynb : extracts the windows around the Bragg peaks from the grain we want to reconstruct and makes the peak files.
    - Reconst_all.ipynb : Reconstructs all grains
    
- Folders
    - AuxData/ : outputs from standard nf-HEDM, the voxelized orientations on a cross section of the sample.
    - Rec_Files/ : outputs from Reconstruct_all.ipynb, the reconstructions.
    - Peak_Files/ : outputs from Make_PeakFiles, the peak files for the reconstructions.
    - micFile/ : microstructure files from FFT simulation or regenerated from the files in AuxData/.
    - ConfigFiles/ : configuration files for reconstruction or simulation.
    - util/ : some basic functions related to nf-HEDM.
  
## TODO
The function "find_grains" in __util/make_micFile.py__ could use some tuning. Currently, it thresholds the confidence map and uses a blob finder to locate the grains. It then takes the euler angles from the center of mass of each blob and searches through all other voxels for matching euler angle values. If the difference between the two sets of euler angles is less than 1 degree for each angle, the voxel is added to that grain. The user can see that grains are missed by adjusting the values of "blob_finding_conf" and "ori_finding_conf" in the first cell of __Calibration.ipynb__.


