import numpy as np
from fractions import Fraction
from math import floor
from matplotlib import path
import json
from util.config import Config
import util.RotRep as Rot
import h5py


def frankie_angles_from_g(g, verbo=True, energy=50):
    """
    Converted from David's code, which converted from Bob's code.
    I9 internal simulation coordinates: x ray direction is positive x direction, positive z direction is upward, y direction can be determined by right hand rule.
    I9 mic file coordinates: x, y directions are the same as the simulation coordinates.
    I9 detector images (include bin and ascii files): J is the same as y, K is the opposite z direction.
    The omega is along positive z direction.

    Parameters
    ------------
    g: array
       One recipropcal vector in the sample frame when omega==0. Unit is ANGSTROM^-1.
    energy:
        Experimental parameters. If use 'wavelength', the unit is 10^-10 meter; if use 'energy', the unit is keV.

    Returns
    -------------
    2Theta and eta are in radian, chi, omega_a and omega_b are in degree. omega_a corresponding to positive y direction scatter, omega_b is negative y direction scatter.
    """

    ghat = g / np.linalg.norm(g) # (1, 3)
    sin_theta = np.linalg.norm(g) / (energy * 0.506773182) / 2 # scalar
    cos_theta = np.sqrt(1 - sin_theta ** 2) # scalar
    cos_chi = ghat[2] # scalar
    sin_chi = np.sqrt(1 - cos_chi ** 2) # scalar
    omega_0 = np.arctan2(ghat[0], ghat[1]) # scalar

    if np.fabs(sin_theta) <= np.fabs(sin_chi):
        phi = np.arccos(sin_theta / sin_chi)
        sin_phi = np.sin(phi)
        eta = np.arcsin(sin_chi * sin_phi / cos_theta)
        delta_omega = np.arctan2(ghat[0], ghat[1])
        delta_omega_b1 = np.arcsin(sin_theta / sin_chi)
        delta_omega_b2 = np.pi - delta_omega_b1
        omega_res1 = delta_omega + delta_omega_b1
        omega_res2 = delta_omega + delta_omega_b2
        if omega_res1 > np.pi:
            omega_res1 -= 2 * np.pi
        if omega_res1 < -np.pi:
            omega_res1 += 2 * np.pi
        if omega_res2 > np.pi:
            omega_res2 -= 2 * np.pi
        if omega_res2 < -np.pi:
            omega_res2 += 2 * np.pi
    else:
        return -1
    if verbo == True:
        print('2theta: ', 2 * np.arcsin(sin_theta) * 180 / np.pi)
        print('chi: ', np.arccos(cos_chi) * 180 / np.pi)
        print('phi: ', phi * 180 / np.pi)
        print('omega_0: ', omega_0 * 180 / np.pi)
        print('omega_a: ', omega_res1 * 180 / np.pi)
        print('omega_b: ', omega_res2 * 180 / np.pi)
        print('eta: ', eta * 180 / np.pi)
    return {'chi': np.arccos(cos_chi) * 180 / np.pi, '2Theta': 2 * np.arcsin(sin_theta), 'eta': eta,
            'omega_a': omega_res1 * 180 / np.pi, 'omega_b': omega_res2 * 180 / np.pi, 'omega_0': omega_0 * 180 / np.pi}


class Detector:
    '''
    "Virtual Detector": Parameters used are explained in section 3.2 Results section of the paper
    '''

    def __init__(self, psizeJ=0.00148, psizeK=0.00148, pnJ=2048, pnK=2048,
                 J=0, K=0, trans=np.array([0, 0, 0]),
                 tilt=np.eye(3), config=False):
        if config:
            psizeJ = config.pixelSize*1e-3
            psizeK = config.pixelSize*1e-3
            J = np.array(config.JCenter)
            K = np.array(config.KCenter)
            pnK = config.KPixelNum
            pnJ = config.JPixelNum
            trans = np.array([config.Ldistance, 0., 0.])
            tilt = Rot.EulerZXZ2Mat(np.array(config.tilt)/180.0*np.pi)

        self.__Norm = np.array([0, 0, 1])
        self.__CoordOrigin = np.array([0., 0., 0.])
        self.__Jvector = np.array([1, 0, 0])
        self.__Kvector = np.array([0, -1, 0])
        self.__PixelJ = psizeJ
        self.__PixelK = psizeK
        self.__NPixelJ = pnJ
        self.__NPixelK = pnK
        self.__J0 = J
        self.__K0 = K
        self.__trans0 = trans
        self.__tilt0 = tilt
        self.Move(J, K, trans, tilt)

    @property
    def Norm(self):
        return self.__Norm.copy()

    @property
    def CoordOrigin(self):
        return self.__CoordOrigin.copy()

    @property
    def Jvector(self):
        return self.__Jvector.copy()

    @property
    def Kvector(self):
        return self.__Kvector.copy()

    def Move(self, J, K, trans, tilt):
        self.__CoordOrigin -= J*self.__Jvector * \
            self.__PixelJ+K*self.__Kvector*self.__PixelK
        self.__CoordOrigin = tilt.dot(self.__CoordOrigin)+trans

        self.__Norm = tilt.dot(self.__Norm)
        self.__Jvector = tilt.dot(self.__Jvector)
        self.__Kvector = tilt.dot(self.__Kvector)

    def IntersectionIdx(self, ScatterSrc, TwoTheta, eta, bIdx=True, checkBoundary=True):
        '''
        ScatterSrc: (newGrainX, newGrainY, 0) shape: (1, 3)
        TwoTheta: radian
        eta: radian
        '''
        dist = self.__Norm.dot(self.__CoordOrigin-ScatterSrc)
        scatterdir = np.array([np.cos(TwoTheta), np.sin(
            TwoTheta)*np.sin(eta), np.sin(TwoTheta)*np.cos(eta)])
        InterPos = dist/(self.__Norm.dot(scatterdir))*scatterdir+ScatterSrc
        J = (self.__Jvector.dot(InterPos-self.__CoordOrigin)/self.__PixelJ)
        K = (self.__Kvector.dot(InterPos-self.__CoordOrigin)/self.__PixelK)
        if checkBoundary:
            if 0 <= np.floor(J) < self.__NPixelJ and 0 <= np.floor(K) < self.__NPixelK:
                if bIdx == True:
                    return np.floor(J), np.floor(K)
                else:
                    return J, K
            else:
                return -1
        return J, K

    def IntersectionIdxs(self, ScatterSrcs, TwoThetas, etas, bIdx=True):
        ScatterSrcs = ScatterSrcs.reshape((3, -1))
        TwoThetas = TwoThetas.ravel()
        etas = etas.ravel()
        dists = self.__Norm.dot(self.__CoordOrigin.reshape((3, 1))-ScatterSrcs)
        scatterdirs = np.array([np.cos(TwoThetas), np.sin(
            TwoThetas)*np.sin(etas), np.sin(TwoThetas)*np.cos(etas)]).reshape((3, -1))
        InterPoss = dists/(self.__Norm.dot(scatterdirs)) * \
            scatterdirs+ScatterSrcs
        Js = (self.__Jvector.dot(
            InterPoss-self.__CoordOrigin.reshape((3, 1)))/self.__PixelJ)
        Ks = (self.__Kvector.dot(
            InterPoss-self.__CoordOrigin.reshape((3, 1)))/self.__PixelK)
        if bIdx == False:
            raise 'Not Implemented'
        else:
            Js = np.floor(Js)
            Ks = np.floor(Ks)
            mask = (Js >= 0)*(Js < self.__NPixelJ) * \
                (Ks >= 0)*(Ks < self.__NPixelK)
            return Js, Ks, mask

    def BackProj(self, HitPos, omega, TwoTheta, eta):
        """
        HitPos: ndarray (3,)
                The position of hitted point on lab coord, unit in mm
        """
        scatterdir = np.array([np.cos(TwoTheta), np.sin(
            TwoTheta) * np.sin(eta), np.sin(TwoTheta) * np.cos(eta)])
        t = HitPos[2] / (np.sin(TwoTheta) * np.cos(eta))
        x = HitPos[0] - t * np.cos(TwoTheta)
        y = HitPos[1] - t * np.sin(TwoTheta) * np.sin(eta)
        truex = np.cos(omega) * x + np.sin(omega) * y
        truey = -np.sin(omega) * x + np.cos(omega) * y
        return np.array([truex, truey])

    def Idx2LabCord(self, J, K):
        return J * self.__PixelJ * self.__Jvector + K * self.__PixelK * self.__Kvector + self.__CoordOrigin

    def Reset(self):
        self.__init__(psizeJ=self.__PixelJ, psizeK=self.__PixelK, pnJ=self.__NPixelJ, pnK=self.__NPixelK,
                      J=self.__J0, K=self.__K0, trans=self.__trans0, tilt=self.__tilt0)

    def Print(self):
        print("Norm: ", self.__Norm)
        print("CoordOrigin: ", self.__CoordOrigin)
        print("J vector: ", self.__Jvector)
        print("K vector: ", self.__Kvector)


class CrystalStr:
    '''
    The class CrystalStr is used to construct crystal structures, 
    currently it supports material of gold, silver and Ti7
    
    Attributes:
    ----------
        AtomPos: list
            records position of the atom
        AtomZs: list
            records atomic number
        PrimA: numpy array
            hexagonal unit cell vector A
        PrimB: numpy array
            hexagonal unit cell vector B
        PrimC: numpy array
            hexagonal unit cell vector C
        RecipA: numpy array
            1 x 3 Reciprocal Lattice Vector A
        RecipB: numpy array
            1 x 3 Reciprocal Lattice Vector B
        RecipC: numpy array
            1 x 3 Reciprocal Lattice Vector C
        Gs: list
            List containing G = h * a' + k * b' + l * c'
        hkls: list
            List containing [h, k, l]
    '''
    def __init__(self, material='new', config=None):
        self.AtomPos = []
        self.AtomZs = []
        if material == 'new' and config is not None:

            self.PrimA = config.lattice[0]*np.array(config.basis[0])
            self.PrimB = config.lattice[1]*np.array(config.basis[1])
            self.PrimC = config.lattice[2]*np.array(config.basis[2])

            for atom in config.atoms:
                pos = [eval(a) for a in atom[:3]]
                self.addAtom(pos, atom[-1])

        if material == 'gold':
            self.PrimA = 4.08 * np.array([1, 0, 0])
            self.PrimB = 4.08 * np.array([0, 1, 0])
            self.PrimC = 4.08 * np.array([0, 0, 1])
            self.addAtom([0, 0, 0], 79)
            self.addAtom([0, 0.5, 0.5], 79)
            self.addAtom([0.5, 0, 0.5], 79)
            self.addAtom([0.5, 0.5, 0], 79)
        elif material == 'copper':
            self.PrimA = 3.61 * np.array([1, 0, 0])
            self.PrimB = 3.61 * np.array([0, 1, 0])
            self.PrimC = 3.61 * np.array([0, 0, 1])
            self.addAtom([0, 0, 0], 29)
            self.addAtom([0, 0.5, 0.5], 29)
            self.addAtom([0.5, 0, 0.5], 29)
            self.addAtom([0.5, 0.5, 0], 29)
        # hexagonal unit cell 
        # paper section 2.2 Measuring orientation and strain by X-ray diffraction
        elif material == 'Ti7':
            self.PrimA = config.lattice[0] * np.array([1, 0, 0])
            self.PrimB = config.lattice[1] * \
                np.array([np.cos(np.pi * 2 / 3), np.sin(np.pi * 2 / 3), 0])
            self.PrimC = config.lattice[2] * np.array([0, 0, 1])
            self.addAtom([1 / 3.0, 2 / 3.0, 1 / 4.0], 22)
            self.addAtom([2 / 3.0, 1 / 3.0, 3 / 4.0], 22)
        else:
            pass

    def setPrim(self, x, y, z):
        '''
        Manually set hexagonal unit cell vector A, B, C
        '''
        self.PrimA = np.array(x)
        self.PrimB = np.array(y)
        self.PrimC = np.array(z)

    def addAtom(self, pos, Z):
        '''
        pos: position of the atom
        Z: atomic number
        '''
        self.AtomPos.append(np.array(pos))
        self.AtomZs.append(Z)

    # Equation (35) from Far-field high-energy diffraction microscopy:
    # a tool for intergranular orientation and strain analysis paper
    def getRecipVec(self):
        '''
        Function to compute the 3 shape 1 x 3 Reciprocal Lattice Vectors (RLV)
        '''
        self.RecipA = 2 * np.pi * \
            np.cross(self.PrimB, self.PrimC) / \
            (self.PrimA.dot(np.cross(self.PrimB, self.PrimC)))
        self.RecipB = 2 * np.pi * \
            np.cross(self.PrimC, self.PrimA) / \
            (self.PrimB.dot(np.cross(self.PrimC, self.PrimA)))
        self.RecipC = 2 * np.pi * \
            np.cross(self.PrimA, self.PrimB) / \
            (self.PrimC.dot(np.cross(self.PrimA, self.PrimB)))

    def calStructFactor(self, hkl):
        '''
        The structure factor F_hkl is a mathematical function describing 
        the amplitude and phase of a wave diffracted from crystal lattice 
        planes characterised by Miller indices h,k,l.
        '''
        F = 0
        for ii in range(len(self.AtomZs)):
            F += self.AtomZs[ii] * \
                np.exp(-2 * np.pi * 1j * (hkl.dot(self.AtomPos[ii])))
        return F

    def getGs(self, maxQ):
        '''
        Function to compute reciprocal lattice of G_hkl, 
        where (h, k, l) corresponds to a set of three Miller indices
        RecipA, RecipB, RecipC represent 1 x 3 Reciprocal Lattice Vectors (RLV)
        For loop is used to pick the right choice of RLV such that they satisfy condition where
        a' * a = 2 * pi * DeltaIJ (Kronecker Symbol), Kronecker delta, deltaIJ = 1 for i = j, 0 for i != j

        '''
        self.Gs = []
        self.hkls = []
        # second norm of a 1 by 3 vector equals to sqrt(x1^2 + x2^2 + x3^2)
        maxh = int(maxQ / float(np.linalg.norm(self.RecipA)))
        maxk = int(maxQ / float(np.linalg.norm(self.RecipB)))
        maxl = int(maxQ / float(np.linalg.norm(self.RecipC)))
        for h in range(-maxh, maxh + 1):
            for k in range(-maxk, maxk + 1):
                for l in range(-maxl, maxl + 1):
                    if h == 0 and k == 0 and l == 0:
                        pass
                    else:
                        G = h * self.RecipA + k * self.RecipB + l * self.RecipC
                        if np.linalg.norm(G) <= maxQ:
                            if np.absolute(self.calStructFactor(np.array([h, k, l]))) > 1e-6:
                                self.Gs.append(G)
                                self.hkls.append(np.array([h, k, l]))
        self.Gs = np.array(self.Gs)
        self.hkls = np.array(self.hkls)


def GetProjectedVertex(Det1, sample, orien, etalimit, 
                       grainpos, getPeaksInfo=False, bIdx=True, omegaL=-90, 
                       omegaU=90, energy=50):
    """
    Get the observable projected vertex on a single detector and their G vectors.
    Caution!!! This function only works for traditional nf-HEDM experiment setup.

    Parameters
    ------------
    Det1: Detector
            Remember to move this detector object to correct position first.
    sample: CrystalStr
            Must calculated G list
    orien:  ndarray (3, 3)
            Active rotation matrix of orientation at that vertex
    etalimit: scalar
            Limit of eta value. Usually is about 85.
    grainpos: array (1, 3)
            Position of that vertex in mic file, unit is mm.
    energy: scalar
        X ray energy in the unit of KeV

    Returns
    ------------
    Peaks: ndarray
            N*3 ndarray, records position of each peak. The first column is the J value, second is K value, third is omega value in degree.
    Gs: ndarray
        N*3 ndarray, records  corresponding G vector in sample frame.
    """
    Peaks = []
    Gs = []
    PeaksInfo = []
    # Active rotation matrix of orientation at that vertex dot product
    # sample Gs: (1048, 3) orien: (3, 3)
    # [(3, 3) * (3, 1048)]' = (1048, 3)
    rotatedG = orien.dot(sample.Gs.T).T
    
    # len(rotatedG) = 1048
    for ii in range(len(rotatedG)):
        # g1: (1, 3) array
        g1 = rotatedG[ii]
        res = frankie_angles_from_g(g1, verbo=False, energy=energy) # res is dict
        if res == -1:
            pass
        elif res['chi'] >= 90:
            pass
        elif res['eta'] > etalimit:
            pass
        else:
            if omegaL <= res['omega_a'] <= omegaU:
                omega = res['omega_a'] / 180.0 * np.pi
                newgrainx = np.cos(omega) * \
                    grainpos[0] - np.sin(omega) * grainpos[1]
                newgrainy = np.cos(omega) * \
                    grainpos[1] + np.sin(omega) * grainpos[0]
                idx = Det1.IntersectionIdx(
                    np.array([newgrainx, newgrainy, 0]), res['2Theta'], res['eta'], bIdx)
                if idx != -1:
                    Peaks.append([idx[0], idx[1], res['omega_a']])
                    Gs.append(g1)
                    if getPeaksInfo:
                        PeaksInfo.append({'WhichOmega': 'a', 'chi': res['chi'], 'omega_0': res['omega_0'],
                                          '2Theta': res['2Theta'], 'eta': res['eta'], 'hkl': sample.hkls[ii]})
            if omegaL <= res['omega_b'] <= omegaU:
                omega = res['omega_b'] / 180.0 * np.pi
                newgrainx = np.cos(omega) * \
                    grainpos[0] - np.sin(omega) * grainpos[1]
                newgrainy = np.cos(omega) * \
                    grainpos[1] + np.sin(omega) * grainpos[0]
                idx = Det1.IntersectionIdx(
                    np.array([newgrainx, newgrainy, 0]), res['2Theta'], -res['eta'], bIdx)
                if idx != -1:
                    Peaks.append([idx[0], idx[1], res['omega_b']])
                    Gs.append(g1)
                    if getPeaksInfo:
                        PeaksInfo.append({'WhichOmega': 'b', 'chi': res['chi'], 'omega_0': res['omega_0'],
                                          '2Theta': res['2Theta'], 'eta': -res['eta'], 'hkl': sample.hkls[ii]})
    Peaks = np.array(Peaks)
    Gs = np.array(Gs)
    if getPeaksInfo:
        return Peaks, Gs, PeaksInfo
    return Peaks, Gs


def digitize(xy):
    """
    xy: ndarray shape(4,2)
        J and K indices in float, four points. This digitize method is far from ideal

    Returns
    -------------
    f: list
        list of integer tuples (J,K) that is hitted. (filled polygon)

    """
    p = path.Path(xy)

    def line(pixels, x0, y0, x1, y1):
        if x0 == x1 and y0 == y1:
            pixels.append((x0, y0))
            return
        brev = True
        if abs(y1 - y0) <= abs(x1 - x0):
            x0, y0, x1, y1 = y0, x0, y1, x1
            brev = False
        if x1 < x0:
            x0, y0, x1, y1 = x1, y1, x0, y0
        leny = abs(y1 - y0)
        for i in range(leny + 1):
            if brev:
                pixels.append(
                    tuple((int(round(Fraction(i, leny) * (x1 - x0))) + x0, int(1 if y1 > y0 else -1) * i + y0)))
            else:
                pixels.append(
                    tuple((int(1 if y1 > y0 else -1) * i + y0, int(round(Fraction(i, leny) * (x1 - x0))) + x0)))

    bnd = p.get_extents().get_points().astype(int)
    ixy = xy.astype(int)
    pixels = []
    line(pixels, ixy[0, 0], ixy[0, 1], ixy[1, 0], ixy[1, 1])
    line(pixels, ixy[1, 0], ixy[1, 1], ixy[2, 0], ixy[2, 1])
    line(pixels, ixy[2, 0], ixy[2, 1], ixy[3, 0], ixy[3, 1])
    line(pixels, ixy[3, 0], ixy[3, 1], ixy[0, 0], ixy[0, 1])
    points = []
    for jj in range(bnd[0, 0], bnd[1, 0] + 1):
        for kk in range(bnd[0, 1], bnd[1, 1] + 1):
            points.append((jj, kk))
    points = np.asarray(points)
    mask = p.contains_points(points)

    ipoints = points[mask]

    f = list([tuple(ii) for ii in ipoints])
    f.extend(pixels)

    return f
