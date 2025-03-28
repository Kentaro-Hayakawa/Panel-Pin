"""
Functions for analysis of rigid origami using panel-pin model
<Reference>
K. Hayakawa and M. Ohsaki, Panel-pin model for kinematic and equilibrium analysis of rigid origami,
J. Int. Assoc. Shell Spat. Struct., Vol. 64 (4), pp. 278-288, 2023, https://doi.org/10.20898/j.iass.2023.025


Explanation of input paramters for class PanelPin
vertex: (3, N) array
        Initial coordinates of panel vertices.
        The i-th column corresponds to the position of vertex i in Face parameter.
        ex: vertex = numpy.array([[-1.0,1.0,-2.0,2.0,2.0,-2.0], [0.0,0.0,-1.0,-1.0,1.0,1.0], [0.0,0.0,0.0,0.0,0.0,0.0]])
face: list of interger list
        List of indices lists of face (panel) vertices.
        Each list contains vertex indices arranged in counter-clockwise.
        ex: face = [[0,5,2], [0,2,3,1], [1,3,4], [0,1,4,5]]
crease: (4, N) array
        Indices of end points and adjacent faces of each crease line.
        The first and second rows represent the start and end points (vertices) indices, and the third and fourth rows represent the left and right faces (panels).
        "Left" and "right" sides of a crease line are determined by seeing from the start point to the end point of the crease line.
        ex: crease = numpy.array([[0,0,1,1,0], [1,2,3,4,5], [3,1,2,3,0], [1,0,1,2,3]])
glued_panels: (2, N) array, optional
        Indices of faces (panels) that are glued togther.
        Each columns of guled_panels parameter corresponds to each pair of glued panel indices.
        Each pair of glued panels shares the same translational and rotational displacement.
        ex: glued_panels = numpy.array([[2,4,13,19], [15,17,24,30]])
stiff_per_length: float or numpy.array, optional
        Rotational stiffness per unit length of crease lines.
        Different stiffness can be assigned to each crease line by using numpy.array with the length of the number of crease lines.
        Defalut value is 1.0 for all crease lines.
weight_per_area: float or numpy.array, optional
        Weight per unit area of faces (panels).
        Different weight can be assigned to each panel by using numpy.array with the length of the number of panels.
        Defalut value is 0.0 for all panels.
boundary_conds: tuple of dictionaries, optional
        Boundary conditions assigned to panel and vertex displacements.
        There are three types; "fix", "disp", and "load" corresponding to support conditions, forced displacements and concentrated loads, respectively.
        ex:
        boundary_conds = ({'type': "fix", 'object': "face", 'index': np.array([[0,0,0,0,0,0],[0,1,2,3,4,5]])},
                          {'type': "disp", 'object': "face", 'index': np.array([[1,1,1],[0,1,2]]), 'target': np.array([1.0,1.0,1.0])},
                          {'type': "load", 'object': "face", 'index': np.array([[2,2,2],[0,1,2]]), 'target': np.array([1.0,1.0,1.0])},
                          {'type': 'fix', 'object': 'vertex', 'index': np.array([[0,0,0],[0,1,2]])},
                          {'type': 'disp', 'object': 'vertex', 'index': np.array([[1,1,1],[0,1,2]]), 'target': np.array([1.0,1.0,1.0])},
                          {'type': 'load', 'object': 'vertex', 'index': np.array([[2,2,2],[0,1,2]]), 'target': np.array([1.0,1.0,1.0])},
                         )
"""


import numpy as np
import math

##################################################
# Main class of panel pin model
##################################################
class PanelPin:

    ##################################################
    ### constructor
    ##################################################
    def __init__(self,
                 vertex,
                 face,
                 crease,
                 glued_panels=np.zeros((2,0)),
                 stiff_per_length=1.0,
                 weight_per_area=0.0,
                 boundary_conds=()
                 ):
        # initialize variables
        self._vert = vertex
        self._face = face
        self._crease = crease
        self._glued = glued_panels
        self._nv = vertex.shape[1]
        self._nf = len(face)
        self._nc = crease.shape[1]
        self._ng = glued_panels.shape[1]
        self._stiffpl = np.ravel(np.array([stiff_per_length]))
        if len(self._stiffpl) > self._nc:
            self._stiffpl = self._stiffpl[0:self._nc]
        elif len(self._stiffpl) < self._nc:
            self._stiffpl = np.tile(self._stiffpl, math.ceil(self._nc/len(self._stiffpl)))[0:self._nc]
        self._weightpa = np.ravel(np.array([weight_per_area]))
        if len(self._weightpa) > self._nf:
            self._weightpa = self._weightpa[0:self._nf]
        elif len(self._weightpa) < self._nf:
            self._weightpa = np.tile(self._weightpa, math.ceil(self._nf/len(self._weightpa)))[0:self._nf]
        # initialize boundary condition paramenters
        self._ffix = np.zeros(0, int)     # indices of fixed panel DOF
        self._fdisp = np.zeros(0, int)    # indices of panel DOF where forced displacements are assigned
        self._df = np.zeros(0, int)       # components of panel forced displacement vectors
        self._fload = np.zeros(0, int)    # indices of loaded panel DOF
        self._pf = np.zeros(0, int)       # components of panel load vectors
        self._vfix = np.zeros(0, int)     # indices of fixed vertex DOF
        self._vdisp = np.zeros(0, int)    # indices of vertex DOF where forced displacements are assigned
        self._dv = np.zeros(0, int)       # components of vertex forced displacement vectors
        self._vload = np.zeros(0, int)    # indices of loaded vertex DOF
        self._pv = np.zeros(0, int)       # components of vertex load vectors
        # check boundary conditions
        if len(boundary_conds) > 0:
            self._check_boundary(boundary_conds)
        # construct panel-pin model
        self.center, self._corner, self._normal, self._binormal, self._angle, \
        self._sum_angle, self._stiff, self._weight, self._refz, self._fold_index, \
        self._conn_index, self._dispv_index, self._ivar, self._iadd, self._iall \
        = self._construct_model(self._vert, self._face, self._crease,
                                self._glued, self._ffix, self._stiffpl, self._weightpa)
        # initialize generalized panel displacement
        self.disp = np.zeros(6*self._nf, float)
        # initialize hinge spring rotation angle
        self.init_angle = self.foldangle()


    ### check and rearrange input baundary conditions
    def _check_boundary(self, boundary_conds):
        # assign boundary conditions
        for bnd in boundary_conds:
            if bnd['type'] == "fix":
                if bnd['object'] == "face":
                    self._ffix = np.append(self._ffix, bnd['index'][0] + self._nf * bnd['index'][1])
                elif bnd['object'] == "vertex":
                    self._vfix = np.append(self._vfix, bnd['index'][0] + self._nv * bnd['index'][1])
                else:
                    raise ValueError("Object is not correctly assigned for fixed boundary condition.")
            elif bnd['type'] == "disp":
                if bnd['object'] == "face":
                    self._fdisp = np.append(self._fdisp, bnd['index'][0] + self._nf * bnd['index'][1])
                    self._df = np.append(self._df, bnd['target'])
                elif bnd['object'] == "vertex":
                    self._vdisp = np.append(self._vdisp, bnd['index'][0] + self._nv * bnd['index'][1])
                    self._dv = np.append(self._dv, bnd['target'])
                else:
                    raise ValueError("Object is not correctly assigned for forced displacement.")
            elif bnd['type'] == "load":
                if bnd['object'] == "face":
                    self._fload = np.append(self._fload, bnd['index'][0] + self._nf * bnd['index'][1])
                    self._pf = np.append(self._pf, bnd['target'])
                elif bnd['object'] == "vertex":
                    self._vload = np.append(self._vload, bnd['index'][0] + self._nv * bnd['index'][1])
                    self._pv = np.append(self._pv, bnd['target'])
                else:
                    raise ValueError("Object is not correctly assigned for external load.")
            else:
                raise ValueError("Invalid boundary condition type:  " + bnd['type'])
        # check validity of inputs
        self._ffix = np.unique(self._ffix)
        _, countsd = np.unique(self._fdisp, return_counts=True)
        _, countsl = np.unique(self._fload, return_counts=True)
        countsd = np.append(countsd, 0)
        countsl = np.append(countsl, 0)
        if (np.max(countsd) > 1) or (np.max(countsl) > 1):
            raise ValueError("Multiple boundary conditions are assigned to same panel DOF.")
        self._vfix = np.unique(self._vfix)
        _, countsd = np.unique(self._vdisp, return_counts=True)
        _, countsl = np.unique(self._vload, return_counts=True)
        countsd = np.append(countsd, 0)
        countsl = np.append(countsl, 0)
        if (np.max(countsd) > 1) or (np.max(countsl) > 1):
            raise ValueError("Multiple boundary conditions are assigned to same vertex DOF.")




    ##################################################
    # general functions
    ##################################################
    ### reduced row echelon form of matrix
    def _rref(self,
              arr,                 # (N, M) array, float; Main matrix whose RREF is calculated
              subarr=np.zeros(0),  # (N, M) array, float; Sub matrix transformed in the same manner as that of main matrix
              tol=1e-12            # float; Threshold below which coefficient values are considered zero
              ):
        arr_new = np.copy(arr)
        arr = arr_new
        rows, cols = arr.shape
        pivot_pos = np.zeros((0,2),dtype=int)
        rows_pos = np.arange(rows)
        if subarr.shape[0] == 0:
            subarr = np.identity(rows)
        else:
            subarr_new = np.copy(subarr)
            subarr = subarr_new
        r = 0
        for c in range(cols):
            # find the pivot point
            pivot = np.argmax(np.abs(arr[r:rows,c]))+r
            maxc = np.abs(arr[pivot,c])
            # skip column c if maxc <= tol
            if maxc <= tol:
                ## make column c accurately zero
                arr[r:rows,c] = 0.
            else:
                pivot_pos = np.append(pivot_pos, [[r,c]], axis=0)
                # swap current row and pivot row
                if pivot != r:
                    arr[[r,pivot],:] = arr[[pivot,r],:]
                    rows_pos[[r,pivot]] = rows_pos[[pivot,r]]
                    subarr[[r,pivot],:] = subarr[[pivot,r],:]
                # normalize pivot row
                div = np.copy(arr[r,c])
                arr[r,c:cols] = arr[r,c:cols]/div
                subarr[r,:] = subarr[r,:]/div
                v = arr[r,c:cols]
                subv = subarr[r,:]
                # eliminate the current column
                # above r
                if r > 0:
                    dif = np.copy(arr[0:r,c])
                    arr[0:r,c:cols] = arr[0:r,c:cols] - np.outer(dif,v)
                    subarr[0:r,:] = subarr[0:r,:] - np.outer(dif,subv)
                # below r
                if r < rows:
                    dif = np.copy(arr[r+1:rows,c])
                    arr[r+1:rows,c:cols] = arr[r+1:rows,c:cols] - np.outer(dif,v)
                    subarr[r+1:rows,:] = subarr[r+1:rows,:] - np.outer(dif,subv)
                r += 1
                # check if done
            if r == rows or c == cols:
                # eliminate nearly zero element
                for i in range(rows):
                    for j in range(cols):
                        if abs(arr[i,j]) <= tol:
                            arr[i,j] = 0.
                break
        pivot_pos = pivot_pos[np.argsort(pivot_pos, axis=0)[:,0]]
        return arr, pivot_pos, rows_pos, subarr


    ### cross product matrix
    def _crossmatrix(self,
                     vec    # (3, N) array, float; List of 3-D vectors
                     ):
        # construct outer product matrix
        mat = np.zeros((3,3,vec.shape[1]),float)
        mat[0,1,:] = -vec[2,:]
        mat[0,2,:] =  vec[1,:]
        mat[1,0,:] =  vec[2,:]
        mat[1,2,:] = -vec[0,:]
        mat[2,0,:] = -vec[1,:]
        mat[2,1,:] =  vec[0,:]
        return mat


    ### 3D rotation matrix around rotation vector
    def _rot3D(self,
               vec    # (3, N) array, float; List of 3-D vectors
               ):
        # construct rotation matrix
        angle = np.linalg.norm(vec, axis=0)
        rot = np.zeros((3,3,vec.shape[1]),float)
        # angle > 0
        jj = np.where(angle > 0.)[0]
        if len(jj) > 0:
            nn = vec[:,jj]/angle[jj]
            nx = self._crossmatrix(nn)
            nnn = np.stack([nn,nn,nn],axis=1)*nn
            cos = np.cos(angle[jj])
            sin = np.sin(angle[jj])
            ee = np.zeros((3,3,len(jj)))
            ee[:,:,:] = np.identity(3)[:,:,None]
            rot[:,:,jj] = cos*ee+(1.-cos)*nnn+sin*nx
        # angle = 0
        rot[:,:,np.where(angle <= 0.)[0]] = np.identity(3)[:,:,None]
        return rot


    ### first-order derivative of rotation matrix with respect to vec[ii]
    def _drot3D(self,
                vec,    # (3, N) array, float; List of 3-D vectors
                ii      # int; index of DOF to compute derivatives
                ):
        # construct derivative of rotation matrix
        angle = np.linalg.norm(vec, axis=0)
        drot = np.zeros((3,3,vec.shape[1]),float)
        # angle > 0
        jj = np.where(angle > 0.)[0]
        if len(jj) > 0:
            nn = vec[:,jj]/angle[jj]
            nx = self._crossmatrix(nn)
            nnn = np.stack([nn,nn,nn],axis=1)*nn
            cos = np.cos(angle[jj])
            sin = np.sin(angle[jj])
            ee = np.zeros((3,3,len(jj)),float)
            ee[:,:,:] = np.identity(3)[:,:,None]
            ev = np.zeros((3,len(jj)),float)
            ev[ii,:] = 1.
            ex = self._crossmatrix(ev)
            ne = np.stack([nn,nn,nn],axis=1)*ev
            en = np.stack([ev,ev,ev],axis=1)*nn
            drot[:,:,jj] += -nn[ii,:]*(sin*ee - (sin-2.*(1.-cos)/angle[jj])*nnn - (cos-sin/angle[jj])*nx)
            drot[:,:,jj] += (1.-cos)/angle[jj]*(ne+en) + sin/angle[jj]*ex
        # angle = 0
        jj = np.where(angle <= 0.)[0]
        if len(jj) > 0:
            ev = np.zeros((3,len(jj)),float)
            ev[ii,:] = 1.
            ex = self._crossmatrix(ev)
            drot[:,:,jj] += ex
        return drot


    ### second-order derivative of rotation matrix with respect to vec[i1] and vec[i2]
    def _ddrot3D(self,
                 vec,    # (3, N) array, float; List of 3-D vectors
                 i1,     # int; first index of DOF to compute derivatives
                 i2      # int; second index of DOF to compute derivatives
                 ):
        # construct derivative of rotation matrix
        angle = np.linalg.norm(vec, axis=0)
        ddrot = np.zeros((3,3,vec.shape[1]),float)
        # angle > 0
        jj = np.where(angle > 0.)[0]
        if len(jj) > 0:
            nn = vec[:,jj]/angle[jj]
            nx = self._crossmatrix(nn)
            nnn = np.stack([nn,nn,nn],axis=1)*nn
            cos = np.cos(angle[jj])
            sin = np.sin(angle[jj])
            ee = np.zeros((3,3,len(jj)),float)
            ee[:,:,:] = np.identity(3)[:,:,None]
            ev1 = np.zeros((3,len(jj)),float)
            ev1[i1,:] = 1.
            ex1 = self._crossmatrix(ev1)
            ne1 = np.stack([nn,nn,nn],axis=1)*ev1
            en1 = np.stack([ev1,ev1,ev1],axis=1)*nn
            ev2 = np.zeros((3,len(jj)),float)
            ev2[i2,:] = 1.
            ex2 = self._crossmatrix(ev2)
            ne2 = np.stack([nn,nn,nn],axis=1)*ev2
            en2 = np.stack([ev2,ev2,ev2],axis=1)*nn
            ee12 = np.stack([ev1,ev1,ev1],axis=1)*ev2
            ee21 = np.stack([ev2,ev2,ev2],axis=1)*ev1
            nn12 = nn[i1,:]*nn[i2,:]
            if i1 == i2:
                del12 = np.ones(len(jj),float)
            else:
                del12 = np.zeros(len(jj),float)
            ddrot[:,:,jj] += ( -nn12*cos - (del12-nn12)*sin/angle[jj] )*ee
            ddrot[:,:,jj] += ( nn12*cos + (del12-5.*nn12)*sin/angle[jj] - 2.*(del12-4.*nn12)*(1.-cos)/angle[jj]**2 )*nnn
            ddrot[:,:,jj] += ( -nn12*sin + (del12-3.*nn12)*(cos/angle[jj]-sin/angle[jj]**2) )*nx
            ddrot[:,:,jj] += ( sin/angle[jj] - 2.*(1.-cos)/angle[jj]**2 )*( nn[i2,:]*(en1+ne1) + nn[i1,:]*(en2+ne2) )
            ddrot[:,:,jj] += ( (1.-cos)/angle[jj]**2 )*( ee12 + ee21 )
            ddrot[:,:,jj] += ( cos/angle[jj] - sin/angle[jj]**2 )*( nn[i1,:]*ex2 + nn[i2,:]*ex1 )
        # angle = 0
        jj = np.where(angle <= 0.)[0]
        if len(jj) > 0:
            ee = np.zeros((3,3,len(jj)),float)
            ee[:,:,:] = np.identity(3)[:,:,None]
            ev1 = np.zeros((3,len(jj)),float)
            ev1[i1,:] = 1.
            ev2 = np.zeros((3,len(jj)),float)
            ev2[i2,:] = 1.
            ee12 = np.stack([ev1,ev1,ev1],axis=1)*ev2
            ee21 = np.stack([ev2,ev2,ev2],axis=1)*ev1
            if i1 == i2:
                ddrot[:,:,jj] += -ee + (ee12 + ee21)/2.
            else:
                ddrot[:,:,jj] += (ee12 + ee21)/2.
        return ddrot




    ##################################################
    ### initial construction of panel-pin model
    ##################################################
    def _construct_model(self,
                         vert,     # (3, N) array, float; Vertex Coordinates
                         face,     # List of list, int; List of face vertices
                         crease,   # (4, N) array, int; Endpoints of crease lines and adjacent faces
                         glued,    # (2, N) array, int; Glued faces
                         ffix,     # 1-D array, int; Fixed face displacement DOFs
                         stiffpl,  # 1-D array, float; Rotation stiffness of crease line per unit length
                         weightpa  # 1-D array, float; Weight of panels per unit area
                         ):
        # number of vertices, face, crease lines, and pairs of glued panels
        nv = vert.shape[1]
        nf = len(face)
        nc = crease.shape[1]
        ng = glued.shape[1]
        # list of connectivity of panels
        connect = np.zeros((3,0), int)    # 1st row: vertex indices, 2nd & 3rd rows: panel indices sharing vertex in 1st row
        for i in range(nv):
            include = np.zeros(0, int)
            for j in range(nf):
                if i in face[j]:
                    include = np.append(include, j)
            include = np.unique(include)
            if len(include) == 2:
                connect = np.append(connect, np.array([[i],[include[0]],[include[1]]]), axis=1)
            elif len(include) >= 3:
                for j in range(len(include)-1):
                    for k in range(j+1,len(include)):
                        connect = np.append(connect, np.array([[i],[include[j]],[include[k]]]), axis=1)
        #  rotation stiffness of crease lines and unit vectors along crease lines
        direction = vert[:,crease[1]] - vert[:,crease[0]]
        length = np.linalg.norm(direction, axis=0)
        direction = direction/length
        stiff = stiffpl*length
        # area, barycenter and unit normal of faces
        area = np.zeros(nf)
        bary = np.zeros((3,nf))
        normal = np.zeros((3,nf))
        for i in range(nf):
            for j in range(1,len(face[i])-1):
                v1 = vert[:,face[i][j]] - vert[:,face[i][0]]
                v2 = vert[:,face[i][j+1]] - vert[:,face[i][0]]
                normal[:,i] += np.cross(v1,v2)/2.
            area[i] = np.linalg.norm(normal[:,i])
            normal[:,i] /= area[i]
            for j in range(1,len(face[i])-1):
                v1 = vert[:,face[i][j]] - vert[:,face[i][0]]
                v2 = vert[:,face[i][j+1]] - vert[:,face[i][0]]
                baryi = (vert[:,face[i][0]] + vert[:,face[i][j]] + vert[:,face[i][j+1]])/3.
                baryi *= np.dot(normal[:,i],np.cross(v1,v2)/2.)
                bary[:,i] += baryi
            bary[:,i] /= area[i]
        # weight of panels
        weight = weightpa*area
        # take average of barycenters if panels are glued
        for i in range(glued.shape[1]):
            bary0 = np.copy(bary[:,glued[0,i]])
            bary1 = np.copy(bary[:,glued[1,i]])
            bary[:,glued[0,i]] = (bary0 + bary1)/2.
            bary[:,glued[1,i]] = (bary0 + bary1)/2.
        # dimensions of panels
        angle = np.zeros(0)         # inner angles of panels
        corner = np.zeros((3,0))    # vectors from barycenters to vertices of panels
        corner_index = np.full((nf,nv), -1, dtype=int)   # index list of components of angle and corner
        k = 0
        for i in range(nf):
            for j in range(len(face[i])):
                r0 = vert[:,face[i][j]]
                r1 = vert[:,face[i][(j+1)%len(face[i])]]
                r2 = vert[:,face[i][(j-1)%len(face[i])]]
                v1 = (r1-r0)/np.linalg.norm(r1-r0)
                v2 = (r2-r0)/np.linalg.norm(r2-r0)
                angle = np.append(angle, np.arccos(np.clip(np.dot(v1,v2), -1, 1)))
                corner = np.append(corner, (r0-bary[:,i]).reshape([3,1]), axis=1)
                corner_index[i,face[i][j]] = k
                k += 1
        angle = np.append(angle, 0.)
        corner = np.append(corner, np.zeros((3,1)), axis=1)
        # reference height for computation of gravity potential
        refz = bary[2]
        # index list for computation of folding angles and incompatibility vector for panel connectivity
        fold_index = crease[2:4]                            # panel indices adjacent to crease lines
        conn_index = np.zeros((4,connect.shape[1]), int)    # panel indices sharing same vertex and indices of vectors from barycenter to vertices
        for i in range(connect.shape[1]):
            conn_index[0,i] = connect[1,i]
            conn_index[1,i] = connect[2,i]
            conn_index[2,i] = corner_index[connect[1,i],connect[0,i]]
            conn_index[3,i] = corner_index[connect[2,i],connect[0,i]]
        # index list for computation of vertex displacement
        count = np.zeros(nv, int)
        dispv_index = np.full((2,nv), -1, dtype=int)    # even row: face indices, odd row: corner indices
        for i in range(nf):
            for j in face[i]:
                count[j] += 1
                if dispv_index.shape[0] < 2*count[j]:
                    dispv_index = np.append(dispv_index, np.full((2,nv), -1, dtype=int), axis=0)
                dispv_index[2*(count[j]-1),j] = i
                dispv_index[2*(count[j]-1)+1,j] = corner_index[i,j]
        # sum of face inner angles around vertices
        sum_angle = np.sum(angle[dispv_index[::2]], axis=0)
        # cross products of crease line vectors and face normals
        binormal = np.cross(direction, normal[:,fold_index[0]], axis=0)
        # indices of independent face displacement
        ivar = np.zeros(0, int)         # indices of free (independent) panel displacement DOFs
        iadd = np.zeros(0, int)         # indices of glued dependent panel displacement DOFs
        iall = np.zeros((6*nf), int)    # indices of panel displacement DOFs in ivar for restoring all DOFs
        for j in range(6):
            for i in range(nf):
                if i+nf*j in ffix:
                    iall[i+nf*j] = -1
                elif i in glued[0]:
                    k = glued[1,np.where(glued[0]==i)[0][0]]
                    if k+nf*j not in ffix:
                        ivar = np.append(ivar, i+nf*j)
                        iadd = np.append(iadd, k+nf*j)
                        iall[i+nf*j] = len(ivar)-1
                        iall[k+nf*j] = len(ivar)-1
                    else:
                        iall[i+nf*j] = -1
                        iall[k+nf*j] = -1
                elif i in glued[1]:
                    continue
                else:
                    ivar = np.append(ivar, i+nf*j)
                    iadd = np.append(iadd, -1)
                    iall[i+nf*j] = len(ivar)-1
        return bary, corner, normal, binormal, angle, sum_angle, stiff, weight, refz, fold_index, conn_index, dispv_index, ivar, iadd, iall




    ##################################################
    ### folding angles at hinges and their derivatives
    ##################################################
    ### folding angle
    def foldangle(self):
        dispf = self.disp.reshape([6,self._nf])
        # rotation matrices
        rot0 = self._rot3D(dispf[3:6,self._fold_index[0]])
        rot1 = self._rot3D(dispf[3:6,self._fold_index[1]])
        # rotated vectors
        bb0 = np.sum(rot0*self._binormal, axis=1)
        nn0 = np.sum(rot0*self._normal[:,self._fold_index[0]], axis=1)
        nn1 = np.sum(rot1*self._normal[:,self._fold_index[1]], axis=1)
        # cosine and sine of folding angles
        cos = np.sum(nn1*nn0, axis=0)
        sin = np.sum(nn1*bb0, axis=0)
        # folding angles
        rho = np.arctan2(sin,cos)
        return rho


    ### first-order derivative of folding angle
    def difffoldangle(self):
        dispf = self.disp.reshape([6,self._nf])
        drho = np.zeros((self._nc,6*self._nf))
        ii = np.arange(self._nc)
        # rotation matrices
        rot0 = self._rot3D(dispf[3:6,self._fold_index[0,:]])
        rot1 = self._rot3D(dispf[3:6,self._fold_index[1,:]])
        # rotated vectors
        bb0 = np.sum(rot0*self._binormal, axis=1)
        nn0 = np.sum(rot0*self._normal[:,self._fold_index[0]], axis=1)
        nn1 = np.sum(rot1*self._normal[:,self._fold_index[1]], axis=1)
        # cosine and sine of folding angles
        cos = np.sum(nn1*nn0, axis=0)
        sin = np.sum(nn1*bb0, axis=0)
        ll2 = sin**2 + cos**2
        for i in range(3):
            # derivatives of rotation matrices
            drot0 = self._drot3D(dispf[3:6,self._fold_index[0]],i)
            drot1 = self._drot3D(dispf[3:6,self._fold_index[1]],i)
            # derivatives of rotated vectors
            dbb0 = np.sum(drot0*self._binormal, axis=1)
            dnn0 = np.sum(drot0*self._normal[:,self._fold_index[0]], axis=1)
            dnn1 = np.sum(drot1*self._normal[:,self._fold_index[1]], axis=1)
            # derivatives of cosine and sine of folding angles
            dcos_0 = np.sum(nn1*dnn0, axis=0)
            dsin_0 = np.sum(nn1*dbb0, axis=0)
            dcos_1 = np.sum(dnn1*nn0, axis=0)
            dsin_1 = np.sum(dnn1*bb0, axis=0)
            # derivatives of folding angles
            drho[ii, self._fold_index[0]+self._nf*i+3*self._nf] = (-sin*dcos_0 + cos*dsin_0)/ll2
            drho[ii, self._fold_index[1]+self._nf*i+3*self._nf] = (-sin*dcos_1 + cos*dsin_1)/ll2
        return drho


    ### second-order derivative of folding angle
    def diff2foldangle(self):
        dispf = self.disp.reshape([6,self._nf])
        ddrho = np.zeros((self._nc,6*self._nf,6*self._nf))
        ii = np.arange(self._nc)
        # rotation matrices
        rot0 = self._rot3D(dispf[3:6,self._fold_index[0]])
        rot1 = self._rot3D(dispf[3:6,self._fold_index[1]])
        # rotated vectors
        bb0 = np.sum(rot0*self._binormal, axis=1)
        nn0 = np.sum(rot0*self._normal[:,self._fold_index[0]], axis=1)
        nn1 = np.sum(rot1*self._normal[:,self._fold_index[1]], axis=1)
        # cosine and sine of folding angles
        cos = np.sum(nn1*nn0, axis=0)
        sin = np.sum(nn1*bb0, axis=0)
        ll2 = sin**2 + cos**2
        for i in range(3):
            # first-order derivatives of rotation matrices
            drot0_i = self._drot3D(dispf[3:6,self._fold_index[0]],i)
            drot1_i = self._drot3D(dispf[3:6,self._fold_index[1]],i)
            # first-order derivatives of rotated vectors
            dbb0_i = np.sum(drot0_i*self._binormal, axis=1)
            dnn0_i = np.sum(drot0_i*self._normal[:,self._fold_index[0]], axis=1)
            dnn1_i = np.sum(drot1_i*self._normal[:,self._fold_index[1]], axis=1)
            # first-order derivatives of cosine and sine of folding angles
            dcos_0i = np.sum(nn1*dnn0_i, axis=0)
            dsin_0i = np.sum(nn1*dbb0_i, axis=0)
            dcos_1i = np.sum(dnn1_i*nn0, axis=0)
            dsin_1i = np.sum(dnn1_i*bb0, axis=0)
            for j in range(3):
                # first-order derivatives of rotation matrices
                drot0_j = self._drot3D(dispf[3:6,self._fold_index[0]],j)
                drot1_j = self._drot3D(dispf[3:6,self._fold_index[1]],j)
                # first-order derivatives of rotated vectors
                dbb0_j = np.sum(drot0_j*self._binormal, axis=1)
                dnn0_j = np.sum(drot0_j*self._normal[:,self._fold_index[0]], axis=1)
                dnn1_j = np.sum(drot1_j*self._normal[:,self._fold_index[1]], axis=1)
                # first-order derivatives of cosine and sine of folding angles
                dcos_0j = np.sum(nn1*dnn0_j, axis=0)
                dsin_0j = np.sum(nn1*dbb0_j, axis=0)
                dcos_1j = np.sum(dnn1_j*nn0, axis=0)
                dsin_1j = np.sum(dnn1_j*bb0, axis=0)
                # second-order derivatives of rotation matrices
                ddrot0_ij = self._ddrot3D(dispf[3:6,self._fold_index[0]],i,j)
                ddrot1_ij = self._ddrot3D(dispf[3:6,self._fold_index[1]],i,j)
                # second-order derivatives of rotated vectors
                ddbb0_ij = np.sum(ddrot0_ij*self._binormal, axis=1)
                ddnn0_ij = np.sum(ddrot0_ij*self._normal[:,self._fold_index[0]], axis=1)
                ddnn1_ij = np.sum(ddrot1_ij*self._normal[:,self._fold_index[1]], axis=1)
                # second-order derivatives of cosine and sine of folding angles
                ddcos_0i0j = np.sum(nn1*ddnn0_ij, axis=0)
                ddcos_0i1j = np.sum(dnn1_j*dnn0_i, axis=0)
                ddcos_0j1i = np.sum(dnn1_i*dnn0_j, axis=0)
                ddcos_1i1j = np.sum(ddnn1_ij*nn0, axis=0)
                ddsin_0i0j = np.sum(nn1*ddbb0_ij, axis=0)
                ddsin_0i1j = np.sum(dnn1_j*dbb0_i, axis=0)
                ddsin_0j1i = np.sum(dnn1_i*dbb0_j, axis=0)
                ddsin_1i1j = np.sum(ddnn1_ij*bb0, axis=0)
                # second-order derivatives of folding angles
                ddrho1_0i0j = ((sin**2 - cos**2) * (dcos_0i*dsin_0j + dcos_0j*dsin_0i) + 2*sin*cos * (dcos_0i*dcos_0j - dsin_0j*dsin_0i)) / (ll2**2)
                ddrho1_0i1j = ((sin**2 - cos**2) * (dcos_0i*dsin_1j + dcos_1j*dsin_0i) + 2*sin*cos * (dcos_0i*dcos_1j - dsin_1j*dsin_0i)) / (ll2**2)
                ddrho1_0j1i = ((sin**2 - cos**2) * (dcos_0j*dsin_1i + dcos_1i*dsin_0j) + 2*sin*cos * (dcos_0j*dcos_1i - dsin_1i*dsin_0j)) / (ll2**2)
                ddrho1_1i1j = ((sin**2 - cos**2) * (dcos_1i*dsin_1j + dcos_1j*dsin_1i) + 2*sin*cos * (dcos_1i*dcos_1j - dsin_1j*dsin_1i)) / (ll2**2)
                ddrho2_0i0j = (-sin * ddcos_0i0j + cos * ddsin_0i0j) / ll2
                ddrho2_0i1j = (-sin * ddcos_0i1j + cos * ddsin_0i1j) / ll2
                ddrho2_0j1i = (-sin * ddcos_0j1i + cos * ddsin_0j1i) / ll2
                ddrho2_1i1j = (-sin * ddcos_1i1j + cos * ddsin_1i1j) / ll2
                ddrho[ii, self._fold_index[0]+self._nf*i+3*self._nf, self._fold_index[0]+self._nf*j+3*self._nf] = ddrho1_0i0j + ddrho2_0i0j
                ddrho[ii, self._fold_index[0]+self._nf*i+3*self._nf, self._fold_index[1]+self._nf*j+3*self._nf] = ddrho1_0i1j + ddrho2_0i1j
                ddrho[ii, self._fold_index[1]+self._nf*j+3*self._nf, self._fold_index[0]+self._nf*i+3*self._nf] = ddrho1_0i1j + ddrho2_0i1j
                ddrho[ii, self._fold_index[0]+self._nf*j+3*self._nf, self._fold_index[1]+self._nf*i+3*self._nf] = ddrho1_0j1i + ddrho2_0j1i
                ddrho[ii, self._fold_index[1]+self._nf*i+3*self._nf, self._fold_index[0]+self._nf*j+3*self._nf] = ddrho1_0j1i + ddrho2_0j1i
                ddrho[ii, self._fold_index[1]+self._nf*i+3*self._nf, self._fold_index[1]+self._nf*j+3*self._nf] = ddrho1_1i1j + ddrho2_1i1j
        return ddrho




    ##################################################
    ### imcompatibility vector and its derivatives for panel connectivity
    ##################################################
    ### incompatibility vector w.r.t panel displacement for panel connectivity
    def connectivity(self):
        dispf = self.disp.reshape([6,self._nf])
        # rotation matrices
        rot0 = self._rot3D(dispf[3:6,self._conn_index[0]])
        rot1 = self._rot3D(dispf[3:6,self._conn_index[1]])
        # displacement of vertices
        vv0 = dispf[0:3,self._conn_index[0]] + np.sum(rot0*self._corner[:,self._conn_index[2]], axis=1) - self._corner[:,self._conn_index[2]]
        vv1 = dispf[0:3,self._conn_index[1]] + np.sum(rot1*self._corner[:,self._conn_index[3]], axis=1) - self._corner[:,self._conn_index[3]]
        # incompatibility vector
        conn = np.ravel(vv0 - vv1)
        return conn


    ### first-order derivative of incompatibility vector for panel connectivity
    def diffconnectivity(self):
        ncmp = self._conn_index.shape[1]
        dispf = self.disp.reshape([6,self._nf])
        dconn = np.zeros((3*ncmp,6*self._nf))
        ii = np.arange(ncmp)
        # first-order derivative of incompatibility vector
        for i in range(3):
            # w.r.t. translation
            dconn[ii+ncmp*i ,self._conn_index[0]+self._nf*i] = 1.
            dconn[ii+ncmp*i ,self._conn_index[1]+self._nf*i] = -1.
            # w.r.t. rotation
            drot0 = self._drot3D(dispf[3:6,self._conn_index[0]],i)
            drot1 = self._drot3D(dispf[3:6,self._conn_index[1]],i)
            ddd0 = np.sum(drot0*self._corner[:,self._conn_index[2]], axis=1)
            ddd1 = np.sum(drot1*self._corner[:,self._conn_index[3]], axis=1)
            dconn[ii       , self._conn_index[0]+self._nf*i+3*self._nf] = ddd0[0]
            dconn[ii+ncmp  , self._conn_index[0]+self._nf*i+3*self._nf] = ddd0[1]
            dconn[ii+2*ncmp, self._conn_index[0]+self._nf*i+3*self._nf] = ddd0[2]
            dconn[ii       , self._conn_index[1]+self._nf*i+3*self._nf] = -ddd1[0]
            dconn[ii+ncmp  , self._conn_index[1]+self._nf*i+3*self._nf] = -ddd1[1]
            dconn[ii+2*ncmp, self._conn_index[1]+self._nf*i+3*self._nf] = -ddd1[2]
        return dconn


    ### second-order derivative of incompatibility vector for panel connectivity
    def diff2connectivity(self):
        ncmp = self._conn_index.shape[1]
        dispf = self.disp.reshape([6,self._nf])
        ddconn = np.zeros((3*ncmp,6*self._nf,6*self._nf))
        ii = np.arange(ncmp)
        # second-order derivative of incompatibility vector
        for i in range(3):
            for j in range(3):
                # w.r.t. rotation (translation = 0)
                ddrot0 = self._ddrot3D(dispf[3:6,self._conn_index[0]],i,j)
                ddrot1 = self._ddrot3D(dispf[3:6,self._conn_index[1]],i,j)
                dddd0 = np.sum(ddrot0*self._corner[:,self._conn_index[2]], axis=1)
                dddd1 = np.sum(ddrot1*self._corner[:,self._conn_index[3]], axis=1)
                ddconn[ii       , self._conn_index[0]+self._nf*i+3*self._nf, self._conn_index[0]+self._nf*j+3*self._nf] = dddd0[0]
                ddconn[ii+ncmp  , self._conn_index[0]+self._nf*i+3*self._nf, self._conn_index[0]+self._nf*j+3*self._nf] = dddd0[1]
                ddconn[ii+2*ncmp, self._conn_index[0]+self._nf*i+3*self._nf, self._conn_index[0]+self._nf*j+3*self._nf] = dddd0[2]
                ddconn[ii       , self._conn_index[1]+self._nf*i+3*self._nf, self._conn_index[1]+self._nf*j+3*self._nf] = -dddd1[0]
                ddconn[ii+ncmp  , self._conn_index[1]+self._nf*i+3*self._nf, self._conn_index[1]+self._nf*j+3*self._nf] = -dddd1[1]
                ddconn[ii+2*ncmp, self._conn_index[1]+self._nf*i+3*self._nf, self._conn_index[1]+self._nf*j+3*self._nf] = -dddd1[2]
        return ddconn




    ##################################################
    ### vertex displacements and their derivatives
    ##################################################
    ### vertex displacement
    def vertexdisp(self):
        dispf = self.disp.reshape([6,self._nf])
        dispv = np.zeros((3,self._nv))
        for i in range(0, self._dispv_index.shape[0], 2):
            rot = self._rot3D(dispf[3:6,self._dispv_index[i]])
            aav = self._angle[self._dispv_index[i+1]]
            ddv = self._corner[:,self._dispv_index[i+1]]
            dispv += aav*(dispf[0:3,self._dispv_index[i]] + np.sum(rot*ddv, axis=1) - ddv)
        dispv = np.ravel(dispv/self._sum_angle)
        return dispv


    ### first-order derivative of vertex displacement
    def diffvertexdisp(self):
        dispf = self.disp.reshape([6,self._nf])
        ddispv = np.zeros((3*self._nv,6*self._nf))
        ii = np.arange(self._nv)
        # first-order derivative of vertex displacement
        for i in range(0, self._dispv_index.shape[0], 2):
            aav = self._angle[self._dispv_index[i+1]]
            ddv = self._corner[:,self._dispv_index[i+1]]
            for j in range(3):
                # w.r.t. translation
                ddispv[ii+self._nv*j, self._dispv_index[i]+self._nf*j] += aav/self._sum_angle
                # w.r.t. rotation
                drot = self._drot3D(dispf[3:6,self._dispv_index[i]],j)
                dddv = np.sum(drot*ddv, axis=1)
                ddispv[ii            , self._dispv_index[i]+self._nf*j+3*self._nf] += (aav/self._sum_angle)*dddv[0]
                ddispv[ii+self._nv  , self._dispv_index[i]+self._nf*j+3*self._nf] += (aav/self._sum_angle)*dddv[1]
                ddispv[ii+2*self._nv, self._dispv_index[i]+self._nf*j+3*self._nf] += (aav/self._sum_angle)*dddv[2]
        return ddispv


    ### second-order derivative of vertex displacement
    def diff2vertexdisp(self):
        dispf = self.disp.reshape([6,self._nf])
        dddispv = np.zeros((3*self._nv,6*self._nf,6*self._nf))
        ii = np.arange(self._nv)
        # second-order derivative of vertex displacement
        for i in range(0, self._dispv_index.shape[0], 2):
            aav = self._angle[self._dispv_index[i+1]]
            ddv = self._corner[:,self._dispv_index[i+1]]
            for j in range(3):
                for k in range(3):
                    # w.r.t. rotation (translation = 0)
                    ddrot = self._ddrot3D(dispf[3:6,self._dispv_index[i]],j,k)
                    ddddv = np.sum(ddrot*ddv, axis=1)
                    dddispv[ii            , self._dispv_index[i]+self._nf*j+3*self._nf, self._dispv_index[i]+self._nf*k+3*self._nf] += (aav/self._sum_angle)*ddddv[0,:]
                    dddispv[ii+self._nv  , self._dispv_index[i]+self._nf*j+3*self._nf, self._dispv_index[i]+self._nf*k+3*self._nf] += (aav/self._sum_angle)*ddddv[1,:]
                    dddispv[ii+2*self._nv, self._dispv_index[i]+self._nf*j+3*self._nf, self._dispv_index[i]+self._nf*k+3*self._nf] += (aav/self._sum_angle)*ddddv[2,:]
        return dddispv




    ##################################################
    ### incompatibility vector including panel connectivity and boundary conditions
    ##################################################
    ### incompatibility vector
    def incompatibility_vector(self,
                               forced_disp=True  # bool; Consider uniformly increasing forced displacement or not
                               ):
        # incompatibility vector for panel connectivity
        incomppc = self.connectivity()
        # incompatibility vector for uniformly increasing forced panel displacements
        if forced_disp and (len(self._fdisp) > 1):
            incomppd = self.disp[self._fdisp[0:-1]] / self._df[0:-1] - self.disp[self._fdisp[-1]] / self._df[-1]
        else:
            incomppd = np.zeros(0)
        # vertex displacements
        if (len(self._vfix) > 0) or (forced_disp and (len(self._vdisp) > 1)):
            dispv = self.vertexdisp()
        # incompatibility vector for fixed vertex displacements
        if len(self._vfix) > 0:
            incompvf = dispv[self._vfix]
        else:
            incompvf = np.zeros(0)
        # incompatibility vector for uniformly increasing forced vertex displacements
        if forced_disp and (len(self._vdisp) > 1):
            incompvd = dispv[self._vdisp[0:-1]] / self._dv[0:-1] - dispv[self._vdisp[-1]] / self._dv[-1]
        else:
            incompvd = np.zeros(0)
        # assemble incompatibility vectors
        incomp = np.concatenate((incomppc,incomppd,incompvf,incompvd))
        return incomp




    ##################################################
    ### first-order infinitesimal mechanism analysis
    ##################################################
    ### compatibility matrix
    def compatibility_matrix(self,
                             forced_disp=True  # bool; Consider uniformly increasing forced displacement or not
                             ):
        # compatibility matrix for panel connectivity
        comppc = self.diffconnectivity()
        # compatibility matrix for uniformly increasing forced panel displacements
        if forced_disp and (len(self._fdisp) > 1):
            comppd = np.identity(6*self._nf)[self._fdisp[0:-1]] / self._df[0:-1].reshape([len(self._fdisp)-1,1])
            comppd[:,self._fdisp[-1]] = -1. / self._df[-1]
        else:
            comppd = np.zeros((0,6*self._nf))
        # first-order derivatives of vertex displacements
        if (len(self._vfix) > 0) or (forced_disp and (len(self._vdisp) > 1)):
            ddispv = self.diffvertexdisp()
        # compatibility matrix for fixed vertex displacements
        if len(self._vfix) > 0:
            compvf = ddispv[self._vfix]
        else:
            compvf = np.zeros((0,6*self._nf))
        # compatibility matrix for uniformly increasing forced vertex displacements
        if forced_disp and (len(self._vdisp) > 1):
            compvd = ddispv[self._vdisp[0:-1]] / self._dv[0:-1].reshape([len(self._vdisp)-1,1])
            compvd -= ddispv[self._vdisp[-1]] / self._dv[-1]
        else:
            compvd = np.zeros((0,6*self._nf))
        # assemble compatibility matrices
        compt = np.vstack((comppc,comppd,compvf,compvd))
        # eliminate fixed and glued DOF
        compt = np.append(compt, np.zeros((compt.shape[0],1)), axis=1)
        comp = compt[:,self._ivar] + compt[:,self._iadd]
        return comp


    ### first-order infinitesimal mechanism analysis
    def first_order_mechanism(self,
                              forced_disp=True,  # bool; Consider uniformly increasing forced displacement or not
                              tol=1.e-8          # float; Threshold below which SVD values are considered zero
                              ):
        # compatibility matrix
        comp = self.compatibility_matrix(forced_disp)
        # sigular value decomposition
        fmode, singular, dmodeT = np.linalg.svd(comp)
        singular = singular[::-1]
        fmode = fmode[:,::-1]
        dmode = (dmodeT.T)[:,::-1]
        dor = comp.shape[0] - np.count_nonzero(singular > tol)    # number of statical indeterminacy
        dof = comp.shape[1] - np.count_nonzero(singular > tol)    # number of kinematic indeterminacy
        # restore all panel displacements
        dmodes = np.copy(dmode)
        dmode = np.append(dmode, np.zeros((1,dmode.shape[1])), axis=0)[self._iall]
        dmodef = dmode[:,0:dof]
        # tangent stiffness matrix only with crease line stiffness
        drho = self.difffoldangle()
        dmoder = np.dot(drho,dmodef)
        kk = np.dot(np.dot(dmoder.T,np.diag(self._stiff)),dmoder)
        # modify first-order infinitesimal mechanism modes using tangent stiffness matrix
        eigen, vv = np.linalg.eigh(kk)
        arg = np.argsort(eigen)
        eigen = eigen[arg]
        vv = vv[:,arg]
        dmodef = np.dot(dmodef,vv)
        dmode[:,0:dof] = dmodef
        dmode = dmode/np.linalg.norm(dmode, axis=0).reshape([1,dmode.shape[1]])
        return singular, eigen, fmode, dmode, dor, dof


    ### vertex displacement modes
    def vertex_modes(self,
                     dmode  # (N, M) array, float; M modes of N-DOF panel displacement
                     ):
        # first order derivative of vertex displacement
        ddispv = self.diffvertexdisp()
        # vertex displacement modes
        dmodev = np.dot(ddispv,dmode)
        ll = np.linalg.norm(dmodev, axis=0)
        dmodev = dmodev/ll.reshape([1,dmodev.shape[1]])
        return dmodev



    ##################################################
    ### second-order infinitesimal mechanism analysis
    ##################################################
    ### Hessian of panel connectivity, support conditions, and uniformly increasing forced displacement conditions
    def compatibility_hessian(self,
                              forced_disp=True  # bool; Consider uniformly increasing forced displacement or not
                              ):
        # Hessian for panel connectivity
        hesspc = self.diff2connectivity()
        # Hessian for uniformly increasing forced panel displacements
        if forced_disp and (len(self._fdisp) > 1):
            hesspd = np.zeros((len(self._fdisp)-1,6*self._nf,6*self._nf))
        else:
            hesspd = np.zeros((0,6*self._nf,6*self._nf))
        # second-order derivatives of vertex displacements
        if (len(self._vfix) > 0) or (forced_disp and (len(self._vdisp) > 1)):
            dddispv = self.diff2vertexdisp()
        # Hessian for fixed vertex displacements
        if len(self._vfix) > 0:
            hessvf = dddispv[self._vfix]
        else:
            hessvf = np.zeros((0,6*self._nf,6*self._nf))
        # Hessian for uniformly increasing forced vertex displacements
        if forced_disp and (len(self._vdisp) > 1):
            hessvd = dddispv[self._vdisp[0:-1]] / self._dv[0:-1].reshape([len(self._vdisp)-1,1,1])
            hessvd -= dddispv[self._vdisp[-1]] / self._dv[-1]
        else:
            hessvd = np.zeros((0,6*self._nf,6*self._nf))
        # assemble Hessian
        hesst = np.concatenate((hesspc,hesspd,hessvf,hessvd), axis=0)
        # eliminate fixed and glued DOF
        hesst = np.append(hesst, np.zeros((hesst.shape[0],hesst.shape[1],1)), axis=2)
        hesst = np.append(hesst, np.zeros((hesst.shape[0],1,hesst.shape[2])), axis=1)
        hess = hesst[:,:,self._ivar] + hesst[:,:,self._iadd]
        hess = hess[:,self._ivar,:] + hess[:,self._iadd,:]
        return hess


    ### quadratic form for existence condition of second-order infinitesimal mechanism
    def second_order_exist(self,
                           sforce,           # (N, M) array, float; M modes of N-dimensional self-equilibrium force modes
                           flex,             # (N, M) array, float; M modes of N-dimensional infinitesimal mechanism modes
                           forced_disp=True  # bool; Consider uniformly increasing forced displacement or not
                           ):
        # degrees of statical and kinematic indeterminacies
        dor = sforce.shape[1]
        dof = flex.shape[1]
        quad = np.zeros((dor,dof,dof))
        if dor > 0:
            # Hessian
            hess = self.compatibility_hessian(forced_disp)
            # quadratic form for existence condition of second-order mechanism
            flex = flex[self._ivar]
            for i in range(dor):
                quad_i = np.sum(hess * sforce[:,i].reshape([sforce.shape[0],1,1]), axis=0)
                quad[i] = np.dot(np.dot(flex.T, quad_i), flex)
        return quad


    def quad_coefficient(self,
                         quad,        # (N, M, M) array, float; N matrices of quadratic forms with the size of N by N
                         tol=1.e-12   # float; Threshold below which coefficient values are considered zero
                         ):
        # degrees of statical and kinematic indeterminacies
        dor = quad.shape[0]
        dof = quad.shape[1]
        # coefficients of quadratic form
        coef = np.zeros((dor,int(dof*(dof+1)/2+0.1)))
        if dor > 0:
            for i in range(dor):
                ii = 0
                for j in range(dof):
                    for k in range(j,dof):
                        if k == j:
                            if np.abs(quad[i,j,j]) >= tol:
                                coef[i,ii] = quad[i,j,j]
                        else:
                            if np.abs(quad[i,j,k]) >= tol:
                                coef[i,ii] = quad[i,j,k] + quad[i,k,j]
                        ii += 1
            # rearrangement of coef
            coef, _, _, _ = self._rref(coef, tol=tol)
        return coef




    ##################################################
    # potential energies and their derivatives
    ##################################################
    ### potential energy of hinge springs
    def hinge_potential(self):
        rho = self.foldangle()
        energy = np.sum(self._stiff * (rho - self.init_angle)**2) / 2.
        return energy


    ### first-order derivative of hinge potential energy
    def hinge_diffpotential(self):
        rho = self.foldangle()
        drho = self.difffoldangle()
        denergy = np.sum((self._stiff * (rho - self.init_angle)).reshape([self._nc,1]) * drho, axis=0)
        return denergy


    ### second-order derivative of hinge potential energy
    def hinge_diff2potential(self):
        rho = self.foldangle()
        drho = self.difffoldangle()
        ddrho = self.diff2foldangle()
        ddenergy = np.dot(drho.T, self._stiff.reshape([self._nc,1]) * drho)
        ddenergy += np.sum((self._stiff * (rho - self.init_angle)).reshape([self._nc,1,1]) * ddrho, axis=0)
        return ddenergy


    ### gravity potential
    def grav_potential(self):
        energy = np.sum(self._weight * (self.disp.reshape([6,self._nf])[2] + self._refz))
        return energy


    ### first-order derivative of gravity potential
    def grav_diffpotential(self):
        denergy = np.zeros(6*self._nf)
        denergy[2*self._nf:3*self._nf] += self._weight
        return denergy


    ### second-order derivative of gravity potential
    def grav_diff2potential(self):
        return np.zeros((6*self._nf, 6*self._nf))


    ### external work of vertex loads
    def vert_work(self):
        dispv = self.vertexdisp()
        work = np.dot(self._pv, dispv[self._vload])
        return work


    ### first-order derivative of external work of vertex loads
    def vert_diffwork(self):
        dispv = self.vertexdisp()
        ddispv = self.diffvertexdisp()
        dwork = np.dot(ddispv[self._vload].T, self._pv)
        return dwork


    ### second-order derivative of external work of vertex loads
    def vert_diff2work(self):
        dispv = self.vertexdisp()
        ddispv = self.diffvertexdisp()
        dddispv = self.diff2vertexdisp()
        ddwork = np.sum(dddispv[self._vload] * self._pv.reshape([len(self._pv),1,1]), axis=0)
        return ddwork


    ### external work of face loads
    def face_work(self):
        work = np.dot(self._pf, self.disp[self._fload])
        return work


    ### first-order derivative of external work of face loads
    def face_diffwork(self):
        dwork = np.dot(np.identity(6*self._nf)[:,self._fload], self._pf)
        return dwork


    ### second-order derivative of external work of face loads
    def face_diff2work(self):
        return np.zeros((6*self._nf,6*self._nf))


    ### total potential energy
    def total_potential(self):
        energyh = self.hinge_potential()
        energyg = self.grav_potential()
        workv = self.vert_work()
        workf = self.face_work()
        energy = energyh + energyg - workv - workf
        return energy


    ### first-order derivative of total potential energy
    def total_diffpotential(self):
        denergyh = self.hinge_diffpotential()
        denergyg = self.grav_diffpotential()
        dworkv = self.vert_diffwork()
        dworkf = self.face_diffwork()
        denergy = denergyh + denergyg - dworkv - dworkf
        return denergy


    ### second-order derivative of total potential energy
    def total_diff2potential(self):
        ddenergyh = self.hinge_diff2potential()
        ddenergyg = self.grav_diff2potential()
        ddworkv = self.vert_diff2work()
        ddworkf = self.face_diff2work()
        ddenergy = ddenergyh + ddenergyg - ddworkv - ddworkf
        return ddenergy








##################################################
# Sub class for path tracing under target folding angles
##################################################
class PathTargetAngle(PanelPin):

    ##################################################
    ### constructor
    ##################################################
    def __init__(self,
                 vertex,                        # (3, N) array, float; Initial coordinates of vertices
                 face,                          # List of list, int; List of indices lists of face (panel) vertices
                 crease,                        # (4, N) array, int; Indices of end points and adjacent faces of each crease line
                 target_angle,                  # N-D array, float; Target folding angles of crease lines
                 glued_panels=np.zeros((2,0)),  # (2, N) array, int; Indices of faces (panels) that are glued togther
                 boundary_conds=(),             # tuple of dictionaries; Boundary conditions assigned to panel and vertex displacements
                 options={}                     # dictionary; Optional paramenters for path tracing
                 ):
        # initialize variables in sub-class
        self.target_angle = target_angle
        if 'weight' in options:
            kkcpl = options['weight']    # Weight of energy at crease lines
        else:
            kkcpl = 1.
        if 'nitr_main' in options:
            self._maxitr = options['nitr_main']    # Maximum number of iterations of path tracing
        else:
            self._maxitr = 1000
        if 'eps_target' in options:
            self._epst = options['eps_target']    # Threshold of squared error of folding angles for convegence of path tracing
        else:
            self._epst = 1.e-4
        if 'eps_grad' in options:
            self._epsg = options['eps_grad']    # Threshold of norm of projected gradient for convegence of path tracing
        else:
            self._epsg = 1.e-4
        if 'tol_singular' in options:
            self._stol = options['tol_singular']    # Threshold for zero singular values of compatibility matrix
        else:
            self._stol = 1.e-8
        if 'delta_pred' in options:
            self._delp = options['delta_pred']    # Initial step size of predictor
        else:
            self._delp = 0.1
        if 'xi_pred' in options:
            self._xip = options['xi_pred']    # Parameter for Armijo line search for length of predictor
        else:
            self._xip = 1.e-4
        if 'tau_pred' in options:
            self._taup = options['tau_pred']    # Reduction rate of length of predictor in Armijo line search
        else:
            self._taup = 0.5
        if 'n_pred' in options:
            self._nitrp = options['n_pred']    # Maximum iteration number of Armijo line search for length of predictor
        else:
            self._nitrp = 10
        if 'delta_corr' in options:
            self._delc = options['delta_corr']    # Initial step size of corrector
        else:
            self._delc = 1.0
        if 'xi_corr' in options:
            self._xic = options['xi_corr']    # Parameter for Armijo line search for length of corrector
        else:
            self._xic = 1.e-4
        if 'tau_corr' in options:
            self._tauc = options['tau_corr']    # Reduction rate of length of corrector in Armijo line search
        else:
            self._tauc = 0.5
        if 'n_corr' in options:
            self._nitrc = options['n_corr']    # Maximum iteration number of Armijo line search for length of corrector
        else:
            self._nitrc = 10
        if 'tol_corr' in options:
            self._ctol = options['tol_corr']    # Tolerance of norm of incompatibility vector
        else:
            self._ctol = 1.e-10
        if 'nitr_corr' in options:
            self._maxcor = options['nitr_corr']    # Maximum iteration number of correction process
        else:
            self._maxcor = 500
        if 'iprint' in options:
            self._iprint = options['iprint']    # Frequency for printing process of path tracing
        else:
            self._iprint = 1
        if 'ihistroy' in options:
            self._ihistory = options['ihistory']    # Frequency for adding panel displacement data to history list
        else:
            self._ihistory = 1
        # construct super-class
        super(PathTargetAngle, self).__init__(vertex, face, crease, glued_panels=glued_panels,
                                              stiff_per_length=kkcpl, weight_per_area=0.0,
                                              boundary_conds=boundary_conds)
        # initialize path tracing
        self.success = False
        self.history = []
        self.message = "Path tracing reached iteration limit"
        self._incomp = self.incompatibility_vector(forced_disp=False)
        self._comp = self.compatibility_matrix(forced_disp=False)
        # execute path tracing
        self.path_tracing()
        # print message
        if self._iprint > 0:
            print(self.message)




    ##################################################
    ### update direction of panel displacements
    ##################################################
    ### projected steepest descent direction to target folding angles
    def _predictor(self):
        # current folding angles and thier derivatives
        rho = self.foldangle()
        drho = self.difffoldangle()
        # eliminate fixed and glued DOF
        drho = drho[:,self._ivar] + drho[:,self._iadd]
        # gradient of squared error of folding angles
        dd = -np.sum((self._stiff * (rho - self.target_angle)).reshape([self._nc,1]) * drho, axis=0)
        # SVD of compatibiliy matrix
        _, ss, dmodeT = np.linalg.svd(self._comp)
        dmode = (dmodeT.T)[:,::-1]
        dof = self._comp.shape[1] - np.count_nonzero(ss > self._stol)
        if dof == 0:
            raise ValueError("Infinitesimal mechanism modes vanished.")
        # space of first-order infinitesimal mechanism
        dmodef = dmode[:,0:dof]
        # projection of steepest descent direction
        ddp = np.dot(dmodef, np.dot(dmodef.T, dd))
        # initial length of predictor
        alpha = np.copy(self._delp)
        # Armijo line search
        for i in range(self._nitrp):
            disp_curr = np.copy(self.disp)
            self.disp += np.append(alpha*ddp, 0)[self._iall]
            rho_new = self.foldangle()
            err_new = 0.5 * np.dot(rho_new - self.target_angle, self._stiff * (rho_new - self.target_angle))
            self.disp = np.copy(disp_curr)
            err_tol = 0.5 * np.dot(rho - self.target_angle, self._stiff * (rho - self.target_angle)) + self._xip * alpha * np.dot(-dd, ddp)
            if err_tol - err_new >= 0:
                break
            alpha *= self._taup
        # restore all panel displacements
        pred = np.append(alpha*ddp, 0)[self._iall]
        return pred


    ### corrector whose length is determined using Armijo line search
    def _corrector(self):
        # direction of corrector
        dd = -np.dot(np.linalg.pinv(self._comp), self._incomp)
        # initial length of corrector
        beta = np.copy(self._delc)
        # Armijo line search
        for i in range(self._nitrc):
            disp_curr = np.copy(self.disp)
            self.disp += np.append(beta*dd, 0)[self._iall]
            incomp_new = np.linalg.norm(self.incompatibility_vector(forced_disp=False))
            self.disp = np.copy(disp_curr)
            incomp_tol = np.linalg.norm(self._incomp) + self._xic * beta * np.dot(np.dot(self._comp.T, self._incomp), dd) / np.linalg.norm(self._incomp)
            if incomp_tol - incomp_new >= 0:
                break
            beta *= self._tauc
        # restore all panel displacements
        corr = np.append(beta*dd, 0)[self._iall]
        return corr




    ##################################################
    ### main part of path tracing
    ##################################################
    def path_tracing(self):
        break_main = False
        if self._iprint > 0:
            print("itr  angle  grad  incomp")
        for itr in range(self._maxitr):
            # update incompatibility vector and compatibility matrix
            self._incomp = self.incompatibility_vector(forced_disp=False)
            self._comp = self.compatibility_matrix(forced_disp=False)
            # squared error of folding angles
            rho = self.foldangle()
            err = 0.5 * np.dot(rho - self.target_angle, self._stiff * (rho - self.target_angle))
            # predictor
            pred = self._predictor()
            # print path tracing process
            if itr%self._iprint == 0:
                print("%i  %.3e  %.3e  %.3e"%(itr,err,np.linalg.norm(pred),np.max(np.abs(self._incomp))))
            # add data to history list
            if itr%self._ihistory == 0:
                self.history.append([itr, err, np.copy(self.disp)])
            # convergence condition
            if err < self._epst:
                if self._iprint > 0 and itr%self._iprint != 0:
                    print("%i  %.3e  %.3e  %.3e"%(itr,err,np.linalg.norm(pred),np.max(np.abs(self._incomp))))
                self.message = "Squared error of folding angles is less than eps_target."
                self.succeess = True
                self.history.append([itr, err, np.copy(self.disp)])
                break
            elif np.linalg.norm(pred) < self._epsg:
                if self._iprint > 0 and itr%self._iprint != 0:
                    print("%i  %.3e  %.3e  %.3e"%(itr,err,np.linalg.norm(pred),np.max(np.abs(self._incomp))))
                self.message = "Norm of projected gradient is less than eps_grad."
                self.history.append([itr, err, np.copy(self.disp)])
                break
            # update panel displacement
            self.disp += pred
            # correction of panel displacement
            for i in range(self._maxcor+1):
                # update incompatibility vector and compatibility matrix
                self._incomp = self.incompatibility_vector(forced_disp=False)
                self._comp = self.compatibility_matrix(forced_disp=False)
                # convergence condition
                if np.linalg.norm(self._incomp) < self._ctol:
                    break
                elif i == self._maxcor:
                    self.message = "Desired compatibility cannot be achieved."
                    break_main = True
                    break
                # corrector
                corr = self._corrector()
                # update panel displacement
                self.disp += corr
            # break main iteration if corrector cannot reduce incompatibility
            if break_main or itr == self._maxitr-1:
                rho = self.foldangle()
                err = 0.5 * np.dot(rho - self.target_angle, self._stiff * (rho - self.target_angle))
                if self._iprint > 0 and itr%self._iprint != 0:
                    self._incomp = self.incompatibility_vector(forced_disp=False)
                    print("%i  %.3e  %.3e  %.3e"%(itr,err,np.linalg.norm(pred),np.max(np.abs(self._incomp))))
                self.history.append([itr, err, np.copy(self.disp)])
                break
