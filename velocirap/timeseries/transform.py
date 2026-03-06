
""" ----------------------------------------------------------------------------
Functions for transforming data (rotation, interpolation)
---------------------------------------------------------------------------- """

import numpy            as np
import scipy.constants  as cst
from velocirap.helpers  import interpolate_func, quaternion_rotation_matrix

def interpolate(var, order=1): 
    """
    Routine that creates an interpolating function for each variables. 
    Contains interpolating function (F.f) 

    Parameters
    ----------
    var : DotMap object
        DotMap object containing all variables.
    order : int, optional
        Order of the interpolation. The default is 1.
        
    Returns
    -------
    var : DotMap object
        DotMap object containing the additional interpolating functions.
    """
    
    for F in var.values() :        
        F.f = interpolate_func(F.t, F.y, k=order)
        
    return var

def kT_tensor_inst2rtn(kT_inst, Q): # From load/
    """
    Rotate the kT tensor from the instrument frame to the RTN frame

    Parameters
    ----------
    kT_inst : ndarray
        The kT tensor in the instrument frame, with shape (6,)
    Q : ndarray
        The quaternions giving the rotation from the instrument frame 
        to the SC frame

    Returns
    -------
    kT_rtn : ndarray
        The kT tensor in the RTN frame, with shape (6,)
    """
    
    #rotation matrix from instrument from to SC frame
    mat_inst_to_sc = np.array([
        [ 0.        ,  0.        ,  1.        ],
        [-0.9396926 ,  0.34202015,  0.        ],
        [-0.34202015, -0.9396926 ,  0.        ]
    ])

    mat_inst_to_rtn = mat_inst_to_sc @ quaternion_rotation_matrix(Q)

    #change of basis matrix  
    A = np.moveaxis(mat_inst_to_rtn, -1, 0)

    #inverse of change of basis matrix  
    A_inv   = np.linalg.inv(A)                      

    # reshape kT tensor to matrix form
    kT_mat = np.array([
        [kT_inst[0], kT_inst[3], kT_inst[4]], 
        [kT_inst[3], kT_inst[1], kT_inst[5]],
        [kT_inst[4], kT_inst[5], kT_inst[2]]
    ]) 
    
    # kT tensor in the RTN frame 
    PA       = np.einsum('ij..., ...jk -> ik...', kT_mat, A)
    A_inv_PA = np.einsum('...ij, jk... -> ik...', A_inv, PA)

    # kT tensor in RTN frame    
    kT_rtn = np.array([
        A_inv_PA[0, 0],
        A_inv_PA[1, 1],
        A_inv_PA[2, 2],
        A_inv_PA[0, 1],
        A_inv_PA[0, 2],
        A_inv_PA[1, 2]
    ])
    return kT_rtn


def kT_tensor_rtn2b(var): # From load/
    """
    Rotate the temperature tensor from the RTN frame 
    to the magnetic field aligned frame
    
    Parameters
    ----------
    var : DotMap object
        The DotMap object where the timeseries are stored
    
    Returns
    -------
    kT_b : array
        Temperature tensor in the magnetic field aligned frame
    """

    if not var.info.get("Tcalc"):
        var.info.Tcalc = 'rotation'
    
    # check which method is used for the temperature calculation and if there is B data (for rotation)
    if np.isfinite(var.B.y).any() and var.info.Tcalc == 'rotation' :   
        # ---------- Define change of basis matrix --------------    
        # velocity unit vector 
        u = var.U.y / np.linalg.norm(var.U.y, axis=0)    

        # if U and B do not have the same length, interpolate B on U
        if len(var.B.y[0]) != len(u[0]): 
            var.B.f = interpolate_func(var.B.t, var.B.y, k=1)
            # magnetic field unit vector 
            b = var.B.f(var.U.t) / np.linalg.norm(var.B.f(var.U.t), axis=0)
        else : 
            # magnetic field unit vector    
            b = var.B.y / np.linalg.norm(var.B.y, axis=0) 
        
        # other unit vectors
        ub        = np.cross(u.T, b.T).T
        u_x_b     = ub / np.linalg.norm(ub, axis=0)

        bub       = np.cross(b.T, u_x_b.T).T
        b_x_u_x_b = bub / np.linalg.norm(bub, axis=0)
        
        #change of basis matrix (columns are b, bxbxv, -bxv)
        A       = np.array([b, b_x_u_x_b, -u_x_b]).T  

        #inverse of change of basis matrix  
        A_inv   = np.linalg.inv(A)                      

        # reshape kT tensor to matrix form
        kT_mat = np.array([
            [var.kT.y[0], var.kT.y[3], var.kT.y[4]], 
            [var.kT.y[3], var.kT.y[1], var.kT.y[5]],
            [var.kT.y[4], var.kT.y[5], var.kT.y[2]]
        ])
        
        # kT tensor in the magnetic field aligned frame 
        PA       = np.einsum('ij..., ...jk -> ik...', kT_mat, A)
        A_inv_PA = np.einsum('...ij, jk... -> ik...', A_inv, PA)

        # kT tensor in b frame    
        kT_b = np.array([
            A_inv_PA[0, 0],
            A_inv_PA[1, 1],
            A_inv_PA[2, 2],
            A_inv_PA[0, 1],
            A_inv_PA[0, 2],
            A_inv_PA[1, 2]
        ])

        # check globally if some B data is missed and filter the data 
        B_idx = np.where(~np.isnan(var.B.y[0,:]))[0]
        mask = ~np.isnan(var.B.y).any(axis=0)
        B_filt = var.B.t[mask]
        
        # B data missed -> mask T at the correct times
        if len(B_idx) < len(var.B.t):   
            #find times of missing data
            xb, xe = var.kT.t[0], var.kT.t[-1]
            kT_idx, kT_idx2 = np.where(var.kT.t<B_filt[0])[0], np.where(var.kT.t>B_filt[-1])[0] 
            
            #No data at the borders
            if B_filt[0] > xb :                    
                for i in range(len(kT_idx)):
                    kT_b[:,i] = np.nan
            if B_filt[-1] < xe :
                for i in range(len(kT_idx2)):
                    kT_b[:,-1-i] = np.nan
            
            #verify B data is not missed in the center
            for idx in range(len(B_idx)-1) :        
                if (np.abs(B_idx[idx+1] - B_idx[idx] - 1) != 0) : 
                    xb, xe = var.B.t[B_idx[idx]], var.B.t[B_idx[idx+1]]
                    kT_idx3 = np.where((xb<var.kT.t)&(var.kT.t<xe))[0]
                    for i in range(len(kT_idx3)):
                        kT_b[:,kT_idx3[i]] = np.nan

    elif var.info.Tcalc == 'diagonalisation':
        # proceed to diagonalisation if this is prefered
        print("Warning : Due to missing magnetic field data and using diagnalisation, it is not possible to track eigenvalues near degenerate points.")
        print('=> At crossings δλ/λ ≈ 1%, switch between eigenvalues λ are expected ! ','\n')
        if var.info.SC == 'SOLO':
            print("Warning : Using SOLO and the diagonalisation of the pressure tensor, for velocities < 350 km/s is not recommended due to instrumental limitations. \n")

        kT_mat = np.array([
            [var.kT.y[0], var.kT.y[3], var.kT.y[4]], 
            [var.kT.y[3], var.kT.y[1], var.kT.y[5]],
            [var.kT.y[4], var.kT.y[5], var.kT.y[2]]
        ]) 

        #diagonalise : loose track of EV at crossings
        kT_diag,_ = np.linalg.eigh(kT_mat.T)
        kT_diag = kT_diag.T
        count_T_par = np.zeros(3)

        #evaluate the most different temperature for each time and keep T// as the one that is overall the most different
        for i in range(np.shape(kT_diag)[1]) : 
            # for each t, find lowest difference between all EV = find the Tperps 
            dT = np.abs(kT_diag[:,i] - kT_diag[0,i])
            dT[0] = np.abs(kT_diag[2,i] - kT_diag[1,i])
            dT[1], dT[2] = dT[2], dT[1]

            # deduce T// by dT construction
            ind_T_par = np.argmin(dT)

            # if local agyrotropy is too high: do not consider Tparallel is well determined
            err = 1 - dT[ind_T_par]/dT
            err = err[err!=0]
            if np.min(err)> 0.1 : # define minimum gap between T// and Tperp to consider the T// found as valid
                count_T_par[ind_T_par] += 1
        
        # check there is not another coordinate that is almost equivalently counted as T// (due to T value crossings)
        indx_Tpar = np.argmax(count_T_par)
        test1, test2 = count_T_par[(indx_Tpar+1)%3]/count_T_par[indx_Tpar], count_T_par[(indx_Tpar+2)%3]/count_T_par[indx_Tpar]
        if test1 > 0.5 or test2 > 0.5 :
            print("! Warning ! : Due to the pressure tensor data at this time, the T_// component is poorly determined: to many switch occur during diagonalisation.")
            print("=> Another coordinate is detected as potential T_//.")
            print("The ratio of the #of times this coordinate is counted as T_// over the #of times the actually shown T_// coordinated is counted is: ", max(test1,test2))

        # return diagonalised tensor with T_par being the most different temperature overall
        kT_b = np.array([
            kT_diag[indx_Tpar],
            kT_diag[(indx_Tpar+1)%3],
            kT_diag[(indx_Tpar+2)%3],
        ])

    else : 
        kT_b = np.nan
    
    return kT_b


def eV_to_kms(E):
    """
    Convert energy in electron volts to velocity in kilometers per second.

    Parameters
    ----------
    E : float
        Energy in electron volts (eV).

    Returns
    -------
    float
        Velocity in kilometers per second (km/s).
    """

    return np.sqrt(E * cst.eV/ cst.m_p) * 1e-3
