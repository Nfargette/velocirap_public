
""" ----------------------------------------------------------------------------
Functions to process VDF data (interpolation, rotation, integration)
---------------------------------------------------------------------------- """

import numpy           as np
import scipy.constants as cst
from scipy.interpolate import RegularGridInterpolator
from velocirap.helpers import integrate

def compute_moments(vdf): 
    """
    Compute the moments of the velocity distribution function (VDF) 
    in the RTN frame
    
    Parameters
    ----------
    vdf : DotMap object
        The DotMap object containing the VDF

    Returns
    -------
    vdf : DotMap object
        The same DotMap object with the added moments variables :
        - vdf.N            density
        - vdf.U_rtn        velocity in RTN frame
        - vdf.U_b          velocity in B frame
        - vdf.kT_rtn       pressure tensor in RTN frame
        - vdf.kT_b         pressure tensor in B frame
    """
    
    #density
    vdf.N = integrate(1, vdf.n_vtp, vdf.Grid_vtp.metric, axis=(0, 1, 2)) * 1e-6 #cm-3
    
    #velocity
    U_xyz       = integrate(vdf.u_xyz, vdf.n_vtp, vdf.Grid_vtp.metric, axis=(0, 1, 2)) #km/s  
    vdf.U_rtn   = vdf.Mat.xyz_to_rtn @ U_xyz + vdf.Usc_rtn  #km/s 
    vdf.U_b     = vdf.Mat.rtn_to_b @ vdf.U_rtn    
    
    #Pressure tensor
    Du  = (vdf.u_xyz - U_xyz[:,None,None,None]) * 1e3 #m/s
    Du2 = np.einsum('i..., j... -> ij...', Du, Du)    
    vdf.kT_xyz = vdf.info.p_mass * cst.m_p * integrate(Du2, vdf.n_vtp, vdf.Grid_vtp.metric, axis=(0, 1, 2)) / cst.eV
    vdf.kT_rtn = vdf.Mat.xyz_to_rtn @ vdf.kT_xyz @ np.linalg.inv(vdf.Mat.xyz_to_rtn)
    vdf.kT_b   = vdf.Mat.xyz_to_b @ vdf.kT_xyz @ np.linalg.inv(vdf.Mat.xyz_to_b)
    
    kT = np.trace(vdf.kT_xyz) / 3  #eV
    Vth =np.sqrt( np.trace(vdf.kT_xyz) /3 * cst.eV  * 2 / cst.m_p) * 1e-3

    # print('VDF moments :')
    # print(f" Density :        {vdf.N :0.1f} cm-3")
    # print(f" Velocity (RTN) : [{vdf.U_rtn[0]:0.1f}, {vdf.U_rtn[1]:0.1f}, {vdf.U_rtn[2]:0.1f}] km/s")
    # print(f" Velocity (L2) :  [{vdf.U0_rtn[0]:0.1f}, {vdf.U0_rtn[1]:0.1f}, {vdf.U0_rtn[2]:0.1f}] km/s")
    # print(f" kT tensor (B) :  [{vdf.kT_b[0,0]:0.1f}, {vdf.kT_b[1,1]:0.1f}, {vdf.kT_b[2,2]:0.1f}] eV")
    # print(f" kT tensor (L2) : [{vdf.kT0_b[0,0]:0.1f}, {vdf.kT0_b[1,1]:0.1f}, {vdf.kT0_b[2,2]:0.1f}] eV")

    
    return vdf

def spher2cart(vdf):      
    """
    Transform spherical coordinates V THETA PHI 
    in cartesian coordinates Ux Uy Uz (sensor frame)
    
    Parameters
    ----------
    vdf : DotMap object
        The DotMap object where the VDF is stored
    
    Returns
    -------
    vdf : DotMap object
        The DotMap object where the VDF is stored including
        - vdf.u_xyz the meshgrid of ux, uy, uz
    """
    # Transform V THETA PHI in Ux Uy Uz (sensor frame) 
    vdf.u_xyz = np.array([
        vdf.Grid_vtp.v * np.cos(vdf.Grid_vtp.theta) * np.cos(vdf.Grid_vtp.phi), 
        vdf.Grid_vtp.v * np.cos(vdf.Grid_vtp.theta) * np.sin(vdf.Grid_vtp.phi), 
        vdf.Grid_vtp.v * np.sin(vdf.Grid_vtp.theta)
    ])    
    return vdf


def rotate_vdf(vdf, interp, frame): 

    """
    Rotate the VDF to a new frame (RTN or B)
    
    Parameters
    ----------
    vdf : DotMap object
        The DotMap object where the VDF is stored
    interp : RegularGridInterpolator object
        The interpolating function of the VTP VDF
    frame : str
        The new frame of the VDF, either 'rtn' or 'b'
    
    Returns
    -------
    vdf : DotMap object
        The updated DotMap object containing the rotated VDF and its axes :
        - vdf.Axis.u_r, vdf.Axis.u_t, vdf.Axis.u_n (1d axes)
        - vdf.u_rtn (meshgrid)
        - vdf.n_rtn (vdf)
        or 
        - vdf.Axis.u_b, vdf.Axis.u_exb, vdf.Axis.u_e (1d axes)
        - vdf.u_b (meshgrid)
        - vdf.n_b (vdf)
    """
    # define rotation matrices and labels depending on chosen frame
    if frame=='rtn':
        mat_xyz_to_new = vdf.Mat.xyz_to_rtn
        mat_rtn_to_new = np.eye(3)
        axes = ['u_r', 'u_t', 'u_n']
        mesh = 'u_rtn'
        vdf_name = 'n_rtn'
    elif frame=='b':
        mat_xyz_to_new = vdf.Mat.xyz_to_b
        mat_rtn_to_new = vdf.Mat.rtn_to_b
        # note : e=-uxb
        axes = ['u_b', 'u_exb', 'u_e']
        mesh = 'u_b'
        vdf_name = 'n_b'
    else:
        raise ValueError("frame must be 'rtn' or 'b'")
    
    # restraining the energy range to where vdf >0
    
    # Compute v histogram
    dist1d = integrate(
            1, vdf.n_vtp, vdf.Grid_vtp.metric, axis=tuple({0, 1, 2} - {0})
        )

    # where vdf >0
    i_vdf = dist1d > 0

    # rotate the original grid u_xyz to the new frame
    u_new = np.einsum('ij, jklm -> iklm', mat_xyz_to_new, vdf.u_xyz[:, i_vdf]) 

    # correct for the spacecraft velocity 
    u_new = u_new + (mat_rtn_to_new @ vdf.Usc_rtn)[:, None, None, None]

    # find the number of points in each directions for a linearly spaced grid
    # of resolution vdf.info.res
    N = [
        int((np.nanmax(u_new[i]) - np.nanmin(u_new[i])) / vdf.info.res)
        for i in range(3)
    ] 
    
    # define new linearly spaced grid in the new frame (here RTN)
    vdf.Axis[axes[0]], vdf.Axis[axes[1]], vdf.Axis[axes[2]] = [
        np.linspace(
            np.nanmin(u_new[i]), 
            np.nanmax(u_new[i]), 
            N[i]
        ) for i in range(3)
    ]

    #define the associated meshgrid
    vdf[mesh] = np.array(
        np.meshgrid(
            vdf.Axis[axes[0]], vdf.Axis[axes[1]], vdf.Axis[axes[2]], 
            indexing="ij"
        )
    )

    # transpose the new grid to xyz
    M_inv = np.linalg.inv(mat_xyz_to_new)   
    u_xyz = np.einsum(
        'ij, jklm -> iklm', 
        M_inv, 
        vdf[mesh]- (mat_rtn_to_new @ vdf.Usc_rtn)[:, None, None, None]
    )
    
    # transpose the new grid to v, theta, phi 
    v       = np.linalg.norm(u_xyz, axis=0)
    theta   = np.pi / 2 - np.arccos(u_xyz[2] / v)
    phi     = np.arctan2(u_xyz[1], u_xyz[0])

    # use the interpolating function to compute the vdf value on the new grid
    # print(f"Interpolating VDF to {frame} frame...")
    # start = time.time()
    vdf[vdf_name] = interp((v, theta, phi))
    # print(f"Interpolation completed in {time.time()-start:.2f} seconds")
            
    return vdf


def find_rtn_to_b(vdf): # From load/
    """
    Computes the rotation matrix from the rtn frame to B frame 
    from the magnetic field and velocity directions

    Parameters
    ----------
    vdf : DotMap object
        The DotMap object where the VDF is stored


    Returns
    -------
    vdf : DotMap object
        The same DotMap object including
        - vdf.data_to_b the rotation matrix from the data frame to B frame 
    """    


    b = vdf.B0_rtn / np.linalg.norm(vdf.B0_rtn)
    u = vdf.U0_rtn / np.linalg.norm(vdf.U0_rtn)
    
    #Construct rotation matrix to B frame
    u_x_b       = np.cross(u, b) / np.linalg.norm(np.cross(u, b))
    b_x_u_x_b   = np.cross(b, u_x_b) / np.linalg.norm(np.cross(b, u_x_b))
    
    #matrix to go from b frame to RTN frame
    B_to_rtn = np.array([b, b_x_u_x_b, -u_x_b]).T
    
    #matrix to go from RTN frame to b frame
    vdf.Mat.rtn_to_b = np.linalg.inv(B_to_rtn)
    
    #matrix to go from sensor frame to b frame
    vdf.Mat.xyz_to_b = vdf.Mat.rtn_to_b @ vdf.Mat.xyz_to_rtn 
        
    return vdf


def interpolate_VTP_vdf(vdf): # From load/
    """
    Create a linear interpolating function for the VTP VDF
    
    Parameters
    ----------
    vdf : DotMap object
        The DotMap object where the VDF is stored
    
    Returns
    -------
    interp : RegularGridInterpolator object
        An interpolating function where interp(v, theta, phi) gives the interpolated
        vdf value for the given v, theta, phi coordinates
    """
    interp = RegularGridInterpolator(
        (vdf.Axis.v, vdf.Axis.theta, vdf.Axis.phi ), 
        vdf.n_vtp, 
        bounds_error=False
    )      
    return interp


def clean_SOLO_vdf(vdf):   
    """
    Clean a SOLO VDF by removing ghost counts.

    A ghost count is defined as a non-zero value in the VDF, 
    which is surrounded by zero or all but one zero values.
    
    Parameters
    ----------
    vdf : DotMap object
        The DotMap object where the VDF is stored

    Returns
    -------
    vdf : DotMap object
        The same DotMap object with the low energy bins and ghost counts removed
    """

    # remove low energy bins
    # vdf.n_vtp[vdf.Axis.v < 300] = 0
    # vdf.n_vtp[vdf.n_vtp < np.nanmax(vdf.n_vtp) * 1e-4] = 0
    
    # #remove ghost counts
    
    ghost = is_ghost(vdf.n_vtp)
    # print(f"Removed {np.sum(ghost)} ghost counts")

    vdf.n_vtp[ghost] = 0      
    
    return vdf

def is_ghost(n_etp, K = 1):   

    """
    Identifies ghost counts in a VDF. A ghost count is defined as a non-zero value
    in the VDF, which is surrounded by zero or all but one zero values.

    Parameters
    ----------
    n_etp : 3D array
        The VDF values in a 3D grid.
    K : int, optional
        The number of non-zero values required in the surrounding cells to
        consider a value as not a ghost count. Default is 1.

    Returns
    -------
    3D boolean array
        A boolean array with the same shape as n_etp, where True values
        indicate ghost counts and False values indicate non-ghost counts.
    """     
    
    # Initialize a boolean array with the same shape as n_etp, 
    # set to False initially
    
    ghost = np.zeros_like(n_etp, dtype=bool)    
    
    # Find positions of non-zero values in n_etp
    pos_non_zero = np.argwhere(n_etp > 0)
    
    # Iterate over each position of non-zero values
    for pos in (pos_non_zero):      
        # Get the coordinates of the adjacent cells
        tiles = get_adjacent_cells(pos, n_etp.shape)
        
        # Get the VDF values of the adjacent cells
        surrounding_values = [n_etp[t[0], t[1], t[2]] for t in tiles]
        
        # Check if all or all except 1 of the surrounding values are zero
        if (np.count_nonzero(surrounding_values) <= K):
            ghost[*pos] = True

    return ghost

def get_adjacent_cells(pos, grid_shape):
    """
    Return a list of adjacent cell coordinates in a grid of shape grid_shape,
    given a cell at coordinates pos = [x, y, z].

    Parameters
    ----------
    pos : tuple of int
        The coordinates of the cell in question
    grid_shape : tuple of int
        The shape of the grid

    Returns
    -------
    list of list of int
        A list of adjacent cell coordinates, each represented as a list of integers

    Raises
    ------
    ValueError
        If the coordinates pos are out of bounds of the grid of shape grid_shape
    """
    result = []

    #check if coordinates are out of bounds
    out_of_bounds = [pos[i] >= grid_shape[i] for i in range(len(grid_shape))]
    if np.any(pos<0)  or np.any(out_of_bounds):
        raise ValueError(
            f"Coordinates ({pos}) are out of bounds of the grid of shape {grid_shape}"
        )
    
    #else compute adjacent tiles
    else:
        for x,y,z in [(pos[0]+i,pos[1]+j, pos[2]+k) for i in (-1,0,1) for j in (-1,0,1) for k in (-1,0,1) if i != 0 or j != 0 or k!=0]:
            if x>=0 and y>=0 and z>=0 and x<grid_shape[0] and y<grid_shape[1] and z<grid_shape[2]:
                result.append([int(x), int(y), int(z)])
        return result
    



