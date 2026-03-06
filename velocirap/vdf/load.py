
""" ----------------------------------------------------------------------------
Functions to load VDF data and surrounding context
---------------------------------------------------------------------------- """

import os
import numpy            as np
import scipy.constants  as cst
import cdflib
import time

from velocirap.vdf.process  import *
from velocirap.vdf.files    import create_1h_cdf_file
from velocirap.helpers      import quaternion_rotation_matrix


def load_vdf(vdf, tc, var, rotate_to=['rtn', 'b'], particle = None, res=10, cleaning = False):    
    """
    Load the VDF data for the given spacecraft and time.
    
    Parameters
    ----------
    vdf : DotMap object
        The DotMap object where the VDF is stored
    tc : datetime object
        The time around which to load the VDF
    var : DotMap object
        The DotMap object containing the surrounding timeseries
    rotate_to : list or string, optional
        The frames to which to rotate the VDF. 
        Defaults to ['rtn', 'b'], where the vdf is rotated to both frames
        If set to None, 'etp' or 'vtp', the VDF will not be rotated.
        if set to 'rtn', the VDF will be rotated to the RTN frame.
        if set to 'b', the VDF will be rotated to the B frame.
    particle : string, optional
        The assumption of the particle type to load. Defaults to None (proton). Can be set to 'alpha'
    res : float, optional
        The resolution of the 'rtn' and 'b' grid in km/s. Defaults to 10 km/s.
    cleaning : bool, optional
        If True, the VDF will be cleaned of ghost counts. Defaults to False.

    Returns
    -------
    vdf : DotMap object
        The DotMap object where the VDF is stored, see read me and documentation 
        for a detailed description of its content
    """
    if particle is None or particle == 'proton':
        vdf.info.p_mass, vdf.info.p_charge = 1, 1
    elif particle == 'alpha':
        vdf.info.p_mass, vdf.info.p_charge = 4, 2
    else:
        raise ValueError("particle must be 'proton' or 'alpha'")
    
    # start = time.time()
    # Create the 1h CDF file if it does not exist    
    create_1h_cdf_file(tc, SC = vdf.info.SC)
    # print(f"cdf download / check --------- {time.time()-start:.6f} s") 

    # Read the 1h vdf file
    # now = time.time()
    read_func = read_SOLO_cdf if vdf.info.SC == 'SOLO' else read_PSP_cdf
    vdf = read_func(vdf, tc, cleaning = cleaning)
    # print(f"read cdf  -------------------- {time.time()-now:.6f} s")
    
    # Load context parameters from other instruments (B, Usc, etc.)
    # now = time.time()
    vdf = load_vdf_context(vdf, var)
    # print(f"load context  ---------------- {time.time()-now:.6f} s")

    # Compute VDF moments
    # now = time.time()
    vdf = compute_moments(vdf)
    # print(f"compute moments  ------------- {time.time()-now:.6f} s")

    if rotate_to not in [None, 'etp', 'vtp']:
        
        now = time.time()
        # Interpolate the ETP VDF
        interp = interpolate_VTP_vdf(vdf)

        # Resolution of the new grid in km/s
        vdf.info.res = res     

        #convert string to list
        if isinstance(rotate_to, str):
            rotate_to = [rotate_to]
        # print(f"Interpolate  ----------------- {time.time()-now:.6f} s") 
        for frame in rotate_to:
            # Rotate the VDF to the RTN frame
            now = time.time()
            if (frame == 'b') and (np.isnan(vdf.B0_rtn).all()):
                print('\nNo Magnetic field data available, skipping B frame rotation of VDF')
            else:
                vdf = rotate_vdf(vdf, interp, frame)
            # print(f"rotate vdf to {frame}  ----------- {time.time()-now:.6f} s") 
    # print(f"Total time ------------------- {time.time()-start:.6f} s")
    return vdf


def load_vdf_context(vdf, var, dt = 1): # From load/
    """
    Load the context of the measurement, i.e. the magnetic field, plasma
    moments (velocity, density, kT tensor) and the spacecraft velocity from
    the timeseries var, for a short time window defined by dt around the
    time of the measurement in vdf.t.

    Parameters
    ----------
    vdf : DotMap object
        The DotMap object containing the VDF
    var : DotMap object
        The DotMap object containing the timeseries
    dt : int, optional
        The half time window in seconds around the time of the measurement to
        average the context over. Defaults to 1 second.

    Returns
    -------
    vdf : DotMap object
        The same DotMap object with the added context variables :
        - B0_rtn          magnetic field
        - U0_rtn          proton velocity
        - N0              proton density
        - Usc_rtn         spacecraft velocity
        - kT0_rtn         kT tensor
        - Mat.xyz_to_rtn  (if SC is PSP)  rotation matrix to rtn frame
        - Mat.xyz_to_b    rotation matrix to b frame

    """
    #magnetic field
    i_b = (
        (var.B.date > vdf.info.t - np.timedelta64(dt, 's')) & 
        (var.B.date < vdf.info.t + np.timedelta64(dt, 's'))
    )
    vdf.B0_rtn = np.nanmean(var.B.y[:, i_b], axis=1)

    # Plasma moments
    # proton velocity
    i_U = (
        (var.U.date > vdf.info.t - np.timedelta64(dt, 's')) & 
        (var.U.date < vdf.info.t + np.timedelta64(dt, 's'))
    )
    vdf.U0_rtn = np.nanmean(var.U.y[:, i_U], axis=1)    #km/s
    
    #proton density
    vdf.N0 = np.nanmean(var.Np.y[:, i_U], axis=1)   #cm-3

    # Spacecraft velocity 
    vdf.Usc_rtn = np.nanmean(var.Usc.y[:, i_U], axis=1)    #km/s
    
    # kT tensor
    kT_tensor = np.nanmean(var.kT.y[:, i_U], axis=1) #eV  
    vdf.kT0_rtn = np.array([
        [kT_tensor[0], kT_tensor[3], kT_tensor[4]], 
        [kT_tensor[3], kT_tensor[1], kT_tensor[5]],
        [kT_tensor[4], kT_tensor[5], kT_tensor[2]]
    ])     


    # construct xy to rtn matrix for PSP
    if vdf.info.SC == 'PSP':            
        #rotation matrix from instrument from to SC frame
        mat_inst_to_sc = np.array([
            [ 0.        ,  0.        ,  1.        ],
            [-0.9396926 ,  0.34202015,  0.        ],
            [-0.34202015, -0.9396926 ,  0.        ]
        ])
        #quaternions from instrument from to SC frame
        Q0 = np.nanmean(var.Q.y[:, i_U], axis=1)        
        vdf.Mat.xyz_to_rtn = mat_inst_to_sc @ quaternion_rotation_matrix(Q0).T
    
    #Find the rotation matrix from RTN to B frame
    vdf         = find_rtn_to_b(vdf)    
    vdf.U0_b    = vdf.Mat.rtn_to_b @ vdf.U0_rtn   
    vdf.kT0_b   = vdf.Mat.rtn_to_b @ vdf.kT0_rtn @ np.linalg.inv(vdf.Mat.rtn_to_b)

    return vdf   


# Temporary functions
def read_SOLO_cdf(vdf, tc, cleaning = False):      
    """
    Read the 1h VDF CDF file containing tc and 
    store the VDF closest to tc with relevant variables in the vdf DotMap object
    
    Parameters
    ----------
    vdf : DotMap object
        The DotMap object where the VDF will be stored
    tc : datetime object
        The datetime object for which to read the VDF. 
    cleaning : bool, optional
        If True, the VDF will be cleaned of ghost counts. Defaults to False.
    
    Returns
    -------
    DotMap
        The updated DotMap object containing the variables
        - vdf.Axis.E        energy axis    (eV)
        - vdf.Axis.theta    elevation axis (rad)
        - vdf.Axis.phi      azimuth axis   (rad)
        - vdf.Axis.v        velocity axis  (km/s), v = sqrt(2 * E / m_p)
        - vdf.Axis.bins_etp bin edges in E, theta, phi (eV, rad, rad)
        - vdf.n_vtp         vdf values on the v theta phi grid (s3 m^(-6))
        - vdf.n_etp         vdf values on the E theta phi grid (eV^(-3) m^(-3))
        
        - vdf.Grid_vtp.v, theta, phi       meshgrid from axis v, theta, phi
        - vdf.Grid_vtp.dv, dtheta, dphi    meshgrid of bin width                        
        - vdf.Grid_vtp.d3v                 meshgrid of bin volume
        
        - vdf.Mat.xyz_to_rtn    rotation matrix from sensor (xyz) to rtn frame 
        
        - vdf.u_xyz         meshgrid of ux, uy, uz
          
    """
      
    mass, charge = vdf.info.p_mass, vdf.info.p_charge

    """ Read CDF file """
    # Convert datetime to string
    date = f"{tc.year}_{tc.month:02d}_{tc.day:02d}"
        
    # Read the relevant CDF file
    cdf = cdflib.CDF(os.path.dirname(__file__)+f"/../../Data/SOLO_VDF/{date}/{date}_{tc.hour:02d}.cdf")
    # cdf = cdflib.CDF(f"./Data/SOLO_VDF/{date}/{date}_{tc.hour:02d}.cdf")
    
    # Time vector in datetime 
    t   = np.array(cdflib.cdfepoch.to_datetime(cdf["Epoch"]))
    # t   = np64_to_datetime(t)

    # Choose the relevant vdf to plot, closest to tc
    i_vdf = np.argmin(np.abs(t - np.datetime64(tc)))
    vdf.info.t = t[i_vdf]
    
    # Raise a warning if no datapoint is found within 2s of tc
    dt = (t[i_vdf] - np.datetime64(tc)) / np.timedelta64(1, 's')
    if np.abs(dt) > 2:
        print(
            f"Could not find a VDF within 2s of {tc} \n" 
            + f"Using the closest one {
                np.round(np.abs(dt), 1)
            }s away"
        )

    """ Store variables in DotMap object """
    
    #vdf axis
    vdf.Axis.E       = cdf["Energy"][::-1] * mass / charge #eV
    vdf.Axis.theta   = cdf["Elevation"] * np.pi / 180 #rad
    vdf.Axis.phi     = cdf["Azimuth"] * np.pi / 180 #rad
    
    #velocity axis assuming only protons
    vdf.Axis.v = np.sqrt(2 * cst.eV * vdf.Axis.E / (mass * cst.m_p)) * 1e-3 #km/s
    
    # vdf values 
    # Transform the vdf so that it matches the desired structure of 
    # E, theta, phi (96, 9, 11) with increasing energy
    vdf.n_vtp       = cdf["vdf"].T[::-1, ..., i_vdf] * (charge) ** (3/2)#s3m-6

    if cleaning:
        vdf = clean_SOLO_vdf(vdf)    
    
    #compute the Energy distribution function (in eV^(-3) m^(-3))
    vdf.n_etp = vdf.n_vtp * 0.5 * (
        (mass * cst.m_p) * vdf.Axis.E[:,None,None] /cst.eV /2
    ) ** (-3/2) #

    # Compute the meshgrids of bin centers and bin widths
    # variations around axis values
    dE_m    = cdf["delta_m_Energy"][::-1] * mass / charge #eV
    dE_p    = cdf["delta_p_Energy"][::-1] * mass / charge #eV
    dtheta  = cdf["delta_Elevation"] * np.pi / 180 #rad
    dphi    = cdf["delta_Azimuth"] * np.pi / 180  #rad
    
    #grid in eV, rad, rad 
    # WARNING : bin definition not always consistent (overlapping in theta, phi) 
    vdf.Axis.bins_etp = [
        np.append(vdf.Axis.E - dE_m,       vdf.Axis.E[-1] + dE_p[-1]),
        np.append(vdf.Axis.theta - dtheta, vdf.Axis.theta[-1] + dtheta[-1]),
        np.append(vdf.Axis.phi - dphi,     vdf.Axis.phi[-1] + dphi[-1])         
    ]    
    
    
    #grid width in E, theta, phi (eV rad rad)
    dBins = [np.diff(b) for b in vdf.Axis.bins_etp]
    #grid width in v, theta, phi
    dBins[0] = dBins[0] * cst.eV / np.sqrt(2 * mass * cst.m_p * vdf.Axis.E * cst.eV)* 1e-3 #km/s

    #meshgrids of bin center and bin width
    vdf.Grid_vtp.v, vdf.Grid_vtp.theta, vdf.Grid_vtp.phi = np.array(
        np.meshgrid(vdf.Axis.v, vdf.Axis.theta, vdf.Axis.phi, indexing='ij')
    ) #km/s, rad, rad

    vdf.Grid_vtp.dv, vdf.Grid_vtp.dtheta, vdf.Grid_vtp.dphi  =np.array(
        np.meshgrid(*dBins, indexing='ij')
    ) #km/s, rad, rad
    
    #meshgrid of the volume of each bin in m3/s3
    vdf.Grid_vtp.d3v = (
        (vdf.Grid_vtp.v * 1e3) **2 
        * np.cos(vdf.Grid_vtp.theta) 
        * vdf.Grid_vtp.dv * 1e3 
        * vdf.Grid_vtp.dtheta 
        * vdf.Grid_vtp.dphi 
    )

    #spherical metric        
    vdf.Grid_vtp.metric = np.array([
        vdf.Grid_vtp.dv * 1e3, 
        vdf.Grid_vtp.v * 1e3 * vdf.Grid_vtp.dtheta, 
        vdf.Grid_vtp.v * 1e3 * np.cos(vdf.Grid_vtp.theta) * vdf.Grid_vtp.dphi
    ])    
    
    #Rotation matrix to rtn frame
    # sensor frame (xyz) to PAS :
    sensor_to_PAS = np.diag([-1., -1., 1.]) 

    # sensor frame to rtn frame (= RTN for SOLO) : 
    pas_to_rtn  = cdf["PAS_to_RTN"][i_vdf]
    vdf.Mat.xyz_to_rtn = pas_to_rtn @ sensor_to_PAS 
    
    #Compute the velocity in the sensor field
    vdf = spher2cart(vdf) #km/s 

    return vdf


def read_PSP_cdf(vdf, tc, cleaning= False):  
    """
    Read the 1h VDF CDF file containing tc and store the VDF closest to tc with 
    relevant variables in the vdf DotMap object
    
    Parameters
    ----------
    vdf : DotMap object
        The DotMap object where the VDF will be stored
    tc : datetime object
        The datetime object for which to read the VDF. 
    cleaning : bool, optional
        Does not apply, here only to be consistent with read_SOLO_cdf
    
    Returns
    -------
    DotMap
        The updated DotMap object containing the variables
        - vdf.Axis.E        energy axis    (eV)
        - vdf.Axis.theta    elevation axis (rad)
        - vdf.Axis.phi      azimuth axis   (rad)
        - vdf.Axis.v        velocity axis  (km/s), v = sqrt(2 * E / m_p)
        - vdf.Axis.bins_etp bin edges in E, theta, phi (eV, rad, rad)
        - vdf.n_vtp         vdf values on the v theta phi grid (s3/m6)
        
        - vdf.Grid_vtp.v, theta, phi       meshgrid from axis v, theta, phi
        - vdf.Grid_vtp.dv, dtheta, dphi    meshgrid of bin width                        
        - vdf.Grid_vtp.d3v                 meshgrid of bin volume
        
        - vdf.u_xyz         meshgrid of ux, uy, uz
        
    Note
    ----    
        The vdf.Mat.xyz_to_rtn rotation matrix is defined 
        in the load context function for PSP (needs quaternion values) 
    """
    
    # Read CDF file 
    # Convert datetime to string
    date = f"{tc.year}_{tc.month:02d}_{tc.day:02d}"
        
    # Read the relevant CDF file
    cdf = cdflib.CDF(os.path.dirname(__file__)+f"/../../Data/PSP_VDF/{date}/{date}_{tc.hour:02d}.cdf")
    
    
    # Time vector in datetime 
    t   = np.array(cdflib.cdfepoch.to_datetime(cdf["Epoch"]))
    # t   = np64_to_datetime(t)

    # Choose the relevant vdf to plot, closest to tc
    i_vdf = np.argmin(np.abs(t - np.datetime64(tc)))
    vdf.info.t = t[i_vdf]
    
    # Raise a warning if no datapoint is found within 2s of tc
    dt = (t[i_vdf] - np.datetime64(tc)) / np.timedelta64(1, 's')
    if np.abs(dt) > 2:
        print(
            f"Could not find a VDF within 2s of {tc} \n" 
            + f"Using the closest one {
                np.round(np.abs(dt), 1)}s away")
    
    # Store variables in DotMap object 

    #vdf axis (eV, rad, rad)
    vdf.Axis.E     = cdf["ENERGY"][i_vdf].reshape(8, 32, 8)[-1, ::-1, -1]
    vdf.Axis.theta = cdf["THETA"][i_vdf].reshape(8, 32, 8)[-1, -1]*np.pi/180
    vdf.Axis.phi   = cdf["PHI"][i_vdf].reshape(8, 32, 8)[::-1, -1, -1]*np.pi/180
    
    #correcting a bug where the last phi value is nan sometimes
    if np.isnan(vdf.Axis.phi[-1]):
        print(
            'Error in azimuth range definition, ' \
            'forcing last phi value to 174.375 instead of nan'
        )
        vdf.Axis.phi[-1] = 174.375 * np.pi / 180
    
    #velocity axis
    vdf.Axis.v = np.sqrt(2 * cst.eV * vdf.Axis.E / cst.m_p) *1e-3 #km/s
    
    # vdf values 
    # Transform the vdf so that it matches the desired structure of 
    # E, theta, phi (32, 8, 8) with increasing energy
    density     = (
        cdf["EFLUX"] * 1e4 
        / cdf["ENERGY"] 
        * (cst.m_p / cst.eV)**2 / 2 
        / cdf["ENERGY"]
    ) #s3m-6
    vdf.n_vtp   = np.moveaxis(
        density[i_vdf].reshape(8, 32, 8), 0, 2
    )[::-1,:,::-1]

    
    #compute the Energy distribution function (in eV^(-3) m^(-3))
    vdf.n_etp = vdf.n_vtp * 0.5 * (
        cst.m_p * vdf.Axis.E[:,None,None] /cst.eV /2
    ) ** (-3/2) #
    
        
    #Compute the meshgrids of bin centers and bin widths
    # variations around axis values
    dE      = np.diff(vdf.Axis.E) /2 #eV
    dtheta  = np.nanmean(np.diff(vdf.Axis.theta)) /2 #rad
    dphi    =  np.nanmean(np.diff(vdf.Axis.phi))  /2 #rad
    
    #grid in eV, rad, rad 
    vdf.Axis.bins_etp = [
        np.append(
            vdf.Axis.E[0] - dE[0], 
            np.append(vdf.Axis.E[1:] - dE, vdf.Axis.E[-1] + dE[-1])
        ),
        np.append(vdf.Axis.theta - dtheta, vdf.Axis.theta[-1] + dtheta),
        np.append(vdf.Axis.phi - dphi,     vdf.Axis.phi[-1] + dphi)         
    ]    
    
    #grid width in V, theta, phi
    dBins = [np.diff(b) for b in vdf.Axis.bins_etp]
    dBins[0] = dBins[0] * cst.eV / np.sqrt(2 * cst.m_p * vdf.Axis.E * cst.eV) * 1e-3 # km/s

    #meshgrids of bin center and bin width
    vdf.Grid_vtp.v, vdf.Grid_vtp.theta, vdf.Grid_vtp.phi = np.array(
        np.meshgrid(vdf.Axis.v, vdf.Axis.theta, vdf.Axis.phi, indexing='ij')
    ) #km/s, rad, rad
    vdf.Grid_vtp.dv, vdf.Grid_vtp.dtheta, vdf.Grid_vtp.dphi  =np.array(
        np.meshgrid(*dBins, indexing='ij')
    ) #km/s, rad, rad
    
    #meshgrid of the volume of each bin in m3/s3
    vdf.Grid_vtp.d3v = (
        (vdf.Grid_vtp.v * 1e3) **2 
        * np.cos(vdf.Grid_vtp.theta) 
        * vdf.Grid_vtp.dv * 1e3 
        * vdf.Grid_vtp.dtheta 
        * vdf.Grid_vtp.dphi
    )
    
    #spherical metric        
    vdf.Grid_vtp.metric = np.array([
        vdf.Grid_vtp.dv * 1e3, 
        vdf.Grid_vtp.v * 1e3 * vdf.Grid_vtp.dtheta, 
        vdf.Grid_vtp.v * 1e3 * np.cos(vdf.Grid_vtp.theta) * vdf.Grid_vtp.dphi
    ])    
    
    #Compute the velocity in the sensor field
    vdf = spher2cart(vdf) #km/s    
    
    return vdf