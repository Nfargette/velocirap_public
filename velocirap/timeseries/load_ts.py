
""" ----------------------------------------------------------------------------
Functions for loading in-situ data
---------------------------------------------------------------------------- """

import numpy            as np
import speasy           as spz
import scipy.constants  as cst
from speasy.products    import SpeasyVariable
from datetime           import timedelta

from velocirap.timeseries.dates     import date_to_sec
from velocirap.timeseries.transform import kT_tensor_rtn2b, kT_tensor_inst2rtn
    
def format_data(folder_name, tb, te): # Merge with load/
    """ Load data from Speasy and format it to be used in the code
     
    Parameters
    ----------
    folder_name : str
        The name of the folder containing the data
    tb : datetime
        The begining of the time interval
    te : datetime
        The end of the time interval
    
    Returns
    -------
    y : numpy array
        The data values
    date : array of datetime
        The time array in datetime
    t : array of float
        The time array in seconds
    u : str
        The units of the data
    E, pad : list of arrays
        The energy and pad axes (if existant)
    """

    # Load data from Speasy
    # note that the data object contains all the metadata if needed, 
    # we only extract the relevant data here
    data: SpeasyVariable = spz.get_data(folder_name, tb, te) 

    #if no data is found
    if (data is None) or (data.values.shape[0] ==0)  :
        # print(f'No data found in {folder_name} between {tb} and {te}.')
        return None, None, None, None, None
    else:
        y       = data.values.T # Return the data values 
        date    = data.time # time
        t       = date_to_sec(data.time, tb) # and time in seconds
        u       = data.unit  # Return the data units 
        
        # if loading pad, extract axis
        E = None
        pad = None
        if len(y.shape)> 2 :
            E   = data.axes[2].values[0] #Energy axis
            pad = data.axes[1].values #pad axis
        
        return y, date, t, u, [E, pad]

    
def load_PSP_timeseries(var, tc, Dt=10, burst = True): # From load/
    """
    Load relevant timeseries data for PSP into the given DotMap object

    Parameters
    ----------
    var : DotMap object
        The DotMap object where the timeseries will be stored
    tc : datetime object
        The central time of the time interval
    Dt : integer
        The desired length of the time interval in minutes, default is 10
    burst : boolean, optional
        If True, load burst mode data if available. Defaults to True.

    Returns
    -------
    DotMap
        The updated DotMap object containing the variables
        - B   : Magnetic field in RTN frame (nT)
        - Ne  : Electron density in cm-3, if available
        - Np  : Proton density in cm-3
        - U   : Bulk velocity in km/s in RTN frame
        - Usc : Spacecraft velocity in km/s in RTN frame
        - kT  : Temperature in eV in RTN frame
        - kT_b : Temperature in eV in B frame
    """
    #beginning and end of the time interval
    tb, te = tc - timedelta(minutes=Dt/2), tc + timedelta(minutes=Dt/2)

    # print(f'\nLoading timeseries...')

    #Load MAG data
    if burst:
        # B (nT)
        var.B.y, var.B.date, var.B.t, var.B.units, _ = format_data(
            'cda/PSP_FLD_L2_MAG_RTN/psp_fld_l2_mag_RTN', tb, te
        ) 
    else:
        var.B.y, var.B.date, var.B.t, var.B.units, _ = format_data(
            'cda/PSP_FLD_L2_MAG_RTN_4_SA_PER_CYC/psp_fld_l2_mag_RTN_4_Sa_per_Cyc', 
            tb, te
        )
    
    # #Load B and U in instrument frame
    # var.B_xyz.y, var.B_xyz.date, var.B_xyz.t, var.B_xyz.units, _ = format_data(
    #     'cda/PSP_SWP_SPI_SF00_L3_MOM/MAGF_INST', tb, te
    # )
    # var.U_xyz.y, var.U_xyz.date, var.U_xyz.t, var.U_xyz.units, _ = format_data(
    #     'cda/PSP_SWP_SPI_SF00_L3_MOM/VEL_INST', tb, te
    # )    
    
    #Load QTN density if possible
    # ne (cm-3)     
    var.Ne.y, var.Ne.date, var.Ne.t, var.Ne.units, _ = format_data(
        'cda/PSP_FLD_L3_SQTN_RFS_V1V2/electron_density', tb, te
    )
    #delete ne if empty  
    if (var.Ne.y is None) or (np.all(var.Ne.y<0)):
        del var['Ne']
    else:
        #remove spurious datapoints
        var.Ne.y[var.Ne.y<0] = np.nan
        
    #Load SPAN ION data
    # np (cm-3)  
    var.Np.y, var.Np.date, var.Np.t, var.Np.units, _ = format_data(
        'cda/PSP_SWP_SPI_SF00_L3_MOM/DENS', tb, te
    )   
    var.Np.y[:, np.argwhere(var.Np.y[0]<1)] = np.nan

    # U (km/s)    
    var.U.y, var.U.date, var.U.t, var.U.units, _ = format_data(
        'cda/PSP_SWP_SPI_SF00_L3_MOM/VEL_RTN_SUN', tb, te
    ) 

    #spacecraft velocity (km/s)
    var.Usc.y, var.Usc.date, var.Usc.t, var.Usc.units, _ = format_data(
        'cda/PSP_SWP_SPI_SF00_L3_MOM/SC_VEL_RTN_SUN', tb, te
    )
    
    #kT tensor in instrument frame (eV)
    kT_inst, var.kT.date, var.kT.t, var.kT.units, _ = format_data(
        'cda/PSP_SWP_SPI_SF00_L3_MOM/T_TENSOR_INST', tb, te
    )
    
    #quaternions SC_to_rtn
    var.Q.y, var.Q.date, var.Q.t, var.Q.units, _ = format_data(
        'cda/PSP_SWP_SPI_SF00_L3_MOM/QUAT_SC_TO_RTN', tb, te
    )  

    # print(f"Timeseries loaded in {time.time()-start:.2f} seconds \n")

    #rotate to RTN
    var.kT.y = kT_tensor_inst2rtn(kT_inst, var.Q.y)
    
    #put kT tensor in B frame
    var.kT_b.y = kT_tensor_rtn2b(var)
    var.kT_b.t, var.kT_b.units = var.kT.t, var.kT.units 
         
    return var


def load_SOLO_timeseries(var, tc, Dt=10, burst = True):  
    """
    Load relevant timeseries data for SOLO into the given DotMap object
    
    Parameters
    ----------
    var : DotMap object
        The DotMap object where the timeseries will be stored
    tc : datetime object
        The central time of the time interval
    Dt : integer
        The desired length of the time interval in minutes, default is 10
    burst : boolean, optional
        If True, load burst mode data if available. Defaults to True.
    
    Returns
    -------
    DotMap
        The updated DotMap object containing the variables
        - B   : Magnetic field in RTN frame (nT)
        - Ne  : Electron density in cm-3, if available
        - Np  : Proton density in cm-3
        - U   : Bulk velocity in km/s in RTN frame
        - Usc : Spacecraft velocity in km/s in RTN frame
        - kT  : Temperature in eV in RTN frame
        - kT_b : Temperature in eV in B frame
    """
    #beginning and end of the time interval
    tb, te = tc - timedelta(minutes=Dt/2), tc + timedelta(minutes=Dt/2)

    # print('\nLoading timeseries...')

    #Load MAG data
    if burst:
        # B (nT)
        var.B.y, var.B.date, var.B.t, var.B.units, _ = format_data(
            'amda/solo_b_rtn_hr', tb, te
        ) 
    else:
        var.B.y, var.B.date, var.B.t, var.B.units, _ = format_data(
            'amda/solo_b_rtn', tb, te
        )
    
    #Load QTN density if possible
    # ne (cm-3)     
    var.Ne.y, var.Ne.date, var.Ne.t, var.Ne.units, _ = format_data(
        'amda/solo_rpw10s_ne', tb, te
    )
    #delete ne if empty  
    if (var.Ne.y is None) or (np.all(var.Ne.y<0)):
        del var['Ne']
    else:
        #remove spurious datapoints
        var.Ne.y[var.Ne.y<0] = np.nan
        
    #Load PAS data
    # np (cm-3)  
    var.Np.y, var.Np.date, var.Np.t, var.Np.units, _ = format_data(
        'amda/pas_momgr_n', tb, te
    )   

    # U (km/s)    
    var.U.y, var.U.date, var.U.t, var.U.units, _ = format_data(
        'amda/pas_momgr1_v_rtn', tb, te
    ) 
    #remove spurious datapoints
    var.U.y[..., (np.linalg.norm(var.U.y, axis=0)>1e25) ] = np.nan

    #spacecraft velocity (km/s)
    var.Usc.y, var.Usc.date, var.Usc.t, var.Usc.units, _ = format_data(
        'amda/pas_momgr1_v_solo_rtn', tb, te
    )
    
    #pressure tensor (J/cm-3)
    var.P.y, var.P.date, var.P.t, var.P.units, _ = format_data(
        'amda/pas_momgr_pressrtn', tb, te
    )
    
    #convert pressure tensor to kT
    var.kT.y = var.P.y / cst.eV / var.Np.y[0]
    var.kT.date, var.kT.t = var.P.date, var.P.t
    var.kT.units = 'eV'

    #remove pressure tensor
    del var['P']
        
    # print(f"Timeseries loaded in {time.time()-start:.2f} seconds \n")
    
    #rotate kT tensor to B frame
    var.kT_b.y = kT_tensor_rtn2b(var)
    var.kT_b.t, var.kT_b.units = var.kT.t, var.kT.units 

    return var



def load_timeseries(var, tc, SC, Dt = 10, burst = True):
    """
    Load the relevant timeseries data for the given spacecraft and time.
    
    Parameters
    ----------
    var : DotMap object
        The DotMap object where the timeseries will be stored
    tc : datetime object
        The central time of the time interval
    SC : string
        The name of the spacecraft, should be PSP or SOLO
    Dt : integer, optional
        The desired length of the time interval in minutes, default is 10 min
    burst : boolean, optional
        If True, load burst mode data of the magnetic field, else load normal mode. Default to True.
    
    Returns
    -------
    DotMap
        The updated DotMap object containing the variables
        - B   : Magnetic field in RTN frame (nT)
        - Ne  : Electron density in cm-3, if available
        - Np  : Proton density in cm-3
        - U   : Bulk velocity in km/s in RTN frame
        - Usc : Spacecraft velocity in km/s in RTN frame
        - kT  : Temperature in eV in RTN frame
        - kT_b : Temperature in eV in B frame
    """
    if SC not in ['SOLO', 'PSP']:
        raise ValueError(f"SC must be 'SOLO' or 'PSP', not {SC}")
    
    if burst and Dt>120:
        print(
            "Warning: loading Burst mode data for Dt>2h. Consider setting burst = False"
        )

    #define load function depending on spacecraft
    load_func = load_SOLO_timeseries if SC == 'SOLO' else load_PSP_timeseries 

    #load timeseries
    var = load_func(var, tc, Dt, burst)
    
    return var
