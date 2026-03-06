
""" ----------------------------------------------------------------------------
Functions for downloading VDF files
---------------------------------------------------------------------------- """

import os
import cdflib
import requests

import numpy        as np
from tqdm           import tqdm
from dataclasses    import asdict
from datetime       import timedelta

            
def download_cdf(tc, dirpath, SC): # Merged from load/
    """
    Download the L2 VDF file for a given datetime and save it under
    {SC}_VDF/YYYY_MM_DD/YYYY_MM_DD.cdf

    Parameters
    ----------
    tc : datetime object
        The central date for which to download the L2 VDF file.
    dirpath : str
        The directory in which to save the downloaded file.
    SC : str
        The spacecraft for which to download the VDF data. 
        Must be 'SOLO' or 'PSP'.

    Raises
    ------
    RuntimeError
        If the file was not successfully downloaded.

    Returns
    -------
    None
    """    
    #convert datetime to useful string formats
    date1 = f"{tc.year}_{tc.month:02d}_{tc.day:02d}"
    date2 = f"{tc.year}{tc.month:02d}{tc.day:02d}"
    
    #get L2 VDF URL depending on spacecraft
    if SC == 'SOLO':     

        id = f"solo_L2_swa-pas-vdf_{date2}"    
        print(f"Downloading {id}.cdf")

        L2_vdf_url  = f"https://soar.esac.esa.int/soar-sl-tap/data?product_type=SCIENCE&RETRIEVAL_TYPE=LAST_PRODUCT&data_item_id={id}"
        response    = requests.get(L2_vdf_url, stream=True)


    elif SC == 'PSP':
        
        id = f"psp_swp_spi_sf00_l2_8dx32ex8a_{date2}_v04"    
        print(f"Downloading {id}.cdf")
        
        L2_vdf_url = f'https://cdaweb.gsfc.nasa.gov/pub/data/psp/sweap/spi/l2/spi_sf00_8dx32ex8a/{tc.year}/{id}.cdf'
        # L2_vdf_url  = f'https://w3sweap.cfa.harvard.edu/pub/data/sci/sweap/spi/L2/spi_sf00/{tc.year}/{tc.month:02d}/{id}.cdf'
        response    = requests.get(L2_vdf_url, stream=True)    
    
    # Size of the file in bytes.
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024
    
    #Download file
    with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
        with open(dirpath + f"{date1}.cdf", "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
    
    #Check if file was successfully downloaded
    if total_size != 0 and progress_bar.n != total_size:
        raise RuntimeError("Could not download file") 
        

def create_1h_cdf_file(tc, SC = 'SOLO'): # From load/
    """
    If the 1h CDF file around tc does not exist, this function:
    - downloads the L2 VDF file (if not already downloaded) and saves it as 
      {SC}_VDF/year_month_day/year_month_day.cdf
    - creates a 1-hour VDF file around the time tc and saves it as 
      {SC}_VDF/year_month_day/year_month_day_hour.cdf

    Parameters
    ----------
    tc : datetime object
        The date around which to download the L2 VDF file. 
        Start and end of the interval are on the hour
    SC : string
        Spacecraft considered, should be 'SOLO' or 'PSP'. Default is SOLO

    Raises
    ------
    RuntimeError
        If the file was not successfully downloaded.

    Returns
    -------
    None
    """

    # Convert datetime to string
    date = f"{tc.year}_{tc.month:02d}_{tc.day:02d}"
    
    # Create directory if it does not exist     
    dirpath = os.path.dirname(__file__)+f"/../../Data/{SC}_VDF/{date}/"
    # dirpath = f"./Data/{SC}_VDF/{date}/"

    if not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)
        
    # Define start time and end time as the hour around tc
    start   = tc.replace(minute=0, second=0, microsecond=0)
    end     = start + timedelta(hours=1)
        
    # if the 1h CDF file does not already exist
    if not os.path.exists(dirpath + f"{date}_{tc.hour:02d}.cdf"):
        
        # then download the 1 day vdf file if it does not already exist
        if not os.path.exists(dirpath + f"{date}.cdf"):
            download_cdf(tc, dirpath, SC)         
            
        # Load the 1-day CDF
        dat = cdflib.CDF(os.path.dirname(__file__)+f"/../../Data/{SC}_VDF/{date}/{date}.cdf")    
        # datetime vector
        t   = np.array(cdflib.cdfepoch.to_datetime(dat["Epoch"])) 
        
        # Find relevant indices
        indices = np.where(
            (t >= np.datetime64(start)) &  (t < np.datetime64(end))
        )[0]
        
        # Create new file
        new_cdf = cdflib.cdfwrite.CDF(
            os.path.dirname(__file__) + 
            f"/../../Data/{SC}_VDF/{date}/{date}_{tc.hour:02d}.cdf", 
            cdf_spec = asdict(dat.cdf_info()), 
            delete=True
        )
        
        # Copy the global attributes
        # Transform global attributes values from a list to a dictionary
        globalAttrs = {
            attr: {i: val for i, val in enumerate(vals)} 
            for attr, vals in dat.globalattsget().items()
        }

        # Write the global attributes
        new_cdf.write_globalattrs(globalAttrs)
        
        for var in dat._get_varnames()[1]:            
            # Get the variable's specification
            varinfo = dat.varinq(var)            
         
            # Get the variable's attributes            
            varattrs=dat.varattsget(var)
            
            if (varinfo.Sparse.lower() == 'no_sparse'):
                # A variable with no sparse records... get the variable data
                varVal = dat[var]
                # Check if the variable is time dependent
                if varVal.shape[0] == len(t):  
                    # Select the corresponding values
                    vardata = varVal[indices]  
                else:
                    vardata = varVal  # Copy the variable as is
                    
                # Create the zVariable, write out the attributes and data
                new_cdf.write_var(
                    asdict(varinfo), var_attrs=varattrs, var_data=vardata
                )
        
        print(f"New CDF created containing VDF from {start} to {end}")   
        new_cdf.close()

