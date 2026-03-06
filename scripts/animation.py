
""" ----------------------------------------------------------------------------
This script generates an animation of the VDF in a given frame, 
together with the surrounding timeseries
---------------------------------------------------------------------------- """
         
import matplotlib.pyplot as plt
import numpy             as np

from pathlib  import Path    
from dotmap   import DotMap
from datetime import datetime, UTC
from tqdm     import tqdm

from velocirap.plot                 import plot_film, create_animation
from velocirap.vdf.load             import load_vdf
from velocirap.timeseries.load_ts   import load_timeseries

vdf = DotMap() #initialize DotMap object to store the VDF
var = DotMap() #initialize DotMap object to store the surrounding timeseries

""" Select Spacecraft, date and frame ---------------------------------------"""

# Spacecraft should be 'SOLO' or 'PSP'
vdf.info.SC = 'SOLO'

#time interval to plot
tb = datetime(2022, 3, 1, 2, ) #begining
te = datetime(2022, 3, 1, 3, ) #end

#frame of projection, should be 'etp', 'vtp', 'rtn' or 'b'
frame = 'rtn'

""" Default saving configuration --------------------------------------------"""

figure_directory    = str(Path.cwd()) + f'/Films/{vdf.info.SC}_vdf_pngs/{frame}/'
animation_directory = str(Path.cwd()) + f'/Films/Animations/'

if not Path(figure_directory).exists():
    Path(figure_directory).mkdir(parents=True, exist_ok=True)

if not Path(animation_directory).exists():
    Path(animation_directory).mkdir(parents=True, exist_ok=True)

animation_name = f'{vdf.info.SC}_{frame}_{tb.strftime("%Y%m%dT%H%M%S")}_{te.strftime("%Y%m%dT%H%M%S")}'
figure_names = []
""" ----------------------- Load timeseries -------------------------------- """

tc  = (tb + (te - tb) /2) #central time
Dt  = (te - tb).seconds / 60 # window length, in minutes

#load timeseries
var = load_timeseries(var, tc, vdf.info.SC, Dt = Dt)

# Generating no more than a 100 images for a movie
n_max_png = 100 # set to np.nan if you want to generate all images
N_pts = np.nanmin([len(var.U.date), n_max_png]) 
step = len(var.U.date) // N_pts

""" Create figures ----------------------------------------------------------"""

# loop over time steps in the timeseries
for i in tqdm(range(0, len(var.U.date), step)): 

    # get date and convert to datetime
    date = var.U.date[i]
    t = datetime.fromtimestamp(int(date) / 1e9, UTC).replace(tzinfo=None)

    
    # load vdf
    vdf = load_vdf(vdf, t, var, rotate_to = frame, cleaning=True)

    
    # Craft figure name
    t_vdf       = datetime.fromtimestamp(int(vdf.info.t) / 1e9, UTC).replace(tzinfo=None)
    figure_name = t_vdf.strftime("%Y%m%dT%H%M%S.%f") + '.png'        
    save_path   = figure_directory + figure_name
    figure_names.append(save_path)

    # plot and save
    plot_film(vdf, var, frame, save_path = save_path)
    plt.close('all')

""" Create Animation --------------------------------------------------------"""

create_animation(
    figure_names, animation_directory, animation_name,
    film_format = 'gif',
    fps = 10,
)


















    
            
             
