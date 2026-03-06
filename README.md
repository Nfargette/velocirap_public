# Velocirap
---
Velocirap is a Python library designed to facilitate the advanced analysis of 
ion velocity distribution functions (VDFs) from space missions such as 
Solar Orbiter, Parker Solar Probe, and the future HelioSwarm mission. 
It is developped at the _Institut de Recherche en Astrophysique et Planétologie_ 
(IRAP) by the Proton and Alpha Sensor (PAS) team of the Solar Orbiter mission.
It provides a wide range of tools for loading, processing, and visualizing VDF
data.

## Table of Contents
1. [Installation](#Install)
2. [Structure](#Structure)
3. [Use Cases](#Use-Cases)


---

## Installation

```
# Get repository from GitHub
git clone git@github.com:vreville/velocirap.git

# Get into the new folder
cd velocirap

# Install all required librairies
pip install -e .
```
Note that a LaTeX installation is needed on your machine. Installing [MiKteX](https://miktex.org/download) may be the easiest way to achieve that. Another option is to comment the following line in the plot.py file

```
rc('text',  usetex=True) #line to comment if not using latex
```

---

## Structure
```
velocirap/
├── Data/                   # Data folder, sorted by mission and by day
│ ├── PSP_VDF/
│ ├── SOLO_VDF/
│   ├──yyy_mm_dd/
│      ├──yyy_mm_dd_hh.cdf
├── docs/                   # Documentation
├── scripts/                # Example gallery 
├── velocirap/              # Source Code
│ ├── folder/
│   ├──function_file.py     
├── pyproject.toml          # Python dependencies
└── README.md               # Main Documentation

```

## Use Cases
A detailed documentation is available in the _docs_ folder. Here, we show the results of running the scripts visualisation.py and animation.py

### Visualising a VDF at time t
In the _visualisation.py_ script, one first needs to specify a time and a spacecraft (SOLO or PSP), through
```
vdf.info.SC = 'SOLO'
tc = datetime(2022, 3, 1, 2, 30 )
```
Then, the timeseries is loaded into the _var_ dotmap object. By default, 10 minutes around _tc_ will be loaded, but this can be adjusted through the _Dt_ parameter (in minutes). Here we choose to load 20 minutes. The _burst_ parameter allows the user to specify if one wants to load burst mode or normal mode magnetic field data. We can then plot the timeseries, with the vertical line indicating _tc_

```
var = load_timeseries(var, tc, vdf.info.SC, Dt = 20, burst = True)
plot_timeseries(var, tc)
```
<p align="center">
<img src="https://github.com/Nfargette/velocirap_public/blob/main/docs/ReadMe_images/0_timeseries.png" alt="Timeseries" width="500"/>
</p>

The VDF closest to _tc_ is then loaded into the _vdf_ dotmap object through the _load_vdf_ function. Optional parameters are the following : .
* The _rotate_to_ parameter allows the user to control which rotation they want to perform. If _rotate_to_ is 'etp', 'vtp' or None, no rotation will be performed (the code is faster). This is useful if ones only need to compute moments for instance. If _rotate_to_ is 'rtn' or 'b', then only the specified rotation is performed. Finally, if _rotate_to_ is ['rtn', 'b'] (default value), both rotations are performed. H
* The _res_ parameter controls the grid size of the interpolation, if the VDF is to be rotated into the _rtn_ or the _b_ frame. The lowest the _res_ value is, the longer it will take to interpolate the VDF. The default value is 10 km/s.  
* The _cleaning_ parameter allows the user to remove one level count dark noise from the VDF
* The _particle_ parameter sets the type of particle considered, which should be 'proton' (default) or 'alpha'. This is particularly relevant for SOLO data and allows to set the mass and charge of the particle, which matters in several calculations.

Here, we load the VDF with default values
```
vdf = load_vdf(vdf, tc, var)
```
We can then plot the vdf using the _plot_vdf_ function. The user needs to specify the frame of visualisation : '_etp_' (energy theta phi), '_vtp_' (velocity theta phi), '_rtn_' or '_b_'. In the _rtn_ plot, the black arrow shows the magnetic field orientation, and a prediction for the helium2+ (alpha) bulk speed is overlayed assuming that the drift speed between alphas and protons is null (see the _Velocirap_documentation.pdf_ in the _docs_ folder for details about the computation of this prediction).

```
plot_vdf(vdf, 'vtp')
plot_vdf(vdf, 'rtn')
plt.show()
```
VTP frame             |  RTN frame
:-------------------------:|:-------------------------:
<img src="https://github.com/Nfargette/velocirap_public/blob/main/docs/ReadMe_images/1_vdf_vtp.png" alt="vdf_etp" width="500"/>|<img src="https://github.com/Nfargette/velocirap_public/blob/main/docs/ReadMe_images/2_vdf_rtn.png" alt="vdf_rtn" width="500"/> 

The _plot_vdf_ function has an optionnal argument, _p_. The _p_ parameter should be a list including the density in cm-3, the RTN velocity in km/s, and the kT tensor in the RTN frame, in eV. The scaled multivariate normal distribution with the defined parameters will then be overlayed. Here is what it looks like for our VDF in the $b$ frame if we choose to overlay the L2 moments :

```
plot_vdf(vdf, 'b', p=[vdf.N0, vdf.U0, vdf.kT0]) 
plt.show()
```

<p align="center">
<img src="https://github.com/Nfargette/velocirap_public/blob/main/docs/ReadMe_images/3_vdf_b_l2_moments.png" alt="Timeseries" width="500"/>
</p>


### Creating an animation on a time interval
In the _animation.py_ script, the user needs to specify a spacecraft (SOLO or PSP), two times delimiting the time interval of interest, and a frame of visualisation :
```
# Spacecraft should be 'SOLO' or 'PSP'
vdf.info.SC = 'SOLO'

#time interval to plot
tb = datetime(2022, 3, 1, 2, ) #begining
te = datetime(2022, 3, 1, 3, ) #end

#frame of projection, should be 'etp', 'rtn' or 'b'
frame = 'rtn'
```
Output of the codes will be saved in the default configuration :
```
velocirap/
├── Films/               
│ ├── Animations/        # Folder where movies are saved
│   ├── sc_frame_yyymmddThhmmss_yyymmddThhmmss.mp4
│ ├── PSP_vdf_pngs/      # Folder where PSP figures are saved
│ ├── SOLO_vdf_pngs/     # Folder where SOLO figures are saved
│   ├── b                # b frame sub folder
│   ├── etp              # etp frame sub folder
│   ├── rtn              # rtn frame sub folder 
│     ├── yyymmddThhmmss.png    
└── 
```
The script will then load the _in-situ_ timeries into _var_ between _tb_ and _te_.
```
tc  = (tb + (te - tb) /2)     # central time
Dt  = (te - tb).seconds / 60  # window length, in minutes

#load timeseries
var = load_timeseries(var, tc, vdf.info.SC, Dt = Dt)
```
Then, the script will limit the number of figures generated to 100. This can be modified by the user : if the 100 factor is replaced by np.nan, and the timestep of the ion instrument is constant, then the script will generate one figure per measurement. This setup can be time consuming and result in high volume output movies. The script will always generate figures at a constant timestep for now, and therefore does not yet account for cases where the timestep of the instrument is not constant within the time interval (burst mode periods in early SOLO data).
```
# Generating no more than a 100 images for a movie
n_max_png = 100 # set to np.nan if you want to generate all images
N_pts = np.nanmin([len(var.U.date), n_max_png]) 
step = len(var.U.date) // N_pts
```
The script will loop and generate png files through the _plot_film_ function.

The final step is to create the animation file through the _create_animation_ function. The user can specify the animation output format (mp4 or gif, default to mp4), the frame per second resolution (default to 24), and the _delete_figure_files_ option that will delete the created png figures if set to _True_ (default to _False_). Let us create a gif animation :
```
create_animation(
    figure_names, animation_directory, animation_name,
    film_format = 'gif',
    fps = 10,
)
```
![Animation](https://github.com/Nfargette/velocirap_public/blob/main/docs/ReadMe_images/5_film.gif)


## Citation

If you use the velocirap module in your research to visualise VDFs, please cite using :

Fargette N., Réville V., Vergé T., Kieokaew R., Rasser C., Génot V., Lavraud B., Louarn P. and Fedorov A., _Velocirap, a python library for advanced analysis of ion velocity distribution functions_ (v1.0.0) https://doi.org/10.5281/zenodo.18888713









