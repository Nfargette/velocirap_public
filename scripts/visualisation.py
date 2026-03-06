
""" ----------------------------------------------------------------------------
This script gives an example of how to plot the VDF in different frames,
together with the surrounding timeseries
---------------------------------------------------------------------------- """

import matplotlib.pyplot as plt

from dotmap   import DotMap
from datetime import datetime

from velocirap.plot                 import plot_timeseries, plot_vdf, plot_film
from velocirap.vdf.load             import load_vdf
from velocirap.timeseries.load_ts   import load_timeseries

vdf = DotMap() #initialize DotMap object to store the VDF
var = DotMap() #initialize DotMap object to store the surrounding timeseries

""" --------------------- Select spacecraft and date ----------------------- """
# Spacecraft should be 'SOLO' or 'PSP'
vdf.info.SC = 'SOLO'

# time of interest
tc = datetime(2022, 3, 1, 2, 30 )
print(f"\n{vdf.info.SC} VDF at {tc}")

""" ----------------------- Load timeseries -------------------------------- """
var = load_timeseries(var, tc, vdf.info.SC, Dt = 20, burst = True)

""" -------------------------- Load VDF ------------------------------------ """
vdf = load_vdf(vdf, tc, var)

""" -------------------------- Plot VDF ------------------------------------ """
plt.close('all')

plot_timeseries(var, tc)

plot_vdf(vdf, 'etp') #distribution in energy theta phi
plot_vdf(vdf, 'vtp') #distribution in velocity theta phi
plot_vdf(vdf, 'rtn') #distribution in RTN
plot_vdf(vdf, 'b', p=[vdf.N0, vdf.U0, vdf.kT0])   #distribution in field-aligned frame

plot_film(vdf, var, 'rtn')
plt.show()

