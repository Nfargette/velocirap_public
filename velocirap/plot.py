
""" ----------------------------------------------------------------------------
Functions for plotting and animating VDFs and timeseries
---------------------------------------------------------------------------- """

import os
import imageio.v2           as imageio   
import numpy                as np
import matplotlib.pyplot    as plt
import matplotlib.gridspec  as gridspec
import scipy.constants      as cst
import cmcrameri.cm         as cmc

from matplotlib         import rc
from matplotlib.colors  import LogNorm
from moviepy            import ImageSequenceClip

from velocirap.helpers              import integrate, multivariate_gaussian
from velocirap.timeseries.transform import eV_to_kms

# Plot options
size=14
rc('text',  usetex=True) #line to comment if not using latex
rc('xtick', labelsize=size)
rc('ytick', labelsize=size)
rc('font',  size=size)


def vec_plot(
        ax, date, y, 
        c = ['b', 'g', 'r'], norm=True, alpha=[1, .2, .2], z = [3,2,1], 
        label=['', '', ''], **kwargs
    ) :
    
    """
    Plot 3 components of a vector y as a function of date on axes ax.

    Parameters
    ----------
    ax : axes
        The ax o subplot on which to plot
    date : array of datetime
        The time array
    y : array of shape (3, len(date))
        The vector values (3 components)
    c : list of 3 str, optional
        The colors of the components, default is ['b', 'g', 'r']
    norm : bool, optional
        If True, plot the norm of the vector, default is True
    alpha : list of 3 float, optional
        The transparency of the components, default is [1, .2, .2]
    z : list of 3 int, optional
        The zorder of the components, default is [3,2,1]
    label : list of 3 str, optional
        The labels of the components, default is ['', '', '']
    **kwargs
        Arguments to be passed to ax.plot
    """
    for Bx, cd, ad, zd, lb in zip(y, c, alpha, z, label):
        ax.plot(date, Bx, color=cd, alpha=ad, zorder=zd, label=lb, **kwargs)
    if norm:
        ax.plot(date, np.linalg.norm(y, axis=0), color='k', **kwargs)
 
        
def plot_timeseries(var, tc):

    """
    Plot the timeseries variables from var.

    Parameters
    ----------
    var : DotMap object
        The DotMap object containing the variables to plot
    tc : datetime object
        The central time of the time interval
    
    Plotting : 
    # - B (nT)
    # - Ur (km/s)
    # - Ut, Un (km/s)
    # - Np, Ne (cm-3)
    # - kT_parallel perp1, perp2 (eV)

    """
    #check temperature data is available 
    n_ax, plot_T = 5, True
    if np.isnan(var.kT_b.y).all() :
        plot_T = False
        n_ax = 4
    
    #creating figure 
    fig = plt.figure(figsize = (8, 8))
    gs = gridspec.GridSpec(n_ax,1)
    
    #creating subplots
    ax = [fig.add_subplot(gs[0])]
    [
        ax.append(
            fig.add_subplot(gs[i_ax], sharex=ax[0])
        ) for i_ax in range(1, n_ax)
    ]    
    
    #plot magnetic field
    vec_plot(
        ax[0], var.B.date, var.B.y, 
        label=["$B_R$", "$B_T$", "$B_N$"], lw=.7
    )
       
    #plot Ur
    ax[1].plot(
        var.U.date, var.U.y[0], 
        color='b', lw=.7, alpha=1, label = "$U_R$"
    ) 
    
    #plot Ut, Un
    ax[2].plot(
        var.U.date, var.U.y[1], 
        color='g', lw=.7, alpha=1, label = "$U_T$"
    ) 
    ax[2].plot(
        var.U.date, var.U.y[2], 
        color='r', lw=.7, alpha=1, label = "$U_N$"
    ) 

    #plot density
    ax[3].plot(
        var.Np.date, var.Np.y[0], 
        color='grey', lw=.7, label = "$n_p$"
    )     
    if 'ne' in var:
        ax[3].plot(
            var.Ne.date, var.Ne.y[0], 
            color='k', lw=.3, label = "$n_e$"
        )    
        
    if plot_T : 
        if var.info.Tcalc == 'rotation' : 
            vec_plot(
                ax[4], var.kT.date, var.kT_b.y[:3] , 
                label = [r"$T_{\parallel}$", r"$T_{\perp 1}$", r"$T_{\perp 2}$"],
                c = ['k', 'purple', 'magenta'], alpha=[1, .7, .7], norm=False, lw=.7
            )
        elif var.info.Tcalc == 'diagonalisation' :
            vec_plot(
                ax[4], var.kT.date, var.kT_b.y[:3] , 
                label = [r"$T_{diag, \parallel}$", r"$T_{diag, \perp 1}$", r"$T_{diag, \perp 2}$"],
                c = ['k', 'purple', 'magenta'], alpha=[1, .7, .7], norm=False, lw=.7
            )
    
    # set date as title
    date_str = str(tc).split()[0]
    ax[0].set_title(f'${date_str}$', fontsize=14 )

    #set units as y labels 
    ax[0].set_ylabel(r'$\mathrm{nT}$') 
    ax[1].set_ylabel(r'$\mathrm{km/s}$') 
    ax[2].set_ylabel(r'$\mathrm{km/s}$') 
    ax[3].set_ylabel(r'$\mathrm{cm}^{-3}$') 
    if plot_T : 
        ax[4].set_ylabel(r'$\mathrm{eV}$')  
    [ax[i].get_yaxis().set_label_coords(-.08,.6) for i in range(len(ax))]

    #legend
    [ax[i].legend(loc=1) for i in range(len(ax))]
    
    # Remove redundant x labels 
    [ax[i].tick_params(axis='x', labelbottom=False) for i in range(len(ax)-1)]     
    
    #format time labels
    #ax[-1].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%d %H:%M'))      
    ax[-1].tick_params(axis='x', labelsize = 10, rotation=25)             

    # add 0 horizontal line and verticl line at tc     
    [ax[i].axhline(0, c='grey', lw=.5) for i in [0, 2]]
    [ax[i].axvline(tc, c='k', lw=.5) for i in range(len(ax))]

    #layout
    ax[0].set_xlim(var.B.date[0], var.B.date[-1],)    
    fig.tight_layout()
    plt.subplots_adjust(hspace=.1)
    
    # plt.show()
       


def plot_vdf(vdf, frame, p = None, cmap = None):    
    """
    Plot 1D and 2D distributions of the given VDF.

    Parameters
    ----------
    vdf : DotMap object
        The DotMap object containing the VDF data
    frame : str
        Frame of visualisation. Should be 'etp', 'vtp', 'rtn' or 'b'.
    p : list, optional
        The Maxwellian parameters :
        - density((1,), cm-3)
        - velocity ((3,), km/s)
        - temperature tensor ((3, 3), eV)
        The default is None.
    cmap : matplotlib colormap, optional
        The colormap to use for the plots. The default is None = cmc.lipari

    
    Returns
    -------
    None

    Notes
    -----
    This function will create a figure with 3x3 subplots. The diagonal subplots
    show the one dimensional distributions of the VDF in the three
    components of the given frame. The off-diagonal subplots show the two
    dimensional distributions of the VDF in the three components of the
    given frame.
    """
    if f'n_{frame}' not in vdf.keys() : 
        print(f"\nThe VDF object does not include the data in the {frame} frame.")
        print(f"Unable to plot the VDF in the {frame} frame.")
        return

    # apply the selected frame
    # ETP
    if frame == 'etp' :
        # distribution function
        f = vdf.n_etp.copy()

        # Bin edges of E, theta, phi
        bin_edges = [ 
            vdf.Axis.bins_etp[0] , #eV
            vdf.Axis.bins_etp[1] * 180 / np.pi, #deg
            vdf.Axis.bins_etp[2] * 180 / np.pi  #deg   
        ]

        # E, theta, phi axis
        bin_centers = [
            vdf.Axis.E,
            vdf.Axis.theta * 180 / np.pi,
            vdf.Axis.phi * 180 / np.pi    
        ]

        #spherical metric        
        metric = np.array([
            cst.m_p * vdf.Grid_vtp.v * 1e3 * vdf.Grid_vtp.dv * 1e3 / cst.eV,
            cst.m_p / 2 * (vdf.Grid_vtp.v * 1e3) ** 2 / cst.eV * vdf.Grid_vtp.dtheta, 
            cst.m_p / 2 * (vdf.Grid_vtp.v * 1e3) ** 2 / cst.eV  * np.cos(vdf.Grid_vtp.theta) * vdf.Grid_vtp.dphi
        ])  
            
        #Labels
        labels = [
            r'$\mathrm{E~(eV)}$', 
            r'$\theta~\mathrm{(deg)}$', 
            r'$\phi~\mathrm{(deg)}$'
        ]
        
    elif frame == 'vtp' :
        # distribution function
        f = vdf.n_vtp.copy()

        # Bin edges of V, theta, phi
        bin_edges = [ 
            np.sqrt(2 * vdf.Axis.bins_etp[0] * cst.eV / (vdf.info.p_mass *cst.m_p)) * 1e-3, #km/s
            vdf.Axis.bins_etp[1] * 180 / np.pi, #deg
            vdf.Axis.bins_etp[2] * 180 / np.pi  #deg   
        ]

        # E, theta, phi axis
        bin_centers = [
            vdf.Axis.v,
            vdf.Axis.theta * 180 / np.pi,
            vdf.Axis.phi * 180 / np.pi    
        ]

        #spherical metric
        metric = vdf.Grid_vtp.metric
            
        #Labels
        labels = [
            r'$\mathrm{V~(km/s)}$', 
            r'$\theta~\mathrm{(deg)}$', 
            r'$\phi~\mathrm{(deg)}$'
        ]


    #RTN or b
    else:
        res = vdf.info.res

        if frame == 'rtn' :
            f = vdf.n_rtn.copy()
            axes = ['u_r', 'u_t', 'u_n']   
            labels = [
                r'$\mathrm{U_R~(km/s)}$', 
                r'$\mathrm{U_T~(km/s)}$', 
                r'$\mathrm{U_N~(km/s)}$'
            ]
            mat_rtn_to_frame = np.eye(3)
            U = vdf.U0_rtn
    
        elif frame == 'b' :
            f = vdf.n_b.copy()
            axes = ['u_b', 'u_exb', 'u_e']
            labels = [
                r'$\mathrm{U_{\parallel}~(km/s)}$', 
                r'$\mathrm{U_{\perp 1}~(km/s)}$', 
                r'$\mathrm{U_{\perp 2}~(km/s)}$'
            ]          
            mat_rtn_to_frame = vdf.Mat.rtn_to_b
            U = vdf.U0_b   
        else:
            raise RuntimeError(
                "The frame of visualisation, should be 'etp', 'rtn' or 'b'"
            )   
            
        #delete bins with too few data points, i.e less than max(f) / 1000  
        f[f < np.nanmax(f) * 1e-3] = 0.    
        
        # Bin edges        
        bin_edges = [
            np.append(
                vdf.Axis[axes[i]] - res/2, vdf.Axis[axes[i]][-1] + res/2
            ) for i in range(3)
        ]

        # Bin center
        bin_centers = [vdf.Axis[axes[i]] for i in range(3)]
        
        metric = np.ones((3,) +  f.shape) * res * 1e3             
    
    if p is not None:
        N0, U0, kT0 = p #rtn
        # L2 bulk flow
        U0 = mat_rtn_to_frame @ U0
        # kT tensor
        kT0 = mat_rtn_to_frame @ kT0 @ np.linalg.inv(mat_rtn_to_frame)
        g = multivariate_gaussian(
            vdf[f'u_{frame}']* 1e3, 
            U0 * 1e3, 
            kT0 * cst.eV / (vdf.info.p_mass * cst.m_p)
        ) * N0 * 1e6               
        g_2d = [integrate(1, g, metric, axis=(i,)) for i in range(3)][::-1] 
        g_1d = [
            integrate(
                1, g, metric, axis=tuple({0, 1, 2} - {i})
            ) for i in range(3)
        ] 

    # Sums along 1 dimension (f_12, f_13, f_23)
    dist2D =  [integrate(1, f, metric, axis=(i,)) for i in range(3)][::-1] 
        
    # Compute the maximum of dist2D
    dist2D_max = np.nanmax([dist2D[i].max() for i in range(3)])   
    dist2D_min = np.nanmin([dist2D[i][dist2D[i]>0].min() for i in range(3)])     

    # Sums along 2 dimensions (f_1, f_2, f_3)   
    dist1d =  [
        integrate(
            1, f, metric, axis=tuple({0, 1, 2} - {i})
        ) for i in range(3)
    ]
        
    
    """ ---------------FIGURE -------------------- """
    N = 3 #3 variables
    
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(N, N)
    if cmap is None:
        cmap = cmc.lipari
    # cmap = plt.get_cmap('jet')
    
    #initialise subplots
    ax_hist = []
    ax_dist = [[] for i in range(N)]
    
    #diagonal 1d histograms
    for i in range(N):        
        ax_hist.append(fig.add_subplot(gs[i,i]))
        ax_hist[i].hist(
            bin_centers[i], bins=bin_edges[i], weights=dist1d[i], 
            alpha=.7, color="#cfd8dc", density=False
        )
        ax_hist[i].hist(
            bin_centers[i], bins=bin_edges[i], weights=dist1d[i],  
            alpha=.7, histtype='step', edgecolor='k', 
        )        
        if frame not in ['etp', 'vtp']:
            # ax_hist[i].axvline(U[i], color='k', lw=.5) 
            if p is not None:
                ax_hist[i].axvline(U0[i], color='b', lw=.5, ls='--') 
                ax_hist[i].plot(bin_centers[i], g_1d[i], color='b', lw=1, alpha=.8)

    #2D plots
    for j in range (N): # columns j       
        for i in range(j+1, N): #rows i
            if j==0:
                ax_dist[j].append(fig.add_subplot(gs[i,j], sharex = ax_hist[j]))                
            else:
                ax_dist[j].append(fig.add_subplot(
                    gs[i,j], sharex = ax_hist[j], sharey = ax_dist[0][i-1])
                    )
            # ax_dist[j][i-1-j].set_facecolor(cmap.colors[0])

            norm=LogNorm(vmax=dist2D_max, vmin = dist2D_min, clip=True)
            im = ax_dist[j][i-1-j].pcolormesh(
                bin_edges[j], bin_edges[i], dist2D[j + i - 1].T, 
                cmap=cmap, shading='flat', norm = norm
            )

            if frame not in ['etp', 'vtp']:                
                #force equal axis in rtn or b frame 
                # (sharex and sharey not working when zooming in this case)
                ax_dist[j][i-1-j].set_aspect('equal', adjustable='box') 
                
                #log contours
                levels = np.logspace(np.log10(dist2D_max) - 3, np.log10(dist2D_max), 4)
                ax_dist[j][i-1-j].contour(
                    vdf.Axis[axes[j]], vdf.Axis[axes[i]], dist2D[j + i - 1].T, colors='w', linewidths=.5, 
                    levels = levels
                ) 


                if p is not None:
                    ax_dist[j][i-1-j].scatter(
                        U0[j], U0[i], 
                        color='b', ec='k', 
                        label = r'$\mathrm{Maxwellian}$' ,                        
                        alpha = .5,
                        zorder=10
                    )
                    
                    ax_dist[j][i-1-j].contour(
                        vdf.Axis[axes[j]], vdf.Axis[axes[i]], g_2d[j + i - 1].T, 
                        colors='b', linewidths=1., #linestyles='dashed',
                        levels = levels,
                    ) 
                else:                    
                    #scatter the L2 bulk flow
                    ax_dist[j][i-1-j].scatter(
                        U[j], U[i], 
                        color='white', ec='k', label = r'$\mathrm{L2~bulk~flow}$',
                        alpha = 1,
                        zorder=10
                    )

                b = vdf.B0_rtn / np.linalg.norm(vdf.B0_rtn)
                # Va = np.linalg.norm(vdf.B0_rtn) * 1e-9 / np.sqrt(cst.mu_0 * cst.m_p * vdf.N0 * 1e6) * 1e-3
                    
                #scatter alpha prediction
                # if vdf.info.SC == 'SOLO':
                #     ax_dist[j][i-1-j].scatter(
                #         np.sqrt(2) * (U[j] + 0 * (mat_rtn_to_frame @ b)[j]), 
                #         np.sqrt(2) * (U[i] + 0 * (mat_rtn_to_frame @ b)[i]), 
                #         color='r', ec='k', label = r'$\alpha \mathrm{~prediction~(no~drift)}$',
                #         alpha = .5, zorder=10
                #     )      
                                   

                if frame == 'rtn':
                    if p is not None:
                        ax_dist[j][i-1-j].quiver(
                        *(U0[j], U0[i]), b[j], b[i], scale=4,
                        zorder=5, color = 'k'
                        )
                    else:
                        #add the magnetic field direction                    
                        ax_dist[j][i-1-j].quiver(
                            *(U[j], U[i]), b[j], b[i], scale=4,
                            zorder=5, color = 'k'
                        )                          
            
                #plot horizontal and vertical lines at 0
                ax_dist[j][i-1-j].axhline(0, color='white', lw=.5) 
                if j==1:
                    ax_dist[j][i-1-j].axvline(0, color='white', lw=.5)
            
            #remove redundant axis tick labels
            if i!=N-1:
                ax_dist[j][i-1-j].tick_params(axis='x', labelbottom=False)
            if j!=0:
                ax_dist[j][i-1-j].tick_params(axis='y', labelbottom=False)
    
    #set axis labels            
    for j in range(N-1):
        ax_dist[j][N-1-1-j].set_xlabel(labels[j]) 
        ax_dist[0][j].set_ylabel(labels[j+1])
        ax_dist[0][j].get_yaxis().set_label_coords(-.3,.5) 
    [ax_hist[j].yaxis.tick_right() for j in range(N)]
    hist_units = r'$\mathrm{1D~VDF~(s/m^4)}$' if frame !='etp' else r'$\mathrm{1D~VDF~(eV^{-1}m^{-3})}$'
    ax_hist[0].set_ylabel(hist_units)
    ax_hist[0].get_yaxis().set_label_coords(-.3,.5)  
    
    #put log scale on energy j if frame = etp or vtp
    if frame in ['etp', 'vtp']:
        ax_hist[0].set_xscale("log")
        ax_hist[0].set_yscale("log")
        if frame == 'etp':
            [ax_hist[i].ticklabel_format(axis='y', style='sci', scilimits=(5,5)) for i in [1, 2]]

    #for rtn and b frame, set plot limits
    else:            
        #find range of variation in each directions :

        #first non zero value of 1d distribution
        i_first = [np.argmax(dist1d[i]>0) for i in range(3)]

        #last non zero value of 1d distribution
        i_last  = [
            len(dist1d[i]) - np.argmax(dist1d[i][::-1]>0) - 1 for i in range(3)
        ] 

        #variation in each direction   
        dV = [
            vdf.Axis[axes[i]][i_last[i]] - vdf.Axis[axes[i]][i_first[i]] 
            for i in range(3)
        ]   
        
        #direction of maximum variation
        i_m = np.argmax(dV)  
        
        #Central V value in each direction
        Vc = [vdf.Axis[axes[i]][i_first[i]] + dV[i]/2 for i in range(3)]
        
        #set ylims 
        [
            ax_dist[0][i].set_ylim(
                Vc[i+1] - dV[i_m]/2 - 3 * res , 
                Vc[i+1] + dV[i_m]/2 + 3 * res
            ) for i in range(N-1)
        ]

        #set xlims
        [
            ax_dist[j][0].set_xlim(
                Vc[j] - dV[i_m]/2 - 3 * res , 
                Vc[j] + dV[i_m]/2 + 3 * res
            ) for j in range(N-1)
        ]    
        ax_hist[-1].set_xlim(
            Vc[-1] - dV[i_m]/2 - 3 * res , 
            Vc[-1] + dV[i_m]/2 + 3 * res
        )
        
    
    #label on last histogram
    ax_hist[N-1].set_xlabel(labels[-1])
    
    plt.suptitle(f'${np.datetime64(vdf.info.t)}$')
    plt.tight_layout()
    
    cbar_ax = fig.add_axes([0.75, 0.39, 0.015, 0.23])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar_units = r'$\mathrm{2D~VDF~(s^2/m^5)}$' if frame !='etp' else r'$\mathrm{2D~VDF~(eV^{-2} m^{-3})}$'
    [cbar.ax.axhline(levels[i], c='w', lw=.5) for i in range(len(levels))] if frame not in ['etp', 'vtp'] else None 
    cbar.ax.set_title(cbar_units, fontsize=10)

    if frame not in ['etp', 'vtp'] :
        ax_dist[0][1].legend(
            loc='center left', bbox_to_anchor=(1.4, 3.5)
        )
    # return fig

def plot_film(vdf, var, frame, save_path=None, cmap = None):
    """
    Plot the timeseries and 2D distribution function in the given frame at a given time

    Parameters
    ----------
    vdf : DotMap object
        The DotMap object containing the VDF data
    var : DotMap object
        The DotMap object containing the timeseries to plot.
    frame : str
        Frame of visualisation. Should be 'etp', 'vtp', 'rtn' or 'b'.
    save_path : str, optional
        The path to save the figure. If None, the figure is not saved.
    cmap : str, optional
        The colormap to use. If None, the default colormap is used (cmc.lipari)

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.

    Notes
    -----
    The function first plots the timeseries of the magnetic field,  the bulk flow velocity, the density and temperatures
    Then it plots the 2D distribution function in the given frame.

    If the frame is 'etp', the function plots the energy distribution function in the Energy theta phi frame.
    If the frame is 'vtp', the function plots the VDF in the velocity theta phi frame.
    If the frame is 'rtn', the function plots the VDF in the RTN frame.
    If the frame is 'b', the function plots the VDF in the magnetic field-aligned frame.

    """
    if f'n_{frame}' not in vdf.keys() : 
        print(f"\nThe VDF object does not include the data in the {frame} frame.")
        print(f"Unable to plot the VDF in the {frame} frame.")
        return
    
    n_ax, plot_T = 5, True
    if np.isnan(var.kT_b.y).all() :
        plot_T = False
        n_ax = 4
    fig = plt.figure(figsize=(14, 7))
    gs = gridspec.GridSpec(n_ax, n_ax*2+1)
    if cmap is None:
        cmap = cmc.lipari

    
    """------------------- plot timeseries ----------------------------------"""
    #creating subplots
    ax = [fig.add_subplot(gs[0, :n_ax])]
    [
        ax.append(
            fig.add_subplot(gs[i_ax, :n_ax], sharex=ax[0])
        ) for i_ax in range(1, n_ax)
    ]    

    #plot magnetic field
    vec_plot(
        ax[0], var.B.date, var.B.y, 
        label=["$B_R$", "$B_T$", "$B_N$"], lw=.7
    )
      
    #plot Ur
    ax[1].plot(
        var.U.date, var.U.y[0], 
        color='b', lw=.7, alpha=1, label = "$U_R$"
    ) 
    
    #plot Ut, Un
    ax[2].plot(
        var.U.date, var.U.y[1], 
        color='g', lw=.7, alpha=1, label = "$U_T$"
    ) 
    ax[2].plot(
        var.U.date, var.U.y[2], 
        color='r', lw=.7, alpha=1, label = "$U_N$"
    ) 
    
    #plot density
    ax[3].plot(
        var.Np.date, var.Np.y[0], 
        color='grey', lw=.7, label = "$n_p$"
    )     
    if 'ne' in var:
        ax[3].plot(
            var.Ne.date, var.Ne.y[0], 
            color='k', lw=.3, label = "$n_e$"
        )    
        
    #plot parallel and perp temperature
    if plot_T : 
        if var.info.Tcalc == 'rotation' : 
            vec_plot(
                ax[4], var.kT.date, var.kT_b.y[:3] , 
                label = [r"$T_{\parallel}$", r"$T_{\perp 1}$", r"$T_{\perp 2}$"],
                c = ['k', 'purple', 'magenta'], alpha=[1, .7, .7], norm=False, lw=.7
            )
        elif var.info.Tcalc == 'diagonalisation' :
            vec_plot(
                ax[4], var.kT.date, var.kT_b.y[:3] , 
                label = [r"$T_{diag, \parallel}$", r"$T_{diag, \perp 1}$", r"$T_{diag, \perp 2}$"],
                c = ['k', 'purple', 'magenta'], alpha=[1, .7, .7], norm=False, lw=.7
            )        
    
    #set units as y labels 
    ax[0].set_ylabel(r'$\mathrm{nT}$') 
    ax[1].set_ylabel(r'$\mathrm{km/s}$') 
    ax[2].set_ylabel(r'$\mathrm{km/s}$') 
    ax[3].set_ylabel(r'$\mathrm{cm}^{-3}$') 
    ax[4].set_ylabel(r'$\mathrm{eV}$')  
    [ax[i].get_yaxis().set_label_coords(-.08,.6) for i in range(len(ax))]

    #legend
    # [ax[i].legend(loc=2) for i in range(len(ax))]
    [ax[i].legend(
        prop={'size': 10},  loc='center left', 
        bbox_to_anchor=(1,0.5)
    ) for i in range(len(ax))] 
    
    # Remove redundant x labels 
    [ax[i].tick_params(axis='x', labelbottom=False) for i in range(len(ax)-1)]     
    
    #format time labels
    ax[-1].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d %H:%M'))      
    ax[-1].tick_params(axis='x', labelsize = 10)             

    # add 0 horizontal line and verticl line at tc     
    [ax[i].axhline(0, c='grey', lw=.5) for i in [2]]
    [ax[i].axvline(vdf.info.t, c='k', lw=1) for i in range(len(ax))]

    # log scale
    # [ax[i].set_yscale('log') for i in []]

    #layout
    ax[0].set_xlim(var.U.date[0], var.U.date[-1],)  

    """-------------------------- plot VDF ----------------------------------"""
    
    if frame == 'etp' :
        # distribution function
        f = vdf.n_etp.copy()

        # Bin edges of E, theta, phi
        bin_edges = [ 
            vdf.Axis.bins_etp[0] , #eV
            vdf.Axis.bins_etp[1] * 180 / np.pi, #deg
            vdf.Axis.bins_etp[2] * 180 / np.pi  #deg   
        ]

        # E, theta, phi axis
        bin_centers = [
            vdf.Axis.E,
            vdf.Axis.theta * 180 / np.pi,
            vdf.Axis.phi * 180 / np.pi    
        ]

        #spherical metric        
        metric = np.array([
            cst.m_p * vdf.Grid_vtp.v * 1e3 * vdf.Grid_vtp.dv * 1e3 / cst.eV,
            cst.m_p / 2 * (vdf.Grid_vtp.v * 1e3) ** 2 / cst.eV * vdf.Grid_vtp.dtheta, 
            cst.m_p / 2 * (vdf.Grid_vtp.v * 1e3) ** 2 / cst.eV  * np.cos(vdf.Grid_vtp.theta) * vdf.Grid_vtp.dphi
        ])  
            
        #Labels
        labels = [
            r'$\mathrm{E~(eV)}$', 
            r'$\theta~\mathrm{(deg)}$', 
            r'$\phi~\mathrm{(deg)}$'
        ]
        

    elif frame == 'vtp' :
        # distribution function
        f = vdf.n_vtp.copy()

        # Bin edges of V, theta, phi
        bin_edges = [ 
            np.sqrt(2 * vdf.Axis.bins_etp[0] * cst.eV / (vdf.info.p_mass *cst.m_p)) * 1e-3, #km/s
            vdf.Axis.bins_etp[1] * 180 / np.pi, #deg
            vdf.Axis.bins_etp[2] * 180 / np.pi  #deg   
        ]

        # E, theta, phi axis
        bin_centers = [
            vdf.Axis.v,
            vdf.Axis.theta * 180 / np.pi,
            vdf.Axis.phi * 180 / np.pi    
        ]

        #spherical metric
        metric = vdf.Grid_vtp.metric
            
        #Labels
        labels = [
            r'$\mathrm{V~(km/s)}$', 
            r'$\theta~\mathrm{(deg)}$', 
            r'$\phi~\mathrm{(deg)}$'
        ]

    #RTN or b
    else:
        res = vdf.info.res

        if frame == 'rtn' :
            f = vdf.n_rtn.copy()
            axes = ['u_r', 'u_t', 'u_n']   
            labels = [
                r'$\mathrm{U_R~(km/s)}$', 
                r'$\mathrm{U_T~(km/s)}$', 
                r'$\mathrm{U_N~(km/s)}$'
            ]
            mat_rtn_to_frame = np.eye(3)
            U = vdf.U0_rtn
            
        elif frame == 'b' :
            f = vdf.n_b.copy()
            axes = ['u_b', 'u_exb', 'u_e']
            labels = [
                r'$\mathrm{U_{\parallel}~(km/s)}$', 
                r'$\mathrm{U_{\perp 1}~(km/s)}$', 
                r'$\mathrm{U_{\perp 2}~(km/s)}$'
            ]          
            mat_rtn_to_frame = vdf.Mat.rtn_to_b
            U = vdf.U0_b
        else:
            raise RuntimeError(
                "The frame of visualisation, should be 'etp', 'vtp', 'rtn' or 'b'"
            )   
            
        #delete bins with too few data points, i.e less than max(f) / 1000  
        f[f < np.nanmax(f) * 1e-3] = 0.    
        
        # Bin edges        
        bin_edges = [
            np.append(
                vdf.Axis[axes[i]] - res/2, vdf.Axis[axes[i]][-1] + res/2
            ) for i in range(3)
        ]

        # Bin center
        bin_centers = [vdf.Axis[axes[i]] for i in range(3)]
        
        #metric
        metric = np.ones((3,) +  f.shape) * res * 1e3   
        
    # Sums along 1 dimension (f_12, f_13, f_23)
    dist2D =  [integrate(1, f, metric, axis=(i,)) for i in range(3)][::-1] 
        
    # Compute the maximum of dist2D
    dist2D_max = np.nanmax([dist2D[i].max() for i in range(3)])   
    dist2D_min = np.nanmin([dist2D[i][dist2D[i]>0].min() for i in range(3)])     

    # Sums along 2 dimensions (f_1, f_2, f_3)   
    dist1d =  [
        integrate(
            1, f, metric, axis=tuple({0, 1, 2} - {i})
        ) for i in range(3)
    ]
        
    
    #initialise subplots
    ax_hist = []
    ax_dist = [[] for x in range(3)]
    
    #diagonal 1d histograms
    
    ax_hist.append(fig.add_subplot(gs[0, n_ax+1:n_ax+3]))
    ax_hist.append(fig.add_subplot(gs[2, n_ax+3:n_ax+5]))
    ax_hist.append(fig.add_subplot(gs[3:5, n_ax+5]))

    #diagonal 1d histograms
    for i in range(3):  
        if i==2:
            orientation = 'horizontal'  
        else:
            orientation = 'vertical'
        ax_hist[i].hist(
            bin_centers[i], bins=bin_edges[i], weights=dist1d[i], 
            alpha=.7, color="#cfd8dc", orientation = orientation
        )
        ax_hist[i].hist(
            bin_centers[i], bins=bin_edges[i], weights=dist1d[i],  
            alpha=.7, histtype='step', edgecolor='k', orientation = orientation
        )
        if frame not in ['etp', 'vtp']:
            ax_hist[i].axvline(U[i], color='k', lw=.5) if i<2 else ax_hist[i].axhline(U[i], color='k', lw=.5)
            

    #2D plots
    for j in range (3): # columns j       
        for i in range(j+1, 3): #rows i
            if j==0:
                ax_dist[j].append(fig.add_subplot(gs[2*i-1:2*i+1, n_ax+2*j+1:n_ax+2*j+3], sharex = ax_hist[j]))                
            else:
                ax_dist[j].append(fig.add_subplot(
                    gs[2*i-1:2*i+1, n_ax+2*j+1:n_ax+2*j+3], sharex = ax_hist[j], sharey = ax_hist[-1])
                    )
            
            if not isinstance(cmap, str):
                ax_dist[j][i-1-j].set_facecolor(cmap.colors[0])
            

            norm=LogNorm(vmax=dist2D_max, vmin = dist2D_min, clip=True)
            im = ax_dist[j][i-1-j].pcolormesh(
                bin_edges[j], bin_edges[i], dist2D[j + i - 1].T, 
                cmap=cmap, shading='flat', norm = norm
            )
            
            if frame not in ['etp', 'vtp']:                
                #force equal axis in rtn or b frame 
                # (sharex and sharey not working when zooming in this case)
                ax_dist[j][i-1-j].set_aspect('equal', adjustable='box')  

                #log contours
                levels = np.logspace(np.log10(dist2D_max) - 3, np.log10(dist2D_max), 4)
                ax_dist[j][i-1-j].contour(
                    vdf.Axis[axes[j]], vdf.Axis[axes[i]], dist2D[j + i - 1].T, colors='w', linewidths=.5, 
                    levels = levels
                ) 

                #scatter the 1st order moment
                ax_dist[j][i-1-j].scatter(
                    U[j], U[i], 
                    color='white', ec='k', label = r'$\mathrm{L2~bulk~flow}$',
                    alpha = 1,
                    zorder=10
                )
                
                b = vdf.B0_rtn / np.linalg.norm(vdf.B0_rtn)
                # Va = np.linalg.norm(vdf.B0_rtn) * 1e-9 / np.sqrt(cst.mu_0 * cst.m_p * vdf.N0 * 1e6) * 1e-3
                    
                # #scatter alpha prediction
                if vdf.info.SC == 'SOLO':
                    ax_dist[j][i-1-j].scatter(
                        np.sqrt(2) * (U[j] + 0 * (mat_rtn_to_frame @ b)[j]), 
                        np.sqrt(2) * (U[i] + 0 * (mat_rtn_to_frame @ b)[i]), 
                        color='r', ec='k', label = r'$\alpha \mathrm{~prediction~(no~drift)}$',
                        alpha = .5, zorder=10
                    )  

                if frame == 'rtn':
                    #add the magnetic field direction                    
                    ax_dist[j][i-1-j].quiver(
                        *(U[j], U[i]), b[j], b[i], scale=4,
                        zorder=5, color = 'k'
                    )                    
            
                #plot horizontal and vertical lines at 0
                ax_dist[j][i-1-j].axhline(0, color='white', lw=.5) 
                if j==1:
                    ax_dist[j][i-1-j].axvline(0, color='white', lw=.5)
            
            #remove redundant axis tick labels
            if i!=2:
                ax_dist[j][i-1-j].tick_params(axis='x', labelbottom=False)
            if j!=0:
                ax_dist[j][i-1-j].tick_params(axis='y', labelleft=False)
                
    
    #set axis labels            
    for j in range(2):
        ax_dist[j][3-1-1-j].set_xlabel(labels[j]) 
        ax_dist[0][j].set_ylabel(labels[j+1])
        ax_dist[0][j].get_yaxis().set_label_coords(-.2,.6)   
        ax_hist[j].xaxis.tick_top()
    [ax_hist[j].yaxis.tick_right() for j in range(3)]
    ax_hist[2].tick_params(axis='x', labelbottom=False)
    hist_units = r'$\mathrm{1D~VDF~(s/m^4)}$' if frame !='etp' else r'$\mathrm{1D~VDF~(eV^{-1}m^{-3})}$'
    ax_hist[0].set_ylabel(hist_units)
    ax_hist[0].get_yaxis().set_label_coords(-.2,.5)  
    
    #put log scale on energy j if frame = etp
    if frame in ['etp', 'vtp']:
        ax_hist[0].set_xscale("log")
        ax_hist[0].set_yscale("log")
        
    #for rtn and b frame, set plot limits
    else:            
        i_Vr_min = np.argmin(var.U.y[0])
        i_Vr_max = np.argmax(var.U.y[0])

        Vr_min = var.U.y[0, i_Vr_min]
        Vr_max = var.U.y[0, i_Vr_max]
        
        Vth_min = eV_to_kms(np.max(var.kT.y[:3, i_Vr_min]))
        Vth_max = eV_to_kms(np.max(var.kT.y[:3, i_Vr_max]))

        #account for alphas being includede in SOLO vdf
        scale_max = 6 if vdf.info.SC == 'PSP' else 12
        scale_min = 6

        V_max = np.min([
            Vr_max + scale_max * Vth_max, 
            np.sqrt(2 * vdf.Axis.E[-1] * cst.eV / cst.m_p) * 1e-3 
        ])
        V_min = np.max([
            Vr_min - scale_min * Vth_min, 
            0 
        ])

        #Central V value in each direction
        Vc = mat_rtn_to_frame @ [(V_max + V_min)/2, 0, 0]
        dV = (V_max - V_min)
        
        #set ylims 
        [
            ax_dist[0][i].set_ylim(
                Vc[i+1] - dV/2  , 
                Vc[i+1] + dV/2 
            ) for i in range(2)
        ]

        #set xlims
        [
            ax_dist[j][0].set_xlim(
                Vc[j] - dV/2 , 
                Vc[j] + dV/2 
            ) for j in range(2)
        ]    
        ax_hist[-1].set_ylim(
            Vc[-1] - dV/2  , 
            Vc[-1] + dV/2 
        )
        
        
    plt.suptitle(f'${np.datetime64(vdf.info.t)}$')
    plt.tight_layout()
    
    cbar_ax = fig.add_axes([0.92, 0.69, 0.015, 0.18])
    cbar = fig.colorbar(im, cax=cbar_ax)
    [cbar.ax.axhline(levels[i], c='w', lw=.5) for i in range(len(levels))] if frame not in ['etp', 'vtp'] else None 
    cbar_units = r'$\mathrm{2D~VDF~(s^2/m^5)}$' if frame !='etp' else r'$\mathrm{2D~VDF~(eV^{-2} m^{-3})}$'
    cbar.ax.set_title(cbar_units, fontsize=10)

    if frame not in ['etp', 'vtp'] :
        ax_dist[0][1].legend(
            loc='center left', bbox_to_anchor=(1.3, 2.5), fontsize = 12
        )
  
    # fig.tight_layout()
    plt.subplots_adjust(hspace=.2)
    plt.subplots_adjust(wspace=.5)



    if save_path is not None :          
        plt.savefig(save_path)


def create_animation(        
        figure_names: list, 
        animation_directory: str, 
        animation_name: str,
        film_format: str = 'mp4', 
        fps: int = 24, 
        delete_figure_files: bool = False
    ):  
    
    """
    Create an animation (mp4 or gif) from an image list.

    Parameters
    ----------
    figure_names : list
        List of figure file names to animate.
    animation_directory : str
        Path to save the animation.
    animation_name : str
        Name of the animation
    film_format : str
        Type of movie to create (mp4 or gif), by default 'mp4'.
    fps : int, optional
        Frames per second, by default 24.
    delete_figure_files : bool, optional
        Delete the figure files after creating the animation, by default False.

    Raises
    -------
    ValueError
        If the list of file names is empty or if the file type is not mp4 or gif.

    Returns
    -------
    None
    """
        
    # --- Check if list is empty
    if not figure_names:
        raise ValueError(f"List: '{figure_names}' is empty, no figures to animate.")
        
    # --- Creating movie
    if film_format == 'mp4':
        clip = ImageSequenceClip(figure_names, fps=fps)
        clip.write_videofile(
            os.path.join(animation_directory, 
            f"{animation_name}.mp4"), 
            codec="libx264"
        )
        
    elif film_format == 'gif':
        with imageio.get_writer(
            os.path.join(animation_directory, 
            f"{animation_name}.gif"),            
            mode='I', fps=fps
        ) as writer:
            for name in figure_names:
                image = imageio.imread(name)
                writer.append_data(image)
                
    else:
        raise ValueError(
            f"Invalid file type: '{film_format}'."
            "Accepted types are 'mp4' and 'gif'."
        )
                
    # --- Deleting figure files
    if delete_figure_files:
        for name in figure_names:
            os.remove(name)
        print("Deletion completed.")