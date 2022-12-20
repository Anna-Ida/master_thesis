import matplotlib as mpl
from matplotlib import gridspec,cm, rcParams, pyplot as plt
import numpy as np
from netCDF4 import Dataset, num2date
from cartopy import config, crs as ccrs
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def plot_mean_plus_diff(mean_data,diff_data,obs_mean,lons,lats,plotting_mode,regdata_type,standard_title,diff_text=''):
    #plt.style.use('ggplot')
    plt.style.use('default')
    cmap = mpl.cm.coolwarm 
    col_mean = mean_data #np.mean(data, axis=0)

    #print(plotting_mode)
    #plotting_modes=['mean+diff','mean+era','diff+era']
    #plotting_mode = plotting_modes[2]


    if (regdata_type=='Temperature'):
        colorlevels = np.linspace(-50,50,11)
        contourlevels = np.linspace(-10,10,11)
        cbarlab = 'Temperature [°C]'
    elif (regdata_type=='Precipitation'):
        cmap = mpl.cm.viridis #- mpl.cm.coolwarm 
        colorlevels = np.linspace(0,9,10) 
        contourlevels = np.linspace(-3,3,11)
        cbarlab = 'Precipitation [mm/day]'
    elif (regdata_type=='Geopotential height'):
        colorlevels = np.linspace(5000,5900,10)
        contourlevels = np.linspace(-100,100,11)
        cbarlab = 'geopotential height at 500 hPa [m]'

    fig_mean, ax = plt.subplots(figsize=(7,5), subplot_kw={'projection': ccrs.EquidistantConic(central_longitude=-20.0)})
    if (plotting_mode=='mean+diff'):
        CF = ax.contourf(lons, lats, col_mean, levels=colorlevels, cmap=cmap, transform=ccrs.PlateCarree())
        CS = ax.contour(lons, lats, diff_data, levels=contourlevels, colors='white', linewidths=0.5, transform=ccrs.PlateCarree())
    elif (plotting_mode=='mean+era'):
        CF = ax.contourf(lons, lats, col_mean, levels=colorlevels, cmap=cmap, transform=ccrs.PlateCarree())
        CS = ax.contour(lons, lats, obs_mean, levels=colorlevels, colors='white', linewidths=0.5, transform=ccrs.PlateCarree())
    elif (plotting_mode=='diff+era'):
        CF = ax.contourf(lons, lats, obs_mean-col_mean, levels=contourlevels, cmap=cmap, transform=ccrs.PlateCarree(),extend='both')
        CS = ax.contour(lons, lats, obs_mean, levels=colorlevels, colors='white', linewidths=0.5, transform=ccrs.PlateCarree())

            
    ax.coastlines()
    cbar = cbar = fig_mean.colorbar(CF)
    cbar.set_label(f'geopotential height at 500 hPa [m] ') 
    ax.clabel(CS, CS.levels, inline=True, fontsize=10) 
    
    if (plotting_mode=='diff+era'):
        fig_mean.suptitle(f'Difference to ERA5 mean {regdata_type} field', fontsize='x-large')
    else:
        fig_mean.suptitle(f'Mean {regdata_type} field', fontsize='x-large')
    plt.text(.5, 0.9, standard_title, transform=fig_mean.transFigure, fontsize='large', horizontalalignment='center')
    if (plotting_mode=='mean+diff'):
        plt.text(.5, 0.85, f'Contours: Difference to ERA5 ({diff_text})', transform=fig_mean.transFigure, fontsize='small', horizontalalignment='center')
    elif (plotting_mode=='mean+era'):
        plt.text(.5, 0.85, f'{diff_text} - Contours: ERA5 mean field', transform=fig_mean.transFigure, fontsize='small', horizontalalignment='center')
    elif (plotting_mode=='diff+era'):
        plt.text(.5, 0.85, f'{diff_text} - Contours: ERA5 mean field', transform=fig_mean.transFigure, fontsize='small', horizontalalignment='center')


    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 15, 'color': 'gray'}
    gl.ylabel_style = {'size': 15, 'color': 'gray'}

    return fig_mean



def plot_diff_map(data, lons, lats, regdata_type, fixed_bounds, standard_title, title='', diff_text=''):
#def plot_diff_map(data, lons, lats, title='',regdata_type):

    fig_means, ax = plt.subplots(1, 1, figsize=(7,5),subplot_kw={'projection': ccrs.EquidistantConic(central_longitude=-20.0)})

    cmap = mpl.cm.coolwarm 
    #colorlevels = np.linspace(5000,5900,10) 
    if (regdata_type=='Geopotential height'):
    	cbarlab = 'geopotential height at 500 hPa [m]'
    	colorlevels = np.linspace(-100,100,11)
    elif (regdata_type=='Temperature'):
    	cbarlab = 'Temperature [°C]'
    	colorlevels = np.linspace(-10,10,11)
    elif (regdata_type=='Precipitation'):
    	cbarlab = 'Precipitation [mm/day]'
    	colorlevels = np.linspace(-3,3,11)

    if not fixed_bounds:
	    n_colorlevels=11
	    maxbound = max(np.abs(np.amin(data)), np.abs(np.amax(data))) 
	    maxbound = float("{:.0e}".format(maxbound))      # round to the first nonzero number
	    cbarticks = np.linspace(-maxbound,maxbound,n_colorlevels) 
	    colorlevels = np.linspace(-maxbound,maxbound,n_colorlevels) 

    im = ax.contourf(lons, lats, data, levels=colorlevels, 
                     cmap=cmap, transform=ccrs.PlateCarree(), extend='both')
    ax.coastlines()
    cbar = fig_means.colorbar(im)
    cbar.set_label(cbarlab)  

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 15, 'color': 'gray'}
    gl.ylabel_style = {'size': 15, 'color': 'gray'}
    
    fig_means.suptitle(title, fontsize='x-large')
    plt.text(.5, 0.9, f'ERA5 - {standard_title}', transform=fig_means.transFigure, fontsize='large', horizontalalignment='center')
    plt.text(.5, 0.85, f'{diff_text}', transform=fig_means.transFigure, fontsize='medium', horizontalalignment='center')

    #ax.set(title=title)

    
    return fig_means






def plot_4_diff_map(data, lons, lats, eofs, regdata_type, fixed_bounds, title='', order=[0,1,2,3]):

    #plt.style.use('default') 

    n_colorlevels=11
    cmap = mpl.cm.coolwarm 

    if (regdata_type=='Geopotential height'):
        cbarlab = 'geopotential height at 500 hPa [m]'
        colorlevels = np.linspace(-20,20,n_colorlevels)
    elif (regdata_type=='Temperature'):
        cbarlab = 'Temperature [°C]'
        colorlevels = np.linspace(-1.5,1.5,n_colorlevels)
    elif (regdata_type=='Precipitation'):
        cbarlab = 'Precipitation [mm/day]'
        colorlevels = np.linspace(-0.5,0.5,n_colorlevels)

    if not fixed_bounds:
        maxbound = max(np.abs(np.amin(data)), np.abs(np.amax(data))) 
        maxbound = float("{:.0e}".format(maxbound))      # round to the first nonzero number
        cbarticks = np.linspace(-maxbound,maxbound,n_colorlevels) 
        colorlevels = np.linspace(-maxbound,maxbound,n_colorlevels) 


    fig, axs = plt.subplots(2, 2, figsize=(12,10), subplot_kw={'projection': ccrs.EquidistantConic(central_longitude=-20.0)})

    i = 0
    for row in range(2):
        for col in range(2):
            ax = axs[row, col]
            # add a thin outline of the EOF patterns:
            line_c = ax.contour(lons, lats, eofs[i,:,:], levels=colorlevels, colors=['grey'], linewidths=0.5 ,transform=ccrs.PlateCarree())
            pcm = ax.contourf(lons, lats, data[i,:,:], levels=colorlevels, cmap=cmap, transform=ccrs.PlateCarree(), extend="both")          
            ax.coastlines()
            ax.set_title(f'PC {order[i]+1}', loc='left', fontsize='xx-large');
            ax.set_title(f'Mean difference: {np.mean(data[i,:]):.2f}', loc='center', fontsize='xx-large');
            #ax.set_title(f'{varfrac[i]*100:.1f} %', loc='right', fontsize='xx-large');
            #ax.set_facecolor('white')
            i += 1

            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
            gl.xlabels_top = False
            gl.ylabels_right = False
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            gl.xlabel_style = {'size': 15, 'color': 'gray'}
            gl.ylabel_style = {'size': 15, 'color': 'gray'}

    cb_ax = fig.add_axes([0.91, 0.2, 0.02, 0.6])  
    cb1 = fig.colorbar(pcm, cax=cb_ax, ticks = colorlevels)
    cb1.set_label(cbarlab)  
    plt.subplots_adjust(left=0.09,bottom=0.1, right=0.89, top=0.85, wspace=0.2, hspace=0.35)

    fig.suptitle(title)
    #fig.suptitle(f'4 leading EOFs of geopotential height at 500 hPa', y=0.98, fontsize='xx-large')
    #plt.text(.5, 0.93, standard_title, transform=fig.transFigure, fontsize='x-large', horizontalalignment='center')
    #plt.text(.5, 0.9, f'Explained variance: {np.sum(varfrac)*100:.1f} %', transform=fig.transFigure, fontsize='x-large', horizontalalignment='center')

    return fig



