import matplotlib as mpl
from matplotlib import gridspec,cm, rcParams, pyplot as plt
import numpy as np
from netCDF4 import Dataset, num2date
from cartopy import config, crs as ccrs
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

def plot_map(data, lons, lats, title=''): #,colorlevels=np.linspace(5000,5900,10),cbarlab = 'geopotential height at 500 hPa [m]'):
    
    fig_means, ax = plt.subplots(1, 1, figsize=(5,5),subplot_kw={'projection': ccrs.EquidistantConic(central_longitude=-20.0)})

    cmap = mpl.cm.coolwarm 
    col_mean = data #np.mean(zg_data, axis=0)
    #colorlevels = np.linspace(5000,5900,10) 

    im = ax.contourf(lons, lats, col_mean, #levels=colorlevels, 
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
    
    ax.set(title=title)
    
    #fig_means.suptitle(title, fontsize='x-large')
    
    
    return fig_means
