from netCDF4 import Dataset, num2date
import matplotlib as mpl
from matplotlib import gridspec,cm, rcParams, pyplot as plt
import numpy as np
import math
from sklearn.linear_model import LinearRegression
from eofs.standard import Eof
from eofs.tools.standard import correlation_map, covariance_map
from os.path import exists
import sys  
from cartopy import config, crs as ccrs
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER





def PC_regression(data, reg_type, regdata_type, regdatatypes_units, lats, lons, pcs, n_colorlevels, eofs, varfrac, 
    #analysis_type,season_mean,extremes,model,variant,season,month,timeframe,running_mean,extr_type,extreme_region_name,
    loaddata, savedata, savePCRdataas, model, saveregmeanas,
    add_eof_pattern, mark_high_corr, mask_low_corr, corr_threshold, standard_title, savePCreg, savePCregas, fixed_bounds,order=[0,1,2,3]):

    #fixed_bounds,regdata_type,regdatatypes_units,reg_type,n_colorlevels,coefficients,lons,lats,add_eof_pattern,
    #mark_high_corr,mask_low_corr,high_corr_reg_coefficients,eofs,corr_threshold,varfrac,savePCregas,savePCreg)


    plot_mean_field(data,lons,lats,regdata_type,standard_title,n_colorlevels,saveregmeanas)

    if savedata:
        try:
            np.save(f'data/{model}/reg_data_{savePCRdataas}', data)
        except NotImplementedError:                                       # in case it is a masked array
            np.save(f'data/{model}/reg_data_{savePCRdataas}', data.data)       

    if loaddata:
        coefficients = np.load(f'data/{model}/coeffs_{savePCRdataas}.npy')
        corr_coefficients = np.load(f'data/{model}/corrcoeffs_{savePCRdataas}.npy')
        high_corr_reg_coefficients = np.where(np.logical_or(corr_coefficients > corr_threshold, corr_coefficients<-corr_threshold), coefficients, np.nan)


    else:
        # calculate anomalies: 
        # ------------------------------------------------    
        col_mean = np.mean(data, axis=0)
        anoms = data - col_mean
        data = anoms
        #print(f' Shape data (anomalies):  {np.shape(data)}')

        # Simon divides PCs again by std here

        
        # calculate coefficients: 
        # ------------------------------------------------
        if (reg_type=='regression'):
            coefficients = np.zeros((4,len(lats),len(lons)))
            for iPC in range(4):
                PC = pcs[:,iPC]
                PC = np.reshape(PC,(len(pcs),1))

                for ilat in range(len(lats)):
                    for ilon in range(len(lons)):
                        y = data[:,ilat,ilon]
                        reg = LinearRegression().fit(PC, y)
                        coefficients[iPC,ilat,ilon] = reg.coef_
                        #coefficients[iPC,ilat,ilon] = np.polynomial.polynomial.polyfit(PC, y, 1)[1]     # same result

        corr_coefficients = correlation_map(pcs, data)
        high_corr_reg_coefficients = np.where(np.logical_or(corr_coefficients > corr_threshold, corr_coefficients<-corr_threshold), coefficients, np.nan)

        if (reg_type=='correlation'):
            print('-- calculating correlation coefficients --')
            coefficients = corr_coefficients
     
        if (reg_type=='covariance'):
            print('-- calculating covariance --')
            coefficients = covariance_map(pcs, data)
        
    # -------------------------------------------------------------------
    # save data:
    if savedata:
        try:
            np.save(f'data/{model}/meanfield_{savePCRdataas}', col_mean)
        except NotImplementedError:                                       # in case it is a masked array
            np.save(f'data/{model}/meanfield_{savePCRdataas}', col_mean.data)
        try:
            np.save(f'data/{model}/coeffs_{savePCRdataas}', coefficients)
        except NotImplementedError:
            np.save(f'data/{model}/coeffs_{savePCRdataas}', coefficients.data)
        try:
            np.save(f'data/{model}/corrcoeffs_{savePCRdataas}', corr_coefficients)
        except NotImplementedError:
            np.save(f'data/{model}/corrcoeffs_{savePCRdataas}', corr_coefficients.data)

    
    # plot maps:
    # ------------------------------------------------
    plot_PCR(fixed_bounds,regdata_type,regdatatypes_units,reg_type,n_colorlevels,coefficients,lons,lats,add_eof_pattern,
    mark_high_corr,mask_low_corr,high_corr_reg_coefficients,eofs,corr_threshold,varfrac,standard_title,savePCregas,savePCreg,order)



    # print maximum regression locations:
    # ------------------------------------------------
    PCnum = 0
    maxreg = np.argmax(coefficients[PCnum,:])
    ind = np.unravel_index(maxreg, (len(lats),len(lons)))
    print(f'Max PC {PCnum+1} {regdata_type} regression value: {coefficients[PCnum,:].flat[maxreg]:.2f} at lat: {lats[ind[0]]} , lon: {lons[ind[1]]} ')

    minreg = np.argmin(coefficients[PCnum,:])
    ind = np.unravel_index(minreg, (len(lats),len(lons)))
    print(f'Min PC {PCnum+1} {regdata_type} regression value: {coefficients[PCnum,:].flat[minreg]:.2f} at lat: {lats[ind[0]]} , lon: {lons[ind[1]]} ')




    return coefficients, corr_coefficients





def plot_mean_field(data,lons,lats,regdata_type,standard_title,n_colorlevels,saveregmeanas):
    #plt.style.use('ggplot')
    plt.style.use('default')
    cmap = mpl.cm.coolwarm 
    col_mean = np.mean(data, axis=0)

    fig_mean, ax = plt.subplots(figsize=(6,4), subplot_kw={'projection': ccrs.EquidistantConic(central_longitude=-20.0)})

    if (regdata_type=='Temperature'):
        maxbound = max(np.abs(np.amin(col_mean)), np.abs(np.amax(col_mean))) 
        maxbound = float("{:.0e}".format(maxbound))      # round to the first nonzero number
        #colorlevels = np.linspace(-maxbound,maxbound,n_colorlevels) 
        colorlevels = np.linspace(-50,50,11) 

    elif (regdata_type=='Precipitation'):
        cmap = mpl.cm.viridis #- mpl.cm.coolwarm 
        colorlevels = np.linspace(0,9,10) 
    
    elif (regdata_type=='Geopotential height'):
        colorlevels = np.linspace(5000,5900,10)

    plt.contourf(lons, lats, col_mean, cmap=cmap, levels=colorlevels, transform=ccrs.PlateCarree())

    ax.coastlines()
    cbar = plt.colorbar()

    if (regdata_type=='Temperature'):
        cbar.set_label(f'Mean {regdata_type} [°C]')
    elif (regdata_type=='Precipitation'):
        cbar.set_label(f'Mean {regdata_type} [mm/day]')
    elif (regdata_type=='Geopotential height'):
        cbar.set_label(f'Mean {regdata_type} [m]')
    else:
        cbar.set_label(f'{regdata_type} [unit?]')  
    #ax.set(title=f'Mean field {season} {timeframe} \n ({model} {variant})')
    fig_mean.suptitle(f'Mean {regdata_type} field', fontsize='x-large')
    plt.text(.5, 0.9, standard_title, transform=fig_mean.transFigure, fontsize='large', horizontalalignment='center')



    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.left_labels = False
    gl.right_labels = False
    #gl.top_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 12, 'color': 'gray'}
    gl.ylabel_style = {'size': 12, 'color': 'gray'}


    #plt.subplots_adjust(left=0.09,bottom=0.1, right=0.89, top=0.85, wspace=0.2, hspace=0.35)
    fig_mean.tight_layout()

    fig_mean.savefig(saveregmeanas)




def plot_PCR(fixed_bounds,regdata_type,regdatatypes_units,reg_type,n_colorlevels,coefficients,lons,lats,add_eof_pattern,
    mark_high_corr,mask_low_corr,high_corr_reg_coefficients,eofs,corr_threshold,varfrac,standard_title,savePCregas,savePCreg,order):
    #fixed_bounds,regdata_type,regdatatypes_units,reg_type,n_colorlevels,coefficients,lons,lats,add_eof_pattern,
    #mark_high_corr,mask_low_corr,high_corr_reg_coefficients,eofs,corr_threshold,varfrac,
    #analysis_type,season_mean,extremes,model,variant,season,month,timeframe,running_mean,extr_type,extreme_region_name,savePCregas,savePCreg):
    
    #plt.style.use('default') 
    # define color levels to use for all 4 contour plots:
    cmap = mpl.cm.coolwarm #viridis
    if fixed_bounds: # standardize colorscale for all regression plots:
        if (regdata_type=='Temperature'):
            #colorlevels = np.linspace(-5.0,5.0,n_colorlevels)  #(-1.0,1.0,n_colorlevels) 
            colorlevels = np.linspace(-1.5,1.5,n_colorlevels)  
        elif (regdata_type=='Precipitation'):
            #colorlevels = np.linspace(-0.00001,0.00001,n_colorlevels)
            #colorlevels = np.linspace(-0.00001,0.00001,n_colorlevels) 
            colorlevels = np.linspace(-1,1,n_colorlevels) 
            #colorlevels = np.linspace(-0.005,0.005,n_colorlevels)    # for ERA5 
        elif (regdata_type=='Geopotential height'):
            colorlevels = np.linspace(-40,40,n_colorlevels) 
    else:  # flexible bounds based on data maximum:
        maxbound = max(np.abs(np.amin(coefficients)), np.abs(np.amax(coefficients)))  
        maxbound = maxbound - (maxbound/3 ) # optional to make differences stand out clearer
        cbarticks = np.linspace(-maxbound,maxbound,n_colorlevels) 
        colorlevels = np.linspace(-maxbound,maxbound,n_colorlevels)   

    fig, axs = plt.subplots(2, 2, figsize=(11,8), subplot_kw={'projection': ccrs.EquidistantConic(central_longitude=-20.0)})  #figsize=(13,10)
    i = 0
    for row in range(2):
        for col in range(2):
            ax = axs[row, col]

            if mask_low_corr:
                pcm = ax.contourf(lons, lats, high_corr_reg_coefficients[i,:,:], levels=colorlevels, cmap=cmap, transform=ccrs.PlateCarree(), extend="both")
                
            else:
                pcm = ax.contourf(lons, lats, coefficients[i,:,:], levels=colorlevels, cmap=cmap, transform=ccrs.PlateCarree(), extend="both")

           
            if add_eof_pattern:
                pcm2 = ax.contour(lons, lats, eofs[i,:,:], colors='grey', linewidths=0.5, transform=ccrs.PlateCarree())
            
            if mark_high_corr:                # mark locations with high correlation for each PC
                high_corr_lats = np.array([])
                high_corr_lons = np.array([])
                for ilat in range(len(lats)):
                    for ilon in range(len(lons)):
                        if (coefficients[i,ilat,ilon]>corr_threshold) or (coefficients[i,ilat,ilon]<-corr_threshold):
                            high_corr_lats = np.append(high_corr_lats, ilat)
                            high_corr_lons = np.append(high_corr_lons, ilon)
                
                ax.scatter(x=lons[high_corr_lons.astype(int)], y=lats[high_corr_lats.astype(int)], color="grey", s=0.5, marker='.', 
                           alpha=0.5, transform=ccrs.PlateCarree(), label=f'Locations with positive/negative correlation > {corr_threshold}')
                ax.legend()
            
            ax.set_title(f'PC {order[i]+1}', loc='left', fontsize='xx-large');
            ax.set_title(f'{varfrac[i]*100:.1f} %', loc='right', fontsize='xx-large');
            
            # draw coastlines:
            ax.coastlines()
            
            # draw lats and lons:
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
            gl.left_labels = False
            gl.right_labels = False
            gl.top_labels = False
            #gl.xlabels_top = False
            #gl.ylabels_right = False
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            gl.xlabel_style = {'size': 12, 'color': 'gray'}
            gl.ylabel_style = {'size': 12, 'color': 'gray'}
            
            i += 1

    #plt.subplots_adjust(left=0.09,bottom=0.1, right=0.88, top=0.85, wspace=0.2, hspace=0.35)
    plt.subplots_adjust(left=0.01,bottom=0.05, right=0.89, top=0.83, wspace=0, hspace=0.3)
    cb_ax = fig.add_axes([0.91, 0.15, 0.02, 0.6])
    #cb_ax = fig.add_axes([0.88, 0.2, 0.02, 0.6])  
    cb1 = fig.colorbar(pcm, cax=cb_ax, ticks = colorlevels)
    #plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.85, wspace=0.2, hspace=0.35)
    cb1.set_label(f'{regdata_type} change {regdatatypes_units[regdata_type]}')
    

    fig.suptitle(f'{regdata_type} field and PC (of geopotential height) {reg_type}', fontsize='xx-large')
    #plt.text(.5, 0.91, standard_title, transform=fig.transFigure, fontsize='x-large', horizontalalignment='center')
    plt.text(.5, 0.925, standard_title, transform=fig.transFigure, fontsize='x-large', horizontalalignment='center')   #0.935
    #plt.text(.5, 0.89, f'Explained variance: {np.sum(varfrac)*100:.1f} %', transform=fig.transFigure, fontsize='x-large', horizontalalignment='center')  #0.913



    if savePCreg:
        fig.savefig(savePCregas) 




