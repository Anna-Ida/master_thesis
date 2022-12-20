

import matplotlib as mpl
from matplotlib import gridspec,cm, rcParams, pyplot as plt
import numpy as np
from netCDF4 import Dataset, num2date
from cartopy import config, crs as ccrs
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from model_comparison import z_test_two_means

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



def plot_PCR1(mask_low_corr, add_eof_pattern, corr_threshold, lons, lats, models, variants, orders, regdata_type, regdatatypes_short, regdatatypes_units,reg_type,n_colorlevels, 
	PCnum=0, fixed_bounds=True, savefig=True, area=''):


    timeframe='1959-2014'
    season='NDJFM'
    meantext = 'mean1'
    extr_text = ''
    regridded_text='_regriddedtoIPSL'
 	

    #plt.style.use('default') 
    # define color levels to use for all 4 contour plots:
    cmap = mpl.cm.coolwarm #viridis
    if fixed_bounds: # standardize colorscale for all regression plots:
        if (regdata_type=='Temperature'):
            colorlevels = np.linspace(-1.5,1.5,n_colorlevels)  
        elif (regdata_type=='Precipitation'):
            colorlevels = np.linspace(-1,1,n_colorlevels) 
        elif (regdata_type=='Geopotential height'):
            colorlevels = np.linspace(-40,40,n_colorlevels) 
    else:  # flexible bounds based on data maximum:
        maxbound = max(np.abs(np.amin(coefficients)), np.abs(np.amax(coefficients)))  
        maxbound = maxbound - (maxbound/3 ) # optional to make differences stand out clearer
        cbarticks = np.linspace(-maxbound,maxbound,n_colorlevels) 
        colorlevels = np.linspace(-maxbound,maxbound,n_colorlevels)   

    fig, axs = plt.subplots(3, 3, figsize=(18,13), subplot_kw={'projection': ccrs.EquidistantConic(central_longitude=-20.0)})
	#fig, axs = plt.subplots(3, 2, figsize=(12,15), subplot_kw={'projection': ccrs.EquidistantConic(central_longitude=-20.0)})

	# for 2x3:
	#plt.subplots(2, 3, figsize=(19,9)
    #rows=[0,0,0,1,1,1]
    #cols=[0,1,2,0,1,2]
    #plt.subplots_adjust(left=0.09,bottom=0.1, right=0.88, top=0.85, wspace=0.2, hspace=0.35)
    #cb_ax = fig.add_axes([0.91, 0.2, 0.02, 0.6])

    rows=[0,0,0,1,1,1,2,2,2]
    cols=[0,1,2,0,1,2,0,1,2]

    # load data:
    for modeli in range(len(models)):
        #if (modeli==0):
        #    continue
        #if (modeli>8):
        #	continue 

        model = models[modeli]
        variant = variants[modeli]
        order = orders[modeli]
        grid = 'gr' if (modeli in [0,1,2,3,8,9]) else 'gn'   
        if (model=='ERA5'):
            eofvar = 'z' 
            regvar = 'tp' if (regdata_type=='Precipitation') else 't2m'
        else:
            eofvar = 'zg'
            regvar = 'pr' if (regdata_type=='Precipitation') else 'tas'
        if (regdata_type=='Geopotential height'):
            regvar = eofvar 

        eofs = np.load(f'data/{model}/eofs_4EOFs_{timeframe}_{model}_{variant}_{season}_{meantext}{regridded_text}.npy')
        varfrac = np.load(f'data/{model}/varfrac_4EOFs_{timeframe}_{model}_{variant}_{season}_{meantext}{regridded_text}.npy')
        coefficients = np.load(f'data/{model}/coeffs_PC_{regdatatypes_short[regdata_type]}_regression_{timeframe}_{model}_{variant}_{season}_{meantext}{regridded_text}.npy')
        corr_coefficients = np.load(f'data/{model}/corrcoeffs_PC_{regdatatypes_short[regdata_type]}_regression_{timeframe}_{model}_{variant}_{season}_{meantext}{regridded_text}.npy')
        high_corr_reg_coefficients = np.where(np.logical_or(corr_coefficients > corr_threshold, corr_coefficients<-corr_threshold), coefficients, np.nan)



        ax = axs[rows[modeli], cols[modeli]]


        if mask_low_corr:
            pcm = ax.contourf(lons, lats, high_corr_reg_coefficients[PCnum,:,:], levels=colorlevels, cmap=cmap, transform=ccrs.PlateCarree(), extend="both")
            
        else:
            pcm = ax.contourf(lons, lats, coefficients[PCnum,:,:], levels=colorlevels, cmap=cmap, transform=ccrs.PlateCarree(), extend="both")

       
        if add_eof_pattern:
            pcm2 = ax.contour(lons, lats, eofs[PCnum,:,:], colors='grey', linewidths=0.5, transform=ccrs.PlateCarree(), label='EOF pattern')
            #ax.legend()
        
        #if (modeli==8):
         #   model = 'EC-Earth3 r10' 
        ax.set_title(model, loc='left', fontsize='xx-large', fontweight='bold');

        ax.set_title(f'PC {order[PCnum]+1}, {varfrac[PCnum]*100:.1f} %', loc='right', fontsize='large');
        
        # draw coastlines:
        ax.coastlines()
        
        # draw lats and lons:
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 15, 'color': 'gray'}
        gl.ylabel_style = {'size': 15, 'color': 'gray'}
        


    plt.subplots_adjust(left=0.09,bottom=0.1, right=0.88, top=0.9, wspace=0.2, hspace=0.35)
    cb_ax = fig.add_axes([0.91, 0.2, 0.02, 0.6])
    cb1 = fig.colorbar(pcm, cax=cb_ax, ticks = colorlevels)
    #plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.85, wspace=0.2, hspace=0.35)
    cb1.set_label(f'{regdata_type} change {regdatatypes_units[regdata_type]}')
    

    #fig.suptitle(f'{regdata_type} field and PC (of geopotential height) {reg_type}', fontsize='xx-large')
    fig.suptitle(f'{regdata_type} PC {PCnum+1} {reg_type}', fontsize='xx-large')
    #plt.text(.5, 0.935, standard_title, transform=fig.transFigure, fontsize='x-large', horizontalalignment='center')
    #plt.text(.5, 0.913, f'Explained variance: {np.sum(varfrac)*100:.1f} %', transform=fig.transFigure, fontsize='x-large', horizontalalignment='center')
    #fig.tight_layout()

    if mask_low_corr:
        savePCregas = f'PC{PCnum+1}_{regdatatypes_short[regdata_type]}_{reg_type}_masked_{timeframe}_{season}{meantext}{extr_text}{regridded_text}.pdf'
    else:
        savePCregas = f'PC{PCnum+1}_{regdatatypes_short[regdata_type]}_{reg_type}_full_{timeframe}_{season}{meantext}{extr_text}{regridded_text}.pdf'

    if savefig:
        fig.savefig(f'output/PCRplots/{savePCregas}') 


    if (PCnum in [0,1,2]):
    	fig.savefig(f'output/plots/{savePCregas}') 





    fig, ax = plt.subplots(figsize=(7,4), subplot_kw={'projection': ccrs.EquidistantConic(central_longitude=-20.0)})
    # load data:
    modeli=0
    model = models[modeli]
    variant = variants[modeli]
    order = orders[modeli]
    eofvar = 'z' 
    regvar = 'tp' if (regdata_type=='Precipitation') else 't2m'
    if (regdata_type=='Geopotential height'):
        regvar = eofvar 
    eofs = np.load(f'data/{model}/eofs_4EOFs_{timeframe}_{model}_{variant}_{season}_{meantext}{regridded_text}.npy')
    varfrac = np.load(f'data/{model}/varfrac_4EOFs_{timeframe}_{model}_{variant}_{season}_{meantext}{regridded_text}.npy')
    coefficients = np.load(f'data/{model}/coeffs_PC_{regdatatypes_short[regdata_type]}_regression_{timeframe}_{model}_{variant}_{season}_{meantext}{regridded_text}.npy')
    corr_coefficients = np.load(f'data/{model}/corrcoeffs_PC_{regdatatypes_short[regdata_type]}_regression_{timeframe}_{model}_{variant}_{season}_{meantext}{regridded_text}.npy')
    high_corr_reg_coefficients = np.where(np.logical_or(corr_coefficients > corr_threshold, corr_coefficients<-corr_threshold), coefficients, np.nan)

    if mask_low_corr:
        pcm = ax.contourf(lons, lats, high_corr_reg_coefficients[PCnum,:,:], levels=colorlevels, cmap=cmap, transform=ccrs.PlateCarree(), extend="both")
    else:
        pcm = ax.contourf(lons, lats, coefficients[PCnum,:,:], levels=colorlevels, cmap=cmap, transform=ccrs.PlateCarree(), extend="both")
    if add_eof_pattern:
        pcm2 = ax.contour(lons, lats, eofs[PCnum,:,:], colors='grey', linewidths=0.5, transform=ccrs.PlateCarree(), label='EOF pattern')
    
    ax.set_title(model, loc='left', fontsize='xx-large', fontweight='bold');
    ax.set_title(f'PC {PCnum+1}, {varfrac[PCnum]*100:.1f} %', loc='right', fontsize='large');
    
    ax.coastlines()
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 15, 'color': 'gray'}
    gl.ylabel_style = {'size': 15, 'color': 'gray'}
    
    plt.subplots_adjust(left=0.09,bottom=0.1, right=0.78, top=0.7, wspace=0.2, hspace=0.35)
    cb_ax = fig.add_axes([0.75, 0.15, 0.03, 0.6])
    cb1 = fig.colorbar(pcm, cax=cb_ax, ticks = colorlevels)
    #plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.85, wspace=0.2, hspace=0.35)
    cb1.set_label(f'{regdata_type} change {regdatatypes_units[regdata_type]}')
    
    #fig.suptitle(f'{regdata_type} field and PC (of geopotential height) {reg_type}', fontsize='large')
    fig.suptitle(f'{regdata_type} PC {reg_type}', fontsize='large')
    standard_title = f'{season} {meantext} {timeframe} {area}' #f'{model} {variant}, {season} {meantext} {timeframe} {area}'
    #plt.text(.5, 0.88, standard_title, transform=fig.transFigure, fontsize='large', horizontalalignment='center')
    #plt.text(.5, 0.913, f'Explained variance: {np.sum(varfrac)*100:.1f} %', transform=fig.transFigure, fontsize='x-large', horizontalalignment='center')


    if mask_low_corr:
        savePCregas = f'{model}_PC{PCnum+1}_{regdatatypes_short[regdata_type]}_{reg_type}_masked_{timeframe}_{season}{meantext}{extr_text}{regridded_text}.pdf'
    else:
        savePCregas = f'{model}_PC{PCnum+1}_{regdatatypes_short[regdata_type]}_{reg_type}_full_{timeframe}_{season}{meantext}{extr_text}{regridded_text}.pdf'

    if savefig:
        fig.savefig(f'output/PCRplots/{savePCregas}') 










def plot_meandiffs(lons, lats, mean_diff_arr, models, variants, orders, regdata_type, regdatatypes_short, regdatatypes_units, savefig=True, area=''):
	

    timeframe='1959-2014'
    season='NDJFM'
    meantext = 'mean1'
    extr_text = ''
    regridded_text='_regriddedtoIPSL'

    if (regdata_type=='Temperature'):
        colorlevels = np.linspace(-50,50,11)
        contourlevels = np.linspace(-10,10,11)
        cbarlab = 'Model - ERA5 Temperature difference [°C]'
    elif (regdata_type=='Precipitation'):
        #cmap = mpl.cm.viridis #- mpl.cm.coolwarm 
        colorlevels = np.linspace(0,9,10) 
        contourlevels = np.linspace(-3,3,11)
        cbarlab = 'Model - ERA5 Precipitation difference [mm/day]'
    elif (regdata_type=='Geopotential height'):
        colorlevels = np.linspace(5000,5900,10)
        contourlevels = np.linspace(-100,100,11)
        contourlevels_masked = np.linspace(-20,20,11)
        cbarlab = 'Model - ERA5 Geopotential height at 500 hPa difference [m]'


    cmap = mpl.cm.coolwarm 


    fig, axs = plt.subplots(3, 3, figsize=(18,13), subplot_kw={'projection': ccrs.EquidistantConic(central_longitude=-20.0)})

    rows=[0,0,0,1,1,1,2,2,2]
    cols=[0,1,2,0,1,2,0,1,2]

    # load data:
    for modeli in range(len(models)):

        model = models[modeli]
        variant = variants[modeli]
        order = orders[modeli] 
        eofvar = 'zgs'
        regvar = 'pr' if (regdata_type=='Precipitation') else 'tas'
        if (regdata_type=='Geopotential height'):
            regvar = eofvar 
           

        obs_regdata_mean = np.load(f'data/ERA5/meanfield_PC_{regvar}_regression_1959-2014_ERA5_1_NDJFM_mean1_regriddedtoIPSL.npy')
        obs_reg_data = np.load(f'data/ERA5/reg_data_PC_{regvar}_regression_1959-2014_ERA5_1_NDJFM_mean1_regriddedtoIPSL.npy')
        obs_zg_mean = np.load(f'data/ERA5/meanfield_PC_zgs_regression_1959-2014_ERA5_1_NDJFM_mean1_regriddedtoIPSL.npy')
        
        model_regdata_mean = np.load(f'data/{model}/meanfield_PC_{regdatatypes_short[regdata_type]}_regression_{timeframe}_{model}_{variant}_{season}_{meantext}{regridded_text}.npy')
        model_reg_data = np.load(f'data/{model}/reg_data_PC_{regdatatypes_short[regdata_type]}_regression_{timeframe}_{model}_{variant}_{season}_{meantext}{regridded_text}.npy')
        

        diff_mean = model_regdata_mean - obs_regdata_mean
        diff_mean_p = z_test_two_means(model_reg_data,obs_reg_data)
        diff_mean_masked = np.ma.masked_where(diff_mean_p > 0.05, diff_mean)



        ax = axs[rows[modeli], cols[modeli]]

        CF = ax.contourf(lons, lats, diff_mean_masked, levels=contourlevels, cmap=cmap, transform=ccrs.PlateCarree(),extend='both')
        CS = ax.contour(lons, lats, obs_zg_mean, levels=np.linspace(5000,5900,10), colors='grey', linewidths=0.5, transform=ccrs.PlateCarree())

        ax.set_title(model, loc='left', fontsize='xx-large', fontweight='bold');
        ax.set_title(f'{mean_diff_arr[1,modeli]:.1f} {regdatatypes_units[regdata_type]}', loc='right', fontsize='large');
        
        # draw coastlines:
        ax.coastlines()
        
        # draw lats and lons:
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 15, 'color': 'gray'}
        gl.ylabel_style = {'size': 15, 'color': 'gray'}
        

    plt.subplots_adjust(left=0.09,bottom=0.1, right=0.88, top=0.9, wspace=0.2, hspace=0.35)
    cb_ax = fig.add_axes([0.91, 0.2, 0.02, 0.6])
    cb1 = fig.colorbar(CF, cax=cb_ax, ticks = contourlevels)
    cb1.set_label(cbarlab)
    #fig.suptitle(f'{regdata_type} PC {PCnum+1} {reg_type}', fontsize='xx-large')

    fig.savefig(f'output/mean_diff/{regvar}_meandiff_masked_maps.pdf') 
    fig.savefig(f'output/plots/{regvar}_meandiff_masked_maps.pdf') 





def plot_mean_plus_diff(model_mean,diff_data,obs_mean,lons,lats,plotting_mode,regdata_type,standard_title,diff_text='',masked=False):
    #plt.style.use('ggplot')
    plt.style.use('default')
    cmap = mpl.cm.coolwarm 

    # to always plot ERA5 Z500 mean field:
    obs_mean = np.load(f'data/ERA5/meanfield_PC_{regvar}_regression_1959-2014_ERA5_1_NDJFM_mean1_regriddedtoIPSL.npy')
    

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
        contourlevels_masked = np.linspace(-20,20,11)
        cbarlab = 'Geopotential height at 500 hPa [m]'

    #if masked:
        #contourlevels = contourlevels_masked

    fig_mean, ax = plt.subplots(figsize=(7,5), subplot_kw={'projection': ccrs.EquidistantConic(central_longitude=-20.0)})
    if (plotting_mode=='mean+diff'):
        CF = ax.contourf(lons, lats, model_mean, levels=colorlevels, cmap=cmap, transform=ccrs.PlateCarree())
        CS = ax.contour(lons, lats, diff_data, levels=contourlevels, colors='white', linewidths=0.5, transform=ccrs.PlateCarree())
    elif (plotting_mode=='mean+era'):
        CF = ax.contourf(lons, lats, model_mean, levels=colorlevels, cmap=cmap, transform=ccrs.PlateCarree())
        CS = ax.contour(lons, lats, obs_mean, levels=colorlevels, colors='white', linewidths=0.5, transform=ccrs.PlateCarree())
    elif (plotting_mode=='diff+era'):
        CF = ax.contourf(lons, lats, diff_data, levels=contourlevels, cmap=cmap, transform=ccrs.PlateCarree(),extend='both')
        CS = ax.contour(lons, lats, obs_mean, levels=colorlevels, colors='grey', linewidths=0.5, transform=ccrs.PlateCarree())

            
    ax.coastlines()
    cbar = fig_mean.colorbar(CF)
    cbar.set_label(cbarlab) 
    ax.clabel(CS, CS.levels, inline=True, fontsize=10) 
    
    if (plotting_mode=='diff+era'):
        if masked:
            fig_mean.suptitle(f'Significant difference to ERA5 mean {regdata_type} field', fontsize='x-large')
        else:
            fig_mean.suptitle(f'Difference to ERA5 mean {regdata_type} field', fontsize='x-large')
    else:
        fig_mean.suptitle(f'Mean {regdata_type} field', fontsize='x-large')
    plt.text(.5, 0.9, standard_title, transform=fig_mean.transFigure, fontsize='large', horizontalalignment='center')
    if (plotting_mode=='mean+diff'):
        plt.text(.5, 0.85, f'Contours: Difference to ERA5 ({diff_text})', transform=fig_mean.transFigure, fontsize='small', horizontalalignment='center')
    elif (plotting_mode=='mean+era'):
        plt.text(.5, 0.85, f'{diff_text} - Contours: ERA5 mean field', transform=fig_mean.transFigure, fontsize='small', horizontalalignment='center')
    elif (plotting_mode=='diff+era'):
        plt.text(.5, 0.85, f'{diff_text} - Contours: ERA5 mean Z500 field', transform=fig_mean.transFigure, fontsize='small', horizontalalignment='center')


    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 15, 'color': 'gray'}
    gl.ylabel_style = {'size': 15, 'color': 'gray'}

    return fig_mean






def plot_eigenvalues(models, variants):

    timeframe='1959-2014'
    season='NDJFM'
    meantext = 'mean1'
    extr_text = ''
    regridded_text='_regriddedtoIPSL'

    #plt.style.use('default') 


    fig, axs = plt.subplots(3, 3, figsize=(16,10))
    rows=[0,0,0,1,1,1,2,2,2]
    cols=[0,1,2,0,1,2,0,1,2]

    # load data:
    for modeli in range(len(models)):

        model = models[modeli]
        variant = variants[modeli]
        ID = f'{timeframe}_{model}_{variant}_{season}_{meantext}{regridded_text}'


        varfrac = np.load(f'data/{model}/varfrac_40_{ID}.npy')
        #varfrac4 = np.load(f'data/{model}/varfrac_4EOFs_{ID}.npy')
        eigenvalues = np.load(f'data/{model}/eigenvals_40_{ID}.npy')
        errors = np.load(f'data/{model}/errors_40_{ID}.npy')
        errors_scaled = np.load(f'data/{model}/errors_scaled_40_{ID}.npy')

        #print(f'varfrac first 4 = {np.sum(varfrac4)*100:.1f} =? {np.sum(varfrac[0:4])*100:.1f}')

        if (modeli==0):
        	ERA_1, ERA_1err = varfrac[0]*100, errors_scaled[0]*100
        	ERA_2, ERA_2err = varfrac[1]*100, errors_scaled[1]*100
        	ERA_3, ERA_3err = varfrac[2]*100, errors_scaled[2]*100

        ax = axs[rows[modeli], cols[modeli]]
        myxaxis = np.arange(len(varfrac))+1
        ax.errorbar(myxaxis, varfrac*100, yerr=errors_scaled*100, fmt='o', color='black', ms=3, ecolor='black', elinewidth=1, capsize=2)
        if (modeli!=0):
        	ax.hlines(ERA_1,0,2, color='grey', alpha=0.5)
        	ax.fill_betweenx((ERA_1+ERA_1err, ERA_1-ERA_1err),0,2,color='grey', alpha=0.3)
        	ax.hlines(ERA_2,1,3, color='grey', alpha=0.5)
        	ax.fill_betweenx((ERA_2+ERA_2err, ERA_2-ERA_2err),1,3,color='grey', alpha=0.3)
        	ax.hlines(ERA_3,2,4, color='grey', alpha=0.5)
        	ax.fill_betweenx((ERA_3+ERA_3err, ERA_3-ERA_3err),2,4,color='grey', alpha=0.3)
        ax.set_title(model, loc='left', fontsize='xx-large', fontweight='bold');
        ax.set_title(f'EOF1-4: {np.sum(varfrac[0:4])*100:.1f} %', loc='right', fontsize='x-large');
        ax.set(xlabel='Rank', ylabel='Eigenvalue [%]', ylim=(0,60));
    
    fig.tight_layout()
    #plt.subplots_adjust(left=0.05,bottom=0.05, right=0.95, top=0.95, wspace=0.2, hspace=0.3)
    
    fig.savefig(f'output/plots/eigenvalue_spectra.pdf')






def plot_1_PCR1(regdata_type, eofs, varfrac, coefficients, corr_coefficients,  
    n_colorlevels, lons, lats, regdatatypes_units, standard_title):


    #plt.style.use('ggplot')
    plt.style.use('default')
    cmap = mpl.cm.coolwarm 

    PCnum=0

    timeframe='1959-2014'
    season='NDJFM'
    meantext = 'mean1'
    extr_text = ''
    regridded_text='_regriddedtoIPSL'
 

    #plt.style.use('default') 
    # define color levels to use for all 4 contour plots:
    cmap = mpl.cm.coolwarm #viridis
    if (regdata_type=='Temperature'):
        colorlevels = np.linspace(-1.5,1.5,n_colorlevels)  
    elif (regdata_type=='Precipitation'):
        colorlevels = np.linspace(-1,1,n_colorlevels) 
    elif (regdata_type=='Geopotential height'):
        colorlevels = np.linspace(-40,40,n_colorlevels) 


    corr_threshold = 0.3
    high_corr_reg_coefficients = np.where(np.logical_or(corr_coefficients > corr_threshold, corr_coefficients<-corr_threshold), coefficients, np.nan)



    fig, ax = plt.subplots(1, 1, figsize=(7,5), subplot_kw={'projection': ccrs.EquidistantConic(central_longitude=-20.0)})
    
    pcm = ax.contourf(lons, lats, high_corr_reg_coefficients[PCnum,:,:], levels=colorlevels, cmap=cmap, transform=ccrs.PlateCarree(), extend="both")
    pcm2 = ax.contour(lons, lats, eofs[PCnum,:,:], colors='grey', linewidths=0.5, transform=ccrs.PlateCarree(), label='EOF pattern')
             
            
    ax.coastlines()
    cb_ax = fig.add_axes([0.83, 0.1, 0.03, 0.73])  
    cbar = fig.colorbar(pcm, cax=cb_ax)#, ticks = cbarticks)
    cbar.set_label(f'{regdata_type} change {regdatatypes_units[regdata_type]}')
    #cbar.set_label(cbarlab) 
    #ax.clabel(CS, CS.levels, inline=True, fontsize=10) 
    #cbar = fig_mean.colorbar(CF)
    #cbar.set_label(cbarlab) 
    #ax.clabel(CS, CS.levels, inline=True, fontsize=10) 
    
    fig.suptitle(f'{regdata_type} PC {PCnum+1} Regression', fontsize='x-large')
    plt.text(.5, 0.9, standard_title, transform=fig.transFigure, fontsize='large', horizontalalignment='center')
    #plt.text(.5, 0.85, f'{diff_text} - Contours: ERA5 mean Z500 field', transform=fig_mean.transFigure, fontsize='medium', horizontalalignment='center')


    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 15, 'color': 'gray'}
    gl.ylabel_style = {'size': 15, 'color': 'gray'}

    plt.subplots_adjust(left=0.09,bottom=0.05, right=0.8, top=0.85, wspace=0.2, hspace=0.35)

    return fig



