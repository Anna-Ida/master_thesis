









def plot_1_PCR1(regdata_type, eofs, varfrac, coefficients, corr_coefficients,  
	n_colorlevels, lons, lats, regdatatypes_units):


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
    
    fig.suptitle(f'{regdata_type} PC {PCnum+1} {reg_type}', fontsize='x-large')
    plt.text(.5, 0.9, standard_title, transform=fig_mean.transFigure, fontsize='large', horizontalalignment='center')
    #plt.text(.5, 0.85, f'{diff_text} - Contours: ERA5 mean Z500 field', transform=fig_mean.transFigure, fontsize='medium', horizontalalignment='center')


    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 15, 'color': 'gray'}
    gl.ylabel_style = {'size': 15, 'color': 'gray'}

    plt.subplots_adjust(left=0.09,bottom=0.05, right=0.8, top=0.85, wspace=0.2, hspace=0.35)

    return fig

