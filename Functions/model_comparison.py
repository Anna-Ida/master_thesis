import matplotlib as mpl
from matplotlib import gridspec,cm, rcParams, pyplot as plt
import numpy as np
from netCDF4 import Dataset, num2date
from cartopy import config, crs as ccrs
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import scipy.stats as stats
from uncertainties import unumpy
import skill_metrics as sm

def z_test_two_means(data1,data2):
    '''
    Two-sample z-Test (Gaussian assumption)
    Rule-of-thumb: If N > 10-20 or σ known then Z-test, else T-test.
    This is a two-tailed test!
    '''
    # calculate mean & uncertainty on the mean:
    x1 = np.mean(data1, axis=0)
    sig_x1 = np.std(data1, axis=0, ddof=1) / np.sqrt(len(data1))   # do ddof=1 to get corrected std

    x2 = np.mean(data2, axis=0)
    sig_x2 = np.std(data2, axis=0, ddof=1) / np.sqrt(len(data2))

    
    z = (x1-x2)/np.sqrt(sig_x1**2 + sig_x2**2)
    #print(f" z-score = {z}")
    #print(f" {x1:.2f} is {z:.4f} sigma away from {x2:.2f}.")

    #find p-value for two-tailed test
    p_val = stats.norm.sf(abs(z))*2
    # sf = Survival function (also defined as 1 - cdf, but sf is sometimes more accurate).
    #print(f" p-value = {p_val:.4f}")

    # Turn a number of sigmas into a probability (i.e. p-value):
    #p_val = 1.0 - stats.norm.cdf(z,loc=0,scale=1)  # To get a p-value, we ask what the chance is to get a larger difference
    # => gives the same result, only difference lies in *2 for 2-sided test. But even that difference is not very large in practice
    
    return p_val


def plot_mean_plus_diff(model_mean,diff_data,obs_mean,lons,lats,plotting_mode,regdata_type,standard_title,diff_text='',masked=False):
    #plt.style.use('ggplot')
    plt.style.use('default')
    cmap = mpl.cm.coolwarm 

    # to always plot ERA5 Z500 mean field:
    obs_mean = np.load(f'data/ERA5/meanfield_PC_zgs_regression_1959-2014_ERA5_1_NDJFM_mean1_regriddedtoIPSL.npy')
    

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
        cbarlab = 'Difference in geopotential height at 500 hPa [m]'

    #if masked:
        #contourlevels = contourlevels_masked

    fig_mean, ax = plt.subplots(1, 1, figsize=(7,5), subplot_kw={'projection': ccrs.EquidistantConic(central_longitude=-20.0)})
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
    cb_ax = fig_mean.add_axes([0.83, 0.1, 0.03, 0.73])  
    cbar = fig_mean.colorbar(CF, cax=cb_ax)#, ticks = cbarticks)
    cbar.set_label(cbarlab) 
    ax.clabel(CS, CS.levels, inline=True, fontsize=10) 
    #cbar = fig_mean.colorbar(CF)
    #cbar.set_label(cbarlab) 
    #ax.clabel(CS, CS.levels, inline=True, fontsize=10) 
    
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
        plt.text(.5, 0.85, f'{diff_text} - Contours: ERA5 mean Z500 field', transform=fig_mean.transFigure, fontsize='medium', horizontalalignment='center')


    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 15, 'color': 'gray'}
    gl.ylabel_style = {'size': 15, 'color': 'gray'}

    plt.subplots_adjust(left=0.09,bottom=0.05, right=0.8, top=0.85, wspace=0.2, hspace=0.35)

    return fig_mean



def plot_diff_map(data, lons, lats, regdata_type, fixed_bounds, standard_title, title='', diff_text=''):
#def plot_diff_map(data, lons, lats, title='',regdata_type):

    fig_means, ax = plt.subplots(1, 1, figsize=(7,5),subplot_kw={'projection': ccrs.EquidistantConic(central_longitude=-20.0)})

    cmap = mpl.cm.coolwarm 

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

    return fig_means




def plot_diff_p_map(data, lons, lats, regdata_type, fixed_bounds, standard_title, title='', diff_text=''):
#def plot_diff_map(data, lons, lats, title='',regdata_type):

    fig_means, ax = plt.subplots(1, 1, figsize=(7,5),subplot_kw={'projection': ccrs.EquidistantConic(central_longitude=-20.0)})

    cmap = mpl.cm.coolwarm 
    colorlevels = np.linspace(0.05,0.95,10) 
    cbarlab = 'p-value'

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
    #plt.text(.5, 0.85, f'{diff_text}', transform=fig_means.transFigure, fontsize='medium', horizontalalignment='center')

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




def calc_gradients(data,lats,lons,):

    mean = np.mean(data, axis=0)
    sig_mean = np.std(data, axis=0, ddof=1) / np.sqrt(len(data))   # = standard error or uncertainty on the mean (ddof=1 to get corrected std)

    meanpluserr = unumpy.uarray(mean, sig_mean) 
    # uncertainties package handles error propagation
    # mean1(+/-sig1) - mean2(+/-sig2) = mean1 - mean2 +/- sqrt(sig1**2+sig2**2)  and also takes correlation btw variables into account: sqrt(sig1**2 + sig2**2 + 2 * cov(mean1,mean2)) 
    
    #high_lats = [44,52,61,52]
    #high_lons = [-10,-30,2,2]
    #low_lats = [63,33,36,43]
    #low_lons = [-48,0,-6,-43]

    NAOH_lat_in, NAOH_lon_in = np.absolute(lats-44).argmin(), np.absolute(lons-(-10)).argmin()
    NAOL_lat_in, NAOL_lon_in = np.absolute(lats-63).argmin(), np.absolute(lons-(-48)).argmin()
    NAO_grad = meanpluserr[NAOH_lat_in, NAOH_lon_in] - meanpluserr[NAOL_lat_in, NAOL_lon_in]
    
    # test:
    #test_errorprop1 = np.sqrt(sig_mean[NAOH_lat_in, NAOH_lon_in]**2 + sig_mean[NAOL_lat_in, NAOL_lon_in]**2)
    #test_errorprop2 = np.sqrt(sig_mean[NAOH_lat_in, NAOH_lon_in]**2 + sig_mean[NAOL_lat_in, NAOL_lon_in]**2 + 2 * np.cov(mean[NAOH_lat_in, NAOH_lon_in],mean[NAOL_lat_in, NAOL_lon_in]))
    #print(NAO_grad, test_errorprop1, test_errorprop1)
    # all 3 are identical! 421.61474609375+/-9.57283594402852, 9.57283594402852 9.57283594402852

    EAH_lat_in, EAH_lon_in = np.absolute(lats-52).argmin(), np.absolute(lons-(-30)).argmin()
    EAL_lat_in, EAL_lon_in = np.absolute(lats-33).argmin(), np.absolute(lons-0).argmin()
    EA_grad = meanpluserr[EAH_lat_in, EAH_lon_in] - meanpluserr[EAL_lat_in, EAL_lon_in]

    NEH_lat_in, NEH_lon_in = np.absolute(lats-61).argmin(), np.absolute(lons-2).argmin()
    NEL_lat_in, NEL_lon_in = np.absolute(lats-36).argmin(), np.absolute(lons-(-6)).argmin()
    NE_grad = meanpluserr[NEH_lat_in, NEH_lon_in] - meanpluserr[NEL_lat_in, NEL_lon_in]

    WEH_lat_in, WEH_lon_in = np.absolute(lats-52).argmin(), np.absolute(lons-2).argmin()
    WEL_lat_in, WEL_lon_in = np.absolute(lats-43).argmin(), np.absolute(lons-(-43)).argmin()
    WE_grad = meanpluserr[WEH_lat_in, WEH_lon_in] - meanpluserr[WEL_lat_in, WEL_lon_in]



    return [NAO_grad, EA_grad, NE_grad, WE_grad]



def test_gradient_overlap(obs_grads,model_grads):
    # tests if 2 values are within 95% confidence interval (= 2 standard deviations, assuming normal distribution)

    overlap1 = np.zeros(4)
    overlap = np.zeros(4)

    for i in range(4):

        # test 1 (see if within range of 2*sigma):
        a = (model_grads[i].n + 2* model_grads[i].s, model_grads[i].n - 2* model_grads[i].s)
        b = (obs_grads[i].n + 2* obs_grads[i].s, obs_grads[i].n - 2* obs_grads[i].s)
        a, b = np.abs(a), np.abs(b)
        if (a[0]>a[1]):
            a = (a[1],a[0]) 
        if (b[0]>b[1]):
            b = (b[1],b[0])
        overlap1[i] = a[0] <= b[0] <= a[1] or a[0] <= b[1] <= a[1]

        # test 2 (2 sample z-test):
        x1 = model_grads[i].n
        sig_x1 = model_grads[i].s
        x2 = obs_grads[i].n
        sig_x2 = obs_grads[i].s  

        z = (x1-x2)/np.sqrt(sig_x1**2 + sig_x2**2)
        #print(f" {x1:.2f} is {z:.4f} sigma away from {x2:.2f}.")
        p_val = stats.norm.sf(abs(z))*2
        #print(p_val, p_val>0.05)
        overlap[i] = p_val>0.05   # if p_val > 5% = same as within range of 2 sigma

        # plot:
        #plt.errorbar(1, obs_grads[i].n, obs_grads[i].s*2,fmt='.')
        #plt.errorbar(2, model_grads[i].n, model_grads[i].s*2,fmt='.')

    if not (overlap1 == overlap).all():
        print('Disagreement!')
        print(overlap1, overlap)

    return overlap



def plot_EOF1_plus_gradlocs(eofs,n_colorlevels,lons,lats,varfrac,standard_title,order):
    #(eofs,n_colorlevels,lons,lats,varfrac,analysis_type,month,season,season_mean,timeframe,model,variant,running_mean,extremes,extr_type,nmonths,extreme_region_name,saveEOFas,saveEOF):
    #plt.style.use('ggplot')
    plt.style.use('default') 
    
    # define color levels to use for all 4 contour plots:
    cmap = mpl.cm.coolwarm #viridis
    maxbound = max(np.abs(np.amin(eofs)), np.abs(np.amax(eofs))) 
    maxbound = float("{:.0e}".format(maxbound))      # round to the first nonzero number
    #maxbound = 0.008 # 0.008 to make it comparable with Simons
    cbarticks = np.linspace(-maxbound,maxbound,n_colorlevels) 
    colorlevels = np.linspace(-maxbound,maxbound,n_colorlevels) 
    #colorlevels = np.linspace(-0.01,0.01,n_colorlevels) 


    high_lats = [44,52,61,52]
    high_lons = [-10,-30,2,2]
    low_lats = [63,33,36,43]
    low_lons = [-48,0,-6,-43]
       
    fig, ax = plt.subplots(1, 1, figsize=(6,4), subplot_kw={'projection': ccrs.EquidistantConic(central_longitude=-20.0)})

    i = 0
    #ax = axs[0, 0]
    # if I want to add a thin outline of the EOF patterns:
    line_c = ax.contour(lons, lats, eofs[i,:,:], levels=colorlevels, colors=['grey'], linewidths=0.5 ,transform=ccrs.PlateCarree())
    pcm = ax.contourf(lons, lats, eofs[i,:,:], levels=colorlevels, cmap=cmap, transform=ccrs.PlateCarree(), extend="both") 

    h_lat, h_lon = np.absolute(lats-(high_lats[i])).argmin(), np.absolute(lons-(high_lons[i])).argmin()
    l_lat, l_lon = np.absolute(lats-(low_lats[i])).argmin(), np.absolute(lons-(low_lons[i])).argmin()
    ax.plot([lons[h_lon],lons[l_lon]], [lats[h_lat],lats[l_lat]], color='black', marker='o', transform=ccrs.Geodetic())
    
    ax.coastlines()
    ax.set_title(f'NAO centers of action (ERA5 EOF {order[i]+1})', loc='left', fontsize='xx-large');
    #ax.set_title(f'{varfrac[i]*100:.1f} %', loc='right', fontsize='xx-large');


    #plt.suptitle('')

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    #gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 15, 'color': 'gray'}
    gl.ylabel_style = {'size': 15, 'color': 'gray'}

    #cb_ax = fig.add_axes([0.91, 0.2, 0.02, 0.6])  
    cb1 = fig.colorbar(pcm, ticks = cbarticks) #(pcm, cax=cb_ax, ticks = cbarticks)
    #cb1.set_label(f'Anomalies of geopotential height at 500 hPa [m]')   # these are not correct

    '''
    # plot mean flow field:
    #if (regdata_type=='Geopotential height'):
    colorlevels = np.linspace(5000,5900,10)
    contourlevels = np.linspace(-100,100,11)
    contourlevels_masked = np.linspace(-20,20,11)
    cbarlab = 'Difference in geopotential height at 500 hPa [m]'
    
    ax = axs[0, 1]
    CF = ax.contourf(lons, lats, model_mean, levels=colorlevels, cmap=cmap, transform=ccrs.PlateCarree())
    ax.coastlines()
    cbar = cbar = fig_mean.colorbar(CF)
    cbar.set_label(cbarlab) 
    ax.clabel(CS, CS.levels, inline=True, fontsize=10) 
    '''


    plt.subplots_adjust(left=0.12,bottom=0.1, right=0.98, top=0.85, wspace=0.2, hspace=0.35)
    #plt.subplots_adjust(left=0.09,bottom=0.1, right=0.8, top=0.85, wspace=0.2, hspace=0.35)



    #fig.suptitle(f'EOF 1 of geopotential height at 500 hPa', y=0.98, fontsize='xx-large')
    #plt.text(.5, 0.93, standard_title, transform=fig.transFigure, fontsize='x-large', horizontalalignment='center')
    #plt.text(.5, 0.9, f'Explained variance: {np.sum(varfrac)*100:.1f} %', transform=fig.transFigure, fontsize='x-large', horizontalalignment='center')


    fig.savefig(f'output/mean_diff/NAOgradient_map.pdf')

    # plot mean field + locations next to it

    # and print gradient + evaluation of comparison to ERA5 immediately in the figure!

    return fig 



def plot_EOFs_plus_gradlocs(eofs,n_colorlevels,lons,lats,varfrac,standard_title,order):
    #(eofs,n_colorlevels,lons,lats,varfrac,analysis_type,month,season,season_mean,timeframe,model,variant,running_mean,extremes,extr_type,nmonths,extreme_region_name,saveEOFas,saveEOF):
    #plt.style.use('ggplot')
    plt.style.use('default') 
    
    # define color levels to use for all 4 contour plots:
    cmap = mpl.cm.coolwarm #viridis
    maxbound = max(np.abs(np.amin(eofs)), np.abs(np.amax(eofs))) 
    maxbound = float("{:.0e}".format(maxbound))      # round to the first nonzero number
    #maxbound = 0.008 # 0.008 to make it comparable with Simons
    cbarticks = np.linspace(-maxbound,maxbound,n_colorlevels) 
    colorlevels = np.linspace(-maxbound,maxbound,n_colorlevels) 
    #colorlevels = np.linspace(-0.01,0.01,n_colorlevels) 


    high_lats = [44,52,61,52]
    high_lons = [-10,-30,2,2]
    low_lats = [63,33,36,43]
    low_lons = [-48,0,-6,-43]
       
    fig, axs = plt.subplots(2, 2, figsize=(12,10), subplot_kw={'projection': ccrs.EquidistantConic(central_longitude=-20.0)})

    i = 0
    for row in range(2):
        for col in range(2):
            ax = axs[row, col]
            # if I want to add a thin outline of the EOF patterns:
            line_c = ax.contour(lons, lats, eofs[i,:,:], levels=colorlevels, colors=['grey'], linewidths=0.5 ,transform=ccrs.PlateCarree())
            pcm = ax.contourf(lons, lats, eofs[i,:,:], levels=colorlevels, cmap=cmap, transform=ccrs.PlateCarree(), extend="both") 

            h_lat, h_lon = np.absolute(lats-(high_lats[i])).argmin(), np.absolute(lons-(high_lons[i])).argmin()
            l_lat, l_lon = np.absolute(lats-(low_lats[i])).argmin(), np.absolute(lons-(low_lons[i])).argmin()
            #print(high_lats[i],high_lons[i],low_lats[i],low_lons[i])
            #ax.plot(lons[h_lon], lats[h_lat], '.', color='blue', transform=ccrs.PlateCarree())
            #ax.plot(lons[l_lon], lats[l_lat], '.', color='red', transform=ccrs.PlateCarree())

            #ax.plot([lons[h_lon],lons[l_lon]], [lats[h_lat],lats[l_lat]], color='black', marker='.', transform=ccrs.PlateCarree())
            ax.plot([lons[h_lon],lons[l_lon]], [lats[h_lat],lats[l_lat]], color='black', marker='o', transform=ccrs.Geodetic())
            
            #ax.plot([lats[h_lat],lons[h_lon]], [lats[l_lat],lons[l_lon]], color='black', marker='o')

            ax.coastlines()
            ax.set_title(f'EOF {order[i]+1}', loc='left', fontsize='xx-large');
            ax.set_title(f'{varfrac[i]*100:.1f} %', loc='right', fontsize='xx-large');
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
    cb1 = fig.colorbar(pcm, cax=cb_ax, ticks = cbarticks)
    plt.subplots_adjust(left=0.09,bottom=0.1, right=0.89, top=0.85, wspace=0.2, hspace=0.35)
    #cb1.set_label(f'Anomalies of geopotential height at 500 hPa [m]')   # these are not correct

    #fig.suptitle(f'4 leading EOFs of geopotential height at 500 hPa - Explained variance: {np.sum(varfrac)*100:.1f} %', y=0.98, fontsize='xx-large')
    #plt.text(.5, 0.91, standard_title, transform=fig.transFigure, fontsize='x-large', horizontalalignment='center')
    fig.suptitle(f'4 leading EOFs of geopotential height at 500 hPa', y=0.98, fontsize='xx-large')
    plt.text(.5, 0.93, standard_title, transform=fig.transFigure, fontsize='x-large', horizontalalignment='center')
    plt.text(.5, 0.9, f'Explained variance: {np.sum(varfrac)*100:.1f} %', transform=fig.transFigure, fontsize='x-large', horizontalalignment='center')

    return fig 




def plot_gradient_ranking(NAO_grad, models):

    cell_text=[]
    colors=[]
    for im in range(len(models)):
        #if (im==0):
         #   continue

        index = np.argsort(NAO_grad[0,:])[::-1][im];                       
        cell_text.append([models[index], np.round(NAO_grad[0,index],2)]) ###, np.round(weighted_score[1,index],3)])    
        colors.append('green' if NAO_grad[1,index] else 'white')

    colors = np.array([colors,colors]).T                   

    fig,ax=plt.subplots(figsize=(3,5))
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    ax.table(cellText=cell_text, cellLoc='left', colLabels=['Model',f'Gradient [m]'],loc='center',bbox = [0, 0, 1, 1], cellColours=colors)
    #ax.table(cellText=cell_text1, cellLoc='left', colLabels=['Model','Score (high corr)'],loc='right',bbox = [0.9, 0, 1.5, 1])
    fig.suptitle(f'NAO gradient', fontsize='small')
    fig.tight_layout()
    fig.savefig(f'output/mean_diff/NAO_gradient.pdf')
    fig.savefig(f'output/plots/NAO_gradient.pdf')





def plot_mean_diff_ranking(mean_diff_arr, regvar, models, regdata_type, regdatatypes_units):

    

    cell_text=[]
    cell_text1=[]
    for im in range(len(models)):
        #if (im==0):
         #   continue

        index = np.argsort(np.abs(mean_diff_arr[0,:]))[im];
        index1 = np.argsort(np.abs(mean_diff_arr[1,:]))[im];
        #cell_text.append([models[index], np.round(weighted_score[0,index],3),models[index2], np.round(weighted_score[1,index2],3)]) 
        if not (models[index]=='ERA5'):                          
            cell_text.append([models[index], np.round(mean_diff_arr[0,index],2)]) ###, np.round(weighted_score[1,index],3)])                               
        if not (models[index1]=='ERA5'):
            cell_text1.append([models[index1], np.round(mean_diff_arr[1,index1],2)]) 

    fig,ax=plt.subplots(figsize=(3,5))
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    ax.table(cellText=cell_text, cellLoc='left', colLabels=['Model',f'Difference {regdatatypes_units[regdata_type]}'],loc='center',bbox = [0, 0, 1, 1])
    #ax.table(cellText=cell_text1, cellLoc='left', colLabels=['Model','Score (high corr)'],loc='right',bbox = [0.9, 0, 1.5, 1])
    fig.suptitle(f'{regdata_type} mean field difference', fontsize='small')
    fig.tight_layout()
    fig.savefig(f'output/mean_diff/{regvar}_meandiff.pdf')


    fig,ax=plt.subplots(figsize=(3,5))
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    ax.table(cellText=cell_text1, cellLoc='left', colLabels=['Model',f'Difference {regdatatypes_units[regdata_type]}'],loc='center',bbox = [0, 0, 1, 1])
    if (regdata_type=='Geopotential height'):
        fig.suptitle(f'Z500 mean field difference', fontsize='small')
    else:
        fig.suptitle(f'{regdata_type} mean field difference', fontsize='small')
    fig.tight_layout()
    fig.savefig(f'output/mean_diff/{regvar}_meandiff_significant.pdf')
    fig.savefig(f'output/plots/{regvar}_meandiff_significant.pdf')













