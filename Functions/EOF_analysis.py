from netCDF4 import Dataset, num2date
import matplotlib as mpl
from matplotlib import gridspec,cm, rcParams, pyplot as plt
import numpy as np
import math
from sklearn.linear_model import LinearRegression
from eofs.standard import Eof
from os.path import exists
import sys  
from cartopy import config, crs as ccrs
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER





def EOF_analysis(data, lats, lons, n_eofs, analysis_type, season, season_mean, timeframe, model, variant, ID,
    running_mean, extremes, startyear, endyear, standard_title, 
    savedata, saveEOFdataas, loaddata, 
    savemean, savemeanas, saveEOF, saveEOFas, norm_PC, savePC, savePCas, n_colorlevels, plot_mean_field=False,order=[0,1,2,3]):

    # -------------------------------------------------------------------
    if plot_mean_field:
        plot_mean_field_func(data,lons,lats,season,timeframe,model,variant,standard_title,savemean,savemeanas)


    # -------------------------------------------------------------------    
    if loaddata:
        eofs = np.load(f'data/{model}/eofs_{saveEOFdataas}.npy')
        pcs = np.load(f'data/{model}/pcs_{saveEOFdataas}.npy')
        varfrac = np.load(f'data/{model}/varfrac_{saveEOFdataas}.npy')

        plot_eigenvalue_spec(data,lats,model,ID,n=40)


    else:
        # -------------------------------------------------------------------
        # EOF analysis:
        wgts = np.sqrt(np.abs(np.cos(np.radians(lats))))[..., np.newaxis]
        solver = Eof(data, weights = wgts, center=True)

        n = n_eofs
        eofs = solver.eofs(neofs = n, eofscaling = 0)   
            # 0 : Un-scaled EOFs (default).
            # 1 : EOFs are divided by the square-root of their eigenvalues.
            # 2 : EOFs are multiplied by the square-root of their eigenvalues.

        pcs = solver.pcs(pcscaling=1, npcs = n)        
            # pcscaling=1 : PCs are scaled to unit variance (divided by the square-root of their eigenvalue).
            # same effect as dividing by standard deviation
            

        if norm_PC:
            # Normalize PCs by the sum per month 
            # => so that in every month the 4 PCs show the relative distribution and add up to 1
            # => don't do this for the analysis, it will mess up the regression values!
            sums = np.zeros(len(pcs))
            pcs_norm = np.zeros([len(pcs),n])
            for imonth in range(len(pcs)):
                sums[imonth] = np.sum(abs(pcs[imonth,:]))
                pcs_norm[imonth] = pcs[imonth,:] / sums[imonth]
            pcs = pcs_norm  
            
        varfrac = solver.varianceFraction(n)
        eigenvalues = solver.eigenvalues(n)
        errors = solver.northTest(n)
        errors_scaled = solver.northTest(n, vfscaled=True)

        # eigenvalues/np.sum(eigenvalues) == errors/np.sum(errors) !
        # but both not identical to neither varfrac nor errors_scaled!
        # IT IS IDENTICAL if including all eigenvalues (i.e. n is very large):  eigenvalues/sum(all_eigenvalues) == varfrac == errors/np.sum(all errors)


        # Plot eigenvalue spectrum:
        plot_eigenvalue_spec(data,lats,model,ID,n=40)


        # rearrange order of EOFs to make them comparable:
        eofs = np.array([eofs[order[0],:], eofs[order[1],:], eofs[order[2],:], eofs[order[3],:]])
        pcs = np.transpose(np.array([pcs[:,order[0]], pcs[:,order[1]], pcs[:,order[2]], pcs[:,order[3]]]))
        varfrac = np.array([varfrac[order[0]], varfrac[order[1]], varfrac[order[2]], varfrac[order[3]]])



        # flip EOFs & PCs: 
        
        # PC1/NAO: criteria: pressure over iceland < azores 
        ice_lat_in, ice_lon_in = np.absolute(lats-65).argmin(), np.absolute(lons-(-22.7)).argmin()
        azo_lat_in, azo_lon_in = np.absolute(lats-37.7).argmin(), np.absolute(lons-(-25.7)).argmin()
        if (eofs[0,ice_lat_in, ice_lon_in] > eofs[0,azo_lat_in, azo_lon_in]):
            print('flipping PC 1 / NAO pattern')
            eofs[0,:] = eofs[0,:] * -1   
            pcs[:,0] = pcs[:,0] * -1     

        # PC2/EA: criteria: pressure over N NA > S NA
        N_NA_lat_in, N_NA_lon_in = np.absolute(lats-53).argmin(), np.absolute(lons-(-29)).argmin()
        S_NA_lat_in, S_NA_lon_in = np.absolute(lats-23).argmin(), np.absolute(lons-(-42)).argmin()
        if (eofs[1,N_NA_lat_in, N_NA_lon_in] < eofs[1,S_NA_lat_in, S_NA_lon_in]):
            print('flipping PC 2 / EA pattern')
            eofs[1,:] = eofs[1,:] * -1   
            pcs[:,1] = pcs[:,1] * -1     

        # PC3/NE: criteria: pressure over NE (southern Norway) > SE (Gibraltar)
        NE_lat_in, NE_lon_in = np.absolute(lats-57).argmin(), np.absolute(lons-5).argmin()
        SE_lat_in, SE_lon_in = np.absolute(lats-35).argmin(), np.absolute(lons-(-5)).argmin()
        if (eofs[2,NE_lat_in, NE_lon_in] < eofs[2,SE_lat_in, SE_lon_in]):
            print('flipping PC 3 / NE pattern')
            eofs[2,:] = eofs[2,:] * -1   
            pcs[:,2] = pcs[:,2] * -1     
        
        # PC4/WE: criteria: pressure over WE (France) > W Atlantic
        WE_lat_in, WE_lon_in = np.absolute(lats-45).argmin(), np.absolute(lons-0).argmin()
        WA_lat_in, WA_lon_in = np.absolute(lats-45).argmin(), np.absolute(lons-(-50)).argmin()
        if (eofs[3,WE_lat_in, WE_lon_in] < eofs[3,WA_lat_in, WA_lon_in]):
            print('flipping PC 4 / WE pattern')
            eofs[3,:] = eofs[3,:] * -1   
            pcs[:,3] = pcs[:,3] * -1     


    

    # -------------------------------------------------------------------
    # save data:
    if savedata:
        try:
            np.save(f'data/{model}/meanzg_{saveEOFdataas}', np.mean(data, axis=0))
        except NotImplementedError:                                       # in case it is a masked array
            np.save(f'data/{model}/meanzg_{saveEOFdataas}', np.mean(data, axis=0).data)
        try:
            np.save(f'data/{model}/eofs_{saveEOFdataas}', eofs)
        except NotImplementedError:                                       # in case it is a masked array
            np.save(f'data/{model}/eofs_{saveEOFdataas}', eofs.data)
            #np.savez_compressed(f'data/{model}/eofs_{saveEOFdataas}', data=eofs.data, mask=eofs.mask)
        np.save(f'data/{model}/pcs_{saveEOFdataas}', pcs)
        np.save(f'data/{model}/varfrac_{saveEOFdataas}', varfrac)


    # -------------------------------------------------------------------
    # plot EOFs:
    plot_EOFs(eofs,n_colorlevels,lons,lats,varfrac,standard_title,saveEOFas,saveEOF,order)
    #plot_EOFs(eofs,n_colorlevels,lons,lats,varfrac,analysis_type,month,season,season_mean,timeframe,model,variant,running_mean,extremes,extr_type,nmonths,extreme_region_name,saveEOFas,saveEOF)
    

        
    # -------------------------------------------------------------------
    # plot PCs:
    #if (running_mean==1) & (not extremes):  # for now easier to plot only if using standard data set
    #if not extremes:
    plot_PCs(pcs,running_mean,analysis_type,model,season,season_mean,startyear,endyear,extremes,standard_title,savePC,savePCas,tickmdist=20,order=order)



    return eofs, pcs, varfrac
        
    



def plot_mean_field_func(data,lons,lats,season,timeframe,model,variant,standard_title,savemean,savemeanas):
    #plt.style.use('ggplot')
    plt.style.use('default')
    cmap = mpl.cm.coolwarm 
    col_mean = np.mean(data, axis=0)
    colorlevels = np.linspace(5000,5900,10) 

    fig_mean, ax = plt.subplots(figsize=(7,5), subplot_kw={'projection': ccrs.EquidistantConic(central_longitude=-20.0)})
    plt.contourf(lons, lats, col_mean, levels=colorlevels, cmap=cmap, transform=ccrs.PlateCarree())
    ax.coastlines()
    cbar = plt.colorbar()
    cbar.set_label(f'geopotential height at 500 hPa [m] ')  
    
    #ax.set(title=f'Mean field {season} {timeframe} \n ({model} {variant})')
    fig_mean.suptitle(f'Mean field of geopotential height at 500 hPa', fontsize='x-large')
    plt.text(.5, 0.9, standard_title, transform=fig_mean.transFigure, fontsize='large', horizontalalignment='center')

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    #gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 15, 'color': 'gray'}
    gl.ylabel_style = {'size': 15, 'color': 'gray'}

    if savemean:
        fig_mean.savefig(savemeanas)



def plot_EOFs(eofs,n_colorlevels,lons,lats,varfrac,standard_title,saveEOFas,saveEOF,order):
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
   
    fig, axs = plt.subplots(2, 2, figsize=(11,8), subplot_kw={'projection': ccrs.EquidistantConic(central_longitude=-20.0)})

    i = 0
    for row in range(2):
        for col in range(2):
            ax = axs[row, col]
            # if I want to add a thin outline of the EOF patterns:
            line_c = ax.contour(lons, lats, eofs[i,:,:], levels=colorlevels, colors=['grey'], linewidths=0.5 ,transform=ccrs.PlateCarree())
            pcm = ax.contourf(lons, lats, eofs[i,:,:], levels=colorlevels, cmap=cmap, transform=ccrs.PlateCarree(), extend="both")          
            ax.coastlines()
            ax.set_title(f'EOF {order[i]+1}', loc='left', fontsize='xx-large');
            ax.set_title(f'{varfrac[i]*100:.1f} %', loc='right', fontsize='xx-large');
            #ax.set_facecolor('white')
            i += 1

            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
            gl.left_labels = False
            gl.right_labels = False
            gl.top_labels = False
            #gl.xlabels_top = False
            #gl.ylabels_top = False
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            gl.xlabel_style = {'size': 12, 'color': 'gray'}
            gl.ylabel_style = {'size': 12, 'color': 'gray'}

    cb_ax = fig.add_axes([0.88, 0.2, 0.02, 0.6])  
    cb1 = fig.colorbar(pcm, cax=cb_ax, ticks = cbarticks)
    plt.subplots_adjust(left=0.01,bottom=0.05, right=0.89, top=0.85, wspace=0, hspace=0.3)
    #cb1.set_label(f'Anomalies of geopotential height at 500 hPa [m]')   # these are not correct

    #fig.suptitle(f'4 leading EOFs of geopotential height at 500 hPa - Explained variance: {np.sum(varfrac)*100:.1f} %', y=0.98, fontsize='xx-large')
    #plt.text(.5, 0.91, standard_title, transform=fig.transFigure, fontsize='x-large', horizontalalignment='center')
    fig.suptitle(f'4 leading EOFs of geopotential height at 500 hPa', y=0.98, fontsize='xx-large')
    plt.text(.5, 0.93, standard_title, transform=fig.transFigure, fontsize='x-large', horizontalalignment='center')
    plt.text(.5, 0.9, f'Explained variance: {np.sum(varfrac)*100:.1f} %', transform=fig.transFigure, fontsize='x-large', horizontalalignment='center')

    if saveEOF:
        fig.savefig(saveEOFas)

    


def plot_PCs(pcs,running_mean,analysis_type,model,season,season_mean,startyear,endyear,extremes,standard_title,savePC,savePCas,tickmdist,order):

    # To DO:
    # change tickmark years from float (1992.0) to int
    # add title

    plt.style.use('default')
    fig_PC, axs = plt.subplots(4, 1, figsize=(6,9), sharex=True)
    tickmdist = 10  # one tick mark every 10 years

    for row in range(4):
        ax = axs[row]
        clrs = ['red' if (x > 0) else 'blue' for x in pcs[:,row]]
        if (running_mean>1):
            rmyears=np.floor(running_mean/2)
          #  myxaxis = np.arange(len(pcs[:,row])-rmyears-rmyears)  # shorten axis by 2 years on each side if running mean = 5
        #else:
         #   myxaxis = np.arange(len(pcs[:,row]))
        myxaxis = np.arange(len(pcs[:,row]))
        ax.bar(myxaxis, pcs[:,row], color=clrs)
        
        if (analysis_type=='allmonths'):
            myticks = np.arange(0,len(myxaxis),tickmdist*12);     
        elif (analysis_type=='monthly'):
            myticks = np.arange(0,len(myxaxis),tickmdist);      
        elif (analysis_type=='seasonal'):
            myticks = np.arange(0,len(myxaxis),tickmdist*3);
            if (season == 'NDJFM'): 
                myticks = np.arange(0,len(myxaxis),tickmdist*5);     
            if season_mean:
                myticks = np.arange(0,len(myxaxis),tickmdist);
        
        if (running_mean>1):
            mylabels = np.arange(startyear+rmyears,endyear-rmyears,tickmdist)
        else:
            mylabels = np.arange(startyear,endyear+1,tickmdist) 
        
        if not extremes:
            if (row==3):
                ax.set_xticks(ticks=myticks); 
                ax.set_xticklabels(labels=mylabels);
                ax.set_xlabel('Model Years');
        
        myylim=3 #4
        ax.set_ylim(- myylim, myylim)
        if (max(np.abs(np.amin(pcs)), np.abs(np.amax(pcs))) > myylim):
            print('!! ATTENTION: PC loading exceeds the set y-axis range !!')   
        
        ax.margins(x=0.005, y=0)
        ax.set_title(f'PC {order[row]+1}', loc='left', fontsize='xx-large');
        ax.set_ylabel('Standard deviations')
    
    fig_PC.suptitle(standard_title)

    fig_PC.tight_layout()
    if savePC:
        fig_PC.savefig(savePCas)




def plot_eigenvalue_spec(data,lats,model,ID,n=40):
    
    wgts = np.sqrt(np.abs(np.cos(np.radians(lats))))[..., np.newaxis]
    solver = Eof(data, weights = wgts, center=True)
        
    varfrac = solver.varianceFraction(n)
    eigenvalues = solver.eigenvalues(n)
    errors = solver.northTest(n)
    errors_scaled = solver.northTest(n, vfscaled=True)


    myxaxis = np.arange(len(varfrac))
    fig = plt.figure(figsize=(7,5))
    ax = plt.axes()
    ax.errorbar(myxaxis, varfrac*100, yerr=errors_scaled*100, fmt='o', color='black', ms=3, ecolor='black', elinewidth=1, capsize=2)
    ax.set(title=f'Eigenvalue spectrum {model}', xlabel='Rank', ylabel='Eigenvalue [%]');
    fig.savefig(f'output/{model}/eigenvalues_{ID}.pdf')

    np.save(f'data/{model}/varfrac_{n}_{ID}', varfrac)
    np.save(f'data/{model}/eigenvals_{n}_{ID}', eigenvalues)
    np.save(f'data/{model}/errors_{n}_{ID}', errors)
    np.save(f'data/{model}/errors_scaled_{n}_{ID}', errors_scaled)









