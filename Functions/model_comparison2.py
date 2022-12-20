import numpy as np
import pandas as pd 
from matplotlib import rcParams, pyplot as plt
from WeightedCorr import WeightedCorr
from statsmodels.stats.weightstats import DescrStatsW
import math
import skill_metrics as sm


def taylor_metrics(ccoef,sdev,mse,rmse,obs_regdata_mean,model_regdata_mean,obs_eofs,model_eofs,obs_coefficients,model_coefficients,obs_corrcoeffs,model_corrcoeffs,corr_threshold,area_weights,modeli,loaddata,models):


	# mean field:
    if loaddata:
        try:
            refdat = obs_regdata_mean.flat
        except AttributeError:
            refdat = obs_regdata_mean.data.flat        
        try:
            modeldat = model_regdata_mean.flat
        except AttributeError:
            modeldat = model_regdata_mean.data.flat
    else:
        refdat = obs_regdata_mean.data.flat
        modeldat = model_regdata_mean.data.flat
        
    # area-weighted comparison metrics:
    ccoef[0,modeli] = WeightedCorr(x=pd.Series(refdat), y=pd.Series(modeldat), w=pd.Series(area_weights.flat))(method='pearson') 
    sdev[0,modeli] = DescrStatsW(modeldat, weights=area_weights.flat, ddof=0).std
    # normalized std:
    #sdev[0,modeli] = DescrStatsW(modeldat, weights=area_weights.flat, ddof=0).std/DescrStatsW(refdat, weights=area_weights.flat, ddof=0).std
    diff = np.square(np.subtract(refdat,modeldat))
    #mse[0,modeli] = np.sum(diff*area_weights.flat)/np.sum(area_weights.flat)
    #rmse[0,modeli] = math.sqrt(mse[0,modeli])
    # centered RMSE:
    modelav = np.average(modeldat, weights=area_weights.flat)
    refav = np.average(refdat, weights=area_weights.flat)
    diff = np.square(np.subtract(np.subtract(refdat,refav),np.subtract(modeldat,modelav)))
    mse[0,modeli] = np.sum(diff*area_weights.flat)/np.sum(area_weights.flat)
    rmse[0,modeli] = math.sqrt(mse[0,modeli])
    
    # old competrics (not weighted):
    #ccoef[0,modeli] = np.corrcoef(refdat, modeldat)[0,1]
    #sdev[0,modeli] = np.std(modeldat)
    #mse[0,modeli] = np.square(np.subtract(refdat,modeldat)).mean() 
    #rmse[0,modeli] = math.sqrt(mse[0,modeli])

    #print(ccoef[0,modeli],sdev[0,modeli],mse[0,modeli],rmse[0,modeli])


    # EOFs:
    #'''
    for ieof in range(len(model_eofs)):
        try:
            refdat = obs_eofs[ieof,:].flat
        except AttributeError:
            refdat = obs_eofs[ieof,:].data.flat
        try:
            modeldat = model_eofs[ieof,:].flat
        except AttributeError:
            modeldat = model_eofs[ieof,:].data.flat
        ccoef[ieof+1,modeli] = WeightedCorr(x=pd.Series(refdat), y=pd.Series(modeldat), w=pd.Series(area_weights.flat))(method='pearson')
        sdev[ieof+1,modeli] = DescrStatsW(modeldat, weights=area_weights.flat, ddof=0).std
        # normalized:
        #sdev[ieof+1,modeli] = DescrStatsW(modeldat, weights=area_weights.flat, ddof=0).std/DescrStatsW(refdat, weights=area_weights.flat, ddof=0).std
        diff = np.square(np.subtract(refdat,modeldat))
        #mse[ieof+1,modeli] = np.sum(diff*area_weights.flat)/np.sum(area_weights.flat)
        #rmse[ieof+1,modeli] = math.sqrt(mse[ieof+1,modeli])
        modelav = np.average(modeldat, weights=area_weights.flat)
        refav = np.average(refdat, weights=area_weights.flat)
        diff = np.square(np.subtract(np.subtract(refdat,refav),np.subtract(modeldat,modelav)))
        mse[ieof+1,modeli] = np.sum(diff*area_weights.flat)/np.sum(area_weights.flat)
        rmse[ieof+1,modeli] = math.sqrt(mse[ieof+1,modeli])

        print(ccoef[ieof+1,modeli],sdev[ieof+1,modeli],mse[ieof+1,modeli],rmse[ieof+1,modeli])
        #'''

    # PCR:
    #'''
    for ieof in range(len(model_eofs)):
        try:
            refdat = obs_coefficients[ieof,:].flat
        except AttributeError:
            refdat = obs_coefficients[ieof,:].data.flat
        try:
            modeldat = model_coefficients[ieof,:].flat
        except AttributeError:
            modeldat = model_coefficients[ieof,:].data.flat
        #new_weights = area_weights*model_corrcoeffs[ieof]
        ccoef[ieof+5,modeli] = WeightedCorr(x=pd.Series(refdat), y=pd.Series(modeldat), w=pd.Series(area_weights.flat))(method='pearson')
        sdev[ieof+5,modeli] = DescrStatsW(modeldat, weights=area_weights.flat, ddof=0).std
        sdev[ieof+5,modeli] = DescrStatsW(modeldat, weights=area_weights.flat, ddof=0).std/DescrStatsW(refdat, weights=area_weights.flat, ddof=0).std
        diff = np.square(np.subtract(refdat,modeldat))
        #mse[ieof+5,modeli] = np.sum(diff*area_weights.flat)/np.sum(area_weights.flat)
        #rmse[ieof+5,modeli] = math.sqrt(mse[ieof+5,modeli])
        modelav = np.average(modeldat, weights=area_weights.flat)
        refav = np.average(refdat, weights=area_weights.flat)
        diff = np.square(np.subtract(np.subtract(refdat,refav),np.subtract(modeldat,modelav)))
        mse[ieof+5,modeli] = np.sum(diff*area_weights.flat)/np.sum(area_weights.flat)
        rmse[ieof+5,modeli] = math.sqrt(mse[ieof+5,modeli])

        print(ccoef[ieof+5,modeli],sdev[ieof+5,modeli],mse[ieof+5,modeli],rmse[ieof+5,modeli])
    
    #'''
       



    # high corr PCR:
    obs_high_corr_reg_coefficients = np.where(np.logical_or(obs_corrcoeffs > corr_threshold, obs_corrcoeffs<-corr_threshold), obs_coefficients, np.nan)
    model_high_corr_reg_coefficients = np.where(np.logical_or(model_corrcoeffs > corr_threshold, model_corrcoeffs<-corr_threshold), model_coefficients, np.nan)

        
    for ieof in range(len(model_eofs)):
        '''
        refdat = obs_high_corr_reg_coefficients[ieof,:].flat
        modeldat = model_high_corr_reg_coefficients[ieof,:].flat
        
        comb_mask = np.ma.mask_or(np.ma.masked_invalid(refdat).mask, np.ma.masked_invalid(modeldat).mask)
        weights_masked = np.ma.masked_array(area_weights.flat, mask=comb_mask)
        
        ccoef[ieof+9,modeli] = WeightedCorr(x=pd.Series(refdat), y=pd.Series(modeldat), w=pd.Series(area_weights.flat))(method='pearson')
        waverage = np.ma.average(np.ma.masked_invalid(modeldat), weights=np.ma.masked_array(area_weights.flat, mask=np.ma.masked_invalid(modeldat).mask))  
        sdev[ieof+9,modeli] = math.sqrt(np.ma.average((np.ma.masked_invalid(modeldat)-waverage)**2, weights=weights_masked))
        diff = np.square(np.ma.masked_invalid(np.subtract(refdat,modeldat)))
        #mse[ieof+9,modeli] = np.sum(diff*weights_masked)/np.sum(weights_masked)
        #rmse[ieof+9,modeli] = math.sqrt(mse[ieof+9,modeli])
        refav = np.ma.average(np.ma.masked_invalid(refdat), weights=np.ma.masked_array(area_weights.flat, mask=np.ma.masked_invalid(refdat).mask))
        diff = np.square(np.ma.masked_invalid(np.subtract(np.subtract(refdat,refav),np.subtract(modeldat,waverage))))
        mse[ieof+9,modeli] = np.sum(diff*weights_masked)/np.sum(weights_masked)
        rmse[ieof+9,modeli] = math.sqrt(mse[ieof+9,modeli])
        
        # unweighted comp metrics:
        #ccoef[ieof+9,modeli] = np.ma.corrcoef(np.ma.masked_invalid(refdat), np.ma.masked_invalid(modeldat)).data[1,0]
        #sdev[ieof+9,modeli] = np.ma.std(np.ma.masked_invalid(modeldat))
        #mse[ieof+9,modeli] = np.mean(np.square(np.ma.masked_invalid(np.subtract(refdat,modeldat))))
        #rmse[ieof+9,modeli] = math.sqrt(mse[ieof+9,modeli])

        print(ccoef[ieof+9,modeli],sdev[ieof+9,modeli],mse[ieof+9,modeli],rmse[ieof+9,modeli])
        '''

        # alternative: convert NaN to 0
        refdat = obs_high_corr_reg_coefficients[ieof,:].flat
        modeldat = model_high_corr_reg_coefficients[ieof,:].flat
        comb_mask = np.logical_and(np.ma.masked_invalid(refdat).mask, np.ma.masked_invalid(modeldat).mask)   # mask out only places where both are nan
        weights_masked = np.ma.masked_array(area_weights.flat, mask=comb_mask)
        refdat = np.nan_to_num(np.ma.masked_array(refdat, mask=comb_mask))
        modeldat = np.nan_to_num(np.ma.masked_array(modeldat, mask=comb_mask))

        ccoef[ieof+9,modeli] = WeightedCorr(x=pd.Series(refdat), y=pd.Series(modeldat), w=pd.Series(weights_masked))(method='pearson')
        sdev[ieof+9,modeli] = DescrStatsW(modeldat, weights=weights_masked, ddof=0).std
        # normalized std:
        sdev[ieof+9,modeli] = DescrStatsW(modeldat, weights=weights_masked, ddof=0).std/DescrStatsW(refdat, weights=weights_masked, ddof=0).std
        diff = np.square(np.subtract(refdat,modeldat))
        #mse[ieof+5,modeli] = np.sum(diff*area_weights.flat)/np.sum(area_weights.flat)
        #rmse[ieof+5,modeli] = math.sqrt(mse[ieof+5,modeli])
        modelav = np.average(modeldat, weights=weights_masked)
        refav = np.average(refdat, weights=weights_masked)
        diff = np.square(np.subtract(np.subtract(refdat,refav),np.subtract(modeldat,modelav)))
        mse[ieof+9,modeli] = np.sum(diff*weights_masked)/np.sum(weights_masked)
        rmse[ieof+9,modeli] = math.sqrt(mse[ieof+9,modeli])

        




    return ccoef, sdev, mse, rmse


def GL_taylor_metrics(ccoef,sdev,mse,rmse,obs_coefficients,model_coefficients,obs_high_corr_reg_coefficients,model_high_corr_reg_coefficients,area_weights,modeli,loaddata,models):

    # PCR:
    #'''
    for ieof in range(len(model_coefficients)):
        try:
            refdat = obs_coefficients[ieof,:].flat
        except AttributeError:
            refdat = obs_coefficients[ieof,:].data.flat
        try:
            modeldat = model_coefficients[ieof,:].flat
        except AttributeError:
            modeldat = model_coefficients[ieof,:].data.flat
        ccoef[ieof+13,modeli] = WeightedCorr(x=pd.Series(refdat), y=pd.Series(modeldat), w=pd.Series(area_weights.flat))(method='pearson')
        sdev[ieof+13,modeli] = DescrStatsW(modeldat, weights=area_weights.flat, ddof=0).std
        diff = np.square(np.subtract(refdat,modeldat))
        #mse[ieof+5,modeli] = np.sum(diff*area_weights.flat)/np.sum(area_weights.flat)
        #rmse[ieof+5,modeli] = math.sqrt(mse[ieof+5,modeli])
        modelav = np.average(modeldat, weights=area_weights.flat)
        refav = np.average(refdat, weights=area_weights.flat)
        diff = np.square(np.subtract(np.subtract(refdat,refav),np.subtract(modeldat,modelav)))
        mse[ieof+13,modeli] = np.sum(diff*area_weights.flat)/np.sum(area_weights.flat)
        rmse[ieof+13,modeli] = math.sqrt(mse[ieof+13,modeli])

        print(ccoef[ieof+13,modeli],sdev[ieof+13,modeli],mse[ieof+13,modeli],rmse[ieof+13,modeli])
        #'''
        
    # high corr PCR: 
    for ieof in range(len(model_coefficients)):
        refdat = obs_high_corr_reg_coefficients[ieof,:].flat
        modeldat = model_high_corr_reg_coefficients[ieof,:].flat
        
        comb_mask = np.ma.mask_or(np.ma.masked_invalid(refdat).mask, np.ma.masked_invalid(modeldat).mask)
        weights_masked = np.ma.masked_array(area_weights.flat, mask=comb_mask)
        
        ccoef[ieof+17,modeli] = WeightedCorr(x=pd.Series(refdat), y=pd.Series(modeldat), w=pd.Series(area_weights.flat))(method='pearson')
        waverage = np.ma.average(np.ma.masked_invalid(modeldat), weights=np.ma.masked_array(area_weights.flat, mask=np.ma.masked_invalid(modeldat).mask))  
        sdev[ieof+17,modeli] = math.sqrt(np.ma.average((np.ma.masked_invalid(modeldat)-waverage)**2, weights=weights_masked))
        diff = np.square(np.ma.masked_invalid(np.subtract(refdat,modeldat)))
        #mse[ieof+9,modeli] = np.sum(diff*weights_masked)/np.sum(weights_masked)
        #rmse[ieof+9,modeli] = math.sqrt(mse[ieof+9,modeli])
        refav = np.ma.average(np.ma.masked_invalid(refdat), weights=np.ma.masked_array(area_weights.flat, mask=np.ma.masked_invalid(refdat).mask))
        diff = np.square(np.ma.masked_invalid(np.subtract(np.subtract(refdat,refav),np.subtract(modeldat,waverage))))
        mse[ieof+17,modeli] = np.sum(diff*weights_masked)/np.sum(weights_masked)
        rmse[ieof+17,modeli] = math.sqrt(mse[ieof+17,modeli])
        
        # unweighted comp metrics:
        #ccoef[ieof+9,modeli] = np.ma.corrcoef(np.ma.masked_invalid(refdat), np.ma.masked_invalid(modeldat)).data[1,0]
        #sdev[ieof+9,modeli] = np.ma.std(np.ma.masked_invalid(modeldat))
        #mse[ieof+9,modeli] = np.mean(np.square(np.ma.masked_invalid(np.subtract(refdat,modeldat))))
        #rmse[ieof+9,modeli] = math.sqrt(mse[ieof+9,modeli])

        print(ccoef[ieof+17,modeli],sdev[ieof+17,modeli],mse[ieof+17,modeli],rmse[ieof+17,modeli])



    return ccoef, sdev, mse, rmse



def skill_score(ccoef, sdev):
    R0 = 1 # ccoef[0] #1
    if (ccoef[0]!=1):
        print(f'First element is not reference...? ccoef={ccoef[0]}')
    sdr = sdev/sdev[0]
    skill_score = (4*(1+ccoef)**4)/((sdr+1/sdr)**2*(1+R0)**4)
    skill_score2 = (1+ccoef)**4 / (4*(sdr+1/sdr)**2)          # = same same
    return skill_score#, skill_score2


def plot_taylor_diagrams(metrics, variable, models, regdata_type,area=''):

    ccoef = metrics[0,:]
    sdev = metrics[1,:]
    mse = metrics[2,:]
    rmse = metrics[3,:]

    #labels = ['ERA5', 'E', 'I', 'C', 'MP', 'U', 'MI', 'CE']#, 'E10', 'E101']  # models
    #labels_expl =  ['ERA5','E = EC-Earth3','I = IPSL-CM6A-LR', 'C = CNRM-ESM2-1', 'MP = MPI-ESM1-2-HR', 'U = UKESM1-0-LL', 'MI = MIROC6', 'CE = CESM2']#, 'E10 = EC-Earth r10', 'E101 = EC-Earth r101']

    labels = ['ERA5', 'CE', 'C', 'E1', 'E10', 'I', 'MI', 'MP', 'U'] #, 'E101']  # models
    labels_expl =  ['ERA5', 'CE = CESM2', 'C = CNRM-ESM2-1', 'E1 = EC-Earth3 r1', 'E10 = EC-Earth r10','I = IPSL-CM6A-LR', 'MI = MIROC6', 'MP = MPI-ESM1-2-HR', 'U = UKESM1-0-LL'] #, 'E101 = EC-Earth r101']


    skill_scores = np.zeros([8,len(models)])

    if (area=='Greenland'):
    	plusnumarr = [13,17]
    else:
    	plusnumarr = [5,9]
    
    for plusnum in plusnumarr: # 4 if wanting to look at PCs, 8 if wanting to look at high corr PCs  => somehow that's 5 and 9 now, not quite sure why...
        if (plusnum==5):
        	compbase = 'fullPCR' 
        elif (plusnum==13):
        	compbase = 'fullPCR' 
        else:
        	compbase = 'high_corr'
        compbasename = 'full field' if (compbase=='fullPCR') else 'high correlation locations'
        print(compbase)

        fig, axs = plt.subplots(2, 2, figsize=(15,10)) ; #, sharey=True);
        rcParams.update({'font.size': 13}) # font size of axes text
        rowsi = [0,0,1,1]
        colsi = [0,1,0,1]
        for ieof in range(4):
            ax = axs[rowsi[ieof], colsi[ieof]]
            #ax.set(adjustable='box', aspect='equal')
            #print(sdev[ieof+plusnum,:],rmse[ieof+plusnum,:],ccoef[ieof+plusnum,:])
            markerclrs = ['green' if (x == np.argsort(ccoef[ieof+plusnum,:])[-2]) else 'red' for x in range(len(labels))]
            sm.taylor_diagram(ax, sdev[ieof+plusnum,:],rmse[ieof+plusnum,:],ccoef[ieof+plusnum,:],
                              markerLabel = labels, styleOBS = '-', colOBS = 'r', markerobs = 'o', titleOBS = 'ERA5',
                              #markercolors = {"face": markerclrs,"edge": markerclrs}, #titleSTD = 'off',titleRMS = 'off',
                             );
            ax.set_title(f'PC {ieof+1}', loc='right', fontsize='xx-large');
            #ax.set_title(models[np.argsort(ccoef[ieof+plusnum,:])[-2]], loc='center', fontsize='small')

            # plot a table with skill score:

            scores = skill_score(ccoef[ieof+plusnum,:],sdev[ieof+plusnum,:])

            cell_text=[]
            for i in range(len(models)):
                if (i==0):
                    continue
                index = np.argsort(scores)[::-1][i];
                cell_text.append([labels_expl[index], np.round(scores[index],3)])


            ax.table(cellText=cell_text, cellLoc='left', colLabels=['Model','Skill Score'],loc='right',bbox = [1.1, 0, 0.6, 1])


            if (plusnum==5):
                skill_scores[ieof]=scores
            elif (plusnum==13):
                skill_scores[ieof]=scores
            else:
                skill_scores[ieof+4]=scores
        
        fig.suptitle(f'{regdata_type} PC Regression ({compbasename}) {area} Model Evaluation', fontsize='xx-large')
        fig.tight_layout();

        fig.savefig(f'output/taylordiagrams/taylor_{variable}_{compbase}_{area}.pdf') # {timeframe}



        fig, axs = plt.subplots(2, 2, figsize=(15,10)) ; #, sharey=True);
        fig.patch.set_visible(False)
        rowsi = [0,0,1,1]
        colsi = [0,1,0,1]
        for ieof in range(4):
            ax = axs[rowsi[ieof], colsi[ieof]]
            ax.axis('off')
            ax.axis('tight')
            scores = skill_score(ccoef[ieof+plusnum,:],sdev[ieof+plusnum,:])
            cell_text=[]
            for i in range(len(models)):
                if (i==0):
                    continue
                index = np.argsort(scores)[::-1][i];
                cell_text.append([models[index], np.round(scores[index],3)])
            ax.table(cellText=cell_text, cellLoc='left', colLabels=['Model','Skill Score'],loc='center',bbox = [0.5, 0, 1, 1])
            ax.set_title(f'PC {ieof+1}', loc='right', fontsize='xx-large')
        fig.suptitle(f'{regdata_type} PC Regression ({compbasename}) {area} Model Evaluation', fontsize='xx-large')
        fig.tight_layout();
        fig.savefig(f'output/taylordiagrams/taylor_scoretables_{variable}_{compbase}_{area}.pdf') # {timeframe}
    

    return skill_scores




def plot_1_taylor_diagram(metrics, variable, models, regdata_type, ieof=0, area=''):

    #models = models[0:-1]
    #metrics = metrics[:,:,0:-1]

    ccoef = metrics[0,:]
    sdev = metrics[1,:]
    mse = metrics[2,:]
    rmse = metrics[3,:]



    labels = ['ERA5', 'CE', 'C', 'E1', 'E10', 'I', 'MI', 'MP', 'U']#, 'E10'] #, 'E101']  # models
    labels_expl =  ['ERA5', 'CE = CESM2', 'C = CNRM-ESM2-1', 'E1 = EC-Earth3 r1', 'E10 = EC-Earth r10', 'I = IPSL-CM6A-LR', 'MI = MIROC6', 'MP = MPI-ESM1-2-HR', 'U = UKESM1-0-LL'] #, 'E101 = EC-Earth r101']

    skill_scores = np.zeros([8,len(models)])

    if (area=='Greenland'):
    	plusnumarr = [13,17]
    else:
    	plusnumarr = [5,9]
    
    for plusnum in plusnumarr: # 4 if wanting to look at PCs, 8 if wanting to look at high corr PCs  => somehow that's 5 and 9 now, not quite sure why...
        if (plusnum==5):
        	compbase = 'fullPCR' 
        elif (plusnum==13):
        	compbase = 'fullPCR' 
        else:
        	compbase = 'high_corr'
        compbasename = 'full field' if (compbase=='fullPCR') else 'high correlation locations'
        print(compbase)

        fig, ax = plt.subplots(figsize=(10,6)) ; #, sharey=True);
        #rcParams.update({'font.size': 13}) # font size of axes text
        #markerclrs = ['green' if (x == np.argsort(ccoef[ieof+plusnum,:])[-2]) else 'red' for x in range(len(labels))]
        sm.taylor_diagram(ax, sdev[ieof+plusnum,:],rmse[ieof+plusnum,:],ccoef[ieof+plusnum,:],
                          markerLabel = labels, styleOBS = '-', colOBS = 'r', markerobs = 'o', titleOBS = 'ERA5',
                          #markercolors = {"face": markerclrs,"edge": markerclrs}, #titleSTD = 'off',titleRMS = 'off',
                         );
        #ax.set_title(f'PC {ieof+1}', loc='right', fontsize='xx-large');

        # plot a table with skill score:
        scores = skill_score(ccoef[ieof+plusnum,:],sdev[ieof+plusnum,:])
        cell_text=[]
        for i in range(len(models)):
            if (i==0):
                continue
            index = np.argsort(scores)[::-1][i];
            cell_text.append([labels_expl[index], np.round(scores[index],3)])
        ax.table(cellText=cell_text, cellLoc='left', colLabels=['Model','Skill Score'],loc='right',bbox = [1.1, 0, 0.8, 1])

        if (plusnum==5):
            skill_scores[ieof]=scores
        elif (plusnum==13):
            skill_scores[ieof]=scores
        else:
            skill_scores[ieof+4]=scores
        
        #fig.suptitle(f'{regdata_type} PC {ieof+1} Regression ({compbasename}) {area} Model Evaluation', fontsize='x-large')
        fig.suptitle(f'{regdata_type} PC {ieof+1} Regression ({compbasename}) {area}', fontsize='x-large')
        #fig.tight_layout();
        plt.subplots_adjust(left=-0.2,bottom=0.1, right=0.9, top=0.85, wspace=0.1, hspace=0.1)

        fig.savefig(f'output/taylordiagrams/taylor_PC{ieof+1}_{variable}_{compbase}_{area}.pdf') # {timeframe}

        if (ieof in [0,1,2]):
        	fig.savefig(f'output/plots/taylor_PC{ieof+1}_{variable}_{compbase}_{area}.pdf') 



        fig, ax = plt.subplots(figsize=(15,10)) ; #, sharey=True);
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        scores = skill_score(ccoef[ieof+plusnum,:],sdev[ieof+plusnum,:])
        cell_text=[]
        for i in range(len(models)):
            if (i==0):
                continue
            index = np.argsort(scores)[::-1][i];
            cell_text.append([models[index], np.round(scores[index],3)])
        ax.table(cellText=cell_text, cellLoc='left', colLabels=['Model','Skill Score'],loc='center',bbox = [0.5, 0, 1, 1])
        ax.set_title(f'PC {ieof+1}', loc='right', fontsize='xx-large')
        fig.suptitle(f'{regdata_type} PC Regression ({compbasename}) {area} Model Evaluation', fontsize='xx-large')
        fig.tight_layout();
        fig.savefig(f'output/taylordiagrams/taylor_scoretables_PC{ieof+1}_{variable}_{compbase}_{area}.pdf') # {timeframe}


        

    

    return skill_scores






def weighted_skillscore(skillscore,models,timeframe,variants,season,meantext,regridded_text,variable,regdata_type,obs_varfrac,area=''):

    weighted_score = np.zeros([2,len(models)])

    for im in range(len(models)):

        model = models[im]

        variant = variants[im]
        model_varfrac = np.load(f'data/{model}/varfrac_4EOFs_{timeframe}_{model}_{variant}_{season}_{meantext}{regridded_text}.npy')
            
        weighted_score[0,im] = np.average(skillscore[0:4,im], weights=model_varfrac)
        weighted_score[1,im] = np.average(skillscore[4:9,im], weights=model_varfrac) 

        # calculate difference to variance fraction:
        print(model, ':', np.round((obs_varfrac - model_varfrac)/obs_varfrac,3))
        # and use orders to reorder them to original


    # plot weighted scores:
    cell_text=[]
    cell_text1=[]
    for im in range(len(models)):
        if (im==0):
            continue

        index = np.argsort(weighted_score[0])[::-1][im];
        index1 = np.argsort(weighted_score[1])[::-1][im];
        #cell_text.append([models[index], np.round(weighted_score[0,index],3),models[index2], np.round(weighted_score[1,index2],3)])                           
        cell_text.append([models[index], np.round(weighted_score[0,index],3)]) ###, np.round(weighted_score[1,index],3)])                               
        cell_text1.append([models[index1], np.round(weighted_score[1,index1],3)])                               
    


    fig,ax=plt.subplots(figsize=(5,5))
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    ax.table(cellText=cell_text, cellLoc='left', colLabels=['Model','Score'],loc='center',bbox = [0.5, 0, 1, 1])
    #ax.table(cellText=cell_text1, cellLoc='left', colLabels=['Model','Score (high corr)'],loc='right',bbox = [0.9, 0, 1.5, 1])
    fig.suptitle(f'{regdata_type} PCR {area} Weighted Skill Score', fontsize='small')
    fig.tight_layout()
    fig.savefig(f'output/taylordiagrams/weightedscores_{variable}_full_{area}.pdf')


    fig,ax=plt.subplots(figsize=(5,5))
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    ax.table(cellText=cell_text1, cellLoc='left', colLabels=['Model','Score'],loc='center',bbox = [0.5, 0, 1, 1])
    fig.suptitle(f'{regdata_type} PCR (high correlation) {area} Weighted Skill Score', fontsize='small')
    fig.tight_layout()
    fig.savefig(f'output/taylordiagrams/weightedscores_{variable}_highcorr_{area}.pdf')

    print(models)
    print(weighted_score[0])
    print(weighted_score[1])

    return weighted_score






def GL_weighted_skillscore(skillscore,models,timeframe,variants,season,meantext,regridded_text,variable,regdata_type,obs_varfrac):

    weighted_score = np.zeros([2,len(models)])

    for im in range(len(models)):

        model = models[im]

        variant = variants[im]
        model_varfrac = np.load(f'data/{model}/varfrac_4EOFs_{timeframe}_{model}_{variant}_{season}_{meantext}{regridded_text}.npy')
            
        weighted_score[0,im] = np.average(skillscore[0:4,im], weights=model_varfrac)
        weighted_score[1,im] = np.average(skillscore[4:9,im], weights=model_varfrac) 

        # calculate difference to variance fraction:
        print(model, ':', np.round((obs_varfrac - model_varfrac)/obs_varfrac,3))
        # and use orders to reorder them to original


    # plot weighted scores:
    cell_text=[]
    cell_text1=[]
    for im in range(len(models)):
        if (im==0):
            continue

        index = np.argsort(weighted_score[0])[::-1][im];
        index1 = np.argsort(weighted_score[1])[::-1][im];
        #cell_text.append([models[index], np.round(weighted_score[0,index],3),models[index2], np.round(weighted_score[1,index2],3)])                           
        cell_text.append([models[index], np.round(weighted_score[0,index],3)]) ###, np.round(weighted_score[1,index],3)])                               
        cell_text1.append([models[index1], np.round(weighted_score[1,index1],3)])                               
    


    fig,ax=plt.subplots(figsize=(5,5))
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    ax.table(cellText=cell_text, cellLoc='left', colLabels=['Model','Score'],loc='center',bbox = [0.5, 0, 1, 1])
    #ax.table(cellText=cell_text1, cellLoc='left', colLabels=['Model','Score (high corr)'],loc='right',bbox = [0.9, 0, 1.5, 1])
    fig.suptitle(f'{regdata_type} PCR Weighted Skill Score', fontsize='small')
    fig.tight_layout()
    fig.savefig(f'output/taylordiagrams/weightedscores_{variable}_full.pdf')


    fig,ax=plt.subplots(figsize=(5,5))
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    ax.table(cellText=cell_text1, cellLoc='left', colLabels=['Model','Score'],loc='center',bbox = [0.5, 0, 1, 1])
    fig.suptitle(f'{regdata_type} PCR (high correlation) Weighted Skill Score', fontsize='small')
    fig.tight_layout()
    fig.savefig(f'output/taylordiagrams/weightedscores_{variable}_highcorr.pdf')

    print(models)
    print(weighted_score[0])
    print(weighted_score[1])

    return weighted_score










