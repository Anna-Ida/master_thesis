

#def input_files(base_filename):
	#myvar = 'pr' if (regdata_type=='Precipitation') else 'tas'

	#base_filename = f'Amon_{model}_historical_{variant}_{grid}_{timeframe_long}'

	#extr_infile = f'data/{model}/{extreme_region}_{myvar}_{base_filename}_detrended.nc'

	#return extr_file


def name_files(analysis_type,month,season,season_mean,running_mean,extremes,extr_type,extreme_region,nmonths,timeframe,model,variant,n_eofs,regdatatypes_short,regdata_type,reg_type,mask_low_corr=False,regridded_text=''):

	if (analysis_type == 'allmonths'):
		monthorseason = 'allmonths'
	elif (analysis_type == 'monthly'):
		monthorseason = month
	elif (analysis_type == 'seasonal'):
		monthorseason = season

	mean_text = '' if not season_mean else f'_mean{running_mean}'
	#if not season_mean:
	#	mean_text = '' if not season_mean else f'_mean{running_mean}'
	#elif season_mean:
	#	mean_text = f'_mean{running_mean}'

	if not extremes:
		extr_text = ''
	if (extremes=='quartiles'):
		extr_text = f'_{extr_type}quart_{extreme_region}'
	elif (extremes=='months'):
		if (extr_type=='dry'):
			extr_type = 'dri' #if (extr_type=='dry')
		elif (extr_type=='wet'):
			extr_type = 'wett' #if (extr_type=='wet')
		extr_text = f'_{extr_type}est{nmonths}months_{extreme_region}'

	regridtext=regridded_text 

	ID = f'{timeframe}_{model}_{variant}_{monthorseason}{mean_text}{extr_text}{regridtext}'
	savemeanas = f'output/{model}/meanpattern_zg_{timeframe}_{model}_{variant}_{monthorseason}{mean_text}{extr_text}{regridtext}.pdf'
	saveEOFas = f'output/{model}/{n_eofs}EOFs_{timeframe}_{model}_{variant}_{monthorseason}{mean_text}{extr_text}{regridtext}.pdf'
	savePCas = f'output/{model}/{n_eofs}PCs_{timeframe}_{model}_{variant}_{monthorseason}{mean_text}{extr_text}{regridtext}.pdf'
	if mask_low_corr:
		savePCregas = f'output/{model}/PC_{regdatatypes_short[regdata_type]}_{reg_type}_masked_{timeframe}_{model}_{variant}_{monthorseason}{mean_text}{extr_text}{regridtext}.pdf'
	else:
		savePCregas = f'output/{model}/PC_{regdatatypes_short[regdata_type]}_{reg_type}_full_{timeframe}_{model}_{variant}_{monthorseason}{mean_text}{extr_text}{regridtext}.pdf'
	saveregmeanas = f'output/{model}/meanpattern_{regdata_type}_{timeframe}_{model}_{variant}_{monthorseason}{mean_text}{extr_text}{regridtext}.pdf'
	saveextremesplotas = f'output/{model}/scatter_{timeframe}_{model}_{variant}_{monthorseason}{mean_text}_{regdata_type}{extremes}_{extreme_region}_{mean_text}{regridtext}'
	saveextremesmapas = f'output/{model}/map_{timeframe}_{model}_{variant}_{monthorseason}{mean_text}_{regdata_type}{extremes}_{extreme_region}_{mean_text}{regridtext}'


	saveEOFdataas =	f'{n_eofs}EOFs_{timeframe}_{model}_{variant}_{monthorseason}{mean_text}{extr_text}{regridtext}'
	savePCRdataas = f'PC_{regdatatypes_short[regdata_type]}_{reg_type}_{timeframe}_{model}_{variant}_{monthorseason}{mean_text}{extr_text}{regridtext}'
	
	#savequartilesas = f'quartile_{timeframe}_{model}_{variant}_{season}{mean_text}{extr_text}.pdf'


	return ID, savemeanas, saveEOFas, savePCas, savePCregas, saveregmeanas, saveEOFdataas, savePCRdataas, saveextremesplotas, saveextremesmapas



def plot_titles(season_mean,running_mean,extremes,extr_type,extreme_region_name,model,variant,season,timeframe):

	if not season_mean:
		mean_text = ''
	elif (running_mean==1):
		mean_text = 'mean'
	elif (running_mean>1):
		mean_text = f'{running_mean} year running mean'

	if (extremes=='quartiles'):
		extr_text = f'- {extr_type} quartile in {extreme_region_name}'
	elif (extremes=='months'):
		if (extr_type=='dry'):
			extr_type = 'dri' #if (extr_type=='dry')
		elif (extr_type=='wet'):
			extr_type = 'wett' #if (extr_type=='wet')
		extr_text = f'- {extr_type}est {nmonths} months in {extreme_region_name}'
	elif not extremes:
		extr_text = ''

	base_title = f'{model} {variant}, {season} {mean_text} {timeframe}'
	#standard_title = f'{model} {variant}, {season} {mean_text} {timeframe} \n {extr_text}'
	standard_title = f'{model} {variant}, {season} {mean_text} {timeframe} {extr_text}'

	#plt.suptitle('Plot specific, fontsize='x-large')
	#plt.title('Experiment info', fontsize='small')
	
	return standard_title, base_title



