import sys
from MDN.product_estimation import get_estimates, print_dataset_stats
from MDN.plot_utils import add_stats_box, add_identity
from MDN.parameters import get_args
from MDN.metrics import performance, mdsa, sspb, rmsle, slope, msa
from MDN.utils import ignore_warnings

from collections import defaultdict as dd
from scipy.interpolate import CubicSpline
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor as XGB
from pathlib import Path 

import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
import seaborn as sns 
import pandas as pd 
import numpy as np 

OLI_WAVELENGTHS = [443, 482, 561, 589, 655]
AER_WAVELENGTHS = [412, 443, 490, 510, 560, 620, 667, 681, 709]

# Ancillary data used as inputs
ANCILLARY = ['senz', 'solz', 'relaz', 'scattang', 'relative_humidity']

def load_insitu_Rrs():
	''' 
	In Situ Rrs which were input into the RT program to create Rho_t
	Used as the training target: Simulated Rho_* -> In Situ Rrs 
	'''
	cols = [*map(str, OLI_WAVELENGTHS)]
	data = pd.read_csv('Data/all_sites/representative_RT_OLI_Rrs_512.csv', usecols=cols)
	return data.rename(columns=int)

def matchups_filter(folder_ac):
	ac_data  = pd.read_csv('Data/all_sites/updated_phase1_and2_new_matchup_test_data_aod.csv')
	#aer_data = pd.read_csv(f'Data/{folder_ac}_aeronet_matchups.csv', header=[0,1])
	#time_diff = pd.read_csv(f'Data/all_sites/ids_60min.csv', header=[0])
	return np.all([
		#ac_data['time_diff'].abs() < (4 * 60), 
		#aer_data[( 'windspeed', 'Unnamed: 421_level_1')]   < 10,
		#time_diff['id']   > 0,
		#ac_data['rhot(1609)'] <= 0.01,
		#ac_data['ins_uid'] != 'Lake_Okeechobee',
		#ac_data['ins_uid'] != 'Helsinki_Lighthouse',
		#ac_data['ins_uid'] != 'Venise',
		#ac_data['ins_uid'] != 'LISCO',
		#ac_data['ins_uid'] != 'Grizzly_Bay',
		#ac_data['ins_uid'] != 'MVCO',
		#ac_data['ins_uid'] != 'WaveCIS_Site_CSI_6',
		#ac_data['ins_uid'] != 'Lake_Erie',
		ac_data['glint_coef']      < 1e4, 
		# ac_data['sstref'] > 0, # Throws out lake okeechobee data - some kind of issue with rhot865?
		# ac_data['Rrs(865)'] < 0.001,
		# ac_data['level_0'] > 5000,
		# ac_data['aot(865)'] < 0.2,
		# ac_data['poc'] < 500,
		# ac_data['pressure'] > 1000, # Throws out lake erie data - too much ice, potentially other problems
		# ac_data['relaz'].abs()    <= 160,
		# aer_data['Rrs']['667.0']   < 0.006,
	], axis=0)

def load_aeronet_Rrs(folder_ac, interp_method='quadratic'):
	'''
	OLI Rrs measurements made using hyperspectral measurements
	'''
	param = 'Rrs'
	cols = [f'{param}({w})' for w in OLI_WAVELENGTHS]
	Rrs = pd.read_csv('Data/all_sites/updated_phase1_and2_new_matchup_test_data_aod.csv').rename(columns=dict(zip(cols, OLI_WAVELENGTHS)))

	print(f'here 2 test Rrs')
	Rrs = Rrs[OLI_WAVELENGTHS]
	Rrs.columns = map(int, map(float, Rrs.columns))
	print(Rrs.columns)

	return Rrs[matchups_filter(folder_ac)]

def load_landsat(folder_ac, param='rhos'):
	''' 
	Rho_t/Rho_s + ancillary measurements observed by landsat (and extracted via acolite) 
	Used as the testing input: Landsat Rho_* + ancillary -> AERONET Rrs
	'''
	wvls = [w for w in OLI_WAVELENGTHS if w != 589]# + [865]
	cols = [f'{param}({w})' for w in wvls]
	data = pd.read_csv(f'Data/all_sites/updated_phase1_and2_new_matchup_test_data_aod.csv').rename(columns=dict(zip(cols, wvls)))
	data['relative_humidity'] = data['humidity'] / 100.
	data['relaz'] = data['relaz'].abs()
	return data[wvls + ANCILLARY][matchups_filter(folder_ac)]

def load_simulated(folder='478_samples', full=True):
	'''
	Rho_t simulated by using in situ Rrs and ancillary parameters as input
	Used as the training input: Simulated Rho_* + ancillary -> In Situ Rrs
	'''
	wvls = [w for w in OLI_WAVELENGTHS if w != 589]# + [865]
	#data = pd.read_csv(f'Simulations/{folder}/simulation_478_oli_case9.csv', header=[0,1])
	#data = pd.read_csv(f'Simulations/{folder}/simulation_all_oli_case1to9.csv', header=[0,1])
	data = pd.read_csv(f'Simulations/simulation_oli_165M_rhos_dummy.csv', header=[0,1])
	print(data.columns)
	#assert(0)
	omit_index = pd.read_csv('omit_index.csv')
	idx = omit_index['id'].to_numpy()
	data = data.loc[~data[('index', 'Unnamed: 0_level_0')].isin(idx)]
	#data = data.loc[data[('index', 'Unnamed: 0_level_0')]<215]
	print(data[('index', 'Unnamed: 0_level_0')].unique())
	#assert(0)
	#if not full: data = data.loc[data.droplevel(1, axis=1)['aerosol_model'] < 26]
	data = data.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='all').dropna(axis=0, how='any')
	rhot = data['Rho_s'].rename(columns=float).rename(columns=round)[wvls]
	invalid = (rhot > 1).any(1)
	data = data.loc[~invalid]
	rhot = rhot.loc[~invalid]
	
	invalid = (rhot < 0.00001).any(1)
	data = data.loc[~invalid]
	rhot = rhot.loc[~invalid]
	
	data = data.rename(columns={
		'Viewing_Azimuth' : 'relaz',
		'Viewing_Zenith'  : 'senz',
		'sza'             : 'solz',
	}).droplevel(1, axis=1)

	rad   = lambda d: d * np.pi / 180.
	deg   = lambda r: r * 180. / np.pi 
	solz  = rad(data['solz'])
	senz  = rad(data['senz'])
	relaz = rad(data['relaz'])
	data['scattang'] = deg(np.arccos(-np.cos(solz) * np.cos(senz) + np.sin(solz) * np.sin(senz) * np.cos(relaz)))
	return pd.concat([rhot, data[ANCILLARY + ['index']]], axis=1)

def load_data(folder_ac): 
	train_x = load_simulated()
	train_y = load_insitu_Rrs()
	test_x  = load_landsat(folder_ac)
	test_y  = load_aeronet_Rrs(folder_ac)

	# Duplicate the in situ Rrs to align with the simulated data
	train_y = train_y.loc[train_x.pop('index').to_list()]

	# Remove samples outside the training domain/range
	valid  = np.ones(len(test_x)).astype(bool)
	valid  = (test_x.iloc[:,0:4] > 0.00001).all(1)
	valid  &= (test_x >= train_x.min(0)).all(1) & (test_x <= train_x.max(0)).all(1)
	valid &= (test_y >= train_y.min(0)).all(1) & (test_y <= train_y.max(0)).all(1)
	np.savetxt("new_valid.csv", valid, delimiter=",")
	# n = 2
	# valid  = ((test_x >= train_x.min(0)).sum(1) >= (test_x.shape[1]-n)) & ((test_x <= train_x.max(0)).sum(1) >= (test_x.shape[1]-n))
	# valid &= ((test_y >= train_y.min(0)).sum(1) >= (test_y.shape[1]-n)) & ((test_y <= train_y.max(0)).sum(1) >= (test_y.shape[1]-n))
	test_x = test_x[valid]
	test_y = test_y[valid]
	print(f'Filtered {(~valid).sum()} out of range samples')

	assert((train_x.columns == test_x.columns).all())
	assert((train_y.columns == test_y.columns).all())
	assert(len(train_x) == len(train_y))
	assert(len(test_x)  == len(test_y))

	print_dataset_stats(x_simulated=train_x, y_insitu=train_y*1e3, label='Training')
	print_dataset_stats(x_landsat=test_x,    y_aeronet_x1000=test_y*1e3, label='Testing')
	return [df.to_numpy() for df in [train_x, train_y, test_x, test_y]] + [valid, test_x.columns]

def get_model_estimate(train_x, train_y, test_x, test_y, *args, use_XGB=True, full=True):
	if use_XGB:
		model_lbl = 'XGB'
		model     = MultiOutputRegressor( XGB(n_estimators=35, max_depth=25, verbosity=1) )
		estimate  = model.fit(train_x, train_y).predict(test_x)	

	else:
		kwargs = {
			'verbose'   : True,
			'no_load'   : False,
			'model_lbl' : 'MDN_AC_reduce_rhos_0_no2_o3_rr_10itr',
			'model_loc' : 'Weights_dev',
			'n_rounds'  : 10,
			'plot_loss' : False,
			'n_iter'    : 30000,
			'l2'        : 1e-3,
			'n_hidden'  : 30,
			'n_layers'  : 5,
			'batch'     : 2048,
			'lr'        : 1e-3,
		}
		model_lbl = 'ACII' if full else 'ACI'
		slices    = {str(wvl): slice(i, i+1) for i, wvl in enumerate(OLI_WAVELENGTHS)}
		estim,  _ = get_estimates(get_args(**kwargs), train_x, train_y, test_x, test_y, slices)
		estimate  = np.median(estim, 0)

	for wvl, y, est in zip(OLI_WAVELENGTHS, test_y.T, estimate.T):
		print(performance(f'{model_lbl}_{wvl}', y, est))
	print()
	np.savetxt('new_estimated_rhorc_Rrs.csv', estimate, delimiter=',')
	np.savetxt('new_test_y_Rrs.csv', test_y, delimiter=',')
	return model_lbl, estimate

@ignore_warnings
def plot_results_new(model_label, estimate, target, other_label, other, aco, valid, folder_ac, save=None):
	idx = OLI_WAVELENGTHS.index(589)
	estimate = np.append(estimate[..., :idx], estimate[..., idx+1:], 1)
	other    = np.append(   other[..., :idx],    other[..., idx+1:], 1)
	target   = np.append(  target[..., :idx],   target[..., idx+1:], 1)
	wavelengths = OLI_WAVELENGTHS[:idx] + OLI_WAVELENGTHS[idx+1:]

	Rrs_label = 'R_{rs} [sr^{-1}]'
	y_label = f'Satellite-derived {Rrs_label}'
	x_label = f'In-Situ {Rrs_label}'
	estimate = (model_label, estimate)
	other    = (other_label, other)
	target   = (None,        target)

	results  = [
		(target, estimate),
		# (other,  estimate),
		(target, other),
	]


	site_name = lambda name: name.title().replace('_', ' ').replace('Wavecis Site Csi 6', 'Wave CIS').replace('Lisco', 'LISCO').replace('Mvco', 'MVCO')
	plt.rc('text', usetex=True)
	plt.rcParams['mathtext.default']='regular'

	locations = pd.read_csv(f'Data/all_sites/updated_phase1_and2_new_matchup_test_data_aod.csv')['ins_uid'].to_numpy()[matchups_filter(folder_ac)][valid]
	for loc in np.unique(locations):
		print(site_name(loc) + ':', (locations == loc).sum(), '/', len(locations))

	cols_num  = len(wavelengths)
	rows_num  = len(results)
	cols_size = 4 * cols_num
	rows_size = 4 * rows_num
	f, all_ax = plt.subplots(rows_num, cols_num, figsize=(cols_size, rows_size))
	
	full_ax  = f.add_subplot(111, frameon=False)
	full_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False, pad=10)
	# full_ax.set_xlabel(fr'$\mathbf{{{x_label} (N={np.isfinite(np.log10(target[1])).any(-1).sum()})}}$'.replace(' ', '\ '), fontsize=18))
	full_ax.set_xlabel(fr'$\mathbf{{{x_label}}}$'.replace(' ', '\ '), fontsize=18)
	full_ax.set_ylabel(fr'$\mathbf{{{y_label}}}$'.replace(' ', '\ '), fontsize=18, labelpad=35)

	stats = dd(lambda: dd(lambda: dd(dict)))
	for i, (row_axes, ((lblx, x), (lbly, y))) in enumerate(zip(all_ax, results)):
		print(lblx, lbly, len(row_axes), x.shape, y.shape)

		row_axes[0].set_ylabel(fr'$\mathbf{{{lbly} (N={np.isfinite(np.log10(x)).any(-1).sum()})}}$'.replace(' ', '\ '), fontsize=16)

		for j, (ax, feat, x_feat, y_feat) in enumerate(zip(row_axes, wavelengths, x.T, y.T)):				
			x_log   = np.log10(x_feat)
			y_log   = np.log10(y_feat)
			x_valid = np.isfinite(x_log)
			y_valid = np.isfinite(y_log)
			print(feat, x_valid.sum(), y_valid.sum())

			x_tmp = x_log[~np.isnan(y_log)]
			y_tmp = y_log[~np.isnan(y_log)]
			b, a = np.polyfit(x_tmp, y_tmp, deg=1)
			xseq = np.linspace(-4, 0, num=len(x_log))

			if len(locations) == len(x_log):
				for loc in np.unique(locations):
					ax.scatter(x_log[locations == loc], y_log[locations == loc], 15, label=r'$\mathbf{%s}$' % site_name(loc).replace(' ','\ '), alpha=0.5)
					ax.plot(xseq, a + b * xseq, color='k', ls='dashed', dashes=[6, 3], lw=0.75)
			else:
				# ax.scatter(x_log, y_log, 15, alpha=0.5)
				for loc in np.unique(aco_locations):
					ax.scatter(x_log[aco_locations == loc], y_log[aco_locations == loc], 15, label=r'$\mathbf{%s}$' % site_name(loc).replace(' ','\ '), alpha=0.5)

			add_identity(ax, color='k', ls='--', alpha=0.5)

			#if i == 0 and feat == 443: 
				#ax.legend(loc='lower right', bbox_to_anchor=(0.5, 1.05), ncol=4, fontsize=8, fancybox=False, framealpha=1)

			if np.any(x_valid & y_valid):
				add_stats_box(ax, x_feat, y_feat, metrics=[mdsa, sspb, slope, msa, rmsle])
			
				minv = -4#round(np.nanmin(x_log)-1) 
				maxv = 0#round(np.nanmax(x_log)+1)
				# loc  = ticker.LinearLocator(numticks=int(round(maxv - minv + 1)))
				loc  = ticker.FixedLocator([-3, -2, -1])
				fmt  = ticker.FuncFormatter(lambda i, _: r'$10$\textsuperscript{%i}' % i)

				ax.set_ylim((minv+(.3 if feat != 651 else 0), maxv-.5))
				ax.set_xlim((minv+(.3 if feat != 651 else 0), maxv-.5))

				ax.xaxis.set_major_locator(loc)
				ax.yaxis.set_major_locator(loc)
				ax.xaxis.set_major_formatter(fmt)
				ax.yaxis.set_major_formatter(fmt)

				ax.grid(True, alpha=0.3)
				ax.tick_params(labelsize=18)

				if True:#i != 1:
					metrics = [mdsa, sspb, slope, msa, rmsle]
					for metric in metrics:
						stats['Overall'][lbly][feat][metric.__name__] = metric(x_feat, y_feat)

						if len(locations) == len(x_feat):
							for loc in sorted(np.unique(locations)):
								stats[loc][lbly][feat][metric.__name__] = metric(x_feat[locations==loc], y_feat[locations==loc])
								# print(loc,(locations==loc).sum())
						else:
							for loc in sorted(np.unique(aco_locations)):
								stats[loc][lbly][feat][metric.__name__] = metric(x_feat[aco_locations==loc], y_feat[aco_locations==loc])

				print(performance(feat, x_feat, y_feat))
				if len(locations) == len(x_feat):
					for loc in np.unique(locations):
						print('\t', performance(site_name(loc), x_feat[locations == loc], y_feat[locations == loc]))
				else:
					for loc in np.unique(aco_locations):
						print('\t', performance(site_name(loc), x_feat[aco_locations == loc], y_feat[aco_locations == loc]))

				print()

			if i == 0: 
				ax.set_title(fr'$\mathbf{{{feat}nm}}$', fontsize=18)
				ax.set_xticklabels([])
			if j != 0:
				ax.set_yticklabels([])

		print()

	df = pd.DataFrame.from_records([(loc, lbl, band, met, val) 
			for loc in stats 
			for lbl in stats[loc] 
			for band in stats[loc][lbl]
			for met, val in stats[loc][lbl][band].items()], columns=['Location', 'Model', 'Band', 'Metric', 'Value'])
	print(df)
	df.to_csv('stats.csv')
	plt.tight_layout()
	if save is not None: 
		plt.savefig(f'{save}.png')
	plt.show()

if __name__ == '__main__':
	folder  = 'all_sites'
	ac_name = 'l2gen'
	data    = *_, target, valid, cols = load_data(f'{folder}/{ac_name}')

	# L2gen Rrs
	ac_estim = load_landsat(f'{folder}/{ac_name}', 'Rrs')
	ac_estim[589] = np.nan 
	ac_estim = ac_estim[OLI_WAVELENGTHS].to_numpy()[valid]

	model_lbl_part, estimate_part = get_model_estimate(*data, use_XGB=False, full=True)
	model_lbl_full, estimate_full = model_lbl_part, estimate_part #get_model_estimate(*data, use_XGB=False, full=True)
	#store_data(f'{folder}/{ac_name}', valid, target, ac_estim, cols, data[2], estimate_part, estimate_full)
	plot_results_new( model_lbl_part, estimate_part, target, model_lbl_full, estimate_full, None,  valid, f'{folder}/{ac_name}', save='MDN_AC_rhorc_new_matchup_new')