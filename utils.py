import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import numpy as np 
import warnings, traceback

from geopy.distance import distance as geo_distance
from collections import defaultdict as dd 
from datetime import datetime as dt 
from pathlib import Path
from netCDF4 import Dataset

from Plot.meta import MODEL_KWARGS, PRODUCT_KWARGS, PRODUCT_BOUNDS, TILE_CHL, LOCATION_CENTER, LOCATION_BOUNDS, KEY_MAPPING
from Plot.Geolocate.img_registration import load_geolocated
from Plot.tile_utils import DatasetBridge

import sys
sys.path.append('..')

from MDN import apply_model
from MDN import get_args
from MDN import run_benchmarks
#from MDN.benchmarks.utils import to_Rrs
from MDN import RatioTransformer
from MDN import get_sensor_bands
from MDN import closest_wavelength, find_wavelength, ignore_warnings, mask_land


def get_model_kwargs(img_param, animate=False):
	method  = img_param['est_method']
	sensor  = img_param['sensor'] + ('-rho' if 'rho' in method or method in ['FAI', 'Cao_XGB'] else '')
	product = img_param['product'].replace('-', '/').split('/')[0]
	kwargs  = dict(MODEL_KWARGS)
	kwargs.update(PRODUCT_KWARGS.get(product, {}))
	kwargs.update({
		'product'   : product,
		'sensor'    : sensor,
		'n_rounds'  : 1 if animate else 10,
	})

	# align = [s for s in ['OLI', 'MSI', 'OLCI'] if s != sensor]
	# kwargs['align'] = ','.join(align)

	if method == 'Mishra_NDCI':
		kwargs['bands'] = np.array([665, 708])

	# MDN specific kwargs
	if 'MDN' in method:

		if img_param.get('simultaneous', False):
			kwargs.update({
				'product'  : 'chl,tss,cdom',
				# 'n_hidden' : 500,
				# 'n_iter': 10000,
			})

		if False and method == 'MDN' and img_param['atm_method'] in ['l2gen'] and sensor == 'OLI':
			kwargs.update({
				'sensor': sensor+'-S2',
				# 'product': 'chl,tss,cdom',
				# 'n_hidden': 500,
				# 'threshold' : 0.75,
				# 'avg_est': True,
				# 'model_lbl' : 'noarctic',
				# 'benchmark' : True,
			})

		# if 'HICO' in sensor:
		# 	kwargs.update({
		# 		'n_iter' : 10000,
		# 		'sat_bands' : True, 
		# 		# 'model_lbl' : 'exclude20140909',
		# 	})

		if 'MOD' in sensor:
			kwargs.update({
				'sensor' : 'MODA',
				'product'  : 'chl,tss,cdom',
				'n_hidden' : 100,
				'n_iter' : 10000,
			})

		if '-opt' in method:
			kwargs.update({
				'n_iter' : 14000,
				'l2' : 0.000283,
				'lr' : 0.000286,
				'n_hidden': 500,
				'n_layers': 3,
				'product' : 'chl,tss,cdom',
			})

		# Models using -rho, -star, -anc, etc.
		elif '-' in method:		
			param  = img_param['est_method'].replace('MDN-', '')
			ackey  = img_param['atm_method'].replace('_snap','').replace('_fullband','')
			loc    = img_param['location']
			date   = img_param['datetime'].strftime('%Y%m%d')
			folder = Path(kwargs['model_loc']).joinpath(kwargs['sensor'])
			assert(folder.exists()), f'No model found at "{folder}"'

			model_lbl = f'{ackey}full_{param}_full'
			loc = loc.replace('_', '')
			if f'{loc}-{date}' in ['Erie-20150914', 'Peipsi-20160614', 'SFBay-20170427']: 
				model_lbl = f'{ackey}full_{param}_{loc}-{date}' 

			# model_lbl = f'{ackey}_{param}_rhosonly'
			final_loc = folder.joinpath(model_lbl)
			assert(final_loc.exists()), f'No trained model exists at {final_loc}'
			kwargs['model_lbl'] = model_lbl

			if '-star' in method:
				kwargs['use_ratio'] = True
				wavelengths = get_sensor_bands(kwargs['sensor'])
				transformer = RatioTransformer(wavelengths)
				transformer.fit_transform(np.ones((1, len(wavelengths))))
				labels = transformer.labels
				if len(labels) == 1: kwargs['model_lbl'] += f'_{labels[-1]}'
				else:                kwargs['model_lbl'] += '_all'

		# Extra aerosol subtraction
		if getattr(img_param, 'AER_SUB', False):
			kwargs['model_lbl'] += '_AER'

	else: kwargs.update({'product': 'aph'})

	# if product in ['ad', 'ag', 'aph']:
	# 	kwargs['align'] = 'MODA'
	# 	pass

	# Choose which bands are shown for Rrs spectra
	if product in ['Rrs']:
		kwargs['product'] = 'aph'

	# Need to include bands necessary for both the original product, as well as chl
	if '/chl' in img_param['product'].replace('-', '/'):
		kwargs['product'] += ',chl'

	if img_param['sensor'] == 'OLCI' and img_param['atm_method'] == 'polymer':
		kwargs['sensor'] = 'OLCI-poly'
	return kwargs


def get_model_bands(img_param):
	kwargs = get_model_kwargs(img_param)
	args   = get_args(kwargs)
	bands  = set()
	for p in img_param['product'].split('/'):
		prod_param = dict(img_param)
		prod_param.update({'product': p})
		args = get_args( get_model_kwargs(prod_param) )
		bands = bands.union(set(get_sensor_bands(args.sensor, args)))
	return sorted(list(bands))


def get_img_preds(Rrs_bands, Rrs_data, img_param, fig_params, valid=None):
	preds = {}
	bands = {}
	for product in img_param['product'].replace('-', '/').split('/'):
		if product == 'AUC': continue
		prod_param = dict(img_param)
		prod_param.update({'product': product})
		prod_bands = get_model_bands(prod_param)
		prod_avail = np.sort([find_wavelength(w, Rrs_bands) for w in prod_bands])	
		assert(len(np.unique(prod_avail)) == len(prod_bands)), [prod_bands, prod_avail]
		
		bands[product] = np.array(Rrs_bands)[prod_avail]
		preds[product] = get_model_preds(bands[product], np.atleast_2d(Rrs_data)[..., prod_avail], prod_param, fig_params, valid)
	return bands, preds


def get_model_preds(bands, inp, img_param, fig_params, valid=None):
	if img_param['product'] == 'Rrs':
		assert(img_param['band_target'] is not None)
		estimates = inp[..., find_wavelength(img_param['band_target'], bands)]
		
	elif 'MDN' in img_param['est_method']:
		product_preds = {}
		for product in img_param['product'].split('/'):
			if product == 'AUC': continue
			prod_param  = dict(img_param)
			prod_param.update({'product': product})
			kwargs      = get_model_kwargs(prod_param)
			prod_bands  = get_model_bands(prod_param)
			prod_avail  = np.sort([find_wavelength(w, bands) for w in prod_bands])
			preds, idxs = apply_model(np.atleast_2d(inp)[..., prod_avail], silent=False, use_gpu=True, **kwargs)
			product     = prod_param['product'] if prod_param['product'] in idxs else kwargs['product'].split(',')[0]
			estimates   = preds[:, idxs[product]]
			product_preds[product] = estimates
			estimates   = np.hstack([preds[:, idxs[p]] for p in fig_params['products']])

		if '/' in img_param['product']:
			p1, p2 = img_param['product'].split('/')
			estimates = product_preds[p1] / product_preds[p2]

	elif img_param['est_method'] == TILE_CHL:
		estimates = load_tile_chl(img_param).flatten()[valid]

	else:
		benchmark = run_benchmarks(img_param['sensor'], inp, bands=bands, verbose=False, product=img_param['product'], kwargs_rs={'method':img_param['est_method']})
		estimates = benchmark[img_param['product']][img_param['est_method']]

	return np.atleast_1d(np.squeeze(estimates))


@ignore_warnings
def NDWI(band_key, avail_wvl, data, part_yx, threshold=2e-1, verbose=False, min_rrs=0, max_rrs=1e2):
	blue  = closest_wavelength(440,  avail_wvl, validate=False)
	green = closest_wavelength(560,  avail_wvl, validate=False)
	red   = closest_wavelength(700,  avail_wvl, validate=False)
	nir   = closest_wavelength(900,  avail_wvl, validate=False)
	swir  = closest_wavelength(1600, avail_wvl, validate=False)
	swir2 = closest_wavelength(2201, avail_wvl, validate=False)

	bands = (green, swir) if swir > 1500 else (red, nir) if red != nir else (min(avail_wvl), max(avail_wvl))
	if verbose: print(f'Using bands {bands[0]} & {bands[1]} for land masking')

	values = []
	for b in bands:
		data[f'{band_key}{b}'].set_auto_mask(False)
		data[f'{band_key}{b}'].set_always_mask(False)
		band = data[f'{band_key}{b}'][part_yx] / (np.pi if band_key == 'Rw' else 1)
		values.append( np.ma.masked_where(~np.isfinite(band) | (band <= min_rrs) | (band > max_rrs), band) )
	return mask_land(np.ma.stack(values, axis=-1), bands, threshold=threshold)


def center_image(im_lon, im_lat, location, img_width=1500, img_height=2500):
	''' Center the image on a given lon/lat coordinate '''
	if True and location in LOCATION_BOUNDS:
		n, s, e, w = LOCATION_BOUNDS[location]

		# Unsure why, but l2gen generates incorrect coordinate grids
		if location == 'Laguna' and im_lat.max() > 0:
			n, s, e, w = 55.5, 55.36, -54.11, -54.29

		ni = np.argmin(np.abs(im_lat[:,im_lat.shape[1]//2] - n))
		ei = np.argmin(np.abs(im_lon[ni] - e))
		wi = np.argmin(np.abs(im_lon[ni] - w))
		ni = min([np.argmin(np.abs(im_lat[:,ei] - n)), np.argmin(np.abs(im_lat[:,wi] - n))]) 

		si = np.argmin(np.abs(im_lat[:,im_lat.shape[1]//2] - s))
		ei2= np.argmin(np.abs(im_lon[si] - e))
		wi2= np.argmin(np.abs(im_lon[si] - w))
		si = max([np.argmin(np.abs(im_lat[:,ei2] - s)), np.argmin(np.abs(im_lat[:,wi2] - s))]) 

		ei = max([ei, ei2])
		wi = min([wi, wi2])
		partialy = slice(ni, si)
		partialx = slice(wi, ei)
		# print(partialy, partialx)
		# print(ni, si, wi, ei)

	else:
		center_lon = im_lon[im_lon.shape[0]//2, im_lon.shape[1]//2]
		center_lat = im_lat[im_lat.shape[0]//2, im_lat.shape[1]//2]
		if location in LOCATION_CENTER:
			center_lon, center_lat = LOCATION_CENTER[location]

		partialy = np.argmin(np.abs(im_lat[:,im_lat.shape[1]//2] - center_lat))
		partialx = np.argmin(np.abs(im_lon[partialy] - center_lon))
		partialy = np.argmin(np.abs(im_lat[:,partialx] - center_lat))
		partialx = slice(max(0, partialx-img_width), min(im_lon.shape[1], partialx+img_width))
		partialy = slice(max(0, partialy-img_height), min(im_lat.shape[0], partialy+img_height))

	if (partialx.stop - partialx.start) <= 0 or (partialy.stop - partialy.start) <= 0:
		print('Slices (y/lat, x/lon):', partialy, partialx)
		print(f'Lat shape={im_lat.shape} [min, max]=[{im_lat.min()}, {im_lat.max()}]')
		print(f'Lon shape={im_lon.shape} [min, max]=[{im_lon.min()}, {im_lon.max()}]')
		coord_def = f'Bounds={dict(zip("NSEW", LOCATION_BOUNDS[location]))}' if location in LOCATION_BOUNDS else f'Center=({center_lat}, {center_lon})'
		raise Exception(f'No pixels within bounds for {location}: {coord_def}')

	return partialy, partialx


def get_filename(img_param, atm_method=None, full=False):
	if 'filename' in img_param:
		if atm_method is not None:
			# return img_param['filename'].with_name(f'{atm_method}.nc')
			filenames = list(img_param['filename'].parent.rglob(f'{atm_method}.*'))
			filenames = [f for f in filenames if f.suffix in ['.nc', '.bsq']]
			if len(filenames):
				return filenames if full else filenames[-1]
		return img_param['filename']

	atm_method = atm_method or img_param['atm_method']
	#filenames  = list(Path(img_param['data_folder']).rglob(f'out/{atm_method}.*'))
	filenames = list(Path(img_param['data_folder']).rglob(f'*{atm_method}.*'))
	assert(len(filenames)), f'Data for {atm_method} does not exist at {img_param["data_folder"]}'
	filenames = [f for f in filenames if f.suffix in ['.nc', '.bsq']]
	# assert(len(filenames) == 1), f'Multiple files for {atm_method} found: {filenames}'
	if len(filenames) > 1: print(f'Multiple files for {atm_method} found:\n\t' + '\n\t'.join(list(map(str, filenames))))
	return filenames if full else filenames[-1]


def load_lonlat(img_param, atm_method=None, center=False):
	if img_param['geolocate']:
		lon, lat = load_geolocated(get_filename(img_param, atm_method))
		lon = np.ma.array(lon)
		lat = np.ma.array(lat)
	else:
		with DatasetBridge(get_filename(img_param, atm_method)) as data:
			if 'navigation_data' in data.groups.keys():
				data = data['navigation_data']
			lon_k, lat_k = ('lon', 'lat') if 'lon' in data.variables.keys() else ('longitude', 'latitude')
			lon,   lat   = data[lon_k][:], data[lat_k][:]

	if len(lon.shape) == 1:
		lon, lat = np.meshgrid(lon, lat)
	lon, lat = lon[img_param['part_yx']], lat[img_param['part_yx']]

	lon[lon < -180] = np.nan 
	lat[lat < -180] = np.nan 

	lon.mask = False 
	lat.mask = False 

	if center and img_param['part_yx'][0].start is None:
		lon_orig, lat_orig   = lon, lat
		img_param['part_yx'] = center_image(lon, lat, img_param['location'])
		lon, lat = lon[img_param['part_yx']], lat[img_param['part_yx']]

	# left, right, bottom, top | west, east, south, north
	img_param['extent'] = np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)
	return lon, lat, img_param['extent']


def load_tile_chl(img_param):
	with Dataset(img_param['filename'], 'r') as data:
		if 'geophysical_data' in data.groups.keys():
			data = data['geophysical_data']
		for key in ['chlor_a', 'conc_chl']:
			if key in data.variables.keys():
				return data[key][img_param['part_yx']]		
		raise Exception(f'Chl key missing:\n{data}')


def get_dto(img_param, verbose=False):
	# Get datatime of image (falling back to seadas tile, if necessary)

	def get_date(path, filename):
		with Dataset(list(path.rglob(f'out/{filename}.nc'))[0], 'r') as nc_data:
			if hasattr(nc_data, 'time_coverage_start'):
				dt_obj = dt.strptime(nc_data.time_coverage_start, '%Y-%m-%dT%H:%M:%S.%fZ')
			elif hasattr(nc_data, 'start_time'):
				dt_obj = dt.strptime(nc_data.start_time, '%Y-%m-%d %H:%M:%S')
			elif hasattr(nc_data, 'sensing_time'):
				dt_obj = dt.strptime(nc_data.sensing_time, '%Y-%m-%d %H:%M:%S')
			elif hasattr(nc_data, 'isodate'):
				try:    dt_obj = dt.strptime(nc_data.isodate, '%Y-%m-%dT%H:%M:%S.%fZ')
				except: dt_obj = dt.strptime(nc_data.isodate.split('.')[0], '%Y-%m-%dT%H:%M:%S')
			elif hasattr(nc_data, 'metadata'):
				dt_obj = data['metadata']['FGDC']['Identification_Information']['Time_Period_of_Content']
				dt_obj = dt.strptime(dt_obj.Ending_Date+'-'+dt_obj.Ending_Time, '%Y%m%d-%H%M%S')
			elif hasattr(nc_data, 'start_date'):
				dt_obj = dt.strptime(nc_data.start_date, '%d-%b-%Y %H:%M:%S.%f')
			else:
				# return dt.now()
				raise Exception(f'{nc_data}\nCan\'t determine image time')
			return dt_obj

	def read_meta(path, sensor):
		def oli_parse(path):
			print(path)
			with open(list(Path(path.as_posix().replace('/Scenes/','/NASA/Plotting/')).rglob('*_MTL.txt'))[0]) as f:
				meta = f.read()
				date = meta.split('DATE_ACQUIRED = ')[1].split('\n')[0]
				time = meta.split('SCENE_CENTER_TIME = "')[1].split('"')[0].split('.')[0]
			return dt.strptime(date+'T'+time, '%Y-%m-%dT%H:%M:%S')

		def msi_parse(path):
			with open(list(path.rglob('MTD_MSIL1C.xml'))[0]) as f:
				meta = f.read()
			datetime = meta.split('<DATATAKE_SENSING_START>')[1].split('</DATATAKE_SENSING_START>')[0]
			return dt.strptime(datetime, '%Y-%m-%dT%H:%M:%S.%fZ')

		sensor_funcs = {
			'OLI' : oli_parse,
			'MSI' : msi_parse,
		}
		duplicates = {
			'MSI' : ['S2A', 'S2B'],
			# 'OLCI': ['S3A', 'S3B'],
		}
		sensor_funcs.update({d: sensor_funcs[k] for k, dupl in duplicates.items() for d in dupl})
		assert(sensor in sensor_funcs), f'No meta parse function available for "{sensor}"'
		return sensor_funcs[sensor](path)

	path = Path(img_param['data_folder'])

	# Try getting from l2gen tile
	try: return get_date(path, 'l2gen')
	except: 

		# Try getting from the current AC tile
		try: return get_date(path, img_param['atm_method'])
		except: 

			# Try getting from the original tile metadata file
			try: return read_meta(path, img_param['sensor'])	
			except: return dt(2020, 9, 20)# dt.now()


def get_wvl(nc_data, key):
	wvl = []
	for v in nc_data.variables.keys():
		if key in v:
			try: wvl.append(int(v.replace(key, '')))
			except: pass
	return np.array(wvl) 


def sci_notation(num, decimal_digits=1, precision=None, exponent=None):
	from math import floor, log10
	if not exponent:
		exponent = int(floor(log10(abs(num))))
	coeff = round(num / float(10**exponent), decimal_digits)
	if not precision:
		precision = decimal_digits
	if coeff == 1:
		return r'$\mathbf{10^{%s}}$' % exponent
	return r'$\mathbf{%i\tiny{\times}10^{%s}}$' % (coeff, exponent)
	# return r"${0:.{2}f}x10^{{{1:d}}}$".format(coeff, exponent, precision)


def to_lon(val, pos):
	direct = 'W' if val < 0 else 'E'
	return '%.1f$^\circ$%s' % (abs(val), direct)


def to_lat(val, pos):
	direct = 'S' if val < 0 else 'N'
	return '%.1f$^\circ$%s' % (abs(val), direct)


def in_bounds(coords, shape):
	# Ensure x,y coordinates are within the bounds of the shape
	return all([c >= 0 and c < s for c,s in zip(coords, shape)])


def make_bold(s):
	# Make a mathtext bold face string
	return r'$\mathbf{%s}$' % str(s).replace('\ ',' ').replace(' ','\ ')


def fix_fig_size(fig, new_wh):
	# Update the figure width / height to be a certain size
	curr_w, curr_h = fig.get_size_inches()
	new_w,  new_h  = new_wh 
	assert(len([s for s in [new_w, new_h] if s is None]) == 1), f'One dimension must be None to maintain aspect ratio: {new_wh}'
	if new_w is None: new_w = (new_h / curr_h) * curr_w
	else:             new_h = (new_w / curr_w) * curr_h
	print(f'Changing figsize from ({curr_w}, {curr_h}) to ({new_w}, {new_h})')
	fig.set_size_inches(new_w, new_h)


def set_loc_date(ax, location, dt_obj, method='title', pad=None):
	# Set the location and date as the axis title (or label, depending on method)
	kwargs = {'y': pad, 'labelpad': pad}
	if pad is None:
		if method == 'title':
			kwargs = {'y': 1}
		else:
			kwargs = {'labelpad': 0.5 if method == 'ylabel' else 0.1}
	dt_label = dt_obj.strftime("%b. %d, %Y")
	print('setting dto at', pad, method)
	getattr(ax, f'set_{method}')(make_bold(location) + f'\n{dt_label}', fontsize=25, **kwargs)


def get_window(lon, lat, im_lon, im_lat, window_size=1):
	# Return a window around the closest pixel to (lon,lat), where window_size=1->3x3 window, 2->5x5, 0->1x1, etc.
	# Can pass flattened im_lon/im_lat arrays (i.e. filtered by valid pixels only) to get the closest valid 
	# points, rather than a strict NxN window 
	shape  = im_lon.shape
	im_lon = im_lon.flatten()
	im_lat = im_lat.flatten()

	# For efficiency, we first approximate the actual distance by finding the 1000 closest
	# pixels according to a total absolute degrees error metric. The actual physical distances
	# between these 1000 points and the target coordinates are then calculated, with the 
	# smallest physical distance (in meters) image coordinate being used as the window center
	pseudo_dist = (np.abs(im_lat - lat) + np.abs(im_lon - lon)).argsort()[:1000]
	distances   = [geo_distance((ilat, ilon), (lat, lon)).km * 1000
					for ilat, ilon in zip(im_lat[pseudo_dist], im_lon[pseudo_dist])]

	# If the passed image lon/lat arrays are 2d grids, then we return a strict NxN window around the center
	if len(shape) == 2:
		min_dist = np.argmin(distances)
		w_center = x, y = np.unravel_index(pseudo_dist[min_dist], shape)
		return [(x+i, y+j) for i in range(-window_size, window_size+1) for j in range(-window_size, window_size+1)]

	# Otherwise, we return the closest image points, regardless of location relative to one another 
	n_samples = (2*window_size + 1) ** 2
	min_dists = np.argsort(distances)[:n_samples]
	return [pseudo_dist[i] for i in min_dists]


def add_label(ax, label, loc='upper right'):
	# Add a label to the top left corner 
	# https://matplotlib.org/3.1.0/gallery/text_labels_and_annotations/demo_annotation_box.html
	from matplotlib.offsetbox import AnchoredText
	ann = AnchoredText(label, frameon=True, prop=dict(size=25, zorder=25, fontfamily='monospace'), loc=loc, borderpad=0.01, pad=0.2)
	ax.add_artist(ann)


def add_scale(ax, im_lon, im_lat, pct_size=0.15, multiple_of=10, include_compass=False, loc='lower left'):
	''' 
	Add a scale to the axis, where:
		- scale width will be approximately <pct_size> of the image width
		- measured length of the scale will be a multiple of <multiple_of> km
	https://matplotlib.org/3.3.1/gallery/axes_grid1/simple_anchored_artists.html

	TODO: 
	- fix scale
		- add 3 tick labels (left, center, right)
		- make 4 colored segments instead of 2
		- see here for basemap's implementation: https://github.com/matplotlib/basemap/blob/3076ec9470cf7dba523bc94ebe5ae9a990e34d08/lib/mpl_toolkits/basemap/__init__.py	
	- fix compass
		- use images
		- separate function "add_compass"
		- see Plotting/geolocated-plotting/utils.py - already finished there, using basemap foundation...
	'''
	from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
	from matplotlib.offsetbox import AuxTransformBox, HPacker, AnchoredOffsetbox
	from matplotlib.font_manager import FontProperties
	from matplotlib.patches import Rectangle, Ellipse

	from matplotlib.transforms import IdentityTransform


	maxim  = im_lon.max() - im_lon.min()
	width  = maxim * pct_size # width in degrees
	height = width * 0.035    # height in degrees
	angle  = 45               # northward rotation angle
	radius = min(im_lon.shape) * 0.05
	kwargs = {
		'loc'       : loc,
		'pad'       : 0.4, # Padding around label & bar, in fraction of font size (default 0.1)
		'borderpad' : 0.5, # Border padding in fraction of font size (default 0.1)
		'frameon'   : True,
		'fontproperties' : FontProperties(size=14, weight='bold'),
	}

	scale = artist = AnchoredSizeBar(ax.transData, width, '', fill_bar=False, size_vertical=height, **kwargs)
	
	if include_compass:
		compass = AuxTransformBox(IdentityTransform())
		compass.add_artist( Ellipse((0, 0), radius, radius, angle, fill=False,) )
		# For whatever reason, nested AnchoredOffsetboxes remain anchored to the outermost axis
		# This means we can't correctly nest and align these independent artists
		# Alternative is to use scale._box, though we then can't control the aesthetics of the scale bounding box, independently of the compass
		boxes   = [compass, scale._box] if 'right' in loc else [scale._box, compass] 
		artist  = AnchoredOffsetbox(loc=loc, child=HPacker(children=boxes, align='center', pad=0., sep=20), frameon=True, pad=0.7, borderpad=0.5)

	# Draw the subplot, so we can determine the size
	ax.add_artist(artist)
	ax.figure.canvas.draw()

	# Get rectangle which actually defines the scale
	bar, aux, *_ = scale.size_bar.findobj()
	assert(isinstance(aux, AuxTransformBox)), f'Did not find AuxTransformBox: {bar}, {aux}'

	# Define some helper methods to easily check the scale length in kilometers
	get_ext = lambda: bar.get_window_extent(ax.figure.canvas.renderer)
	get_pts = lambda: ax.transData.inverted().transform(get_ext())
	get_len = lambda p: geo_distance(p[0][::-1], p[1][::-1]).km
	is_nice = lambda d: (round(d % multiple_of, 1) % multiple_of) == 0 and d > 0

	# Calculate the approximate 
	curr_len = get_len( get_pts() )
	targ_len = round(curr_len - curr_len % multiple_of) + multiple_of
	bar.set_width(width * (targ_len / curr_len))
	new_len  = get_len( get_pts() )
	assert(is_nice(new_len)), [pct_size, curr_len, targ_len, new_len]
	scale.txt_label.set_text(f'{round(new_len):.0f} km')

	# Finally, for some aesthetic appeal, color in half the scale bar
	scale.size_bar.add_artist(Rectangle((0, 0), 0.5*bar.get_width(), height, fill=True, facecolor='k', edgecolor='k'))


def _get_colorbar_axis(fig_params, img_param, ax_dict, pgrid, vertical=True):
	''' Create and return the axis on which the colorbar will be plotted, adjusting its position as necessary '''
	bar_y_loc  = 0
	bar_x_loc  = fig_params['n_plots_w'] if     vertical else 0 # Location is top right subplot if vertical, and top left if not
	bar_height = fig_params['n_plots_h'] if     vertical else 1 # Height of bar is 1 if horizontal
	bar_width  = fig_params['n_plots_w'] if not vertical else 1 # Width  of bar is 1 if vertical

	first_row  = fig_params['first_row'] 
	first_col  = fig_params['first_col'] 
	last_row   = fig_params['last_row']
	last_col   = fig_params['last_col']

	# Only extend the bar across the current image row (or column) if there are multiple products
	if len(fig_params['products']) > 1:
		if vertical: bar_y_loc = img_param['row_idx']
		else:        bar_x_loc = img_param['col_idx']
		bar_height = bar_width = 1
		first_row  = last_row  = img_param['row_idx']
		first_col  = last_col  = img_param['col_idx']

	bottom = ax_dict[( last_row,  last_col)].get_position()
	top    = ax_dict[(first_row,  last_col)].get_position()
	left   = ax_dict[( last_row, first_col)].get_position()
	right  = ax_dict[( last_row,  last_col)].get_position()

	spec = pgrid.new_subplotspec((bar_y_loc, bar_x_loc), rowspan=bar_height, colspan=bar_width)
	cax  = plt.subplot(spec)

	# Align colorbar top+bottom / left+right with the top+bottom / left+right subplots
	if vertical: right = left = cax.get_position()
	else:        bottom = top = cax.get_position()
	bottom, top, left, right  = bottom.y0, top.y1, left.x0, right.x1
	
	# With multiple products, adjust axis edges inward slightly to ensure multiple colorbar ticks do not overlap
	if len(fig_params['products']) > 1:
		pct    = 0.05
		width  = right - left
		height = top - bottom
		if vertical: top, bottom = top - pct * height, bottom + pct * height 
		else:        left, right = left + pct * width, right - pct * width 

	cax.set_position((left, bottom, right - left, top - bottom))	
	return cax


@ignore_warnings
def add_colorbar(im, fig_params, img_param, ax_dict, pgrid):
	vertical   = fig_params['cbar_vertical']
	log_norm   = fig_params['cbar_lognorm'] and 'diff-' not in img_param['est_method']
	product    = img_param['product']
	cmin, cmax = PRODUCT_BOUNDS[product] if 'diff-' not in img_param['est_method'] else (0, 100)

	bar_label  = KEY_MAPPING['product'][product]
	bar_orient = 'vertical' if vertical else 'horizontal'
	bar_axis   = _get_colorbar_axis(fig_params, img_param, ax_dict, pgrid, vertical)
	colorbar   = plt.colorbar(im, orientation=bar_orient, cax=bar_axis)

	# colorbar.set_label(make_bold(bar_label), fontsize=28)
	colorbar.ax.tick_params(labelsize=22)
	ticks = np.append(colorbar.ax.get_yticks(), cmax)
	ticks = np.append([cmin], ticks)
	ticks = sorted(np.unique(ticks))
	fmt   = sci_notation if log_norm else lambda x: x 
	colorbar.set_ticks(ticks, [fmt(n) for n in ticks])
	colorbar.set_ticklabels([fmt(n) for n in ticks])

	# Adjust colorbar top+bottom / left+right ticks inward slightly, to ensure multiple colorbar ticks do not overlap
	# Moving the outermost ticks 5 points inward, given matplotlib uses a PPI of 72 
	if vertical:
		ticks = colorbar.ax.yaxis.get_majorticklabels()
		first = transforms.ScaledTranslation(0,  5 / 72., fig_params['figure'].dpi_scale_trans)
		last  = transforms.ScaledTranslation(0, -5 / 72., fig_params['figure'].dpi_scale_trans)
	else:
		ticks = colorbar.ax.xaxis.get_majorticklabels()
		first = transforms.ScaledTranslation( 5 / 72., 0, fig_params['figure'].dpi_scale_trans)
		last  = transforms.ScaledTranslation(-5 / 72., 0, fig_params['figure'].dpi_scale_trans)
		
		# Move ticks to the top of the bar
		colorbar.ax.xaxis.set_ticks_position('top')
		colorbar.ax.xaxis.set_label_position('top')

	# ticks[ 0].set_transform(ticks[ 0].get_transform() + first)
	# ticks[-1].set_transform(ticks[-1].get_transform() + last)
	colorbar.update_ticks()
	return colorbar


_triangulations = {}
def fix_projection(y, lon, lat, reproject=True, exact=False):
	''' 
	Project y into its native rectangular coordinate grid 
		(e.g. diagonal image -> rectangular)
	Return the full diagonalized image, including lon/lat
		extent, with reproject=False
	Must pass name parameter if inverse transformations 
	are desired.
	'''
	# extent = 0, y.shape[1], 0, y.shape[0]
	# return y, extent, (lon, lat)
	from scipy.interpolate import griddata, LinearNDInterpolator
	from scipy.spatial import ConvexHull, Delaunay
	import skimage.transform as st

	shape = lon.shape 
	dtype = y.dtype	
	y     = np.ma.masked_invalid(y.astype(np.float32)).filled(fill_value=np.nan)

	if len(y.shape) == 3: y = y.reshape((-1, y.shape[-1]))
	else:                 y = y.ravel()

	# Get lon/lat min/max values
	lonlat   = np.array(np.vstack([lon.ravel(), lat.ravel()]))
	min_val, min_val2 = np.partition(lonlat, 1).T[:2]
	max_val2, max_val = np.partition(lonlat,-2).T[-2:]

	extent   = np.vstack((min_val, max_val)).T.ravel()
	tri_key  = tuple(extent)
	step_val = np.mean([(min_val2 - min_val)/2, (max_val - max_val2)/2], 0)

	# Create a Delaunay triangulation for the original grid, if it doesn't already exist
	if tri_key not in _triangulations:
		print('Calculating Delaunay triangulation...')
		_triangulations[tri_key] = Delaunay(lonlat.T)

	# Apply a linear interpolation over the values for the new grid
	if exact or step_val.min() <= 1e-5 or (max_val[0]-min_val[0])/step_val[0] > 5000:
		size = [min(1000, lon.shape[1]), min(1000, lat.shape[0])]
		lon2 = np.linspace(min_val[0], max_val[0], size[0])
		lat2 = np.linspace(min_val[1], max_val[1], size[1])[::-1]
	else: 
		lon2 = np.arange(min_val[0], max_val[0], step_val[0])
		lat2 = np.arange(min_val[1], max_val[1], step_val[1])[::-1]

	interp = LinearNDInterpolator(_triangulations[tri_key], y, fill_value=np.nan)
	#interp = LinearNDInterpolator(lonlat.T, y, fill_value=np.nan)
	grid   = np.meshgrid(lon2, lat2)
	square = interp(tuple(grid))

	if not reproject:
		square = np.ma.masked_invalid(square).astype(dtype)
		mask   = square.mask if len(square.shape) == 2 else square.mask.any(-1)
		valid  = np.ix_(~np.all(mask, 1), ~np.all(mask, 0)) # remove any completely masked rows / columns
		# return square[valid], extent, (lonlat[0][valid], lonlat[1][valid]) # need to adjust extent if cutting off edges
		return square, extent, grid
		
	def minimum_bounding_rectangle(points):
		"""
		https://gis.stackexchange.com/questions/22895/finding-minimum-area-rectangle-for-given-points
		https://stackoverflow.com/questions/38409156/minimal-enclosing-parallelogram-in-python
		Find the smallest bounding rectangle for a set of points.
		Returns a set of points representing the corners of the bounding box.

		:param points: an nx2 matrix of coordinates
		:rval: an nx2 matrix of coordinates
		"""
		pi2 = np.pi/2.

		# get the convex hull for the points
		hull_points = points[ConvexHull(points).vertices]

		# calculate edge angles
		edges = np.zeros((len(hull_points)-1, 2))
		edges = hull_points[1:] - hull_points[:-1]

		angles = np.zeros((len(edges)))
		angles = np.arctan2(edges[:, 1], edges[:, 0])

		angles = np.abs(np.mod(angles, pi2))
		angles = np.unique(angles)

		# find rotation matrices
		rotations = np.vstack([
			np.cos(angles),
			np.cos(angles-pi2),
			np.cos(angles+pi2),
			np.cos(angles)]).T
		rotations = rotations.reshape((-1, 2, 2))

		# apply rotations to the hull
		rot_points = np.dot(rotations, hull_points.T)

		# find the bounding points
		min_x = np.nanmin(rot_points[:, 0], axis=1)
		max_x = np.nanmax(rot_points[:, 0], axis=1)
		min_y = np.nanmin(rot_points[:, 1], axis=1)
		max_y = np.nanmax(rot_points[:, 1], axis=1)

		# find the box with the best area
		areas = (max_x - min_x) * (max_y - min_y)
		best_idx = np.argmin(areas)

		# return the best box
		x1 = max_x[best_idx]
		x2 = min_x[best_idx]
		y1 = max_y[best_idx]
		y2 = min_y[best_idx]
		r  = rotations[best_idx]

		rval = np.zeros((4, 2))
		rval[0] = np.dot([x1, y2], r)
		rval[1] = np.dot([x2, y2], r)
		rval[2] = np.dot([x2, y1], r)
		rval[3] = np.dot([x1, y1], r)

		return rval

	# There's a way to use a convex hull, but would need 
	# to flatten the sides of the hull into a rectangle, 
	# in order to have an equal number of points between
	# src and dst in the projective transform
	# Can do this by iterating through hull points, and 
	# finding the nearest point on the rectangle perimeter
	#bounds   = minimum_bounding_rectangle(lonlat.T)
	#bot_left, top_left, top_right, bot_right = bounds 

	# assert(0), 'likely incorrect, currently - or at least inefficient'
	''' 
	Need to check if we can just use the square matrix nan values to determine corners
	lonlat has now been reassigned above, so check if it's necessary to have down here
	'''
	# Get lon/lat 0-based corner values
	grid   = griddata(lonlat.T, np.ones_like(y), tuple(grid))
	lonlat = np.vstack(np.where(np.isfinite(grid.T if len(grid.shape) == 2 else grid.T[0])))
	top_left, top_right = lonlat[:, np.argmin(lonlat, 1)].T 
	bot_right, bot_left = lonlat[:, np.argmax(lonlat, 1)].T
	left_side  = np.sum(np.abs(bot_left  - top_left )**2)**0.5
	right_side = np.sum(np.abs(bot_right - top_right)**2)**0.5 
	top_side   = np.sum(np.abs(top_left  - top_right)**2)**0.5
	bot_side   = np.sum(np.abs(bot_left  - bot_right)**2)**0.5

	if exact:
		height, width = shape
	else:
		width  = int(np.max([top_side, bot_side])+1)
		height = int(np.max([left_side, right_side])+1)

	# Project the image onto a rectangle (fixing skew & rotation)
	proj = st.ProjectiveTransform()
	src  = np.array([[0,0], [0,height],[width,height],[width,0]])
	dst  = np.asarray([top_left, bot_left, bot_right, top_right])
	assert(proj.estimate(src, dst)), 'Failed to estimate warping parameters'

	lon, lat  = np.meshgrid(lon2, lat2)
	longitude = st.warp(lon, proj, output_shape=(height,width))
	latitude  = st.warp(lat, proj, output_shape=(height,width))
	projected = st.warp(square, proj, output_shape=(height,width), cval=np.nan)

	if projected.shape[0] > projected.shape[1]: # vertical image
	# if projected.shape[1] > projected.shape[0]: # horizontal image
		n = 1 # 1 = 90 degrees, 2 = 180, ...
		longitude = np.rot90(longitude, n)
		latitude  = np.rot90(latitude, n)
		projected = np.rot90(projected, n)
	extent = left, right, bottom, top = 0, projected.shape[1], 0, projected.shape[0]
	return np.ma.masked_invalid(projected).astype(dtype), extent, (longitude, latitude)


def interpolate_negative(img_data, band):
	# Interpolate negative/zero values to be positive
	from scipy.interpolate import griddata, LinearNDInterpolator
	invalid = np.logical_and(img_data > -0.5, img_data <= 0)
	valid   = img_data > 0
	if valid.sum() and invalid.sum():
		print(f'({band}) Pos: {valid.sum()} Neg: {invalid.sum()} Min: {img_data[invalid].min()} Max: {img_data[valid].max()}')
		
		valid_idxs = np.dstack(np.where(valid))[0]
		if len(img_data.shape) == 2: # 2D interpolation
			img_data[invalid] = griddata(valid_idxs, img_data[valid], np.where(invalid))
		else: # 3d interpolation
			img_data[invalid] = LinearNDInterpolator(valid_idxs, im_data[valid])(np.dstack(np.where(invalid))[0])
	return img_data 


def var_name(v):
	'''
	Gets the name of the passed variable.
	https://stackoverflow.com/questions/18425225/getting-the-name-of-a-variable-as-a-string
	'''
	import inspect
	for fi in reversed(inspect.stack()):
		names = [name for name, val in fi.frame.f_locals.items() if val is v]
		if len(names) > 0: return names[0]


def parse_bitmask(bitmask, debug=False):
	''' 
	Parse a bitmask, returning a boolean array with bits unpacked
	along the last dimension
	'''
	debug = True
	if debug:
		# Take a random subset rather than entirety, to save time / memory
		random_idx = np.random.choice(np.arange(bitmask.size), 10000)
		orig = bitmask.flatten()[random_idx]

	shape   = bitmask.shape
	bitmask = bitmask.flatten()[:, None].byteswap()
	bitmask = np.unpackbits(bitmask.view(np.uint8), axis=1)[:, ::-1]
	bitmask = bitmask.astype(bool)

	if debug:
		recalc  = np.zeros(orig.shape)
		row,col = np.where(bitmask[random_idx])
		np.add.at(recalc, row, 2 ** col)
		if not (recalc == orig).all():
			idxs = np.where(recalc != orig)[0]
			print('Original:',orig[idxs[0]])
			print('Recalced:',recalc[idxs[0]])
			print('Bitmask: ',bitmask[idxs[0]])
			raise Exception('Bitmask flags calculated incorrectly')
	return bitmask.reshape(shape+(-1,))


def print_bitmask_stats(bitmask):
	print('\nBitmask stats:')
	for i, (key, mask) in enumerate(bitmask.items()):
		print(f'\t{key:>12} | {mask.sum():,} / {mask.size:,}')


def polymer_bitmask(bitmask, *args, verbose=True, debug=False, **kwargs):
	''' Polymer bitmask flags '''
	flags = [
		'land',         # 'LAND'          : 1
		'cloud',        # 'CLOUD_BASE'    : 2
		'invalid_L1',   # 'L1_INVALID'    : 4
		'neg_bb',       # 'NEGATIVE_BB'   : 8
		'out_bounds',   # 'OUT_OF_BOUNDS' : 16
		'exception',    # 'EXCEPTION'     : 32
		'aerosol',      # 'THICK_AEROSOL' : 64
		'high_airmass', # 'HIGH_AIR_MASS' : 128
		'_unused',      # 'UNUSED'        : 256
		'other_mask',   # 'EXTERNAL_MASK' : 512
		'case2_water',  # 'CASE2'         : 1024
		'inconsistent', # 'INCONSISTENCY' : 2048
	]

	assert((bitmask < 4096).all()), bitmask.max()
	bitmask = parse_bitmask(bitmask.astype(np.uint16), debug)
	labeled = {k: bitmask[..., i] for i, k in enumerate(flags)}
	assert(not np.any(labeled['_unused']))
	assert(not np.any(bitmask[..., len(flags):])), [bitmask[np.where(bitmask[..., len(flags):])[0]][0]]

	nan_val   = ['land', 'cloud', 'invalid_L1', 'exception', 'other_mask'] # polymer calculated nans
	suggested = nan_val + ['neg_bb', 'out_bounds', 'aerosol', 'high_airmass'] # suggested invalid mask (all < 1024)
	w_incons  = suggested + ['inconsistent'] # with inconsistency flag
	bitmask   = {k: labeled[k] for k in w_incons}

	if verbose: print_bitmask_stats(bitmask)
	return np.any(list(bitmask.values()), 0)


def l2gen_bitmask(bitmask, mask_flags='l2gen', verbose=True, debug=False):
	''' l2gen bitmask flags (https://oceancolor.gsfc.nasa.gov/atbd/ocl2flags/) '''
	flags = [
		'atmfail',    # 00    ATMFAIL      Atmospheric correction failure   
		'land',       # 01    LAND         Pixel is over land 
		'prodwarn',   # 02    PRODWARN     One or more product algorithms generated a warning
		'higlint',    # 03    HIGLINT      Sunglint: reflectance exceeds threshold
		'hilt',       # 04    HILT         Observed radiance very high or saturated
		'hisatzen',   # 05    HISATZEN     Sensor view zenith angle exceeds threshold
		'coastz',     # 06    COASTZ       Pixel is in shallow water
		'spare1',     # 07    spare
		'straylight', # 08    STRAYLIGHT   Probable stray light contamination
		'cldice',     # 09    CLDICE       Probable cloud or ice contamination
		'coccolith',  # 10    COCCOLITH    Coccolithophores detected 
		'turbidw',    # 11    TURBIDW      Turbid water detected
		'hisolzen',   # 12    HISOLZEN     Solar zenith exceeds threshold
		'spare2',     # 13    spare
		'lowlw',      # 14    LOWLW        Very low water-leaving radiance
		'chlfail',    # 15    CHLFAIL      Chlorophyll algorithm failure
		'navwarn',    # 16    NAVWARN      Navigation quality is suspect 
		'absaer',     # 17    ABSAER       Absorbing Aerosols determined 
		'spare3',     # 18    spare
		'maxaeriter', # 19    MAXAERITER   Maximum iterations reached for NIR iteration
		'modglint',   # 20    MODGLINT     Moderate sun glint contamination
		'chlwarn',    # 21    CHLWARN      Chlorophyll out-of-bounds 
		'atmwarn',    # 22    ATMWARN      Atmospheric correction is suspect 
		'spare4',     # 23    spare 
		'seaice',     # 24    SEAICE       Probable sea ice contamination
		'navfail',    # 25    NAVFAIL      Navigation failure
		'filter',     # 26    FILTER       Pixel rejected by user-defined filter OR Insufficient data for smoothing filter 
		'spare5',     # 27    spare 
		'bowtiedel',  # 28    BOWTIEDEL    Deleted off-nadir, overlapping pixels (VIIRS only) 
		'hipol',      # 29    HIPOL        High degree of polarization determined
		'prodfail',   # 30    PRODFAIL     Failure in any product
		'spare6',     # 31    spare
	]
	bitflag = parse_bitmask(bitmask, debug)
	labeled = {k: bitflag[..., i] for i, k in enumerate(flags)}
	masks   = {
		'L2': ['land', 'hilt', 'straylight', 'cldice'],
		'L3': ['atmfail', 'land', 'higlint', 'hilt', 'hisatzen', 'straylight', 'cldice', 
				'coccolith', 'hisolzen', 'lowlw', 'chlfail', 'navwarn', 'absaer', 'maxaeriter',
				'atmwarn', 'navfail'],
		'Custom' : ['land', 'hilt', 'straylight', 'cldice', 'atmfail', 'higlint', 'hisatzen', 'hisolzen', 'atmwarn'],
		'l2gen'  : ['cldice', 'land', 'hilt', 'straylight', 'cldice', 'atmfail', 'higlint', 'hisolzen'],# chlfail],
		'polymer': ['land', 'hilt'],
		'rhos'   : ['cldice'],
		'land'   : ['land'],
	}
	
	assert(mask_flags in masks), f'Unrecognized mask "{mask_flags}"'
	bitflag = {k: labeled[k] for k in masks[mask_flags]}

	if verbose: print_bitmask_stats(bitflag)
	return np.any(list(bitflag.values()) + [np.zeros_like(bitmask)], 0)

def find_geographic_pixelloc(chosen_locations, lat, lon):
    """
    This function can be used to find the pixel location of the point that is closest to the specific chosen location

    :param chosen_locations: [[lat_1, lon_1], [lat_2, lon_2]....[latn, lon_n]]
    This parameter is list of 2D lists with the the appropropriate latitude and longitude values

    :param lat: [ndarray]
    A 2d array that holds the list of latitude values

    :param lon: [ndarray]
    A 2d array that holds the list of longitude values

    :return:
    """

    assert isinstance(chosen_locations, list), "The <chosen_locations> variable must be a list"
    assert all(isinstance(item, list) for item in chosen_locations), "All elements of <chosen_locations> must also " \
                                                                     "be lists"
    assert all(len(item)==2 for item in chosen_locations), "All elements of <chosen_locations> must have 2 entries, a latitude and " \
                                               "a longitude"
    chk_val = max(chosen_locations)
    assert all(isinstance(item, np.floating) or isinstance(item, float) for
               substr in chosen_locations for item in substr), "All the values in <chosen_locations> must" \
                                                                           " be floating-point numbers"
    assert -90 <= chk_val[0] <= 90, "The latitude values must be between -90 and 90"
    assert -180 < chk_val[1] <= 180, "The longitude values must be between -180 and 180"

    "Iterate over the locations"
    location = []
    for item in chosen_locations:
        if not ((lat.min() <= item[0] <= lat.max()) or (lon.min() <= item[1] <= lon.max())):
            return False, []
        else:
            'Find the closest point to specified point'
            lat_loc = ((lat - item[0]) / 90.)**2
            lon_loc = ((lon - item[1]) / 180.)**2

            dist = (lat_loc + lon_loc)**0.5
            temp = np.where(dist == dist.min())

            location += [[temp[0][0], temp[1][0]]]

            """if len(temp[0]) == 1:
                location += [[temp[0][0], temp[1][0]]]
            else:
                location += [[temp[0][0], temp[0][1]]]"""

    return True, location



