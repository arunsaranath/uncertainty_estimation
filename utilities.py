# -*- coding: utf-8 -*-
"""
File Name:      utilities
Description:    This code file contains a set of supporting files which can be used for generating ensemble prediction
maps and uncertainty maps


Date Created:   February 3rd, 2023
"""

import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

plt.rc('text', usetex=True)
plt.rcParams['mathtext.default']='regular'

SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 24
mrkSize = 15

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


from evaluate_MDN_AC import load_data, OLI_WAVELENGTHS
from MDN import get_estimates, get_args, mask_land, get_tile_data
from uncertainty_package_final.uncert_support_lib import get_sample_uncertainity
from Plot.tile_utils import DatasetBridge
from Plot.plot_scenes import load_Rrs
from utils import center_image, fix_projection, get_window
from plot_utils import get_tile_data as get_tile_rgb

'Base properties for imshow'
ASPECT = 'auto'
cmap = 'jet'

SENSOR_ARGS = {
    "MSI" : {
            "land_mask" : True,
            "landmask_threshold" : 0.2,
            "flg_subsmpl" : True,
            "subsmpl_rate" : 4
    },
    "HICO" : {
        "land_mask" : False,
        "landmask_threshold" : 0.2,
        "flg_subsmpl" : False,
        "subsmpl_rate": 5
        },
    "PRISMA-chl" : {
        "land_mask" : True,
        "landmask_threshold" : 0.2,
        "flg_subsmpl" : False,
        "subsmpl_rate": 1
        },
    "PRISMA-noBnoNIR" : {
        "land_mask" : True,
        "landmask_threshold" : 0.2,
        "flg_subsmpl" : False,
        "subsmpl_rate": 1
        },
}


def get_mdn_preds_full(args, train_x=None, train_y=None, test_x=None, test_y=None, slices=None):
    """
    This function is  used get the full MDN models predictions including the coefs for analysis

    :param args (dictionary)
    A set of arguments to setup the MDN

    :param train_x: (np.ndarray: nSamples X (nBands + 4)) [Default: None]
    A numpy array which contains the data for Training the models [MODEL INPUT]

    :param train_y: (np.ndarray: nSamples X nBands ) [Default: None]
    A numpy array which contains the appropriately atmospheric corrected data which we can use to train the models

    :param test_x: (np.ndarray: nSamples X (nBands + 4))[Default: None]
    A numpy array which contains the input data for testing the models [MODEL INPUT]

    :param train_y: (np.ndarray: nSamples X nBands ) [Default: None][Default: None]
    A numpy array which contains the atmospehrically corrected data for comparing to the  models output

    :return: dist [list]
    A list of dictionaries which contains the details of the distribution from the models

    :return: scalers [list]
    A list of scalers which contain the scalers used to fit the data prior to training

    :return: estimates [list]
    A list of point-predictions from the individual models in the suite
    """
    if train_y is not None and train_x is not None:
        assert isinstance(train_x, np.ndarray) and len(train_x.shape)==2, "<train_x> must be a 2d numpy array"
        assert isinstance(train_y, np.ndarray) and len(train_y.shape)==2, "<train_y> must be a 2d numpy array"
        assert train_x.shape[0] == train_y.shape[0], "The parameters <train_x> and <train_y> should have the same number" \
                                                     " of rows (samples)"

    if test_x is not None:
        assert isinstance(test_x, np.ndarray) and len(test_x.shape) == 2, "<test_x> must be a 2d numpy array"
        if test_y is not None:
            assert isinstance(test_y, np.ndarray) and len(test_y.shape) == 2, "<test_y> must be a 2d numpy array"
            assert test_x.shape[0] == test_y.shape[0], "The parameters <test_x> and <test_y> should have the same number" \
                                                         " of rows (samples)"

    if train_y is None and train_x is None and test_x is None:
        assert 1, "Function can only be run for training or testing MDN but neither training data or test data" \
                  " is provided"


    'If needed perform the model training'
    if train_x is not None and test_y is not None:
        outputs, _ = get_estimates(args, x_train=train_x, y_train=train_y, x_test=None, y_test=None,
                                 output_slices=slices, return_coefs=True)

    'If prediction mode get the predictions'
    if test_x is not None and test_y is None:
        outputs, _ = get_estimates(args, x_train=train_x, y_train=train_y, x_test=test_x, y_test=None,
                                 output_slices=slices, return_coefs=True)

    if test_x is not None and test_y is not None:
        outputs, _ = get_estimates(args, x_train=train_x, y_train=train_y, x_test=test_x, y_test=test_y,
                                 output_slices=slices, return_coefs=True)

    return outputs

def get_mdn_uncert_ensemble(ensmeble_distribution, scaler_y_list):
    """
    This function accepts the a dictionary with the distribution details for the entire ensemble and calculates the
    uncertainty for the entire ensmeble

    :param ensmeble_distribution (list of dictionaries)
    A dictionary that has all the distribution information provided by Brandons MDN package

    :param scaler_y_list (list of model scalers)
    To convert uncertianty to appropriate scale

    :return: ensmeble_uncertainties (list)
    A list containing the uncertainties for the entire ensemble set
    """

    'Create a variable to hold the uncertainties'
    ensmeble_uncertainties= []
    'create a counter to track model number'
    ctr = 0
    'iterate over models'
    for item in ensmeble_distribution:
        'Seperate out the different components of the distribution'
        dist = {'pred_wts': item[0], 'pred_mu': item[1],
                'pred_sigma': item[2]}

        scaler_y = scaler_y_list[0]

        'Use the MDN output to get the uncertainty estimates'
        aleatoric, epistemic = get_sample_uncertainity(dist)
        aleatoric, epistemic = np.sum(aleatoric, axis=1), np.sum(epistemic, axis=1)
        ensmeble_uncertainties += [scaler_y.inverse_transform(np.sqrt(aleatoric + epistemic))]

        ctr +=1

    return np.asarray(ensmeble_uncertainties)


def arg_median(X, axis=0):
    """
    This function can be used to identify the location of the sample which corresponds to the median in the specific
    samples.

    :param X:[np.ndarray]
    A numpy array in which we want to find the position of the median from the samples

    :param axis: [int] (Default: 0)
    An integer axis along which we are doing this process

    :return:

    A numpy array of the median location
    """
    assert isinstance(X, np.ndarray), "The variable <X> must be a numpy array"
    assert isinstance(axis, int) and axis >= 0, f"The variable <axis> must be a integer >= 0"
    assert axis <= len(X.shape), f"Given matrix has {len(X.shape)} dimensions, but asking meidan along axis {axis}" \
                                 f"dimension"

    'Find the median along axis of interest'
    amedian = np.nanmedian(X, axis=axis)

    'Find difference from median'
    aabs = np.abs((X.T - np.expand_dims(amedian.T, axis=-1))).T

    'Find the sample with smallest difference'
    return np.nanargmin(aabs, axis=axis)

def colorbar(mappable, ticks_list=None, lbl_list=None,):
    """
    This function can be used to create a custom colorbar
    :param mappable:
    :param ticks_list:
    :param lbl_list:
    :return:
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    if ticks_list is not None:
        cbar.set_ticks(ticks_list)
        if lbl_list is not None:
            cbar.set_ticklabels(lbl_list)
    plt.sca(last_axes)
    return cbar

def create_satellite_uncertainty_plots(rgb_img, model_preds, extent,img_uncert=None, mu=0, sigma=1,
                                       product_name='Parameter', pred_ticks= [-1, 0, 1, 2],
                                       pred_labels= [r'$10^{-1}$',r'$10^{0}$', '$10^{1}$', r'$10^{2}$'],
                                       ll=0, ul=0.25):
    """
    This function can be used to overlay the MDN-prediction maps over the RGB compostite of a satellite image for display

    :param rgb_img: [np.ndarray, rows X cols X 3]
    The RGB commposite of the scene

    :param model_preds: [np.ndarray, rows X cols]
    The MDN predictions for that location

    :param extent: [np.array]
    A descrtption of the extent of the location

    :param img_uncert:  [np.ndarray, rows X cols]
    The uncertainty associated with the MDN predictions for that location

    :param mu: [float] (Default: 0)
    The shifting that needs to applied to the estimated uncertainty (Default is 0 or no shifting)

    :param sigma: [float] (Default: 1)
    The scaling that needs to applied to the estimated uncertainty (Default is 0 or no scaling)

    :param product_name: (string) (Default: "Parameter")
    The name of the product that has been predicted

    :param pred_tricks: (list) (Default: [-1, 0, 1, 2]])
    A list with the colorbar for the ticks along with the MDN predictions. By default assumes log-scale range between
    0.1 and  100

    :param pred_labels: (list) (Default: [r'$10^{-1}$',r'$10^{0}$', '$10^{1}$', r'$10^{2}$'])
    A list of labels for the  colorbar ticks of the MDN predictions.

    :param ll: (float) (Default: 0)
    Lower limit for uncertainty maps colorbar

    :param ul: (float) (Default: 1)
    Upper limit for uncertainty maps colorbar

    :return: fig1: A figure with appropriate plots
    """

    'Check data properties'
    assert rgb_img.shape[:2] == model_preds.shape[:2], f"The base RGB and prediction image should have the same" \
                                                       f" spatial dimensions"
    assert rgb_img.shape[2] == 3, "The <rgb_img> can only have three bands"
    if len(model_preds.shape) == 3:
        assert model_preds.shape[2] == 1, "This function is only set up to the overlay the predictions of a single " \
                                          "parameter at a time"

    assert len(extent) == 4, "Need to provide the spatial extent of the image to be displayed"
    if img_uncert is not None:
        assert rgb_img.shape[:2] == img_uncert.shape[
                                    :2], f"The base RGB and uncertainty image should have the same spatial dimensions"
        if len(img_uncert.shape) > 2:
            assert model_preds.shape[2] == 1, "This function is only set up to the overlay the predictions of a single " \
                                              "parameter at a time"
        assert isinstance(mu, float), "The shifting fcator <mu> must be a float"
        assert isinstance(sigma, float), "The scaling factor <sigma> must be a float"


    'Create the basic figure and set its properties'
    if img_uncert is not None:
        fig1, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 5))
    else:
        fig1, ax1= plt.subplots(figsize=(7,7))

    fig1.patch.set_visible(True)
    ord = 0

    'Display the results - model predictions'
    img1 = ax1.imshow(rgb_img, extent=extent, aspect=ASPECT, zorder=ord)
    img2 = ax1.imshow(np.ma.masked_where(model_preds <= -5.9, model_preds), cmap=cmap,
                      extent=extent, aspect=ASPECT, zorder=ord + 1)
    ax1.set_title("Model predictions-" + product_name, fontsize=14, fontweight="bold")
    'Apply colorbar'
    colorbar(img2, ticks_list=pred_ticks, lbl_list=pred_labels)


    'Display the results - model uncertainty'
    if img_uncert is not None:
        img3 = ax2.imshow(rgb_img, extent=extent, aspect=ASPECT, zorder=ord)
        'Normalize uncertainty'
        img_uncert = (img_uncert - mu) / sigma
        img4 = ax2.imshow(np.ma.masked_where(img_uncert == (-mu /sigma), img_uncert), cmap=cmap,
                          extent=extent, aspect=ASPECT, zorder=ord + 1)
        ax2.set_title(r"Total Uncertainty ($\sigma_{UNC}$)", fontsize=14, fontweight="bold")
        img4.set_clim(ll, ul)
        colorbar(img4)

    #ax2.set_xlim([-123, -121.8])
    #ax2.set_ylim([37, 38.4])


    return fig1

def extract_sensor_data(image_name, sensor_option):
    """
    Get the rgb image associated with a specific image

    :param image_name:
    :return:
    """
    if "PRISMA" not in sensor_option:
        '--------------------------------------------------------------------------------------------------------------'
        'Get the latittude and longitude information for the file'
        with DatasetBridge(image_name) as data:
            if 'navigation_data' in data.groups.keys():
                data = data['navigation_data']
            lon_k, lat_k = ('lon', 'lat') if 'lon' in data.variables.keys() else ('longitude', 'latitude')
            lon, lat = data[lon_k][:], data[lat_k][:]
            'If subsampling -perform on latitude and longitude data'
            if SENSOR_ARGS[sensor_option]["flg_subsmpl"]:
                ss_rate = SENSOR_ARGS[sensor_option]["subsmpl_rate"]
                lat = lat[::ss_rate, ::ss_rate]
                lon = lon[::ss_rate, ::ss_rate]

        'Eliminate any spurious values'
        lon[lon < -180] = np.nan
        lat[lat < -90] = np.nan

        'Are we masking the lonitude/latitude'
        lon.mask = False
        lat.mask = False


        'Read in the image from the netcdf file'
        wvl_bands, img_data = get_tile_data(image_name, sensor_option, rhos=False)
        'Get the Extent'
        extent = np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)

        return img_data, np.asarray(wvl_bands), (lon, lat), extent,(None, None)

    if "PRISMA" in sensor_option:
        'Set the params and load the PRISMA data'
        img_params = {
            'est_method': 'None',
            'atm_method': 'None',
            'filename': image_name,
            'sensor': sensor_option,
            'part_yx': (slice(None, None, None), slice(None, None, None)),
            'geolocate': False,
            'location': 'default',
            'product': 'Chl'
        }
        lon, lat, im_data, mask, im_out, wvl_bands = load_Rrs(img_params)
        img_data = np.ones(im_out)
        #img_data[:] = np.nan
        for ii in range(im_data.shape[1]):
            temp = (np.squeeze(img_data[:, :, ii])).flatten()
            temp[~mask] = im_data[:, ii]
            img_data[:, :, ii] = temp.reshape((im_out[0], im_out[1]))

        'Get the Extent'
        extent = np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)
        #img_data, extent, (lon_proj, lat_proj) = fix_projection(img_data, lon, lat, reproject=False

        return img_data, np.asarray(wvl_bands), (lon, lat), extent,(None, None)

def find_rgb_img(file_name):
    import netCDF4

    f = netCDF4.Dataset(file_name)

    'Get the Image data'
    img = (f.groups['products']).variables['Lt']  # temperature variable
    print(img)

    'Get the RGB Bands'
    rgb_bands = [656, 564, 444]
    wvl_bands = img.wavelengths
    img_rgb = np.zeros((img.shape[0], img.shape[1], 3))

    for ii in range(len(rgb_bands)):
        idx = np.argmin(np.abs(wvl_bands - rgb_bands[ii]))
        img_rgb[:, :, ii] = img[:, :, idx]
        print(wvl_bands[idx])

    bd_dn = [10, 15, 40]
    bd_up = [40, 80, 100]
    for ii in range(len(rgb_bands)):
        temp = np.squeeze(img_rgb[:, :, ii])
        temp = (temp - bd_dn[ii]) / (bd_up[ii] - bd_dn[ii])
        temp[temp <= 0] = 0
        temp[temp >= 1] = 0.4
        img_rgb[:, :, ii] = temp

    return img_rgb


class create_uncertainty_maps(object):
    def __init__(self, sensor,  n_layers=5, n_nodes=100, dropout=None,
                 model_name="/Users/arunsaranathan/SSAI/Code/uncert_hyper_experiments/experiment_1/weights_folder/MDN_3params/",
                 output='chl,tss,cdom'):
        """
        This class can be used to create maps for satellite based image cube measurements. A different object has to be
        created for each sensor. The object also allows the user to decide the specific model which is to be used in the
        mapping.

        :param sensor: [string in ['OLI', 'OLCI', 'MSI']]
        The sensor for which we are creating the image maps.

        :param n_layers: [int] (Default:5)
        Number of layers in the MDN

        :param n_nodes: [int] (Default:100)
        Number of node in the MDN

        :param dropout: (0<= float <=1) (Default:None)
        The dropout to be added to the model. The default values is no dropout

        :param model_name (str) (Defaults: 'Weights')
        The name which the model weights etc are stored.
        """

        'Set the parameters for the MDN'
        self.args = get_args()
        #args.sensor = sensor_option
        self.args.no_ratio = True
        self.args.n_layers = n_layers
        self.args.n_hidden = n_nodes
        self.args.model_loc = model_name
        self.args.dropout= dropout

        self.args.product = output

        "Get the data"
        np.random.seed(self.args.seed)
        self.args.sensor = sensor

    def map_cube(self, img_data, wvl_bands=None, land_mask=True, landmask_threshold=0.2, flg_subsmpl=False,
                 flg_uncert=True, subsmpl_rate=10, slices=None):
        """
        This function is used tomap the pixels in a 3D numpy array, in terms of both parameters and the associated
        model uncertainty.

        :param img_data: [np.ndarray: nRow X nCols X nBands]
        The 3D array for which we need MDN predictions

        :param rhos_flag: [Bool] (default: False)
        Whether the maps use the classical Rrs or the more questionable Rhos

        :param land_mask: [Bool] (default: True)
        Should a heuristic be applied to mask out the land pixels

        :param landmask_threshold: [-1 <= float <= 1] (default: 0.2)
        The value with which the land mask is being calculated.

        :param flg_subsmpl: [bool] (Default: False)
        Does the image have to be subsampled.

        :param subsmpl_rate: [int > 0] (Default: 10)
        The subsampling rate. Must be an integer. For e.g. if provided rate is 2, one pixel is chosen in each 2X2
        spatial bin.

        :return:
        """

        assert isinstance(img_data, np.ndarray), "The <image_data> variable must be a numpy array"
        assert len(img_data.shape) == 3, "The <image_data> variable must be a 3D numpy array"
        assert isinstance(land_mask, bool), "The <mask_land> parameter must be a boolean variable"
        assert isinstance(landmask_threshold, float) and (np.abs(landmask_threshold) <= 1), "The <landmask_threshold>" \
                                                                                            "must be in range [-1, 1]"
        assert isinstance(flg_subsmpl, bool), f"The variable <flg_subsmpl> must be boolean"
        if flg_subsmpl:
            assert isinstance(subsmpl_rate, int) and (subsmpl_rate > 0), f"The variable <subsmpl_rate> must be a " \
                                                                         f"positive integer"



        'Sub-sample the image if needed'
        if flg_subsmpl:
            img_data = img_data[::subsmpl_rate, ::subsmpl_rate, :]

        'Apply the mask to find and remove the Land pixels or just remove nan values'
        if land_mask:
            'Get the mask which mask out the land pixels'
            #wvl_bands_m, img_data_m = get_tile_data(image_name, 'OLCI-no760', rhos=rhos_flag)
            img_mask = mask_land(img_data, wvl_bands, threshold=landmask_threshold)

            'Get the locations/spectra for the water pixels'
            water_pixels = np.where(img_mask == 0)
            water_spectra = img_data[water_pixels[0], water_pixels[1], :]
        else:
            'Get a simple mask removing pixels with Nan values'
            img_mask = np.asarray((np.isnan(np.min(img_data, axis=2))), dtype=np.float)

            'Get the locations/spectra for the water pixels'
            water_pixels = np.where(img_mask == 0)
            water_spectra = img_data[water_pixels[0], water_pixels[1], :]

        'Mask out the spectra with invalid pixels'
        if water_spectra.size != 0:
            water_spectra = np.expand_dims(water_spectra, axis=1)
            water_final = np.ma.masked_invalid(water_spectra.reshape((-1, water_spectra.shape[-1])))
            water_mask = np.any(water_final.mask, axis=1)
            water_final = water_final[~water_mask]
            #water_pixels = water_pixels[~water_mask]

            'Get the estimates and predictions for each sample'
            outputs = get_mdn_preds_full(self.args, test_x=water_final, test_y=None,
                                                          output_slice=slices)
            estimates = np.asarray(outputs['estimates'])
            'Get the location of the prediction closest to the median -- may need to select uncertainty of median'
            est_med_loc = arg_median(estimates, axis=0)



            'Create the Chl-a prediction map'
            model_preds = np.zeros((img_data.shape[0], img_data.shape[1], estimates.shape[1]))
            model_preds[water_pixels[0][~water_mask], water_pixels[1][~water_mask], :] = estimates

            if not flg_uncert:
                return model_preds

            'Perform the Uncertainity estimation'
            ensmeble_uncertainties = get_mdn_uncert_ensemble(ensmeble_distribution=outputs['coefs'],
                                                             scaler_y_list=outputs['scalery'])

            'Iterate over the data and ectract the values'
            final_uncertainties = []
            for ii in range(estimates.shape[1]):
                samp_uncert = []
                for jj in range(estimates.shape[2]):
                    samp_uncert += [ensmeble_uncertainties[est_med_loc[ii, jj], ii, jj]]

                final_uncertainties += [np.asarray(samp_uncert)]

            'Get the image uncertainty'
            img_uncert = np.zeros((img_data.shape[0], img_data.shape[1], estimates.shape[1]))
            img_uncert[water_pixels[0][~water_mask], water_pixels[1][~water_mask], ] = np.squeeze(final_uncertainties)

        else:
            if not flg_uncert:
                return np.zeros((img_data.shape[:-1]))

            model_preds = np.zeros((img_data.shape[:-1]))
            img_uncert= np.zeros((img_data.shape[:-1]))

        return model_preds, img_uncert


if __name__ == "__main__":
    'Create a mapping object'
    location = "chesapeake_bay"
    sensor = "HICO"
    sensor_option = sensor.split('-')

    sensor_option = sensor_option[0]
    model_loc = "/Users/arunsaranathan/SSAI/Code/uncert_hyper_experiments/experiment_1/weights_folder/MDN_3params/"
    products = 'chl,tss,cdom'
    mObj = create_uncertainty_maps(sensor, model_name=model_loc, output=products)

    'Set the name of the folder which contains the data and extract the data'
    """base_folder = f"/Volumes/AMS_HDD/Satellite Data/{location}/{sensor_option}"
    if sensor_option == "PRISMA":
        image_name = list(Path(base_folder).rglob("*.[bB][sS][qQ]"))
    else:
        image_name = list(Path(base_folder).rglob("*L2*.[nN][cC]"))"""
    image_name = ["/Volumes/AMS_HDD/Satellite Data/chesapeake_bay/HICO/2013263142213/H2013263142213.L2_ISS_OC.nc"]
    #image_name  = [image_name[1]]

    'Iterate over the the NetCDF files available for that location'
    for item in image_name:
        'Get the image data'
        img_data, wvl_bands, (lon, lat), extent, (lon_o, lat_o) = extract_sensor_data(item, sensor)

        if "PRISMA" not in sensor_option:
            img_rgb = find_rgb_img((str(item).replace('L2_ISS_OC.nc', 'L1B_ISS.nc')))

        if "PRISMA" in sensor_option:
            rgb_bands = [660, 560, 440]
            img_rgb = np.ones((img_data.shape[0], img_data.shape[1], 3))
            for ii in range(3):
                idx = np.nanargmin(np.abs(wvl_bands - rgb_bands[ii]))
                img_rgb[:, :, ii] = img_data[:, :, idx]

            'NORMALIZE THE RGB image for better clarity'
            interpolate = lambda data, hi=0.1: np.interp(data, [0, hi], [0, 1])

            for ii in range(img_rgb.shape[2]):
                temp = np.squeeze(img_rgb[:, :, ii])
                temp[temp < 0] = 0
                if sensor_option == "OLCI":
                    temp = interpolate(temp, 0.015)
                else:
                    temp = interpolate(temp, 0.05)
                img_rgb[:, :, ii] = 255. * temp

            img_rgb = img_rgb.astype(np.uint8)

        img_rgb[np.isnan(img_rgb)] = 1


        """fig2 = plt.figure()
        plt.imshow(rgb_img, extent=extent, aspect=ASPECT)
        fig2.savefig(str(item).replace('.nc', f'_rgb.png'), bbox_inches='tight')"""


        'Extract the Chl-a predictions and estimated uncertainty from the MDN'
        model_preds, img_uncert = mObj.map_cube(img_data, wvl_bands, **SENSOR_ARGS[sensor])


        'Convert output to log scale for analysis'
        model_preds = np.log10(model_preds + 1e-6)

        'Again fix the projections - if needed'
        if sensor_option == "OLCI":
            model_preds, _, (_, _) = fix_projection(model_preds, lon, lat, False)
            img_uncert, _, (_, _) = fix_projection(img_uncert, lon, lat, False)

        data_products = [r"Chl$a$ [$mg~m^{-3}$]", r"TSS [$g~m^{-3}$]", r"a$_{cdom}$ [$\frac{1}{m}$]"]
        prdct_name = ['chla', 'tss', 'cdom']
        assert len(data_products) == model_preds.shape[2], f"The number of data products given is {len(data_products)}," \
                                                           f"but model predicts {model_preds.shape[2]} shape"

        for ii in range(len(data_products)):
            'Create the figures for the specific product'
            fig1  = create_satellite_uncertainty_plots(img_rgb, np.squeeze(model_preds[:,:, ii]),
                                                       np.squeeze(img_uncert[:,:, ii]), data_products[ii], extent,
                                                       0.0, 1.0)

            fig1.savefig(f'./figures/uncert_{location}_{sensor}_{prdct_name[ii]}_glr_MDN.png', bbox_inches='tight')

        plt.show()
        plt.close('all')