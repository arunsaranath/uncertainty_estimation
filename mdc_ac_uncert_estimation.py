# -*- coding: utf-8 -*-
"""
File Name:      mdn_ac_uncert_estimation
Description:    This code file will be used to evaluate the performance of the MDN-AC models in terms of atmospheric
                corrections.

Date Created:   February 2nd, 2023
"""

import numpy as np
from matplotlib import pyplot as plt

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
from MDN import get_estimates, get_args
from uncertainty_package_final.uncert_support_lib import get_sample_uncertainity
from utilities import get_mdn_preds_full, get_mdn_uncert_ensemble, arg_median

if __name__ == "__main__":
    'Get the test data'
    folder = 'all_sites'
    ac_name = 'l2gen'
    data = *_, target, valid, cols = load_data(f'{folder}/{ac_name}')

    'Get the Full MDN predictions'
    outputs = get_mdn_preds_full(test_x=data[2], test_y=data[3])

    'Now calculate the uncertainties'
    """
    The outputs variable of the MDN package has the structure:
        0) ScalerX
        1) ScalerY
        2) estimates
        3) Distribution
    """
    'Extract the estimates and uncertainties of the ensemble'
    estimates = np.asarray(outputs['estimates'])
    ensmeble_uncertainties = get_mdn_uncert_ensemble(ensmeble_distribution=outputs['coefs'], scaler_y_list=outputs['scalery'])

    'Get the location of the prediction closest to the median'
    est_med_loc = arg_median(estimates, axis=0)


    'Iterate over the data and ectract the values'
    final_uncertainties = []
    final_estimates = []
    for ii in range(estimates.shape[1]):
        samp_uncert = []
        samp_est = []
        for jj in range(estimates.shape[2]):
            samp_est += [estimates[est_med_loc[ii, jj], ii, jj]]
            samp_uncert += [ensmeble_uncertainties[est_med_loc[ii, jj], ii, jj]]

        final_estimates += [np.asarray(samp_est)]
        final_uncertainties += [np.asarray(samp_uncert)]


    final_uncertainties, final_estimates = np.asarray(final_uncertainties), np.asarray(final_estimates)
    np.savetxt('new_estimated_rhorc_Rrs_uncert.csv', final_uncertainties, delimiter=',')

    '------------------------------------------------------------------------------------------------------------------'
    'Calculate and plot the band-level statistics'

    fig1 = plt.figure(figsize=(10, 10))
    plt.errorbar(OLI_WAVELENGTHS, np.mean(final_uncertainties, 0), np.std(final_uncertainties, 0),linestyle='None',
                 fmt='o', ecolor='r')
    plt.xlabel('Wavelength [nm]', fontsize=MEDIUM_SIZE, labelpad=10)
    plt.ylabel(r'Estimated Average Uncertainty [$sr^{-1}$]', fontsize=MEDIUM_SIZE, labelpad=10)
    plt.title(f'Band Level Uncertainty statistics', fontsize=BIGGER_SIZE, weight="bold")
    plt.tight_layout()
    fig1.savefig('oli_bandLevel_uncert_stats.png', bbox_inches='tight')

    '------------------------------------------------------------------------------------------------------------------'
    'Plot the sample-level uncertainty'
    total_error = np.abs((np.median(estimates, 0) - data[3]))

    fig2, axs = plt.subplots(3, 2, figsize=(20, 20))
    for ctr in range(data[3].shape[1]):
        r,c = ctr//2, ctr%2
        ax1 = axs[r,c]
        color = 'tab:blue'
        ax1.set_xlabel('Sample ID', fontsize=MEDIUM_SIZE, labelpad=10)
        ax1.set_ylabel('Total sample uncertainty', fontsize=MEDIUM_SIZE, labelpad=10)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.plot(np.arange(final_uncertainties.shape[0]), final_uncertainties[:, ctr], linewidth='2.0'
                 , color=color, label='Total Uncertainty')

        """ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        ax2.set_xlabel('Sample ID', fontsize=MEDIUM_SIZE, labelpad=10)
        ax2.set_ylabel('MAE', fontsize=MEDIUM_SIZE, labelpad=10)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.plot(np.arange(final_uncertainties.shape[0]), total_error[:, ctr], linewidth='2.0',
                 color=color, label='RMSE')"""

        ax1.set_title(f'Model Performance: Band {OLI_WAVELENGTHS[ctr]}', fontsize=BIGGER_SIZE, weight="bold")
        fig2.savefig('oli_sampLevel_uncert_total.png', bbox_inches='tight')

    plt.show()
