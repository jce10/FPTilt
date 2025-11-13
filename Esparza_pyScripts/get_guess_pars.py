#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, peak_widths
import math
import pandas as pd

# convert FWHM to standard deviation
# where FWHM = sigma * 2*sqrt(2*ln(2))
def convert_FWHM_to_sigma(widths):
    factor = 1 / (2 * math.sqrt(2 * np.log(2)))
    return widths*factor

# to get initial guesses for later fitting
def getGuessPars(inFile, isFile:bool = True):

    # assign data to variables
    if isFile == True:
        data = pd.read_pickle(inFile)
    else:
        data = inFile
    files = np.array(data['files'])
    hs = np.array(data['Hs'])
    xlows = np.array([int(xl) for xl in data['xlows']])
    xhighs = np.array([int(xl) for xl in data['xhighs']])

    # make some arrays to hold info of peaks
    noOfFiles = len(files)
    amps = np.empty(noOfFiles, dtype=np.ndarray)
    means = np.empty(noOfFiles, dtype=np.ndarray)
    sigmas = np.empty(noOfFiles, dtype=np.ndarray)

    # get the guesses
    fig, axs = plt.subplots(noOfFiles, gridspec_kw = {'wspace':0, 'hspace':0})
    for i in range(noOfFiles):
        
        xvals_hist, bin_edges, _ = axs[i].hist(files[i], int(xhighs[0] - xlows[0]), (xlows[i],xhighs[i]),
                                            histtype='step',log=False,lw=0.5,stacked=True)
        peaks, properties = find_peaks(xvals_hist, prominence=100, height=100, width=2.355)
        amps[i] = properties['peak_heights']
        means[i] = bin_edges[peaks]
        widths,width_heights,left_ips,right_ips = peak_widths(xvals_hist,peaks,rel_height=0.5)
        sigmas[i] = convert_FWHM_to_sigma(widths)
        axs[i].plot(bin_edges[peaks],xvals_hist[peaks],'vr')

        # turn off tick labels
        axs[i].set_yticklabels([])
        axs[i].set_xticklabels([])

        # turn off tick marks
        axs[i].set_xticks([])
        axs[i].set_yticks([])

        # set y-axis title
        axs[i].set_ylabel("{:.2f}".format(hs[i]))
    
    # save means and sigmas then return
    pars = {}
    pars['amplitudes'] = amps
    pars['means'] = means
    pars['sigmas'] = sigmas

    return pars

#####################################################
###                                               ###
###                 MAIN FUNCTION                 ###
###                                               ###
#####################################################
# to control what the whole program does
def main(file):
    
    getGuessPars(file)

# everything starts here
if __name__ == '__main__':

    # handle the command line arguments
    parser = argparse.ArgumentParser(description="To get peak guesses for later fitting of peaks\
                                        within a specific region")
    parser.add_argument("file", metavar="good_info_file", type=str, help="file produced by plot_specific_region.py;\
                                                                        this file should have the pkl format")

   # pass all of the arguments into args
    args = parser.parse_args()
    
    # pass the args into main function
    main(**vars(args))

    plt.show()