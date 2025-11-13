#!/usr/bin/env python3

from unittest import result
import numpy as np
import argparse
import ROOT
import plot_specific_region as psr
import get_guess_pars as ggp
import pandas as pd

# to fit peaks using ROOT
def fitPeaks(data, noOfPeaks, guess, canvas, histos):

    # assign data to variables
    files = np.array(data['files'])
    hs = np.array(data['Hs'])
    xlows = np.array([int(xl) for xl in data['xlows']])
    xhighs = np.array([int(xl) for xl in data['xhighs']])

    fitRes = {} # to hold results of fit

    # define the fit function name based on how many peaks
    # are expected within the region; note that the fit 
    # function is n*Gaussian + linear background, where n is
    # the number of peaks
    fName = "pol1(0)"
    for i in range(noOfPeaks):
        fName += f"+gaus({i*3 +2})"
        fitRes[f"peak{i+1}"] = {}

    # get the pad from canvas
    bigPad = canvas.GetListOfPrimitives().FindObject("pads")
    
    # go through and fit all histograms
    for i in range(len(files)):

        # skip current iteration if guess is empty
        if len(guess['amplitudes'][i])==0 or len(guess['means'][i])==0 or len(guess['sigmas'][i])==0:
            print(f"Check the guesses for {hs[i]:.4f}H")
            continue

        # get rid of the stats box
        ROOT.gStyle.SetOptStat(0)
        ROOT.gStyle.SetOptFit(0)
        
        # define fit function
        fitf = ROOT.TF1("fitf", fName, xlows[i], xhighs[i])

        # pass all the guesses into a parameter array
        # to use for the fit function
        pars = np.empty(2 + noOfPeaks*3)
        pars[0] = 0.0
        pars[1] = 0.0
        for j in range(noOfPeaks):
            pars[3*j + 2] = guess['amplitudes'][i]
            pars[3*j + 3] = guess['means'][i]
            pars[3*j + 4] = guess['sigmas'][i]
        #print(pars)
        
        # set the guesses
        fitf.SetParameters(pars)

        # retrieve histogram from each sub-pad
        bigPad.cd(i+1)
        currentPad = bigPad.GetPad(i+1)
        currentHist = currentPad.GetListOfPrimitives().FindObject(f"{hs[i]:.4f}H")
        currentHist.Fit(fitf,"BRQ")
        canvas.Update()

        # calculate reduced Chi2 from fit
        chi2 = fitf.GetChisquare()
        ndf = fitf.GetNDF()
        rchi2 = chi2/ndf

        # save fit results to a dictionary for returning
        resTmp = {}
        for k in range(noOfPeaks):
            resTmp['amps']   = float(f"{fitf.GetParameter(3*k + 2):.4f}")
            resTmp['means']  = float(f"{fitf.GetParameter(3*k + 3):.4f}")
            resTmp['sigmas'] = float(f"{fitf.GetParameter(3*k + 4):.4f}")
            fitRes[f'peak{k+1}'][float(f'{hs[i]:.4f}')] = resTmp.copy()
            fitRes[f'peak{k+1}'][float(f'{hs[i]:.4f}')]['Chi2'] = float(f"{rchi2:.4f}")

    canvas.Draw()

    return fitRes
   
#####################################################
###                                               ###
###                 MAIN FUNCTION                 ###
###                                               ###
#####################################################
# to control what the whole program does
def main(file, peakNum, alpha, path_to_xtilt, nbinsx):
    
    # plot the region and get the files that exist
    #canvas, pads, hists, info = psr.main(file, alpha, path_to_xtilt, nbinsx)
    canvas, hists, info = psr.main(file, alpha, path_to_xtilt, nbinsx)

    # get initial guesses
    # note that while the means and sigmas are likely to
    # be good guesses, the amplitudes (or peak heights) may
    # not; this is due to the getGuessPars() function using a
    # different value for bins
    fitGuesses = ggp.getGuessPars(info, isFile=False)

    # fit using ROOT
    results = fitPeaks(info, peakNum, fitGuesses, canvas, hists)

    # write the results into a file
    for p in range(peakNum):

        results_df = pd.DataFrame(results[f"peak{p+1}"])
        results_df_Transposed = pd.DataFrame(results_df.T)
        results_df_Transposed.to_csv(f"peak{p+1}_fit_results.csv")

    # this line is needed for the canvas to stay on screen
    # alternatively, run this script with "python3 -i ..."
    ROOT.gApplication.Run()

# everything starts here
if __name__ == '__main__':

    # handle the command line arguments
    parser = argparse.ArgumentParser(description="Fit peak or peaks within a specific region")
    parser.add_argument("file", metavar="windowFile", type=str, help="input file from createWindows.py")
    parser.add_argument("peakNum", metavar="noOfPeaks", type=int, help="number of peaks in this region")
    parser.add_argument("-a", "--alpha", type=float, default=0.0, help="tilt angle alpha; the\
                                                                    default value is taken to\
                                                                    be 0.0")
    parser.add_argument("-p", "--path_to_xtilt", type=str, default="./", 
                                                 help="path to directory where xtilt files exist;\
                                                    the default is taken to be the current dir,\
                                                    where the program is being run")
    parser.add_argument("-bin", "--nbinsx", type=int, default=-10, help="number of bins; if no value\
                                                                    is given, bins = xhigh - xlow is used")

    # pass all of the arguments into args
    args = parser.parse_args()
    
    # pass the args into main function
    main(**vars(args))
