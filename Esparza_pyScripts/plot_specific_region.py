#!/usr/bin/env python3

import numpy as np
import argparse
from pathlib import Path
import sys
import ROOT
import pandas as pd

# to plot one or multiple histograms of a specific region
def plotThem(noOfFiles, files, Hs, xlows, xhighs, bins):

    # (name, title, top left x coord, top left y coord, window x width, window y width)
    c1 = ROOT.TCanvas('c1', 'Specific Region', 300, 0, 1300, 1000)

    # (name, title, bottomLeftX, bottomLeftY, topRightX, topRightY, color)
    # note that the coordinates are between [0,1]
    pads = ROOT.TPad('pads', 'Total Pad', 0.03, 0.03, 0.97, 0.97, 0)

    # divide the pad into sub-pads based on how many files exist
    pads.Divide(1, noOfFiles, 0.0, 0.0, 0)
    pads.Draw()

    # set number of bins to xhighs - xlows unless a specific value is given
    if bins == -10:
        bins = int(xhighs[0] - xlows[0])

    # fill the pads with histograms
    binMax = -1
    tmpMax = -1
    hMax = -1
    hists = np.empty(noOfFiles, dtype=ROOT.TH1F)
    for i in range(noOfFiles):

        # (name, title, numOfBins, xlow, xup)
        hists[i] = ROOT.TH1F(f'{Hs[i]:.4f}H', f'{Hs[i]:.4f}H', bins, xlows[i], xhighs[i])
        
        # make titles more visible
        ROOT.gStyle.SetTitleSize(0.3,"t")

        # fill histogram with values from files
        for val in files[i]:
            hists[i].Fill(val)
        
        # find max bin for y-axis scaling
        tmpMax = hists[i].GetMaximum()
        if binMax < tmpMax: 
            binMax = tmpMax
            hMax = Hs[i]

    # print out the value at which the peak is highest 
    print(f"Highest peak occured at: {hMax:.4f}H")

    # plot histograms
    for j in range(noOfFiles):

        pads.cd(j+1)
        hists[j].SetMaximum(int(1.1*binMax))
        hists[j].Draw()
    
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptFit(1111)
    c1.Update()

    return c1, hists

#####################################################
###                                               ###
###                 MAIN FUNCTION                 ###
###                                               ###
#####################################################
# to control what the whole program does
def main(file, alpha, path_to_xtilt, nbinsx:int = -10, save_good_info:bool = False):

    # check if file exist
    inFile = Path(file)
    if inFile.is_file() != True:
        sys.exit(f"file does not exits!")

    # unpack inFile into corresponding variables
    hs, xLows, xHighs = np.loadtxt(inFile, unpack=True)

    # check for existence of files based on combo of hs and alpha
    # then load the existing file into a dictionary for later use
    exist = 0 # keep track of how many file actually exist
    goodFiles = [] # keep the contents of files that exist
    goodFileNames = [] # keep the names of files that exist
    nonExistIndex = [] # keep the indices of bad files (non-existence)
    for i in range(len(hs)):
        fileName = Path(f"{path_to_xtilt}/{hs[i]:.4f}H_{alpha:.1f}_degrees_xavg_tilt.txt")
        if fileName.is_file():
            goodFiles.append(np.loadtxt(fileName))
            goodFileNames.append(fileName)
            exist += 1
        else:
            nonExistIndex.append(i)
            print(f"{fileName} does not exist!")
    
    # if none of the file exists, exit the program
    if exist == 0:
        print("None of the file exist! Check directory and H-alpha combo!")
        sys.exit()
    
    # remove info of bad files
    good_hs = np.delete(hs, nonExistIndex)
    good_xLows = np.delete(xLows, nonExistIndex)
    good_xHighs = np.delete(xHighs, nonExistIndex)

    # plot things
    cv, hist = plotThem(exist, goodFiles, good_hs, good_xLows, good_xHighs, nbinsx)

    # put info of good (i.e. existing) H into a dictionary
    good_info = {}
    good_info['files'] = goodFiles
    good_info['Hs'] = good_hs
    good_info['xlows'] = good_xLows
    good_info['xhighs'] = good_xHighs

    # save good_info into a file
    if save_good_info:
        good_info_df = pd.DataFrame(good_info)
        good_info_df.to_pickle("good_info_of_specific_region.pkl")

    return cv, hist, good_info
    
# everything starts here
if __name__ == '__main__':

    # handle the command line arguments
    parser = argparse.ArgumentParser(description="Plot a specific region of the focal plane spectrum\
                                        for a range of values of H and alpha. These values are\
                                        given in a file produced by the createWindows.py program")                                       
    parser.add_argument("file", metavar="windowFile", type=str, help="input file from createWindows.py")
    parser.add_argument("-a", "--alpha", type=float, default=0.0, help="tilt angle alpha")
    parser.add_argument("-p", "--path_to_xtilt", type=str, default="./", 
                                                 help="path to directory where xtilt files exist")
    parser.add_argument("-bin", "--nbinsx", type=int, default=-10, help="number of bins; if no value\
                                                                    is given, bins = xhigh - xlow is used")
    parser.add_argument("-s", "--save_good_info", action="store_true", help="save info of existing H\
                                                                            into a txt file")
        
    # pass all of the arguments into args
    args = parser.parse_args()
    
    # pass the args into main function
    results = main(**vars(args)) # need to assign the results to a variable for plots to show up
                                 # when this program is run as a standalone

    # this line is needed for the canvas to stay on screen
    # alternatively, run this script with "python3 -i ..."
    ROOT.gApplication.Run()
