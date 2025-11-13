#!/usr/bin/env python3

import numpy as np
import argparse

#####################################################
###                                               ###
###                 MAIN FUNCTION                 ###
###                                               ###
#####################################################
# to control what the whole program does
def main(h1, x1, h2, x2, w, s, bounds):

    # make an array of H-values between h1 and h2
    # note that np.arange() is exclusive on the right boundary
    # so, an increment should be added to the right boundary to include it
    if bounds == [-10.0, -10.0]:
        hs = np.arange(h1, h2+s, s)
    else:
        hs = np.arange(bounds[0], bounds[1]+s, s)

    # arrays to hold left and right ends of the window
    starts = np.empty(len(hs))
    ends = np.empty(len(hs))

    # calculate the boundaries of the peak window for each val of Hs    
    slope = (x2 - x1) / (h2 - h1)
    for i in range(len(hs)):
        starts[i] = int(slope*(hs[i]-h1) + x1)
        #print(f'{hs[i]:.3f}', starts[i], sep='\t')
    ends = starts + w

    # write the boundaries to a file
    output = np.column_stack((hs, starts, ends))
    np.savetxt(f'boundariesFile_{hs[0]:.2f}_to_{hs[-1]:.2f}_w_{s:.2f}spacing.txt', output, fmt='%.4f %d %d')

# everything starts here
if __name__ == '__main__':

    # handle the command line arguments
    parser = argparse.ArgumentParser(description="Calculate the low and high boundaries of a\
                                       window in which a specific peak will appear for different\
                                       values of H. The program requires two values of H, two\
                                       values for the lower boundaries of each H, the size of the\
                                       window, and the spacing between H")                                       
    parser.add_argument("h1", type=float, help="value of the smaller H")
    parser.add_argument("x1", type=float, help="value of the lower boundary for h1")
    parser.add_argument("h2", type=float, help="value of the higher H")
    parser.add_argument("x2", type=float, help="value of the lower boundary for h2")
    parser.add_argument("w", metavar="windowSize", type=int, help="size of the window containing the peak")
    parser.add_argument("s", metavar="HSpacing", type=float, help="spacing between each H to calculate for")
    parser.add_argument("-bs", "--bounds", nargs=2, type=float, default=[-10.0,-10.0],
                                           help="assume h1 and h2 to be the boundaries of H unless\
                                                 this option is provided")
    
    # pass all of the arguments into args
    args = parser.parse_args()
    
    # pass the args into main function
    main(**vars(args))
