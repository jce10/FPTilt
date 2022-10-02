# User Guide for the Focal Plane Detector Tilt Correction Scripts

This collection of python scripts is aimed to find the tilt angle (relative to the wires), $\alpha$, of the focal plane detector as well as the distance between the focal plane and any predetermined origin, according to the method described in https://doi.org/10.1016/0029-554X(75)90121-4. The overall scheme of the scripts is to find the optimal H values (while the tilt angle is set parallel to the wires, i.e. $\alpha$ = 0.0) of several peaks of the position spectrum such that the FWHM of each peak is smallest at its corresponding optimal H value. The rest of this document is meant to describe the functionality of each script and the overall scheme of how the scripts are meant to be used together.

In general, the order to use the scripts in are as follows:

1. Get x1-x2 coordinates (positions of incident particles when they cross the front and back wire, respectively) from the SPS-SABRE eventbuilder. This can be done by either printing out the coordinates, then grabbing them from the terminal, or adding in a way to save these values into a text file. Note that the coordinates should be extracted AFTER cuts are made. For now, let's call this text file "x1x2\_withcuts.txt".

2. Use the calculate\_xtilt.py script together x1x2\_withcuts.txt as its argument to produce the focal plane positions (i.e. the points where the incident particles cross the focal plane).

3. Use the createWindows.py together with the output files of calculate\_xtilt.py to calculate the boundaries of a specific region of the xavg spectrum.

If running these scripts for the first time, do the following two steps:

+ Use the plot\_specific\_region.py script together with the output file of createWindows.py to confirm that the boundaries are correct.

+ Use the get\_guess\_pars.py script together with the output file of plot\_specific\_region.py to confirm that peaks in specific regions are always recognized. Most likely, one would need to adjust parameters in the script in order for the peaks to be reliably recognized every time.

If everything is optimized in plot\_specific\_region.py and get\_guess\_pars.py, skip the above two steps and go straight to step 4:

4. Use the fit\_peaks.py script for the final results.

Here is the link to a video that demonstrates the usage of these scripts: https://youtu.be/nnAzjDd-1k8.
