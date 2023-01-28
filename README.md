# MiCAM Generator for Convolutional Neural Networks (CNN)

This location holds the code and plot example for the CNN MiCAM visualization tool.
There where two folders, one that contains the plots and code for version-1, and the remaining is for version-2.

Version-1 was a proof of concept and protoype code, and is included here since it was submitted to academic publications.

Version-2 is an improvment over version-1 for the following reasons:
- First, it uses 4 degrees of information using the RGBA color pallete to provide a deeper understanding for the variations in the positive and negative extremes between layers' plots.  Version-1 only uses gray scale, a single degree of information.
- Second, it contains increase proficiency in the code, steamlining the pixel generation by interacting with the graphics engine only after all of the pixles for a layers has been generated.  This is opposed to generating a single pixel at a time in version-1 which was inefficient.
- Third, it includes the biases and weights in the CAM generation and guaranteed that all of the activations were included in the final CAM plot.  It also establishes concrete reference point within the deconvolution step for combined assimilation.  Version-1 uses a uniform scaling instead of weights and biases.
- Fourth, modularize the code, extracting the MiCAM generator and task specific routines into their own modules.

