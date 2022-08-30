#!/usr/bin/env python
##                            Research
## Randy Klepetko                                 08/16/2022
#
## CNN MiCam Plotter V3
#
# This code was written for research, and isn't considered production quality.
# 
# Version 3 changes:
# - Included four degrees of parameters for CAM visualization taking advantage of the RGBA pallete.
# - Updated graphics engine to pass pixel parameter to the graphics engine as a grid instead of by pixel incresing performance.
# - Modularized code. 
#
# This code is the lead training module that processes CNN models and exporting a MiCAM plot.
# 
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Python Module Imports
import math, time, os, sys
from PIL import Image

# this routine takes a set of images files and merges them horizontaly consecutively
def merge_horizontal_image(filenames,path):
    height = 0
    width = 0
    for filename in filenames:
        im = Image.open(filename)
        if im.height > height:
            height = im.height
        width += im.width
    imt = Image.new('RGBA',(width,height))
    width = 0
    for filename in filenames:
        im = Image.open(filename)
        imt.paste(im,(width,0))
        width += im.width
    imt.save(path, dpi=(300,300))

# this routine takes a set of images files and merges them vertically consecutively
def merge_vertical_image(filenames,path):
    height = 0
    width = 0
    for filename in filenames:
        im = Image.open(filename)
        if im.width > width:
            width = im.width
        height += im.height
    imt = Image.new('RGBA',(width,height))
    height = 0
    for filename in filenames:
        im = Image.open(filename)
        imt.paste(im,(0,height))
        height += im.height
    imt.save(path, dpi=(300,300))

# this routine takes the saved image and truncates it down to remove excess white space
def truncate_image(source,dest,size):
    im = Image.open(source)
    im = im.crop(size)
    ns = im.size
    im = im.resize([int(im.width/2),int(im.height/2)])
    im.save(dest, dpi=(300,300))

