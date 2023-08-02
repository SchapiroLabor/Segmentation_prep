#!/usr/bin/env python

### This script takes a list of images and stacks them into a single image stack using Dask

import numpy as np
import argparse
import tifffile as tf
from aicsimageio.writers import OmeTiffWriter
from aicsimageio import aics_image as AI
from os.path import abspath
from argparse import ArgumentParser as AP
import aicsimageio
import time

def get_args():
    # Script description
    description="""Writes selected channels as single images"""

    # Add parser
    parser = AP(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i","--input", dest="input", type=str, required=True, help="Image to unstack")
    parser.add_argument("-o","--output", dest="output", type=str, required=True, help="Output directory")
    parser.add_argument("--channels", dest="channels", nargs="+", type=int, required=True, help="Channels to unstack")
    parser.add_argument("--pixel-size", dest="pixel_size", type=float, required=True, help="Pixel size in microns")
    arg = parser.parse_args()
    arg.input = abspath(arg.input)
    arg.output = abspath(arg.output)

    return arg

def main(args):

    img = AI.AICSImage(args.input).get_image_dask_data("CYX")

    for i in args.channels:
        img_out = img[i]
        out_path = f'{args.output}/channel_{i}.ome.tif'
        tf.imwrite(out_path, img_out, imagej=False, resolution=(10000 / args.pixel_size, 10000 / args.pixel_size, "centimeter"))
        #OmeTiffWriter.save(img_out, f'{args.output}/channel_{i}.ome.tiff', dim_order = "YX")
#        with tf.TiffWriter(args.output.join(f'channel_{i}.ome.tiff'), ome=True, bigtiff=True) as tiff:
#            tiff.write(
##                data = img_out,
#               shape = img_out.shape,
#                dtype=img_out.dtype,
#                resolution=(10000 / args.pixel_size, 10000 / args.pixel_size, "centimeter")
#            )


if __name__ == '__main__':
    # Read in arguments
    args = get_args()

    # Run script
    st = time.time()
    main(args)
    rt = time.time() - st
    print(f"Script finished in {rt // 60:.0f}m {rt % 60:.0f}s")