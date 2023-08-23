from __future__ import print_function, division
from distutils.log import error
import time
import argparse
from argparse import ArgumentParser as AP
from os.path import abspath
import os
import numpy as np
from skimage.exposure import equalize_adapthist
import sys
import copy
import argparse
import numpy as np
import tifffile
import zarr
import skimage.transform
from aicsimageio import aics_image as AI
from ome_types import from_tiff, to_xml
from os.path import abspath
from argparse import ArgumentParser as AP
import time
import dask.array as da
# This API is apparently changing in skimage 1.0 but it's not clear to
# me what the replacement will be, if any. We'll explicitly import
# this so it will break loudly if someone tries this with skimage 1.0.
try:
    from skimage.util.dtype import _convert as dtype_convert
except ImportError:
    from skimage.util.dtype import convert as dtype_convert


def get_args():
    # Script description
    description="""Easy-to-use, large scale CLAHE"""

    # Add parser
    parser = AP(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Sections
    inputs = parser.add_argument_group(title="Required Input", description="Path to required input file")
    inputs.add_argument("-r", "--input", dest="raw", action="store", required=True, help="File path to input image.")
    inputs.add_argument("-o", "--output", dest="output", action="store", required=True, help="Path to output image.")
    inputs.add_argument("--nuclear-channels", dest="nuclear", nargs="+", action="store", required=True, help="Channels to be used for max projection for nucleus")
    inputs.add_argument("--membrane-channels", dest="membrane", nargs="+", action="store", required=True, help="Channels to be used for max projection for membrane")
    inputs.add_argument("--pixel-size", metavar="SIZE", dest = "pixel_size", type=float, default = None, action = "store",help="pixel size in microns; default is 1.0")
    inputs.add_argument("--pyramid", dest="pyramid", required=False, default=True, help="Generate pyramid")
    inputs.add_argument("--tile-size", dest="tile_size", action="store", required=False, default=1024, help="Tile size for pyramid generation")
    inputs.add_argument("--first", dest="first", action="store", required=False, default="nuclear", help="Is the first channel 'nuclear' or 'membrane'")


    arg = parser.parse_args()

    # Standardize paths
    arg.raw = abspath(arg.raw)
    arg.output = abspath(arg.output)
    arg.nuclear = [int(channel) for channel in arg.nuclear]
    arg.membrane = [int(channel) for channel in arg.membrane]
    arg.pyramid = bool(arg.pyramid)
    arg.pixel_size = float(arg.pixel_size)

    return arg

def preduce(coords, img_in, img_out):
    print(img_in.dtype)
    (iy1, ix1), (iy2, ix2) = coords
    (oy1, ox1), (oy2, ox2) = np.array(coords) // 2
    tile = skimage.img_as_float32(img_in[iy1:iy2, ix1:ix2])
    tile = skimage.transform.downscale_local_mean(tile, (2, 2))
    tile = dtype_convert(tile, 'uint16')
    #tile = dtype_convert(tile, img_in.dtype)
    img_out[oy1:oy2, ox1:ox2] = tile

def format_shape(shape):
    return "%dx%d" % (shape[1], shape[0])

def subres_tiles(level, level_full_shapes, tile_shapes, outpath, scale):
    print(f"\n processing level {level}")
    assert level >= 1
    num_channels, h, w = level_full_shapes[level]
    tshape = tile_shapes[level] or (h, w)
    tiff = tifffile.TiffFile(outpath)
    zimg = zarr.open(tiff.aszarr(series=0, level=level-1, squeeze=False))
    for c in range(num_channels):
        sys.stdout.write(
            f"\r  processing channel {c + 1}/{num_channels}"
        )
        sys.stdout.flush()
        th = tshape[0] * scale
        tw = tshape[1] * scale
        for y in range(0, zimg.shape[1], th):
            for x in range(0, zimg.shape[2], tw):
                a = zimg[c, y:y+th, x:x+tw, 0]
                a = skimage.transform.downscale_local_mean(
                    a, (scale, scale)
                )
                if np.issubdtype(zimg.dtype, np.integer):
                    a = np.around(a)
                a = a.astype('uint16')
                yield a

def max_projection(img, channels):
    return np.max(img[channels], axis=0)

def check_power_of_two(x):
    return x != 0 and ((x & (x - 1)) == 0)

def main(args):
    print()
    print(f"Input image path = {args.raw}")
    print(f"Output image path = {args.output}")
    print(f"Nuclear channel(s) = {args.nuclear}, membrane channel(s) = {args.membrane}")
    print()

    if check_power_of_two(int(args.tile_size)) == False:
        raise Exception("Tile_size must be a power of 2")
    
    # load image
    img_raw = AI.AICSImage(args.raw)
    img_dask = img_raw.get_image_dask_data("CYX")
    img_dask = img_dask.rechunk({0: img_dask.shape[0], 1: args.tile_size, 2: args.tile_size})

    ### create output_img variable with 2 channels, one for nuclear and one for membrane with zeros
    output_img = np.zeros((2, img_dask.shape[1], img_dask.shape[2]), dtype=img_dask.dtype)

    ### max project nuclear channels
    print()
    output_img[0] = max_projection(img_dask, args.nuclear)

    ### max project membrane channels
    output_img[1] = max_projection(img_dask, args.membrane)

    if args.first == "membrane":
        output_img = np.flip(output_img, axis=0)

    # keep only specified channels
    metadata = None

    # process and adapt pixel size --> if user input, this will be used, otherwise, it will be taken from the metadata. If no metadata, default 1.0
    if args.pixel_size != None:
        # If specified, the input pixel size is used
        pixel_size = args.pixel_size   
    else:
        try:
            if img_raw.metadata.images[0].pixels.physical_size_x != None:
                pixel_size = img_raw.metadata.images[0].pixels.physical_size_x
            else:
                pixel_size = 1.0
        except:
            # If no pixel size specified anywhere, use default 1.0
            pixel_size = 1.0
    
    # if pyramid, construct pyramid
    if (args.pyramid == True) and (int(args.tile_size)<=max(output_img[0].shape)):
            
        # construct levels
        tile_size = int(args.tile_size)
        scale = 2

        print()
        dtype = output_img.dtype
        base_shape = output_img[0].shape
        num_channels = output_img.shape[0]
        num_levels = (np.ceil(np.log2(max(base_shape) / tile_size)) + 1).astype(int)
        factors = 2 ** np.arange(num_levels)
        shapes = (np.ceil(np.array(base_shape) / factors[:,None])).astype(int)
        print(base_shape)
        print(np.ceil(np.log2(max(base_shape)/tile_size))+1)

        print("Pyramid level sizes: ")
        for i, shape in enumerate(shapes):
            print(f"   level {i+1}: {format_shape(shape)}", end="")
            if i == 0:
                print("(original size)", end="")
            print()
        print()
        print(shapes)  

        level_full_shapes = []
        for shape in shapes:
            level_full_shapes.append((num_channels, shape[0], shape[1]))
        level_shapes = shapes
        tip_level = np.argmax(np.all(level_shapes < tile_size, axis=1))
        tile_shapes = [
            (tile_size, tile_size) if i <= tip_level else None
            for i in range(len(level_shapes))
        ]
        
        # write pyramid
        with tifffile.TiffWriter(args.output, ome=True, bigtiff=True) as tiff:
            tiff.write(
                data = output_img,
                shape = level_full_shapes[0],
                subifds=int(num_levels-1),
                dtype=dtype,
                resolution=(10000 / pixel_size, 10000 / pixel_size, "centimeter"),
                tile=tile_shapes[0]
            )
            for level, (shape, tile_shape) in enumerate(
                    zip(level_full_shapes[1:], tile_shapes[1:]), 1
            ):
                tiff.write(
                    data = subres_tiles(level, level_full_shapes, tile_shapes, args.output, scale),
                    shape=shape,
                    subfiletype=1,
                    dtype=dtype,
                    tile=tile_shape
                )
    else:
        # write image
        with tifffile.TiffWriter(args.output, ome=True, bigtiff=True) as tiff:
            tiff.write(
                data = output_img,
                shape = output_img.shape,
                dtype=output_img.dtype,
                resolution=(10000 / pixel_size, 10000 / pixel_size, "centimeter"),
            )
    try:
        tifffile.tiffcomment(args.output, to_xml(metadata))
    except:
        pass
    # note about metadata: the channels, planes etc were adjusted not to include the removed channels, however
    # the channel ids have stayed the same as before removal. E.g if channels 1 and 2 are removed,
    # the channel ids in the metadata will skip indices 1 and 2 (channel_id:0, channel_id:3, channel_id:4 ...)
    print()


if __name__ == '__main__':
    # Read in arguments
    args = get_args()

    # Run script
    st = time.time()
    main(args)
    rt = time.time() - st
    print(f"Script finished in {rt // 60:.0f}m {rt % 60:.0f}s")
