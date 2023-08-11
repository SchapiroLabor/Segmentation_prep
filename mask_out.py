from __future__ import print_function, division
from multiprocessing.spawn import import_main_path
import sys
import copy
import argparse
import numpy as np
import tifffile
import zarr
import skimage.transform
from aicsimageio import aics_image as AI
import pandas as pd
import numexpr as ne
from ome_types import from_tiff, to_xml
from os.path import abspath
from argparse import ArgumentParser as AP
import time
import dask
import dask.array as da
# This API is apparently changing in skimage 1.0 but it's not clear to
# me what the replacement will be, if any. We'll explicitly import
# this so it will break loudly if someone tries this with skimage 1.0.
try:
    from skimage.util.dtype import _convert as dtype_convert
except ImportError:
    from skimage.util.dtype import convert as dtype_convert


# arg parser
def get_args():
    # Script description
    description="""Subtracts background - Lunaphore platform"""

    # Add parser
    parser = AP(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Sections
    inputs = parser.add_argument_group(title="Required Input", description="Path to required input file")
    inputs.add_argument("-r", "--input", dest="root", action="store", required=True, help="File path to image file to be masked out.")
    inputs.add_argument("--pixel-size", metavar="SIZE", dest = "pixel_size", type=float, default = None, action = "store",help="pixel size in microns; default is 1.0")
    inputs.add_argument("--pyramid", dest="pyramid", required=False, default=True, help="Should output be pyramidal?")
    inputs.add_argument("--tile-size", dest="tile_size", required=False, default=1024, help="Tile size for pyramid generation")
    inputs.add_argument("--version", action="version", version="v0.5.0")
    inputs.add_argument("--mask", dest="mask", action="store", required=True, help="File path to binary mask image file specifying regions to be excluded (1 to keep, 0 to exclude).")
    
    outputs = parser.add_argument_group(title="Output", description="Path to output file")
    outputs.add_argument("-o", "--output", dest="output", action="store", required=True, help="Path to output file")

    arg = parser.parse_args()

    # Standardize paths
    arg.root = abspath(arg.root)
    arg.mask = abspath(arg.mask)
    arg.output = abspath(arg.output)

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


# NaN values return True for the statement below in this version of Python. Did not use math.isnan() since the values
# are strings if present
def isNaN(x):
    return x != x
 
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

# Define a function to apply the binary mask to a chunk of the image
def apply_mask(chunk, mask_chunk):
    return chunk * mask_chunk

def main(args):
    img_raw = AI.AICSImage(args.root)
    mask_raw = AI.AICSImage(args.mask)

    img = img_raw.get_image_dask_data("CYX")
    mask = mask_raw.get_image_dask_data("CYX")
    img = img.rechunk({0: img.shape[0], 1: args.tile_size, 2: args.tile_size})
    mask = mask.rechunk({0: mask.shape[0], 1: args.tile_size, 2: args.tile_size})

    # Stack the masked chunks back into a dask array
    dask_masked_image = da.map_blocks(apply_mask, img, mask, dtype=img.dtype)

    # Processing metadata - highly adapted to Lunaphore outputs
    # check if metadata is present
    try:
        print(img_raw.metadata.images[0])
        metadata = img_raw.metadata
    except:
        metadata = None   

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
    print(args.pyramid)
    print(int(args.tile_size)<=max(dask_masked_image[0].shape))

    if (args.pyramid == True) and (int(args.tile_size)<=max(dask_masked_image[0].shape)):
            
        # construct levels
        tile_size = int(args.tile_size)
        scale = 2

        print()
        dtype = dask_masked_image.dtype
        base_shape = dask_masked_image[0].shape
        num_channels = dask_masked_image.shape[0]
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
                data = dask_masked_image,
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
                data = dask_masked_image.compute(),
                shape = dask_masked_image.shape,
                dtype=dask_masked_image.dtype,
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
