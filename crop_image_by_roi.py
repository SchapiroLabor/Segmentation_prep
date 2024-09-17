from os.path import abspath
from argparse import ArgumentParser as AP
import argparse
import numpy as np
import tifffile
import zarr
import dask.array
import pandas as pd
import time


# arg parser
def get_args():
    # Script description
    description="""Subtracts background - Lunaphore platform"""

    # Add parser
    parser = AP(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Sections
    inputs = parser.add_argument_group(title="Required Input", description="Path to required input file")
    inputs.add_argument("-i", "--input", dest="input", action="store", required=True, help="File path to input image file.")
    inputs.add_argument("-r", "--roi", dest="roi", action="store", required=True, help="File path to required roi.csv file")
    inputs.add_argument("--pixel-size", metavar="SIZE", dest = "pixel_size", type=float, default = None, action = "store",help="pixel size in microns; default is 1.0")
    
    outputs = parser.add_argument_group(title="Output", description="Path to output folder")
    outputs.add_argument("-o", "--output", dest="output", action="store", required=True, help="Path to output folder")

    arg = parser.parse_args()

    # Standardize paths
    arg.input = abspath(arg.input)
    arg.roi = abspath(arg.roi)
    arg.output = abspath(arg.output)

    return arg

def write_to_tif(image, output, pixel_size):
    with tifffile.TiffWriter(output, ome=True, bigtiff=False) as tiff:
        tiff.write(
            data = image,
            shape = image.shape,
            dtype = image.dtype,
            resolution = (10000 / pixel_size, 10000 / pixel_size, "centimeter")
        )

def main(args):
    store = tifffile.imread(input, aszarr=True)
    cache = zarr.LRUStoreCache(store, max_size=2**30)
    zobj = zarr.open(cache, mode='r')
    data = dask.array.from_zarr(zobj[0])

    # roi dataframe needs to have columns specifying y_min, y_max, x_min, x_max, and roi_name
    roi = pd.read_csv(args.roi)

    for index in range(roi.shape[0]):
        row = roi.iloc[index,:]
        crop = data[:, int(row.y_min):int(row.y_max), int(row.x_min): int(row.x_max)]
        print(f"Cropped roi {row.roi_name}, shape: {crop.shape}, YX coordinates: {row.y_min}:{row.y_max}, {row.x_min}:{row.x_max}")
        write_to_tif(crop, output=f"{args.output}/crop_{row.roi_name}.tif", pixel_size=args.pixel_size)       
    
if __name__ == '__main__':
    # Read in arguments
    args = get_args()

    # Run script
    st = time.time()
    main(args)
    rt = time.time() - st
    print(f"Script finished in {rt // 60:.0f}m {rt % 60:.0f}s")
