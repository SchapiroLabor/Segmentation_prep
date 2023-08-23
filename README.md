# Segmentation_prep

Image preprocessing for different segmentation approaches

## Docker container

The Docker container for this repository is currently at kbestak/seg_prep:0.1.4
```
docker pull kbestak/seg_prep:0.1.6
```

## List of scripts and functionalities

### background_sub.py --> Backsub (v0.3.4)

Refers to [https://github.com/SchapiroLabor/Background_subtraction](https://github.com/SchapiroLabor/Background_subtraction) which also works within this container.


### clahe_segmentation_prep.py --> implementation of Skimage CLAHE

[https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.equalize_adapthist](https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.equalize_adapthist)

Reshapes the image to only include selected channels. Possibility to apply Contrast-limited adaptive histogram equalization (CLAHE) on selected channels.

#### Arguments:

* `--input` (required), path to input image
* `--output` (required), path to output image
* `--keep-channel` (required), channel indices which should be included in output image
* `--clahe-channel` (required), channel indices on which CLAHE should be applied.
* `--pyramid` (default=`True`), should the output be pyramidal?
* `--tile-size` (default=`1024`), tile size for pyramid generation)
* `--pixel-size` (required), physical pixel size for metadata in output
* CLAHE options:
  * `--cliplimit` (required), clip limit for CLAHE
  * `--nbins` (default=`256`), number of bins for CLAHE

#### Example command with Docker:

```
docker run \
  -v /path/to/folder/with/input:/input \
  -v /path/to/folder/with/output:/output \
  kbestak/seg_prep:0.1.6 \
  python3 /seg_prep/clahe_segmentation_prep.py \
  --input "/input/image.ome.tif" \
  --output "/output/image_out.ome.tif" \
  --keep-channel 0 1 2 19 \
  --clahe-channel 0 19 \
  --pixel-size 0.23 \
  --cliplimit 0.01 \
  --nbins 256 \
  --pyramid True \
  --tile-size 1024
```

#### Example command with Singularity on Helix:

```
#!/bin/sh
#SBATCH --job-name="demo"
#SBATCH --partition=single
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mem=16gb

module load devel/java_jdk/1.18
module load system/singularity/3.9.2

singularity run \
  docker://kbestak/seg_prep:0.1.6 \
  python3 /seg_prep/clahe_segmentation_prep.py \
  --input "/path/to/input/image.ome.tif" \
  --output "/path/to/output/image_out.ome.tif" \
  --keep-channel 0 1 2 19 \
  --clahe-channel 0 19 \
  --pixel-size 0.23 \
  --cliplimit 0.01 \
  --nbins 256 \
  --pyramid True \
  --tile-size 1024
```


### crop_image_by_roi.py

Region of interest extraction based on input `csv` file. 
Not intended for large ROIs which would require pyramidal output (this functionality could be implemented if required).

#### Arguments:

* `--input` (required), path to input image file
* `--output` (required), path to output folder
* `--pixel-size` (default=`1.0`), physical pixel size in microns
* `--roi` (required), path to the `csv` file containing ROI information

#### ROI `csv` file columns:

Region of interest boundaries (only rectangles currently possible) are given as y(min), y(max), x(min), x(max) values for the ROI.
* `y_min` specifies the minimum y coordinate of the ROI
* `x_min` specifies the minimum x coordinate of the ROI
* `y_max` specifies the maximum y coordinate of the ROI
* `x_max` specifies the maximum x coordinate of the ROI
* `roi_name` specifies the name with which the ROI gets saved (should *not* have the suffix `.tif`) 

#### Example command with Docker:

```
docker run \
  -v /path/to/folder/with/input:/input \
  -v /path/to/folder/with/roi:/roi \
  -v /path/to/output/folder:/output \
  kbestak/seg_prep:0.1.6 \
  python3 /seg_prep/crop_image_by_roi.py \
  --input "/input/image.ome.tif" \
  --output "/output" \
  --roi "/roi/roi.csv" \
  --pixel-size 0.23
```

#### Example command with Singularity on Helix:

```
#!/bin/sh
#SBATCH --job-name="demo"
#SBATCH --partition=single
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mem=16gb

module load devel/java_jdk/1.18
module load system/singularity/3.9.2

singularity run \
  docker://kbestak/seg_prep:0.1.6 \
  python3 /seg_prep/crop_image_by_roi.py \
  --input "/path/to/input/image.ome.tif" \
  --output "/path/to/output/folder" \
  --roi "/path/to/roi.csv" \
  --pixel-size 0.23 
```

### unstack_image.py

Saves specified channels as separate `tif`˙files. Channel names are saved based on the index.
Channel naming based on channel name metadata will be implemented and toggleable.

#### Arguments:

* `--input` (required), path to input image file
* `--output` (required), path to output folder
* `--channels` (required), channel indices that should be saved as separate images
* `--pixel-size` (required), physical pixel size for metadata in output

#### Example command with Docker:

```
docker run \
  -v /path/to/folder/with/input:/input \
  -v /path/to/output/folder:/output \
  kbestak/seg_prep:0.1.6 \
  python3 /seg_prep/unstack_image.py \
  --input "/input/image.ome.tif" \
  --output "/output" \
  --channels 0 1 2 19 \
  --pixel-size 0.23
```

#### Example command with Singularity on Helix:

```
#!/bin/sh
#SBATCH --job-name="demo"
#SBATCH --partition=single
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mem=16gb

module load devel/java_jdk/1.18
module load system/singularity/3.9.2

singularity run \
  docker://kbestak/seg_prep:0.1.6 \
  python3 /seg_prep/unstack_image.py \
  --input "/path/to/input/image.ome.tif" \
  --output "/path/to/output/folder" \
  --channels 0 1 2 19 \
  --pixel-size 0.23
```


### stack_channels.py

Takes a list of separate images (single-, or multi-channel `tif` files) to stack to one `ome.tif` image. 
A pyramidal writer will be included in the future for larger images. Functionality to stack all images within a folder could be added if needed.

#### Arguments:

* `--input` (required), paths to input images
* `--output` (required), path to output image
* `--num-channels`, total number of channels in output image (saves computing time, could be removed)

#### Example command with Docker:

In this example image1.ome.tif is a 2-channel image, and image2-ome.tif is a single-channel image.
```
docker run \
  -v /path/to/folder/with/inputs:/input \
  -v /path/to/output:/output \
  kbestak/seg_prep:0.1.6 \
  python3 /seg_prep/stack_channels.py \
  --input "/input/image1.ome.tif" "/input/image2.ome.tif" \
  --output "/output/image_out.ome.tif" \
  --num-channels 3
```

#### Example command with Singularity on Helix:

```
#!/bin/sh
#SBATCH --job-name="demo"
#SBATCH --partition=single
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mem=16gb

module load devel/java_jdk/1.18
module load system/singularity/3.9.2

singularity run \
  docker://kbestak/seg_prep:0.1.6 \
  python3 /seg_prep/stack_channels.py \
  --input "/path/to/folder/with/inputs/image1.ome.tif" "/path/to/folder/with/inputs/image2.ome.tif" \
  --output "/output/image_out.ome.tif" \
  --num-channels 3
```


### mask_out.py

Takes a single-, or multi-channel `tif` image and multiplies it with a binary mask to produce a pyramidal `ome tif` image. 

#### Arguments:

* `--input` (required), paths to input image
* `--output` (required), path to output image
* `--mask` (required), path to input binary mask file
* `--pixel-size`, add pixel size to metadata
* `--pyramid` (default `True`), should the output be pyramidal
* `--tile-size` (default `1024`), tile size for pyramidal outputs

#### Example command with Docker:

In this example image1.ome.tif is a 2-channel image, and image2-ome.tif is a single-channel image.
```
docker run \
  -v /path/to/input:/input \
  -v /path/to/mask:/input \
  -v /path/to/output:/output \
  kbestak/seg_prep:0.1.6 \
  python3 /seg_prep/mask_out.py \
  --input "/input/image.ome.tif" \
  --mask "/input/mask.tif" \
  --pixel-size 0.23 \
  --output "/output/output.ome.tif"
```

#### Example command with Singularity on Helix:

```
#!/bin/sh
#SBATCH --job-name="demo"
#SBATCH --partition=single
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mem=16gb

module load devel/java_jdk/1.18
module load system/singularity/3.9.2

singularity run \
  docker://kbestak/seg_prep:0.1.6 \
  python3 /seg_prep/mask_out.py \
  --input "/path/to/folder/with/inputs/image.ome.tif" \
  --mask "/path/to/folder/with/mask/mask.tif" \
  --output "/output/output.ome.tif" \
  --pixel-size 0.23
```


### max_projection.py

Takes a multi-channel `tif` image and applies max projection on nuclear and membrane channels. Outputs a 2-channel image with the nuclear max projections, and membrane max projections with adjustable order.

#### Arguments:

* `--input` (required), paths to input image
* `--output` (required), path to output image
* `--nuclear-channels` (required), channel indices specifying the nuclear channels on which max projection should be applied
* `--membrane-channels` (required), channel indices specifying the membrane channels on which max projection should be applied
* `--first` (default `nuclear`), write `--first membrane` if the membrane channel should be the first channel in the output image
* `--pixel-size`, add pixel size to metadata
* `--pyramid` (default `True`), should the output be pyramidal
* `--tile-size` (default `1024`), tile size for pyramidal outputs

#### Example command with Docker:

In this example image.ome.tif is a 20-channel image
```
docker run \
  -v /path/to/input:/input \
  -v /path/to/output:/output \
  kbestak/seg_prep:0.1.6 \
  python3 /seg_prep/max_projection.py \
  --input "/input/image.ome.tif" \
  --nuclear-channels 0 \
  --membrane-channels 16 17 19 \
  --pixel-size 0.23 \
  --output "/output/output.ome.tif"
```

#### Example command with Singularity on Helix:

```
#!/bin/sh
#SBATCH --job-name="demo"
#SBATCH --partition=single
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mem=16gb

module load devel/java_jdk/1.18
module load system/singularity/3.9.2

singularity run \
  docker://kbestak/seg_prep:0.1.6 \
  python3 /seg_prep/max_projection.py \
  --input "/path/to/folder/with/inputs/image.ome.tif" \
  --output "/output/output.ome.tif" \
  --pixel-size 0.23 \
  --nuclear-channels 0 \
  --membrane-channels 16 17 19
```


