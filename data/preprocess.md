# Data Preparing

## Datasets for clean and corrupted object recognition.

The processed CO3D10, CO3D10-C, IN10, and IN10-C datasets can be downloaded [here](https://drive.google.com/drive/folders/1PCfdYbyn_-aVJIDqIsXyDhnFZloP4dfm?usp=drive_link).

For the remaining datasets, please preprocess them using the following pipeline.

### 1. Download the datasets.

Download [CO3D](https://github.com/facebookresearch/co3d), [SAYCam](https://nyu.databrary.org/volume/564), [ImageNet](https://www.image-net.org/), and [ImageNet-C](https://github.com/hendrycks/robustness/tree/master/ImageNet-C) from their respective official websites or access portals.

For SAYCam, after downloading, decode the videos using `decode_SAY.py`.
Specify your *data directory* and run:

```bash
python3 decode_SAY.py
```

### 2. Download Meta files.

Download meta files for CO3D-50 [here](https://drive.google.com/drive/folders/1PCfdYbyn_-aVJIDqIsXyDhnFZloP4dfm?usp=drive_link).


### 3. Transform to beton format.

As this repo uses [ffcv-ssl](https://github.com/facebookresearch/FFCV-SSL) to speed up training, all the data needs to be transformed into beton format.

**Note that `.beton` files usually take up significantly more disk space than the original data.**


Specify the *data directories*, *metadata paths*, and *output paths*, then run:

```bash
python3 generate_beton.py
```

After running the script, the following beton files will be generated:
- CO3D_pretrain.beton
- CO3D_linprobe_train.beton
- CO3D_linprobe_test.beton
- CO3D-C/{corruption type}_{severity}.beton
- SAY_pretrain.beton
- SAY_linprobe_train.beton
- SAY_linprobe_test.beton
- SAY-C/{corruption type}_{severity}.beton
- IN_linprobe_train.beton
- IN_linprobe_test.beton
- IN-C/{corruption type}_{severity}.beton


## Datasets for Silhouette and Cue-Conflict Shape Bias.

Download [Silhouette](https://github.com/rgeirhos/texture-vs-shape/tree/master/stimuli/filled-silhouettes) and [Cue-Conflict Shape Bias](https://github.com/rgeirhos/texture-vs-shape/tree/master/stimuli/style-transfer-preprocessed-512).

## Datasets for Depth estimation.

- 3D-PC depth estimation: The processed version used in this repository can be downloaded [here](https://drive.google.com/drive/folders/1PCfdYbyn_-aVJIDqIsXyDhnFZloP4dfm?usp=drive_link).
- Visual Cliff: The infant's egocentric pictures used in this repository can be downloaded [here](https://drive.google.com/drive/folders/1PCfdYbyn_-aVJIDqIsXyDhnFZloP4dfm?usp=drive_link).
