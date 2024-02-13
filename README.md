# homography-estimation-v0

## Overview

This repository contains the implementation for the adding a module to
multi-view SSL algorithms that estimates the parameters of a random affine
transformation in order to improve performance.

## SSL Pretraining

```bash
python -m he.train --config_path <config_path>
```

## Linear Evaluation

```bash
python -m he.linear_eval.le --config_path <config_path> --model_path <model_path>
```

## Scripts

### [extract_backbone.py](scripts%2Fextract_backbone.py)

This script extracts the backbone encoder from a checkpoint file outputted from
`he.train`. This needs to be done prior to linear evaluation. Further, the
`--model_path` argument to the `he.linear_eval.le` script will be the path to
this extracted backbone.

```bash
python scripts/extract_backbone.py --checkpoint_path <checkpoint_path> --export_path <export_path>
```

### [show_ablation_results.py](scripts%2Fshow_ablation_results.py)

This script outputs the results of the affine transformation component ablation
study. This is the set of experiments in which the rotation, translation,
scale, and shear components are separately included into the module to see 
which of these 4 components plays the biggest role in performance.

```bash
python scripts/show_ablation_results.py [--ci]
```

### [show_bounded_results.py](scripts%2Fshow_bounded_results.py)

This script outputs the results of the set of experiments which investigate
cropping the image after the affine transformation to ensure none of the
induced background is included into the image prior to feeding it into the
model. This is done by solving an optimisation problem using [maxrect.py](he%2Fdata%2Fmaxrect.py).

```bash
python scripts/show_bounded_results.py [--ci]
```

### [show_concat_results.py](scripts%2Fshow_concat_results.py)

This scripts outputs the results of the set of experiments which investigate
performing concatenation as the aggregation function when feeding into the
affine module MLP which estimated the parameters of the transformation.

```bash
python scripts/show_concat_results.py [--ci]
```

### [show_double_results.py](scripts%2Fshow_double_results.py)

This scripts outputs the results of the set of experiments which investigate
performing the affine parameter estimation for both input random views, instead
of just one.

```bash
python scripts/show_double_results.py [--ci]
```

### [show_epoch_results.py](scripts%2Fshow_epoch_results.py)

This scripts outputs the results for linear evaluation for all epochs of the
'main' models (i.e. those that using vector difference as the aggeregation
function, estimate the parameters for only 1 random view, and don't perform
bounded affine transformations).

```bash
python scripts/show_epoch_results.py
```

### [show_epoch_results_raw.py](scripts%2Fshow_epoch_results_raw.py)

This script outputs the same results as [show_epoch_results.py](scripts%2Fshow_epoch_results.py)
but without taking the average over trials for each epoch/checkpoint. This is
to enable the plotting of confidence intervals on the line plots for the paper.

```bash
python scripts/show_epoch_results_raw.py
```

### [show_main_results.py](scripts%2Fshow_main_results.py)

The script outputs the results for the linear evaluation for the final epoch's
checkpoint (`90_model_tiny_imagenet`) for the 'main' checkpoint/model (defined
above).

```bash
python scripts/show_main_results.py
```
