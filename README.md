# EfficientLEAF: A Slightly Modified Faster LEarnable Audio Frontend for Personal Use

This is a fork of the original repo by Jan Schlüter and Gerald Gutenbrunner. The only changes made thus far are some clean up and formatting so that I can more easily use the code in some of my own experiments, and I may add further modification

I have included the parts of the readme from the original repo that pertain to the frontend itself. [`efficientleaf.py`](efficientleaf.py) contains the dependencies from the original repo's `leaf.py` and as such, should be complete.

From here on in the README you are in Jan and Gerald's hands!

# EfficientLEAF: A Faster LEarnable Audio Frontend of Questionable Use
- Authors: Jan Schlüter, Gerald Gutenbrunner
- Paper: https://arxiv.org/abs/2207.05508

This is the (un-)official (parentheses added by Mark ;) ) PyTorch implementation for our EUSIPCO 2022 paper "EfficientLEAF: A Faster LEarnable Audio Frontend of Questionable Use".

## Introduction

[LEAF](https://openreview.net/forum?id=jM76BCb6F9m) is an audio frontend with Gabor filters of learnable center frequencies and bandwidths. It was proposed as an alternative to mel spectrograms, but is about 300x slower to compute. EfficientLEAF is a drop-in replacement for LEAF, only about 10x slower than mel spectrograms. We achieve this by dynamically adapting convolution filter sizes and strides, and by replacing PCEN (Per-Channel Energy Normalization) with better parallizable operations (median subtraction and temporal batch normalization).

## Reusing the frontend

When running a Python session from the repository root, an EfficientLEAF frontend can be initialized with:
```python
from model.efficientleaf import EfficientLeaf

frontend = EfficientLeaf(n_filters=80, min_freq=60, max_freq=7800,
                         sample_rate=16000,
                         num_groups=8, conv_win_factor=6, stride_factor=16)
```
This corresponds to the configuration "Gabor 8G-opt" from the paper. A smaller convolution window factor or a larger stride factor will be even faster, but too extreme settings will produce artifacts in the generated spectrograms.

Alternatively, the `main.py` script can be used to extract a trained or newly initialized network by appending the command `--ret-network` (with `--data-set "None"` it returns the network without dataloaders). This returns the entire network (frontend and EfficientNet backend), with access to the frontend via `network.frontend`.

## Implementation notes

We started off by porting the official [TensorFlow LEAF implementation](https://github.com/google-research/leaf-audio) to PyTorch, module by module, verifying that we get numerically identical output. This implementation is preserved in [`model/OG_leaf.py`](model/OG_leaf.py). We then modified the implementation in the following:
* The original implementation initializes the Gabor filterbank by computing a mel filterbank matrix, then [measuring the center and width](model/OG_leaf.py#L183) of each triangular filter in this (discretized) matrix. We instead [compute the center frequency and bandwidth](model/leaf.py#L12) analytically.
* The original implementation computes the real and complex responses as interleaved channels, we compute them as two blocks instead.
* We simplified the PCEN implementation to a single self-contained module.
* The original implementation learns PCEN parameters in linear space and does not guard the delta parameter to be nonnegative. We optionally learn parameters in log space (as in the original PCEN paper) instead. This was needed for training on longer inputs, as the original implementation regularly crashed with NaN in this case.

We verified that none of these changes affected classification performance after training.

## Citation

Please cite our paper if you use this repository in a publication:
```
@INPROCEEDINGS{2022eleaf,
author={Schl{\"u}ter, Jan and Gutenbrunner, Gerald},
  booktitle={Proceedings of the 30th European Signal Processing Conference (EUSIPCO)},
  title={{EfficientLEAF}: A Faster {LEarnable} Audio Frontend of Questionable Use},
  year=2022,
  month=sep}
```
