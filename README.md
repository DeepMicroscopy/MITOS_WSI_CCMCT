![Large-Scale Canine Cutaneous Mast Cell Tumor Data Set for Mitotic Figure Assessment on Whole Slide Images](ccmct_logo.png)

## MITOS_WSI_CCMCT data set

This repository contains all code needed to derive the Technical Validation of our paper:
Bertram, C. A., Aubreville, M., Marzahl, C., Maier, A., & Klopfleisch, R. (2019). A large-scale dataset for mitotic figure assessment on whole slide images of canine cutaneous mast cell tumor. Scientific Data, 6(274), 1--9.

A short video about it can be found on [youtube](https://youtu.be/1UV1_a5qyQM).

It contains two main parts:

### Data set variant evaluation

This folder contains the evaluation for all variants, i.e. the manually labelled (MEL), the hard-example-augmented manually labelled (HEAEL) and the object-detection augmented manually expert labelled (ODAEL) variant.

Main results of the data set variants based on a one- and two-stage-detector can be found in [Evaluation.ipynb](Evaluation.ipynb).

### Ablation Study

One main question behind our research was: How big does a data set need to be? In order to find out, we reduced the set both in quantity of WSIs and area. All previous data sets typically had an annotated area of around 10 High Power Fields (approx 2 square millimeters) per tumor. We assumed that this would not be enough to account for all data variance, and were able to show this for our data set.

Main results of the ablation study were calculated in the ipython notebook: [AblationStudy_Evaluation.ipynb](AblationStudy_Evaluation.ipynb)

## Setting up the environment

Besides [https://github.com/fastai/](fast.ai) you can use the following notebook to set up the dataset for you: [Setup.ipynb](Setup.ipynb). The download of the WSI from figshare will take a while. Once everything has been downloaded, you can either use the data loaders provided in this repository, or, if you want to get a visual impression of the dataset, use [our annotation tool SlideRunner](https://github.com/maubreville/SlideRunner)

## Training notebooks

The training process can be seen in the notebooks for the respective dataset variants:

[RetinaNet-CCMCT-MEL.ipynb](RetinaNet-CCMCT-MEL.ipynb)

[RetinaNet-CCMCT-HEAEL.ipynb](RetinaNet-CCMCT-HEAEL.ipynb)

[RetinaNet-CCMCT-ODAEL.ipynb](RetinaNet-CCMCT-ODAEL.ipynb)

Note: The results (as submitted) were done with a previous version of the notebook, which was afterwards simplified and cleaned up. However, besides the random factor in sampling, there should be no difference between the networks generated with these notebooks and the ones used for the manuscript.
