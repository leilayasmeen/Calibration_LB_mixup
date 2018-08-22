# Calibration_LB_mixup

## Creating mixed examples with different types of interpolations
**To create mixed-label augmentation sets of fixed sizes using SLI, Slerp, and mixup, type:**
*CUDA_VISIBLE_DEVICES = <devices you wish to use> python interpolate_sli_slerp_mixup.py*

**To create mixed-label augmentation sets of fixed sizes using SLI-CP and Slerp-CP, type:**
*CUDA_VISIBLE_DEVICES = <devices you wish to use> python interpolate_slicp_slerpcp.py*

These two interpolation files assumes that you have pre-trained PixelVAE parameters in the same directory, and it operates on CIFAR-10 by default. The dataset, as well as number and type of interpolations, can be adjusted using the commented lines within the code.

## Training ResNet-110's 

**To train a ResNet-110 on CIFAR-10 as a baseline:**
*CUDA_VISIBLE_DEVICES = <devices you wish to use> python train_baseline_model.py*

**To train a ResNet-110 on CIFAR-10 using a pre-created mixed label augmentation set, type:**
*CUDA_VISIBLE_DEVICES = <devices you wish to use> python train_fixed_augmentations_model.py*

**To train a ResNet-110 on CIFAR-10 using mixup, type:**
*CUDA_VISIBLE_DEVICES = <devices you wish to use> python train_mixup_model.py*

The file above draws on mixup_generator.py, as it generates mixed examples within every training batch. This file was adjusted based on the implementation by INSERT CITATION.

## Finding test predictions for ResNet 110's

**To obtain ResNet-110 predictions on the CIFAR-10 test set, type:**
*python eval_baseline_model.py*
*python eval_mixup_model.py*
*python eval_fixed_augmentations_model.py*

These three files should to be adjusted to run on the weights for the ResNet-110(s) of interest. Comments in the code have indicated where the adjustments need to be made.

## Training a PixelVAE to use for *Latent Blending*
**To train a PixelVAE using the architecture described in the paper, type:**
*CUDA_VISIBLE_DEVICES = <devices you wish to use> python train_pixelvae.py*

This file trains a 3-pixel receptive field PixelCNN on CIFAR-10 by default. The lines which need to be adjusted in order to train a PixelVAE on a different dataset, or with a different architecture, have been indicated with comments.

## Evaluating calibration and generalization ability

**To evaluate the calibration of ResNet-110’s, type:**
*python calibration.py*

**To produce reliability diagrams, type:**
*python reliability.py*

These two files, and the backup files they draw on (contained in the utility folder), were obtained from Kangsepp (2018) and adjusted as needed for this paper. They calculates ECE, MCE, error, and cross-entropy loss given the logit vectors for a set of neural networks. Thus, each neural network of interest must be evaluated using the "evaluations" files prior to running these files.







