## README: About this repository
This repository contains the code needed to run the experiments in "The Uncertainty in Uncertainty: Confidence Calibration in Neural Networks with Mixed-Label Data Augmentation".

## Creating mixed examples with different types of interpolations
**To create mixed-label augmentation sets of fixed sizes using SLI, *Slerp*, and *mixup*, SLI-CP, *Slerp*-CP, type:**

```
CUDA_VISIBLE_DEVICES = <gpus> python interpolate_sli_slerp_mixup.py

CUDA_VISIBLE_DEVICES = <gpus> python interpolate_slicp_slerpcp.py
```
These two interpolation files assume that you have pre-trained PixelVAE parameters in your directory, and they operate on CIFAR-10 by default. The dataset, as well as the number and types of interpolations, can be adjusted using the commented lines within the code.

## Training ResNet-110's 

**To train and get test-set predictions for a ResNet-110 using a pre-created mixed-label augmentation set:**

```
CUDA_VISIBLE_DEVICES = <gpus> python train_fixed_augmentations_model.py

CUDA_VISIBLE_DEVICES = <gpus> python eval_fixed_augmentations_model.py
```

**To train and get test-set predictions for a ResNet-110 with pixel-space interpolations (*mixup*) applied in every training batch:**

```
CUDA_VISIBLE_DEVICES = <gpus> python train_mixup_model.py

CUDA_VISIBLE_DEVICES = <gpus> python eval_mixup_model.py
```
The *mixup* files above draw on *mixup_generator.py* to generate mixed examples within each training batch. The file *mixup_generator.py* was adjusted based on the implementation by [yu4u](https://github.com/yu4u/mixup-generator). All six files above should to be adjusted to run on the weights for the ResNet-110(s) of interest.

**To train and get test-set predictions for a baseline ResNet-110:**

```
CUDA_VISIBLE_DEVICES = <gpus> python train_baseline_model.py

CUDA_VISIBLE_DEVICES = <gpus> python eval_baseline_model.py
```

## Training a PixelVAE for *Latent Blending*
**To train a PixelVAE using the architecture described in the paper, type:**

```

CUDA_VISIBLE_DEVICES = <gpus> python train_pixelvae.py

```

This file trains a 3-pixel receptive field PixelCNN on CIFAR-10 by default. The lines which need to be adjusted in order to train a PixelVAE on a different dataset, or with a different architecture, have been indicated with comments.

## Evaluating calibration

**To evaluate the calibration of ResNet-110’s, and then produce reliability diagrams, type:**

```
python calibration.py

python reliability.py
```
These two files, and the backup files they draw on (contained in the utility folder), were obtained from [markus93](https://github.com/markus93/NN_calibration) and adjusted as needed for this paper. They calculate ECE, MCE, error, and cross-entropy loss given the logit vectors for a set of neural networks. Thus, each neural network of interest must be evaluated using the appropriate "evaluation" file prior to running these two files.

**Citation**

If you use our methododology or code, please cite it using:

```
@misc{
leila2018nncalibrationmixedlabels,
title={The Uncertainty in Uncertainty: Confidence Calibration in Neural Networks with Mixed-Label Data Augmentation},
author={Leila Islam}, url={https://github.com/leilayasmeen/Calibration_LB_mixup},
}
```

**Requirements**: Python; Tensorflow; Keras

**Acknowledgements**

The *mixup* implementation is adapted from [yu4u](https://github.com/yu4u/mixup-generator); the PixelVAE training code is adapted from [igul222](https://github.com/igul222/PixelVAE); and the neural network training, evaluation, and calibration scripts from [markus93](https://github.com/markus93/NN_calibration).


