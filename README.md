# DeepAnthing - MonocularDepthEstimation CIL Project

We tackle the prominent problem of Depth Estimation in this project and have various solution attempts:
- [UNet](#features)
- [Finetuning](#installation)
- [Depth Enhancement](#depth-enhancement)

## UNet



## Finetuning



## Depth Enhancement
The Depth Enhancement process consist of two steps:
- Uncertainty Map
- Low Rank Approximation

For the Uncertainty Model we train the the respective double decoder model with the `double_decoder_setup.py` file.

Then we use outputs of this model in the file `lowrank.py` file and post-process the outputs according to the steps in the paper.

## Additional Files
Helper files such as the model architecutre, loss functions and train process specifications can be found in the `utils/` folder.

## Reproducability of Results
The results of the paper should be easily reproducable with the given code and the use of a random seed.


TODO: Nick :) 

bis am Freitag, code hochgeladen. Jeder passt das README entsprechend an.

todo: mit unet_experiments.py kann man den UNet teil des papers reproducen
