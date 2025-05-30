# DeepAnthing - MonocularDepthEstimation CIL Project

We tackle the prominent problem of Depth Estimation in this project and have various solution attempts:
- [UNet](#unet)
- [Finetuning](#finetuning)
- [Depth Enhancement](#depth-enhancement)

## UNet
The main function of the file `unet_experiments.py` can be used to reproduce the results for the UNet models as shown in the paper.

## Finetuning
### DepthAnythingV2
In the branch ```LucaTestingBranch```, one can find the code to fine-tune the DepthAnythingV2 model. The pretrained weights for (depending on the encoder size) can be downloaded in the original repository: (https://github.com/DepthAnything/Depth-Anything-V2). To reproduce the results, one must clone the DepthAnythingV2 repository and add it to the system path. The notebook FineTuneDepthAnything.ipynb contains the pipeline to fine-tune the model.

### IndoorDepth
Due to this model's size, we could not run it locally. So, we used a Google Colab subscription to let it use dedicated GPUs for this task. I included a zip file with the code's changes in the branch ```FineTuneIndoorDepth```.

## Depth Enhancement
The Depth Enhancement process consists of two steps:
- Uncertainty Map
- Low Rank Approximation

For the Uncertainty Model, we train the respective double decoder model with the `double_decoder_setup.py` file.

Then, we use the outputs of this model in the file `lowrank.py` and post-process them according to the steps in the paper.

## Additional Files
The ' utils/' folder contains helper files such as the model architecture, loss functions, and train process specifications.

## Reproducability of Results
The paper's results should be easily reproducible using the given code and a random seed.

## Authors
Luca Conconi, Marino Eisenegger, Timon Fopp, Nick Hofstetter


TODO: Nick :) 

bis am Freitag, code hochgeladen. Jeder passt das README entsprechend an.

todo: mit unet_experiments.py kann man den UNet teil des papers reproducen
