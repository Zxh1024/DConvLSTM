# DConvLSTM-SAC

This repository includes code for the paper: DConvLSTM-SAC: Incorporating spatial autocorrelation into Deformable ConvLSTM for hourly precipitation forecasting.

We present a new model, DConvLSTM-SAC, that employs deformable convolution for irregularly distributed precipitation data and measures spatial autocorrelation through the local Moran index to enhance spatial information extraction. Compared to decision tree regression, random forest regression, recurrent neural networks, and convlstm, the DConvLSTM-SAC model exhibits superior performance.

![image](https://github.com/Zxh1024/DConvLSTM/assets/51319967/efa57cc7-2c02-4f17-8e3d-de207cf1ffcd)

## Setup

All code was developed and tested on Nvidia RTX3060Ti in the following environment:

- Python 3.6
- torch
- matplotlib 
- opencv3
- scikit-image
- numpy
- tensorflow>=2.0
- cuda>=8.0
- cudnn>=5.0

## Quick Start

The training script has a number of command-line flags that you can use to configure the model architecture, hyperparameters, and input / output settings.
Below are some parameters about our model:

- `--model_name`: The model name. Default value is `dconvlstm_sac`.
- `--pretrained_model`: Directory to find our pretrained models. See below for the download instruction.
- `--num_hidden`: Comma separated number of units of dconvlstm_sac
- `--filter_size`: Filter of a single dconvlstm_sac layer.
- `--layer_norm`: Whether to apply tensor layer norm.

- `--is_training`: Is it training or testing.
- `--train_data_paths`, `--valid_data_paths`: Training and validation dataset path.
- `--gen_frm_dir`: Directory to store the prediction results.
- `--allow_gpu_growth`: Whether allows GPU to grow.
- `--input_length 6`: Input sequence length.
- `--total_length 3`: Input and output sequence length in total.
