# Predicting bitrate ladder using pre-trained DNN features

This Repository contains code and data relating to the paper "**Efficient Bitrate Ladder Construction using Transfer Learning and Spatio-Temporal Features**"

## Prerequisites 
To install the prerequisites on `Ubuntu:20:04` using `miniconda3:23.10.0-1` run the following:
```shell
apt update
apt install gcc g++ libgl1-mesa-glx libsm6 libxext6
conda config --set channel_priority strict
conda config --add channels conda-forge
conda env create -f env_part01.yml
conda env update -n torch_env -f env_part02.yml --prune
conda activate torch_env
```
You might need to change `prefix:` in both `env_part01.yml` and `env_part02_yml` based on the installation directory of `conda`.

## How to train
To perform training you need to download the following files from [here]() and put them into the repository:

1. Download the Slowfast model weights (`SLOWFAST_8x8_R50.pkl`) and store it under `data/checkpoints/Kinetics`.
2. Download the video (`videos_dataframe.csv`) and encode (`encodes_dataframe.csv`) information tables and store them under `data/dataframes`.
3. Download and extract the DNN features (`features.tar.gz`) and store them under `data/features`.

After downloading the data, you can run the following command to train the model on the extracted spatial and temporal features:
```shell
python3 src/SME_main.py
```
The training outputs will be stored in `data/results/train`. After training you can use the following command to do inference:
```shell
python3 src/SME_main.py -inference
```
The inference outputs will be stored in `data/results/inference`. Finally, you can use the following command to construct the actual and predicted bitrate ladders:
```shell
python3 src/bitrate_ladder_constructor.py
```
The final result tables will be stored in `data/results/final`.
