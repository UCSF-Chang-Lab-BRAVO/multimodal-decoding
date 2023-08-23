# Neuroprosthesis for Speech Decoding and Avatar Control

This repository contains the code used to train and evaluate text-decoding models for our paper: 'A high-performance neuroprosthesis for speech decoding and avatar control'.

## Getting Started

To set up the project, follow these steps:

### 1. Set up Conda Environment

NOTE: You must make separate environments for text and synthesis. 

Create a Conda environment with Python 3.9.13:

```shell
conda create -n bci_env python=3.9.13
conda activate bci_env
```

### 2. Install the requirements
```shell
pip install -r requirements.txt
```

You will also need to set up an account with [wandb.ai](https://wandb.ai) to track model loss and results. 
Visit their website to learn more about features and sign up for an account. 

### 3. Try out the models on 1024-word-General dataset. 

You can run the notebook in `text_example_decoding.ipynb` to train a decoder and evaluate it on our test data from the paper.
This notebook trains a model on the 1024-word-General data, and evaluates it on our realtime test data and uses early stopping
to enable fast real-time decoding using neural acivity. Your performance metrics should be very close to the metrics in the paper. 


Note - the neural data is at 33.33 Hz

Set the `data_dir` to be where the downloaded data lives, 
and set `curdir` to be your current directory

Change `device` to be `cpu` or `cuda` depending on the hardware you have available


### 5. Using limited sets
You can train and decode with the 50-phrase AAC set using `50_phrase_nb.ipynb`. 
This notebooks are designed for model training only and will be updated shortly with full early stopping code.

