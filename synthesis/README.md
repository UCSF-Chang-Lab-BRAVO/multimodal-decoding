Set up instructions

First you will need to create a conda env with at least Python 3.8

```shell
conda create -n bci_env python=3.8
conda activate bci_env
```

Next, install dependencies
```
cd synthesis

pip install -r requirements.txt
pip install -U pip setuptools 
pip install -e .
```

Please install the GSLM library from source

```
https://github.com/facebookresearch/fairseq
```

Please add the directory where data is stored to under root (~41 GB)


The synthesis_example.ipynb notebook does the following:
1. Trains the 50 phrase model.
2. Performs inference on neural data real-time block from each sentence set using real-time models.
3. Allows for performing inference using model trained from step 1.
4. Displays example waveforms (chance, personalized, ground-truth, decoded).
5. Generates the figures from steps 2-4.
6. Generate perceptual accuracy figures from the Amazon Mechanical Turk perceptual study.