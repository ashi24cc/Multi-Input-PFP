# Incremental-Training-Protein
This is repository for the Incremental learning.

This has following main python files.
  1) Model.py - This python file contains the code for CNN-based segment encoder.
  2) Incremental_Training_Predict.py - This contains the code for segmentation + incremental training + Testing.
  3) Predict.py - This contains the code for the predicting on unseen data.
  4) Evaluate.py - This contains the code for the evaluation metrics.

This repository contains the dataset for the BP and MF gene ontology. The CAFA3 dataset can be downlaoded from https://deepgo.cbrc.kaust.edu.sa/data/

# Requirements
1. Python 3.14
2. Tensorflow >= 2.0
3. Keras
4. Pandas, Numpy

# Dataset
The dataset for BP and MF is presents in the BP and MF folders, respectively.

# Training from scratch
Download the BP and MF datasets and set the training path in `Model.py` and `Incremental.py`.
The major steps with training are as follows:

Step 1: Run `python Model.py`                   <==== Create an instance of LiteSeqCNN

Step 2: Run `python Evaluate.py`                <==== Code for the evaluation metrics.

Step 3: Run `python Incremental_Training_Predict.py`    <==== Start the incremental training + Predict

# Alternative approach to run code:

Rune the example python file inside Example directory in the Google Colab. Set the dataset directory if needed.

The paper is under review at Scientific Reports.
