# Multi-Input-Protein
This is repository for the Multi-Input PFP.

This has following main python files.
  1) Model.py - This python file contains the code for CNN-based segment encoder.
  2) Multi_Input.py - This contains the code for segmentation + Multi-Input training + Testing.
  3) Multi_Input_Hybrid.py - This contains the code for segmentation + Multi-Input training + ESM2 + Testing.
  4) Predict.py - This contains the code for the predicting on unseen data.
  5) Evaluate.py - This contains the code for the evaluation metrics.

This repository contains the dataset for the BP and MF gene ontology. The CAFA3 dataset can be downlaoded from https://deepgo.cbrc.kaust.edu.sa/data/

# Requirements
1. Python 3.14
2. Tensorflow >= 2.0
3. Keras
4. Pandas, Numpy

# Dataset
The dataset for UniprotKB, BP and MF is presents in the BP and MF folders, respectively.

# Training from scratch
Download the BP and MF datasets and set the training path in `Multi_Input.py` and `Multi_Input_Hybrid.py`.
The major steps with training are as follows:

Step 1: Run `python Model.py`                   <==== Create an instance of multi-input classifier

Step 2: Run `python Evaluate.py`                   <==== Execute code for evauation metrics

Step 3: Run `python Multi_Input_Hybrid.py`    <==== Start the training + Predict

# Alternative approach to run code:

Run the example python file inside Example directory in the Google Colab. Set the dataset directory if needed.
