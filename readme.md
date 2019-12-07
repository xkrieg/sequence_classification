#### Experiments on Sequence Analysis

The purpose of this project is to explore the methodology related to emotion sequence classification. In particular, we aim to identify the minimum parameters and data requirements for classification while avoiding overfitting. This project will contain several components:

1. Random Data Simulation 
2. Comparisons Deep learning model training and testing 
3. Real case use examples 

##### Random Data Simulation

By generating sample datasets with random sequence information that does not map on to meaningful outcome data, we hope to identify how much data is required for the model to "break" (show adequately low accuracy statistics for random data). Overfitting is a problem with sequence classification projects with a small number of sequences because the deep learning model can simply learn to identify the sequence (and its corresponding output) rather than learn about the meaningful features contained within that sequence.  

Several paramters we will test include number of sequences, window size, number of channels, and different types of outputs (binary; multi-class; multi-label, multi-class).

The project is found in the `simulation` folder and can be run using the `main.py` command.