# LeNet-5 Experiment Reproduction Guide

This guide explains how to reproduce the LeNet-5 (Baseline, Pruned, and Quantized-Pruned) experiments from the "Deep Microcompression" paper.

## Required File Structure

This script assumes it is located within the original project's directory structure. The development module must be accessible three levels up from this script.


## Instructions

Open your terminal and navigate to this directory:

```
cd path/to/your/project/experiments/lenet5/
```


Make the bash script executable:

```
chmod +x setup_and_run.sh
```

Run the script. It will create a local Python virtual environment (venv/), install the required packages, and then execute the experiment.

```
./setup_and_run.sh
```


## What to Expect

The script will run the full experiment, which involves three stages:

1. Baseline Model: Trains the original LeNet-5 model (20 epochs with early stopping) and saves it as lenet5_state_dict.pth.

1. Pruned Model: Loads the baseline weights, applies the optimal structured pruning (conv2d_1: 9, linear_0: 50), and retrains the model (20 epochs).

1. Quantized-Pruned Model: Applies 4-bit static quantization to the pruned model and performs Quantization-Aware Training (QAT) (15 epochs).

The script will print the Accuracy and Model Size for each of these three stages, allowing your supervisor to easily verify the results from Table 2 in your paper.