# TFLite vs. DMC Comparison Reproduction Guide

This guide explains how to reproduce the TFLite vs. DMC comparison (Table 2) from the "Deep Microcompression" paper.

## Required File Structure

This script assumes it is located within the original project's directory structure under the experiments directory. The development module must be accessible two levels up.


## What to Expect

The script will run the full comparison, which involves:

1. Baseline TF Model: Trains the TensorFlow/Keras LeNet-5 model (25 epochs) and saves it as lenet5_model.keras. This is the single source for both TFLite and DMC models.

1. Comparison Stage 1 (Float32): Compares the accuracy and size of the TFLite Float32 model vs. the DMC Float32 model.

1. Comparison Stage 2 (Dynamic Quantization): Compares TFLite Dynamic Quantization vs. DMC Dynamic Quantization.

1. Comparison Stage 3 (Static Quantization): Compares TFLite Static INT8 Quantization vs. DMC Static INT8 Quantization.

The script will print the Accuracy and Model Size for all six models, finishing with a summary table that directly reproduces the data for Table 2.