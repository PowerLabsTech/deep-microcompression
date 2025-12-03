# VGG-13 TFLite vs. DMC Comparison Guide

This notebook reproduces the VGG-13 experiments comparing TensorFlow Lite (TFLite) and Deep Microcompression (DMC) performanc (Table 3) from the "Deep Microcompression" paper.

## Required File Structure

This script assumes it is located within the original project's directory structure under the experiments directory. The development module must be accessible two levels up.

## Experiment Overview

The experiment compares accuracy and model size across three quantization schemes:
1. **Float32 (Baseline):** No quantization.
2. **Dynamic Quantization:** Weights are quantized, activations dynamically quantized at runtime.
3. **Static Quantization (Int8):** Weights and activations are quantized using calibration data.

## Methodology

1. **Source of Truth:** Uses a pre-trained VGG-13 (Batch Norm) model from PyTorch Hub (`cifar100_vgg13_bn`).
2. **Weight Transfer:** Copies weights from the PyTorch model to an equivalent TensorFlow/Keras model to ensure an exact baseline match.
3. **DMC Conversion:** Converts the PyTorch model to a DMC `Sequential` model.
4. **TFLite Conversion:** Converts the Keras model to TFLite flatbuffers using standard optimization flags.
5. **Evaluation:** Both frameworks evaluate on the same CIFAR-100 test set. PyTorch is forced to CPU to match the TFLite execution environment.
