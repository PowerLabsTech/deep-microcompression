/*
 * @file deep_microcompression.h
 * @brief Main Header for the Deep Microcompression (DMC) Inference Engine.
 *
 * This file serves as the single entry point for the generated C library.
 * It aggregates all layer definitions and the sequential container required
 * to run the optimized models on bare-metal hardware.
 *
 * Usage in MCU Project:
 * 1. Include this header: #include "deep_microcompression.h"
 * 2. Include the generated model header: #include "my_model.h"
 * 3. Call inference: my_model.forward(input_buffer);
 *
 * Reference:
 * "DMC generates generic C code that performs consistently on non-ARM architectures...
 * producing a standalone library."
 */

#ifndef DEEP_MICROCOMPRESSION
#define DEEP_MICROCOMPRESSION

// Core Container
#include "models/sequential.h"

// Layer Implementations
#include "layers/activation.h"
#include "layers/batchnorm.h"
#include "layers/branch.h"
#include "layers/conv.h"
#include "layers/flatten.h"
#include "layers/fused_layers.h"
#include "layers/linear.h"
#include "layers/padding.h"
#include "layers/pooling.h"

#endif // DEEP_MICROCOMPRESSION