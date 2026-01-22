/**
 * @file flatten.h
 * @brief Header for Flatten layer with support for:
 *      1. Non-quantized models (float)
 *      2. Static quantized models per tensor (int8_t)
 */

#ifndef FLATTEN_H
#define FLATTEN_H

#include "layer.h"

#if !defined(QUANTIZATION_SCHEME) || QUANTIZATION_SCHEME != STATIC



/**
 * @class Flatten
 * @brief Flatten layer implementation for floating-point models
 * 
 * This layer reshapes multi-dimensional input into 1D output without
 * modifying values. Used for transition between convolutional and dense layers.
 */
class Flatten : public Layer {
private:
    uint32_t input_size;  ///< Total number of elements in input tensor

public:
    /**
     * @brief Constructor for floating-point Flatten layer
     * @param size Number of elements in input tensor
     */
    Flatten(uint32_t size);

    /**
     * @brief Forward pass for floating-point flatten operation
     * @param input Pointer to input tensor (float)
     * @param output Pointer to output tensor (float)
     */
    float* forward(float* input, float* workspace_start, uint32_t workspace_size);
};




#else // QUANTIZATION_SCHEME

class Flatten : public Layer {
private:
    uint32_t input_size;  ///< Total number of elements in input tensor

public:
    /**
     * @brief Constructor for quantized Flatten layer
     * @param size Number of elements in input tensor
     */
    Flatten(uint32_t size);

    /**
     * @brief Forward pass for quantized flatten operation
     * @param input Pointer to input tensor (int8_t)
     * @param output Pointer to output tensor (int8_t)
     */
    int8_t* forward(int8_t* input, int8_t* workspace_start, uint32_t workspace_size);
};


#endif // QUANTIZATION_SCHEME

#endif // FLATTEN_H