/**
 * @file layer.h
 * @brief Base layer interface with support for:
 *       1. Non-quantized models (float)
 *       2. Static quantized models per tensor (int8_t)
 */

#ifndef LAYER_H
#define LAYER_H

#include "../core/define.h"

#include <stdint.h>  // For int8_t type
#include <float.h>   // For floating-point constants
#include <math.h>    // For math operations


// #define QUANTIZATION_SCHEME DYNAMIC 
// #define QUANTIZATION_GRANULARITY PER_TENSOR

#if !defined(QUANTIZATION_SCHEME) || QUANTIZATION_SCHEME != STATIC

/**
 * @class Layer
 * @brief Abstract base class for all floating-point layers
 * 
 * Provides the interface for forward propagation in non-quantized networks.
 * All concrete layer types must implement the forward() method.
 */
class Layer {
public:
    /**
     * @brief Forward pass interface for floating-point layers
     * @param input Pointer to input tensor (float)
     * @param output Pointer to output tensor (float)
     * 
     * @note Pure virtual function - must be implemented by derived classes
     */
    virtual float* forward(float* input, float* output) = 0;
};

#else // QUANTIZATION_SCHEME

class Layer {
public:
    /**
     * @brief Forward pass interface for floating-point layers
     * @param input Pointer to input tensor (float)
     * @param output Pointer to output tensor (float)
     * 
     * @note Pure virtual function - must be implemented by derived classes
     */
    virtual int8_t* forward(int8_t* input, int8_t* output) = 0;
};



#endif // QUANTIZATION_SCHEME

#endif // LAYER_H