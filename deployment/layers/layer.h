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
     * @param input Pointer to input tensor for the layer(float)
     * @param workspace_start Pointer to workspace start start (float)
     * @param workspace_size size of the pre-allocated workspace
     * 
     * @note Pure virtual function - must be implemented by derived classes
     */
    virtual float* forward(float* input, float* workspace_start, uint32_t workspace_size) = 0;
};


class Layer_SQ {
public:
    /**
     * @brief Forward pass interface for floating-point layers
     * @param input Pointer to input tensor for the layer(int8_t)
     * @param workspace_start Pointer to workspace start start (int8_t)
     * @param workspace_size size of the pre-allocated workspace
     * 
     * @note Pure virtual function - must be implemented by derived classes
     */
    uint8_t quantize_property;
    virtual int8_t* forward(int8_t* input, int8_t* workspace_start, uint32_t workspace_size) = 0;
};


#endif // LAYER_H