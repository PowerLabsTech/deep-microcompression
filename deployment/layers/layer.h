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
protected:
    uint8_t *buffer;

public:
    /**
     * @brief In-place forward pass for floating-point layers.
     * @param workspace_start Shared workspace buffer; input on entry, output on exit.
     * @param workspace_size  Number of float elements in workspace_start.
     */
    Layer(uint8_t* buffer);
    virtual void forward(float* workspace_start, uint32_t workspace_size) = 0;
    virtual uint32_t get_output_size(void) = 0;
};


class Layer_SQ {
protected:
    uint8_t *buffer;

public:
    /**
     * @brief In-place forward pass for statically-quantized layers.
     * @param workspace_start Shared workspace buffer (int8_t*); input on entry, output on exit.
     * @param workspace_size  Size of workspace_start in bytes.
     */
    Layer_SQ(uint8_t* buffer);
    virtual void forward(int8_t* workspace_start, uint32_t workspace_size) = 0;
    virtual uint32_t get_output_size(void) = 0;
};


#endif // LAYER_H