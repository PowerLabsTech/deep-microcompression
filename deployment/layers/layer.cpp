/**
 * @file layer.cpp
 * @brief Base layer implementation with support for:
 *      1. Non-quantized models (float)
 *      2. Static quantized models per tensor (int8_t)
 */

#include "layer.h"


float* Layer::forward(float* input, float* workspace_start, uint32_t workspace_size) {
    // Intentionally empty - to be implemented by derived classes
    return nullptr;
}

/**
 * @brief Default forward pass implementation for floating-point layers
 * @param input Pointer to input tensor (float)
 * @param output Pointer to output tensor (float)
 * 
 * @note This base implementation does nothing and should be overridden
 *       by derived layer classes.
 */
int8_t* Layer_SQ::forward(int8_t* input, int8_t* workspace_start, uint32_t workspace_size) {
    // Intentionally empty - to be implemented by derived classes
    return nullptr;
}
