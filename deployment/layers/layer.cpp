/**
 * @file layer.cpp
 * @brief Base layer implementation with support for:
 *      1. Non-quantized models (float)
 *      2. Static quantized models per tensor (int8_t)
 */

#include "layer.h"

#if !defined(QUANTIZATION_SCHEME) || QUANTIZATION_SCHEME != STATIC

float* Layer::forward(float* input, float* output) {
    // Intentionally empty - to be implemented by derived classes
    return nullptr;
}


#else // QUANTIZATION_SCHEME

/**
 * @brief Default forward pass implementation for floating-point layers
 * @param input Pointer to input tensor (float)
 * @param output Pointer to output tensor (float)
 * 
 * @note This base implementation does nothing and should be overridden
 *       by derived layer classes.
 */
int8_t* Layer::forward(int8_t* input, int8_t* output) {
    // Intentionally empty - to be implemented by derived classes
    return nullptr;
}



#endif // QUANTIZATION_SCHEME
