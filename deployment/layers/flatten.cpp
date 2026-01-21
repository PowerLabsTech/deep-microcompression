/**
 * @file flatten.cpp
 * @brief Implementation of Flatten layer with support for:
 *      1. None quantized model (float)
 *      2. Static quantized model per tensor (int8_t)
 */

#include "flatten.h"


#if !defined(QUANTIZATION_SCHEME) || QUANTIZATION_SCHEME != STATIC

/**
 * @brief Constructor for floating-point Flatten layer
 * @param input_size Number of elements in input tensor
 * 
 * @note Flatten operation doesn't modify values, just reshapes the tensor
 */
Flatten::Flatten(uint32_t input_size) {
    this->input_size = input_size;
}

/**
 * @brief Forward pass for floating-point Flatten
 * @param input Pointer to input tensor (float)
 * @param output Pointer to output tensor (float)
 * 
 * Simply copies input to output as flattening is just a view operation.
 * Maintains same memory layout but changes tensor shape interpretation.
 */
void Flatten::forward(float* input, float* output) {
    // Perform element-wise copy (no transformation needed)
    for (uint32_t i = 0; i < this->input_size; i++) {
        act_write_float(output, i, act_read_float(input, i));
    }
}


#else // QUANTIZATION_SCHEME


Flatten::Flatten(uint32_t input_size) {
    this->input_size = input_size;
}

void Flatten::forward(int8_t* input, int8_t* output) {
    // Perform element-wise copy (no transformation needed)
    for (uint32_t i = 0; i < this->input_size; i++) {
        act_write_packed_intb(output, i, act_read_packed_intb(input, i));
    }
}

#endif // QUANTIZATION_SCHEME
