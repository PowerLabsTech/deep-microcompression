/**
 * @file flatten.cpp
 * @brief Implementation of Flatten layer with support for:
 *      1. None quantized model (float)
 *      2. Static quantized model per tensor (int8_t)
 */

#include "flatten.h"


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
void Flatten::forward(float* workspace_start, uint32_t workspace_size) {
    // No-op: flatten is a shape-only operation, data stays in place
}



Flatten_SQ::Flatten_SQ(uint32_t input_size, uint8_t quantize_property) {
    this->input_size = input_size;
    this->quantize_property = quantize_property;
}

void Flatten_SQ::forward(int8_t* workspace_start, uint32_t workspace_size) {
    // No-op: flatten is a shape-only operation, data stays in place
}



uint32_t Flatten_SQ::get_output_size(void) {
    return this->input_size;
}

