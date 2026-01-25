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
float* Flatten::forward(float* input, float* workspace_start, uint32_t workspace_size) {
    // Getting the output start address with the input size as offset
    float* output = input == workspace_start ? workspace_start + workspace_size - this->input_size : workspace_start;

    // Perform element-wise copy (no transformation needed)
    for (uint32_t i = 0; i < this->input_size; i++) {
        activation_write_float(output, i, activation_read_float(input, i));
    }

    return output;
}


Flatten_SQ::Flatten_SQ(uint32_t input_size, uint8_t quantize_property) {
    this->input_size = input_size;
    this->quantize_property = quantize_property;
}

int8_t* Flatten_SQ::forward(int8_t* input, int8_t* workspace_start, uint32_t workspace_size) {
    // Getting the output start address with the input size as offset
    int8_t* output = input == workspace_start ? workspace_start + workspace_size - (uint32_t)ceil((float)this->input_size / get_activation_data_per_byte(this->quantize_property)) : workspace_start;

    void (*activation_write_packed_intb) (int8_t*, uint32_t, int8_t);
    int8_t (*activation_read_packed_intb) (int8_t*, uint32_t);
    
    
    get_activation_write_packed_intb(this->quantize_property, &activation_write_packed_intb);
    get_activation_read_packed_intb(this->quantize_property, &activation_read_packed_intb);

    // Perform element-wise copy (no transformation needed)
    for (uint32_t i = 0; i < this->input_size; i++) {
        activation_write_packed_intb(output, i, activation_read_packed_intb(input, i));
    }

    return output;
}
