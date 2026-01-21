/**
 * @file activation.cpp
 * @brief Implementation of ReLU activation layer with support:
 *      1. None quantized model
 *      2. Dynamic quantized model per tensor
 *          - 8 bit
 *          - 4 bit
 *      3. Static quantized model per tensor
 *          - 8 bit
 *          - 4 bit
 */

#include "activation.h"


#if !defined(QUANTIZATION_SCHEME) || QUANTIZATION_SCHEME != STATIC

/**
 * @brief Constructor for floating-point ReLU layer
 * @param input_size Number of elements in input tensor
 */
ReLU::ReLU(uint32_t input_size) {
    this->input_size = input_size;
}

/**
 * @brief Forward pass for floating-point ReLU
 * @param input Pointer to input tensor (float)
 * @param output Pointer to output tensor (float)
 * 
 * Computes: output[i] = max(0, input[i]) for each element
 */
void ReLU::forward(float* input, float* output) {
    // Apply ReLU function element-wise
    for (uint32_t i = 0; i < this->input_size; i++) {
        act_write_float(output, i, relu(act_read_float(input, i)));
    }
}


ReLU6::ReLU6(uint32_t input_size) {
    this->input_size = input_size;
}

void ReLU6::forward(float* input, float* output) {
    // Apply ReLU6 function element-wise
    for (uint32_t i = 0; i < this->input_size; i++) {
        act_write_float(output, i, relu6(act_read_float(input, i)));
    }
}


#else // QUANTIZATION_SCHEME


ReLU::ReLU(uint32_t input_size, int8_t input_zero_point) {
    this->input_size = input_size;
    this->input_zero_point = input_zero_point;
}

void ReLU::forward(int8_t* input, int8_t* output) {
    // Apply quantized ReLU function element-wise
    for (uint32_t i = 0; i < this->input_size; i++) {
        // output[i] = relu_zero_point(input[i], this->input_zero_point);
        act_write_packed_intb(output, i, relu_zero_point(act_read_packed_intb(input, i), this->input_zero_point));
        
    }
}

ReLU6::ReLU6(uint32_t input_size, int8_t input_zero_point, int8_t input_six_point) {
    this->input_size = input_size;
    this->input_zero_point = input_zero_point;
    this->input_six_point = input_six_point;
}

void ReLU6::forward(int8_t* input, int8_t* output) {
    // Apply quantized ReLU6 function element-wise
    for (uint32_t i = 0; i < this->input_size; i++) {
        act_write_packed_intb(output, i, relu6_zero_point(act_read_packed_intb(input, i), this->input_zero_point, this->input_six_point));
    }
}


#endif // QUANTIZATION_SCHEME
