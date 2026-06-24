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
void ReLU::forward(float* workspace_start, uint32_t workspace_size) {
    // Apply ReLU function element-wise
    for (uint32_t i = 0; i < this->input_size; i++) {
        activation_write_float(workspace_start, i, relu(activation_read_float(workspace_start, i)));
    }
}

uint32_t ReLU::get_output_size(void) {
    return this->input_size;
}


ReLU6::ReLU6(uint32_t input_size) {
    this->input_size = input_size;
}

void ReLU6::forward(float* workspace_start, uint32_t workspace_size) {
    // Apply ReLU6 function element-wise
    for (uint32_t i = 0; i < this->input_size; i++) {
        activation_write_float(workspace_start, i, relu6(activation_read_float(workspace_start, i)));
    }
}


uint32_t ReLU6::get_output_size(void) {
    return this->input_size;
}


ReLU_SQ::ReLU_SQ(uint32_t input_size, int8_t input_zero_point, uint8_t quantize_property) {
    this->input_size = input_size;
    this->input_zero_point = input_zero_point;
    this->quantize_property = quantize_property;
}

void ReLU_SQ::forward(int8_t* workspace_start, uint32_t workspace_size) {    
    void (*activation_write_packed_intb) (int8_t*, uint32_t, int8_t);
    int8_t (*activation_read_packed_intb) (int8_t*, uint32_t);
    
    get_activation_write_packed_intb(this->quantize_property, &activation_write_packed_intb);
    get_activation_read_packed_intb(this->quantize_property, &activation_read_packed_intb);

    // Apply quantized ReLU function element-wise
    for (uint32_t i = 0; i < this->input_size; i++) {
        activation_write_packed_intb(workspace_start, i, relu_zero_point(activation_read_packed_intb(workspace_start, i), this->input_zero_point));
        
    }
}


uint32_t ReLU_SQ::get_output_size(void) {
    return this->input_size;
}



ReLU6_SQ::ReLU6_SQ(uint32_t input_size, int8_t input_zero_point, int8_t input_six_point, uint8_t quantize_property) {
    this->input_size = input_size;
    this->input_zero_point = input_zero_point;
    this->input_six_point = input_six_point;
    this->quantize_property = quantize_property;
}

void ReLU6_SQ::forward(int8_t* workspace_start, uint32_t workspace_size) {        
    void (*activation_write_packed_intb) (int8_t*, uint32_t, int8_t);
    int8_t (*activation_read_packed_intb) (int8_t*, uint32_t);
    
    get_activation_write_packed_intb(this->quantize_property, &activation_write_packed_intb);
    get_activation_read_packed_intb(this->quantize_property, &activation_read_packed_intb);

    // Apply quantized ReLU6 function element-wise
    for (uint32_t i = 0; i < this->input_size; i++) {
        activation_write_packed_intb(workspace_start, i, relu6_zero_point(activation_read_packed_intb(workspace_start, i), this->input_zero_point, this->input_six_point));
    }
}



uint32_t ReLU6_SQ::get_output_size(void) {
    return this->input_size;
}