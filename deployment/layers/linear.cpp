/**
 * @file linear.cpp
 * @brief Implementation of Linear (fully-connected) layer with support for:
 *       1. Non-quantized models (float)
 *       2. Dynamic quantized models per tensor (float input + quantized weights)
 *       3. Static quantized models per tensor (all quantized)
 */

#include "linear.h"


/**
 * @brief Constructor for floating-point Linear layer
 * @param output_size Number of output neurons
 * @param input_size Number of input features
 * @param weight Pointer to weight matrix (row-major, shape [output_size, input_size])
 * @param bias Pointer to bias vector (size [output_size])
 */
Linear::Linear(uint16_t output_size, uint16_t input_size, 
              const float* weight, const float* bias) {
    this->output_size = output_size;
    this->input_size = input_size;
    this->weight = weight;
    this->bias = bias;
}

/**
 * @brief Forward pass for floating-point Linear layer
 * @param input Pointer to input tensor (float)
 * @param output Pointer to output tensor (float)
 * 
 * Computes: output = input * weight^T + bias
 */
float* Linear::forward(float* input, float* workspace_start, uint32_t workspace_size) {
    // Getting the output start address with the input size as offset
    float* output = input == workspace_start ? workspace_start + workspace_size - this->output_size : workspace_start;

    float output_temp;
    for (uint16_t j = 0; j < this->output_size; j++) {
        output_temp = this->bias ? parameter_read_float(this->bias, j) : 0;
        // Matrix-vector multiplication
        for (uint16_t i = 0; i < this->input_size; i++) {
            output_temp += activation_read_float(input, i) * parameter_read_float(this->weight, (j * this->input_size) + i);
        }
        activation_write_float(output, j, output_temp);
    }

    return output;
}


/**
 * @brief Constructor for dynamically quantized Linear layer
 * @param output_size Number of output neurons
 * @param input_size Number of input features
 * @param weight Pointer to quantized weight matrix (int8_t)
 * @param weight_scale Scaling factor for weights
 * @param bias Pointer to floating-point bias vector
 */
Linear_DQ::Linear_DQ(uint16_t output_size, uint16_t input_size,
              const int8_t* weight, const float* bias,
              float* weight_scale, uint8_t quantize_property) {
    this->output_size = output_size;
    this->input_size = input_size;
    this->weight = weight;
    this->bias = bias;
    this->weight_scale = weight_scale;

    this->quantize_property = quantize_property;
}

/**
 * @brief Forward pass for dynamically quantized Linear layer
 * @param input Pointer to floating-point input tensor
 * @param output Pointer to floating-point output tensor
 * 
 * Computes: output = input * dequant(weight)^T + bias
 */
float* Linear_DQ::forward(float* input, float* workspace_start, uint32_t workspace_size) {
    // Getting the output start address with the input size as offset
    float* output = input == workspace_start ? workspace_start + workspace_size - this->output_size : workspace_start;

    int8_t (*parameter_read_packed_intb) (const int8_t*, uint32_t);
    
    get_parameter_read_packed_intb(this->quantize_property, &parameter_read_packed_intb);

    float output_temp;

    for (uint16_t j = 0; j < this->output_size; j++) {
        output_temp = 0;
        for (uint16_t i = 0; i < this->input_size; i++) {
            output_temp += activation_read_float(input, i) * parameter_read_packed_intb(this->weight, (j * this->input_size) + i);
        }
        uint8_t scale_index = get_granularity(this->quantize_property) == PER_CHANNEL ? j : 0;

        activation_write_float(output,
            j,
            (this->bias ? 
            (output_temp * parameter_read_float(this->weight_scale, scale_index) + parameter_read_float(this->bias, j)) :
            (output_temp * parameter_read_float(this->weight_scale, scale_index)))
        );
    }
    return output;
}


Linear_SQ::Linear_SQ(uint16_t output_size, uint16_t input_size, const int8_t* weight, const int32_t* bias,
          float output_scale, int8_t output_zero_point, int8_t input_zero_point,  float* bias_scale, uint8_t quantize_property) {

    this->output_size = output_size;
    this->input_size = input_size;

    this->weight = weight;
    this->bias = bias;

    this->output_scale = output_scale;
    this->output_zero_point = output_zero_point;
    this->input_zero_point = input_zero_point;
    
    this->bias_scale = bias_scale;

    this->quantize_property = quantize_property;
}

/**
 * @brief Forward pass for statically quantized Linear layer
 * @param input Pointer to quantized input tensor (int8_t)
 * @param output Pointer to quantized output tensor (int8_t)
 * 
 * Computes quantized output with proper scaling and zero-point adjustments
 */
int8_t* Linear_SQ::forward(int8_t* input, int8_t* workspace_start, uint32_t workspace_size) {
    // Getting the output start address with the input size as offset
    int8_t* output = input == workspace_start ? workspace_start + workspace_size - (uint32_t)ceil((float)this->output_size / get_activation_data_per_byte(this->quantize_property)) : workspace_start;

    int32_t output_temp;

    void (*activation_write_packed_intb) (int8_t*, uint32_t, int8_t);
    int8_t (*activation_read_packed_intb) (int8_t*, uint32_t);
    int8_t (*parameter_read_packed_intb) (const int8_t*, uint32_t);
    int8_t (*clamp_intb) (int32_t);
        
    get_activation_write_packed_intb(this->quantize_property, &activation_write_packed_intb);
    get_activation_read_packed_intb(this->quantize_property, &activation_read_packed_intb);
    get_parameter_read_packed_intb(this->quantize_property, &parameter_read_packed_intb);
    get_activation_clamp_intb(this->quantize_property, &clamp_intb);

    for (uint16_t j = 0; j < this->output_size; j++) {
        output_temp = this->bias ? parameter_read_int32(this->bias, j) : 0;

        for (uint16_t i = 0; i < this->input_size; i++) {
            output_temp += ((int32_t)activation_read_packed_intb(input, i) - this->input_zero_point) *
                            parameter_read_packed_intb(this->weight, (j * this->input_size) + i);
        }
        uint8_t scale_index = get_granularity(this->quantize_property) == PER_CHANNEL ? j : 0;
        
        // Requantize to 8-bit
        output_temp = roundf(output_temp * parameter_read_float(this->bias_scale, scale_index)/ this->output_scale);
        output_temp += this->output_zero_point;
        output_temp = clamp_intb(output_temp);
        
        activation_write_packed_intb(output, j, output_temp);
    }
    return output;
}
