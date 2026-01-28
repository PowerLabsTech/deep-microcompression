/**
 * @file conv.cpp
 * @brief Implementation of 2D convolution layer with support for:
 *      1. None quantized model
 *      2. Dynamic quantized model per tensor
 *          - 8 bit
 *          - 4 bit
 *      3. Static quantized model per tensor
 *          - 8 bit
 *          - 4 bit
 * 
 * Supports 4-bit and 8-bit weight packing for quantized modes.
 */

#include "conv.h"

// Padding type constants
#define PADDING_VALID 0
#define PADDING_SAME  1

Conv2d::Conv2d(uint16_t input_channel_size, uint16_t input_row_size, uint16_t input_col_size,
               uint16_t output_channel_size, uint8_t kernel_row_size, uint8_t kernel_col_size,
               uint8_t stride_row, uint8_t stride_col, uint8_t groups,
               const float* weight, const float* bias) {
    
    // Store layer parameters
    this->input_channel_size = input_channel_size;
    this->input_row_size = input_row_size;
    this->input_col_size = input_col_size;
    this->output_channel_size = output_channel_size;
    this->kernel_row_size = kernel_row_size;
    this->kernel_col_size = kernel_col_size;
    
    this->stride_row = stride_row;
    this->stride_col = stride_col;
    this->groups = groups;
    
    this->weight = weight;
    this->bias = bias;

    // Compute output dimensions
    this->output_row_size = ((this->input_row_size - this->kernel_row_size) / this->stride_row) + 1;
    this->output_col_size = ((this->input_col_size - this->kernel_col_size) / this->stride_col) + 1;
}

/**
 * @brief Forward pass for floating-point Conv2d
 * @param input Input tensor (float)
 * @param output Output tensor (float)
 */
float* Conv2d::forward(float* input, float* workspace_start, uint32_t workspace_size) {
    // Getting the output start address with the input size as offset
    float* output = input == workspace_start ? workspace_start + workspace_size - this->get_output_size() : workspace_start;

    float output_temp;

    // Handle Grouped Convolution (e.g., Depthwise)
    uint16_t input_channel_per_group = this->input_channel_size / this->groups;
    uint16_t output_channel_per_group = this->output_channel_size / this->groups;

    uint16_t n, k; // Indices for Output/Input channels

    for (uint8_t g = 0; g < this->groups; g++){
        // Loop Output Channels (Filters)
        for (uint16_t c_out = 0; c_out < output_channel_per_group; c_out++) {
            n = g * output_channel_per_group + c_out;

            // Output spatial dimensions loops
            for (uint16_t m = 0; m < this->output_row_size; m++) {
                for (uint16_t l = 0; l < this->output_col_size; l++) {
                    output_temp = this->bias ? parameter_read_float(this->bias, n) : 0;

                    for (uint16_t c_in = 0; c_in < input_channel_per_group; c_in++) {
                        k = g * input_channel_per_group + c_in;
                        for (uint8_t j = 0; j < this->kernel_row_size; j++) {
                            for (uint8_t i = 0; i < this->kernel_col_size; i++) {

                                // Standard MAC Operation
                                output_temp += activation_read_float(
                                    input, 
                                    (k * this->input_row_size * this->input_col_size) +
                                    ((j + m * this->stride_row) * this->input_col_size) + 
                                    (i + l * this->stride_col)
                                ) * parameter_read_float(
                                    this->weight, 
                                    (n * input_channel_per_group * this->kernel_row_size * this->kernel_col_size) +
                                    (c_in * this->kernel_row_size * kernel_col_size) + 
                                    (j * this->kernel_col_size) + 
                                    i
                                );
                            }
                        }
                    }
                    activation_write_float(output, 
                        (n * this->output_row_size * this->output_col_size) + 
                        (m * this->output_col_size) + 
                        l,
                        output_temp
                    );
                }
            }
        }
    }
    return output;
}


uint32_t Conv2d::get_output_size(void) {
    return (this->output_channel_size * this->output_row_size * this->output_col_size);
}

/**
 * @brief Constructor for dynamically quantized Conv2d layer
 * @param weight_scale Scale factor for quantized weights
 */
Conv2d_DQ::Conv2d_DQ(uint16_t input_channel_size, uint16_t input_row_size, uint16_t input_col_size,
               uint16_t output_channel_size, uint8_t kernel_row_size, uint8_t kernel_col_size,
               uint8_t stride_row, uint8_t stride_col, uint8_t groups,
               const int8_t* weight, const float* bias, const float* weight_scale, uint8_t quantize_property) {

    // Store layer parameters
    this->input_channel_size = input_channel_size;
    this->input_row_size = input_row_size;
    this->input_col_size = input_col_size;
    this->output_channel_size = output_channel_size;
    this->kernel_row_size = kernel_row_size;
    this->kernel_col_size = kernel_col_size;
    
    this->stride_row = stride_row;
    this->stride_col = stride_col;
    this->groups = groups;
    
    this->weight = weight;
    this->bias = bias;
    this->weight_scale = weight_scale;

    // Compute output dimensions
    this->output_row_size = ((this->input_row_size - this->kernel_row_size) / this->stride_row) + 1;
    this->output_col_size = ((this->input_col_size - this->kernel_col_size) / this->stride_col) + 1;

    this->quantize_property = quantize_property;
}


/**
 * @brief Forward pass for floating-point Conv2d
 * @param input Input tensor (float)
 * @param output Output tensor (float)
 */
float* Conv2d_DQ::forward(float* input, float* workspace_start, uint32_t workspace_size) {
    // Getting the output start address with the input size as offset
    float* output = input == workspace_start ? workspace_start + workspace_size - this->get_output_size() : workspace_start;

    int8_t (*parameter_read_packed_intb) (const int8_t*, uint32_t);
    
    get_parameter_read_packed_intb(this->quantize_property, &parameter_read_packed_intb);

    // uint32_t output_index;
    float output_temp;

    uint16_t input_channel_per_group = this->input_channel_size / this->groups;
    uint16_t output_channel_per_group = this->output_channel_size / this->groups;

    uint16_t n, k;

    for (uint8_t g = 0; g < this->groups; g++){
        // Output channel loop
        for (uint16_t c_out = 0; c_out < output_channel_per_group; c_out++) {
            n = g * output_channel_per_group + c_out;
            // Output spatial dimensions loops
            for (uint16_t m = 0; m < this->output_row_size; m++) {
                for (uint16_t l = 0; l < this->output_col_size; l++) {
                    
                    output_temp = 0;
                    for (uint16_t c_in = 0; c_in < input_channel_per_group; c_in++) {
                        k = g * input_channel_per_group + c_in;
                        for (uint8_t j = 0; j < this->kernel_row_size; j++) {
                            for (uint8_t i = 0; i < this->kernel_col_size; i++) {

                                // // Convolution operation
                                output_temp += activation_read_float(
                                    input, 
                                    (k * this->input_row_size * this->input_col_size) +
                                    ((j + m * this->stride_row) * this->input_col_size) + 
                                    (i + l * this->stride_col)
                                ) * 
                                parameter_read_packed_intb(
                                    this->weight, 
                                    (n * input_channel_per_group * this->kernel_row_size * this->kernel_col_size) +
                                    (c_in * this->kernel_row_size * kernel_col_size) + 
                                    (j * this->kernel_col_size) + i
                                );
                            }
                        }
                    }
                    uint8_t scale_index = get_granularity(this->quantize_property) == PER_CHANNEL ? n : 0;

                    activation_write_float(output, 
                        (n * this->output_row_size * this->output_col_size) + 
                        (m * this->output_col_size) + 
                        l,
                        (this->bias ? 
                        (output_temp * parameter_read_float(this->weight_scale, scale_index) + parameter_read_float(this->bias, n)) :
                        (output_temp * parameter_read_float(this->weight_scale, scale_index)))
                    );
                }
            }
        }
    }
    return output;
}


uint32_t Conv2d_DQ::get_output_size(void) {
    return (this->output_channel_size * this->output_row_size * this->output_col_size);
}

Conv2d_SQ::Conv2d_SQ(uint16_t input_channel_size, uint16_t input_row_size, uint16_t input_col_size,
               uint16_t output_channel_size, uint8_t kernel_row_size, uint8_t kernel_col_size,
               uint8_t stride_row, uint8_t stride_col, uint8_t groups,
               const int8_t* weight, const int32_t* bias, float output_scale, 
               int8_t output_zero_point, int8_t input_zero_point,  float* bias_scale, uint8_t quantize_property) {
                
    // Store layer parameters
    this->input_channel_size = input_channel_size;
    this->input_row_size = input_row_size;
    this->input_col_size = input_col_size;
    this->output_channel_size = output_channel_size;
    this->kernel_row_size = kernel_row_size;
    this->kernel_col_size = kernel_col_size;

    this->stride_row = stride_row;
    this->stride_col = stride_col;
    this->groups = groups;

    this->weight = weight;
    this->bias = bias;

    this->output_scale = output_scale;
    this->output_zero_point = output_zero_point;
    this->input_zero_point = input_zero_point;

    this->bias_scale = bias_scale;

    this->output_row_size = ((this->input_row_size - this->kernel_row_size) / this->stride_row) + 1;
    this->output_col_size = ((this->input_col_size - this->kernel_col_size) / this->stride_col) + 1;

    this->quantize_property = quantize_property;
}

/**
 * @brief Forward pass for statically quantized Conv2d
 * @param input Input tensor (int8_t)
 * @param output Output tensor (int8_t)
 */
int8_t* Conv2d_SQ::forward(int8_t* input, int8_t* workspace_start, uint32_t workspace_size) {
    // Getting the output start address with the input size as offset
    int8_t* output = input == workspace_start ? workspace_start + workspace_size - (uint32_t)ceil(
        (float)this->get_output_size() / get_activation_data_per_byte(this->quantize_property)
    ) : workspace_start;

    void (*activation_write_packed_intb) (int8_t*, uint32_t, int8_t);
    int8_t (*activation_read_packed_intb) (int8_t*, uint32_t);
    int8_t (*parameter_read_packed_intb) (const int8_t*, uint32_t);
    int8_t (*clamp_intb) (int32_t);
        
    get_activation_write_packed_intb(this->quantize_property, &activation_write_packed_intb);
    get_activation_read_packed_intb(this->quantize_property, &activation_read_packed_intb);
    get_parameter_read_packed_intb(this->quantize_property, &parameter_read_packed_intb);
    get_activation_clamp_intb(this->quantize_property, &clamp_intb);

    uint16_t input_channel_per_group = this->input_channel_size / this->groups;
    uint16_t output_channel_per_group = this->output_channel_size / this->groups;

    int32_t output_temp;
    uint16_t n, k;

    for (uint8_t g = 0; g < this->groups; g++){
        // Output channel loop
        for (uint16_t c_out = 0; c_out < output_channel_per_group; c_out++) {
            n = g * output_channel_per_group + c_out;
            // Output spatial dimensions loops
            for (uint16_t m = 0; m < this->output_row_size; m++) {
                for (uint16_t l = 0; l < this->output_col_size; l++) {

                    output_temp = this->bias ? parameter_read_int32(this->bias, n) : 0;

                    for (uint16_t c_in = 0; c_in < input_channel_per_group; c_in++) {
                        k = g * input_channel_per_group + c_in;
                        for (uint8_t j = 0; j < this->kernel_row_size; j++) {
                            for (uint8_t i = 0; i < this->kernel_col_size; i++) {
                                // Convolution operation

                                output_temp += ((int32_t)activation_read_packed_intb(
                                    input,
                                    (k * this->input_row_size * this->input_col_size) +
                                    ((j + m * this->stride_row) * this->input_col_size) + 
                                    (i + l * this->stride_col)) - this->input_zero_point) *
                                    parameter_read_packed_intb(
                                        this->weight,
                                        (n * input_channel_per_group * this->kernel_row_size * this->kernel_col_size) +
                                        (c_in * this->kernel_row_size * kernel_col_size) + 
                                        (j * this->kernel_col_size) + i
                                    );
                            }
                        }
                    }
                    uint8_t scale_index = get_granularity(this->quantize_property) == PER_CHANNEL ? n : 0;

                    // Apply bias, scaling and clamping
                    output_temp = roundf(output_temp * parameter_read_float(this->bias_scale, scale_index) / this->output_scale);
                    output_temp += this->output_zero_point;
                    output_temp = clamp_intb(output_temp);
                    
                    activation_write_packed_intb(
                        output,
                        (n * this->output_row_size * this->output_col_size) + 
                        (m * this->output_col_size) + 
                        l, 
                        output_temp
                    );
                }
            }
        }
    }
    return output;
}


uint32_t Conv2d_SQ::get_output_size(void) {
    return (this->output_channel_size * this->output_row_size * this->output_col_size);
}
