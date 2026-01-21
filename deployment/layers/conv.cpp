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


#if !defined(QUANTIZATION_SCHEME) || QUANTIZATION_SCHEME == NONE


Conv2d::Conv2d(uint16_t input_channel_size, uint16_t input_row_size, uint16_t input_col_size,
               uint16_t output_channel_size, uint8_t kernel_row_size, uint8_t kernel_col_size,
               uint8_t stride_row, uint8_t stride_col, Padding_t padding, uint8_t groups,
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
    this->padding = padding;
    this->groups = groups;
    
    this->weight = weight;
    this->bias = bias;

    // Compute output dimensions
    this->output_row_size = ((this->input_row_size + this->padding.padding_top + this->padding.padding_bottom - this->kernel_row_size) / this->stride_row) + 1;
    this->output_col_size = ((this->input_col_size + this->padding.padding_left + this->padding.padding_right - this->kernel_col_size) / this->stride_col) + 1;
}

/**
 * @brief Forward pass for floating-point Conv2d
 * @param input Input tensor (float)
 * @param output Output tensor (float)
 */
void Conv2d::forward(float* input, float* output) {
    float output_temp;

    // Handle Grouped Convolution (e.g., Depthwise)
    uint16_t input_channel_per_group = this->input_channel_size / this->groups;
    uint16_t output_channel_per_group = this->output_channel_size / this->groups;

    uint16_t n, k; // Indices for Output/Input channels

    // Padding calculations
    uint16_t padded_row_size = this->input_row_size + this->padding.padding_top + this->padding.padding_bottom;
    uint16_t padded_col_size = this->input_col_size + this->padding.padding_left + this->padding.padding_right;

    pad_input(input, this->padding, input_channel_size, input_row_size, input_col_size, padded_row_size, padded_col_size);

    for (uint8_t g = 0; g < this->groups; g++){
        // Loop Output Channels (Filters)
        for (uint16_t c_out = 0; c_out < output_channel_per_group; c_out++) {
            n = g * output_channel_per_group + c_out;

            // Output spatial dimensions loops
            for (uint16_t m = 0; m < this->output_row_size; m++) {
                for (uint16_t l = 0; l < this->output_col_size; l++) {
                    output_temp = this->bias ? par_read_float(this->bias, n) : 0;

                    for (uint16_t c_in = 0; c_in < input_channel_per_group; c_in++) {
                        k = g * input_channel_per_group + c_in;
                        for (uint8_t j = 0; j < this->kernel_row_size; j++) {
                            for (uint8_t i = 0; i < this->kernel_col_size; i++) {

                                // Standard MAC Operation
                                output_temp += act_read_float(
                                    input, 
                                    (k * padded_row_size * padded_col_size) +
                                    ((j + m * this->stride_row) * padded_col_size) + 
                                    (i + l * this->stride_col)
                                ) * par_read_float(
                                    this->weight, 
                                    (n * input_channel_per_group * this->kernel_row_size * this->kernel_col_size) +
                                    (c_in * this->kernel_row_size * kernel_col_size) + 
                                    (j * this->kernel_col_size) + 
                                    i
                                );
                            }
                        }
                    }
                    act_write_float(output, 
                        (n * this->output_row_size * this->output_col_size) + 
                        (m * this->output_col_size) + 
                        l,
                        output_temp
                    );
                }
            }
        }
    }
}




#elif QUANTIZATION_SCHEME == DYNAMIC // QUANTIZATION_SCHEME
#if QUANTIZATION_GRANULARITY == PER_TENSOR


/**
 * @brief Constructor for dynamically quantized Conv2d layer
 * @param weight_scale Scale factor for quantized weights
 */
Conv2d::Conv2d(uint16_t input_channel_size, uint16_t input_row_size, uint16_t input_col_size,
               uint16_t output_channel_size, uint8_t kernel_row_size, uint8_t kernel_col_size,
               uint8_t stride_row, uint8_t stride_col, Padding_t padding, uint8_t groups,
               const int8_t* weight, const float* bias, float weight_scale) {

    // Store layer parameters
    this->input_channel_size = input_channel_size;
    this->input_row_size = input_row_size;
    this->input_col_size = input_col_size;
    this->output_channel_size = output_channel_size;
    this->kernel_row_size = kernel_row_size;
    this->kernel_col_size = kernel_col_size;
    
    this->stride_row = stride_row;
    this->stride_col = stride_col;
    this->padding = padding;
    this->groups = groups;
    
    this->weight = weight;
    this->bias = bias;
    this->weight_scale = weight_scale;

    // Compute output dimensions
    this->output_row_size = ((this->input_row_size - this->kernel_row_size) / this->stride_row) + 1;
    this->output_col_size = ((this->input_col_size - this->kernel_col_size) / this->stride_col) + 1;


    // Compute output dimensions
    this->output_row_size = ((this->input_row_size + this->padding.padding_top + this->padding.padding_bottom - this->kernel_row_size) / this->stride_row) + 1;
    this->output_col_size = ((this->input_col_size + this->padding.padding_left + this->padding.padding_right - this->kernel_col_size) / this->stride_col) + 1;
}


/**
 * @brief Forward pass for floating-point Conv2d
 * @param input Input tensor (float)
 * @param output Output tensor (float)
 */
void Conv2d::forward(float* input, float* output) {
    // uint32_t output_index;
    float output_temp;

    uint16_t input_channel_per_group = this->input_channel_size / this->groups;
    uint16_t output_channel_per_group = this->output_channel_size / this->groups;

    uint16_t n, k;

    uint16_t padded_row_size = this->input_row_size + this->padding.padding_top + this->padding.padding_bottom;
    uint16_t padded_col_size = this->input_col_size + this->padding.padding_left + this->padding.padding_right;

    pad_input(input, this->padding, input_channel_size, input_row_size, input_col_size, padded_row_size, padded_col_size);

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
                                output_temp += act_read_float(
                                    input, 
                                    (k * padded_row_size * padded_col_size) +
                                    ((j + m * this->stride_row) * padded_col_size) + 
                                    (i + l * this->stride_col)
                                ) * 
                                par_read_packed_intb(
                                    this->weight, 
                                    (n * input_channel_per_group * this->kernel_row_size * this->kernel_col_size) +
                                    (c_in * this->kernel_row_size * kernel_col_size) + 
                                    (j * this->kernel_col_size) + i
                                );
                            }
                        }
                    }
                    act_write_float(output, 
                        (n * this->output_row_size * this->output_col_size) + 
                        (m * this->output_col_size) + 
                        l,
                        (this->bias ? 
                        (output_temp * this->weight_scale + par_read_float(this->bias, n)) :
                        (output_temp * this->weight_scale))
                    );
                }
            }
        }
    }
}

#endif // QUANTIZATION_GRANULARITY


#elif QUANTIZATION_SCHEME == STATIC // QUANTIZATION_SCHEME

#if QUANTIZATION_GRANULARITY == PER_TENSOR

Conv2d::Conv2d(uint16_t input_channel_size, uint16_t input_row_size, uint16_t input_col_size,
               uint16_t output_channel_size, uint8_t kernel_row_size, uint8_t kernel_col_size,
               uint8_t stride_row, uint8_t stride_col, Padding_t padding, uint8_t groups,
               const int8_t* weight, const int32_t* bias, float output_scale, 
               int8_t output_zero_point, int8_t input_zero_point,  float bias_scale) {
                
    // Store layer parameters
    this->input_channel_size = input_channel_size;
    this->input_row_size = input_row_size;
    this->input_col_size = input_col_size;
    this->output_channel_size = output_channel_size;
    this->kernel_row_size = kernel_row_size;
    this->kernel_col_size = kernel_col_size;

    this->stride_row = stride_row;
    this->stride_col = stride_col;
    this->padding = padding;
    this->groups = groups;

    this->weight = weight;
    this->bias = bias;

    this->output_scale = output_scale;
    this->output_zero_point = output_zero_point;
    this->input_zero_point = input_zero_point;

    this->bias_scale = bias_scale;

    this->output_row_size = ((this->input_row_size + this->padding.padding_top + this->padding.padding_bottom - this->kernel_row_size) / this->stride_row) + 1;
    this->output_col_size = ((this->input_col_size + this->padding.padding_left + this->padding.padding_right - this->kernel_col_size) / this->stride_col) + 1;

}

/**
 * @brief Forward pass for statically quantized Conv2d
 * @param input Input tensor (int8_t)
 * @param output Output tensor (int8_t)
 */
void Conv2d::forward(int8_t* input, int8_t* output) {

    uint16_t input_channel_per_group = this->input_channel_size / this->groups;
    uint16_t output_channel_per_group = this->output_channel_size / this->groups;

    int32_t output_temp;
    uint16_t n, k;

    uint16_t padded_row_size = this->input_row_size + this->padding.padding_top + this->padding.padding_bottom;
    uint16_t padded_col_size = this->input_col_size + this->padding.padding_left + this->padding.padding_right;

    pad_input(input, this->input_zero_point, this->padding, input_channel_size, input_row_size, input_col_size, padded_row_size, padded_col_size);

    for (uint8_t g = 0; g < this->groups; g++){
        // Output channel loop
        for (uint16_t c_out = 0; c_out < output_channel_per_group; c_out++) {
            n = g * output_channel_per_group + c_out;
            // Output spatial dimensions loops
            for (uint16_t m = 0; m < this->output_row_size; m++) {
                for (uint16_t l = 0; l < this->output_col_size; l++) {

                    output_temp = this->bias ? par_read_int32(this->bias, n) : 0;

                    for (uint16_t c_in = 0; c_in < input_channel_per_group; c_in++) {
                        k = g * input_channel_per_group + c_in;
                        for (uint8_t j = 0; j < this->kernel_row_size; j++) {
                            for (uint8_t i = 0; i < this->kernel_col_size; i++) {
                                // Convolution operation

                                output_temp += ((int32_t)act_read_packed_intb(
                                    input,
                                    (k * padded_row_size * padded_col_size) +
                                    ((j + m * this->stride_row) * padded_col_size) + 
                                    (i + l * this->stride_col)) - this->input_zero_point) *
                                    par_read_packed_intb(
                                        this->weight,
                                        (n * input_channel_per_group * this->kernel_row_size * this->kernel_col_size) +
                                        (c_in * this->kernel_row_size * kernel_col_size) + 
                                        (j * this->kernel_col_size) + i
                                    );
                            }
                        }
                    }
                    // Apply bias, scaling and clamping
                    output_temp = roundf(output_temp * this->bias_scale / this->output_scale);
                    output_temp += this->output_zero_point;
                    output_temp = clampb(output_temp);
                    
                    act_write_packed_intb(
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
}
#endif // QUANTIZATION_GRANULARITY


#endif // QUANTIZATION_SCHEME
