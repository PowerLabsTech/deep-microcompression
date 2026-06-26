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
#include "padding.h"
#include <string.h>

// Padding type constants
#define PADDING_VALID 0
#define PADDING_SAME  1

Conv2d::Conv2d(uint16_t input_channel_size, uint16_t input_row_size, uint16_t input_col_size,
               uint16_t output_channel_size, uint8_t kernel_row_size, uint8_t kernel_col_size,
               uint8_t stride_row, uint8_t stride_col, uint8_t padding_row, uint8_t padding_col,
               uint8_t groups, const float* weight, const float* bias) {

    this->input_channel_size = input_channel_size;
    this->input_row_size = input_row_size;
    this->input_col_size = input_col_size;
    this->output_channel_size = output_channel_size;
    this->kernel_row_size = kernel_row_size;
    this->kernel_col_size = kernel_col_size;
    this->padding_row = padding_row;
    this->padding_col = padding_col;
    this->stride_row = stride_row;
    this->stride_col = stride_col;
    this->groups = groups;
    this->weight = weight;
    this->bias = bias;

    this->output_row_size = ((this->input_row_size + 2*this->padding_row - this->kernel_row_size) / this->stride_row) + 1;
    this->output_col_size = ((this->input_col_size + 2*this->padding_col - this->kernel_col_size) / this->stride_col) + 1;
}

/**
 * @brief Forward pass for floating-point Conv2d
 * @param input Input tensor (float)
 * @param output Output tensor (float)
 */
void Conv2d::forward(float* workspace_start, uint32_t workspace_size) {

    uint16_t input_channel_per_group  = this->input_channel_size / this->groups;
    uint16_t output_channel_per_group = this->output_channel_size / this->groups;

    uint16_t padded_row = this->input_row_size + 2 * this->padding_row;
    uint16_t padded_col = this->input_col_size + 2 * this->padding_col;

    // Last output_channel_per_group floats in workspace reserved as accumulator scratch
    float* pixel = workspace_start + workspace_size - output_channel_per_group;

    float* input;
    if (this->output_channel_size > this->input_channel_size) {
        input = workspace_start + workspace_size - output_channel_per_group
                                 - ((uint32_t)padded_row * padded_col * this->input_channel_size);
        if (this->padding_row + this->padding_col) {
            Padding_t pad = {this->padding_col, this->padding_col, this->padding_row, this->padding_row};
            constantPad2d(workspace_start, input,
                          this->input_channel_size, this->input_row_size, this->input_col_size, pad, 0.0f);
        } else {
            memmove(input, workspace_start,
                (uint32_t)padded_row * padded_col * this->input_channel_size * sizeof(float));
        }
    } else {
        input = workspace_start;
        if (this->padding_row + this->padding_col) {
            Padding_t pad = {this->padding_col, this->padding_col, this->padding_row, this->padding_row};
            constantPad2d(workspace_start, workspace_start,
                          this->input_channel_size, this->input_row_size, this->input_col_size,
                          pad, 0.0f);
        }
    }

    uint16_t n, k;

    for (uint8_t g = 0; g < this->groups; g++) {
        for (uint16_t m = 0; m < this->output_row_size; m++) {
            for (uint16_t l = 0; l < this->output_col_size; l++) {
                for (uint16_t c_out = 0; c_out < output_channel_per_group; c_out++) {
                    n = g * output_channel_per_group + c_out;
                    pixel[c_out] = this->bias ? parameter_read_float(this->bias, n) : 0;

                    for (uint16_t c_in = 0; c_in < input_channel_per_group; c_in++) {
                        k = g * input_channel_per_group + c_in;
                        for (uint8_t j = 0; j < this->kernel_row_size; j++) {
                            for (uint8_t i = 0; i < this->kernel_col_size; i++) {
                                pixel[c_out] += activation_read_float(
                                    input,
                                    ((j + m * this->stride_row) * padded_col * this->input_channel_size) +
                                    ((i + l * this->stride_col) * this->input_channel_size) +
                                    k
                                ) * parameter_read_float(
                                    this->weight,
                                    (n * this->kernel_row_size * this->kernel_col_size * input_channel_per_group) +
                                    (j * this->kernel_col_size * input_channel_per_group) +
                                    (i * input_channel_per_group) +
                                    c_in
                                );
                            }
                        }
                    }
                }

                for (uint16_t c_out = 0; c_out < output_channel_per_group; c_out++) {
                    n = g * output_channel_per_group + c_out;
                    activation_write_float(workspace_start,
                        (m * this->output_col_size * this->output_channel_size) +
                        (l * this->output_channel_size) +
                        n,
                        pixel[c_out]
                    );
                }
            }
        }
    }
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
               uint8_t stride_row, uint8_t stride_col, uint8_t padding_row, uint8_t padding_col,
               uint8_t groups, const int8_t* weight, const float* bias,
               const float* weight_scale, uint8_t quantize_property) {

    this->input_channel_size = input_channel_size;
    this->input_row_size = input_row_size;
    this->input_col_size = input_col_size;
    this->output_channel_size = output_channel_size;
    this->kernel_row_size = kernel_row_size;
    this->kernel_col_size = kernel_col_size;
    this->padding_row = padding_row;
    this->padding_col = padding_col;
    this->stride_row = stride_row;
    this->stride_col = stride_col;
    this->groups = groups;
    this->weight = weight;
    this->bias = bias;
    this->weight_scale = weight_scale;
    this->quantize_property = quantize_property;

    this->output_row_size = ((this->input_row_size + 2*this->padding_row - this->kernel_row_size) / this->stride_row) + 1;
    this->output_col_size = ((this->input_col_size + 2*this->padding_col - this->kernel_col_size) / this->stride_col) + 1;
}


void Conv2d_DQ::forward(float* workspace_start, uint32_t workspace_size) {

    uint16_t input_channel_per_group  = this->input_channel_size / this->groups;
    uint16_t output_channel_per_group = this->output_channel_size / this->groups;

    uint16_t padded_row = this->input_row_size + 2 * this->padding_row;
    uint16_t padded_col = this->input_col_size + 2 * this->padding_col;

    float* pixel = workspace_start + workspace_size - output_channel_per_group;

    float* input;
    if (this->output_channel_size > this->input_channel_size) {
        input = workspace_start + workspace_size - output_channel_per_group
                                 - ((uint32_t)padded_row * padded_col * this->input_channel_size);
        if (this->padding_row + this->padding_col) {
            Padding_t pad = {this->padding_col, this->padding_col, this->padding_row, this->padding_row};
            constantPad2d(workspace_start, input,
                          this->input_channel_size, this->input_row_size, this->input_col_size, pad, 0.0f);
        } else {
            memmove(input, workspace_start,
                (uint32_t)padded_row * padded_col * this->input_channel_size * sizeof(float));
        }
    } else {
        input = workspace_start;
        if (this->padding_row + this->padding_col) {
            Padding_t pad = {this->padding_col, this->padding_col, this->padding_row, this->padding_row};
            constantPad2d(workspace_start, workspace_start,
                          this->input_channel_size, this->input_row_size, this->input_col_size, pad, 0.0f);
        }
    }

    int8_t (*parameter_read_packed_intb) (const int8_t*, uint32_t);
    get_parameter_read_packed_intb(this->quantize_property, &parameter_read_packed_intb);

    uint16_t n, k;

    for (uint8_t g = 0; g < this->groups; g++) {
        for (uint16_t m = 0; m < this->output_row_size; m++) {
            for (uint16_t l = 0; l < this->output_col_size; l++) {

                for (uint16_t c_out = 0; c_out < output_channel_per_group; c_out++) {
                    pixel[c_out] = 0.0f;
                }

                for (uint16_t c_out = 0; c_out < output_channel_per_group; c_out++) {
                    n = g * output_channel_per_group + c_out;
                    for (uint16_t c_in = 0; c_in < input_channel_per_group; c_in++) {
                        k = g * input_channel_per_group + c_in;
                        for (uint8_t j = 0; j < this->kernel_row_size; j++) {
                            for (uint8_t i = 0; i < this->kernel_col_size; i++) {
                                pixel[c_out] += activation_read_float(
                                    input,
                                    ((j + m * this->stride_row) * padded_col * this->input_channel_size) +
                                    ((i + l * this->stride_col) * this->input_channel_size) +
                                    k
                                ) * parameter_read_packed_intb(
                                    this->weight,
                                    (n * this->kernel_row_size * this->kernel_col_size * input_channel_per_group) +
                                    (j * this->kernel_col_size * input_channel_per_group) +
                                    (i * input_channel_per_group) +
                                    c_in
                                );
                            }
                        }
                    }
                }

                for (uint16_t c_out = 0; c_out < output_channel_per_group; c_out++) {
                    n = g * output_channel_per_group + c_out;
                    uint8_t scale_index = get_granularity(this->quantize_property) == PER_CHANNEL ? n : 0;
                    activation_write_float(workspace_start,
                        (m * this->output_col_size * this->output_channel_size) +
                        (l * this->output_channel_size) + n,
                        this->bias ?
                        (pixel[c_out] * parameter_read_float(this->weight_scale, scale_index) + parameter_read_float(this->bias, n)) :
                        (pixel[c_out] * parameter_read_float(this->weight_scale, scale_index))
                    );
                }
            }
        }
    }
}


uint32_t Conv2d_DQ::get_output_size(void) {
    return (this->output_channel_size * this->output_row_size * this->output_col_size);
}

Conv2d_SQ::Conv2d_SQ(uint16_t input_channel_size, uint16_t input_row_size, uint16_t input_col_size,
               uint16_t output_channel_size, uint8_t kernel_row_size, uint8_t kernel_col_size,
               uint8_t stride_row, uint8_t stride_col, uint8_t padding_row, uint8_t padding_col,
               uint8_t groups, const int8_t* weight, const int32_t* bias, float output_scale,
               int8_t output_zero_point, int8_t input_zero_point, float* bias_scale, uint8_t quantize_property) {

    this->input_channel_size = input_channel_size;
    this->input_row_size = input_row_size;
    this->input_col_size = input_col_size;
    this->output_channel_size = output_channel_size;
    this->kernel_row_size = kernel_row_size;
    this->kernel_col_size = kernel_col_size;
    this->padding_row = padding_row;
    this->padding_col = padding_col;
    this->stride_row = stride_row;
    this->stride_col = stride_col;
    this->groups = groups;
    this->weight = weight;
    this->bias = bias;
    this->output_scale = output_scale;
    this->output_zero_point = output_zero_point;
    this->input_zero_point = input_zero_point;
    this->bias_scale = bias_scale;
    this->quantize_property = quantize_property;

    this->output_row_size = ((this->input_row_size + 2*this->padding_row - this->kernel_row_size) / this->stride_row) + 1;
    this->output_col_size = ((this->input_col_size + 2*this->padding_col - this->kernel_col_size) / this->stride_col) + 1;
}

void Conv2d_SQ::forward(int8_t* workspace_start, uint32_t workspace_size) {

    uint16_t input_channel_per_group  = this->input_channel_size / this->groups;
    uint16_t output_channel_per_group = this->output_channel_size / this->groups;

    uint16_t padded_row = this->input_row_size + 2 * this->padding_row;
    uint16_t padded_col = this->input_col_size + 2 * this->padding_col;

    uint32_t data_per_byte = get_activation_data_per_byte(this->quantize_property);
    uint32_t padded_input_bytes = (uint32_t)ceil(
        (float)((uint32_t)padded_row * padded_col * this->input_channel_size) / data_per_byte
    );

    // Last output_channel_per_group int32 values in workspace reserved as accumulator scratch
    int32_t* pixel = (int32_t*)(workspace_start + workspace_size) - output_channel_per_group;

    int8_t* input;
    if (this->output_channel_size > this->input_channel_size) {
        input = workspace_start + workspace_size
                - (uint32_t)(output_channel_per_group * sizeof(int32_t))
                - padded_input_bytes;
        if (this->padding_row + this->padding_col) {
            Padding_t pad = {this->padding_col, this->padding_col, this->padding_row, this->padding_row};
            constantPad2d_SQ(workspace_start, input,
                             this->input_channel_size, this->input_row_size, this->input_col_size,
                             pad, this->input_zero_point, this->quantize_property);
        } else {
            memmove(input, workspace_start, padded_input_bytes);
        }
    } else {
        input = workspace_start;
        if (this->padding_row + this->padding_col) {
            Padding_t pad = {this->padding_col, this->padding_col, this->padding_row, this->padding_row};
            constantPad2d_SQ(workspace_start, workspace_start,
                             this->input_channel_size, this->input_row_size, this->input_col_size,
                             pad, this->input_zero_point, this->quantize_property);
        }
    }

    void (*activation_write_packed_intb) (int8_t*, uint32_t, int8_t);
    int8_t (*activation_read_packed_intb) (int8_t*, uint32_t);
    int8_t (*parameter_read_packed_intb) (const int8_t*, uint32_t);
    int8_t (*clamp_intb) (int32_t);

    get_activation_write_packed_intb(this->quantize_property, &activation_write_packed_intb);
    get_activation_read_packed_intb(this->quantize_property, &activation_read_packed_intb);
    get_parameter_read_packed_intb(this->quantize_property, &parameter_read_packed_intb);
    get_activation_clamp_intb(this->quantize_property, &clamp_intb);

    uint16_t n, k;

    for (uint8_t g = 0; g < this->groups; g++) {
        for (uint16_t m = 0; m < this->output_row_size; m++) {
            for (uint16_t l = 0; l < this->output_col_size; l++) {

                for (uint16_t c_out = 0; c_out < output_channel_per_group; c_out++) {
                    n = g * output_channel_per_group + c_out;
                    pixel[c_out] = this->bias ? parameter_read_int32(this->bias, n) : 0;
                }

                for (uint16_t c_out = 0; c_out < output_channel_per_group; c_out++) {
                    n = g * output_channel_per_group + c_out;
                    for (uint16_t c_in = 0; c_in < input_channel_per_group; c_in++) {
                        k = g * input_channel_per_group + c_in;
                        for (uint8_t j = 0; j < this->kernel_row_size; j++) {
                            for (uint8_t i = 0; i < this->kernel_col_size; i++) {
                                pixel[c_out] += ((int32_t)activation_read_packed_intb(
                                    input,
                                    ((j + m * this->stride_row) * padded_col * this->input_channel_size) +
                                    ((i + l * this->stride_col) * this->input_channel_size) +
                                    k) - this->input_zero_point) *
                                    parameter_read_packed_intb(
                                        this->weight,
                                        (n * this->kernel_row_size * this->kernel_col_size * input_channel_per_group) +
                                        (j * this->kernel_col_size * input_channel_per_group) +
                                        (i * input_channel_per_group) +
                                        c_in
                                    );
                            }
                        }
                    }
                }

                for (uint16_t c_out = 0; c_out < output_channel_per_group; c_out++) {
                    n = g * output_channel_per_group + c_out;
                    uint8_t scale_index = get_granularity(this->quantize_property) == PER_CHANNEL ? n : 0;
                    int32_t val = roundf(pixel[c_out] * parameter_read_float(this->bias_scale, scale_index) / this->output_scale);
                    val += this->output_zero_point;
                    activation_write_packed_intb(workspace_start,
                        (m * this->output_col_size * this->output_channel_size) +
                        (l * this->output_channel_size) + n,
                        clamp_intb(val)
                    );
                }
            }
        }
    }
}


uint32_t Conv2d_SQ::get_output_size(void) {
    return (this->output_channel_size * this->output_row_size * this->output_col_size);
}
