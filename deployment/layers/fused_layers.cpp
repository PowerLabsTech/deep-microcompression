#include "fused_layers.h"
#include "padding.h"
#include <string.h>


// ============================================================================
// Float variants
// ============================================================================

void LinearReLU::forward(float* workspace_start, uint32_t workspace_size) {
    float* input;
    float* output;

    if (this->input_size > this->output_size) {
        input  = workspace_start;
        output = workspace_start + workspace_size - this->output_size;
    } else {
        input  = workspace_start + workspace_size - this->input_size;
        output = workspace_start;
        memcpy(input, workspace_start, this->input_size * sizeof(float));
    }

    for (uint16_t j = 0; j < this->output_size; j++) {
        float output_temp = this->bias ? parameter_read_float(this->bias, j) : 0.0f;
        for (uint16_t i = 0; i < this->input_size; i++) {
            output_temp += activation_read_float(input, i) * parameter_read_float(this->weight, (j * this->input_size) + i);
        }
        activation_write_float(output, j, relu(output_temp));
    }

    if (this->input_size > this->output_size)
        memcpy(workspace_start, output, this->output_size * sizeof(float));
}

void LinearReLU6::forward(float* workspace_start, uint32_t workspace_size) {
    float* input;
    float* output;

    if (this->input_size > this->output_size) {
        input  = workspace_start;
        output = workspace_start + workspace_size - this->output_size;
    } else {
        input  = workspace_start + workspace_size - this->input_size;
        output = workspace_start;
        memcpy(input, workspace_start, this->input_size * sizeof(float));
    }

    for (uint16_t j = 0; j < this->output_size; j++) {
        float output_temp = this->bias ? parameter_read_float(this->bias, j) : 0.0f;
        for (uint16_t i = 0; i < this->input_size; i++) {
            output_temp += activation_read_float(input, i) * parameter_read_float(this->weight, (j * this->input_size) + i);
        }
        activation_write_float(output, j, relu6(output_temp));
    }

    if (this->input_size > this->output_size)
        memcpy(workspace_start, output, this->output_size * sizeof(float));
}

void Conv2dReLU::forward(float* workspace_start, uint32_t workspace_size) {
    uint16_t input_channel_per_group  = this->input_channel_size / this->groups;
    uint16_t output_channel_per_group = this->output_channel_size / this->groups;
    uint16_t padded_row = this->input_row_size  + 2 * this->padding_row;
    uint16_t padded_col = this->input_col_size  + 2 * this->padding_col;

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

    uint16_t n, k;

    for (uint8_t g = 0; g < this->groups; g++) {
        for (uint16_t m = 0; m < this->output_row_size; m++) {
            for (uint16_t l = 0; l < this->output_col_size; l++) {

                for (uint16_t c_out = 0; c_out < output_channel_per_group; c_out++) {
                    n = g * output_channel_per_group + c_out;
                    pixel[c_out] = this->bias ? parameter_read_float(this->bias, n) : 0.0f;

                    for (uint16_t c_in = 0; c_in < input_channel_per_group; c_in++) {
                        k = g * input_channel_per_group + c_in;
                        for (uint8_t j = 0; j < this->kernel_row_size; j++) {
                            for (uint8_t i = 0; i < this->kernel_col_size; i++) {
                                pixel[c_out] += activation_read_float(
                                    input,
                                    ((j + m * this->stride_row) * padded_col * this->input_channel_size) +
                                    ((i + l * this->stride_col) * this->input_channel_size) + k
                                ) * parameter_read_float(
                                    this->weight,
                                    (n * this->kernel_row_size * this->kernel_col_size * input_channel_per_group) +
                                    (j * this->kernel_col_size * input_channel_per_group) +
                                    (i * input_channel_per_group) + c_in
                                );
                            }
                        }
                    }
                }

                for (uint16_t c_out = 0; c_out < output_channel_per_group; c_out++) {
                    n = g * output_channel_per_group + c_out;
                    activation_write_float(workspace_start,
                        (m * this->output_col_size * this->output_channel_size) +
                        (l * this->output_channel_size) + n,
                        relu(pixel[c_out])
                    );
                }
            }
        }
    }
}

void Conv2dReLU6::forward(float* workspace_start, uint32_t workspace_size) {
    uint16_t input_channel_per_group  = this->input_channel_size / this->groups;
    uint16_t output_channel_per_group = this->output_channel_size / this->groups;
    uint16_t padded_row = this->input_row_size  + 2 * this->padding_row;
    uint16_t padded_col = this->input_col_size  + 2 * this->padding_col;

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

    uint16_t n, k;

    for (uint8_t g = 0; g < this->groups; g++) {
        for (uint16_t m = 0; m < this->output_row_size; m++) {
            for (uint16_t l = 0; l < this->output_col_size; l++) {

                for (uint16_t c_out = 0; c_out < output_channel_per_group; c_out++) {
                    n = g * output_channel_per_group + c_out;
                    pixel[c_out] = this->bias ? parameter_read_float(this->bias, n) : 0.0f;

                    for (uint16_t c_in = 0; c_in < input_channel_per_group; c_in++) {
                        k = g * input_channel_per_group + c_in;
                        for (uint8_t j = 0; j < this->kernel_row_size; j++) {
                            for (uint8_t i = 0; i < this->kernel_col_size; i++) {
                                pixel[c_out] += activation_read_float(
                                    input,
                                    ((j + m * this->stride_row) * padded_col * this->input_channel_size) +
                                    ((i + l * this->stride_col) * this->input_channel_size) + k
                                ) * parameter_read_float(
                                    this->weight,
                                    (n * this->kernel_row_size * this->kernel_col_size * input_channel_per_group) +
                                    (j * this->kernel_col_size * input_channel_per_group) +
                                    (i * input_channel_per_group) + c_in
                                );
                            }
                        }
                    }
                }

                for (uint16_t c_out = 0; c_out < output_channel_per_group; c_out++) {
                    n = g * output_channel_per_group + c_out;
                    activation_write_float(workspace_start,
                        (m * this->output_col_size * this->output_channel_size) +
                        (l * this->output_channel_size) + n,
                        relu6(pixel[c_out])
                    );
                }
            }
        }
    }
}


// ============================================================================
// Dynamic quantization variants
// ============================================================================

void LinearReLU_DQ::forward(float* workspace_start, uint32_t workspace_size) {
    float* input;
    float* output;

    if (this->input_size > this->output_size) {
        input  = workspace_start;
        output = workspace_start + workspace_size - this->output_size;
    } else {
        input  = workspace_start + workspace_size - this->input_size;
        output = workspace_start;
        memcpy(input, workspace_start, this->input_size * sizeof(float));
    }

    int8_t (*parameter_read_packed_intb) (const int8_t*, uint32_t);
    get_parameter_read_packed_intb(this->quantize_property, &parameter_read_packed_intb);

    for (uint16_t j = 0; j < this->output_size; j++) {
        float output_temp = 0.0f;
        for (uint16_t i = 0; i < this->input_size; i++) {
            output_temp += activation_read_float(input, i) * parameter_read_packed_intb(this->weight, (j * this->input_size) + i);
        }
        uint8_t scale_index = get_granularity(this->quantize_property) == PER_CHANNEL ? j : 0;
        activation_write_float(output, j, relu(
            output_temp * parameter_read_float(this->weight_scale, scale_index) +
            (this->bias ? parameter_read_float(this->bias, j) : 0.0f)
        ));
    }

    if (this->input_size > this->output_size)
        memcpy(workspace_start, output, this->output_size * sizeof(float));
}

void LinearReLU6_DQ::forward(float* workspace_start, uint32_t workspace_size) {
    float* input;
    float* output;

    if (this->input_size > this->output_size) {
        input  = workspace_start;
        output = workspace_start + workspace_size - this->output_size;
    } else {
        input  = workspace_start + workspace_size - this->input_size;
        output = workspace_start;
        memcpy(input, workspace_start, this->input_size * sizeof(float));
    }

    int8_t (*parameter_read_packed_intb) (const int8_t*, uint32_t);
    get_parameter_read_packed_intb(this->quantize_property, &parameter_read_packed_intb);

    for (uint16_t j = 0; j < this->output_size; j++) {
        float output_temp = 0.0f;
        for (uint16_t i = 0; i < this->input_size; i++) {
            output_temp += activation_read_float(input, i) * parameter_read_packed_intb(this->weight, (j * this->input_size) + i);
        }
        uint8_t scale_index = get_granularity(this->quantize_property) == PER_CHANNEL ? j : 0;
        activation_write_float(output, j, relu6(
            output_temp * parameter_read_float(this->weight_scale, scale_index) +
            (this->bias ? parameter_read_float(this->bias, j) : 0.0f)
        ));
    }

    if (this->input_size > this->output_size)
        memcpy(workspace_start, output, this->output_size * sizeof(float));
}

void Conv2dReLU_DQ::forward(float* workspace_start, uint32_t workspace_size) {
    uint16_t input_channel_per_group  = this->input_channel_size / this->groups;
    uint16_t output_channel_per_group = this->output_channel_size / this->groups;
    uint16_t padded_row = this->input_row_size  + 2 * this->padding_row;
    uint16_t padded_col = this->input_col_size  + 2 * this->padding_col;

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

                for (uint16_t c_out = 0; c_out < output_channel_per_group; c_out++)
                    pixel[c_out] = 0.0f;

                for (uint16_t c_out = 0; c_out < output_channel_per_group; c_out++) {
                    n = g * output_channel_per_group + c_out;
                    for (uint16_t c_in = 0; c_in < input_channel_per_group; c_in++) {
                        k = g * input_channel_per_group + c_in;
                        for (uint8_t j = 0; j < this->kernel_row_size; j++) {
                            for (uint8_t i = 0; i < this->kernel_col_size; i++) {
                                pixel[c_out] += activation_read_float(
                                    input,
                                    ((j + m * this->stride_row) * padded_col * this->input_channel_size) +
                                    ((i + l * this->stride_col) * this->input_channel_size) + k
                                ) * parameter_read_packed_intb(
                                    this->weight,
                                    (n * this->kernel_row_size * this->kernel_col_size * input_channel_per_group) +
                                    (j * this->kernel_col_size * input_channel_per_group) +
                                    (i * input_channel_per_group) + c_in
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
                        relu(pixel[c_out] * parameter_read_float(this->weight_scale, scale_index) +
                             (this->bias ? parameter_read_float(this->bias, n) : 0.0f))
                    );
                }
            }
        }
    }
}

void Conv2dReLU6_DQ::forward(float* workspace_start, uint32_t workspace_size) {
    uint16_t input_channel_per_group  = this->input_channel_size / this->groups;
    uint16_t output_channel_per_group = this->output_channel_size / this->groups;
    uint16_t padded_row = this->input_row_size  + 2 * this->padding_row;
    uint16_t padded_col = this->input_col_size  + 2 * this->padding_col;

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

                for (uint16_t c_out = 0; c_out < output_channel_per_group; c_out++)
                    pixel[c_out] = 0.0f;

                for (uint16_t c_out = 0; c_out < output_channel_per_group; c_out++) {
                    n = g * output_channel_per_group + c_out;
                    for (uint16_t c_in = 0; c_in < input_channel_per_group; c_in++) {
                        k = g * input_channel_per_group + c_in;
                        for (uint8_t j = 0; j < this->kernel_row_size; j++) {
                            for (uint8_t i = 0; i < this->kernel_col_size; i++) {
                                pixel[c_out] += activation_read_float(
                                    input,
                                    ((j + m * this->stride_row) * padded_col * this->input_channel_size) +
                                    ((i + l * this->stride_col) * this->input_channel_size) + k
                                ) * parameter_read_packed_intb(
                                    this->weight,
                                    (n * this->kernel_row_size * this->kernel_col_size * input_channel_per_group) +
                                    (j * this->kernel_col_size * input_channel_per_group) +
                                    (i * input_channel_per_group) + c_in
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
                        relu6(pixel[c_out] * parameter_read_float(this->weight_scale, scale_index) +
                              (this->bias ? parameter_read_float(this->bias, n) : 0.0f))
                    );
                }
            }
        }
    }
}


// ============================================================================
// Static quantization variants
// ============================================================================

void LinearReLU_SQ::forward(int8_t* workspace_start, uint32_t workspace_size) {
    uint32_t data_per_byte = get_activation_data_per_byte(this->quantize_property);
    uint32_t input_bytes   = (uint32_t)ceil((float)this->input_size  / data_per_byte);
    uint32_t output_bytes  = (uint32_t)ceil((float)this->output_size / data_per_byte);

    int8_t* input;
    int8_t* output;

    if (this->input_size > this->output_size) {
        input  = workspace_start;
        output = workspace_start + workspace_size - output_bytes;
    } else {
        input  = workspace_start + workspace_size - input_bytes;
        output = workspace_start;
        memcpy(input, workspace_start, input_bytes);
    }

    void (*activation_write_packed_intb) (int8_t*, uint32_t, int8_t);
    int8_t (*activation_read_packed_intb) (int8_t*, uint32_t);
    int8_t (*parameter_read_packed_intb)  (const int8_t*, uint32_t);
    int8_t (*clamp_intb) (int32_t);

    get_activation_write_packed_intb(this->quantize_property, &activation_write_packed_intb);
    get_activation_read_packed_intb(this->quantize_property,  &activation_read_packed_intb);
    get_parameter_read_packed_intb(this->quantize_property,   &parameter_read_packed_intb);
    get_activation_clamp_intb(this->quantize_property,        &clamp_intb);

    for (uint16_t j = 0; j < this->output_size; j++) {
        int32_t output_temp = this->bias ? parameter_read_int32(this->bias, j) : 0;
        for (uint16_t i = 0; i < this->input_size; i++) {
            output_temp += ((int32_t)activation_read_packed_intb(input, i) - this->input_zero_point) *
                           parameter_read_packed_intb(this->weight, (j * this->input_size) + i);
        }
        uint8_t scale_index = get_granularity(this->quantize_property) == PER_CHANNEL ? j : 0;
        int32_t val = roundf(output_temp * parameter_read_float(this->bias_scale, scale_index) / this->output_scale);
        val += this->output_zero_point;
        activation_write_packed_intb(output, j, relu_zero_point(clamp_intb(val), this->output_zero_point));
    }

    if (this->input_size > this->output_size)
        memcpy(workspace_start, output, output_bytes);
}

void LinearReLU6_SQ::forward(int8_t* workspace_start, uint32_t workspace_size) {
    uint32_t data_per_byte = get_activation_data_per_byte(this->quantize_property);
    uint32_t input_bytes   = (uint32_t)ceil((float)this->input_size  / data_per_byte);
    uint32_t output_bytes  = (uint32_t)ceil((float)this->output_size / data_per_byte);

    int8_t* input;
    int8_t* output;

    if (this->input_size > this->output_size) {
        input  = workspace_start;
        output = workspace_start + workspace_size - output_bytes;
    } else {
        input  = workspace_start + workspace_size - input_bytes;
        output = workspace_start;
        memcpy(input, workspace_start, input_bytes);
    }

    void (*activation_write_packed_intb) (int8_t*, uint32_t, int8_t);
    int8_t (*activation_read_packed_intb) (int8_t*, uint32_t);
    int8_t (*parameter_read_packed_intb)  (const int8_t*, uint32_t);
    int8_t (*clamp_intb) (int32_t);

    get_activation_write_packed_intb(this->quantize_property, &activation_write_packed_intb);
    get_activation_read_packed_intb(this->quantize_property,  &activation_read_packed_intb);
    get_parameter_read_packed_intb(this->quantize_property,   &parameter_read_packed_intb);
    get_activation_clamp_intb(this->quantize_property,        &clamp_intb);

    // Quantize 6.0 into output space once (constant across all outputs)
    int8_t six_point = clamp_int8(
        (int32_t)roundf(6.0f / this->output_scale) + (int32_t)this->output_zero_point
    );

    for (uint16_t j = 0; j < this->output_size; j++) {
        int32_t output_temp = this->bias ? parameter_read_int32(this->bias, j) : 0;
        for (uint16_t i = 0; i < this->input_size; i++) {
            output_temp += ((int32_t)activation_read_packed_intb(input, i) - this->input_zero_point) *
                           parameter_read_packed_intb(this->weight, (j * this->input_size) + i);
        }
        uint8_t scale_index = get_granularity(this->quantize_property) == PER_CHANNEL ? j : 0;
        int32_t val = roundf(output_temp * parameter_read_float(this->bias_scale, scale_index) / this->output_scale);
        val += this->output_zero_point;
        activation_write_packed_intb(output, j, relu6_zero_point(clamp_intb(val), this->output_zero_point, six_point));
    }

    if (this->input_size > this->output_size)
        memcpy(workspace_start, output, output_bytes);
}

void Conv2dReLU_SQ::forward(int8_t* workspace_start, uint32_t workspace_size) {
    uint16_t input_channel_per_group  = this->input_channel_size / this->groups;
    uint16_t output_channel_per_group = this->output_channel_size / this->groups;
    uint16_t padded_row = this->input_row_size  + 2 * this->padding_row;
    uint16_t padded_col = this->input_col_size  + 2 * this->padding_col;

    uint32_t data_per_byte     = get_activation_data_per_byte(this->quantize_property);
    uint32_t padded_input_bytes = (uint32_t)ceil(
        (float)((uint32_t)padded_row * padded_col * this->input_channel_size) / data_per_byte
    );

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
    int8_t (*parameter_read_packed_intb)  (const int8_t*, uint32_t);
    int8_t (*clamp_intb) (int32_t);

    get_activation_write_packed_intb(this->quantize_property, &activation_write_packed_intb);
    get_activation_read_packed_intb(this->quantize_property,  &activation_read_packed_intb);
    get_parameter_read_packed_intb(this->quantize_property,   &parameter_read_packed_intb);
    get_activation_clamp_intb(this->quantize_property,        &clamp_intb);

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
                                    ((i + l * this->stride_col) * this->input_channel_size) + k
                                ) - this->input_zero_point) *
                                    parameter_read_packed_intb(
                                        this->weight,
                                        (n * this->kernel_row_size * this->kernel_col_size * input_channel_per_group) +
                                        (j * this->kernel_col_size * input_channel_per_group) +
                                        (i * input_channel_per_group) + c_in
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
                        relu_zero_point(clamp_intb(val), this->output_zero_point)
                    );
                }
            }
        }
    }
}

void Conv2dReLU6_SQ::forward(int8_t* workspace_start, uint32_t workspace_size) {
    uint16_t input_channel_per_group  = this->input_channel_size / this->groups;
    uint16_t output_channel_per_group = this->output_channel_size / this->groups;
    uint16_t padded_row = this->input_row_size  + 2 * this->padding_row;
    uint16_t padded_col = this->input_col_size  + 2 * this->padding_col;

    uint32_t data_per_byte      = get_activation_data_per_byte(this->quantize_property);
    uint32_t padded_input_bytes = (uint32_t)ceil(
        (float)((uint32_t)padded_row * padded_col * this->input_channel_size) / data_per_byte
    );

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
    int8_t (*parameter_read_packed_intb)  (const int8_t*, uint32_t);
    int8_t (*clamp_intb) (int32_t);

    get_activation_write_packed_intb(this->quantize_property, &activation_write_packed_intb);
    get_activation_read_packed_intb(this->quantize_property,  &activation_read_packed_intb);
    get_parameter_read_packed_intb(this->quantize_property,   &parameter_read_packed_intb);
    get_activation_clamp_intb(this->quantize_property,        &clamp_intb);

    // Quantize 6.0 into output space once (constant across all outputs)
    int8_t six_point = clamp_int8(
        (int32_t)roundf(6.0f / this->output_scale) + (int32_t)this->output_zero_point
    );

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
                                    ((i + l * this->stride_col) * this->input_channel_size) + k
                                ) - this->input_zero_point) *
                                    parameter_read_packed_intb(
                                        this->weight,
                                        (n * this->kernel_row_size * this->kernel_col_size * input_channel_per_group) +
                                        (j * this->kernel_col_size * input_channel_per_group) +
                                        (i * input_channel_per_group) + c_in
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
                        relu6_zero_point(clamp_intb(val), this->output_zero_point, six_point)
                    );
                }
            }
        }
    }
}
