#include "fused_layers.h"
#include "padding.h"
#include <string.h>


// ============================================================================
// Float variants
// ============================================================================

void LinearReLU::forward(float* workspace_start, uint32_t workspace_size) {
    const uint8_t* p = this->buffer;
    uint16_t output_size = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    uint16_t input_size  = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    const float* weight  = (const float*)dmc_pgm_read_ptr(p);     p += DMC_PTR_SIZE;
    const float* bias    = (const float*)dmc_pgm_read_ptr(p);

    float* input;
    float* output;
    if (input_size > output_size) {
        input  = workspace_start;
        output = workspace_start + workspace_size - output_size;
    } else {
        input  = workspace_start + workspace_size - input_size;
        output = workspace_start;
        memcpy(input, workspace_start, input_size * sizeof(float));
    }

    for (uint16_t j = 0; j < output_size; j++) {
        float output_temp = bias ? parameter_read_float(bias, j) : 0.0f;
        for (uint16_t i = 0; i < input_size; i++)
            output_temp += activation_read_float(input, i) * parameter_read_float(weight, (j * input_size) + i);
        activation_write_float(output, j, relu(output_temp));
    }

    if (input_size > output_size)
        memcpy(workspace_start, output, output_size * sizeof(float));
}

void LinearReLU6::forward(float* workspace_start, uint32_t workspace_size) {
    const uint8_t* p = this->buffer;
    uint16_t output_size = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    uint16_t input_size  = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    const float* weight  = (const float*)dmc_pgm_read_ptr(p);     p += DMC_PTR_SIZE;
    const float* bias    = (const float*)dmc_pgm_read_ptr(p);

    float* input;
    float* output;
    if (input_size > output_size) {
        input  = workspace_start;
        output = workspace_start + workspace_size - output_size;
    } else {
        input  = workspace_start + workspace_size - input_size;
        output = workspace_start;
        memcpy(input, workspace_start, input_size * sizeof(float));
    }

    for (uint16_t j = 0; j < output_size; j++) {
        float output_temp = bias ? parameter_read_float(bias, j) : 0.0f;
        for (uint16_t i = 0; i < input_size; i++)
            output_temp += activation_read_float(input, i) * parameter_read_float(weight, (j * input_size) + i);
        activation_write_float(output, j, relu6(output_temp));
    }

    if (input_size > output_size)
        memcpy(workspace_start, output, output_size * sizeof(float));
}

void Conv2dReLU::forward(float* workspace_start, uint32_t workspace_size) {
    const uint8_t* p = this->buffer;
    uint16_t input_channel_size, input_row_size, input_col_size, output_channel_size;
    uint8_t kernel_row_size, kernel_col_size, padding_row, padding_col, stride_row, stride_col, groups;
    read_conv_header(&p,
        &input_channel_size, &input_row_size, &input_col_size,
        &output_channel_size,
        &kernel_row_size, &kernel_col_size, &padding_row, &padding_col,
        &stride_row, &stride_col, &groups);
    const float* weight = (const float*)dmc_pgm_read_ptr(p); p += DMC_PTR_SIZE;
    const float* bias   = (const float*)dmc_pgm_read_ptr(p);

    uint16_t input_channel_per_group  = input_channel_size / groups;
    uint16_t output_channel_per_group = output_channel_size / groups;
    uint16_t padded_row = input_row_size + 2 * padding_row;
    uint16_t padded_col = input_col_size + 2 * padding_col;
    uint16_t output_row_size = (padded_row - kernel_row_size) / stride_row + 1;
    uint16_t output_col_size = (padded_col - kernel_col_size) / stride_col + 1;

    float* pixel = workspace_start + workspace_size - output_channel_per_group;

    float* input;
    if (output_channel_size > input_channel_size) {
        input = workspace_start + workspace_size - output_channel_per_group
                                 - ((uint32_t)padded_row * padded_col * input_channel_size);
        if (padding_row + padding_col) {
            Padding_t pad = {padding_col, padding_col, padding_row, padding_row};
            constantPad2d(workspace_start, input, input_channel_size, input_row_size, input_col_size, pad, 0.0f);
        } else {
            memmove(input, workspace_start, (uint32_t)padded_row * padded_col * input_channel_size * sizeof(float));
        }
    } else {
        input = workspace_start;
        if (padding_row + padding_col) {
            Padding_t pad = {padding_col, padding_col, padding_row, padding_row};
            constantPad2d(workspace_start, workspace_start, input_channel_size, input_row_size, input_col_size, pad, 0.0f);
        }
    }

    uint16_t n, k;
    for (uint8_t g = 0; g < groups; g++) {
        for (uint16_t m = 0; m < output_row_size; m++) {
            for (uint16_t l = 0; l < output_col_size; l++) {
                for (uint16_t c_out = 0; c_out < output_channel_per_group; c_out++) {
                    n = g * output_channel_per_group + c_out;
                    pixel[c_out] = bias ? parameter_read_float(bias, n) : 0.0f;
                    for (uint16_t c_in = 0; c_in < input_channel_per_group; c_in++) {
                        k = g * input_channel_per_group + c_in;
                        for (uint8_t j = 0; j < kernel_row_size; j++) {
                            for (uint8_t i = 0; i < kernel_col_size; i++) {
                                pixel[c_out] += activation_read_float(input,
                                    ((j + m * stride_row) * padded_col * input_channel_size) +
                                    ((i + l * stride_col) * input_channel_size) + k
                                ) * parameter_read_float(weight,
                                    (n * kernel_row_size * kernel_col_size * input_channel_per_group) +
                                    (j * kernel_col_size * input_channel_per_group) +
                                    (i * input_channel_per_group) + c_in);
                            }
                        }
                    }
                }
                for (uint16_t c_out = 0; c_out < output_channel_per_group; c_out++) {
                    n = g * output_channel_per_group + c_out;
                    activation_write_float(workspace_start,
                        (m * output_col_size * output_channel_size) + (l * output_channel_size) + n,
                        relu(pixel[c_out]));
                }
            }
        }
    }
}

void Conv2dReLU6::forward(float* workspace_start, uint32_t workspace_size) {
    const uint8_t* p = this->buffer;
    uint16_t input_channel_size, input_row_size, input_col_size, output_channel_size;
    uint8_t kernel_row_size, kernel_col_size, padding_row, padding_col, stride_row, stride_col, groups;
    read_conv_header(&p,
        &input_channel_size, &input_row_size, &input_col_size, &output_channel_size,
        &kernel_row_size, &kernel_col_size, &padding_row, &padding_col,
        &stride_row, &stride_col, &groups);
    const float* weight = (const float*)dmc_pgm_read_ptr(p); p += DMC_PTR_SIZE;
    const float* bias   = (const float*)dmc_pgm_read_ptr(p);

    uint16_t input_channel_per_group  = input_channel_size / groups;
    uint16_t output_channel_per_group = output_channel_size / groups;
    uint16_t padded_row = input_row_size + 2 * padding_row;
    uint16_t padded_col = input_col_size + 2 * padding_col;
    uint16_t output_row_size = (padded_row - kernel_row_size) / stride_row + 1;
    uint16_t output_col_size = (padded_col - kernel_col_size) / stride_col + 1;

    float* pixel = workspace_start + workspace_size - output_channel_per_group;

    float* input;
    if (output_channel_size > input_channel_size) {
        input = workspace_start + workspace_size - output_channel_per_group
                                 - ((uint32_t)padded_row * padded_col * input_channel_size);
        if (padding_row + padding_col) {
            Padding_t pad = {padding_col, padding_col, padding_row, padding_row};
            constantPad2d(workspace_start, input, input_channel_size, input_row_size, input_col_size, pad, 0.0f);
        } else {
            memmove(input, workspace_start, (uint32_t)padded_row * padded_col * input_channel_size * sizeof(float));
        }
    } else {
        input = workspace_start;
        if (padding_row + padding_col) {
            Padding_t pad = {padding_col, padding_col, padding_row, padding_row};
            constantPad2d(workspace_start, workspace_start, input_channel_size, input_row_size, input_col_size, pad, 0.0f);
        }
    }

    uint16_t n, k;
    for (uint8_t g = 0; g < groups; g++) {
        for (uint16_t m = 0; m < output_row_size; m++) {
            for (uint16_t l = 0; l < output_col_size; l++) {
                for (uint16_t c_out = 0; c_out < output_channel_per_group; c_out++) {
                    n = g * output_channel_per_group + c_out;
                    pixel[c_out] = bias ? parameter_read_float(bias, n) : 0.0f;
                    for (uint16_t c_in = 0; c_in < input_channel_per_group; c_in++) {
                        k = g * input_channel_per_group + c_in;
                        for (uint8_t j = 0; j < kernel_row_size; j++) {
                            for (uint8_t i = 0; i < kernel_col_size; i++) {
                                pixel[c_out] += activation_read_float(input,
                                    ((j + m * stride_row) * padded_col * input_channel_size) +
                                    ((i + l * stride_col) * input_channel_size) + k
                                ) * parameter_read_float(weight,
                                    (n * kernel_row_size * kernel_col_size * input_channel_per_group) +
                                    (j * kernel_col_size * input_channel_per_group) +
                                    (i * input_channel_per_group) + c_in);
                            }
                        }
                    }
                }
                for (uint16_t c_out = 0; c_out < output_channel_per_group; c_out++) {
                    n = g * output_channel_per_group + c_out;
                    activation_write_float(workspace_start,
                        (m * output_col_size * output_channel_size) + (l * output_channel_size) + n,
                        relu6(pixel[c_out]));
                }
            }
        }
    }
}


// ============================================================================
// Dynamic quantization variants
// ============================================================================

void LinearReLU_DQ::forward(float* workspace_start, uint32_t workspace_size) {
    const uint8_t* p = this->buffer;
    uint16_t output_size      = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    uint16_t input_size       = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    const int8_t* weight      = (const int8_t*)dmc_pgm_read_ptr(p);    p += DMC_PTR_SIZE;
    const float* bias         = (const float*)dmc_pgm_read_ptr(p);     p += DMC_PTR_SIZE;
    const float* weight_scale = (const float*)dmc_pgm_read_ptr(p);     p += DMC_PTR_SIZE;
    uint8_t quantize_property = dmc_pgm_read_byte(p);

    float* input;
    float* output;
    if (input_size > output_size) {
        input  = workspace_start;
        output = workspace_start + workspace_size - output_size;
    } else {
        input  = workspace_start + workspace_size - input_size;
        output = workspace_start;
        memcpy(input, workspace_start, input_size * sizeof(float));
    }

    int8_t (*parameter_read_packed_intb) (const int8_t*, uint32_t);
    get_parameter_read_packed_intb(quantize_property, &parameter_read_packed_intb);

    for (uint16_t j = 0; j < output_size; j++) {
        float output_temp = 0.0f;
        for (uint16_t i = 0; i < input_size; i++)
            output_temp += activation_read_float(input, i) * parameter_read_packed_intb(weight, (j * input_size) + i);
        uint8_t scale_index = get_granularity(quantize_property) == PER_CHANNEL ? j : 0;
        activation_write_float(output, j, relu(
            output_temp * parameter_read_float(weight_scale, scale_index) +
            (bias ? parameter_read_float(bias, j) : 0.0f)));
    }

    if (input_size > output_size)
        memcpy(workspace_start, output, output_size * sizeof(float));
}

void LinearReLU6_DQ::forward(float* workspace_start, uint32_t workspace_size) {
    const uint8_t* p = this->buffer;
    uint16_t output_size      = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    uint16_t input_size       = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    const int8_t* weight      = (const int8_t*)dmc_pgm_read_ptr(p);    p += DMC_PTR_SIZE;
    const float* bias         = (const float*)dmc_pgm_read_ptr(p);     p += DMC_PTR_SIZE;
    const float* weight_scale = (const float*)dmc_pgm_read_ptr(p);     p += DMC_PTR_SIZE;
    uint8_t quantize_property = dmc_pgm_read_byte(p);

    float* input;
    float* output;
    if (input_size > output_size) {
        input  = workspace_start;
        output = workspace_start + workspace_size - output_size;
    } else {
        input  = workspace_start + workspace_size - input_size;
        output = workspace_start;
        memcpy(input, workspace_start, input_size * sizeof(float));
    }

    int8_t (*parameter_read_packed_intb) (const int8_t*, uint32_t);
    get_parameter_read_packed_intb(quantize_property, &parameter_read_packed_intb);

    for (uint16_t j = 0; j < output_size; j++) {
        float output_temp = 0.0f;
        for (uint16_t i = 0; i < input_size; i++)
            output_temp += activation_read_float(input, i) * parameter_read_packed_intb(weight, (j * input_size) + i);
        uint8_t scale_index = get_granularity(quantize_property) == PER_CHANNEL ? j : 0;
        activation_write_float(output, j, relu6(
            output_temp * parameter_read_float(weight_scale, scale_index) +
            (bias ? parameter_read_float(bias, j) : 0.0f)));
    }

    if (input_size > output_size)
        memcpy(workspace_start, output, output_size * sizeof(float));
}

void Conv2dReLU_DQ::forward(float* workspace_start, uint32_t workspace_size) {
    const uint8_t* p = this->buffer;
    uint16_t input_channel_size, input_row_size, input_col_size, output_channel_size;
    uint8_t kernel_row_size, kernel_col_size, padding_row, padding_col, stride_row, stride_col, groups;
    read_conv_header(&p,
        &input_channel_size, &input_row_size, &input_col_size, &output_channel_size,
        &kernel_row_size, &kernel_col_size, &padding_row, &padding_col,
        &stride_row, &stride_col, &groups);
    const int8_t* weight      = (const int8_t*)dmc_pgm_read_ptr(p);  p += DMC_PTR_SIZE;
    const float* bias         = (const float*)dmc_pgm_read_ptr(p);   p += DMC_PTR_SIZE;
    const float* weight_scale = (const float*)dmc_pgm_read_ptr(p);   p += DMC_PTR_SIZE;
    uint8_t quantize_property = dmc_pgm_read_byte(p);

    uint16_t input_channel_per_group  = input_channel_size / groups;
    uint16_t output_channel_per_group = output_channel_size / groups;
    uint16_t padded_row = input_row_size + 2 * padding_row;
    uint16_t padded_col = input_col_size + 2 * padding_col;
    uint16_t output_row_size = (padded_row - kernel_row_size) / stride_row + 1;
    uint16_t output_col_size = (padded_col - kernel_col_size) / stride_col + 1;

    float* pixel = workspace_start + workspace_size - output_channel_per_group;

    float* input;
    if (output_channel_size > input_channel_size) {
        input = workspace_start + workspace_size - output_channel_per_group
                                 - ((uint32_t)padded_row * padded_col * input_channel_size);
        if (padding_row + padding_col) {
            Padding_t pad = {padding_col, padding_col, padding_row, padding_row};
            constantPad2d(workspace_start, input, input_channel_size, input_row_size, input_col_size, pad, 0.0f);
        } else {
            memmove(input, workspace_start, (uint32_t)padded_row * padded_col * input_channel_size * sizeof(float));
        }
    } else {
        input = workspace_start;
        if (padding_row + padding_col) {
            Padding_t pad = {padding_col, padding_col, padding_row, padding_row};
            constantPad2d(workspace_start, workspace_start, input_channel_size, input_row_size, input_col_size, pad, 0.0f);
        }
    }

    int8_t (*parameter_read_packed_intb) (const int8_t*, uint32_t);
    get_parameter_read_packed_intb(quantize_property, &parameter_read_packed_intb);

    uint16_t n, k;
    for (uint8_t g = 0; g < groups; g++) {
        for (uint16_t m = 0; m < output_row_size; m++) {
            for (uint16_t l = 0; l < output_col_size; l++) {
                for (uint16_t c_out = 0; c_out < output_channel_per_group; c_out++)
                    pixel[c_out] = 0.0f;

                for (uint16_t c_out = 0; c_out < output_channel_per_group; c_out++) {
                    n = g * output_channel_per_group + c_out;
                    for (uint16_t c_in = 0; c_in < input_channel_per_group; c_in++) {
                        k = g * input_channel_per_group + c_in;
                        for (uint8_t j = 0; j < kernel_row_size; j++) {
                            for (uint8_t i = 0; i < kernel_col_size; i++) {
                                pixel[c_out] += activation_read_float(input,
                                    ((j + m * stride_row) * padded_col * input_channel_size) +
                                    ((i + l * stride_col) * input_channel_size) + k
                                ) * parameter_read_packed_intb(weight,
                                    (n * kernel_row_size * kernel_col_size * input_channel_per_group) +
                                    (j * kernel_col_size * input_channel_per_group) +
                                    (i * input_channel_per_group) + c_in);
                            }
                        }
                    }
                }

                for (uint16_t c_out = 0; c_out < output_channel_per_group; c_out++) {
                    n = g * output_channel_per_group + c_out;
                    uint8_t scale_index = get_granularity(quantize_property) == PER_CHANNEL ? n : 0;
                    activation_write_float(workspace_start,
                        (m * output_col_size * output_channel_size) + (l * output_channel_size) + n,
                        relu(pixel[c_out] * parameter_read_float(weight_scale, scale_index) +
                             (bias ? parameter_read_float(bias, n) : 0.0f)));
                }
            }
        }
    }
}

void Conv2dReLU6_DQ::forward(float* workspace_start, uint32_t workspace_size) {
    const uint8_t* p = this->buffer;
    uint16_t input_channel_size, input_row_size, input_col_size, output_channel_size;
    uint8_t kernel_row_size, kernel_col_size, padding_row, padding_col, stride_row, stride_col, groups;
    read_conv_header(&p,
        &input_channel_size, &input_row_size, &input_col_size,
        &output_channel_size,
        &kernel_row_size, &kernel_col_size, &padding_row, &padding_col,
        &stride_row, &stride_col, &groups);
    const int8_t* weight      = (const int8_t*)dmc_pgm_read_ptr(p);  p += DMC_PTR_SIZE;
    const float* bias         = (const float*)dmc_pgm_read_ptr(p);   p += DMC_PTR_SIZE;
    const float* weight_scale = (const float*)dmc_pgm_read_ptr(p);   p += DMC_PTR_SIZE;
    uint8_t quantize_property = dmc_pgm_read_byte(p);

    uint16_t input_channel_per_group  = input_channel_size / groups;
    uint16_t output_channel_per_group = output_channel_size / groups;
    uint16_t padded_row = input_row_size + 2 * padding_row;
    uint16_t padded_col = input_col_size + 2 * padding_col;
    uint16_t output_row_size = (padded_row - kernel_row_size) / stride_row + 1;
    uint16_t output_col_size = (padded_col - kernel_col_size) / stride_col + 1;

    float* pixel = workspace_start + workspace_size - output_channel_per_group;

    float* input;
    if (output_channel_size > input_channel_size) {
        input = workspace_start + workspace_size - output_channel_per_group
                                 - ((uint32_t)padded_row * padded_col * input_channel_size);
        if (padding_row + padding_col) {
            Padding_t pad = {padding_col, padding_col, padding_row, padding_row};
            constantPad2d(workspace_start, input, input_channel_size, input_row_size, input_col_size, pad, 0.0f);
        } else {
            memmove(input, workspace_start, (uint32_t)padded_row * padded_col * input_channel_size * sizeof(float));
        }
    } else {
        input = workspace_start;
        if (padding_row + padding_col) {
            Padding_t pad = {padding_col, padding_col, padding_row, padding_row};
            constantPad2d(workspace_start, workspace_start, input_channel_size, input_row_size, input_col_size, pad, 0.0f);
        }
    }

    int8_t (*parameter_read_packed_intb) (const int8_t*, uint32_t);
    get_parameter_read_packed_intb(quantize_property, &parameter_read_packed_intb);

    uint16_t n, k;
    for (uint8_t g = 0; g < groups; g++) {
        for (uint16_t m = 0; m < output_row_size; m++) {
            for (uint16_t l = 0; l < output_col_size; l++) {
                for (uint16_t c_out = 0; c_out < output_channel_per_group; c_out++)
                    pixel[c_out] = 0.0f;

                for (uint16_t c_out = 0; c_out < output_channel_per_group; c_out++) {
                    n = g * output_channel_per_group + c_out;
                    for (uint16_t c_in = 0; c_in < input_channel_per_group; c_in++) {
                        k = g * input_channel_per_group + c_in;
                        for (uint8_t j = 0; j < kernel_row_size; j++) {
                            for (uint8_t i = 0; i < kernel_col_size; i++) {
                                pixel[c_out] += activation_read_float(input,
                                    ((j + m * stride_row) * padded_col * input_channel_size) +
                                    ((i + l * stride_col) * input_channel_size) + k
                                ) * parameter_read_packed_intb(weight,
                                    (n * kernel_row_size * kernel_col_size * input_channel_per_group) +
                                    (j * kernel_col_size * input_channel_per_group) +
                                    (i * input_channel_per_group) + c_in);
                            }
                        }
                    }
                }

                for (uint16_t c_out = 0; c_out < output_channel_per_group; c_out++) {
                    n = g * output_channel_per_group + c_out;
                    uint8_t scale_index = get_granularity(quantize_property) == PER_CHANNEL ? n : 0;
                    activation_write_float(workspace_start,
                        (m * output_col_size * output_channel_size) + (l * output_channel_size) + n,
                        relu6(pixel[c_out] * parameter_read_float(weight_scale, scale_index) +
                              (bias ? parameter_read_float(bias, n) : 0.0f)));
                }
            }
        }
    }
}


// ============================================================================
// Static quantization variants
// ============================================================================

void LinearReLU_SQ::forward(int8_t* workspace_start, uint32_t workspace_size) {
    const uint8_t* p = this->buffer;
    uint16_t output_size     = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    uint16_t input_size      = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    float output_scale       = dmc_pgm_read_float((const float*)p);   p += 4;
    const int8_t* weight     = (const int8_t*)dmc_pgm_read_ptr(p);    p += DMC_PTR_SIZE;
    const int32_t* bias      = (const int32_t*)dmc_pgm_read_ptr(p);   p += DMC_PTR_SIZE;
    const float* bias_scale  = (const float*)dmc_pgm_read_ptr(p);     p += DMC_PTR_SIZE;
    int8_t output_zero_point = (int8_t)dmc_pgm_read_byte(p);          p += 1;
    int8_t input_zero_point  = (int8_t)dmc_pgm_read_byte(p);          p += 1;
    uint8_t quantize_property = (uint8_t)dmc_pgm_read_byte(p);

    uint32_t data_per_byte = get_activation_data_per_byte(quantize_property);
    uint32_t input_bytes   = (uint32_t)ceil((float)input_size  / data_per_byte);
    uint32_t output_bytes  = (uint32_t)ceil((float)output_size / data_per_byte);

    int8_t* input;
    int8_t* output;
    if (input_size > output_size) {
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
    get_activation_write_packed_intb(quantize_property, &activation_write_packed_intb);
    get_activation_read_packed_intb(quantize_property,  &activation_read_packed_intb);
    get_parameter_read_packed_intb(quantize_property,   &parameter_read_packed_intb);
    get_activation_clamp_intb(quantize_property,        &clamp_intb);

    for (uint16_t j = 0; j < output_size; j++) {
        int32_t output_temp = bias ? parameter_read_int32(bias, j) : 0;
        for (uint16_t i = 0; i < input_size; i++) {
            output_temp += ((int32_t)activation_read_packed_intb(input, i) - input_zero_point) *
                           parameter_read_packed_intb(weight, (j * input_size) + i);
        }
        uint8_t scale_index = get_granularity(quantize_property) == PER_CHANNEL ? j : 0;
        int32_t val = roundf(output_temp * parameter_read_float(bias_scale, scale_index) / output_scale);
        val += output_zero_point;
        activation_write_packed_intb(output, j, relu_zero_point(clamp_intb(val), output_zero_point));
    }

    if (input_size > output_size)
        memcpy(workspace_start, output, output_bytes);
}

void LinearReLU6_SQ::forward(int8_t* workspace_start, uint32_t workspace_size) {
    const uint8_t* p = this->buffer;
    uint16_t output_size     = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    uint16_t input_size      = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    float output_scale       = dmc_pgm_read_float((const float*)p);   p += 4;
    const int8_t* weight     = (const int8_t*)dmc_pgm_read_ptr(p);    p += DMC_PTR_SIZE;
    const int32_t* bias      = (const int32_t*)dmc_pgm_read_ptr(p);   p += DMC_PTR_SIZE;
    const float* bias_scale  = (const float*)dmc_pgm_read_ptr(p);     p += DMC_PTR_SIZE;
    int8_t output_zero_point = (int8_t)dmc_pgm_read_byte(p);          p += 1;
    int8_t input_zero_point  = (int8_t)dmc_pgm_read_byte(p);          p += 1;
    uint8_t quantize_property = (uint8_t)dmc_pgm_read_byte(p);

    uint32_t data_per_byte = get_activation_data_per_byte(quantize_property);
    uint32_t input_bytes   = (uint32_t)ceil((float)input_size  / data_per_byte);
    uint32_t output_bytes  = (uint32_t)ceil((float)output_size / data_per_byte);

    int8_t* input;
    int8_t* output;
    if (input_size > output_size) {
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
    get_activation_write_packed_intb(quantize_property, &activation_write_packed_intb);
    get_activation_read_packed_intb(quantize_property,  &activation_read_packed_intb);
    get_parameter_read_packed_intb(quantize_property,   &parameter_read_packed_intb);
    get_activation_clamp_intb(quantize_property,        &clamp_intb);

    int8_t six_point = clamp_int8(
        (int32_t)roundf(6.0f / output_scale) + (int32_t)output_zero_point);

    for (uint16_t j = 0; j < output_size; j++) {
        int32_t output_temp = bias ? parameter_read_int32(bias, j) : 0;
        for (uint16_t i = 0; i < input_size; i++) {
            output_temp += ((int32_t)activation_read_packed_intb(input, i) - input_zero_point) *
                           parameter_read_packed_intb(weight, (j * input_size) + i);
        }
        uint8_t scale_index = get_granularity(quantize_property) == PER_CHANNEL ? j : 0;
        int32_t val = roundf(output_temp * parameter_read_float(bias_scale, scale_index) / output_scale);
        val += output_zero_point;
        activation_write_packed_intb(output, j, relu6_zero_point(clamp_intb(val), output_zero_point, six_point));
    }

    if (input_size > output_size)
        memcpy(workspace_start, output, output_bytes);
}

void Conv2dReLU_SQ::forward(int8_t* workspace_start, uint32_t workspace_size) {
    const uint8_t* p = this->buffer;
    uint16_t input_channel_size, input_row_size, input_col_size, output_channel_size;
    uint8_t kernel_row_size, kernel_col_size, padding_row, padding_col, stride_row, stride_col, groups;
    read_conv_header(&p,
        &input_channel_size, &input_row_size, &input_col_size,
        &output_channel_size,
        &kernel_row_size, &kernel_col_size, &padding_row, &padding_col,
        &stride_row, &stride_col, &groups);
    const int8_t* weight     = (const int8_t*)dmc_pgm_read_ptr(p);  p += DMC_PTR_SIZE;
    const int32_t* bias      = (const int32_t*)dmc_pgm_read_ptr(p); p += DMC_PTR_SIZE;
    const float* bias_scale  = (const float*)dmc_pgm_read_ptr(p);   p += DMC_PTR_SIZE;
    float output_scale       = dmc_pgm_read_float((const float*)p);  p += 4;
    int8_t output_zero_point = (int8_t)dmc_pgm_read_byte(p);         p += 1;
    int8_t input_zero_point  = (int8_t)dmc_pgm_read_byte(p);         p += 1;
    uint8_t quantize_property = (uint8_t)dmc_pgm_read_byte(p);

    uint16_t input_channel_per_group  = input_channel_size / groups;
    uint16_t output_channel_per_group = output_channel_size / groups;
    uint16_t padded_row = input_row_size + 2 * padding_row;
    uint16_t padded_col = input_col_size + 2 * padding_col;
    uint16_t output_row_size = (padded_row - kernel_row_size) / stride_row + 1;
    uint16_t output_col_size = (padded_col - kernel_col_size) / stride_col + 1;

    uint32_t data_per_byte      = get_activation_data_per_byte(quantize_property);
    uint32_t padded_input_bytes = (uint32_t)ceil(
        (float)((uint32_t)padded_row * padded_col * input_channel_size) / data_per_byte);

    int32_t* pixel = (int32_t*)(workspace_start + workspace_size) - output_channel_per_group;

    int8_t* input;
    if (output_channel_size > input_channel_size) {
        input = workspace_start + workspace_size
                - (uint32_t)(output_channel_per_group * sizeof(int32_t))
                - padded_input_bytes;
        if (padding_row + padding_col) {
            Padding_t pad = {padding_col, padding_col, padding_row, padding_row};
            constantPad2d_SQ(workspace_start, input, input_channel_size, input_row_size, input_col_size,
                             pad, input_zero_point, quantize_property);
        } else {
            memmove(input, workspace_start, padded_input_bytes);
        }
    } else {
        input = workspace_start;
        if (padding_row + padding_col) {
            Padding_t pad = {padding_col, padding_col, padding_row, padding_row};
            constantPad2d_SQ(workspace_start, workspace_start, input_channel_size, input_row_size, input_col_size,
                             pad, input_zero_point, quantize_property);
        }
    }

    void (*activation_write_packed_intb) (int8_t*, uint32_t, int8_t);
    int8_t (*activation_read_packed_intb) (int8_t*, uint32_t);
    int8_t (*parameter_read_packed_intb)  (const int8_t*, uint32_t);
    int8_t (*clamp_intb) (int32_t);
    get_activation_write_packed_intb(quantize_property, &activation_write_packed_intb);
    get_activation_read_packed_intb(quantize_property,  &activation_read_packed_intb);
    get_parameter_read_packed_intb(quantize_property,   &parameter_read_packed_intb);
    get_activation_clamp_intb(quantize_property,        &clamp_intb);

    uint16_t n, k;
    for (uint8_t g = 0; g < groups; g++) {
        for (uint16_t m = 0; m < output_row_size; m++) {
            for (uint16_t l = 0; l < output_col_size; l++) {
                for (uint16_t c_out = 0; c_out < output_channel_per_group; c_out++) {
                    n = g * output_channel_per_group + c_out;
                    pixel[c_out] = bias ? parameter_read_int32(bias, n) : 0;
                }
                for (uint16_t c_out = 0; c_out < output_channel_per_group; c_out++) {
                    n = g * output_channel_per_group + c_out;
                    for (uint16_t c_in = 0; c_in < input_channel_per_group; c_in++) {
                        k = g * input_channel_per_group + c_in;
                        for (uint8_t j = 0; j < kernel_row_size; j++) {
                            for (uint8_t i = 0; i < kernel_col_size; i++) {
                                pixel[c_out] += ((int32_t)activation_read_packed_intb(input,
                                    ((j + m * stride_row) * padded_col * input_channel_size) +
                                    ((i + l * stride_col) * input_channel_size) + k
                                ) - input_zero_point) *
                                    parameter_read_packed_intb(weight,
                                        (n * kernel_row_size * kernel_col_size * input_channel_per_group) +
                                        (j * kernel_col_size * input_channel_per_group) +
                                        (i * input_channel_per_group) + c_in);
                            }
                        }
                    }
                }
                for (uint16_t c_out = 0; c_out < output_channel_per_group; c_out++) {
                    n = g * output_channel_per_group + c_out;
                    uint8_t scale_index = get_granularity(quantize_property) == PER_CHANNEL ? n : 0;
                    int32_t val = roundf(pixel[c_out] * parameter_read_float(bias_scale, scale_index) / output_scale);
                    val += output_zero_point;
                    activation_write_packed_intb(workspace_start,
                        (m * output_col_size * output_channel_size) + (l * output_channel_size) + n,
                        relu_zero_point(clamp_intb(val), output_zero_point));
                }
            }
        }
    }
}

void Conv2dReLU6_SQ::forward(int8_t* workspace_start, uint32_t workspace_size) {
    const uint8_t* p = this->buffer;
    uint16_t input_channel_size, input_row_size, input_col_size, output_channel_size;
    uint8_t kernel_row_size, kernel_col_size, padding_row, padding_col, stride_row, stride_col, groups;
    read_conv_header(&p,
        &input_channel_size, &input_row_size, &input_col_size,
        &output_channel_size,
        &kernel_row_size, &kernel_col_size, &padding_row, &padding_col,
        &stride_row, &stride_col, &groups);
    const int8_t* weight     = (const int8_t*)dmc_pgm_read_ptr(p);  p += DMC_PTR_SIZE;
    const int32_t* bias      = (const int32_t*)dmc_pgm_read_ptr(p); p += DMC_PTR_SIZE;
    const float* bias_scale  = (const float*)dmc_pgm_read_ptr(p);   p += DMC_PTR_SIZE;
    float output_scale       = dmc_pgm_read_float((const float*)p);  p += 4;
    int8_t output_zero_point = (int8_t)dmc_pgm_read_byte(p);         p += 1;
    int8_t input_zero_point  = (int8_t)dmc_pgm_read_byte(p);         p += 1;
    uint8_t quantize_property = (uint8_t)dmc_pgm_read_byte(p);

    uint16_t input_channel_per_group  = input_channel_size / groups;
    uint16_t output_channel_per_group = output_channel_size / groups;
    uint16_t padded_row = input_row_size + 2 * padding_row;
    uint16_t padded_col = input_col_size + 2 * padding_col;
    uint16_t output_row_size = (padded_row - kernel_row_size) / stride_row + 1;
    uint16_t output_col_size = (padded_col - kernel_col_size) / stride_col + 1;

    uint32_t data_per_byte      = get_activation_data_per_byte(quantize_property);
    uint32_t padded_input_bytes = (uint32_t)ceil(
        (float)((uint32_t)padded_row * padded_col * input_channel_size) / data_per_byte);

    int32_t* pixel = (int32_t*)(workspace_start + workspace_size) - output_channel_per_group;

    int8_t* input;
    if (output_channel_size > input_channel_size) {
        input = workspace_start + workspace_size
                - (uint32_t)(output_channel_per_group * sizeof(int32_t))
                - padded_input_bytes;
        if (padding_row + padding_col) {
            Padding_t pad = {padding_col, padding_col, padding_row, padding_row};
            constantPad2d_SQ(workspace_start, input, input_channel_size, input_row_size, input_col_size,
                             pad, input_zero_point, quantize_property);
        } else {
            memmove(input, workspace_start, padded_input_bytes);
        }
    } else {
        input = workspace_start;
        if (padding_row + padding_col) {
            Padding_t pad = {padding_col, padding_col, padding_row, padding_row};
            constantPad2d_SQ(workspace_start, workspace_start, input_channel_size, input_row_size, input_col_size,
                             pad, input_zero_point, quantize_property);
        }
    }

    void (*activation_write_packed_intb) (int8_t*, uint32_t, int8_t);
    int8_t (*activation_read_packed_intb) (int8_t*, uint32_t);
    int8_t (*parameter_read_packed_intb)  (const int8_t*, uint32_t);
    int8_t (*clamp_intb) (int32_t);
    get_activation_write_packed_intb(quantize_property, &activation_write_packed_intb);
    get_activation_read_packed_intb(quantize_property,  &activation_read_packed_intb);
    get_parameter_read_packed_intb(quantize_property,   &parameter_read_packed_intb);
    get_activation_clamp_intb(quantize_property,        &clamp_intb);

    int8_t six_point = clamp_int8(
        (int32_t)roundf(6.0f / output_scale) + (int32_t)output_zero_point);

    uint16_t n, k;
    for (uint8_t g = 0; g < groups; g++) {
        for (uint16_t m = 0; m < output_row_size; m++) {
            for (uint16_t l = 0; l < output_col_size; l++) {
                for (uint16_t c_out = 0; c_out < output_channel_per_group; c_out++) {
                    n = g * output_channel_per_group + c_out;
                    pixel[c_out] = bias ? parameter_read_int32(bias, n) : 0;
                }
                for (uint16_t c_out = 0; c_out < output_channel_per_group; c_out++) {
                    n = g * output_channel_per_group + c_out;
                    for (uint16_t c_in = 0; c_in < input_channel_per_group; c_in++) {
                        k = g * input_channel_per_group + c_in;
                        for (uint8_t j = 0; j < kernel_row_size; j++) {
                            for (uint8_t i = 0; i < kernel_col_size; i++) {
                                pixel[c_out] += ((int32_t)activation_read_packed_intb(input,
                                    ((j + m * stride_row) * padded_col * input_channel_size) +
                                    ((i + l * stride_col) * input_channel_size) + k
                                ) - input_zero_point) *
                                    parameter_read_packed_intb(weight,
                                        (n * kernel_row_size * kernel_col_size * input_channel_per_group) +
                                        (j * kernel_col_size * input_channel_per_group) +
                                        (i * input_channel_per_group) + c_in);
                            }
                        }
                    }
                }
                for (uint16_t c_out = 0; c_out < output_channel_per_group; c_out++) {
                    n = g * output_channel_per_group + c_out;
                    uint8_t scale_index = get_granularity(quantize_property) == PER_CHANNEL ? n : 0;
                    int32_t val = roundf(pixel[c_out] * parameter_read_float(bias_scale, scale_index) / output_scale);
                    val += output_zero_point;
                    activation_write_packed_intb(workspace_start,
                        (m * output_col_size * output_channel_size) + (l * output_channel_size) + n,
                        relu6_zero_point(clamp_intb(val), output_zero_point, six_point));
                }
            }
        }
    }
}
