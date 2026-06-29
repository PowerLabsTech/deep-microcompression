#include "linear.h"
#include <string.h>


void Linear::forward(float* workspace_start, uint32_t workspace_size) {
    const uint8_t* p = this->buffer;
    uint16_t output_size  = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    uint16_t input_size   = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    const float* weight   = (const float*)dmc_pgm_read_ptr(p);     p += DMC_PTR_SIZE;
    const float* bias     = (const float*)dmc_pgm_read_ptr(p);

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
        for (uint16_t i = 0; i < input_size; i++) {
            output_temp += activation_read_float(input, i) * parameter_read_float(weight, (j * input_size) + i);
        }
        activation_write_float(output, j, output_temp);
    }

    if (input_size > output_size)
        memcpy(workspace_start, output, output_size * sizeof(float));
}

uint32_t Linear::get_output_size(void) {
    return dmc_pgm_read_word((const uint16_t*)this->buffer);
}


void Linear_DQ::forward(float* workspace_start, uint32_t workspace_size) {
    const uint8_t* p = this->buffer;
    uint16_t output_size       = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    uint16_t input_size        = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    const int8_t* weight       = (const int8_t*)dmc_pgm_read_ptr(p);   p += DMC_PTR_SIZE;
    const float* bias          = (const float*)dmc_pgm_read_ptr(p);     p += DMC_PTR_SIZE;
    const float* weight_scale  = (const float*)dmc_pgm_read_ptr(p);     p += DMC_PTR_SIZE;
    uint8_t quantize_property  = dmc_pgm_read_byte(p);

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
        for (uint16_t i = 0; i < input_size; i++) {
            output_temp += activation_read_float(input, i) * parameter_read_packed_intb(weight, (j * input_size) + i);
        }
        uint8_t scale_index = get_granularity(quantize_property) == PER_CHANNEL ? j : 0;
        activation_write_float(output, j,
            output_temp * parameter_read_float(weight_scale, scale_index) +
            (bias ? parameter_read_float(bias, j) : 0.0f));
    }

    if (input_size > output_size)
        memcpy(workspace_start, output, output_size * sizeof(float));
}

uint32_t Linear_DQ::get_output_size(void) {
    return dmc_pgm_read_word((const uint16_t*)this->buffer);
}


void Linear_SQ::forward(int8_t* workspace_start, uint32_t workspace_size) {
    const uint8_t* p = this->buffer;
    uint16_t output_size       = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    uint16_t input_size        = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    float output_scale         = dmc_pgm_read_float((const float*)p);   p += 4;
    const int8_t* weight       = (const int8_t*)dmc_pgm_read_ptr(p);    p += DMC_PTR_SIZE;
    const int32_t* bias        = (const int32_t*)dmc_pgm_read_ptr(p);   p += DMC_PTR_SIZE;
    const float* bias_scale    = (const float*)dmc_pgm_read_ptr(p);     p += DMC_PTR_SIZE;
    int8_t output_zero_point   = (int8_t)dmc_pgm_read_byte(p);          p += 1;
    int8_t input_zero_point    = (int8_t)dmc_pgm_read_byte(p);          p += 1;
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
    int8_t (*input_activation_read_packed_intb) (int8_t*, uint32_t);
    int8_t (*parameter_read_packed_intb)  (const int8_t*, uint32_t);
    int8_t (*clamp_intb) (int32_t);
    get_activation_write_packed_intb(quantize_property, &activation_write_packed_intb);
    get_input_activation_read_packed_intb(quantize_property,  &input_activation_read_packed_intb);
    get_parameter_read_packed_intb(quantize_property,   &parameter_read_packed_intb);
    get_activation_clamp_intb(quantize_property,        &clamp_intb);

    for (uint16_t j = 0; j < output_size; j++) {
        int32_t output_temp = bias ? parameter_read_int32(bias, j) : 0;
        for (uint16_t i = 0; i < input_size; i++) {
            output_temp += ((int32_t)input_activation_read_packed_intb(input, i) - input_zero_point) *
                           parameter_read_packed_intb(weight, (j * input_size) + i);
        }
        uint8_t scale_index = get_granularity(quantize_property) == PER_CHANNEL ? j : 0;
        int32_t val = roundf(output_temp * parameter_read_float(bias_scale, scale_index) / output_scale);
        val += output_zero_point;
        activation_write_packed_intb(output, j, clamp_intb(val));
    }

    if (input_size > output_size)
        memcpy(workspace_start, output, output_bytes);
}

uint32_t Linear_SQ::get_output_size(void) {
    return dmc_pgm_read_word((const uint16_t*)this->buffer);
}
