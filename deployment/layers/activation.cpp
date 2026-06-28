#include "activation.h"


void ReLU::forward(float* workspace_start, uint32_t workspace_size) {
    uint32_t input_size = dmc_pgm_read_dword((const uint32_t*)this->buffer);
    for (uint32_t i = 0; i < input_size; i++) {
        activation_write_float(workspace_start, i, relu(activation_read_float(workspace_start, i)));
    }
}

uint32_t ReLU::get_output_size(void) {
    return dmc_pgm_read_dword((const uint32_t*)this->buffer);
}


void ReLU6::forward(float* workspace_start, uint32_t workspace_size) {
    uint32_t input_size = dmc_pgm_read_dword((const uint32_t*)this->buffer);
    for (uint32_t i = 0; i < input_size; i++) {
        activation_write_float(workspace_start, i, relu6(activation_read_float(workspace_start, i)));
    }
}

uint32_t ReLU6::get_output_size(void) {
    return dmc_pgm_read_dword((const uint32_t*)this->buffer);
}


// Buffer layout: [input_size: uint32_t | input_zero_point: int8_t | quantize_property: uint8_t]
void ReLU_SQ::forward(int8_t* workspace_start, uint32_t workspace_size) {
    const uint8_t* p    = this->buffer;
    uint32_t input_size = dmc_pgm_read_dword((const uint32_t*)p); p += 4;
    int8_t input_zero_point = (int8_t)dmc_pgm_read_byte(p); p += 1;
    uint8_t quantize_property = (uint8_t)dmc_pgm_read_byte(p);

    void (*activation_write_packed_intb) (int8_t*, uint32_t, int8_t);
    int8_t (*activation_read_packed_intb) (int8_t*, uint32_t);

    get_activation_write_packed_intb(quantize_property, &activation_write_packed_intb);
    get_activation_read_packed_intb(quantize_property, &activation_read_packed_intb);


    for (uint32_t i = 0; i < input_size; i++) {
        activation_write_packed_intb(workspace_start, i,
            relu_zero_point(activation_read_packed_intb(workspace_start, i), input_zero_point));
    }
}

uint32_t ReLU_SQ::get_output_size(void) {
    return dmc_pgm_read_dword((const uint32_t*)this->buffer);
}


// Buffer layout: [input_size: uint32_t | input_zero_point: int8_t | input_six_point: int8_t | quantize_property: uint8_t]
void ReLU6_SQ::forward(int8_t* workspace_start, uint32_t workspace_size) {
    const uint8_t* p    = this->buffer;
    uint32_t input_size = dmc_pgm_read_dword((const uint32_t*)p); p += 4;
    int8_t input_zero_point = (int8_t)dmc_pgm_read_byte(p); p += 1;
    int8_t input_six_point  = (int8_t)dmc_pgm_read_byte(p); p += 1;
    uint8_t quantize_property = (uint8_t)dmc_pgm_read_byte(p);

    void (*activation_write_packed_intb) (int8_t*, uint32_t, int8_t);
    int8_t (*activation_read_packed_intb) (int8_t*, uint32_t);
    get_activation_write_packed_intb(quantize_property, &activation_write_packed_intb);
    get_activation_read_packed_intb(quantize_property, &activation_read_packed_intb);

    for (uint32_t i = 0; i < input_size; i++) {
        activation_write_packed_intb(workspace_start, i,
            relu6_zero_point(activation_read_packed_intb(workspace_start, i), input_zero_point, input_six_point));
    }
}

uint32_t ReLU6_SQ::get_output_size(void) {
    return dmc_pgm_read_dword((const uint32_t*)this->buffer);
}
