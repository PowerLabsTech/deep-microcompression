#include "sequential.h"


// ── Sequential (float) ────────────────────────────────────────────────────────

Sequential::Sequential(const uint8_t* buffer, float* workspace_start, uint32_t workspace_size) {
    this->buffer          = buffer;
    this->workspace_start = workspace_start;
    this->workspace_size  = workspace_size;
}

void Sequential::predict(void) {
    const uint8_t* p = this->buffer;
    uint32_t required_size = dmc_pgm_read_dword((const uint32_t*)p); p += 4;
    uint8_t  layers_len    = dmc_pgm_read_byte(p);                   p += 1;
    if (this->workspace_size < required_size) return;
    for (uint8_t i = 0; i < layers_len; i++) {
        Layer* layer = (Layer*)dmc_pgm_read_ptr(p); p += DMC_PTR_SIZE;
        layer->forward(this->workspace_start, required_size);
    }
}

float Sequential::get_output(uint32_t index) {
    return activation_read_float(this->workspace_start, index);
}

void Sequential::set_input(uint32_t index, float value) {
    activation_write_float(this->workspace_start, index, value);
}


// ── Sequential_SQ (static quantization) ──────────────────────────────────────

// Buffer layout: [workspace_size:u32 | layers_len:u8 | quantize_property:u8 | layer0:ptr | ...]

Sequential_SQ::Sequential_SQ(const uint8_t* buffer, int8_t* workspace_start, uint32_t workspace_size) {
    this->buffer          = buffer;
    this->workspace_start = workspace_start;
    this->workspace_size  = workspace_size;
}

void Sequential_SQ::predict(void) {
    const uint8_t* p = this->buffer;
    uint32_t required_size     = dmc_pgm_read_dword((const uint32_t*)p); p += 4;
    uint8_t  layers_len        = dmc_pgm_read_byte(p);                   p += 1;
    uint8_t  quantize_property = dmc_pgm_read_byte(p);                   p += 1;
    if (this->workspace_size < required_size) return;
    for (uint8_t i = 0; i < layers_len; i++) {
        Layer_SQ* layer = (Layer_SQ*)dmc_pgm_read_ptr(p); p += DMC_PTR_SIZE;
        layer->forward(this->workspace_start, required_size);
    }
}

int8_t Sequential_SQ::get_output(uint32_t index) {
    uint8_t quantize_property = dmc_pgm_read_byte(this->buffer + 5);
    int8_t (*activation_read_packed_intb)(int8_t*, uint32_t);
    get_activation_read_packed_intb(quantize_property, &activation_read_packed_intb);
    return activation_read_packed_intb(this->workspace_start, index);
}

void Sequential_SQ::set_input(uint32_t index, int8_t value) {
    uint8_t quantize_property = dmc_pgm_read_byte(this->buffer + 5);
    void (*activation_read_packed_intb)(int8_t*, uint32_t, int8_t);
    get_activation_write_packed_intb(quantize_property, &activation_read_packed_intb);
    activation_read_packed_intb(this->workspace_start, index, value);
}
