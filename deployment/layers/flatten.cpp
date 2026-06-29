#include "flatten.h"


void Flatten::forward(float* workspace_start, uint32_t workspace_size) {
    // No-op: flatten is a shape-only operation, data stays in place
}

uint32_t Flatten::get_output_size(void) {
    return dmc_pgm_read_dword((const uint32_t*)this->buffer);
}


void Flatten_SQ::forward(int8_t* workspace_start, uint32_t workspace_size) {
    // No-op: flatten is a shape-only operation, data stays in place
}

uint32_t Flatten_SQ::get_output_size(void) {
    return dmc_pgm_read_dword((const uint32_t*)this->buffer);
}
