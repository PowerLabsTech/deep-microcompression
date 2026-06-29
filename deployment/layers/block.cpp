#include "block.h"


// Buffer layout: [num_layers: uint16_t | layer0: ptr | layer1: ptr | ...]
void Block::forward(float* workspace_start, uint32_t workspace_size) {
    const uint8_t* p = this->buffer;
    uint16_t num_layers = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    for (uint16_t i = 0; i < num_layers; i++) {
        Layer* layer = (Layer*)dmc_pgm_read_ptr(p); p += DMC_PTR_SIZE;
        layer->forward(workspace_start, workspace_size);
    }
}

uint32_t Block::get_output_size(void) {
    const uint8_t* p = this->buffer;
    uint16_t num_layers = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    p += DMC_PTR_SIZE * (num_layers - 1);
    return ((Layer*)dmc_pgm_read_ptr(p))->get_output_size();
}


void Block_SQ::forward(int8_t* workspace_start, uint32_t workspace_size) {
    const uint8_t* p = this->buffer;
    uint16_t num_layers = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    for (uint16_t i = 0; i < num_layers; i++) {
        Layer_SQ* layer = (Layer_SQ*)dmc_pgm_read_ptr(p); p += DMC_PTR_SIZE;
        layer->forward(workspace_start, workspace_size);
    }
}

uint32_t Block_SQ::get_output_size(void) {
    const uint8_t* p = this->buffer;
    uint16_t num_layers = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    p += DMC_PTR_SIZE * (num_layers - 1);
    return ((Layer_SQ*)dmc_pgm_read_ptr(p))->get_output_size();
}
