#include "block.h"



Block::Block(Layer** layers, uint8_t num_layers) {
    this->layers = layers;
    this->num_layers = num_layers;
}

void Block::forward(float* workspace_start, uint32_t workspace_size) {
    for (uint8_t i=0; i<this->num_layers; i++) {
        this->layers[i]->forward(workspace_start, workspace_size);
    }
}

uint32_t Block::get_output_size(void) {
    return this->layers[this->num_layers-1]->get_output_size();
}



Block_SQ::Block_SQ(Layer_SQ** layers, uint8_t num_layers) {
    this->layers = layers;
    this->num_layers = num_layers;
}

void Block_SQ::forward(int8_t* workspace_start, uint32_t workspace_size) {
    for (uint8_t i=0; i<this->num_layers; i++) {
        this->layers[i]->forward(workspace_start, workspace_size);
    }
}

uint32_t Block_SQ::get_output_size(void) {
    return this->layers[this->num_layers-1]->get_output_size();
}
