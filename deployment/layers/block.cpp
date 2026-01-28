#include "block.h"



Block::Block(Layer** layers, uint8_t num_layers) {
    this->layers = layers;
    this->num_layers = num_layers;
}

float* Block::forward(float* input, float* workspace_start, uint32_t workspace_size) {
    // Getting the output start address with the input size as offset

    for (uint8_t i=0; i<this->num_layers; i++) {
        input = this->layers[i]->forward(input, workspace_start, workspace_size);
    }
    return input;
}


    
uint32_t Block::get_output_size(void) {
    return this->layers[this->num_layers-1]->get_output_size();
}
