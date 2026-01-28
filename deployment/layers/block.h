#ifndef BLOCK_LAYER_H
#define BLOCK_LAYER_H

#include "layer.h"




class Block : public Layer {
private:
    Layer** layers;
    uint8_t num_layers;

public:
    Block(Layer** layers, uint8_t num_layers);

    float* forward(float* input, float* workspace_start, uint32_t workspace_size);
    uint32_t get_output_size(void);
    
    // uint32_t get_output_size(void);

};


class Block_SQ : public Layer_SQ {
private:
    Layer_SQ** layers;
    uint8_t num_layers;

public:
    Block_SQ(Layer_SQ** layers, uint8_t num_layers);

    int8_t* forward(int8_t* input, int8_t* workspace_start, uint32_t workspace_size);
    // uint32_t get_output_size(void);
    uint32_t get_output_size(void);

};



#endif // BLOCK_LAYER_H