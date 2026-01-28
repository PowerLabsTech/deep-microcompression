#ifndef BRANCH_LAYER_H
#define BRANCH_LAYER_H

#include "layer.h"




class Branch : public Layer {
private:
    Layer* sublayer1;
    Layer* sublayer2;

public:
    Branch(Layer* sublayer1, Layer* sublayer2);

    float* forward(float* input, float* workspace_start, uint32_t workspace_size);
    
    uint32_t get_output_size(void);

};


class Branch_SQ : public Layer_SQ {
private:
    Layer_SQ* sublayer1;
    Layer_SQ* sublayer2;

public:
    Branch_SQ(Layer_SQ* sublayer1, Layer_SQ* sublayer2, uint8_t quantize_property);

    int8_t* forward(int8_t* input, int8_t* workspace_start, uint32_t workspace_size);
    uint32_t get_output_size(void);

};



#endif // BRANCH_LAYER_H