#ifndef BRANCH_LAYER_H
#define BRANCH_LAYER_H

#include "layer.h"




class Branch : public Layer {
private:
    Layer* sublayer1;
    Layer* sublayer2;
    uint32_t sublayer1_workspace_size;

public:
    Branch(Layer* sublayer1, Layer* sublayer2, uint32_t sublayer1_workspace_size);

    void forward(float* workspace_start, uint32_t workspace_size);
    uint32_t get_output_size(void);

};


class Branch_SQ : public Layer_SQ {
private:
    Layer_SQ* sublayer1;
    Layer_SQ* sublayer2;
    uint32_t sublayer1_workspace_size;
    uint8_t* quantize_parameters;

public:
    Branch_SQ(Layer_SQ* sublayer1, Layer_SQ* sublayer2, 
        uint32_t sublayer1_workspace_size, uint8_t quantize_property,
        uint8_t* quantize_parameters);

    void forward(int8_t* workspace_start, uint32_t workspace_size);
    uint32_t get_output_size(void);
};



#endif // BRANCH_LAYER_H