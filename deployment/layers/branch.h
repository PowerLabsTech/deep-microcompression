#ifndef BRANCH_LAYER_H
#define BRANCH_LAYER_H

#include "layer.h"


// Branch buffer layout:
// [sublayer1: ptr | sublayer2: ptr | sublayer1_workspace_size: uint32_t]
class Branch : public Layer {
public:
    using Layer::Layer;
    void forward(float* workspace_start, uint32_t workspace_size);
    uint32_t get_output_size(void);
};


// Branch_SQ buffer layout:
// [sublayer1: ptr | sublayer2: ptr | sublayer1_workspace_size: uint32_t |
//  quantize_parameters: ptr | quantize_property: uint8_t]
class Branch_SQ : public Layer_SQ {
public:
    using Layer_SQ::Layer_SQ;
    void forward(int8_t* workspace_start, uint32_t workspace_size);
    uint32_t get_output_size(void);
};



#endif // BRANCH_LAYER_H
