#ifndef LINEAR_H
#define LINEAR_H

#include "layer.h"


// Linear buffer layout:
// [output_size: uint16_t | input_size: uint16_t |
//  weight: ptr | bias: ptr]
//
// Linear_DQ buffer layout:
// [output_size: uint16_t | input_size: uint16_t |
//  weight: ptr | bias: ptr | weight_scale: ptr |
//  quantize_property: uint8_t]
//
// Linear_SQ buffer layout:
// [output_size: uint16_t | input_size: uint16_t |
//  output_scale: float |
//  weight: ptr | bias: ptr | bias_scale: ptr |
//  output_zero_point: int8_t | input_zero_point: int8_t]


class Linear : public Layer {
public:
    using Layer::Layer;
    void forward(float* workspace_start, uint32_t workspace_size);
    uint32_t get_output_size(void);
};


class Linear_DQ : public Layer {
public:
    using Layer::Layer;
    void forward(float* workspace_start, uint32_t workspace_size);
    uint32_t get_output_size(void);
};


class Linear_SQ : public Layer_SQ {
public:
    using Layer_SQ::Layer_SQ;
    void forward(int8_t* workspace_start, uint32_t workspace_size);
    uint32_t get_output_size(void);
};


#endif // LINEAR_H
