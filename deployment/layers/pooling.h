#ifndef POOLING_H
#define POOLING_H

#include "layer.h"


// Buffer layout (all pooling classes):
// [input_channel_size: uint16_t | input_row_size: uint16_t | input_col_size: uint16_t |
//  output_row_size: uint16_t | output_col_size: uint16_t |
//  kernel_size: uint8_t | stride: uint8_t | padding: uint8_t]

class MaxPool2d : public Layer {
public:
    using Layer::Layer;
    void forward(float* workspace_start, uint32_t workspace_size);
    uint32_t get_output_size(void);
};


class AvgPool2d : public Layer {
public:
    using Layer::Layer;
    void forward(float* workspace_start, uint32_t workspace_size);
    uint32_t get_output_size(void);
};


class MaxPool2d_SQ : public Layer_SQ {
public:
    using Layer_SQ::Layer_SQ;
    void forward(int8_t* workspace_start, uint32_t workspace_size);
    uint32_t get_output_size(void);
};


class AvgPool2d_SQ : public Layer_SQ {
public:
    using Layer_SQ::Layer_SQ;
    void forward(int8_t* workspace_start, uint32_t workspace_size);
    uint32_t get_output_size(void);
};


#endif // POOLING_H
