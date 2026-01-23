#ifndef FUSED_LAYERS_H
#define FUSED_LAYERS_H

#include "layer.h"
#include "conv.h"
#include "linear.h"
#include "pad.h"


class Conv2dReLU: public Conv2d {
public:
    using Conv2d::Conv2d;

    float* forward(float* input, float* workspace_start, uint32_t workspace_size);
    int8_t* forward(int8_t* input, int8_t* workspace_start, uint32_t workspace_size);
};


class LinearReLU: public Linear {
public:

    using Linear::Linear;

    float* forward(float* input, float* workspace_start, uint32_t workspace_size);
    int8_t* forward(int8_t* input, int8_t* workspace_start, uint32_t workspace_size);
};

class Conv2dReLU6: public Conv2d {
public:
    using Conv2d::Conv2d;

    float* forward(float* input, float* workspace_start, uint32_t workspace_size);
    int8_t* forward(int8_t* input, int8_t* workspace_start, uint32_t workspace_size);
};


class LinearReLU6: public Linear {
public:

    using Linear::Linear;

    float* forward(float* input, float* workspace_start, uint32_t workspace_size);
    int8_t* forward(int8_t* input, int8_t* workspace_start, uint32_t workspace_size);
};

#endif// FUSED_LAYERS_H