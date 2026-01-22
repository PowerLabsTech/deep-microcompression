#ifndef FUSED_LAYERS_H
#define FUSED_LAYERS_H

#include "layer.h"
#include "conv.h"
#include "linear.h"
#include "pad.h"


class Conv2dReLU: public Conv2d {
public:
    using Conv2d::Conv2d;

    float* forward(float* input, float* output);
    int8_t* forward(int8_t* input, int8_t* output);
};


class LinearReLU: public Linear {
public:

    using Linear::Linear;

    float* forward(float* input, float* output);
    int8_t* forward(int8_t* input, int8_t* output);
};

class Conv2dReLU6: public Conv2d {
public:
    using Conv2d::Conv2d;

    float* forward(float* input, float* output);
    int8_t* forward(int8_t* input, int8_t* output);
};


class LinearReLU6: public Linear {
public:

    using Linear::Linear;

    float* forward(float* input, float* output);
    int8_t* forward(int8_t* input, int8_t* output);
};

#endif// FUSED_LAYERS_H