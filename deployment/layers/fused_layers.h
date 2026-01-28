#ifndef FUSED_LAYERS_H
#define FUSED_LAYERS_H

#include "layer.h"
#include "conv.h"
#include "linear.h"


class LinearReLU: public Linear {
public:
    using Linear::Linear;
    float* forward(float* input, float* workspace_start, uint32_t workspace_size);
    uint32_t get_output_size(void);
};


class LinearReLU6: public Linear {
public:
    using Linear::Linear;
    float* forward(float* input, float* workspace_start, uint32_t workspace_size);
    uint32_t get_output_size(void);
};


class Conv2dReLU: public Conv2d {
public:
    using Conv2d::Conv2d;
    float* forward(float* input, float* workspace_start, uint32_t workspace_size);
    uint32_t get_output_size(void);
};


class Conv2dReLU6: public Conv2d {
public:
    using Conv2d::Conv2d;
    float* forward(float* input, float* workspace_start, uint32_t workspace_size);
    uint32_t get_output_size(void);
};


class LinearReLU_DQ: public Linear_DQ {
public:
    using Linear_DQ::Linear_DQ;
    float* forward(float* input, float* workspace_start, uint32_t workspace_size);
    uint32_t get_output_size(void);
};


class LinearReLU6_DQ: public Linear_DQ {
public:
    using Linear_DQ::Linear_DQ;
    float* forward(float* input, float* workspace_start, uint32_t workspace_size);
    uint32_t get_output_size(void);
};


class Conv2dReLU_DQ: public Conv2d_DQ {
public:
    using Conv2d_DQ::Conv2d_DQ;
    float* forward(float* input, float* workspace_start, uint32_t workspace_size);
    uint32_t get_output_size(void);
};


class Conv2dReLU6_DQ: public Conv2d_DQ {
public:
    using Conv2d_DQ::Conv2d_DQ;
    float* forward(float* input, float* workspace_start, uint32_t workspace_size);
    uint32_t get_output_size(void);
};


class LinearReLU_SQ: public Linear_SQ {
public:
    using Linear_SQ::Linear_SQ;
    int8_t* forward(int8_t* input, int8_t* workspace_start, uint32_t workspace_size);
    uint32_t get_output_size(void);
};


class LinearReLU6_SQ: public Linear_SQ {
public:
    using Linear_SQ::Linear_SQ;
    int8_t* forward(int8_t* input, int8_t* workspace_start, uint32_t workspace_size);
    uint32_t get_output_size(void);
};


class Conv2dReLU_SQ: public Conv2d_SQ {
public:
    using Conv2d_SQ::Conv2d_SQ;
    int8_t* forward(int8_t* input, int8_t* workspace_start, uint32_t workspace_size);
    uint32_t get_output_size(void);
};


class Conv2dReLU6_SQ: public Conv2d_SQ {
public:
    using Conv2d_SQ::Conv2d_SQ;
    int8_t* forward(int8_t* input, int8_t* workspace_start, uint32_t workspace_size);
    uint32_t get_output_size(void);
};

#endif// FUSED_LAYERS_H