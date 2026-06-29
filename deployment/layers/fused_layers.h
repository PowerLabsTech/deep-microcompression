/**
 * @file fused_layers.h
 * @brief Fused Conv2d/Linear + ReLU/ReLU6 layers.
 *
 * Each class inherits from its base layer, calls the base forward() for the
 * linear/conv computation, then applies the activation in-place over the output.
 *
 * All classes conform to the in-place 2-argument forward() API:
 *   void forward(T* workspace_start, uint32_t workspace_size)
 */

#ifndef FUSED_LAYERS_H
#define FUSED_LAYERS_H

#include "layer.h"
#include "conv.h"
#include "linear.h"


// ============================================================================
// Float variants
// ============================================================================

class LinearReLU : public Linear {
public:
    using Linear::Linear;
    void forward(float* workspace_start, uint32_t workspace_size);
};

class LinearReLU6 : public Linear {
public:
    using Linear::Linear;
    void forward(float* workspace_start, uint32_t workspace_size);
};

class Conv2dReLU : public Conv2d {
public:
    using Conv2d::Conv2d;
    void forward(float* workspace_start, uint32_t workspace_size);
};

class Conv2dReLU6 : public Conv2d {
public:
    using Conv2d::Conv2d;
    void forward(float* workspace_start, uint32_t workspace_size);
};


// ============================================================================
// Dynamic quantization variants (float activations, int weights)
// ============================================================================

class LinearReLU_DQ : public Linear_DQ {
public:
    using Linear_DQ::Linear_DQ;
    void forward(float* workspace_start, uint32_t workspace_size);
};

class LinearReLU6_DQ : public Linear_DQ {
public:
    using Linear_DQ::Linear_DQ;
    void forward(float* workspace_start, uint32_t workspace_size);
};

class Conv2dReLU_DQ : public Conv2d_DQ {
public:
    using Conv2d_DQ::Conv2d_DQ;
    void forward(float* workspace_start, uint32_t workspace_size);
};

class Conv2dReLU6_DQ : public Conv2d_DQ {
public:
    using Conv2d_DQ::Conv2d_DQ;
    void forward(float* workspace_start, uint32_t workspace_size);
};


// ============================================================================
// Static quantization variants (int8 activations)
// ============================================================================

class LinearReLU_SQ : public Linear_SQ {
public:
    using Linear_SQ::Linear_SQ;
    void forward(int8_t* workspace_start, uint32_t workspace_size);
};

class LinearReLU6_SQ : public Linear_SQ {
public:
    using Linear_SQ::Linear_SQ;
    void forward(int8_t* workspace_start, uint32_t workspace_size);
};

class Conv2dReLU_SQ : public Conv2d_SQ {
public:
    using Conv2d_SQ::Conv2d_SQ;
    void forward(int8_t* workspace_start, uint32_t workspace_size);
};

class Conv2dReLU6_SQ : public Conv2d_SQ {
public:
    using Conv2d_SQ::Conv2d_SQ;
    void forward(int8_t* workspace_start, uint32_t workspace_size);
};


#endif // FUSED_LAYERS_H
