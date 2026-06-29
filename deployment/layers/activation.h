/**
 * @file activation.h
 * @brief Header for ReLU activation layer with support:
 *      1. None quantized model
 *      2. Dynamic quantized model per tensor
 *          - 8 bit
 *          - 4 bit
 *      3. Static quantized model per tensor
 *          - 8 bit
 *          - 4 bit
 * 
 * The implementation switches between modes based on STATIC_QUANTIZATION_PER_TENSOR:
 * - Floating-point mode: Operates on float tensors
 * - Static Quantized mode: Operates on int8_t tensors with zero-point
 */

#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "layer.h"


/**
 * @brief Floating-point ReLU activation layer
 * 
 * Implements standard ReLU: output = max(0, input)
 */
class ReLU : public Layer {
    
public:
    /**
     * @brief Constructor for floating-point ReLU
     * @param input_size Number of elements in input tensor
     */
    /**
     * @brief Forward pass for floating-point ReLU
     * @param input Pointer to input tensor (float)
     * @param output Pointer to output tensor (float)
     */
    void forward(float* workspace_start, uint32_t workspace_size);
    uint32_t get_output_size(void);
};


class ReLU6 : public Layer {
public:
    using Layer::Layer;
    void forward(float* workspace_start, uint32_t workspace_size);
    uint32_t get_output_size(void);
};


class ReLU_SQ : public Layer_SQ {
public:
    using Layer_SQ::Layer_SQ;
    void forward(int8_t* workspace_start, uint32_t workspace_size);
    uint32_t get_output_size(void);
};


class ReLU6_SQ : public Layer_SQ {
public:
    using Layer_SQ::Layer_SQ;
    void forward(int8_t* workspace_start, uint32_t workspace_size);
    uint32_t get_output_size(void);
};

#endif // ACTIVATION_H