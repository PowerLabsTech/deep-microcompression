#ifndef BATCHNORM_H
#define BATCHNORM_H

#include "layer.h"


#if !defined(QUANTIZATION_SCHEME) || QUANTIZATION_SCHEME != STATIC


class BatchNorm2d : public Layer{

private:
    // Input tensor dimensions
    uint16_t input_channel_size;  ///< Number of input channels
    uint16_t input_row_size;      ///< Height of input feature map
    uint16_t input_col_size;      ///< Width of input feature map

    // Weight and bias tensors
    const float* folded_weight;         ///< Pointer to weight tensor
    const float* folded_bias;           ///< Pointer to bias tensor    

public:
    BatchNorm2d(uint16_t input_channel_size, uint16_t input_row_size, uint16_t input_col_size,
                const float* folded_weight, const float* folded_bias);

    void forward(float* input, float* output);
};


#else // QUANTIZATION_SCHEME

#endif // QUANTIZATION_SCHEME

#endif // BATCHNORM_H