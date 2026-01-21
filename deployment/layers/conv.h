/**
 * @file conv.h
 * @brief Header for 2D convolution layer with support or:
 *      1. None quantized model
 *      2. Dynamic quantized model per tensor
 *          - 8 bit
 *          - 4 bit
 *      3. Static quantized model per tensor
 *          - 8 bit
 *          - 4 bit
 * 
 * The implementation is selected via compile-time definitions:
 * - QUANTIZATION_NONE: Floating-point
 * - DYNAMIC_QUANTIZATION_PER_TENSOR: Dynamic quantization
 * - STATIC_QUANTIZATION_PER_TENSOR: Static quantization
 * 
 * - QUANTIZATION_BITWIDTH: 8, 4
 */

#ifndef CONV_H
#define CONV_H

#include "layer.h"
#include "pad.h"

#if !defined(QUANTIZATION_SCHEME) || QUANTIZATION_SCHEME == NONE

/**
 * @brief Floating-point 2D convolution layer
 */
class Conv2d : public Layer {
protected:
    // Input tensor dimensions
    uint8_t input_channel_size;  ///< Number of input channels
    uint16_t input_row_size;      ///< Height of input feature map
    uint16_t input_col_size;      ///< Width of input feature map

    // Output tensor dimensions
    uint8_t output_channel_size; ///< Number of output channels
    uint16_t output_row_size;     ///< Height of output feature map
    uint16_t output_col_size;     ///< Width of output feature map

    // Kernel parameters
    uint8_t kernel_row_size;     ///< Height of convolution kernel
    uint8_t kernel_col_size;     ///< Width of convolution kernel

    // Operation parameters
    uint8_t stride_row;         ///< Vertical stride
    uint8_t stride_col;         ///< Horizontal stride
    Padding_t  padding;            ///< Padding type (0=VALID, 1=SAME)
    uint8_t groups;

    // Weight and bias tensors
    const float* weight;         ///< Pointer to weight tensor
    const float* bias;           ///< Pointer to bias tensor

public:
    /**
     * @brief Constructor for floating-point Conv2d
     * @param input_channel_size Number of input channels
     * @param input_row_size Input height in pixels
     * @param input_col_size Input width in pixels
     * @param output_channel_size Number of output channels
     * @param kernel_row_size Kernel height
     * @param kernel_col_size Kernel width
     * @param stride_row Vertical stride
     * @param stride_col Horizontal stride
     * @param padding Padding type (0=VALID, 1=SAME)
     * @param weight Pointer to weight tensor
     * @param bias Pointer to bias tensor
     */
    Conv2d(uint16_t input_channel_size, uint16_t input_row_size, uint16_t input_col_size,
           uint16_t output_channel_size, uint8_t kernel_row_size, uint8_t kernel_col_size,
           uint8_t stride_row, uint8_t stride_col, Padding_t padding, uint8_t groups,
           const float* weight, const float* bias);
    
    /**
     * @brief Forward pass for floating-point Conv2d
     * @param input Input tensor (float)
     * @param output Output tensor (float)
     */
    void forward(float* input, float* output);
};

#elif QUANTIZATION_SCHEME == DYNAMIC // QUANTIZATION_SCHEME


/**
 * @brief Dynamically quantized 2D convolution layer
 * 
 * Uses int8_t weights with float input/output and per-tensor scaling
 */
class Conv2d : public Layer {
protected:
    // Input tensor dimensions
    uint16_t input_channel_size;  ///< Number of input channels
    uint16_t input_row_size;      ///< Height of input feature map
    uint16_t input_col_size;      ///< Width of input feature map

    // Output tensor dimensions
    uint16_t output_channel_size; ///< Number of output channels
    uint16_t output_row_size;     ///< Height of output feature map
    uint16_t output_col_size;     ///< Width of output feature map

    // Kernel parameters
    uint8_t kernel_row_size;     ///< Height of convolution kernel
    uint8_t kernel_col_size;     ///< Width of convolution kernel

    // Operation parameters
    uint8_t stride_row;         ///< Vertical stride
    uint8_t stride_col;         ///< Horizontal stride
    Padding_t padding;            ///< Padding type (0=VALID, 1=SAME)
    uint8_t groups;

    // Quantization parameters
    const int8_t* weight;       ///< Pointer to quantized weight tensor
    const float* bias;          ///< Pointer to bias tensor (float)
    float weight_scale;         ///< Scale factor for weights

public:
    /**
     * @brief Constructor for dynamically quantized Conv2d
     * @param weight_scale Scale factor for quantized weights
     * @param other parameters same as floating-point version
     */
    Conv2d(uint16_t input_channel_size, uint16_t input_row_size, uint16_t input_col_size,
           uint16_t output_channel_size, uint8_t kernel_row_size, uint8_t kernel_col_size,
           uint8_t stride_row, uint8_t stride_col, Padding_t padding, uint8_t groups,
           const int8_t* weight, const float* bias, float weight_scale);

    /**
     * @brief Forward pass for dynamically quantized Conv2d
     * @param input Input tensor (float)
     * @param output Output tensor (float)
     */
    void forward(float* input, float* output);
};


#elif QUANTIZATION_SCHEME == STATIC // QUANTIZATION_SCHEME

class Conv2d : public Layer {
protected:
    // Input tensor dimensions
    uint16_t input_channel_size;  ///< Number of input channels
    uint16_t input_row_size;      ///< Height of input feature map
    uint16_t input_col_size;      ///< Width of input feature map

    // Output tensor dimensions
    uint16_t output_channel_size; ///< Number of output channels
    uint16_t output_row_size;     ///< Height of output feature map
    uint16_t output_col_size;     ///< Width of output feature map

    // Kernel parameters
    uint8_t kernel_row_size;     ///< Height of convolution kernel
    uint8_t kernel_col_size;     ///< Width of convolution kernel

    // Operation parameters
    uint8_t stride_row;         ///< Vertical stride
    uint8_t stride_col;         ///< Horizontal stride
    Padding_t padding;            ///< Padding type (0=VALID, 1=SAME)
    uint8_t groups;

    // Weight and bias tensors
    const int8_t* weight;       ///< Pointer to quantized weight tensor
    const int32_t* bias;        ///< Pointer to quantized bias tensor

    // Quantization parameters
    float output_scale;          ///< Output tensor scale factor
    int8_t output_zero_point;    ///< Output tensor zero point
    int8_t input_zero_point;     ///< Input tensor zero point

    float bias_scale;           ///< Bias scale factor

public:
    Conv2d(uint16_t input_channel_size, uint16_t input_row_size, uint16_t input_col_size,
           uint16_t output_channel_size, uint8_t kernel_row_size, uint8_t kernel_col_size,
           uint8_t stride_row, uint8_t stride_col, Padding_t padding, uint8_t groups,
           const int8_t* weight, const int32_t* bias, float output_scale, 
           int8_t output_zero_point, int8_t input_zero_point,  float bias_scale);

    void forward(int8_t* input, int8_t* output);
};


#endif // QUANTIZATION_SCHEME

#endif // CONV_H