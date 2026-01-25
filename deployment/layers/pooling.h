/**
 * @file pooling.h
 * @brief MaxPool2d layer definition for 2D max pooling operations.
 * 
 * Supports both floating-point and quantized (int8_t) inference modes.
 */

#ifndef POOLING_H
#define POOLING_H

#include "layer.h"


/**
 * @brief MaxPool2d layer for floating-point inference.
 * 
 * Performs 2D max pooling operation on float input tensors.
 */
class MaxPool2d : public Layer {
private:
    uint16_t input_channel_size;  ///< Number of input channels
    uint16_t input_row_size;      ///< Height of input feature map
    uint16_t input_col_size;      ///< Width of input feature map

    uint16_t output_row_size;     ///< Height of output feature map
    uint16_t output_col_size;     ///< Width of output feature map

    uint8_t kernel_size;         ///< Size of pooling window (square)
    uint8_t stride;              ///< Stride for pooling operation
    uint8_t padding;             ///< Padding size around input

public:
    /**
     * @brief Constructor for floating-point MaxPool2d layer.
     * 
     * @param input_channel_size Number of input channels
     * @param input_row_size Height of input feature map
     * @param input_col_size Width of input feature map
     * @param kernel_size Size of pooling window
     * @param stride Stride for pooling operation
     * @param padding Padding size around input
     */
    MaxPool2d(uint16_t input_channel_size, 
              uint16_t input_row_size, 
              uint16_t input_col_size,
              uint8_t kernel_size, 
              uint8_t stride, 
              uint8_t padding);

    /**
     * @brief Forward pass for floating-point max pooling.
     * 
     * @param input Pointer to input tensor (float)
     * @param output Pointer to output tensor (float)
     */
    float* forward(float* input, float* workspace_start, uint32_t workspace_size);
};



class AvgPool2d : public Layer {
private:
    uint16_t input_channel_size;  ///< Number of input channels
    uint16_t input_row_size;      ///< Height of input feature map
    uint16_t input_col_size;      ///< Width of input feature map

    uint16_t output_row_size;     ///< Height of output feature map
    uint16_t output_col_size;     ///< Width of output feature map

    uint8_t kernel_size;         ///< Size of pooling window (square)
    uint8_t stride;              ///< Stride for pooling operation
    uint8_t padding;             ///< Padding size around input

public:
    /**
     * @brief Constructor for floating-point AvgPool2d layer.
     * 
     * @param input_channel_size Number of input channels
     * @param input_row_size Height of input feature map
     * @param input_col_size Width of input feature map
     * @param kernel_size Size of pooling window
     * @param stride Stride for pooling operation
     * @param padding Padding size around input
     */
    AvgPool2d(uint16_t input_channel_size, 
              uint16_t input_row_size, 
              uint16_t input_col_size,
              uint8_t kernel_size, 
              uint8_t stride, 
              uint8_t padding);

    /**
     * @brief Forward pass for floating-point max pooling.
     * 
     * @param input Pointer to input tensor (float)
     * @param output Pointer to output tensor (float)
     */
    float* forward(float* input, float* workspace_start, uint32_t workspace_size);
};




class MaxPool2d_SQ : public Layer_SQ {
private:
    uint16_t input_channel_size;  ///< Number of input channels
    uint16_t input_row_size;      ///< Height of input feature map
    uint16_t input_col_size;      ///< Width of input feature map

    uint16_t output_row_size;     ///< Height of output feature map
    uint16_t output_col_size;     ///< Width of output feature map

    uint8_t kernel_size;         ///< Size of pooling window (square)
    uint8_t stride;              ///< Stride for pooling operation
    uint8_t padding;             ///< Padding size around input

public:
    /**
     * @brief Constructor for quantized MaxPool2d layer.
     * 
     * @param input_channel_size Number of input channels
     * @param input_row_size Height of input feature map
     * @param input_col_size Width of input feature map
     * @param kernel_size Size of pooling window
     * @param stride Stride for pooling operation
     * @param padding Padding size around input
     */
    MaxPool2d_SQ(
        uint16_t input_channel_size, 
        uint16_t input_row_size, 
        uint16_t input_col_size,
        uint8_t kernel_size, 
        uint8_t stride, 
        uint8_t padding,
        uint8_t quantize_property
    );

    /**
     * @brief Forward pass for quantized max pooling.
     * 
     * @param input Pointer to input tensor (int8_t)
     * @param output Pointer to output tensor (int8_t)
     */
    int8_t* forward(int8_t* input, int8_t* workspace_start, uint32_t workspace_size);
};




class AvgPool2d_SQ : public Layer_SQ {
private:
    uint16_t input_channel_size;  ///< Number of input channels
    uint16_t input_row_size;      ///< Height of input feature map
    uint16_t input_col_size;      ///< Width of input feature map

    uint16_t output_row_size;     ///< Height of output feature map
    uint16_t output_col_size;     ///< Width of output feature map

    uint8_t kernel_size;         ///< Size of pooling window (square)
    uint8_t stride;              ///< Stride for pooling operation
    uint8_t padding;             ///< Padding size around input

public:
    /**
     * @brief Constructor for quantized AvgPool2d layer.
     * 
     * @param input_channel_size Number of input channels
     * @param input_row_size Height of input feature map
     * @param input_col_size Width of input feature map
     * @param kernel_size Size of pooling window
     * @param stride Stride for pooling operation
     * @param padding Padding size around input
     */
    AvgPool2d_SQ(
        uint16_t input_channel_size, 
        uint16_t input_row_size, 
        uint16_t input_col_size,
        uint8_t kernel_size, 
        uint8_t stride, 
        uint8_t padding,
        uint8_t quantize_property
    );

    /**
     * @brief Forward pass for quantized max pooling.
     * 
     * @param input Pointer to input tensor (int8_t)
     * @param output Pointer to output tensor (int8_t)
     */
    int8_t* forward(int8_t* input, int8_t* workspace_start, uint32_t workspace_size);
};


#endif // POOLING_H