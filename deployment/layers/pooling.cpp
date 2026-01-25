/**
 * @file pooling.cpp
 * @brief Implementation of MaxPool2d layer with support for:
 *       1. Non-quantized models (float)
 *       2. Static quantized models per tensor (int8_t)
 */

#include "pooling.h"


/**
 * @brief Constructor for floating-point MaxPool2d layer
 * @param input_channel_size Number of input channels
 * @param input_row_size Height of input feature map
 * @param input_col_size Width of input feature map
 * @param kernel_size Size of pooling window (square)
 * @param stride Stride of pooling operation
 * @param Padding Padding size (currently unused, reserved for future)
 */
MaxPool2d::MaxPool2d(uint16_t input_channel_size, uint16_t input_row_size, 
                    uint16_t input_col_size, uint8_t kernel_size, 
                    uint8_t stride, uint8_t padding) {
    // Store layer parameters
    this->input_channel_size = input_channel_size;
    this->input_row_size = input_row_size;
    this->input_col_size = input_col_size;
    this->kernel_size = kernel_size;
    this->stride = stride;
    this->padding = padding;

    // Calculate output dimensions
    this->output_row_size = ((this->input_row_size - this->kernel_size) / this->stride) + 1;
    this->output_col_size = ((this->input_col_size - this->kernel_size) / this->stride) + 1;
}

/**
 * @brief Forward pass for floating-point MaxPool2d
 * @param input Pointer to input tensor (float) in CHW format
 * @param output Pointer to output tensor (float) in CHW format
 * 
 * Performs 2D max pooling operation with the specified kernel size and stride.
 * The input is assumed to be in CHW (channels, height, width) format.
 */
float* MaxPool2d::forward(float* input, float* workspace_start, uint32_t workspace_size) {
    // Getting the output start address with the input size as offset
    float* output = input == workspace_start ? workspace_start + workspace_size - (this->input_channel_size * this->output_row_size * this->output_col_size) : workspace_start;

    float temp, input_val;

    // Loop over all channels
    for (uint16_t n = 0; n < this->input_channel_size; n++) {
        // Loop over output spatial dimensions
        for (uint16_t m = 0; m < this->output_row_size; m++) {
            for (uint16_t l = 0; l < this->output_col_size; l++) {
                // Initialize max value to smallest possible float
                temp = -FLT_MAX;

                // Iterate through pooling window
                for (uint8_t j = 0; j < this->kernel_size; j++) {
                    for (uint8_t i = 0; i < this->kernel_size; i++) {
                        // Calculate input index
                        input_val = activation_read_float(input, 
                            (n * this->input_row_size * this->input_col_size) +
                            ((m * this->stride + j) * this->input_col_size) +
                            (l * this->stride + i));
                        // Update max value
                        if (input_val > temp) {
                            temp = input_val;
                        }
                    }
                }

                // Store max value in output
                activation_write_float(output, 
                    (n * this->output_row_size * this->output_col_size) +
                    (m * this->output_col_size) + l,
                    temp
                    );
            }
        }
    }
    
    return output;
}



AvgPool2d::AvgPool2d(uint16_t input_channel_size, uint16_t input_row_size, 
                    uint16_t input_col_size, uint8_t kernel_size, 
                    uint8_t stride, uint8_t padding) {
    // Store layer parameters
    this->input_channel_size = input_channel_size;
    this->input_row_size = input_row_size;
    this->input_col_size = input_col_size;
    this->kernel_size = kernel_size;
    this->stride = stride;
    this->padding = padding;

    // Calculate output dimensions
    this->output_row_size = ((this->input_row_size - this->kernel_size) / this->stride) + 1;
    this->output_col_size = ((this->input_col_size - this->kernel_size) / this->stride) + 1;
}

float* AvgPool2d::forward(float* input, float* workspace_start, uint32_t workspace_size) {
    // Getting the output start address with the input size as offset
    float* output = input == workspace_start ? workspace_start + workspace_size - (this->input_channel_size * this->output_row_size * this->output_col_size) : workspace_start;

    float total;
    uint8_t pool_size = this->kernel_size * this->kernel_size;


    // Loop over all channels
    for (uint16_t n = 0; n < this->input_channel_size; n++) {
        // Loop over output spatial dimensions
        for (uint16_t m = 0; m < this->output_row_size; m++) {
            for (uint16_t l = 0; l < this->output_col_size; l++) {
                // Initialize max value to smallest possible float
                total = 0;

                // Iterate through pooling window
                for (uint8_t j = 0; j < this->kernel_size; j++) {
                    for (uint8_t i = 0; i < this->kernel_size; i++) {
                        // Calculate input index
                        total += activation_read_float(input, 
                            (n * this->input_row_size * this->input_col_size) +
                            ((m * this->stride + j) * this->input_col_size) +
                            (l * this->stride + i)
                        );
                    }
                }
                activation_write_float(output,
                    (n * this->output_row_size * this->output_col_size) +
                    (m * this->output_col_size) + l,
                    total / pool_size
                );
            }
        }
    }

    return output;
}


MaxPool2d_SQ::MaxPool2d_SQ(uint16_t input_channel_size, uint16_t input_row_size,
                    uint16_t input_col_size, uint8_t kernel_size,
                    uint8_t stride, uint8_t padding, uint8_t quantize_property) {
    // Store layer parameters
    this->input_channel_size = input_channel_size;
    this->input_row_size = input_row_size;
    this->input_col_size = input_col_size;
    this->kernel_size = kernel_size;
    this->stride = stride;
    this->padding = padding;

    // Calculate output dimensions
    this->output_row_size = ((this->input_row_size - this->kernel_size) / this->stride) + 1;
    this->output_col_size = ((this->input_col_size - this->kernel_size) / this->stride) + 1;
    
    this->quantize_property = quantize_property;
}

int8_t* MaxPool2d_SQ::forward(int8_t* input, int8_t* workspace_start, uint32_t workspace_size) {
    // Getting the output start address with the input size as offset
    int8_t* output = input == workspace_start ? workspace_start + workspace_size - (uint32_t)ceil((float)(this->input_channel_size * this->output_row_size * this->output_col_size) / get_activation_data_per_byte(this->quantize_property)) : workspace_start;

    int8_t temp, input_val;
    
    void (*activation_write_packed_intb) (int8_t*, uint32_t, int8_t);
    int8_t (*activation_read_packed_intb) (int8_t*, uint32_t);

    get_activation_write_packed_intb(this->quantize_property, &activation_write_packed_intb);
    get_activation_read_packed_intb(this->quantize_property, &activation_read_packed_intb);

    // Loop over all channels
    for (uint16_t n = 0; n < this->input_channel_size; n++) {
        // Loop over output spatial dimensions
        for (uint16_t m = 0; m < this->output_row_size; m++) {
            for (uint16_t l = 0; l < this->output_col_size; l++) {
                // Initialize max value to smallest possible int8
                temp = -128;

                // Iterate through pooling window
                for (uint8_t j = 0; j < this->kernel_size; j++) {
                    for (uint8_t i = 0; i < this->kernel_size; i++) {
                        // Calculate input index

                        input_val = activation_read_packed_intb(
                            input,
                            (n * this->input_row_size * this->input_col_size) +
                            ((m * this->stride + j) * this->input_col_size) +
                            (l * this->stride + i)
                        );

                        // Update max value
                        if (input_val > temp) {
                            temp = input_val;
                        }
                    }
                }

                // Store max value in output
                activation_write_packed_intb(
                    output, 
                    (n * this->output_row_size * this->output_col_size) +
                    (m * this->output_col_size) + l,
                    temp
                );
            }
        }
    }

    return output;
}


AvgPool2d_SQ::AvgPool2d_SQ(uint16_t input_channel_size, uint16_t input_row_size,
                    uint16_t input_col_size, uint8_t kernel_size,
                    uint8_t stride, uint8_t padding, uint8_t quantize_property) {
    // Store layer parameters
    this->input_channel_size = input_channel_size;
    this->input_row_size = input_row_size;
    this->input_col_size = input_col_size;
    this->kernel_size = kernel_size;
    this->stride = stride;
    this->padding = padding;

    // Calculate output dimensions
    this->output_row_size = ((this->input_row_size - this->kernel_size) / this->stride) + 1;
    this->output_col_size = ((this->input_col_size - this->kernel_size) / this->stride) + 1;

    this->quantize_property = quantize_property;
}

int8_t* AvgPool2d_SQ::forward(int8_t* input, int8_t* workspace_start, uint32_t workspace_size) {
    // Getting the output start address with the input size as offset
    int8_t* output = input == workspace_start ? workspace_start + workspace_size - (uint32_t)ceil((float)(this->input_channel_size * this->output_row_size * this->output_col_size) / get_activation_data_per_byte(this->quantize_property)) : workspace_start;

    int16_t total;

    uint8_t pool_size = this->kernel_size * this->kernel_size;
        
    void (*activation_write_packed_intb) (int8_t*, uint32_t, int8_t);
    int8_t (*activation_read_packed_intb) (int8_t*, uint32_t);
    
    get_activation_write_packed_intb(this->quantize_property, &activation_write_packed_intb);
    get_activation_read_packed_intb(this->quantize_property, &activation_read_packed_intb);

    // Loop over all channels
    for (uint16_t n = 0; n < this->input_channel_size; n++) {
        // Loop over output spatial dimensions
        for (uint16_t m = 0; m < this->output_row_size; m++) {
            for (uint16_t l = 0; l < this->output_col_size; l++) {
                // Initialize max value to smallest possible int8
                total = 0;

                // Iterate through pooling window
                for (uint8_t j = 0; j < this->kernel_size; j++) {
                    for (uint8_t i = 0; i < this->kernel_size; i++) {
                        total += activation_read_packed_intb(
                            input, 
                            (n * this->input_row_size * this->input_col_size) +
                            ((m * this->stride + j) * this->input_col_size) +
                            (l * this->stride + i)
                        );
                    }
                }

                // Store average value in output
                activation_write_packed_intb(
                    output, 
                    (n * this->output_row_size * this->output_col_size) +
                    (m * this->output_col_size) + l,
                    (int8_t)((float)total / pool_size)
                );
            }
        }
    }

    return output;
}


