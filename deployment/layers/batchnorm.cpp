#include "batchnorm.h"


#if !defined(QUANTIZATION_SCHEME) || QUANTIZATION_SCHEME != STATIC

BatchNorm2d::BatchNorm2d(uint16_t input_channel_size, uint16_t input_row_size, uint16_t input_col_size,
                const float* folded_weight, const float* folded_bias) {
    
    // Store layer parameters
    this->input_channel_size = input_channel_size;
    this->input_row_size = input_row_size;
    this->input_col_size = input_col_size;
    
    this->folded_weight = folded_weight;
    this->folded_bias = folded_bias;
}

float* BatchNorm2d::forward(float* input, float* workspace_start, uint32_t workspace_size) {

    // Getting the output start address with the input size as offset
    float* output = input == workspace_start ? workspace_start + workspace_size - (this->input_channel_size * this->input_row_size * this->input_col_size) : workspace_start;

    for (uint16_t n = 0; n < this->input_channel_size; n++) {
        for (uint16_t m = 0; m < this->input_row_size; m++) {
            for (uint16_t l = 0; l < this->input_col_size; l++) {

                activation_write_float(output, 
                        ((n * this->input_row_size * this->input_col_size) + 
                        (m * this->input_col_size) + 
                        l),
                        ((activation_read_float(input,
                            ((n * this->input_row_size * this->input_col_size) + 
                            (m * this->input_col_size) + 
                            l)
                        ) * parameter_read_float(this->folded_weight, n)) + 
                        parameter_read_float(this->folded_bias, n))
                );
            }
        }
    }
    
    return output;
}


#else // QUANTIZATION_SCHEME

#endif // QUANTIZATION_SCHEME
