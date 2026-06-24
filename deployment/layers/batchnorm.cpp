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

void BatchNorm2d::forward(float* workspace_start, uint32_t workspace_size) {
    // n outermost: each Flash read (folded_weight/bias) is cached once per channel
    // HWC index used so correct channel weight is applied to each element
    for (uint16_t n = 0; n < this->input_channel_size; n++) {
        float w = parameter_read_float(this->folded_weight, n);
        float b = parameter_read_float(this->folded_bias, n);
        for (uint16_t m = 0; m < this->input_row_size; m++) {
            for (uint16_t l = 0; l < this->input_col_size; l++) {
                uint32_t idx = (m * this->input_col_size * this->input_channel_size) +
                               (l * this->input_channel_size) + n;
                activation_write_float(workspace_start, idx,
                    activation_read_float(workspace_start, idx) * w + b);
            }
        }
    }
}


uint32_t BatchNorm2d::get_output_size(void) {
    return (this->input_channel_size * this->input_row_size * this->input_col_size);
}



#else // QUANTIZATION_SCHEME

#endif // QUANTIZATION_SCHEME
