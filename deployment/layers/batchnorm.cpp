#include "batchnorm.h"


#if !defined(QUANTIZATION_SCHEME) || QUANTIZATION_SCHEME != STATIC


// Buffer layout: [channel_size: uint16_t | row_size: uint16_t | col_size: uint16_t |
//                 folded_weight: ptr (DMC_PTR_SIZE bytes) | folded_bias: ptr (DMC_PTR_SIZE bytes)]
void BatchNorm2d::forward(float* workspace_start, uint32_t workspace_size) {
    const uint8_t* p = this->buffer;
    uint16_t input_channel_size = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    uint16_t input_row_size     = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    uint16_t input_col_size     = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    float* folded_weight = (float*)dmc_pgm_read_ptr(p); p += DMC_PTR_SIZE;
    float* folded_bias   = (float*)dmc_pgm_read_ptr(p);

    // n outermost: each Flash read (folded_weight/bias) is cached once per channel
    // HWC index used so correct channel weight is applied to each element
    for (uint16_t n = 0; n < input_channel_size; n++) {
        float w = parameter_read_float(folded_weight, n);
        float b = parameter_read_float(folded_bias, n);
        for (uint16_t m = 0; m < input_row_size; m++) {
            for (uint16_t l = 0; l < input_col_size; l++) {
                uint32_t idx = (m * input_col_size * input_channel_size) +
                               (l * input_channel_size) + n;
                activation_write_float(workspace_start, idx,
                    activation_read_float(workspace_start, idx) * w + b);
            }
        }
    }
}


uint32_t BatchNorm2d::get_output_size(void) {
    const uint8_t* p = this->buffer;
    uint16_t channel = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    uint16_t rows    = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    uint16_t cols    = dmc_pgm_read_word((const uint16_t*)p);
    return (uint32_t)channel * rows * cols;
}



#else // QUANTIZATION_SCHEME

#endif // QUANTIZATION_SCHEME
