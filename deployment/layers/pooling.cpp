#include "pooling.h"


void MaxPool2d::forward(float* workspace_start, uint32_t workspace_size) {
    const uint8_t* p = this->buffer;
    uint16_t input_channel_size = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    uint16_t input_row_size     = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    uint16_t input_col_size     = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    uint8_t kernel_size         = dmc_pgm_read_byte(p); p += 1;
    uint8_t stride              = dmc_pgm_read_byte(p);
    uint16_t output_row_size    = (input_row_size - kernel_size) / stride + 1;
    uint16_t output_col_size    = (input_col_size - kernel_size) / stride + 1;

    float temp, input_val;
    for (uint16_t n = 0; n < input_channel_size; n++) {
        for (uint16_t m = 0; m < output_row_size; m++) {
            for (uint16_t l = 0; l < output_col_size; l++) {
                temp = -FLT_MAX;
                for (uint8_t j = 0; j < kernel_size; j++) {
                    for (uint8_t i = 0; i < kernel_size; i++) {
                        input_val = activation_read_float(workspace_start,
                            ((m * stride + j) * input_col_size * input_channel_size) +
                            ((l * stride + i) * input_channel_size) + n);
                        if (input_val > temp) temp = input_val;
                    }
                }
                activation_write_float(workspace_start,
                    (m * output_col_size * input_channel_size) +
                    (l * input_channel_size) + n, temp);
            }
        }
    }
}

uint32_t MaxPool2d::get_output_size(void) {
    const uint8_t* p = this->buffer;
    uint16_t input_channel_size = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    uint16_t input_row_size     = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    uint16_t input_col_size     = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    uint8_t kernel_size         = dmc_pgm_read_byte(p); p += 1;
    uint8_t stride              = dmc_pgm_read_byte(p);
    uint16_t output_row_size = (input_row_size - kernel_size) / stride + 1;
    uint16_t output_col_size = (input_col_size - kernel_size) / stride + 1;
    return (uint32_t)input_channel_size * output_row_size * output_col_size;
}


void AvgPool2d::forward(float* workspace_start, uint32_t workspace_size) {
    const uint8_t* p = this->buffer;
    uint16_t input_channel_size = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    uint16_t input_row_size     = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    uint16_t input_col_size     = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    uint8_t kernel_size         = dmc_pgm_read_byte(p); p += 1;
    uint8_t stride              = dmc_pgm_read_byte(p);
    uint16_t output_row_size    = (input_row_size - kernel_size) / stride + 1;
    uint16_t output_col_size    = (input_col_size - kernel_size) / stride + 1;
    uint8_t pool_size           = kernel_size * kernel_size;

    float total;
    for (uint16_t n = 0; n < input_channel_size; n++) {
        for (uint16_t m = 0; m < output_row_size; m++) {
            for (uint16_t l = 0; l < output_col_size; l++) {
                total = 0;
                for (uint8_t j = 0; j < kernel_size; j++) {
                    for (uint8_t i = 0; i < kernel_size; i++) {
                        total += activation_read_float(workspace_start,
                            ((m * stride + j) * input_col_size * input_channel_size) +
                            ((l * stride + i) * input_channel_size) + n);
                    }
                }
                activation_write_float(workspace_start,
                    (m * output_col_size * input_channel_size) +
                    (l * input_channel_size) + n, total / pool_size);
            }
        }
    }
}

uint32_t AvgPool2d::get_output_size(void) {
    const uint8_t* p = this->buffer;
    uint16_t input_channel_size = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    uint16_t input_row_size     = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    uint16_t input_col_size     = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    uint8_t kernel_size         = dmc_pgm_read_byte(p); p += 1;
    uint8_t stride              = dmc_pgm_read_byte(p);
    uint16_t output_row_size = (input_row_size - kernel_size) / stride + 1;
    uint16_t output_col_size = (input_col_size - kernel_size) / stride + 1;
    return (uint32_t)input_channel_size * output_row_size * output_col_size;
}


void MaxPool2d_SQ::forward(int8_t* workspace_start, uint32_t workspace_size) {
    const uint8_t* p = this->buffer;
    uint16_t input_channel_size = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    uint16_t input_row_size     = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    uint16_t input_col_size     = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    uint8_t kernel_size         = dmc_pgm_read_byte(p); p += 1;
    uint8_t stride              = dmc_pgm_read_byte(p); p += 1;
    uint8_t quantize_property   = (uint8_t)dmc_pgm_read_byte(p);
    uint16_t output_row_size    = (input_row_size - kernel_size) / stride + 1;
    uint16_t output_col_size    = (input_col_size - kernel_size) / stride + 1;

    void (*activation_write_packed_intb) (int8_t*, uint32_t, int8_t);
    int8_t (*activation_read_packed_intb) (int8_t*, uint32_t);
    get_activation_write_packed_intb(quantize_property, &activation_write_packed_intb);
    get_activation_read_packed_intb(quantize_property, &activation_read_packed_intb);

    int8_t temp, input_val;
    for (uint16_t n = 0; n < input_channel_size; n++) {
        for (uint16_t m = 0; m < output_row_size; m++) {
            for (uint16_t l = 0; l < output_col_size; l++) {
                temp = -128;
                for (uint8_t j = 0; j < kernel_size; j++) {
                    for (uint8_t i = 0; i < kernel_size; i++) {
                        input_val = activation_read_packed_intb(workspace_start,
                            ((m * stride + j) * input_col_size * input_channel_size) +
                            ((l * stride + i) * input_channel_size) + n);
                        if (input_val > temp) temp = input_val;
                    }
                }
                activation_write_packed_intb(workspace_start,
                    (m * output_col_size * input_channel_size) +
                    (l * input_channel_size) + n, temp);
            }
        }
    }
}

uint32_t MaxPool2d_SQ::get_output_size(void) {
    const uint8_t* p = this->buffer;
    uint16_t input_channel_size = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    uint16_t input_row_size     = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    uint16_t input_col_size     = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    uint8_t kernel_size         = dmc_pgm_read_byte(p); p += 1;
    uint8_t stride              = dmc_pgm_read_byte(p);
    uint16_t output_row_size = (input_row_size - kernel_size) / stride + 1;
    uint16_t output_col_size = (input_col_size - kernel_size) / stride + 1;
    return (uint32_t)input_channel_size * output_row_size * output_col_size;
}


void AvgPool2d_SQ::forward(int8_t* workspace_start, uint32_t workspace_size) {
    const uint8_t* p = this->buffer;
    uint16_t input_channel_size = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    uint16_t input_row_size     = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    uint16_t input_col_size     = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    uint8_t kernel_size         = dmc_pgm_read_byte(p); p += 1;
    uint8_t stride              = dmc_pgm_read_byte(p); p += 1;
    uint8_t quantize_property   = (uint8_t)dmc_pgm_read_byte(p);
    uint16_t output_row_size    = (input_row_size - kernel_size) / stride + 1;
    uint16_t output_col_size    = (input_col_size - kernel_size) / stride + 1;
    uint8_t pool_size           = kernel_size * kernel_size;

    void (*activation_write_packed_intb) (int8_t*, uint32_t, int8_t);
    int8_t (*activation_read_packed_intb) (int8_t*, uint32_t);
    get_activation_write_packed_intb(quantize_property, &activation_write_packed_intb);
    get_activation_read_packed_intb(quantize_property, &activation_read_packed_intb);

    int16_t total;
    for (uint16_t n = 0; n < input_channel_size; n++) {
        for (uint16_t m = 0; m < output_row_size; m++) {
            for (uint16_t l = 0; l < output_col_size; l++) {
                total = 0;
                for (uint8_t j = 0; j < kernel_size; j++) {
                    for (uint8_t i = 0; i < kernel_size; i++) {
                        total += activation_read_packed_intb(workspace_start,
                            ((m * stride + j) * input_col_size * input_channel_size) +
                            ((l * stride + i) * input_channel_size) + n);
                    }
                }
                activation_write_packed_intb(workspace_start,
                    (m * output_col_size * input_channel_size) +
                    (l * input_channel_size) + n,
                    (int8_t)((float)total / pool_size));
            }
        }
    }
}

uint32_t AvgPool2d_SQ::get_output_size(void) {
    const uint8_t* p = this->buffer;
    uint16_t input_channel_size = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    uint16_t input_row_size     = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    uint16_t input_col_size     = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    uint8_t kernel_size         = dmc_pgm_read_byte(p); p += 1;
    uint8_t stride              = dmc_pgm_read_byte(p);
    uint16_t output_row_size = (input_row_size - kernel_size) / stride + 1;
    uint16_t output_col_size = (input_col_size - kernel_size) / stride + 1;
    return (uint32_t)input_channel_size * output_row_size * output_col_size;
}
