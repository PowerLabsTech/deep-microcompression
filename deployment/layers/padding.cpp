#include "padding.h"


void constantPad2d(float* input, float* output,
                   uint16_t channel_size, uint16_t row_size, uint16_t col_size,
                   Padding_t padding, float value) {

    uint16_t padded_row_size = row_size + padding.padding_top + padding.padding_bottom;
    uint16_t padded_col_size = col_size + padding.padding_left + padding.padding_right;

    // Loop order m→l→n backwards: writes to higher addresses first, safe for in-place expansion
    for (int32_t m = padded_row_size-1; m > -1; m--) {
        for (int32_t l = padded_col_size-1; l > -1; l--) {
            for (int32_t n = channel_size-1; n > -1; n--) {

                if (m < padding.padding_top || m >= padded_row_size - padding.padding_bottom ||
                    l < padding.padding_left || l >= padded_col_size - padding.padding_right) {

                    activation_write_float(output,
                        (m * padded_col_size * channel_size) +
                        (l * channel_size) +
                        n,
                        value
                    );
                } else {
                    activation_write_float(output,
                        (m * padded_col_size * channel_size) +
                        (l * channel_size) +
                        n,
                        activation_read_float(input,
                            ((m - padding.padding_top) * col_size * channel_size) +
                            ((l - padding.padding_left) * channel_size) +
                            n
                        )
                    );
                }
            }
        }
    }
}

void ConstantPad2d::forward(float* workspace_start, uint32_t workspace_size) {
    const uint8_t* p = this->buffer;
    uint16_t input_channel_size = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    uint16_t input_row_size     = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    uint16_t input_col_size     = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    p += 2;  // alignment pad to reach float at offset 8
    float value                 = dmc_pgm_read_float((const float*)p);   p += 4;
    Padding_t padding;
    padding.padding_left        = dmc_pgm_read_byte(p); p += 1;
    padding.padding_right       = dmc_pgm_read_byte(p); p += 1;
    padding.padding_top         = dmc_pgm_read_byte(p); p += 1;
    padding.padding_bottom      = dmc_pgm_read_byte(p);

    if (padding.is_padded()) {
        constantPad2d(workspace_start, workspace_start,
                      input_channel_size, input_row_size, input_col_size,
                      padding, value);
    }
}

uint32_t ConstantPad2d::get_output_size(void) {
    const uint8_t* p = this->buffer;
    uint16_t input_channel_size = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    uint16_t input_row_size     = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    uint16_t input_col_size     = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    p += 6;  // skip _pad (2) + value (4)
    uint8_t padding_left        = dmc_pgm_read_byte(p); p += 1;
    uint8_t padding_right       = dmc_pgm_read_byte(p); p += 1;
    uint8_t padding_top         = dmc_pgm_read_byte(p); p += 1;
    uint8_t padding_bottom      = dmc_pgm_read_byte(p);
    return (uint32_t)input_channel_size *
           (input_row_size + padding_top + padding_bottom) *
           (input_col_size + padding_left + padding_right);
}



void constantPad2d_SQ(int8_t* input, int8_t* output,
                      uint16_t channel_size, uint16_t row_size, uint16_t col_size,
                      Padding_t padding, int8_t value, uint8_t quantize_property) {

    uint16_t padded_row_size = row_size + padding.padding_top + padding.padding_bottom;
    uint16_t padded_col_size = col_size + padding.padding_left + padding.padding_right;

    void (*activation_write_packed_intb) (int8_t*, uint32_t, int8_t);
    int8_t (*activation_read_packed_intb) (int8_t*, uint32_t);

    get_activation_write_packed_intb(quantize_property, &activation_write_packed_intb);
    get_activation_read_packed_intb(quantize_property, &activation_read_packed_intb);

    // Loop order m→l→n backwards: writes to higher addresses first, safe for in-place expansion
    for (int32_t m = padded_row_size-1; m > -1; m--) {
        for (int32_t l = padded_col_size-1; l > -1; l--) {
            for (int32_t n = channel_size-1; n > -1; n--) {

                if (m < padding.padding_top || m >= padded_row_size - padding.padding_bottom ||
                    l < padding.padding_left || l >= padded_col_size - padding.padding_right) {

                    activation_write_packed_intb(output,
                        (m * padded_col_size * channel_size) + (l * channel_size) + n,
                        value
                    );
                } else {
                    activation_write_packed_intb(output,
                        (m * padded_col_size * channel_size) + (l * channel_size) + n,
                        activation_read_packed_intb(input,
                            ((m - padding.padding_top) * col_size * channel_size) +
                            ((l - padding.padding_left) * channel_size) + n
                        )
                    );
                }
            }
        }
    }
}

void ConstantPad2d_SQ::forward(int8_t* workspace_start, uint32_t workspace_size) {
    const uint8_t* p = this->buffer;
    uint16_t input_channel_size = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    uint16_t input_row_size     = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    uint16_t input_col_size     = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    int8_t input_value_point    = (int8_t)dmc_pgm_read_byte(p);          p += 1;
    Padding_t padding;
    padding.padding_left        = dmc_pgm_read_byte(p); p += 1;
    padding.padding_right       = dmc_pgm_read_byte(p); p += 1;
    padding.padding_top         = dmc_pgm_read_byte(p); p += 1;
    padding.padding_bottom      = dmc_pgm_read_byte(p); p += 1;
    uint8_t quantize_property = (uint8_t)dmc_pgm_read_byte(p);

    if (padding.is_padded()) {
        constantPad2d_SQ(workspace_start, workspace_start,
                         input_channel_size, input_row_size, input_col_size,
                         padding, input_value_point, quantize_property);
    }
}

uint32_t ConstantPad2d_SQ::get_output_size(void) {
    const uint8_t* p = this->buffer;
    uint16_t input_channel_size = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    uint16_t input_row_size     = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    uint16_t input_col_size     = dmc_pgm_read_word((const uint16_t*)p); p += 2;
    p += 1;  // skip input_value_point
    uint8_t padding_left        = dmc_pgm_read_byte(p); p += 1;
    uint8_t padding_right       = dmc_pgm_read_byte(p); p += 1;
    uint8_t padding_top         = dmc_pgm_read_byte(p); p += 1;
    uint8_t padding_bottom      = dmc_pgm_read_byte(p);
    return (uint32_t)input_channel_size *
           (input_row_size + padding_top + padding_bottom) *
           (input_col_size + padding_left + padding_right);
}