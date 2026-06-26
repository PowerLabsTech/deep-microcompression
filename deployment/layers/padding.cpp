#include "padding.h"


ConstantPad2d::ConstantPad2d(uint16_t input_channel_size, uint16_t input_row_size, 
                    uint16_t input_col_size, float value, Padding_t padding) {
    this->input_channel_size = input_channel_size;
    this->input_row_size = input_row_size;
    this->input_col_size = input_col_size;

    this->value = value;
    this->padding = padding;                      
}

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
    if (this->padding.is_padded()) {
        constantPad2d(workspace_start, workspace_start,
                      this->input_channel_size, this->input_row_size, this->input_col_size,
                      this->padding, this->value);
    }
}


uint32_t ConstantPad2d::get_output_size(void) {
    return this->input_channel_size * \
            (this->input_row_size + this->padding.padding_top + this->padding.padding_bottom) * \
            (this->input_col_size + this->padding.padding_left + this->padding.padding_right);
}



ConstantPad2d_SQ::ConstantPad2d_SQ(uint16_t input_channel_size, uint16_t input_row_size, 
                uint16_t input_col_size, int8_t input_value_point, Padding_t padding, uint8_t quantize_property) {
    this->input_channel_size = input_channel_size;
    this->input_row_size = input_row_size;
    this->input_col_size = input_col_size;

    this->input_value_point = input_value_point;
    this->padding = padding;      
    
    this->quantize_property = quantize_property;
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
    if (this->padding.is_padded()) {
        constantPad2d_SQ(workspace_start, workspace_start,
                         this->input_channel_size, this->input_row_size, this->input_col_size,
                         this->padding, this->input_value_point, this->quantize_property);
    }
}

uint32_t ConstantPad2d_SQ::get_output_size(void) {
    return this->input_channel_size * \
        (this->input_row_size + this->padding.padding_top + this->padding.padding_bottom) * \
        (this->input_col_size + this->padding.padding_left + this->padding.padding_right);
}