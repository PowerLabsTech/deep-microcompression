#include "padding.h"


ConstantPad2d::ConstantPad2d(uint16_t input_channel_size, uint16_t input_row_size, 
                    uint16_t input_col_size, float value, Padding_t padding) {
    this->input_channel_size = input_channel_size;
    this->input_row_size = input_row_size;
    this->input_col_size = input_col_size;

    this->value = value;
    this->padding = padding;                      
}

float* ConstantPad2d::forward(float* input, float* workspace_start, uint32_t workspace_size) {

    uint16_t padded_row_size = this->input_row_size + this->padding.padding_top + this->padding.padding_bottom;
    uint16_t padded_col_size = this->input_col_size + this->padding.padding_left + this->padding.padding_right;

    float* output = input == workspace_start ? workspace_start + workspace_size - this->get_output_size() : workspace_start;

    if (this->padding.is_padded()) {
        for (int32_t n = this->input_channel_size-1; n > -1; n--) {
            for (int32_t m = padded_row_size-1; m > -1; m--) {
                for (int32_t l = padded_col_size-1; l > -1; l--) {

                    if (m < this->padding.padding_top || m >= padded_row_size - this->padding.padding_bottom || 
                        l < this->padding.padding_left || l >= padded_col_size - this->padding.padding_right){
                        
                            activation_write_float(output, 
                                ((n * padded_row_size * padded_col_size) + 
                                (m * padded_col_size) + 
                                l),
                                this->value
                            );
                        }
                    else {
                        activation_write_float(output, 
                            ((n * padded_row_size * padded_col_size) + 
                            (m * padded_col_size) + 
                            l),
                            activation_read_float(input, 
                                ((n * this->input_row_size * this->input_col_size) + 
                                ((m-this->padding.padding_top) * this->input_col_size) + 
                                (l-this->padding.padding_left))
                            )
                        );
                    }
                }
            }
        }

    }
    return output;
}


uint32_t ConstantPad2d::get_output_size(void) {
    return this->input_channel_size * \
            this->input_row_size + this->padding.padding_top + this->padding.padding_bottom * \
            this->input_col_size + this->padding.padding_left + this->padding.padding_right;
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


int8_t* ConstantPad2d_SQ::forward(int8_t* input, int8_t* workspace_start, uint32_t workspace_size) {

    uint16_t padded_row_size = this->input_row_size + this->padding.padding_top + this->padding.padding_bottom;
    uint16_t padded_col_size = this->input_col_size + this->padding.padding_left + this->padding.padding_right;

    int8_t* output = input == workspace_start ? workspace_start + workspace_size - (uint32_t)ceil(
        (float)this->get_output_size() / get_activation_data_per_byte(this->quantize_property)
    ) : workspace_start;

    void (*activation_write_packed_intb) (int8_t*, uint32_t, int8_t);
    int8_t (*activation_read_packed_intb) (int8_t*, uint32_t);
        
    get_activation_write_packed_intb(this->quantize_property, &activation_write_packed_intb);
    get_activation_read_packed_intb(this->quantize_property, &activation_read_packed_intb);

    if (this->padding.is_padded()) {
        for (int32_t n = this->input_channel_size-1; n > -1; n--) {
            for (int32_t m = padded_row_size-1; m > -1; m--) {
                for (int32_t l = padded_col_size-1; l > -1; l--) {

                    if (m < this->padding.padding_top || m >= padded_row_size - this->padding.padding_bottom || 
                        l < this->padding.padding_left || l >= padded_col_size - this->padding.padding_right){
                        
                            activation_write_packed_intb(output, 
                                ((n * padded_row_size * padded_col_size) + 
                                (m * padded_col_size) + 
                                l),
                                this->input_value_point
                            );
                        }
                    else {
                            activation_write_packed_intb(output,
                                ((n * padded_row_size * padded_col_size) + 
                                (m * padded_col_size) + 
                                l),
                                activation_read_packed_intb(input,
                                    ((n * this->input_row_size * this->input_col_size) + 
                                    ((m-this->padding.padding_top) * this->input_col_size) + 
                                    (l-this->padding.padding_left))
                                )
                            );
                    }
                }
            }
        }
    }
    return output;
}

uint32_t ConstantPad2d_SQ::get_output_size(void) {
    return this->input_channel_size * \
        this->input_row_size + this->padding.padding_top + this->padding.padding_bottom * \
        this->input_col_size + this->padding.padding_left + this->padding.padding_right;
}