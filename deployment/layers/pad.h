#ifndef PAD_H
#define PAD_H

#include <stdint.h>


struct Padding_t {
    uint8_t padding_left;
    uint8_t padding_right;
    uint8_t padding_top;
    uint8_t padding_bottom;

    bool is_padded() {
        return (this->padding_bottom + this->padding_top + 
                this->padding_left + this->padding_right) > 0;
    }

};



void pad_input(float* input, Padding_t padding, 
                const uint16_t input_channel_size, const uint16_t input_row_size, const uint16_t input_col_size, 
                const uint16_t padded_row_size, const uint16_t padded_col_size);


void pad_input(int8_t* input, int8_t zero_point, Padding_t padding, 
                const uint16_t input_channel_size, const uint16_t input_row_size, const uint16_t input_col_size, 
                const uint16_t padded_row_size, const uint16_t padded_col_size);


#endif // PAD_H