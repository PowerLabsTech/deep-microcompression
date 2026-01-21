#include "pad.h"
#include "layer.h"

void pad_input(float* input, Padding_t padding, 
                const uint16_t input_channel_size, const uint16_t input_row_size, const uint16_t input_col_size, 
                const uint16_t padded_row_size, const uint16_t padded_col_size) {
    if (padding.is_padded()) {
        for (int32_t n = input_channel_size-1; n > -1; n--) {
            for (int32_t m = padded_row_size-1; m > -1; m--) {
                for (int32_t l = padded_col_size-1; l > -1; l--) {

                    if (m < padding.padding_top || m >= padded_row_size - padding.padding_bottom || 
                        l < padding.padding_left || l >= padded_col_size - padding.padding_right){
                        
                            act_write_float(input, 
                                ((n * padded_row_size * padded_col_size) + 
                                (m * padded_col_size) + 
                                l),
                                0
                            );
                        }
                    else {
                        act_write_float(input, 
                            ((n * padded_row_size * padded_col_size) + 
                            (m * padded_col_size) + 
                            l),
                            act_read_float(input, 
                                ((n * input_row_size * input_col_size) + 
                                ((m-padding.padding_top) * input_col_size) + 
                                (l-padding.padding_left))
                            )
                        );
                    }
                }
            }
        }

    }
}


void pad_input(int8_t* input, int8_t zero_point, Padding_t padding, 
                const uint16_t input_channel_size, const uint16_t input_row_size, const uint16_t input_col_size, 
                const uint16_t padded_row_size, const uint16_t padded_col_size) {
    if (padding.is_padded()) {
        for (int32_t n = input_channel_size-1; n > -1; n--) {
            for (int32_t m = padded_row_size-1; m > -1; m--) {
                for (int32_t l = padded_col_size-1; l > -1; l--) {

                    if (m < padding.padding_top || m >= padded_row_size - padding.padding_bottom || 
                        l < padding.padding_left || l >= padded_col_size - padding.padding_right){
                        
                            act_write_packed_intb(input, 
                                ((n * padded_row_size * padded_col_size) + 
                                (m * padded_col_size) + 
                                l),
                                zero_point
                            );
                        }
                    else {
                            act_write_packed_intb(input,
                                ((n * padded_row_size * padded_col_size) + 
                                (m * padded_col_size) + 
                                l),
                                act_read_packed_intb(input,
                                    ((n * input_row_size * input_col_size) + 
                                    ((m-padding.padding_top) * input_col_size) + 
                                    (l-padding.padding_left))
                                )
                            );
                    }
                }
            }
        }
    }
}

