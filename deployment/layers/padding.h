#ifndef PAD_H
#define PAD_H

#include <stdint.h>
#include "layer.h"

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



// void pad_input(float* input, Padding_t padding, 
//                 const uint16_t input_channel_size, const uint16_t input_row_size, const uint16_t input_col_size, 
//                 const uint16_t padded_row_size, const uint16_t padded_col_size);


// void pad_input(int8_t* input, int8_t zero_point, Padding_t padding, 
//                 const uint16_t input_channel_size, const uint16_t input_row_size, const uint16_t input_col_size, 
//                 const uint16_t padded_row_size, const uint16_t padded_col_size, uint8_t quantize_property);


class ConstantPad2d : public Layer {
private:
    uint16_t input_channel_size;  ///< Number of input channels
    uint16_t input_row_size;      ///< Height of input feature map
    uint16_t input_col_size;      ///< Width of input feature map
    float value;
    Padding_t padding;

public:
    ConstantPad2d(uint16_t input_channel_size, uint16_t input_row_size, 
                    uint16_t input_col_size, float value, Padding_t padding);

    float* forward(float* input, float* workspace_start, uint32_t workspace_size);
};


class ConstantPad2d_SQ : public Layer_SQ {
private:
    uint16_t input_channel_size;  ///< Number of input channels
    uint16_t input_row_size;      ///< Height of input feature map
    uint16_t input_col_size;      ///< Width of input feature map
    Padding_t padding;

    int8_t input_value_point;
    uint8_t quantize_property;

    
public:
    ConstantPad2d_SQ(uint16_t input_channel_size, uint16_t input_row_size, 
                    uint16_t input_col_size, int8_t value, Padding_t padding,  uint8_t quantize_property);
    int8_t* forward(int8_t* input, int8_t* workspace_start, uint32_t workspace_size);
};


#endif // PAD_H