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



void constantPad2d(float* input, float* output,
                   uint16_t channel_size, uint16_t row_size, uint16_t col_size,
                   Padding_t padding, float value);

void constantPad2d_SQ(int8_t* input, int8_t* output,
                      uint16_t channel_size, uint16_t row_size, uint16_t col_size,
                      Padding_t padding, int8_t value, uint8_t quantize_property);


// ConstantPad2d buffer layout:
// [input_channel_size: uint16_t | input_row_size: uint16_t | input_col_size: uint16_t |
//  _pad: uint8_t x2 | value: float |
//  padding_left: uint8_t | padding_right: uint8_t | padding_top: uint8_t | padding_bottom: uint8_t]

class ConstantPad2d : public Layer {
public:
    using Layer::Layer;
    void forward(float* workspace_start, uint32_t workspace_size);
    uint32_t get_output_size(void);
};


// ConstantPad2d_SQ buffer layout:
// [input_channel_size: uint16_t | input_row_size: uint16_t | input_col_size: uint16_t |
//  input_value_point: int8_t |
//  padding_left: uint8_t | padding_right: uint8_t | padding_top: uint8_t | padding_bottom: uint8_t]

class ConstantPad2d_SQ : public Layer_SQ {
public:
    using Layer_SQ::Layer_SQ;
    void forward(int8_t* workspace_start, uint32_t workspace_size);
    uint32_t get_output_size(void);
};


#endif // PAD_H