#ifndef CONV_H
#define CONV_H

#include "layer.h"


// Conv2d buffer layout:
// [input_channel_size: uint16_t | input_row_size: uint16_t | input_col_size: uint16_t |
//  output_channel_size: uint16_t |
//  kernel_row_size: uint8_t | kernel_col_size: uint8_t |
//  padding_row: uint8_t | padding_col: uint8_t |
//  stride_row: uint8_t | stride_col: uint8_t | groups: uint8_t | [pad: uint8_t] |
//  weight: ptr | bias: ptr]
//
// Conv2d_DQ appends after bias:
//  weight_scale: ptr | quantize_property: uint8_t
//
// Conv2d_SQ appends after bias:
//  bias_scale: ptr | output_scale: float |
//  output_zero_point: int8_t | input_zero_point: int8_t
//
// output_row/col are not stored; compute as (padded - kernel) / stride + 1


// Reads the 16-byte spatial header common to all Conv2d buffer layouts.
// On return, *p points to the first byte after the header.
inline void read_conv_header(
    const uint8_t** p,
    uint16_t* input_channel_size, uint16_t* input_row_size, uint16_t* input_col_size,
    uint16_t* output_channel_size,
    uint8_t* kernel_row_size, uint8_t* kernel_col_size,
    uint8_t* padding_row, uint8_t* padding_col,
    uint8_t* stride_row, uint8_t* stride_col, uint8_t* groups)
{
    *input_channel_size  = dmc_pgm_read_word((const uint16_t*)*p); *p += 2;
    *input_row_size      = dmc_pgm_read_word((const uint16_t*)*p); *p += 2;
    *input_col_size      = dmc_pgm_read_word((const uint16_t*)*p); *p += 2;
    *output_channel_size = dmc_pgm_read_word((const uint16_t*)*p); *p += 2;
    *kernel_row_size     = dmc_pgm_read_byte(*p); *p += 1;
    *kernel_col_size     = dmc_pgm_read_byte(*p); *p += 1;
    *padding_row         = dmc_pgm_read_byte(*p); *p += 1;
    *padding_col         = dmc_pgm_read_byte(*p); *p += 1;
    *stride_row          = dmc_pgm_read_byte(*p); *p += 1;
    *stride_col          = dmc_pgm_read_byte(*p); *p += 1;
    *groups              = dmc_pgm_read_byte(*p); *p += 2;  // +2: groups byte + 1 alignment pad
}


class Conv2d : public Layer {
public:
    using Layer::Layer;
    void forward(float* workspace_start, uint32_t workspace_size);
    uint32_t get_output_size(void);
};


class Conv2d_DQ : public Layer {
public:
    using Layer::Layer;
    void forward(float* workspace_start, uint32_t workspace_size);
    uint32_t get_output_size(void);
};


class Conv2d_SQ : public Layer_SQ {
public:
    using Layer_SQ::Layer_SQ;
    void forward(int8_t* workspace_start, uint32_t workspace_size);
    uint32_t get_output_size(void);
};


#endif // CONV_H
