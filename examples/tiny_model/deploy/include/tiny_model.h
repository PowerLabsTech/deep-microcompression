#ifndef TINY_MODEL_H
#define TINY_MODEL_H

#include <stdint.h>
#include "deep_microcompression.h"


#define MAX_OUTPUT_EVEN_SIZE 784
#define MAX_OUTPUT_ODD_SIZE 784
extern int8_t workspace[MAX_OUTPUT_EVEN_SIZE + MAX_OUTPUT_ODD_SIZE];

#define LAYERS_LEN 4
extern Layer* layers[LAYERS_LEN];

extern Sequential tiny_model;

extern const uint8_t conv2d_0_weight[];
extern const uint8_t conv2d_0_bias[];
extern const uint8_t conv2d_0_output_scale[];
extern const uint8_t conv2d_0_output_zero_point[];
extern const uint8_t conv2d_0_input_zero_point[];
extern const uint8_t conv2d_0_bias_scale[];
extern Conv2d conv2d_0;

extern const uint8_t relu_0_input_zero_point[];
extern ReLU relu_0;

extern Flatten flatten_0;

extern const uint8_t linear_0_weight[];
extern const uint8_t linear_0_bias[];
extern const uint8_t linear_0_output_scale[];
extern const uint8_t linear_0_output_zero_point[];
extern const uint8_t linear_0_input_zero_point[];
extern const uint8_t linear_0_bias_scale[];
extern Linear linear_0;


#endif //TINY_MODEL_h
