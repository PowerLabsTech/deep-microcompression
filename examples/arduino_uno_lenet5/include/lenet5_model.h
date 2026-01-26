#ifndef LENET5_MODEL_H
#define LENET5_MODEL_H

#include <stdint.h>
#include <Arduino.h>
#include "deep_microcompression.h"


#define WORKSPACE_SIZE 1296
extern int8_t workspace[WORKSPACE_SIZE];

#define LAYERS_LEN 8
extern Layer_SQ* layers[LAYERS_LEN];

extern Sequential_SQ lenet5_model;

extern ConstantPad2d_SQ constantpad2d_0;

extern const uint8_t constantpad2d_0_input_value_point[];
extern const uint8_t conv2d_0_weight[] PROGMEM;
extern const uint8_t conv2d_0_bias[] PROGMEM;
extern Conv2dReLU_SQ conv2d_0;

extern const uint8_t conv2d_0_output_scale[];
extern const uint8_t conv2d_0_output_zero_point[];
extern const uint8_t conv2d_0_input_zero_point[];
extern const uint8_t conv2d_0_bias_scale[] PROGMEM;
extern MaxPool2d_SQ maxpool2d_0;

extern const uint8_t conv2d_1_weight[] PROGMEM;
extern const uint8_t conv2d_1_bias[] PROGMEM;
extern Conv2dReLU_SQ conv2d_1;

extern const uint8_t conv2d_1_output_scale[];
extern const uint8_t conv2d_1_output_zero_point[];
extern const uint8_t conv2d_1_input_zero_point[];
extern const uint8_t conv2d_1_bias_scale[] PROGMEM;
extern MaxPool2d_SQ maxpool2d_1;

extern Flatten_SQ flatten_0;

extern const uint8_t linear_0_weight[] PROGMEM;
extern const uint8_t linear_0_bias[] PROGMEM;
extern LinearReLU_SQ linear_0;

extern const uint8_t linear_0_output_scale[];
extern const uint8_t linear_0_output_zero_point[];
extern const uint8_t linear_0_input_zero_point[];
extern const uint8_t linear_0_bias_scale[] PROGMEM;
extern const uint8_t linear_1_weight[] PROGMEM;
extern const uint8_t linear_1_bias[] PROGMEM;
extern Linear_SQ linear_1;

extern const uint8_t linear_1_output_scale[];
extern const uint8_t linear_1_output_zero_point[];
extern const uint8_t linear_1_input_zero_point[];
extern const uint8_t linear_1_bias_scale[] PROGMEM;

#endif //LENET5_MODEL_h
