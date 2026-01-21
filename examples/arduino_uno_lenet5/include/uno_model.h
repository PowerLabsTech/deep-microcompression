#ifndef UNO_MODEL_H
#define UNO_MODEL_H

#include <stdint.h>
#include <Arduino.h>
#include "deep_microcompression.h"


#define MAX_OUTPUT_EVEN_SIZE 512
#define MAX_OUTPUT_ODD_SIZE 784
extern int8_t workspace[MAX_OUTPUT_EVEN_SIZE + MAX_OUTPUT_ODD_SIZE];

#define LAYERS_LEN 7
extern Layer* layers[LAYERS_LEN];

extern Sequential uno_model;

extern const uint8_t conv2d_0_weight[] PROGMEM;
extern const uint8_t conv2d_0_bias[] PROGMEM;
extern const uint8_t conv2d_0_output_scale[];
extern const uint8_t conv2d_0_output_zero_point[];
extern const uint8_t conv2d_0_input_zero_point[];
extern const uint8_t conv2d_0_bias_scale[];
extern Conv2dReLU conv2d_0;

extern MaxPool2d maxpool2d_0;

extern const uint8_t conv2d_1_weight[] PROGMEM;
extern const uint8_t conv2d_1_bias[] PROGMEM;
extern const uint8_t conv2d_1_output_scale[];
extern const uint8_t conv2d_1_output_zero_point[];
extern const uint8_t conv2d_1_input_zero_point[];
extern const uint8_t conv2d_1_bias_scale[];
extern Conv2dReLU conv2d_1;

extern MaxPool2d maxpool2d_1;

extern Flatten flatten_0;

extern const uint8_t linear_0_weight[] PROGMEM;
extern const uint8_t linear_0_bias[] PROGMEM;
extern const uint8_t linear_0_output_scale[];
extern const uint8_t linear_0_output_zero_point[];
extern const uint8_t linear_0_input_zero_point[];
extern const uint8_t linear_0_bias_scale[];
extern LinearReLU linear_0;

extern const uint8_t linear_1_weight[] PROGMEM;
extern const uint8_t linear_1_bias[] PROGMEM;
extern const uint8_t linear_1_output_scale[];
extern const uint8_t linear_1_output_zero_point[];
extern const uint8_t linear_1_input_zero_point[];
extern const uint8_t linear_1_bias_scale[];
extern Linear linear_1;


#endif //UNO_MODEL_h
