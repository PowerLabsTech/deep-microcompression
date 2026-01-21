#include "uno_model.h"

int8_t workspace[MAX_OUTPUT_EVEN_SIZE + MAX_OUTPUT_ODD_SIZE];

Conv2dReLU conv2d_0(1, 28, 28, 2, 5, 5, 1, 1, {2, 2, 2, 2}, 1, (int8_t*)conv2d_0_weight, (int32_t*)conv2d_0_bias, *(float*)conv2d_0_output_scale, *(int8_t*)conv2d_0_output_zero_point, *(int8_t*)conv2d_0_input_zero_point, *(float*)conv2d_0_bias_scale);
MaxPool2d maxpool2d_0(2, 28, 28, 2, 2, 0);
Conv2dReLU conv2d_1(2, 14, 14, 9, 5, 5, 1, 1, {0, 0, 0, 0}, 1, (int8_t*)conv2d_1_weight, (int32_t*)conv2d_1_bias, *(float*)conv2d_1_output_scale, *(int8_t*)conv2d_1_output_zero_point, *(int8_t*)conv2d_1_input_zero_point, *(float*)conv2d_1_bias_scale);
MaxPool2d maxpool2d_1(9, 10, 10, 2, 2, 0);
Flatten flatten_0(225);
LinearReLU linear_0(29, 225, (int8_t*)linear_0_weight, (int32_t*)linear_0_bias, *(float*)linear_0_output_scale, *(int8_t*)linear_0_output_zero_point, *(int8_t*)linear_0_input_zero_point, *(float*)linear_0_bias_scale);
Linear linear_1(10, 29, (int8_t*)linear_1_weight, (int32_t*)linear_1_bias, *(float*)linear_1_output_scale, *(int8_t*)linear_1_output_zero_point, *(int8_t*)linear_1_input_zero_point, *(float*)linear_1_bias_scale);
Sequential uno_model(layers, LAYERS_LEN, workspace, MAX_OUTPUT_EVEN_SIZE);

Layer* layers[LAYERS_LEN] = {
    &conv2d_0,
    &maxpool2d_0,
    &conv2d_1,
    &maxpool2d_1,
    &flatten_0,
    &linear_0,
    &linear_1,
};
