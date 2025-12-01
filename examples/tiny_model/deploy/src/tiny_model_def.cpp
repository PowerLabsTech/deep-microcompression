#include "tiny_model.h"

int8_t workspace[MAX_OUTPUT_EVEN_SIZE + MAX_OUTPUT_ODD_SIZE];

Conv2d conv2d_0(1, 28, 28, 2, 3, 3, 1, 1, {1, 1, 1, 1}, 1, (int8_t*)conv2d_0_weight, (int32_t*)conv2d_0_bias, *(float*)conv2d_0_output_scale, *(int8_t*)conv2d_0_output_zero_point, *(int8_t*)conv2d_0_input_zero_point, *(float*)conv2d_0_bias_scale);
ReLU relu_0(1568, *(int8_t*)relu_0_input_zero_point);
Flatten flatten_0(1568);
Linear linear_0(10, 1568, (int8_t*)linear_0_weight, (int32_t*)linear_0_bias, *(float*)linear_0_output_scale, *(int8_t*)linear_0_output_zero_point, *(int8_t*)linear_0_input_zero_point, *(float*)linear_0_bias_scale);
Sequential tiny_model(layers, LAYERS_LEN, workspace, MAX_OUTPUT_EVEN_SIZE);

Layer* layers[LAYERS_LEN] = {
    &conv2d_0,
    &relu_0,
    &flatten_0,
    &linear_0,
};
