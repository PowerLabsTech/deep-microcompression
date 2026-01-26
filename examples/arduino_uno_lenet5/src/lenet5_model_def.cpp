#include "lenet5_model.h"

int8_t workspace[WORKSPACE_SIZE];

ConstantPad2d_SQ constantpad2d_0(1, 28, 28, *(int8_t*)constantpad2d_0_input_value_point, {2, 2, 2, 2}, A4);
Conv2dReLU_SQ conv2d_0(1, 32, 32, 2, 5, 5, 1, 1, 1, (int8_t*)conv2d_0_weight, (int32_t*)conv2d_0_bias, *(float*)conv2d_0_output_scale, *(int8_t*)conv2d_0_output_zero_point, *(int8_t*)conv2d_0_input_zero_point, (float*)conv2d_0_bias_scale, PER_TENSOR_A4_P4);
MaxPool2d_SQ maxpool2d_0(2, 28, 28, 2, 2, 0, A4);
Conv2dReLU_SQ conv2d_1(2, 14, 14, 5, 5, 5, 1, 1, 1, (int8_t*)conv2d_1_weight, (int32_t*)conv2d_1_bias, *(float*)conv2d_1_output_scale, *(int8_t*)conv2d_1_output_zero_point, *(int8_t*)conv2d_1_input_zero_point, (float*)conv2d_1_bias_scale, PER_TENSOR_A4_P4);
MaxPool2d_SQ maxpool2d_1(5, 10, 10, 2, 2, 0, A4);
Flatten_SQ flatten_0(125, A4);
LinearReLU_SQ linear_0(26, 125, (int8_t*)linear_0_weight, (int32_t*)linear_0_bias, *(float*)linear_0_output_scale, *(int8_t*)linear_0_output_zero_point, *(int8_t*)linear_0_input_zero_point, (float*)linear_0_bias_scale, PER_TENSOR_A4_P4);
Linear_SQ linear_1(10, 26, (int8_t*)linear_1_weight, (int32_t*)linear_1_bias, *(float*)linear_1_output_scale, *(int8_t*)linear_1_output_zero_point, *(int8_t*)linear_1_input_zero_point, (float*)linear_1_bias_scale, PER_TENSOR_A4_P4);
Sequential_SQ lenet5_model(layers, LAYERS_LEN, workspace, WORKSPACE_SIZE, A4);

Layer_SQ* layers[LAYERS_LEN] = {
    &constantpad2d_0,
    &conv2d_0,
    &maxpool2d_0,
    &conv2d_1,
    &maxpool2d_1,
    &flatten_0,
    &linear_0,
    &linear_1,
};
