/**
 * @file layer.cpp
 * @brief Base layer implementation with support for:
 *      1. Non-quantized models (float)
 *      2. Static quantized models per tensor (int8_t)
 */

#include "layer.h"

Layer::Layer(uint8_t* buffer) {
    this->buffer = buffer;
}


void Layer::forward(float* workspace_start, uint32_t workspace_size) {
    // Intentionally empty - to be implemented by derived classes
}


uint32_t Layer::get_output_size(void) {
    return 0;
}


Layer_SQ::Layer_SQ(uint8_t* buffer) {
    this->buffer = buffer;
}

void Layer_SQ::forward(int8_t* workspace_start, uint32_t workspace_size) {
    // Intentionally empty - to be implemented by derived classes
}


uint32_t Layer_SQ::get_output_size(void) {
    return 0;
}

