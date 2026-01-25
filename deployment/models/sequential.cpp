/**
 * @file sequential.cpp
 * @brief Implementation of sequential neural network model with:
 *      1. Support for both floating-point and quantized inference
 *      2. Double-buffering memory strategy for efficient layer processing
 *      3. Workspace optimization for memory-constrained devices
 * 
 * The implementation uses compile-time switching between floating-point
 * and quantized versions via STATIC_QUANTIZATION_PER_TENSOR define
 */

#include "sequential.h"

/**
 * @brief Constructs a floating-point sequential model
 * @param layers Array of layer pointers
 * @param layers_len Number of layers in model
 * @param workspace Pre-allocated workspace buffer (float)
 * @param workspace_size Size of the pre-allocated workspace memory
 * 
 * @note Uses ping-pong strategy to alternate between workspace_even_layer
 *       and workspace_odd_layer for memory efficiency
 */
Sequential::Sequential(Layer **layers, uint8_t layers_len, float *workspace, uint32_t workspace_size) {
    this->layers = layers;
    this->layers_len = layers_len;
    this->input = workspace;
    this->workspace_size = workspace_size;
}

/**
 * @brief Executes forward pass through all layers
 * 
 * Alternates between workspace buffers for each layer to minimize memory usage:
 * - Even layers write to odd workspace
 * - Odd layers write to even workspace
 */
void Sequential::predict(void) {
    float* current_input = this->input;

    for (uint8_t i = 0; i < this->layers_len; i++) {
        current_input = this->layers[i]->forward(current_input, this->input, workspace_size);
    }
    this->output = current_input;
}

float Sequential::get_output(uint32_t index) {
    return activation_read_float(this->output, index);
}

void Sequential::set_input(uint32_t index, float value) {
    activation_write_float(this->input, index, value);
}

Sequential_SQ::Sequential_SQ(Layer_SQ **layers, uint8_t layers_len, int8_t *workspace, uint32_t workspace_size, uint8_t quantize_property) {
    this->layers = layers;
    this->layers_len = layers_len;
    this->input = workspace;
    this->workspace_size = workspace_size;
    this->quantize_property = quantize_property;
}

void Sequential_SQ::predict(void) {
    int8_t* current_input = this->input;

    for (uint8_t i = 0; i < this->layers_len; i++) {
        current_input = this->layers[i]->forward(current_input, this->input, workspace_size);
    }
    this->output = current_input;
}

int8_t Sequential_SQ::get_output(uint32_t index) {
    int8_t (*activation_read_packed_intb) (int8_t*, uint32_t);
    get_activation_read_packed_intb(this->quantize_property, &activation_read_packed_intb);
    return activation_read_packed_intb(this->output, index);
}

void Sequential_SQ::set_input(uint32_t index, int8_t value) {
    void (*activation_write_packed_intb) (int8_t*, uint32_t, int8_t);
    get_activation_write_packed_intb(this->quantize_property, &activation_write_packed_intb);
    activation_write_packed_intb(this->input, index, value);
}
