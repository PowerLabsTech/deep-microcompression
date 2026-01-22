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

#if !defined(QUANTIZATION_SCHEME) || QUANTIZATION_SCHEME != STATIC

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
float* Sequential::predict(void) {
    float* current_input = this->input;

    for (uint8_t i = 0; i < this->layers_len; i++) {
        current_input = this->layers[i]->forward(current_input, this->input, workspace_size);
        // switch (i % 2) {
        //     case DLAI_EVEN:
        //         next_input = this->layers[i]->forward(this->input, nullptr);
        //         break;
        //     default:
        //         next_input = this->layers[i]->forward(next_input, this->input);
        //         break;
        // }
    }
    return current_input;
}


#else // QUATIZATION_SCHEME

Sequential::Sequential(Layer **layers, uint8_t layers_len, int8_t *workspace), uint32_t workspace_size {
    this->layers = layers;
    this->layers_len = layers_len;
    this->input = workspace;
    this->workspace_size = workspace_size;
}

int8_t* Sequential::predict(void) {
    int8_t* current_input = this->input;

    for (uint8_t i = 0; i < this->layers_len; i++) {
        current_input = this->layers[i]->forward(current_input, this->input, workspace_size);
        // switch (i % 2) {
        //     case DLAI_EVEN:
        //         next_input = this->layers[i]->forward(input=this->input, start=nullptr);
        //         break;
        //     default:
        //         next_input = this->layers[i]->forward(input=next_input, start=this->input);
        //         break;
        // }
    }
    return current_input;
}

#endif // QUANTIZATION_SCHEME