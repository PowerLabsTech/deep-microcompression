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
 * @param workspace_even_layer_size Size of even layer workspace partition
 * 
 * @note Uses ping-pong strategy to alternate between workspace_even_layer
 *       and workspace_odd_layer for memory efficiency
 */
Sequential::Sequential(Layer **layers, uint8_t layers_len, float *workspace, 
                      uint32_t workspace_even_layer_size) {
    this->layers = layers;
    this->layers_len = layers_len;
    this->workspace_even_layer = workspace;
    this->workspace_odd_layer = workspace + workspace_even_layer_size;

    // Set model input/output buffers based on double-buffering strategy
    this->input = this->workspace_even_layer;
    this->output = (layers_len % 2 == DLAI_EVEN) ? this->workspace_even_layer 
                                               : this->workspace_odd_layer;   
}

/**
 * @brief Executes forward pass through all layers
 * 
 * Alternates between workspace buffers for each layer to minimize memory usage:
 * - Even layers write to odd workspace
 * - Odd layers write to even workspace
 */
void Sequential::predict(void) {
    for (uint8_t i = 0; i < this->layers_len; i++) {
        switch (i % 2) {
            case DLAI_EVEN:
                this->layers[i]->forward(this->workspace_even_layer, 
                                       this->workspace_odd_layer);
                break;
            default:
                this->layers[i]->forward(this->workspace_odd_layer, 
                                       this->workspace_even_layer);
                break;
        }
    }
}


#else // QUATIZATION_SCHEME

Sequential::Sequential(Layer **layers, uint8_t layers_len, int8_t *workspace, 
                      uint32_t workspace_even_layer_size) {
    this->layers = layers;
    this->layers_len = layers_len;
    this->workspace_even_layer = workspace;
    this->workspace_odd_layer = workspace + workspace_even_layer_size;

    // Set model input/output buffers based on ping-pong strategy
    this->input = this->workspace_even_layer;
    this->output = (layers_len % 2 == DLAI_EVEN) ? this->workspace_even_layer 
                                               : this->workspace_odd_layer;   
}

void Sequential::predict(void) {
    for (uint8_t i = 0; i < this->layers_len; i++) {
        switch (i % 2) {
            case DLAI_EVEN:
                this->layers[i]->forward(this->workspace_even_layer, 
                                       this->workspace_odd_layer);
                break;
            default:
                this->layers[i]->forward(this->workspace_odd_layer, 
                                       this->workspace_even_layer);
                break;
        }
    }
}

#endif // QUANTIZATION_SCHEME