/**
 * @file sequential.h
 * @brief Header for Sequential neural network model container with support for:
 *      1. Both floating-point and quantized inference modes
 *      2. Double-buffering memory strategy
 *      3. Workspace optimization for constrained devices
 * 
 * The class manages a sequence of neural network layers and their execution,
 * with compile-time selection between float and int8_t data types.
 */

#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

#include <stdint.h>
#include "layers/layer.h"

#define NONE 0
#define DYNAMIC 1
#define STATIC 2


/**
 * @brief Sequential container for floating-point neural network layers
 * 
 * Manages execution of a sequence of layers using alternating workspace buffers
 * to minimize memory usage during inference.
 */
class Sequential {
private:
    Layer** layers;                  ///< Array of layer pointers
    uint8_t layers_len;             ///< Number of layers in the model
    uint32_t workspace_size;        ///< Size of the pre-allocated memory

public:
    float* workspace_start;
    /**
     * @brief Constructs a floating-point sequential model
     * @param layers Array of layer pointers
     * @param layers_len Number of layers
     * @param workspace Pre-allocated workspace memory
     * @param workspace_size Size of the pre-allocated workspace memory
     */
    Sequential(Layer** layers, uint8_t layers_len, float* workspace_start, uint32_t workspace_size);

    /**
     * @brief Executes forward pass through all layers
     * 
     * Uses alternating buffers between layers to optimize memory usage
     */
    void predict(void);
    float get_output(uint32_t index);
    void set_input(uint32_t index, float value);
};


class Sequential_SQ {
private:
    Layer_SQ** layers;                  ///< Array of layer pointers
    uint8_t layers_len;             ///< Number of layers in the model
    uint32_t workspace_size;        ///< Size of the pre-allocated memory
    uint8_t quantize_property;

public:
    int8_t* workspace_start;                  ///< Pointer to quantized workspace buffer
    /**
     * @brief Constructs a quantized sequential model
     * @param layers Array of layer pointers
     * @param layers_len Number of layers
     * @param workspace Pre-allocated workspace memory
     * @param workspace_size Size of the pre-allocated workspace memory
     */
    Sequential_SQ(Layer_SQ** layers, uint8_t layers_len, int8_t* workspace_start, uint32_t workspace_size, uint8_t quantize_property);
    /**
     * @brief Executes forward pass through all quantized layers
     * 
     * Uses same alternating buffer strategy as floating-point version
     */
    void predict(void);
    int8_t get_output(uint32_t index);
    void set_input(uint32_t index, int8_t value);
};

#endif // SEQUENTIAL_H