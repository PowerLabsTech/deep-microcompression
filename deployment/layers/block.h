#ifndef BLOCK_LAYER_H
#define BLOCK_LAYER_H

#include "layer.h"

/**
 * @brief Sequential sub-network of floating-point layers sharing one workspace.
 *
 * Useful for grouping layers inside a Branch (e.g. a residual arm).
 * All layers are executed in order; get_output_size() returns the last layer's output size.
 */
class Block : public Layer {
private:
    Layer** layers;
    uint8_t num_layers;

public:
    Block(Layer** layers, uint8_t num_layers);

    void forward(float* workspace_start, uint32_t workspace_size);
    uint32_t get_output_size(void);
};

/**
 * @brief Sequential sub-network of statically-quantized layers sharing one workspace.
 */
class Block_SQ : public Layer_SQ {
private:
    Layer_SQ** layers;
    uint8_t num_layers;

public:
    Block_SQ(Layer_SQ** layers, uint8_t num_layers);

    void forward(int8_t* workspace_start, uint32_t workspace_size);
    uint32_t get_output_size(void);
};



#endif // BLOCK_LAYER_H