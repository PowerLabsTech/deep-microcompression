#ifndef BLOCK_LAYER_H
#define BLOCK_LAYER_H

#include "layer.h"

/**
 * @brief Sequential sub-network of floating-point layers sharing one workspace.
 *
 * Buffer layout: [num_layers: uint32_t | layer0: ptr | layer1: ptr | ...]
 * All entries are DMC_PTR_SIZE bytes for layer pointers.
 */
class Block : public Layer {
public:
    using Layer::Layer;
    void forward(float* workspace_start, uint32_t workspace_size);
    uint32_t get_output_size(void);
};

/**
 * @brief Sequential sub-network of statically-quantized layers sharing one workspace.
 *
 * Buffer layout: [num_layers: uint32_t | layer0: ptr | layer1: ptr | ...]
 */
class Block_SQ : public Layer_SQ {
public:
    using Layer_SQ::Layer_SQ;
    void forward(int8_t* workspace_start, uint32_t workspace_size);
    uint32_t get_output_size(void);
};



#endif // BLOCK_LAYER_H
