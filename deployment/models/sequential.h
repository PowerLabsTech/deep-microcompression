/**
 * @file sequential.h
 * @brief Header for Sequential neural network model container with support for:
 *      1. Both floating-point and quantized inference modes
 *      2. Double-buffering memory strategy
 *      3. Workspace optimization for constrained devices
 *
 * Sequential is buffer-based. Its Flash buffer stores the required workspace
 * size, layers_len, and the layer pointer array. The caller supplies
 * workspace_start and workspace_size at construction; predict() reads the
 * required size from Flash and validates before running.
 */

#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

#include <stdint.h>
#include "../layers/layer.h"

#define NONE 0
#define DYNAMIC 1
#define STATIC 2


// Buffer layout (float):
//   [workspace_size: uint32_t | layers_len: uint8_t | layer0: ptr | layer1: ptr | ...]
class Sequential {
    const uint8_t* buffer;
    float*         workspace_start;
    uint32_t       workspace_size;
public:
    Sequential(const uint8_t* buffer, float* workspace_start, uint32_t workspace_size);
    void predict(void);
    float get_output(uint32_t index);
    void set_input(uint32_t index, float value);
};


// Buffer layout (static quantization):
//   [workspace_size: uint32_t | layers_len: uint8_t | quantize_property: uint8_t | layer0: ptr | ...]
class Sequential_SQ {
    const uint8_t* buffer;
    int8_t*        workspace_start;
    uint32_t       workspace_size;
public:
    Sequential_SQ(const uint8_t* buffer, int8_t* workspace_start, uint32_t workspace_size);
    void predict(void);
    int8_t get_output(uint32_t index);
    void set_input(uint32_t index, int8_t value);
};

#endif // SEQUENTIAL_H
