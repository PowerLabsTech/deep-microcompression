#ifndef FLATTEN_H
#define FLATTEN_H

#include "layer.h"


// Buffer layout: [input_size: uint32_t]

class Flatten : public Layer {
public:
    using Layer::Layer;
    void forward(float* workspace_start, uint32_t workspace_size);
    uint32_t get_output_size(void);
};


class Flatten_SQ : public Layer_SQ {
public:
    using Layer_SQ::Layer_SQ;
    void forward(int8_t* workspace_start, uint32_t workspace_size);
    uint32_t get_output_size(void);
};


#endif // FLATTEN_H
