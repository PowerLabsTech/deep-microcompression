#ifndef BATCHNORM_H
#define BATCHNORM_H

#include "layer.h"


#if !defined(QUANTIZATION_SCHEME) || QUANTIZATION_SCHEME != STATIC


class BatchNorm2d : public Layer{

public:
    void forward(float* workspace_start, uint32_t workspace_size);
    uint32_t get_output_size(void);
};


#else // QUANTIZATION_SCHEME

#endif // QUANTIZATION_SCHEME

#endif // BATCHNORM_H