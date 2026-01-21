/**
 * @file linear.cpp
 * @brief Implementation of Linear (fully-connected) layer with support for:
 *       1. Non-quantized models (float)
 *       2. Dynamic quantized models per tensor (float input + quantized weights)
 *       3. Static quantized models per tensor (all quantized)
 */

#include "linear.h"


#if !defined(QUANTIZATION_SCHEME) || QUANTIZATION_SCHEME == NONE


/**
 * @brief Constructor for floating-point Linear layer
 * @param output_size Number of output neurons
 * @param input_size Number of input features
 * @param weight Pointer to weight matrix (row-major, shape [output_size, input_size])
 * @param bias Pointer to bias vector (size [output_size])
 */
Linear::Linear(uint16_t output_size, uint16_t input_size, 
              const float* weight, const float* bias) {
    this->output_size = output_size;
    this->input_size = input_size;
    this->weight = weight;
    this->bias = bias;
}

/**
 * @brief Forward pass for floating-point Linear layer
 * @param input Pointer to input tensor (float)
 * @param output Pointer to output tensor (float)
 * 
 * Computes: output = input * weight^T + bias
 */
void Linear::forward(float* input, float* output) {
    float output_temp;
    for (uint16_t j = 0; j < this->output_size; j++) {
        output_temp = this->bias ? par_read_float(this->bias, j) : 0;
        // Matrix-vector multiplication
        for (uint16_t i = 0; i < this->input_size; i++) {
            output_temp += act_read_float(input, i) * par_read_float(this->weight, (j * this->input_size) + i);
        }
        act_write_float(output, j, output_temp);
    }
}



#elif QUANTIZATION_SCHEME == DYNAMIC

#if QUANTIZATION_GRANULARITY == PER_TENSOR

/**
 * @brief Constructor for dynamically quantized Linear layer
 * @param output_size Number of output neurons
 * @param input_size Number of input features
 * @param weight Pointer to quantized weight matrix (int8_t)
 * @param weight_scale Scaling factor for weights
 * @param bias Pointer to floating-point bias vector
 */
Linear::Linear(uint16_t output_size, uint16_t input_size,
              const int8_t* weight, const float* bias,
              float weight_scale) {
    this->output_size = output_size;
    this->input_size = input_size;
    this->weight = weight;
    this->bias = bias;
    this->weight_scale = weight_scale;
}

/**
 * @brief Forward pass for dynamically quantized Linear layer
 * @param input Pointer to floating-point input tensor
 * @param output Pointer to floating-point output tensor
 * 
 * Computes: output = input * dequant(weight)^T + bias
 */
void Linear::forward(float* input, float* output) {
    float output_temp;

    for (uint16_t j = 0; j < this->output_size; j++) {
        output_temp = 0;
        for (uint16_t i = 0; i < this->input_size; i++) {
            output_temp += act_read_float(input, i) * par_read_packed_intb(this->weight, (j * this->input_size) + i);
        }
        act_write_float(output,
            j,
            (this->bias ? 
            (output_temp * this->weight_scale + par_read_float(this->bias, j)) :
            (output_temp * this->weight_scale))
        );
    }
}

#endif // QUANTIZATION_GRANULARITY


#elif QUANTIZATION_SCHEME == STATIC

#if QUANTIZATION_GRANULARITY == PER_TENSOR


Linear::Linear(uint16_t output_size, uint16_t input_size, const int8_t* weight, const int32_t* bias,
          float output_scale, int8_t output_zero_point, int8_t input_zero_point,  float bias_scale) {

    this->output_size = output_size;
    this->input_size = input_size;

    this->weight = weight;
    this->bias = bias;

    this->output_scale = output_scale;
    this->output_zero_point = output_zero_point;
    this->input_zero_point = input_zero_point;
    
    this->bias_scale = bias_scale;
}

/**
 * @brief Forward pass for statically quantized Linear layer
 * @param input Pointer to quantized input tensor (int8_t)
 * @param output Pointer to quantized output tensor (int8_t)
 * 
 * Computes quantized output with proper scaling and zero-point adjustments
 */
void Linear::forward(int8_t* input, int8_t* output) {
    int32_t output_temp;

    for (uint16_t j = 0; j < this->output_size; j++) {
        output_temp = this->bias ? par_read_int32(this->bias, j) : 0;

        for (uint16_t i = 0; i < this->input_size; i++) {
            output_temp += ((int32_t)act_read_packed_intb(input, i) - this->input_zero_point) *
                            par_read_packed_intb(this->weight, (j * this->input_size) + i);
        }
        
        // Requantize to 8-bit
        output_temp = roundf(output_temp * this->bias_scale / this->output_scale);
        output_temp += this->output_zero_point;
        output_temp = clampb(output_temp);
        
        act_write_packed_intb(output, j, output_temp);
    }
}
#endif // QUANTIZATION_GRANULARITY

#endif // QUANTIZATION_SCHEME
