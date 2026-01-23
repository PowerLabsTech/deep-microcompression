#include "fused_layers.h"


#if !defined(QUANTIZATION_SCHEME) || QUANTIZATION_SCHEME == NONE

float* LinearReLU::forward(float* input, float* workspace_start, uint32_t workspace_size) {
    // Getting the output start address with the input size as offset
    float* output = input == workspace_start ? workspace_start + workspace_size - this->output_size : workspace_start;

    float output_temp;
    for (uint16_t j = 0; j < this->output_size; j++) {
        output_temp = this->bias ? par_read_float(this->bias, j) : 0;
        // Matrix-vector multiplication
        for (uint16_t i = 0; i < this->input_size; i++) {
            output_temp += act_read_float(input, i) * par_read_float(this->weight, (j * this->input_size) + i);
        }
        act_write_float(output, j, (relu(output_temp)));
    }

    return output;
}



float* Conv2dReLU::forward(float* input, float* workspace_start, uint32_t workspace_size) {
    // Getting the output start address with the input size as offset
    float* output = input == workspace_start ? workspace_start + workspace_size - (
        this->output_channel_size * this->output_row_size * this->output_col_size
    ) : workspace_start;

    float output_temp;

    uint16_t input_channel_per_group = this->input_channel_size / this->groups;
    uint16_t output_channel_per_group = this->output_channel_size / this->groups;

    uint16_t n, k;

    uint16_t padded_row_size = this->input_row_size + this->padding.padding_top + this->padding.padding_bottom;
    uint16_t padded_col_size = this->input_col_size + this->padding.padding_left + this->padding.padding_right;

    pad_input(input, this->padding, input_channel_size, input_row_size, input_col_size, padded_row_size, padded_col_size);

    for (uint8_t g = 0; g < this->groups; g++){
        // Output channel loop
        for (uint16_t c_out = 0; c_out < output_channel_per_group; c_out++) {
            n = g * output_channel_per_group + c_out;
            // Output spatial dimensions loops
            for (uint16_t m = 0; m < this->output_row_size; m++) {
                for (uint16_t l = 0; l < this->output_col_size; l++) {
                    output_temp = this->bias ? par_read_float(this->bias, n) : 0;

                    for (uint16_t c_in = 0; c_in < input_channel_per_group; c_in++) {
                        k = g * input_channel_per_group + c_in;
                        for (uint8_t j = 0; j < this->kernel_row_size; j++) {
                            for (uint8_t i = 0; i < this->kernel_col_size; i++) {

                                // Convolution operation
                                output_temp += act_read_float(
                                    input, 
                                    (k * padded_row_size * padded_col_size) +
                                    ((j + m * this->stride_row) * padded_col_size) + 
                                    (i + l * this->stride_col)
                                ) * par_read_float(
                                    this->weight, 
                                    (n * input_channel_per_group * this->kernel_row_size * this->kernel_col_size) +
                                    (c_in * this->kernel_row_size * kernel_col_size) + 
                                    (j * this->kernel_col_size) + 
                                    i
                                );
                            }
                        }
                    }
                    act_write_float(output, 
                        (n * this->output_row_size * this->output_col_size) + 
                        (m * this->output_col_size) + 
                        l,
                        relu(output_temp)
                    );
                }
            }
        }
    }

    return output;
}


float* LinearReLU6::forward(float* input, float* workspace_start, uint32_t workspace_size) {
    // Getting the output start address with the input size as offset
    float* output = input == workspace_start ? workspace_start + workspace_size - this->output_size : workspace_start;

    float output_temp;
    for (uint16_t j = 0; j < this->output_size; j++) {
        output_temp = this->bias ? par_read_float(this->bias, j) : 0;
        // Matrix-vector multiplication
        for (uint16_t i = 0; i < this->input_size; i++) {
            output_temp += act_read_float(input, i) * par_read_float(this->weight, (j * this->input_size) + i);
        }
        act_write_float(output, j, relu6(output_temp));
    }

    return output;
}



float* Conv2dReLU6::forward(float* input, float* workspace_start, uint32_t workspace_size) {
    // Getting the output start address with the input size as offset
    float* output = input == workspace_start ? workspace_start + workspace_size - (
        this->output_channel_size * this->output_row_size * this->output_col_size
    ) : workspace_start;

    float output_temp;

    uint16_t input_channel_per_group = this->input_channel_size / this->groups;
    uint16_t output_channel_per_group = this->output_channel_size / this->groups;

    uint16_t n, k;

    uint16_t padded_row_size = this->input_row_size + this->padding.padding_top + this->padding.padding_bottom;
    uint16_t padded_col_size = this->input_col_size + this->padding.padding_left + this->padding.padding_right;

    pad_input(input, this->padding, input_channel_size, input_row_size, input_col_size, padded_row_size, padded_col_size);

    for (uint8_t g = 0; g < this->groups; g++){
        // Output channel loop
        for (uint16_t c_out = 0; c_out < output_channel_per_group; c_out++) {
            n = g * output_channel_per_group + c_out;
            // Output spatial dimensions loops
            for (uint16_t m = 0; m < this->output_row_size; m++) {
                for (uint16_t l = 0; l < this->output_col_size; l++) {
                    output_temp = this->bias ? par_read_float(this->bias, n) : 0;

                    for (uint16_t c_in = 0; c_in < input_channel_per_group; c_in++) {
                        k = g * input_channel_per_group + c_in;
                        for (uint8_t j = 0; j < this->kernel_row_size; j++) {
                            for (uint8_t i = 0; i < this->kernel_col_size; i++) {

                                // Convolution operation
                                output_temp += act_read_float(
                                    input, 
                                    (k * padded_row_size * padded_col_size) +
                                    ((j + m * this->stride_row) * padded_col_size) + 
                                    (i + l * this->stride_col)
                                ) * par_read_float(
                                    this->weight, 
                                    (n * input_channel_per_group * this->kernel_row_size * this->kernel_col_size) +
                                    (c_in * this->kernel_row_size * kernel_col_size) + 
                                    (j * this->kernel_col_size) + 
                                    i
                                );
                            }
                        }
                    }
                    act_write_float(output, 
                        (n * this->output_row_size * this->output_col_size) + 
                        (m * this->output_col_size) + 
                        l,
                        relu6(output_temp)
                    );
                }
            }
        }
    }

    return output;
}


#elif QUANTIZATION_SCHEME == DYNAMIC 


float* LinearReLU::forward(float* input, float* workspace_start, uint32_t workspace_size) {
    // Getting the output start address with the input size as offset
    float* output = input == workspace_start ? workspace_start + workspace_size - this->output_size : workspace_start;

    float output_temp;
    for (uint16_t j = 0; j < this->output_size; j++) {
        output_temp = 0;
        // Matrix-vector multiplication
        for (uint16_t i = 0; i < this->input_size; i++) {
            output_temp += act_read_float(input, i) * par_read_packed_intb(this->weight, (j * this->input_size) + i);
        }
        act_write_float(output,
            j,
            (this->bias ? 
            relu((output_temp * this->weight_scale + par_read_float(this->bias, j))) :
            relu((output_temp * this->weight_scale)))
        );
    }

    return output;
}



float* Conv2dReLU::forward(float* input, float* workspace_start, uint32_t workspace_size) {
    // Getting the output start address with the input size as offset
    float* output = input == workspace_start ? workspace_start + workspace_size - (
        this->output_channel_size * this->output_row_size * this->output_col_size
    ) : workspace_start;

    float output_temp;

    uint16_t input_channel_per_group = this->input_channel_size / this->groups;
    uint16_t output_channel_per_group = this->output_channel_size / this->groups;

    uint16_t n, k;

    uint16_t padded_row_size = this->input_row_size + this->padding.padding_top + this->padding.padding_bottom;
    uint16_t padded_col_size = this->input_col_size + this->padding.padding_left + this->padding.padding_right;

    pad_input(input, this->padding, input_channel_size, input_row_size, input_col_size, padded_row_size, padded_col_size);

    for (uint8_t g = 0; g < this->groups; g++){
        // Output channel loop
        for (uint16_t c_out = 0; c_out < output_channel_per_group; c_out++) {
            n = g * output_channel_per_group + c_out;
            // Output spatial dimensions loops
            for (uint16_t m = 0; m < this->output_row_size; m++) {
                for (uint16_t l = 0; l < this->output_col_size; l++) {
                    
                    output_temp = 0;

                    for (uint16_t c_in = 0; c_in < input_channel_per_group; c_in++) {
                        k = g * input_channel_per_group + c_in;
                        for (uint8_t j = 0; j < this->kernel_row_size; j++) {
                            for (uint8_t i = 0; i < this->kernel_col_size; i++) {

                                // Convolution operation
                                output_temp += act_read_float(
                                    input, 
                                    (k * padded_row_size * padded_col_size) +
                                    ((j + m * this->stride_row) * padded_col_size) + 
                                    (i + l * this->stride_col)
                                ) * 
                                par_read_packed_intb(
                                    this->weight, 
                                    (n * input_channel_per_group * this->kernel_row_size * this->kernel_col_size) +
                                    (c_in * this->kernel_row_size * kernel_col_size) + 
                                    (j * this->kernel_col_size) + i
                                );
                            }
                        }
                    }
                    act_write_float(output, 
                        (n * this->output_row_size * this->output_col_size) + 
                        (m * this->output_col_size) + 
                        l,
                        (this->bias ? 
                        relu((output_temp * this->weight_scale + par_read_float(this->bias, n))) :
                        relu((output_temp * this->weight_scale)))
                    );
                }
            }
        }
    }

    return output;
}


float* LinearReLU6::forward(float* input, float* workspace_start, uint32_t workspace_size) {
    // Getting the output start address with the input size as offset
    float* output = input == workspace_start ? workspace_start + workspace_size - this->output_size : workspace_start;

    float output_temp;
    for (uint16_t j = 0; j < this->output_size; j++) {
        output_temp = 0;
        // Matrix-vector multiplication
        for (uint16_t i = 0; i < this->input_size; i++) {
            output_temp += act_read_float(input, i) * par_read_packed_intb(this->weight, (j * this->input_size) + i);
        }
        act_write_float(output,
            j,
            (this->bias ? 
            relu6((output_temp * this->weight_scale + par_read_float(this->bias, j))) :
            relu6((output_temp * this->weight_scale)))
        );
    }

    return output;
}



float* Conv2dReLU6::forward(float* input, float* workspace_start, uint32_t workspace_size) {
    // Getting the output start address with the input size as offset
    float* output = input == workspace_start ? workspace_start + workspace_size - (
        this->output_channel_size * this->output_row_size * this->output_col_size
    ) : workspace_start;

    float output_temp;

    uint16_t input_channel_per_group = this->input_channel_size / this->groups;
    uint16_t output_channel_per_group = this->output_channel_size / this->groups;

    uint16_t n, k;

    uint16_t padded_row_size = this->input_row_size + this->padding.padding_top + this->padding.padding_bottom;
    uint16_t padded_col_size = this->input_col_size + this->padding.padding_left + this->padding.padding_right;

    pad_input(input, this->padding, input_channel_size, input_row_size, input_col_size, padded_row_size, padded_col_size);

    for (uint8_t g = 0; g < this->groups; g++){
        // Output channel loop
        for (uint16_t c_out = 0; c_out < output_channel_per_group; c_out++) {
            n = g * output_channel_per_group + c_out;
            // Output spatial dimensions loops
            for (uint16_t m = 0; m < this->output_row_size; m++) {
                for (uint16_t l = 0; l < this->output_col_size; l++) {
                    
                    output_temp = 0;

                    for (uint16_t c_in = 0; c_in < input_channel_per_group; c_in++) {
                        k = g * input_channel_per_group + c_in;
                        for (uint8_t j = 0; j < this->kernel_row_size; j++) {
                            for (uint8_t i = 0; i < this->kernel_col_size; i++) {

                                // Convolution operation
                                output_temp += act_read_float(
                                    input, 
                                    (k * padded_row_size * padded_col_size) +
                                    ((j + m * this->stride_row) * padded_col_size) + 
                                    (i + l * this->stride_col)
                                ) * 
                                par_read_packed_intb(
                                    this->weight, 
                                    (n * input_channel_per_group * this->kernel_row_size * this->kernel_col_size) +
                                    (c_in * this->kernel_row_size * kernel_col_size) + 
                                    (j * this->kernel_col_size) + i
                                );
                            }
                        }
                    }
                    act_write_float(output, 
                        (n * this->output_row_size * this->output_col_size) + 
                        (m * this->output_col_size) + 
                        l,
                        (this->bias ? 
                        relu6((output_temp * this->weight_scale + par_read_float(this->bias, n))) :
                        relu6((output_temp * this->weight_scale)))
                    );
                }
            }
        }
    }

    return output;
}

#elif QUANTIZATION_SCHEME == STATIC



int8_t* LinearReLU::forward(int8_t* input, int8_t* workspace_start, uint32_t workspace_size) {
    // Getting the output start address with the input size as offset
    int8_t* output = input == workspace_start ? workspace_start + workspace_size - (uint32_t)ceil(this->output_size / DATA_PER_BYTE) : workspace_start;

    int32_t output_temp;

    for (uint16_t j = 0; j < this->output_size; j++) {
        output_temp = this->bias ? par_read_int32(this->bias, j) : 0;

        for (uint16_t i = 0; i < this->input_size; i++) {
            output_temp += ((int32_t)act_read_packed_intb(input, i) - this->input_zero_point) *
                            par_read_packed_intb(this->weight, (j * this->input_size) + i);
    
        }

        output_temp = relu(output_temp);
        
        // Requantize to 8-bit
        output_temp = roundf(output_temp * this->bias_scale / this->output_scale);
        output_temp += this->output_zero_point;
        output_temp = clampb(output_temp);
        act_write_packed_intb(output, j, output_temp);

        // set_packed_value(output, j, output_temp);
    }

    return output;
}

int8_t* Conv2dReLU::forward(int8_t* input, int8_t* workspace_start, uint32_t workspace_size) {
    // Getting the output start address with the input size as offset
    int8_t* output = input == workspace_start ? workspace_start + workspace_size - (uint32_t)ceil(
        (this->output_channel_size * this->output_row_size * this->output_col_size) / DATA_PER_BYTE
    ) : workspace_start;

    int32_t output_temp;

    uint16_t input_channel_per_group = this->input_channel_size / this->groups;
    uint16_t output_channel_per_group = this->output_channel_size / this->groups;

    uint16_t n, k;

    uint16_t padded_row_size = this->input_row_size + this->padding.padding_top + this->padding.padding_bottom;
    uint16_t padded_col_size = this->input_col_size + this->padding.padding_left + this->padding.padding_right;

    pad_input(input, this->input_zero_point, this->padding, input_channel_size, input_row_size, input_col_size, padded_row_size, padded_col_size);

    for (uint8_t g = 0; g < this->groups; g++){
        // Output channel loop
        for (uint16_t c_out = 0; c_out < output_channel_per_group; c_out++) {
            n = g * output_channel_per_group + c_out;
            // Output spatial dimensions loops
            for (uint16_t m = 0; m < this->output_row_size; m++) {
                for (uint16_t l = 0; l < this->output_col_size; l++) {
                    
                    // Calculate output index
                    output_temp = this->bias ? par_read_int32(this->bias, n) : 0;

                    for (uint16_t c_in = 0; c_in < input_channel_per_group; c_in++) {
                        k = g * input_channel_per_group + c_in;
                        for (uint8_t j = 0; j < this->kernel_row_size; j++) {
                            for (uint8_t i = 0; i < this->kernel_col_size; i++) {

                                output_temp += ((int32_t)act_read_packed_intb(
                                    input,
                                    (k * padded_row_size * padded_col_size) +
                                    ((j + m * this->stride_row) * padded_col_size) + 
                                    (i + l * this->stride_col)) - this->input_zero_point) *
                                    par_read_packed_intb(
                                        this->weight,
                                        (n * input_channel_per_group * this->kernel_row_size * this->kernel_col_size) +
                                        (c_in * this->kernel_row_size * kernel_col_size) + 
                                        (j * this->kernel_col_size) + i
                                    );
                            }
                        }
                    }

                    output_temp = relu(output_temp);

                    // Apply bias, scaling and clamping
                    output_temp = roundf(output_temp * this->bias_scale / this->output_scale);
                    output_temp += this->output_zero_point;
                    output_temp = clampb(output_temp);
                    
                    act_write_packed_intb(
                        output,
                        (n * this->output_row_size * this->output_col_size) + 
                        (m * this->output_col_size) + 
                        l, 
                        output_temp
                    );
                }
            }
        }
    }

    return output;
}





int8_t* LinearReLU6::forward(int8_t* input, int8_t* workspace_start, uint32_t workspace_size) {
    // Getting the output start address with the input size as offset
    int8_t* output = input == workspace_start ? workspace_start + workspace_size - (uint32_t)ceil(this->output_size / DATA_PER_BYTE) : workspace_start;

    int32_t output_temp;
    int32_t six_point = (int32_t)((float)6. / this->bias_scale);

    for (uint16_t j = 0; j < this->output_size; j++) {
        output_temp = this->bias ? par_read_int32(this->bias, j) : 0;

        for (uint16_t i = 0; i < this->input_size; i++) {
            output_temp += ((int32_t)act_read_packed_intb(input, i) - this->input_zero_point) *
                            par_read_packed_intb(this->weight, (j * this->input_size) + i);
    
        }

        output_temp = relux(output_temp, six_point);
        
        // Requantize to 8-bit
        output_temp = roundf(output_temp * this->bias_scale / this->output_scale);
        output_temp += this->output_zero_point;
        output_temp = clampb(output_temp);
        
        act_write_packed_intb(output, j, output_temp);
    }

    return output;
}

int8_t* Conv2dReLU6::forward(int8_t* input, int8_t* workspace_start, uint32_t workspace_size) {
    // Getting the output start address with the input size as offset
    int8_t* output = input == workspace_start ? workspace_start + workspace_size - (uint32_t)ceil(
        (this->output_channel_size * this->output_row_size * this->output_col_size) / DATA_PER_BYTE
    ) : workspace_start;

    int32_t output_temp;
    int32_t six_point = (int32_t)((float)6. / this->bias_scale);

    uint16_t input_channel_per_group = this->input_channel_size / this->groups;
    uint16_t output_channel_per_group = this->output_channel_size / this->groups;

    uint16_t n, k;

    uint16_t padded_row_size = this->input_row_size + this->padding.padding_top + this->padding.padding_bottom;
    uint16_t padded_col_size = this->input_col_size + this->padding.padding_left + this->padding.padding_right;

    pad_input(input, this->input_zero_point, this->padding, input_channel_size, input_row_size, input_col_size, padded_row_size, padded_col_size);

    for (uint8_t g = 0; g < this->groups; g++){
        // Output channel loop
        for (uint16_t c_out = 0; c_out < output_channel_per_group; c_out++) {
            n = g * output_channel_per_group + c_out;
            // Output spatial dimensions loops
            for (uint16_t m = 0; m < this->output_row_size; m++) {
                for (uint16_t l = 0; l < this->output_col_size; l++) {
                    
                    output_temp = this->bias ? par_read_int32(this->bias, n) : 0;

                    for (uint16_t c_in = 0; c_in < input_channel_per_group; c_in++) {
                        k = g * input_channel_per_group + c_in;
                        for (uint8_t j = 0; j < this->kernel_row_size; j++) {
                            for (uint8_t i = 0; i < this->kernel_col_size; i++) {                                // Convolution operation
                                                         
                                output_temp += ((int32_t)act_read_packed_intb(
                                    input,
                                    (k * padded_row_size * padded_col_size) +
                                    ((j + m * this->stride_row) * padded_col_size) + 
                                    (i + l * this->stride_col)) - this->input_zero_point) *
                                    par_read_packed_intb(
                                        this->weight,
                                        (n * input_channel_per_group * this->kernel_row_size * this->kernel_col_size) +
                                        (c_in * this->kernel_row_size * kernel_col_size) + 
                                        (j * this->kernel_col_size) + i
                                    );
                            }
                        }
                    }

                    output_temp = relux(output_temp, six_point);

                    // Apply bias, scaling and clamping
                    output_temp = roundf(output_temp * this->bias_scale / this->output_scale);
                    output_temp += this->output_zero_point;
                    output_temp = clampb(output_temp);
                    
                    act_write_packed_intb(
                        output,
                        (n * this->output_row_size * this->output_col_size) + 
                        (m * this->output_col_size) + 
                        l, 
                        output_temp
                    );
                }
            }
        }
    }

    return output;
}



#endif // QUANTIZATION_SCHEME

