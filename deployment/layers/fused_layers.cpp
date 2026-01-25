#include "fused_layers.h"

float* LinearReLU::forward(float* input, float* workspace_start, uint32_t workspace_size) {
    // Getting the output start address with the input size as offset
    float* output = input == workspace_start ? workspace_start + workspace_size - this->output_size : workspace_start;

    float output_temp;
    for (uint16_t j = 0; j < this->output_size; j++) {
        output_temp = this->bias ? parameter_read_float(this->bias, j) : 0;
        // Matrix-vector multiplication
        for (uint16_t i = 0; i < this->input_size; i++) {
            output_temp += activation_read_float(input, i) * parameter_read_float(this->weight, (j * this->input_size) + i);
        }
        activation_write_float(output, j, (relu(output_temp)));
    }

    return output;
}


float* LinearReLU6::forward(float* input, float* workspace_start, uint32_t workspace_size) {
    // Getting the output start address with the input size as offset
    float* output = input == workspace_start ? workspace_start + workspace_size - this->output_size : workspace_start;

    float output_temp;
    for (uint16_t j = 0; j < this->output_size; j++) {
        output_temp = this->bias ? parameter_read_float(this->bias, j) : 0;
        // Matrix-vector multiplication
        for (uint16_t i = 0; i < this->input_size; i++) {
            output_temp += activation_read_float(input, i) * parameter_read_float(this->weight, (j * this->input_size) + i);
        }
        activation_write_float(output, j, relu6(output_temp));
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

    for (uint8_t g = 0; g < this->groups; g++){
        // Output channel loop
        for (uint16_t c_out = 0; c_out < output_channel_per_group; c_out++) {
            n = g * output_channel_per_group + c_out;
            // Output spatial dimensions loops
            for (uint16_t m = 0; m < this->output_row_size; m++) {
                for (uint16_t l = 0; l < this->output_col_size; l++) {
                    output_temp = this->bias ? parameter_read_float(this->bias, n) : 0;

                    for (uint16_t c_in = 0; c_in < input_channel_per_group; c_in++) {
                        k = g * input_channel_per_group + c_in;
                        for (uint8_t j = 0; j < this->kernel_row_size; j++) {
                            for (uint8_t i = 0; i < this->kernel_col_size; i++) {

                                // Convolution operation
                                output_temp += activation_read_float(
                                    input, 
                                    (k * this->input_row_size * this->input_col_size) +
                                    ((j + m * this->stride_row) * this->input_col_size) + 
                                    (i + l * this->stride_col)
                                ) * parameter_read_float(
                                    this->weight, 
                                    (n * input_channel_per_group * this->kernel_row_size * this->kernel_col_size) +
                                    (c_in * this->kernel_row_size * kernel_col_size) + 
                                    (j * this->kernel_col_size) + 
                                    i
                                );
                            }
                        }
                    }
                    activation_write_float(output, 
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


float* Conv2dReLU6::forward(float* input, float* workspace_start, uint32_t workspace_size) {
    // Getting the output start address with the input size as offset
    float* output = input == workspace_start ? workspace_start + workspace_size - (
        this->output_channel_size * this->output_row_size * this->output_col_size
    ) : workspace_start;

    float output_temp;

    uint16_t input_channel_per_group = this->input_channel_size / this->groups;
    uint16_t output_channel_per_group = this->output_channel_size / this->groups;

    uint16_t n, k;

    for (uint8_t g = 0; g < this->groups; g++){
        // Output channel loop
        for (uint16_t c_out = 0; c_out < output_channel_per_group; c_out++) {
            n = g * output_channel_per_group + c_out;
            // Output spatial dimensions loops
            for (uint16_t m = 0; m < this->output_row_size; m++) {
                for (uint16_t l = 0; l < this->output_col_size; l++) {
                    output_temp = this->bias ? parameter_read_float(this->bias, n) : 0;

                    for (uint16_t c_in = 0; c_in < input_channel_per_group; c_in++) {
                        k = g * input_channel_per_group + c_in;
                        for (uint8_t j = 0; j < this->kernel_row_size; j++) {
                            for (uint8_t i = 0; i < this->kernel_col_size; i++) {

                                // Convolution operation
                                output_temp += activation_read_float(
                                    input, 
                                    (k * this->input_row_size * this->input_col_size) +
                                    ((j + m * this->stride_row) * this->input_col_size) + 
                                    (i + l * this->stride_col)
                                ) * parameter_read_float(
                                    this->weight, 
                                    (n * input_channel_per_group * this->kernel_row_size * this->kernel_col_size) +
                                    (c_in * this->kernel_row_size * kernel_col_size) + 
                                    (j * this->kernel_col_size) + 
                                    i
                                );
                            }
                        }
                    }
                    activation_write_float(output, 
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


float* LinearReLU_DQ::forward(float* input, float* workspace_start, uint32_t workspace_size) {
    // Getting the output start address with the input size as offset
    float* output = input == workspace_start ? workspace_start + workspace_size - this->output_size : workspace_start;

    int8_t (*parameter_read_packed_intb) (const int8_t*, uint32_t);
    
    get_parameter_read_packed_intb(this->quantize_property, &parameter_read_packed_intb);

    float output_temp;
    for (uint16_t j = 0; j < this->output_size; j++) {
        output_temp = 0;
        // Matrix-vector multiplication
        for (uint16_t i = 0; i < this->input_size; i++) {
            output_temp += activation_read_float(input, i) * parameter_read_packed_intb(this->weight, (j * this->input_size) + i);
        }
        uint8_t scale_index = get_granularity(this->quantize_property) == PER_CHANNEL ? j : 0;

        activation_write_float(output,
            j,
            (this->bias ? 
            relu((output_temp * parameter_read_float(this->weight_scale, scale_index) + parameter_read_float(this->bias, j))) :
            relu((output_temp * parameter_read_float(this->weight_scale, scale_index))))
        );
    }

    return output;
}


float* LinearReLU6_DQ::forward(float* input, float* workspace_start, uint32_t workspace_size) {
    // Getting the output start address with the input size as offset
    float* output = input == workspace_start ? workspace_start + workspace_size - this->output_size : workspace_start;

    int8_t (*parameter_read_packed_intb) (const int8_t*, uint32_t);
    
    get_parameter_read_packed_intb(this->quantize_property, &parameter_read_packed_intb);

    float output_temp;
    for (uint16_t j = 0; j < this->output_size; j++) {
        output_temp = 0;
        // Matrix-vector multiplication
        for (uint16_t i = 0; i < this->input_size; i++) {
            output_temp += activation_read_float(input, i) * parameter_read_packed_intb(this->weight, (j * this->input_size) + i);
        }
        uint8_t scale_index = get_granularity(this->quantize_property) == PER_CHANNEL ? j : 0;

        activation_write_float(output,
            j,
            (this->bias ? 
            relu6((output_temp * parameter_read_float(this->weight_scale, scale_index) + parameter_read_float(this->bias, j))) :
            relu6((output_temp * parameter_read_float(this->weight_scale, scale_index))))
        );
    }

    return output;
}



float* Conv2dReLU_DQ::forward(float* input, float* workspace_start, uint32_t workspace_size) {
    // Getting the output start address with the input size as offset
    float* output = input == workspace_start ? workspace_start + workspace_size - (
        this->output_channel_size * this->output_row_size * this->output_col_size
    ) : workspace_start;

    int8_t (*parameter_read_packed_intb) (const int8_t*, uint32_t);
    
    get_parameter_read_packed_intb(this->quantize_property, &parameter_read_packed_intb);

    float output_temp;

    uint16_t input_channel_per_group = this->input_channel_size / this->groups;
    uint16_t output_channel_per_group = this->output_channel_size / this->groups;

    uint16_t n, k;

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
                                output_temp += activation_read_float(
                                    input, 
                                    (k * this->input_row_size * this->input_col_size) +
                                    ((j + m * this->stride_row) * this->input_col_size) + 
                                    (i + l * this->stride_col)
                                ) * 
                                parameter_read_packed_intb(
                                    this->weight, 
                                    (n * input_channel_per_group * this->kernel_row_size * this->kernel_col_size) +
                                    (c_in * this->kernel_row_size * kernel_col_size) + 
                                    (j * this->kernel_col_size) + i
                                );
                            }
                        }
                    }
                    uint8_t scale_index = get_granularity(this->quantize_property) == PER_CHANNEL ? n : 0;


                    activation_write_float(output, 
                        (n * this->output_row_size * this->output_col_size) + 
                        (m * this->output_col_size) + 
                        l,
                        (this->bias ? 
                        relu((output_temp * parameter_read_float(this->weight_scale, scale_index) + parameter_read_float(this->bias, n))) :
                        relu((output_temp * parameter_read_float(this->weight_scale, scale_index))))
                    );
                }
            }
        }
    }

    return output;
}



float* Conv2dReLU6_DQ::forward(float* input, float* workspace_start, uint32_t workspace_size) {
    // Getting the output start address with the input size as offset
    float* output = input == workspace_start ? workspace_start + workspace_size - (
        this->output_channel_size * this->output_row_size * this->output_col_size
    ) : workspace_start;

    int8_t (*parameter_read_packed_intb) (const int8_t*, uint32_t);
    
    get_parameter_read_packed_intb(this->quantize_property, &parameter_read_packed_intb);

    float output_temp;

    uint16_t input_channel_per_group = this->input_channel_size / this->groups;
    uint16_t output_channel_per_group = this->output_channel_size / this->groups;

    uint16_t n, k;

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
                                output_temp += activation_read_float(
                                    input, 
                                    (k * this->input_row_size * this->input_col_size) +
                                    ((j + m * this->stride_row) * this->input_col_size) + 
                                    (i + l * this->stride_col)
                                ) * 
                                parameter_read_packed_intb(
                                    this->weight, 
                                    (n * input_channel_per_group * this->kernel_row_size * this->kernel_col_size) +
                                    (c_in * this->kernel_row_size * kernel_col_size) + 
                                    (j * this->kernel_col_size) + i
                                );
                            }
                        }
                    }
                    uint8_t scale_index = get_granularity(this->quantize_property) == PER_CHANNEL ? n : 0;

                    activation_write_float(output, 
                        (n * this->output_row_size * this->output_col_size) + 
                        (m * this->output_col_size) + 
                        l,
                        (this->bias ? 
                        relu6((output_temp * parameter_read_float(this->weight_scale, scale_index) + parameter_read_float(this->bias, n))) :
                        relu6((output_temp * parameter_read_float(this->weight_scale, scale_index))))
                    );
                }
            }
        }
    }

    return output;
}


int8_t* LinearReLU_SQ::forward(int8_t* input, int8_t* workspace_start, uint32_t workspace_size) {
    // Getting the output start address with the input size as offset
    int8_t* output = input == workspace_start ? workspace_start + workspace_size - (uint32_t)ceil((float)this->output_size / get_activation_data_per_byte(this->quantize_property)) : workspace_start;

    int32_t output_temp;

    void (*activation_write_packed_intb) (int8_t*, uint32_t, int8_t);
    int8_t (*activation_read_packed_intb) (int8_t*, uint32_t);
    int8_t (*parameter_read_packed_intb) (const int8_t*, uint32_t);
    int8_t (*clamp_intb) (int32_t);
        
    get_activation_write_packed_intb(this->quantize_property, &activation_write_packed_intb);
    get_activation_read_packed_intb(this->quantize_property, &activation_read_packed_intb);
    get_parameter_read_packed_intb(this->quantize_property, &parameter_read_packed_intb);
    get_activation_clamp_intb(this->quantize_property, &clamp_intb);

    for (uint16_t j = 0; j < this->output_size; j++) {
        output_temp = this->bias ? parameter_read_int32(this->bias, j) : 0;

        for (uint16_t i = 0; i < this->input_size; i++) {
            output_temp += ((int32_t)activation_read_packed_intb(input, i) - this->input_zero_point) *
                            parameter_read_packed_intb(this->weight, (j * this->input_size) + i);
    
        }
        uint8_t scale_index = get_granularity(this->quantize_property) == PER_CHANNEL ? j : 0;

        output_temp = relu(output_temp);
        
        // Requantize to 8-bit
        output_temp = roundf(output_temp * parameter_read_float(this->bias_scale, scale_index) / this->output_scale);
        output_temp += this->output_zero_point;
        output_temp = clamp_intb(output_temp);

        activation_write_packed_intb(output, j, output_temp);
    }

    return output;
}


int8_t* LinearReLU6_SQ::forward(int8_t* input, int8_t* workspace_start, uint32_t workspace_size) {
    // Getting the output start address with the input size as offset
    int8_t* output = input == workspace_start ? workspace_start + workspace_size - (uint32_t)ceil((float)this->output_size / get_activation_data_per_byte(this->quantize_property)) : workspace_start;

    int32_t output_temp;
    int32_t six_point;

    void (*activation_write_packed_intb) (int8_t*, uint32_t, int8_t);
    int8_t (*activation_read_packed_intb) (int8_t*, uint32_t);
    int8_t (*parameter_read_packed_intb) (const int8_t*, uint32_t);
    int8_t (*clamp_intb) (int32_t);
        
    get_activation_write_packed_intb(this->quantize_property, &activation_write_packed_intb);
    get_activation_read_packed_intb(this->quantize_property, &activation_read_packed_intb);
    get_parameter_read_packed_intb(this->quantize_property, &parameter_read_packed_intb);
    get_activation_clamp_intb(this->quantize_property, &clamp_intb);

    for (uint16_t j = 0; j < this->output_size; j++) {
        output_temp = this->bias ? parameter_read_int32(this->bias, j) : 0;

        for (uint16_t i = 0; i < this->input_size; i++) {
            output_temp += ((int32_t)activation_read_packed_intb(input, i) - this->input_zero_point) *
                            parameter_read_packed_intb(this->weight, (j * this->input_size) + i);
    
        }
        uint8_t scale_index = get_granularity(this->quantize_property) == PER_CHANNEL ? j : 0;

        six_point = (int32_t)((float)6. / parameter_read_float(this->bias_scale, scale_index));
        output_temp = relux(output_temp, six_point);
        
        // Requantize to 8-bit
        output_temp = roundf(output_temp * parameter_read_float(this->bias_scale, scale_index) / this->output_scale);
        output_temp += this->output_zero_point;
        output_temp = clamp_intb(output_temp);
        
        activation_write_packed_intb(output, j, output_temp);
    }

    return output;
}


int8_t* Conv2dReLU_SQ::forward(int8_t* input, int8_t* workspace_start, uint32_t workspace_size) {
    // Getting the output start address with the input size as offset
    int8_t* output = input == workspace_start ? workspace_start + workspace_size - (uint32_t)ceil(
        (float)(this->output_channel_size * this->output_row_size * this->output_col_size) / get_activation_data_per_byte(this->quantize_property)
    ) : workspace_start;

    void (*activation_write_packed_intb) (int8_t*, uint32_t, int8_t);
    int8_t (*activation_read_packed_intb) (int8_t*, uint32_t);
    int8_t (*parameter_read_packed_intb) (const int8_t*, uint32_t);
    int8_t (*clamp_intb) (int32_t);
        
    get_activation_write_packed_intb(this->quantize_property, &activation_write_packed_intb);
    get_activation_read_packed_intb(this->quantize_property, &activation_read_packed_intb);
    get_parameter_read_packed_intb(this->quantize_property, &parameter_read_packed_intb);
    get_activation_clamp_intb(this->quantize_property, &clamp_intb);

    int32_t output_temp;

    uint16_t input_channel_per_group = this->input_channel_size / this->groups;
    uint16_t output_channel_per_group = this->output_channel_size / this->groups;

    uint16_t n, k;

    for (uint8_t g = 0; g < this->groups; g++){
        // Output channel loop
        for (uint16_t c_out = 0; c_out < output_channel_per_group; c_out++) {
            n = g * output_channel_per_group + c_out;
            // Output spatial dimensions loops
            for (uint16_t m = 0; m < this->output_row_size; m++) {
                for (uint16_t l = 0; l < this->output_col_size; l++) {
                    
                    // Calculate output index
                    output_temp = this->bias ? parameter_read_int32(this->bias, n) : 0;

                    for (uint16_t c_in = 0; c_in < input_channel_per_group; c_in++) {
                        k = g * input_channel_per_group + c_in;
                        for (uint8_t j = 0; j < this->kernel_row_size; j++) {
                            for (uint8_t i = 0; i < this->kernel_col_size; i++) {

                                output_temp += ((int32_t)activation_read_packed_intb(
                                    input,
                                    (k * this->input_row_size * this->input_col_size) +
                                    ((j + m * this->stride_row) * this->input_col_size) + 
                                    (i + l * this->stride_col)) - this->input_zero_point) *
                                    parameter_read_packed_intb(
                                        this->weight,
                                        (n * input_channel_per_group * this->kernel_row_size * this->kernel_col_size) +
                                        (c_in * this->kernel_row_size * kernel_col_size) + 
                                        (j * this->kernel_col_size) + i
                                    );
                            }
                        }
                    }
                    uint8_t scale_index = get_granularity(this->quantize_property) == PER_CHANNEL ? n : 0;

                    output_temp = relu(output_temp);

                    // Apply bias, scaling and clamping
                    output_temp = roundf(output_temp * parameter_read_float(this->bias_scale, scale_index) / this->output_scale);
                    output_temp += this->output_zero_point;
                    output_temp = clamp_intb(output_temp);
                    
                    activation_write_packed_intb(
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



int8_t* Conv2dReLU6_SQ::forward(int8_t* input, int8_t* workspace_start, uint32_t workspace_size) {
    // Getting the output start address with the input size as offset
    int8_t* output = input == workspace_start ? workspace_start + workspace_size - (uint32_t)ceil(
        (float)(this->output_channel_size * this->output_row_size * this->output_col_size) / get_activation_data_per_byte(this->quantize_property)
    ) : workspace_start;


    void (*activation_write_packed_intb) (int8_t*, uint32_t, int8_t);
    int8_t (*activation_read_packed_intb) (int8_t*, uint32_t);
    int8_t (*parameter_read_packed_intb) (const int8_t*, uint32_t);
    int8_t (*clamp_intb) (int32_t);
        
    get_activation_write_packed_intb(this->quantize_property, &activation_write_packed_intb);
    get_activation_read_packed_intb(this->quantize_property, &activation_read_packed_intb);
    get_parameter_read_packed_intb(this->quantize_property, &parameter_read_packed_intb);
    get_activation_clamp_intb(this->quantize_property, &clamp_intb);

    int32_t output_temp;
    int32_t six_point;

    uint16_t input_channel_per_group = this->input_channel_size / this->groups;
    uint16_t output_channel_per_group = this->output_channel_size / this->groups;

    uint16_t n, k;

    for (uint8_t g = 0; g < this->groups; g++){
        // Output channel loop
        for (uint16_t c_out = 0; c_out < output_channel_per_group; c_out++) {
            n = g * output_channel_per_group + c_out;
            // Output spatial dimensions loops
            for (uint16_t m = 0; m < this->output_row_size; m++) {
                for (uint16_t l = 0; l < this->output_col_size; l++) {
                    
                    output_temp = this->bias ? parameter_read_int32(this->bias, n) : 0;

                    for (uint16_t c_in = 0; c_in < input_channel_per_group; c_in++) {
                        k = g * input_channel_per_group + c_in;
                        for (uint8_t j = 0; j < this->kernel_row_size; j++) {
                            for (uint8_t i = 0; i < this->kernel_col_size; i++) {                                // Convolution operation
                                                         
                                output_temp += ((int32_t)activation_read_packed_intb(
                                    input,
                                    (k * this->input_row_size * this->input_col_size) +
                                    ((j + m * this->stride_row) * this->input_col_size) + 
                                    (i + l * this->stride_col)) - this->input_zero_point) *
                                    parameter_read_packed_intb(
                                        this->weight,
                                        (n * input_channel_per_group * this->kernel_row_size * this->kernel_col_size) +
                                        (c_in * this->kernel_row_size * kernel_col_size) + 
                                        (j * this->kernel_col_size) + i
                                    );
                            }
                        }
                    }
                    uint8_t scale_index = get_granularity(this->quantize_property) == PER_CHANNEL ? n : 0;

                    six_point = (int32_t)((float)6. / parameter_read_float(this->bias_scale, scale_index));
                    output_temp = relux(output_temp, six_point);

                    // Apply bias, scaling and clamping
                    output_temp = roundf(output_temp * parameter_read_float(this->bias_scale, scale_index)/ this->output_scale);
                    output_temp += this->output_zero_point;
                    output_temp = clamp_intb(output_temp);
                    
                    activation_write_packed_intb(
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
