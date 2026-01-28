

#include "branch.h"



Branch::Branch(Layer* sublayer1, Layer* sublayer2) {
    this->sublayer1 = sublayer1;
    this->sublayer2 = sublayer2;
}

float* Branch::forward(float* input, float* workspace_start, uint32_t workspace_size) {

    float* output1 = this->sublayer1->forward(input, workspace_start, workspace_size);
    uint32_t output_size1 = this->get_output_size();

    float* output2 = input;
    if (this->sublayer2) {
        if (input == workspace_start) {
            // if the input is left aligned, to put the output of sublayer 2 in the middle
            output2 = this->sublayer2->forward(input, workspace_start, workspace_size - output_size1);
        }
        else {
            // if the input is right aligned, to put the output of sublayer 2 in the middle
            output2 = this->sublayer2->forward(input, workspace_start + output_size1, workspace_size - output_size1);
        }
    }

    float* output = output1;

    for(uint32_t i=0; i<output_size1; i++) {
        activation_write_float(output, i, (activation_read_float(output1, i) + activation_read_float(output2, i)));
    } 
    return output;
}


uint32_t Branch::get_output_size(void) {
    return this->sublayer1->get_output_size();
}




Branch_SQ::Branch_SQ(Layer_SQ* sublayer1, Layer_SQ* sublayer2, uint8_t quantize_property) {
    this->sublayer1 = sublayer1;
    this->sublayer2 = sublayer2;
    this->quantize_property = quantize_property;
}

int8_t* Branch_SQ::forward(int8_t* input, int8_t* workspace_start, uint32_t workspace_size) {
    // Getting the output start address with the input size as offset
    // int8_t* output = input == workspace_start ? workspace_start + workspace_size - 
    // (uint32_t)ceil((float)this->input_size / get_activation_data_per_byte(this->quantize_property)) : workspace_start;
    int8_t* output1 = this->sublayer1->forward(input, workspace_start, workspace_size);
    uint32_t output_size1_in_byte = (uint32_t)ceil((float)this->get_output_size() / get_activation_data_per_byte(this->quantize_property));

    int8_t* output2 = input;
    if (this->sublayer2) {
        if (input == workspace_start) {
            // if the input is left aligned, to put the output of sublayer 2 in the middle
            output2 = this->sublayer2->forward(input, workspace_start, workspace_size - output_size1_in_byte);
        }
        else {
            // if the input is right aligned, to put the output of sublayer 2 in the middle
            output2 = this->sublayer2->forward(input, workspace_start + output_size1_in_byte, workspace_size - output_size1_in_byte);
        }
    }

    int8_t* output = output1;

    void (*activation_write_packed_intb) (int8_t*, uint32_t, int8_t);
    int8_t (*activation_read_packed_intb) (int8_t*, uint32_t);
    
    get_activation_write_packed_intb(this->quantize_property, &activation_write_packed_intb);
    get_activation_read_packed_intb(this->quantize_property, &activation_read_packed_intb);

    for(uint32_t i=0; i<this->sublayer1->get_output_size(); i++) {
        activation_write_packed_intb(output, i, (activation_read_packed_intb(output1, i)/2 + activation_read_packed_intb(output2, i)/2));
    } 
    return output;
}


uint32_t Branch_SQ::get_output_size(void) {
    return this->sublayer1->get_output_size();
}