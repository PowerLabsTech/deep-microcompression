

#include "branch.h"
#include <string.h>



Branch::Branch(Layer* sublayer1, Layer* sublayer2, uint32_t sublayer1_workspace_size) {
    this->sublayer1 = sublayer1;
    this->sublayer2 = sublayer2;
    this->sublayer1_workspace_size = sublayer1_workspace_size;
}

void Branch::forward(float* workspace_start, uint32_t workspace_size) {

    // NOTE: this->sublayer1_workspace_size should be input size, if copy is an expensive operation, I will
    //       need to add that parameter, just this save a 4 bytes in memeory
    memcpy((float*)((uint8_t*)workspace_start + this->sublayer1_workspace_size), workspace_start, this->sublayer1_workspace_size);

    this->sublayer1->forward(workspace_start, this->sublayer1_workspace_size);
    if (this->sublayer2) {
        this->sublayer2->forward((float*)((uint8_t*)workspace_start + this->sublayer1_workspace_size), (workspace_size - this->sublayer1_workspace_size));
    }

    for(uint32_t i=0; i < this->get_output_size(); i++) {
        activation_write_float(
            workspace_start, i, (
                activation_read_float(workspace_start, i) +
                activation_read_float((float*)((uint8_t*)workspace_start + this->sublayer1_workspace_size), i)
            )
        );
    } 
}



uint32_t Branch::get_output_size(void) {
    return this->sublayer1->get_output_size();
}




Branch_SQ::Branch_SQ(Layer_SQ* sublayer1, Layer_SQ* sublayer2, uint32_t sublayer1_workspace_size, uint8_t quantize_property, uint8_t* quantize_parameters) {
    this->sublayer1 = sublayer1;
    this->sublayer2 = sublayer2;
    this->sublayer1_workspace_size = sublayer1_workspace_size;
    this->quantize_property = quantize_property;
    this->quantize_parameters = quantize_parameters;
}

void Branch_SQ::forward(int8_t* workspace_start, uint32_t workspace_size) {

    void (*activation_write_packed_intb) (int8_t*, uint32_t, int8_t);
    int8_t (*activation_read_packed_intb) (int8_t*, uint32_t);
    int8_t (*clamp_intb) (int32_t);

    get_activation_write_packed_intb(this->quantize_property, &activation_write_packed_intb);
    get_activation_read_packed_intb(this->quantize_property, &activation_read_packed_intb);
    get_activation_clamp_intb(this->quantize_property, &clamp_intb);

    // copying the input to sublayer2 workspace
    int8_t* sublayer2_workspace_start = workspace_start + this->sublayer1_workspace_size;
    memcpy(sublayer2_workspace_start, workspace_start, this->sublayer1_workspace_size);


    this->sublayer1->forward(workspace_start, this->sublayer1_workspace_size);
    if (this->sublayer2) {
        this->sublayer2->forward(sublayer2_workspace_start, (workspace_size - this->sublayer1_workspace_size));
    }

    float s1_so = parameter_read_float((float*)this->quantize_parameters, 0);
    float s2_so = parameter_read_float((float*)this->quantize_parameters, 1);
    int8_t zo = (int8_t)dmc_pgm_read_byte(this->quantize_parameters + 8);

    for(uint32_t i=0; i < this->get_output_size(); i++) {
        activation_write_packed_intb(
            workspace_start, i, (int8_t)clamp_intb((int16_t)roundf(
                (s1_so * (float)activation_read_packed_intb(workspace_start, i)) +
                (s2_so * (float)activation_read_packed_intb(sublayer2_workspace_start, i))
            ) + (int16_t)zo
        ));
    }
}


uint32_t Branch_SQ::get_output_size(void) {
    return this->sublayer1->get_output_size();
}