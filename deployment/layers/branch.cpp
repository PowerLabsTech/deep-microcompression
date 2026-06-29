

#include "branch.h"
#include <string.h>



void Branch::forward(float* workspace_start, uint32_t workspace_size) {
    const uint8_t* p = this->buffer;
    Layer* sublayer1 = (Layer*)dmc_pgm_read_ptr(p); p += DMC_PTR_SIZE;
    Layer* sublayer2 = (Layer*)dmc_pgm_read_ptr(p); p += DMC_PTR_SIZE;
    uint32_t sublayer1_workspace_size = (uint32_t)dmc_pgm_read_dword(p);

    memcpy((float*)((uint8_t*)workspace_start + sublayer1_workspace_size), workspace_start, sublayer1_workspace_size);

    sublayer1->forward(workspace_start, sublayer1_workspace_size);
    if (sublayer2) {
        sublayer2->forward((float*)((uint8_t*)workspace_start + sublayer1_workspace_size), (workspace_size - sublayer1_workspace_size));
    }

    for(uint32_t i=0; i < get_output_size(); i++) {
        activation_write_float(
            workspace_start, i, (
                activation_read_float(workspace_start, i) +
                activation_read_float((float*)((uint8_t*)workspace_start + sublayer1_workspace_size), i)
            )
        );
    }
}



uint32_t Branch::get_output_size(void) {
    return ((Layer*)dmc_pgm_read_ptr(this->buffer))->get_output_size();
}




void Branch_SQ::forward(int8_t* workspace_start, uint32_t workspace_size) {
    const uint8_t* p = this->buffer;
    Layer_SQ* sublayer1        = (Layer_SQ*)dmc_pgm_read_ptr(p); p += DMC_PTR_SIZE;
    Layer_SQ* sublayer2        = (Layer_SQ*)dmc_pgm_read_ptr(p); p += DMC_PTR_SIZE;
    uint32_t sublayer1_workspace_size = (uint32_t)dmc_pgm_read_dword(p); p += 4;
    uint8_t* quantize_parameters = (uint8_t*)dmc_pgm_read_ptr(p); p += DMC_PTR_SIZE;
    uint8_t quantize_property  = (uint8_t)dmc_pgm_read_byte(p);

    void (*activation_write_packed_intb) (int8_t*, uint32_t, int8_t);
    int8_t (*activation_read_packed_intb) (int8_t*, uint32_t);
    int8_t (*clamp_intb) (int32_t);

    get_activation_write_packed_intb(quantize_property, &activation_write_packed_intb);
    get_activation_read_packed_intb(quantize_property, &activation_read_packed_intb);
    get_activation_clamp_intb(quantize_property, &clamp_intb);

    // copying the input to sublayer2 workspace
    int8_t* sublayer2_workspace_start = workspace_start + sublayer1_workspace_size;
    memcpy(sublayer2_workspace_start, workspace_start, sublayer1_workspace_size);


    sublayer1->forward(workspace_start, sublayer1_workspace_size);
    if (sublayer2) {
        sublayer2->forward(sublayer2_workspace_start, (workspace_size - sublayer1_workspace_size));
    }

    float s1_so = parameter_read_float((float*)quantize_parameters, 0);
    float s2_so = parameter_read_float((float*)quantize_parameters, 1);
    int8_t zo = (int8_t)dmc_pgm_read_byte(quantize_parameters + 8);

    for(uint32_t i=0; i < get_output_size(); i++) {
        activation_write_packed_intb(
            workspace_start, i, (int8_t)clamp_intb((int16_t)roundf(
                (s1_so * (float)activation_read_packed_intb(workspace_start, i)) +
                (s2_so * (float)activation_read_packed_intb(sublayer2_workspace_start, i))
            ) + (int16_t)zo
        ));
    }
}


uint32_t Branch_SQ::get_output_size(void) {
    return ((Layer_SQ*)dmc_pgm_read_ptr(this->buffer))->get_output_size();
}
