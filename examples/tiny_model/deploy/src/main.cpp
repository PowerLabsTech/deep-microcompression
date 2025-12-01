#include <stdio.h>

// Include your generated model artifacts
#include "tiny_model.h"         
#include "tiny_model_test_input.h" 

// Simulation configs
#define INPUT_SIZE (1 * 28 * 28)
#define NUM_CLASSES 10

int main() {
    printf("--- Deep MicroCompression Bare-Metal Inference ---\n");

    // 1. Memory Access
    // DMC generates optimized 'int8_t' buffers for 4-bit quantized models
    // to save RAM. We access them here.
    int8_t* input_buffer = tiny_model.input;
    int8_t* output_buffer = tiny_model.output;

    // 2. Load Data (Flash -> RAM)
    // The test input might be packed (two 4-bit pixels per byte).
    // 'get_packed_value' handles the unpacking automatically (Algorithm 2).
    printf("Loading data...\n");
    for (int i = 0; i < INPUT_SIZE; i++) {
        // Reads from 'test_input' (Flash) and writes to 'input_buffer' (RAM)
        int value = get_packed_value(test_input, i);
        set_packed_value(input_buffer, i, value);
    }

    // 3. Run Inference
    // Executes the Integer-Only pipeline. 
    // No floating point math is used here!
    printf("Running Inference...\n");
    tiny_model.predict();

    // 4. Print Results
    printf("Predictions (Logits):\n");
    for (int i = 0; i < NUM_CLASSES; i++) {
        // Output is also int8_t. We unpack it to view the value.
        int val = (int)get_packed_value(output_buffer, i);
        printf("Class %d: %d\n", i, val);
    }

    return 0;
}