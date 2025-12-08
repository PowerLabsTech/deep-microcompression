#include <Arduino.h>
#include "deep_microcompression.h" 

// Include your generated model artifacts
#include "uno_model.h"
#include "uno_model_test_input.h" 

#define AVERAGING_TIME 1
#define INPUT_SIZE (1*28*28)
#define NUM_CLASSES 10

// --- Global Pointers ---
// We declare pointers globally, but assign them in setup()
// because the model instance is created in uno_model_def.cpp
int8_t* input_ptr; 
int8_t* output_ptr;

void setup() {
  Serial.begin(9600);
  while (!Serial); 

  Serial.println("\n\n--- DMC Arduino Uno Inference ---");

  // 2. Link Pointers to Model Buffers
  // The 'uno_model' object is instantiated in src/uno_model_def.cpp
  input_ptr = uno_model.input;
  output_ptr = uno_model.output;

  Serial.print("Model Initialized. RAM usage check suggested.\n");
}

void loop() {
  uint32_t t0;
  uint32_t dt;

  Serial.println("Loading Input Data...");
  
  // Load Data (Flash -> RAM)
  // We use 'set_packed_value' to handle cases where input is bit-packed (e.g. 4-bit images)
  // This unpacks 'test_input' (from header) into the working 'input_ptr' buffer.
  for (int j = 0; j < INPUT_SIZE; j++) {
      int val = get_packed_value(test_input, j);
      set_packed_value(input_ptr, j, val);
  }

  Serial.println("Running Inference...");
  t0 = millis();

  // Run Model
  for (int t = 0; t < AVERAGING_TIME; t++) {
    uno_model.predict();
  }
  
  dt = millis() - t0;

  // Report Results
  Serial.print("Inference Time: "); Serial.print(dt / AVERAGING_TIME); Serial.println(" ms");
  
  Serial.print("Predictions: ");
  for(int i=0; i < NUM_CLASSES; i++) {
      // Decode output (unpacks 4-bit/2-bit to int if needed)
      int val = (int)get_packed_value(output_ptr, i);
      Serial.print(val); Serial.print(" ");
  }
  Serial.println("\n-----------------------------");

  delay(2000); 
}