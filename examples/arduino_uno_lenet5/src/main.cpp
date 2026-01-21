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

  Serial.print(F("\n\n--- DMC Arduino Uno Inference ---\n"));

  // 2. Link Pointers to Model Buffers
  // The 'uno_model' object is instantiated in src/uno_model_def.cpp
  input_ptr = uno_model.input;
  output_ptr = uno_model.output;

  Serial.print(F("Model Initialized. RAM usage check suggested.\n"));
}

void loop() {
  uint32_t t0;
  uint32_t dt;

  Serial.print(F("Loading Input Data...\n"));
  
  // Load Data (Flash -> RAM)
  // We use 'set_packed_value' to handle cases where input is bit-packed (e.g. 4-bit images)
  // This unpacks 'test_input' (from header) into the working 'input_ptr' buffer.
  for (int j = 0; j < INPUT_SIZE; j++) {
      int val = par_read_packed_intb(test_input, j);
      act_write_packed_intb(input_ptr, j, val);
  }

  Serial.print(F("Running Inference...\n"));
  t0 = millis();

  // Run Model
  for (int t = 0; t < AVERAGING_TIME; t++) {
    uno_model.predict();
  }
  
  dt = millis() - t0;

  // Report Results
  Serial.print(F("Inference Time: ")); Serial.print(dt / AVERAGING_TIME); Serial.println(F(" ms"));
  
  Serial.print(F("Predictions: "));
  for(int i=0; i < NUM_CLASSES; i++) {
      // Decode output (unpacks 4-bit/2-bit to int if needed)
      int val = (int)act_read_packed_intb(output_ptr, i);
      Serial.print(val); Serial.print(" ");
  }
  Serial.print(F("\n-----------------------------\n\n"));

  delay(2000); 
}