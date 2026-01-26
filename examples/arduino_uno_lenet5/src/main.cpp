#include <Arduino.h>
#include "deep_microcompression.h" 

// Include your generated model artifacts
#include "lenet5_model.h"
// Stores a sample image in pack c array
#include "lenet5_model_test_input.h" 

#define AVERAGING_TIME 1
#define INPUT_SIZE (1*28*28)
#define NUM_CLASSES 10
#define DELAY 1000

#define sprint(str) Serial.print(str)
#define sprintln(str) Serial.println(str)

void setup() {
  Serial.begin(9600);
  while (!Serial); 

  sprint(F("\n\n--- DMC Arduino Uno Inference ---\n"));

  sprint(F("Model Initialized. RAM usage check suggested.\n"));
}

void loop() {

  uint32_t t0;
  uint32_t dt;

  sprint(F("Loading Input Data...\n"));
  
  // Load Data (Flash -> RAM)
  // We use 'parameter_read_packed_int4' to handle cases where input is bit-packed (e.g. 4-bit images)
  // This unpacks 'test_input' (from header) into the working 'input' buffer.
  for (int j = 0; j < INPUT_SIZE; j++) {
      int8_t val = parameter_read_packed_int4((int8_t*)test_input, j); 
      lenet5_model.set_input(j, val);
  }

  sprint(F("Running Inference...\n"));
  t0 = millis();

  // Run Model
  for (int t = 0; t < AVERAGING_TIME; t++) {
    lenet5_model.predict();
  }
  
  dt = millis() - t0;

  // Report Results
  sprint(F("Inference Time: ")); sprint(dt / AVERAGING_TIME); sprintln(F(" ms"));
  
  sprintln(F("Logits predictions:"));
  for(int i=0; i < NUM_CLASSES; i++) {
      // Decode output (unpacks 4-bit/2-bit to int if needed)
      sprint("Digit "); sprint(i); sprint(" - "); sprintln(lenet5_model.get_output(i));
  }
  sprint(F("\n-----------------------------\n\n"));

  delay(DELAY); 
}



