# ⚡ DMC Arduino Deployment Example

This project demonstrates how to deploy a pre-compressed **LeNet-5** model on an **Arduino Uno (ATmega328P)** using PlatformIO. The model and packed image is include in the directory.

It leverages **Deep MicroCompression (DMC)** to fit a neural network into the strict 2KB SRAM limit of the ATmega328P by using:
* **4-bit Static Quantization** (Integer-only arithmetic)
* **Bit-Packing** (Storing 2 weights per byte)
* **Ping-Pong Buffer Management** (Minimal RAM footprint)

## 📂 Project Structure

```text
.
├── include/
│   ├── uno_model.h             # Generated Model Header
│   └── uno_model_test_input.h  # Generated Test Image
├── lib/
│   └── deep_microcompression/  # The DMC Inference Engine
├── src/
│   ├── main.cpp                # Arduino Sketch
│   ├── uno_model_def.cpp       # Layer Definitions
│   └── uno_model_params.cpp    # Bit-Packed Weights
└── platformio.ini              # Build Configuration
```

## 🚀 Setup & Usage

### 1\. Prerequisites

  * **VSCode** with **PlatformIO** extension installed.
  * **Python DMC Environment** (to generate the model files).

### 2\. Install the Runtime Library

The model requires the DMC C++ library to run.. Use the provided Python script to generate the library tree into your lib/ folder:

```bash
# Run from within this directory
python ../../development/tools/generate_library_tree.py lib/
```


### 3\. Configure Quantization

Open `platformio.ini` and ensure the flags match your Python configuration:

```ini
build_flags = 
    -DQUANTIZATION_SCHEME=STATIC
    -DQUANTIZATION_BITWIDTH=4 ; 4 = 4-bit Weights
    -DQUANTIZATION_GRANULARITY=PER_TENSOR
```

### 4\. Upload

1.  Connect your Arduino Uno.
2.  Click the **PlatformIO Upload** button (Right Arrow icon).
3.  Open the **Serial Monitor** (Plug icon) to see inference results.

## ⚠️ Memory constraints

The Arduino Uno has only 2048 bytes of SRAM.

- **Flash (PROGMEM):** The model weights (uno_model_params.cpp) are stored in Flash memory to save RAM.

- **Stack/Heap:** The input image and intermediate buffers (MAX_OUTPUT_EVEN/ODD) must fit in the remaining SRAM.
## 📊 Expected Output

```text
--- DMC Arduino Uno Inference ---
Loading Input Data...
Running Inference...
Inference Time: 2288 ms
Predictions: 0 1 -2 2 -3 4 2 -1 3 2
```
