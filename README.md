# 🧠 Deep MicroCompression (DMC)

> **Bare-Metal Deep Learning Inference for Resource-Constrained Microcontrollers**

**Deep MicroCompression (DMC)** is a hardware-aware compression pipeline designed to bridge the gap between modern neural networks and ultra-low-power microcontrollers (e.g., ATmega328P, Cortex-M4).

Unlike frameworks that rely on heavy runtime interpreters (like TFLite Micro), DMC optimizes the model at the bit-level and generates a **standalone, dependency-free C library** tailored for integer-only execution.

---

## 🚀 Key Features

### 1. Structured Channel Pruning
* **Dependency-Aware:** Automatically handles channel dependency propagation across layers (Conv2d $\to$ BatchNorm $\to$ ReLU).
* **Physical Reduction:** Unlike "mask-only" pruning, DMC physically removes filters and kernels, resulting in a smaller, dense model that requires no sparse matrix libraries.
* **Sensitivity Analysis:** Built-in tools to profile layer sensitivity (L2 norm) for optimal sparsity configuration.

### 2. Quantization
* **Flexible Precision Schemes:** Supports both Dynamic and Static quantization across configurable bitwidths (8-bit, 4-bit, 2-bit).
* **Quantization-Aware Training (QAT):** Simulates quantization noise during training to recover accuracy at low bitwidths (4-bit/2-bit).
* **Pure Integer Inference:** The Static Quantization mode pre-computes scaling factors and zero-points, enabling fully integer-based arithmetic. This eliminates the need for floating-point operations (FLOPs) at runtime, drastically reducing latency on bare-metal MCUs .

### 3. Hardware-Aware Bit-Packing
* **Compression:** Implements a fixed-length bit-packing scheme. Packs multiple weights (e.g., four 2-bit weights) into single bytes (`uint8_t`), maximizing flash storage density.
* **Fast Unpacking:** Uses optimized C macros (`define.h`) with shift/mask operations to decode weights on-the-fly, avoiding complex Huffman decoders.

### 4. Zero-Dependency C Export
* **Portable:** Generates generic C++ code compatible with any compiler (GCC, Clang) and architecture (AVR, RISC-V, ARM).
* **Static Allocation:** Calculates "Ping-Pong" SRAM buffer requirements (`MAX_OUTPUT_EVEN/ODD`) at compile time, eliminating dynamic memory allocation (`malloc`).

---

## 🛠️ Project Architecture
```plaintext
.
├── deployment/                # C/C++ Inference Engine
│   ├── core/                  # Core macros and definitions
│   ├── layers/                # Bare-metal C++ layer implementations
│   ├── models/                # Generated C++ model containers
│   └── deep_microcompression.h # Single include entry point
│
├── development/               # Python Training & Compression Pipeline
│   ├── compressors/           # Pruning and Quantization engines
│   ├── layers/                # PyTorch layers augmented with masking/observers
│   ├── models/                # Sequential container & model definitions
│   ├── tools/                 # Utilities (Library generation, NAS Estimators)
│   └── utils.py               # Bit-packing math (Algorithm 1 & 3)
│
└── requirements.txt
```

## 💻 Usage Guide

### Phase 1: Compression Pipeline

**1. Define & Train Baseline**

```python
from development import Sequential, Conv2d, Linear, ReLU, Flatten

model = Sequential(
    Conv2d(1, 6, kernel_size=5, pad=(2,2,2,2)),
    ReLU(),
    Flatten(),
    Linear(6*28*28, 10)
)
model.fit(train_loader, epochs=10, ...)
```

**2. Configure Pipeline (Pruning + Quantization)**
Apply the settings derived from your analysis.

```python
from development import QuantizationScheme

compression_config = {
    # Stage 1: Structured Pruning
    "prune_channel": {
        "sparsity": {
            "conv2d_0": 0.0,  # Keep sensitive input layers dense
            "conv2d_1": 0.5,  # Prune 50%
            "linear_0": 0.8   # Prune 80% of dense layer
        },
        "metric": "l2"
    },
    # Stage 2: Static Quantization
    "quantize": {
        "scheme": QuantizationScheme.STATIC,
        "bitwidth": 4,         # 4-bit Weights
        "granularity": "PER_TENSOR"
    }
}

# Transform model to Compressed State
# calibration_data is MANDATORY for Static Quantization to determine min/max ranges
compressed_model = model.init_compress(
    compression_config, 
    input_shape=(1, 1, 28, 28), 
    calibration_data=train_sample+data
)
```

**3. Retrain (Fine-Tuning / QAT)**
Recover accuracy lost due to pruning and quantization noise.

```python
# The model is now in "Fake Quantization" mode with Masks active
compressed_model.fit(train_loader, epochs=5, ...)
```

**4. Export to C (Stage 3)**
Generate the dependency-free C library.

```python
# Generates: model.h, model_def.cpp, model_params.cpp
compressed_model.convert_to_c(
    input_shape=(1, 1, 28, 28),
    var_name="model",
    src_dir="./deployment_src",
    include_dir="./deployment_include"
)
```

-----

### Phase 3: Embedded Integration

**1. Generate the Runtime Engine**
Run the utility script to copy the core C++ files to your project.

```bash
python generate_library_tree.py /path/to/my/mcu/project/lib
```

**2. Include & Compile**
Add the generated files to your firmware project (PlatformIO, Keil, CMake).

```c
#include "deep_microcompression.h"
#include "model.h"      // Generated model definition


int main(void) {
    // 1. Access Model Buffers defined in model.h
    int8_t* input_ptr = model.input; 
    int8_t* output_ptr = model.output;

    while(1) {
        // 2. Load Data
        // You can use 'set_packed_value' to handle bit-packing automatically if the input 
        // buffer requires sub-byte addressing.
        get_input(input_ptr);

        // 3. Run Inference
        // Executes layer-by-layer forward pass on the device.
        model.predict();

        // 4. Access Predictions
        for (int i=0; i < 10; i++) {
            // 'get_packed_value' decodes 8-bit, 4-bit, or 2-bit outputs on-the-fly
            int val = (int)get_packed_value(output_ptr, i);
            sprint(val); sprint(" ");
        }
        sprint("\n");
    }
}
```
