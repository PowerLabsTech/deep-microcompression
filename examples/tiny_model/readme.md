# 🚀 DMC Quick Start: End-to-End Tiny Model

This example demonstrates the complete **Deep MicroCompression (DMC)** workflow: designing a model in Python, applying hardware-aware compression (Pruning & Quantization), and deploying it to a C++ environment using CMake.

## 📋 Prerequisites

Ensure your directory structure is set up correctly. This example assumes it sits in the example directory of the repo.

```plaintext
root/
├── development/       # Main DMC Python library
├── deployment/        # Main DMC C++ runtime
└── example/
    └──tiny_model/        # This example directory
        ├── generate.py
        └── deploy/
```

-----

## 🛠️ Step 1: Model Generation (Python)

The `generate.py` script serves as the orchestrator. It performs the following pipeline steps:

1.  **Definition:** Creates a simple `Lenet-style` CNN using DMC layers (`Conv2d`, `ReLU`, `Flatten`, `Linear`).
2.  **Configuration:** Sets up the compression parameters:
      * **Pruning:** 50% sparsity using L2-norm ranking.
      * **Quantization:** 4-bit Static Quantization (Integer-only inference).
3.  **Training:** Simulates Quantization-Aware Training (QAT) to recover accuracy.
4.  **Export:** Generates the C++ header and source files into the `deploy/` directory.

**Run the generator:**

```bash
# Run from the tiny_model directory
python generate.py
```

**Output:**
This will create a `deploy` directory containing your model artifacts:

  * `include/tiny_model.h`: Model definition header.
  * `src/tiny_model_def.cpp`: Layer definitions.
  * `src/tiny_model_params.cpp`: Bit-packed weights and quantization scales.

-----

## ⚙️ Step 2: Runtime Library Setup

The generated model requires the DMC C++ Inference Engine to run. Use the provided tool to copy the core library into your deployment folder.

```bash
# Syntax: python <PATH_TO_TOOL> <DESTINATION_LIB_DIR>
python ../../development/tools/generate_library_tree.py deploy/lib/
```

*This command copies the bare-metal kernels from the main `deployment` directory into `tiny_model/deploy/lib/deep_microcompression`.*

-----

## 🏗️ Step 3: Building the C++ Application

The provided `CMakeLists.txt` configures the build for the generated model.

**Understanding the CMake Configuration:**

  * **`add_executable(main ...)`**: Compiles the test harness (`main.cpp`) alongside the generated model files (`tiny_model_def.cpp`, `tiny_model_params.cpp`).
  * **`target_compile_definitions(... "ST4")`**: This flag tells the DMC engine which quantization scheme to use during compilation.
      * `ST4` = **S**atic **per Tensor** **4**-bit quantization.
      * *Note: Ensure this matches the `bitwidth: 4` and `scheme: STATIC` config in `generate.py`.*

**Build Commands:**

```bash
cd deploy
mkdir build
cd build

# Generate Makefiles, Configure Build (passing the compression format)
cmake .. -DCOMPRESSION_FORMAT=ST4

# Compile
make

# Run the inference test
./main
```

-----

## 📂 Project Structure

After running the steps above, your directory should look like this:

```plaintext
tiny_model/
├── generate.py                 # Pipeline Orchestrator
├── readme.md
└── deploy/                     # Deployment Artifacts
    ├── CMakeLists.txt          # Build Configuration
    ├── include/
    │   ├── tiny_model.h
    │   └── tiny_model_test_input.h
    ├── lib/
    │   └── deep_microcompression/  # Core DMC Runtime Engine
    ├── src/
    │   ├── main.cpp            # Your C++ Entry Point
    │   ├── tiny_model_def.cpp  # Generated Layer Definitions
    │   └── tiny_model_params.cpp # Generated Bit-Packed Weights
    └── build/                  # Compiled Binaries
        └── main
```

-----

## 📝 Customization

To adapt this for your own use case:

1.  **Change Compression:** Edit `generate.py` to change `bitwidth` (e.g., to 2 or 8) or `scheme`.
2.  **Update Build Flag:** If you change the bitwidth in Python, simply update the flag when running cmake (e.g., to `"ST2"` or `"ST8"`).
```bash
cmake -B build/ -DCOMPRESSION_FORMAT=ST8
```
3.  **Deployment Directory:** You can change `DEPLOYMENT_BASE_DIR` in `generate.py` to output files directly to your MCU project folder.

