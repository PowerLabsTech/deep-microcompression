import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force TensorFlow to CPU

import sys
import os
import copy
import random
import torch
from torch import nn, optim
from torch.utils import data
from torchvision import datasets, transforms
from tqdm.auto import tqdm
import numpy as np

import tensorflow as tf
from tensorflow.keras.datasets import mnist # type: ignore

sys.path.append("../../../")

try:
    from development import (
        Sequential, Conv2d, Linear, ReLU, BatchNorm2d,
        MaxPool2d, Flatten, QuantizationScheme, QuantizationGranularity
    )
except ImportError:
    print("Error: Could not import the 'development' module.")
    print("Please ensure this script is run from 'experiments/lenet5/'")
    print("and the 'development' module is in the project root ('../../../').")
    sys.exit(1)

# --- Constants ---
DEVICE = "cpu"  # Force PyTorch to CPU for fair comparison
LUCKY_NUMBER = 25
LENET5_TF_FILE = "lenet5_model.keras"
INPUT_SHAPE_TORCH = (1, 28, 28)
INPUT_SHAPE_TF = (28, 28, 1)

# Set random seeds for reproducibility
torch.manual_seed(LUCKY_NUMBER)
tf.random.set_seed(LUCKY_NUMBER)
np.random.seed(LUCKY_NUMBER)
random.seed(LUCKY_NUMBER)

# --- 1. Load Data ---
def get_data_loaders():
    """Loads MNIST data for both TF/Numpy and PyTorch."""
    print("Loading MNIST dataset...")
    (mnist_train_image, mnist_train_label), (mnist_test_image, mnist_test_label) = mnist.load_data()
    
    # TF/Numpy data (normalized, reshaped)
    np_train_img = (mnist_train_image / 255.0).astype(np.float32).reshape(-1, 28, 28, 1)
    np_train_lbl = mnist_train_label.astype(np.int64)
    np_test_img = (mnist_test_image / 255.0).astype(np.float32).reshape(-1, 28, 28, 1)
    np_test_lbl = mnist_test_label.astype(np.int64)

    # PyTorch data (needs channel-first)
    torch_train_img = torch.from_numpy(np_train_img.transpose(0, 3, 1, 2))
    torch_train_lbl = torch.from_numpy(np_train_lbl)
    torch_test_img = torch.from_numpy(np_test_img.transpose(0, 3, 1, 2))
    torch_test_lbl = torch.from_numpy(np_test_lbl)

    torch_train_dataset = data.TensorDataset(torch_train_img, torch_train_lbl)
    torch_test_dataset = data.TensorDataset(torch_test_img, torch_test_lbl)
    
    torch_train_loader = data.DataLoader(torch_train_dataset, batch_size=32, shuffle=True)
    torch_test_loader = data.DataLoader(torch_test_dataset, batch_size=32, shuffle=False)
    
    return (np_train_img, np_train_lbl), (np_test_img, np_test_lbl), torch_train_loader, torch_test_loader

# --- 2. TF Model ---
def get_tf_model(train_data, test_data):
    """Trains or loads the baseline TF/Keras model."""
    print("\n--- STAGE 1: Training/Loading TF Baseline Model ---")
    (train_img, train_lbl) = train_data
    (test_img, test_lbl) = test_data
    
    if os.path.exists(LENET5_TF_FILE):
        print(f"Loading existing TF model from {LENET5_TF_FILE}...")
        return tf.keras.models.load_model(LENET5_TF_FILE)
    
    print("No TF model found. Training from scratch (up to 25 epochs)...")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=INPUT_SHAPE_TF),
        tf.keras.layers.ZeroPadding2D(padding=2),
        tf.keras.layers.Conv2D(filters=6, kernel_size=5, padding="valid"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5, padding="valid"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=84),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(units=10)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    
    model.fit(
        train_img, train_lbl,
        epochs=25,
        batch_size=32, 
        validation_data=(test_img, test_lbl),
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
        verbose=2
    )
    
    print(f"Saving TF model to {LENET5_TF_FILE}...")
    model.save(LENET5_TF_FILE)
    return model

# --- 3. TF to DMC (PyTorch) Converter ---
@torch.no_grad()
def copy_tensor(tensor_source, tensor_destination):
    tensor_destination.copy_(tensor_source)

@torch.no_grad()
def convert_tf_to_dmc(tf_model):
    """Converts the trained Keras model to the DMC Sequential model."""
    print("Converting TF model to DMC (PyTorch) model...")
    prev_layer_is_flatten = False
    last_conv_channel = None
    dmc_layers = []
    pad_next_conv = None

    for layer in tf_model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            pass
        elif isinstance(layer, tf.keras.layers.ZeroPadding2D):
            pad_next_conv = layer.padding

        elif isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_channel = layer.weights[0].shape[-1]
            weight_np = np.transpose(layer.weights[0].numpy(), (3, 2, 0, 1)) # TF(kH,kW,in,out) -> Torch(out,in,kH,kW)
            out_channels, in_channels, kernel_size, _ = weight_np.shape
            stride = layer.strides[0]
            padding_str = layer.padding
            pad = [0]*4 if padding_str == "valid" else [(kernel_size - 1)//2]*4
            if pad_next_conv is not None:
                for i, padding in enumerate(pad_next_conv):
                    pad[i*2] = padding[0]
                    pad[i*2 + 1] = padding[1]
                pad_next_conv = None
            has_bias = len(layer.weights) > 1
            conv_layer = Conv2d(in_channels, out_channels, kernel_size, stride, pad=pad, bias=has_bias)
            if has_bias:
                copy_tensor(torch.from_numpy(layer.weights[1].numpy()), conv_layer.bias)
            copy_tensor(torch.from_numpy(weight_np), conv_layer.weight)
            dmc_layers.append(conv_layer)


        elif isinstance(layer, tf.keras.layers.Dense):
            weight_np = layer.weights[0].numpy() # TF(in, out)
            if prev_layer_is_flatten:
                # Handle TF's channel-last flatten for Dense layer
                weight_np = weight_np.reshape(-1, last_conv_channel, weight_np.shape[-1])
                weight_np = np.transpose(weight_np, (1, 0, 2))
                weight_np = weight_np.reshape(-1, weight_np.shape[-1])
                prev_layer_is_flatten = False
            
            weight_np = np.transpose(weight_np, (1, 0)) # TF(in,out) -> Torch(out,in)
            out_features, in_features = weight_np.shape
            
            linear_layer = Linear(out_features=out_features, in_features=in_features, bias=True)
            copy_tensor(torch.from_numpy(layer.weights[1].numpy()), linear_layer.bias)
            copy_tensor(torch.from_numpy(weight_np), linear_layer.weight)
            dmc_layers.append(linear_layer)         

        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            bn_layer = BatchNorm2d(num_features=layer.gamma.shape[0], affine=layer.scale or layer.center, eps=layer.epsilon, momentum=(1 - layer.momentum))
            copy_tensor(torch.from_numpy(layer.gamma.numpy()), bn_layer.weight)
            copy_tensor(torch.from_numpy(layer.beta.numpy()), bn_layer.bias)
            copy_tensor(torch.from_numpy(layer.moving_mean.numpy()), bn_layer.running_mean)
            copy_tensor(torch.from_numpy(layer.moving_variance.numpy()), bn_layer.running_var)
            dmc_layers.append(bn_layer)

        elif isinstance(layer, tf.keras.layers.ReLU):
            dmc_layers.append(ReLU())
        elif isinstance(layer, tf.keras.layers.Flatten):
            prev_layer_is_flatten = True
            dmc_layers.append(Flatten())
        elif isinstance(layer, tf.keras.layers.MaxPool2D):
            stride = layer.strides[0]
            kernel_size = layer.pool_size[0]
            dmc_layers.append(MaxPool2d(kernel_size=kernel_size, stride=stride))
        else: 
            raise RuntimeError(f"Unknown layer type: {type(layer)}")
        
    return Sequential(*dmc_layers)

# --- 4. TFLite Helper Functions ---
def convert_tf_to_tflite(tf_model, scheme=QuantizationScheme.NONE, test_data=None):
    """Converts Keras model to TFLite flatbuffer."""
    converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
    
    if scheme == QuantizationScheme.DYNAMIC:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    if scheme == QuantizationScheme.STATIC:        
        (train_img, _) = test_data
        def representative_dataset():
            for i in range(100): # Use 100 batches for calibration
                yield [train_img[i*32:(i+1)*32]]
                
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    return converter.convert()

def get_tflite_model_accuracy(tflite_model, test_data, scheme=QuantizationScheme.NONE):
    """Evaluates a TFLite flatbuffer model."""
    (_, test_lbl) = test_data
    (test_img, _) = test_data
    
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    tflite_predicted = []
    
    for image in tqdm(test_img, desc="Evaluating TFLite Model"):
        if scheme == QuantizationScheme.STATIC:
            scale, zero_point = input_details["quantization"]
            image = ((image / scale) + zero_point).astype(np.int8)
        
        interpreter.set_tensor(input_details["index"], image.reshape(1, 28, 28, 1))
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details["index"])
        tflite_predicted.append(np.argmax(output_data))

    tflite_predicted = np.array(tflite_predicted)
    return (tflite_predicted == test_lbl).sum() / len(test_lbl)

# --- 5. PyTorch (DMC) Accuracy Helper ---
def accuracy_fun(y_pred, y_true):
    return (y_pred.argmax(dim=1) == y_true).sum().item()

# --- Main Execution ---
if __name__ == "__main__":
    
    results = []
    
    # --- Load Data ---
    tf_train_data, tf_test_data, torch_train_loader, torch_test_loader = get_data_loaders()
    
    # --- Get Baseline Models ---
    tf_model = get_tf_model(tf_train_data, tf_test_data)
    dmc_base_model = convert_tf_to_dmc(tf_model).to(DEVICE)
    dmc_metrics = {"acc": accuracy_fun}

    # --- STAGE 1: NO QUANTIZATION (Float32) ---
    print("\n--- STAGE 2: Running Float32 (No Quantization) Comparison ---")
    
    # TFLite (Float)
    tflite_float_model = convert_tf_to_tflite(tf_model, QuantizationScheme.NONE)
    tflite_float_acc = get_tflite_model_accuracy(tflite_float_model, tf_test_data, QuantizationScheme.NONE)
    tflite_float_size = len(tflite_float_model)
    results.append(("TFLite (Float32)", tflite_float_acc, tflite_float_size))

    # DMC (Float)
    dmc_float_model = dmc_base_model.init_compress({
        "quantize": {"scheme": QuantizationScheme.NONE, "bitwidth": None, "granularity": None}
        }, INPUT_SHAPE_TORCH)
    dmc_float_eval = dmc_float_model.evaluate(torch_test_loader, dmc_metrics, device=DEVICE)
    dmc_float_size = dmc_float_model.get_size_in_bits() // 8
    results.append(("DMC (Float32)", dmc_float_eval['acc'], dmc_float_size))
    
    # --- STAGE 2: DYNAMIC QUANTIZATION ---
    print("\n--- STAGE 3: Running Dynamic Quantization Comparison ---")
    
    # TFLite (Dynamic)
    tflite_dyn_model = convert_tf_to_tflite(tf_model, QuantizationScheme.DYNAMIC)
    tflite_dyn_acc = get_tflite_model_accuracy(tflite_dyn_model, tf_test_data, QuantizationScheme.DYNAMIC)
    tflite_dyn_size = len(tflite_dyn_model)
    results.append(("TFLite (Dynamic)", tflite_dyn_acc, tflite_dyn_size))
    
    # DMC (Dynamic)
    dmc_dyn_model = dmc_base_model.init_compress({
        "quantize": {"scheme": QuantizationScheme.DYNAMIC, "bitwidth": 8, "granularity": QuantizationGranularity.PER_TENSOR}
    }, INPUT_SHAPE_TORCH)
    dmc_dyn_eval = dmc_dyn_model.evaluate(torch_test_loader, dmc_metrics, device=DEVICE)
    dmc_dyn_size = dmc_dyn_model.get_size_in_bits() // 8
    results.append(("DMC (Dynamic)", dmc_dyn_eval['acc'], dmc_dyn_size))

    # --- STAGE 3: STATIC QUANTIZATION ---
    print("\n--- STAGE 4: Running Static Quantization (INT8) Comparison ---")
    
    # TFLite (Static)
    tflite_static_model = convert_tf_to_tflite(tf_model, QuantizationScheme.STATIC, tf_train_data)
    tflite_static_acc = get_tflite_model_accuracy(tflite_static_model, tf_test_data, QuantizationScheme.STATIC)
    tflite_static_size = len(tflite_static_model)
    results.append(("TFLite (Static)", tflite_static_acc, tflite_static_size))
    
    # DMC (Static)
    calib_data_torch = next(iter(torch_train_loader))[0].to(DEVICE)
    dmc_static_model = dmc_base_model.init_compress({
        "quantize": {"scheme": QuantizationScheme.STATIC, "bitwidth": 8, "granularity": QuantizationGranularity.PER_TENSOR}
    }, INPUT_SHAPE_TORCH, calibration_data=calib_data_torch)
    dmc_static_eval = dmc_static_model.evaluate(torch_test_loader, dmc_metrics, device=DEVICE)
    dmc_static_size = dmc_static_model.get_size_in_bits() // 8
    results.append(("DMC (Static)", dmc_static_eval['acc'], dmc_static_size))

    # --- Print Final Summary Table ---
    print("\n\n--- REPRODUCTION FINISHED: TFLITE vs. DMC (TABLE 4) ---")
    print("=" * 60)
    print(f"{'Method':<20} | {'Accuracy (%)':<15} | {'Size (Bytes)':<15}")
    print("-" * 60)
    for name, acc, size in results:
        print(f"{name:<20} | {acc * 100:<15.2f} | {size:<15}")
    print("=" * 60)