"""
One-shot generator: writes search.ipynb and reproduce.ipynb for all three
model experiments.  Run from the paper_experiments/ directory:

    python _generate_notebooks.py
"""
import json
import os


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {"name": "python", "version": "3.12.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": list(lines)}

def code(*lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": list(lines),
    }

def write_nb(path, cells):
    with open(path, "w") as f:
        json.dump(nb(cells), f, indent=1)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Model-specific config
# ---------------------------------------------------------------------------

MODELS = {
    "lenet5_mnist": {
        "title":        "LeNet-5 / MNIST",
        "model_import": "from development.experiments.lenet5  import get_model",
        "data_import":  "from development.experiments.mnist   import get_data_loaders, get_metric",
        "input_shape":  "(1, 28, 28)",
        "hardware_var": "ATMEGA328P",
        "board":        "ATmega328P (Arduino Uno)",
        "model_key":    "lenet5_model",
        "pretrained":   False,   # must load from models/baseline.pth
        "retrain_epochs": 40,
        "baseline_epochs": 30,
    },
    "vgg13_cifar100": {
        "title":        "VGG-13 / CIFAR-100",
        "model_import": "from development.experiments.vgg13   import get_model",
        "data_import":  "from development.experiments.cifar100 import get_data_loaders, get_metric",
        "input_shape":  "(3, 32, 32)",
        "hardware_var": "ATMEGA2560",
        "board":        "ATmega2560 (Arduino Mega)",
        "model_key":    "vgg13_model",
        "pretrained":   True,
        "retrain_epochs": 40,
        "baseline_epochs": None,
    },
    "resnet56_cifar100": {
        "title":        "ResNet-56 / CIFAR-100",
        "model_import": "from development.experiments.resnet   import get_model",
        "data_import":  "from development.experiments.cifar100 import get_data_loaders, get_metric",
        "input_shape":  "(3, 32, 32)",
        "hardware_var": "ATMEGA2560",
        "board":        "ATmega2560 (Arduino Mega)",
        "model_key":    "resnet56_model",
        "pretrained":   True,
        "retrain_epochs": 40,
        "baseline_epochs": None,
    },
}


# ---------------------------------------------------------------------------
# search.ipynb template
# ---------------------------------------------------------------------------

def make_search_nb(cfg):
    hw   = cfg["hardware_var"]
    isp  = cfg["input_shape"]
    title = cfg["title"]

    cells = [
        md(f"# NAS Search — {title}\n",
           f"Loads accumulated NAS samples from `nas_data/`, trains an MLP "
           f"accuracy estimator, then runs evolutionary search for two "
           f"compression objectives:\n",
           f"- **DMC-Ultra**: minimise weight storage subject to fitting on "
           f"the {cfg['board']}.\n",
           f"- **DMC-Board**: maximise accuracy subject to the same hardware "
           f"constraint.\n"),

        # ── 1. Imports & paths ──────────────────────────────────────────────
        code(
            "import os, sys, json, torch\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "_HERE      = os.path.dirname(os.path.abspath('.'))\n",
            "_PROJ_ROOT = os.path.abspath(os.path.join('.', '../../..'))\n",
            "_SHARED    = os.path.abspath(os.path.join('.', '../shared'))\n",
            "sys.path.insert(0, _PROJ_ROOT)\n",
            "sys.path.insert(0, _SHARED)\n",
            "\n",
            "from development import (\n",
            "    ConfigEncoder, evolutionary_search_compression_config,\n",
            "    QuantizationScheme,\n",
            ")\n",
            f"{cfg['model_import']}\n",
            f"{cfg['data_import']}\n",
            "from hardware_specs import " + hw + ", make_ultra_condition, make_board_condition\n",
            "from nas_utils      import load_nas_data, train_estimator, make_estimate_fn\n",
        ),

        # ── 2. Config ───────────────────────────────────────────────────────
        code(
            "DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
            "SEED        = 25\n",
            f"INPUT_SHAPE = {isp}\n",
            f"HARDWARE    = {hw}\n",
            "NAS_DATA_DIR = 'nas_data'\n",
            "MODELS_DIR   = 'models'\n",
            "os.makedirs(MODELS_DIR, exist_ok=True)\n",
            "\n",
            "torch.manual_seed(SEED)\n",
            "print(f'Device: {DEVICE}')\n",
        ),

        # ── 3. Load NAS data ────────────────────────────────────────────────
        md("## 1 — Load & aggregate NAS samples"),
        code(
            "nas_data = load_nas_data(NAS_DATA_DIR)\n",
            "print('Keys:', list(nas_data.keys())[:4], '...')\n",
        ),

        # ── 4. Build model + encoder ────────────────────────────────────────
        md("## 2 — Build baseline model and ConfigEncoder"),
        code(
            "print('Loading model …')\n",
            "baseline_model = get_model().to(DEVICE)\n",
            "nas_encoder    = ConfigEncoder(baseline_model)\n",
        ),

        # ── 5. Encode data ──────────────────────────────────────────────────
        md("## 3 — Encode NAS data"),
        code(
            "encoded_data = nas_encoder.encode(nas_data, with_metric=True)\n",
            "print(f'Encoded shape: {encoded_data.shape}   "
            "(rows=samples, cols=features+1 metric)')\n",
        ),

        # ── 6. Train estimator ──────────────────────────────────────────────
        md("## 4 — Train accuracy estimator"),
        code(
            "print('Training estimator …')\n",
            "estimator, x_mu, x_std, y_mu, y_std, history = train_estimator(\n",
            "    encoded_data,\n",
            "    device     = DEVICE,\n",
            "    hidden_dim = 256,\n",
            "    dropout    = 0.2,\n",
            "    epochs     = 2000,\n",
            "    batch_size = 128,\n",
            "    val_split  = 0.2,\n",
            "    seed       = SEED,\n",
            ")\n",
        ),

        # ── 7. Plot training ────────────────────────────────────────────────
        md("## 5 — Estimator training curves"),
        code(
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n",
            "axes[0].plot(history['train_loss'][50:], label='train')\n",
            "axes[0].plot(history['val_loss'][50:],   label='val')\n",
            "axes[0].set_title('Huber loss'); axes[0].legend()\n",
            "\n",
            "axes[1].plot(history['train_mae'][50:], label='train')\n",
            "axes[1].plot(history['val_mae'][50:],   label='val')\n",
            "axes[1].set_title('MAE (accuracy %)'); axes[1].legend()\n",
            "plt.tight_layout(); plt.show()\n",
            "\n",
            "print(f'Final val MAE: {history[\"val_mae\"][-1]:.3f}%')\n",
        ),

        # ── 8. Calibration scatter ──────────────────────────────────────────
        code(
            "# Predicted vs actual on the full encoded dataset\n",
            "import torch\n",
            "X_all = ((encoded_data[:, :-1].float() - x_mu) / x_std).to(DEVICE)\n",
            "with torch.no_grad():\n",
            "    Y_pred = (estimator(X_all) * y_std.to(DEVICE) + y_mu.to(DEVICE)).cpu().numpy()\n",
            "Y_true = encoded_data[:, -1].numpy()\n",
            "\n",
            "plt.figure(figsize=(5, 5))\n",
            "plt.scatter(Y_true, Y_pred, s=8, alpha=0.6)\n",
            "lims = [min(Y_true.min(), Y_pred.min()), max(Y_true.max(), Y_pred.max())]\n",
            "plt.plot(lims, lims, 'r--', lw=1)\n",
            "plt.xlabel('True accuracy (%)'); plt.ylabel('Predicted accuracy (%)')\n",
            "plt.title('Estimator calibration'); plt.tight_layout(); plt.show()\n",
        ),

        # ── 9. Prepare search ───────────────────────────────────────────────
        md("## 6 — Evolutionary search"),
        code(
            "train_loader, test_loader = get_data_loaders()\n",
            "calibration_data = next(iter(train_loader))[0].to(DEVICE)\n",
            "\n",
            "metric_fn = get_metric()\n",
            "baseline_acc = baseline_model.fuse(device=DEVICE).evaluate(\n",
            "    test_loader, {'acc': metric_fn}, device=DEVICE\n",
            ")['acc']\n",
            "print(f'Baseline accuracy: {baseline_acc:.2f}%')\n",
            "\n",
            "estimate = make_estimate_fn(estimator, nas_encoder, x_mu, x_std, y_mu, y_std, DEVICE)\n",
            "original_size = baseline_model.fuse(device=DEVICE).get_size_in_bytes()\n",
        ),

        # ── 10. DMC-Ultra search ─────────────────────────────────────────────
        md("### DMC-Ultra — minimise weight storage"),
        code(
            "ultra_condition = make_ultra_condition(HARDWARE)\n",
            "\n",
            "best_ultra_raw, ultra_info = evolutionary_search_compression_config(\n",
            "    baseline_model,\n",
            "    estimate,\n",
            "    INPUT_SHAPE,\n",
            "    calibration_data,\n",
            "    condition  = ultra_condition,\n",
            "    objective  = lambda metric, size, ram, cfg: size,\n",
            "    maximize   = False,\n",
            "    verbose    = True,\n",
            "    population_size = 75,\n",
            "    generations     = 75,\n",
            ")\n",
            "\n",
            "ultra_config = baseline_model.decode_compression_dict_hyperparameter(best_ultra_raw)\n",
            "print('\\nDMC-Ultra config:')\n",
            "print(json.dumps({k: str(v) for k, v in ultra_config.items()}, indent=2))\n",
            "print('Search info:', ultra_info)\n",
        ),

        # ── 11. DMC-Board search ─────────────────────────────────────────────
        md("### DMC-Board — maximise accuracy within board constraints"),
        code(
            "board_condition = make_board_condition(HARDWARE, baseline_acc, max_drop=5.0)\n",
            "\n",
            "best_board_raw, board_info = evolutionary_search_compression_config(\n",
            "    baseline_model,\n",
            "    estimate,\n",
            "    INPUT_SHAPE,\n",
            "    calibration_data,\n",
            "    condition  = board_condition,\n",
            "    objective  = lambda metric, size, ram, cfg: metric,\n",
            "    maximize   = True,\n",
            "    verbose    = True,\n",
            "    population_size = 75,\n",
            "    generations     = 75,\n",
            ")\n",
            "\n",
            "board_config = baseline_model.decode_compression_dict_hyperparameter(best_board_raw)\n",
            "print('\\nDMC-Board config:')\n",
            "print(json.dumps({k: str(v) for k, v in board_config.items()}, indent=2))\n",
            "print('Search info:', board_info)\n",
        ),

        # ── 12. Save configs ─────────────────────────────────────────────────
        md("## 7 — Save configs"),
        code(
            "torch.save(ultra_config, os.path.join(MODELS_DIR, 'dmc_ultra_config.pth'))\n",
            "torch.save(board_config, os.path.join(MODELS_DIR, 'dmc_board_config.pth'))\n",
            "print('Configs saved to models/')\n",
        ),
    ]

    return cells


# ---------------------------------------------------------------------------
# reproduce.ipynb template
# ---------------------------------------------------------------------------

def make_reproduce_nb(cfg):
    hw        = cfg["hardware_var"]
    isp       = cfg["input_shape"]
    title     = cfg["title"]
    pretrained = cfg["pretrained"]
    retrain_e  = cfg["retrain_epochs"]

    baseline_cell_lines = []
    if pretrained:
        baseline_cell_lines = [
            "# Pretrained weights loaded by get_model() from torch hub.\n",
            "print('Baseline model loaded with pretrained weights.')\n",
        ]
    else:
        baseline_cell_lines = [
            "# LeNet-5 has no pretrained weights — train from scratch.\n",
            "baseline_ckpt = os.path.join(MODELS_DIR, 'baseline.pth')\n",
            "if os.path.exists(baseline_ckpt):\n",
            "    print(f'Loading baseline from {baseline_ckpt}')\n",
            "    baseline_model.load_state_dict(\n",
            "        torch.load(baseline_ckpt, weights_only=True)['model']\n",
            "    )\n",
            "else:\n",
            f"    print('Training baseline for {cfg['baseline_epochs']} epochs …')\n",
            "    from shared.train_utils import train_baseline\n",
            f"    train_baseline(baseline_model, train_loader, test_loader, metric_fn,\n",
            f"                   epochs={cfg['baseline_epochs']}, device=DEVICE)\n",
            "    os.makedirs(MODELS_DIR, exist_ok=True)\n",
            "    torch.save({'model': baseline_model.state_dict()}, baseline_ckpt)\n",
            "    print(f'Baseline saved → {baseline_ckpt}')\n",
        ]

    cells = [
        md(f"# Reproduce — {title}\n",
           f"Trains the two compressed models (DMC-Ultra and DMC-Board) using the\n",
           f"configs found by `search.ipynb`, evaluates them, generates C headers,\n",
           f"and compiles to report binary size for the {cfg['board']}.\n"),

        # ── 1. Imports ───────────────────────────────────────────────────────
        code(
            "import os, sys, json, torch, subprocess\n",
            "from torch import nn\n",
            "\n",
            "_HERE      = os.path.dirname(os.path.abspath('.'))\n",
            "_PROJ_ROOT = os.path.abspath(os.path.join('.', '../../..'))\n",
            "_SHARED    = os.path.abspath(os.path.join('.', '../shared'))\n",
            "sys.path.insert(0, _PROJ_ROOT)\n",
            "sys.path.insert(0, _SHARED)\n",
            "\n",
            f"{cfg['model_import']}\n",
            f"{cfg['data_import']}\n",
            "from train_utils import train_compressed, evaluate_model, "
            "save_results, print_results_table\n",
        ),

        # ── 2. Config ────────────────────────────────────────────────────────
        code(
            "DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
            "SEED        = 25\n",
            f"INPUT_SHAPE = {isp}\n",
            "MODELS_DIR  = 'models'\n",
            "DEPLOY_DIR  = 'deployment'\n",
            "os.makedirs(MODELS_DIR, exist_ok=True)\n",
            "os.makedirs(DEPLOY_DIR, exist_ok=True)\n",
            "\n",
            "torch.manual_seed(SEED)\n",
            "print(f'Device: {DEVICE}')\n",
        ),

        # ── 3. Load data & model ─────────────────────────────────────────────
        md("## 1 — Data and baseline model"),
        code(
            "print('Loading dataset …')\n",
            "train_loader, test_loader = get_data_loaders()\n",
            "metric_fn        = get_metric()\n",
            "calibration_data = next(iter(train_loader))[0].to(DEVICE)\n",
            "\n",
            "print('Loading model …')\n",
            "baseline_model = get_model().to(DEVICE)\n",
        ),

        # ── 4. Baseline training (or load) ───────────────────────────────────
        md("## 2 — Baseline model (FP32)"),
        code(*baseline_cell_lines),

        # ── 5. Evaluate baseline ─────────────────────────────────────────────
        code(
            "print('Evaluating baseline …')\n",
            "baseline_results = evaluate_model(\n",
            "    baseline_model, test_loader, metric_fn, INPUT_SHAPE, DEVICE\n",
            ")\n",
            "print('Baseline:', baseline_results)\n",
        ),

        # ── 6. Load configs ──────────────────────────────────────────────────
        md("## 3 — Load compression configs from search.ipynb"),
        code(
            "ultra_config = torch.load(os.path.join(MODELS_DIR, 'dmc_ultra_config.pth'),\n",
            "                          weights_only=False)\n",
            "board_config = torch.load(os.path.join(MODELS_DIR, 'dmc_board_config.pth'),\n",
            "                          weights_only=False)\n",
            "print('Configs loaded.')\n",
        ),

        # ── 7. Train DMC-Ultra ───────────────────────────────────────────────
        md("## 4 — Train DMC-Ultra"),
        code(
            "print('Training DMC-Ultra …')\n",
            "dmc_ultra = train_compressed(\n",
            "    baseline_model, ultra_config, INPUT_SHAPE,\n",
            "    train_loader, test_loader, metric_fn, calibration_data,\n",
            f"   epochs={retrain_e}, device=DEVICE, two_step=True,\n",
            ")\n",
            "torch.save(dmc_ultra.state_dict(), os.path.join(MODELS_DIR, 'dmc_ultra.pth'))\n",
            "ultra_results = evaluate_model(dmc_ultra, test_loader, metric_fn, INPUT_SHAPE, DEVICE)\n",
            "print('DMC-Ultra:', ultra_results)\n",
        ),

        # ── 8. Train DMC-Board ───────────────────────────────────────────────
        md("## 5 — Train DMC-Board"),
        code(
            "print('Training DMC-Board …')\n",
            "dmc_board = train_compressed(\n",
            "    baseline_model, board_config, INPUT_SHAPE,\n",
            "    train_loader, test_loader, metric_fn, calibration_data,\n",
            f"   epochs={retrain_e}, device=DEVICE, two_step=True,\n",
            ")\n",
            "torch.save(dmc_board.state_dict(), os.path.join(MODELS_DIR, 'dmc_board.pth'))\n",
            "board_results = evaluate_model(dmc_board, test_loader, metric_fn, INPUT_SHAPE, DEVICE)\n",
            "print('DMC-Board:', board_results)\n",
        ),

        # ── 9. Results table ─────────────────────────────────────────────────
        md("## 6 — Results summary"),
        code(
            "all_results = {\n",
            "    'Baseline (FP32)': baseline_results,\n",
            "    'DMC-Ultra':       ultra_results,\n",
            "    'DMC-Board':       board_results,\n",
            "}\n",
            "print_results_table(all_results)\n",
            "save_results(all_results, MODELS_DIR)\n",
        ),

        # ── 10. Compression ratios ───────────────────────────────────────────
        code(
            "base_size = baseline_results['size_bytes']\n",
            "base_ws   = baseline_results['workspace_bytes']\n",
            "for name, r in all_results.items():\n",
            "    cr  = base_size / r['size_bytes']   if r['size_bytes']   else float('inf')\n",
            "    wsr = base_ws   / r['workspace_bytes'] if r['workspace_bytes'] else float('inf')\n",
            "    print(f\"{name:<20} size_CR={cr:.1f}x  workspace_CR={wsr:.1f}x\")\n",
        ),

        # ── 11. Generate C headers ───────────────────────────────────────────
        md("## 7 — Generate C headers for deployment"),
        code(
            "import random, string\n",
            "test_input = torch.rand(INPUT_SHAPE, device=DEVICE)\n",
            "\n",
            "for model_name, model in [('dmc_ultra', dmc_ultra), ('dmc_board', dmc_board)]:\n",
            "    out_dir = os.path.join(DEPLOY_DIR, model_name)\n",
            "    os.makedirs(out_dir, exist_ok=True)\n",
            "    fused = model.fuse(device=DEVICE)\n",
            f"    fused.convert_to_c(\n",
            f"        INPUT_SHAPE, '{cfg['model_key']}',\n",
            "        out_dir, out_dir,\n",
            "        for_arduino=True,\n",
            "        test_input=test_input,\n",
            "    )\n",
            "    print(f'C headers written to {out_dir}/')\n",
        ),

        # ── 12. Compile & measure binary ─────────────────────────────────────
        md("## 8 — Compile with avr-gcc and report binary size"),
        code(
            "# Requires avr-gcc to be installed: sudo apt install gcc-avr binutils-avr\n",
            "import glob\n",
            "\n",
            "def avr_compile(src_dir, mcu='atmega2560'):\n",
            "    c_files = glob.glob(os.path.join(src_dir, '*.cpp')) + \\\n",
            "              glob.glob(os.path.join(src_dir, '*.c'))\n",
            "    if not c_files:\n",
            "        print(f'  No C/C++ files found in {src_dir}'); return\n",
            "    out_elf = os.path.join(src_dir, 'model.elf')\n",
            "    cmd = ['avr-g++', f'-mmcu={mcu}', '-Os', '-o', out_elf] + c_files\n",
            "    result = subprocess.run(cmd, capture_output=True, text=True)\n",
            "    if result.returncode != 0:\n",
            "        print('avr-g++ error:', result.stderr[:500]); return\n",
            "    size_out = subprocess.run(['avr-size', out_elf],\n",
            "                              capture_output=True, text=True).stdout\n",
            "    print(f'\\n{src_dir}:')\n",
            "    print(size_out)\n",
            "\n",
            "for model_name in ['dmc_ultra', 'dmc_board']:\n",
            "    avr_compile(os.path.join(DEPLOY_DIR, model_name))\n",
        ),
    ]

    return cells


# ---------------------------------------------------------------------------
# Generate all notebooks
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))

for exp_dir, cfg in MODELS.items():
    exp_path = os.path.join(_HERE, exp_dir)
    os.makedirs(exp_path, exist_ok=True)

    search_path = os.path.join(exp_path, "search.ipynb")
    write_nb(search_path, make_search_nb(cfg))

    repro_path = os.path.join(exp_path, "reproduce.ipynb")
    write_nb(repro_path, make_reproduce_nb(cfg))

print("\nAll notebooks generated.")
