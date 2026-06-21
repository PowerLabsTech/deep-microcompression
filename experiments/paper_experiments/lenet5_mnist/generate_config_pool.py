#!/usr/bin/env python3
import os, sys
_HERE   = os.path.dirname(os.path.abspath(__file__))
_SHARED = os.path.abspath(os.path.join(_HERE, "../shared"))
sys.path.insert(0, _SHARED)
from generate_config_pool import main
main(defaults={
    "model":    "lenet5",
    "out_dir":  _HERE,
    "baseline": os.path.join(_HERE, "models", "baseline.pth"),
})
