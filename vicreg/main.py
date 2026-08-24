#!/usr/bin/env python3
"""VICReg_baseline — run from this folder: python main.py"""

import os
import sys
import warnings

import Utils
from Config import get_arguments
import runner

warnings.filterwarnings("ignore", message="enable_nested_tensor is True")

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(here)
    os.chdir(parent)  # so `import Datasets`, `Utils`, etc. resolve to MFRL root
    if parent not in sys.path:
        sys.path.insert(0, parent)
    if here not in sys.path:
        sys.path.insert(0, here)

    args = get_arguments()
    Utils.set_seed(42)

    results_dir = os.path.join(here, "Results")
    os.makedirs(results_dir, exist_ok=True)
    print("=" * 40, "VICReg_baseline", "=" * 40)
    print(f"Views: official VICReg two-branch recipe (asymmetric blur/solarize)")
    print(f"Loss: {args.vicreg_lambda_inv}*inv + {args.vicreg_lambda_var}*var "
          f"+ {args.vicreg_lambda_cov}*cov")
    runner.run(args)
    print("=" * 40, "Done", "=" * 40)