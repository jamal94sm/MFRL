"""
Config.py — VICReg on STL-10 (unlabeled) → linear/kNN eval on CIFAR-10/etc.

Same section structure as MFRL's own Config.py, pruned to what VICReg uses.
No corruption/Gabor/struct/EMA args -- VICReg has one trainable encoder,
no target network, no masking.
"""

import argparse
import torch


def get_arguments():
    parser = argparse.ArgumentParser(description="VICReg_baseline")

    parser.add_argument('--device', type=str,
        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--dataset', type=str, default='stl10',
        choices=['stl-10', 'stl10', 'tiny-imagenet'])
    parser.add_argument('--eval_dataset', type=str, default='cifar10',
        choices=['cifar10', 'cifar100', 'stl10', 'tiny-imagenet'])
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--stl10_split', type=str, default='unlabeled',
        choices=['unlabeled', 'train+unlabeled', 'train'],
        help="STL-10 split used for VICReg pretraining. Default matches "
             "the standard self-supervised protocol (100k images). Set "
             "this to whatever split MFRL's Datasets.load_stl10 actually "
             "uses if it differs, to keep the effective training set "
             "size identical between VICReg and the proposed method.")

    parser.add_argument('--num_patches', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs=2, default=[96, 96])
    parser.add_argument('--evaluation', type=str, default='knn',
        choices=['linear', 'knn'])
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--encoder_depth', type=int, default=6)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--eval_only', type=int, default=0, choices=[0, 1])
    parser.add_argument('--initialization', type=str, default=None)
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--eval_epochs', type=int, default=20)
    parser.add_argument('--eval_lr', type=float, default=1e-2)
    parser.add_argument('--num_ep_for_eval', type=int, default=1)

    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--start_lr', type=float, default=1e-6)
    parser.add_argument('--final_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--final_weight_decay', type=float, default=0.1)

    # ─── VICReg projector + loss ────────────────────────────
    parser.add_argument('--projector_hidden_dim', type=int, default=None,
        help="Defaults to embed_dim.")
    parser.add_argument('--projector_out_dim', type=int, default=None,
        help="Defaults to embed_dim.")
    parser.add_argument('--vicreg_lambda_inv', type=float, default=25.0)
    parser.add_argument('--vicreg_lambda_var', type=float, default=25.0)
    parser.add_argument('--vicreg_lambda_cov', type=float, default=1.0)
    parser.add_argument('--vicreg_gamma', type=float, default=1.0)
    parser.add_argument('--vicreg_eps', type=float, default=1e-4)

    args = parser.parse_args()
    return args