"""MyUtils.py -- self-contained (not imported from MFRL root: unlike
Datasets/Utils/periodic_eval/checkpoint_init, this can't be safely reused
as-is -- MFRL's own MyUtils.load_frozen_context_encoder hardcodes
Context_Encoder, so at least part of that file is method-specific).
Checkpoint dict shape matches the {"models": {name: state_dict}, "opt":,
"epoch":, "global_step":, "best_acc":} convention used across your other
files -- verify against checkpoint_init.py / periodic_eval.py if anything
fails to load.
"""

import os
import math
import torch
from . import MyModels


class WarmupCosineSchedule:
    def __init__(self, optimizer, warmup_steps, start_lr, ref_lr, total_steps, final_lr=0.0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.T_max = total_steps - warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        if self.step_num < self.warmup_steps:
            p = self.step_num / max(1, self.warmup_steps)
            lr = self.start_lr + p * (self.ref_lr - self.start_lr)
        else:
            p = (self.step_num - self.warmup_steps) / max(1, self.T_max)
            lr = self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1 + math.cos(math.pi * p))
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        return lr


class CosineWDSchedule:
    def __init__(self, optimizer, ref_wd, total_steps, final_wd=0.0):
        self.optimizer = optimizer
        self.ref_wd = ref_wd
        self.final_wd = final_wd
        self.total_steps = total_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        p = self.step_num / self.total_steps
        wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (1 + math.cos(math.pi * p))
        for g in self.optimizer.param_groups:
            g["weight_decay"] = wd
        return wd


def prepare_checkpoint_state(models, opt, ckpt_base, args):
    run_name = getattr(args, "ckpt_run_name", None) or "run"
    run_dir = os.path.join(ckpt_base, run_name)
    os.makedirs(run_dir, exist_ok=True)
    last_path = os.path.join(run_dir, "last.ckpt")
    if os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
        for name, m in models.items():
            if name in ckpt["models"]:
                m.load_state_dict(ckpt["models"][name])
        opt.load_state_dict(ckpt["opt"])
        return {
            "run_dir": run_dir,
            "start_epoch": ckpt["epoch"] + 1,
            "global_step": ckpt["global_step"],
            "best_acc": ckpt.get("best_acc", float("-inf")),
            "eval_history": ckpt.get("eval_history", []),
        }
    return {"run_dir": run_dir, "start_epoch": 0, "global_step": 0,
            "best_acc": float("-inf"), "eval_history": []}


def save_epoch(run_dir, models, opt, epoch, global_step, best_acc, eval_history=None):
    ckpt = {
        "models": {name: m.state_dict() for name, m in models.items()},
        "opt": opt.state_dict(),
        "epoch": epoch, "global_step": global_step, "best_acc": best_acc,
        "eval_history": eval_history or [],
    }
    torch.save(ckpt, os.path.join(run_dir, "last.ckpt"))


def resolve_ckpt_path(folder_name, args):
    ckpt_base = os.path.join(folder_name, "checkpoints")
    runs = [d for d in os.listdir(ckpt_base)
            if os.path.isdir(os.path.join(ckpt_base, d))]
    run = max(runs, key=lambda d: os.path.getmtime(os.path.join(ckpt_base, d)))
    return os.path.join(ckpt_base, run, "last.ckpt")


def load_frozen_encoder(ckpt_path, args):
    enc = MyModels.Encoder(
        image_size=(args.image_size[0], args.image_size[1]),
        num_patches=args.num_patches, embed_dim=args.embed_dim,
        depth=args.encoder_depth, num_heads=args.heads,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    enc.load_state_dict(ckpt["models"]["encoder"], strict=True)
    enc.to(args.device).eval()
    for p in enc.parameters():
        p.requires_grad = False
    return enc


def linear_probe(feature_extractor, train_loader, test_loader,
                 num_classes, lr, epochs, device):
    with torch.no_grad():
        x, _ = next(iter(train_loader))
        feat_dim = feature_extractor(x.to(device)).shape[-1]
    clf = torch.nn.Linear(feat_dim, num_classes).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        clf.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                feats = feature_extractor(x)
            loss = loss_fn(clf(feats), y)
            opt.zero_grad(); loss.backward(); opt.step()
    clf.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            feats = feature_extractor(x.to(device))
            pred = clf(feats).argmax(1).cpu()
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


@torch.no_grad()
def knn_evaluate(feature_extractor, train_loader, test_loader,
                 k, num_classes, device, temperature=0.07):
    feats, labels = [], []
    for x, y in train_loader:
        z = torch.nn.functional.normalize(feature_extractor(x.to(device)), dim=1)
        feats.append(z.cpu()); labels.append(y)
    feat_train, labels_train = torch.cat(feats), torch.cat(labels)

    correct = total = 0
    for x, y in test_loader:
        z = torch.nn.functional.normalize(feature_extractor(x.to(device)), dim=1).cpu()
        sim = z @ feat_train.T
        topk_sim, topk_idx = sim.topk(k, dim=1)
        topk_labels = labels_train[topk_idx]
        weights = torch.exp(topk_sim / temperature)
        scores = torch.zeros(z.size(0), num_classes)
        for c in range(num_classes):
            scores[:, c] = (weights * (topk_labels == c)).sum(dim=1)
        correct += (scores.argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total


def Plot(losses, plot_name, results_dir):
    import matplotlib.pyplot as plt
    os.makedirs(results_dir, exist_ok=True)
    plt.figure()
    plt.plot(losses)
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title(plot_name)
    plt.savefig(os.path.join(results_dir, f"{plot_name}_loss.png"), dpi=150)
    plt.close()