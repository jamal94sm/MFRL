import os
import torch
from torch.utils.data import DataLoader

import checkpoint_init
import Utils as root_utils
import Datasets as datasets_root

import MyFuncs
import MyModels
import MyUtils
from paired_dataset import TwoViewSTL10

baseline_name = "VICReg_baseline"


def run(args):
    args.baseline_name = baseline_name
    ckpt_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")

    encoder = MyModels.Encoder(
        (args.image_size[0], args.image_size[1]), args.num_patches,
        args.embed_dim, depth=args.encoder_depth, num_heads=args.heads,
    ).to(args.device)
    expander = MyModels.Expander(
        args.embed_dim, args.projector_hidden_dim, args.projector_out_dim,
    ).to(args.device)
    models = {"encoder": encoder, "expander": expander}

    checkpoint_init.maybe_load_initialization(
        args, models, profile="vicreg", method_name=baseline_name)

    if root_utils.maybe_eval_only(
        encoder, args, MyFuncs._extract_eval_features, models, ckpt_base
    ):
        return

    # Same root path the proposed method resolves to -- same underlying
    # images, independent of that loader's internal transform.
    dataset_key = datasets_root.normalize_dataset_name(args.dataset)
    root = os.path.join(datasets_root.data_bank_root(args), dataset_key)
    train_ds = TwoViewSTL10(root=root, split=args.stl10_split,
                             img_size=args.image_size[0], download=True)
    dataloader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=False, drop_last=True,
    )

    opt = torch.optim.AdamW(
        list(encoder.parameters()) + list(expander.parameters()),
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    total_steps = args.epochs * len(dataloader)
    lr_scheduler = MyUtils.WarmupCosineSchedule(
        optimizer=opt, warmup_steps=int(args.warmup_ratio * total_steps),
        start_lr=args.start_lr, ref_lr=args.learning_rate,
        total_steps=total_steps, final_lr=args.final_lr)
    wd_scheduler = MyUtils.CosineWDSchedule(
        optimizer=opt, ref_wd=args.weight_decay,
        total_steps=total_steps, final_wd=args.final_weight_decay)

    args.ckpt_run_name = getattr(args, "ckpt_run_name", None) or "VICReg"
    checkpoint_state = MyUtils.prepare_checkpoint_state(models, opt, ckpt_base, args)
    print(f"Checkpoints → {checkpoint_state['run_dir']}")
    print(f"Train set: {len(train_ds)} images (split={args.stl10_split})")

    epoch_losses, eval_history = MyFuncs.Train(
        dataloader, encoder, expander, opt, lr_scheduler, wd_scheduler,
        checkpoint_state, args)

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")
    MyUtils.Plot(epoch_losses, plot_name=baseline_name, results_dir=results_dir)
    plot_path = root_utils.plot_eval_progress(
        eval_history, baseline_name, args.evaluation, results_dir=results_dir)
    if plot_path:
        print(f"Saved eval progress plot: {plot_path}")
    args.results_dir = results_dir
    tsne_path = root_utils.live_eval_tsne(
        encoder, args, MyFuncs._extract_eval_features, args.epochs)
    print(f"Saved t-SNE: {tsne_path}")
