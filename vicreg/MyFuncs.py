import torch
from torch.utils.data import DataLoader

import Datasets as datasets_root
import periodic_eval
import Utils as root_utils

import MyModels
import MyUtils
from vicreg_loss import vicreg_loss


def _extract_eval_features(encoder, x):
    return encoder(x).mean(dim=1)


def Train(dataloader, encoder, expander, opt, lr_scheduler, wd_scheduler,
          checkpoint_state, args):
    """Real VICReg: paired augmented views, loss = inv + var + cov."""
    device = args.device
    epoch_losses = []
    global_step = checkpoint_state["global_step"]
    start_epoch = checkpoint_state["start_epoch"]
    run_dir = checkpoint_state["run_dir"]
    best_acc = checkpoint_state.get("best_acc", float("-inf"))
    eval_history = list(checkpoint_state.get("eval_history", []))

    for _ in range(global_step):
        lr_scheduler.step(); wd_scheduler.step()

    print(f"\n{'='*70}\n TRAINING SETUP (VICReg)\n{'='*70}")
    n_enc = sum(p.numel() for p in encoder.parameters())
    n_exp = sum(p.numel() for p in expander.parameters())
    print(f" Encoder: {n_enc/1e6:.2f}M | Expander: {n_exp/1e6:.2f}M")
    print(f" Weights: inv={args.vicreg_lambda_inv} var={args.vicreg_lambda_var} "
          f"cov={args.vicreg_lambda_cov}\n{'='*70}\n")

    for epoch in range(start_epoch, args.epochs):
        encoder.train(); expander.train()
        pbar = root_utils.make_epoch_progress_bar(dataloader, epoch, args)
        epoch_loss = ep_inv = ep_var = ep_cov = 0.0
        n_batches = 0

        for view1, view2, _ in pbar:
            view1, view2 = view1.to(device), view2.to(device)

            z1 = expander(encoder(view1).mean(dim=1))
            z2 = expander(encoder(view2).mean(dim=1))
            loss, stats = vicreg_loss(
                z1, z2, lambda_inv=args.vicreg_lambda_inv,
                lambda_var=args.vicreg_lambda_var,
                lambda_cov=args.vicreg_lambda_cov,
                gamma=args.vicreg_gamma, eps=args.vicreg_eps)

            lr_scheduler.step(); wd_scheduler.step()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            global_step += 1
            epoch_loss += loss.item(); ep_inv += stats["inv"]
            ep_var += stats["var"]; ep_cov += stats["cov"]
            n_batches += 1
            pbar.set_postfix(loss=f"{epoch_loss/n_batches:.3f}")

        root_utils.finish_epoch_progress_bar(pbar)
        epoch_loss /= max(n_batches, 1)
        epoch_losses.append(epoch_loss)
        print(f"Epoch {epoch+1} | loss={epoch_loss:.4f} "
              f"| inv={ep_inv/max(n_batches,1):.4f} "
              f"var={ep_var/max(n_batches,1):.4f} "
              f"cov={ep_cov/max(n_batches,1):.4f}")

        models = {"encoder": encoder, "expander": expander}
        eval_history, best_acc = periodic_eval.maybe_eval_epoch(
            epoch, encoder, args, eval_history, _extract_eval_features,
            best_acc=best_acc, models=models, opt=opt,
            run_dir=run_dir, global_step=global_step,
        )
        checkpoint_state["eval_history"] = eval_history
        checkpoint_state["best_acc"] = best_acc
        MyUtils.save_epoch(run_dir, models, opt, epoch, global_step,
                            best_acc, eval_history)

    return epoch_losses, eval_history


def run_linear_probing(folder_name, args):
    ckpt_path = MyUtils.resolve_ckpt_path(folder_name, args)
    train_set, test_set = datasets_root.load_eval_dataset(args)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)
    encoder = MyUtils.load_frozen_encoder(ckpt_path, args)
    fe = MyModels.FeatureExtractor(encoder)
    return MyUtils.linear_probe(
        fe, train_loader, test_loader,
        num_classes=datasets_root.eval_num_classes(args),
        lr=args.eval_lr, epochs=args.eval_epochs, device=args.device)


def run_knn_evaluation(folder_name, args):
    ckpt_path = MyUtils.resolve_ckpt_path(folder_name, args)
    train_set, test_set = datasets_root.load_eval_dataset(args)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    encoder = MyUtils.load_frozen_encoder(ckpt_path, args)
    fe = MyModels.FeatureExtractor(encoder)
    return MyUtils.knn_evaluate(
        fe, train_loader, test_loader,
        k=args.K, num_classes=datasets_root.eval_num_classes(args), device=args.device)
