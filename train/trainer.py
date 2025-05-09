import inspect
import math
import os
import time
from collections import defaultdict
from functools import partial
from test import evaluate_ppl

import torch
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_scheduler

def cosine_scheduler(
    optimizer, num_warmup_steps, num_training_steps, lr, min_lr, last_epoch=-1
):
    def _get_cosine_schedule_with_warmup_lr_lambda(
        current_step: int,
        *,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float,
        lr: float,
        min_lr: float,
    ):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        if current_step >= num_training_steps:
            return min_lr
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        coeff = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        return min_lr + coeff * (lr - min_lr)

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=0.5,
        lr=lr,
        min_lr=min_lr,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def trainer(
    cfg,
    train_func,
    base_lm,
    train_loader,
    val_loaders,
    logger,
    accelerator,
    concept_extractor=None,
    start_step=0,
    optimizer_state=None,
    scheduler_state=None,
):
    main_process = accelerator.is_main_process

    trainable_params = list(base_lm.parameters())
    fused_available = (
        "fused" in inspect.signature(torch.optim.AdamW).parameters
        and "gpt2" in cfg.base_model
    )
    if fused_available:
        logger.log("Fused AdamW is available")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        fused=fused_available,
    )
    scheduler = get_scheduler(
        name=cfg.lr_schedule,
        optimizer=optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=cfg.train_steps,
        scheduler_specific_kwargs={"min_lr": cfg.min_lr},
    )

    # ✅ 加载 optimizer 和 scheduler 状态（使用 main.py传进来的）
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        logger.log(f"✅ Loaded optimizer state from checkpoint")

    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
        logger.log(f"✅ Loaded scheduler state from checkpoint")

    optimizer, train_loader = accelerator.prepare(optimizer, train_loader)
    if len(val_loaders) > 0:
        val_loaders = {k: accelerator.prepare(v) for k, v in val_loaders.items()}

    kwargs = {}
    if concept_extractor is not None:
        kwargs["concept_extractor"] = concept_extractor

    logger.log_dirname(f"Start training")
    metrics_dic = defaultdict(lambda: [])

    if cfg.use_torch_compile:
        logger.log("Using torch compile... after first ppl evaluation it may take sometime to run...")

    for i_epoch in range(0, cfg.n_epochs):
        for local_step, batch in enumerate(train_loader, start=start_step):
            if (
                cfg.global_step != 0
                and cfg.global_step % cfg.save_step_freq == 0
                and local_step % cfg.grad_acc_steps == 0
            ):
                accelerator.wait_for_everyone()
                base_lm_origin = base_lm._orig_mod if cfg.use_torch_compile else base_lm
                unwrapped_model = accelerator.unwrap_model(base_lm_origin)
                unwrapped_model.save_pretrained(
                    os.path.join(logger.logdir, f"step_{cfg.global_step}"),
                    is_main_process=main_process,
                    save_function=accelerator.save,
                    state_dict=accelerator.get_state_dict(base_lm_origin, unwrap=False),
                )
                if main_process:
                    torch.save(optimizer.state_dict(), os.path.join(logger.logdir, f"step_{cfg.global_step}", "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(logger.logdir, f"step_{cfg.global_step}", "scheduler.pt"))
                    last_ckpt = {
                        "model": unwrapped_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "step": cfg.global_step,
                    }
                    torch.save(last_ckpt, os.path.join(logger.logdir, "last.ckpt"))
                    logger.log(f"💾 Saved last.ckpt at step {cfg.global_step}")
                accelerator.wait_for_everyone()

            if (
                cfg.global_step % cfg.eval_step_freq == 0
                and local_step % cfg.grad_acc_steps == 0
                and val_loaders is not None
            ):
                logger.log("-" * 50)
                logger.log("Start ppl evaluation")
                results = {}
                stime = time.time()
                for val_name, val_loader in val_loaders.items():
                    results[f"{val_name}.ppl"] = evaluate_ppl(
                        cfg, base_lm, val_loader, accelerator, eval_limit=cfg.eval_limit
                    )
                logger.log(f"Eval time for ppl: {time.time() - stime:.2f}s")
                logger.wandb_log(results, step=cfg.global_step)
                for k, v in results.items():
                    logger.log(f"Step {cfg.global_step}: Eval {k}: {v}")
                logger.log("End lm evaluation")
                logger.log("-" * 50)
                torch.cuda.empty_cache()

            train_func(
                cfg,
                base_lm,
                optimizer,
                scheduler,
                accelerator,
                batch,
                logger,
                metrics_dic,
                **kwargs,
            )

            if cfg.global_step >= cfg.train_steps:
                break
        if cfg.global_step >= cfg.train_steps:
            break

    accelerator.wait_for_everyone()
    base_lm_origin = base_lm._orig_mod if cfg.use_torch_compile else base_lm
    unwrapped_model = accelerator.unwrap_model(base_lm_origin)
    unwrapped_model.save_pretrained(
        os.path.join(logger.logdir, "last"),
        is_main_process=main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(base_lm_origin, unwrap=False),
    )
    if main_process:
        torch.save(optimizer.state_dict(), os.path.join(logger.logdir, "last", "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(logger.logdir, "last", "scheduler.pt"))


