import os
import torch
import torch._dynamo
from safetensors.torch import load_file
from accelerate import Accelerator
import torch.distributed as dist
from omegaconf import OmegaConf
import hydra
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.distributed.fsdp._runtime_utils import _lazy_init
import torch.distributed.fsdp._runtime_utils as fsdp_runtime_utils

from data.data import get_train_dataloader, get_val_dataloaders
from models import get_base_lm, get_concept_extractor
from utils import Logger, set_random_seed
from train.trainer import trainer
from train import setup as train_setup

@hydra.main(config_path="conf", config_name="config", version_base="1.3.2")
def main(cfg):
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.grad_acc_steps,
    )
    accelerator.wait_for_everyone()

    num_gpus = dist.get_world_size()
    cfg.distributed = num_gpus > 1
    cfg.world_size = num_gpus

    set_random_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if cfg.use_torch_compile:
        torch._dynamo.config.cache_size_limit = cfg.compile_dynamo_cache_size_limit

    # 准备数据集
    train_loader = get_train_dataloader(cfg)
    val_loaders = get_val_dataloaders(cfg)

    # 创建裸模型
    base_lm = get_base_lm(cfg)

    # ✅ 设置 FSDP state dict policy
    FSDP.set_state_dict_type(base_lm, StateDictType.FULL_STATE_DICT)

    # 显式转 float16
    base_lm = base_lm.to(torch.float16)

    # 概念提取器
    concept_extractor = get_concept_extractor(cfg, accelerator)

    train_func, fname, wandb_name = train_setup(cfg.mode, cfg)

    logger = Logger(
        fname,
        cfg,
        main_process=accelerator.is_main_process,
        use_wandb=cfg.wandb_log,
        wandb_name=wandb_name,
        log_path=cfg.log_path,
    )
    logger.log(OmegaConf.to_yaml(cfg))

    start_step = 0
    optimizer_state = None
    scheduler_state = None
    saved_state_dict = None

    # 🛠️ 加载 checkpoint (只保存 state_dict，不立刻load)
    if cfg.get("load_path", None):
        if os.path.exists(cfg.load_path):
            model_path = os.path.join(cfg.load_path, "model.safetensors")
            optimizer_path = os.path.join(cfg.load_path, "optimizer.pt")
            scheduler_path = os.path.join(cfg.load_path, "scheduler.pt")

            if os.path.exists(model_path):
                saved_state_dict = load_file(model_path, device="cpu")
                logger.log(f"✅ Loaded model weights (deferred) from {model_path}")

            if os.path.exists(optimizer_path):
                optimizer_state = torch.load(optimizer_path, map_location="cpu")
                logger.log(f"✅ Loaded optimizer state from {optimizer_path}")

            if os.path.exists(scheduler_path):
                scheduler_state = torch.load(scheduler_path, map_location="cpu")
                logger.log(f"✅ Loaded scheduler state from {scheduler_path}")

            try:
                step_str = os.path.basename(cfg.load_path)
                step_num = int(step_str.replace("step_", ""))
                start_step = step_num
                cfg.global_step = start_step
                logger.log(f"✅ Resuming from step {start_step}")
            except Exception:
                logger.log("⚠️ Failed to parse step number from load_path")

    # 现在才 prepare
    base_lm = accelerator.prepare(base_lm)
    print(type(base_lm))

    # 🛠️ prepare之后再 load weights
    #if saved_state_dict is not None:
    #    unwrapped_model = accelerator.unwrap_model(base_lm)
    #    if hasattr(unwrapped_model, "module"):
    #        unwrapped_model = unwrapped_model.module

    #    if cfg.distributed:  # 只有真正多卡训练时，才需要 set_state_dict_type
    #        FSDP.set_state_dict_type(unwrapped_model, StateDictType.FULL_STATE_DICT)

    #    unwrapped_model.load_state_dict(saved_state_dict, strict=False)

        # _lazy_init(unwrapped_model._fsdp_state, unwrapped_model)

    #    logger.log(f"✅ Reloaded model weights after prepare")
    if saved_state_dict is not None:
        unwrapped_model = accelerator.unwrap_model(base_lm)
        FSDP.set_state_dict_type(unwrapped_model, StateDictType.FULL_STATE_DICT)
        unwrapped_model.load_state_dict(saved_state_dict, strict=False)
        logger.log(f"✅ Reloaded model weights after prepare")


    # compile (prepare之后)
    if cfg.use_torch_compile:
        base_lm = torch.compile(base_lm)
        logger.log("✅ Compiled model")

    # 启动训练
    trainer(
        cfg,
        train_func,
        base_lm,
        train_loader,
        val_loaders,
        logger,
        accelerator,
        concept_extractor,
        start_step=start_step,
        optimizer_state=optimizer_state,
        scheduler_state=scheduler_state,
    )

    logger.close_writer()

if __name__ == "__main__":
    main()

