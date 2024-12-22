import os
import platform
import argparse
import time
import math
import warnings
import shutil

import pandas as pd
import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext

from transformers import AutoTokenizer

from src.models.model import MiniChat
from src.models.model_config import MiniChatConfig
from src.models.datasets import PretrainDataset
import logging

# 配置日志格式和输出级别
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
warnings.filterwarnings("ignore")


def get_available_device():
    """
    判断当前环境下 CUDA、MPS 和 CPU 中哪个可用，并返回设备名称。
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def log(content):
    if not ddp or dist.get_rank() == 0:
        logging.info(content)


def get_lr(it, all):
    warmup_iters = args.warmup_iters
    lr_decay_iters = all
    min_lr = args.learning_rate / 10

    if it < warmup_iters:
        return args.learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (args.learning_rate - min_lr)


def train_epoch(epoch):
    start_time = time.time()
    for step, (input_ids, labels, attn_mask) in enumerate(train_loader):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        attn_mask = attn_mask.to(args.device)

        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with ctx:
            out = model(input_ids, labels)
            loss = out.last_loss / args.accumulation_steps
            attn_mask = attn_mask.view(-1)
            loss = torch.sum(loss * attn_mask) / attn_mask.sum()

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            log(
                "Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min speed:{:.2f} s/it".format(
                    epoch,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]["lr"],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60,
                    spend_time,
                )
            )

        if (step) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            log("Saving model checkpoint ...")
            auto_map = {
                "AutoConfig": "model_config.MiniChatConfig",
                "AutoModelForCausalLM": "model.MiniChat",
            }
            # torch.save(state_dict, ckp)
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model.module.save_pretrained(
                    args.save_dir, safe_serialization=False, auto_map=auto_map
                )
            else:
                model.save_pretrained(args.save_dir, safe_serialization=False, auto_map=auto_map)

            current_folder = os.path.dirname(os.path.realpath(__file__))
            shutil.copyfile(
                os.path.join(current_folder, "./src/models/model.py"),
                os.path.join(current_folder, "model.py")
            )
            shutil.copyfile(
                os.path.join(current_folder, "./src/models/model_config.py"),
                os.path.join(current_folder, "model_config.py"),
            )
            model.train()


def init_model():
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    tokenizer = AutoTokenizer.from_pretrained("./tests/data/tokenizer")

    model = MiniChat(lm_config).to(args.device)

    log(f"模型总参数量：{count_parameters(model) / 1e6:.3f} milion")
    return model, tokenizer


def init_distributed_mode():
    if not ddp:
        return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


# torchrun --nproc_per_node 2 1-pretrain.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniChat Pretraining")
    parser.add_argument(
        "--out_dir", type=str, default="checkpoints", help="Output directory"
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=2e-4, help="Learning rate"
    )
    parser.add_argument("--device", type=str, default=None, help="Device to use")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type")
    parser.add_argument(
        "--num_workers", type=int, default=1, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./tests/data/datasets/test_train_pretrain_data.jsonl",
        help="Path to training data",
    )
    parser.add_argument(
        "--ddp", action="store_true", help="Use DistributedDataParallel"
    )
    parser.add_argument(
        "--accumulation_steps", type=int, default=8, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--grad_clip", type=float, default=1.0, help="Gradient clipping threshold"
    )
    parser.add_argument(
        "--warmup_iters", type=int, default=0, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--log_interval", type=int, default=100, help="Logging interval"
    )
    parser.add_argument(
        "--save_interval", type=int, default=1000, help="Model saving interval"
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="local rank for distributed training"
    )

    args = parser.parse_args()

    lm_config = MiniChatConfig()
    max_seq_len = lm_config.max_seq_len
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * max_seq_len
    torch.manual_seed(1337)
    device_type = get_available_device()

    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    ddp_local_rank, DEVICE = 0, device_type
    if ddp:
        init_distributed_mode()
        args.device = torch.device(device_type)

    model, tokenizer = init_model()
    df = pd.read_csv(args.data_path)
    df = df.sample(frac=1.0)
    train_ds = PretrainDataset(df, tokenizer, max_length=max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ["float16", "bfloat16"]))
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch)
