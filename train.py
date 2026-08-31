"""학습 스크립트 (Colab / 로컬 공용).

  python train.py                          # base 프리셋 (24L, d768), GPU 권장
  python train.py --preset tiny --max-steps 30   # 로컬 smoke test

- {train,val}.jsonl 이 없으면 tar.xz 에서 자동으로 푼다 (data.py).
- bf16 지원 GPU 는 bf16 autocast, 그 외 CUDA 는 fp16+GradScaler, CPU 는 fp32.
- --ckpt-dir 에 주기 저장하며 --resume 으로 재개한다 (Google Drive 경로 가능).
"""

import argparse
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import load_tokenizer, make_dataset
from model import KoreanSLLM, ModelConfig

ROOT = Path(__file__).resolve().parent

PRESETS = {
    "base": {},
    "tiny": {"n_layers": 2, "d_model": 64, "n_heads": 4, "n_kv_heads": 2, "head_dim": 16,
             "ffn_hidden": 128, "mtp_ffn_hidden": 64, "max_seq_len": 128, "sliding_window": 32},
}


def parse_args():
    p = argparse.ArgumentParser(description="Korean sLLM 학습")
    p.add_argument("--preset", choices=PRESETS, default="base")
    p.add_argument("--data-dir", default=str(ROOT))
    p.add_argument("--cache-dir", default=str(ROOT / "cache"))
    p.add_argument("--ckpt-dir", default=str(ROOT / "checkpoints"))
    p.add_argument("--resume", default=None, help="재개할 체크포인트 경로")
    p.add_argument("--seq-len", type=int, default=None, help="기본: config.max_seq_len")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=20_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=50)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--grad-checkpointing", action="store_true")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample", default=None, help="학습 종료 후 이 프롬프트로 생성 데모")
    return p.parse_args()


def lr_at(step: int, args) -> float:
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    t = (step - args.warmup_steps) / max(args.max_steps - args.warmup_steps, 1)
    return args.lr * (args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))


def setup_amp(device: torch.device):
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16, None
    if device.type == "cuda":
        return torch.float16, torch.amp.GradScaler("cuda")
    return None, None  # CPU: fp32


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, max_batches: int) -> dict[str, float]:
    model.eval()
    sums = {"main_loss": 0.0, "mtp_loss": 0.0}
    n = 0
    for batch in loader:
        if n >= max_batches:
            break
        ids = batch["input_ids"].to(device)
        mask = batch["loss_mask"].to(device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(ids, mask)
        sums["main_loss"] += out["main_loss"].item()
        sums["mtp_loss"] += out["mtp_loss"].item()
        n += 1
    model.train()
    return {k: v / max(n, 1) for k, v in sums.items()}


def save_ckpt(path: Path, model, optim, step: int, cfg: ModelConfig):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optim": optim.state_dict(),
                "step": step, "config": cfg.to_dict()}, path)
    print(f"[ckpt] step {step} -> {path}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = ModelConfig(**PRESETS[args.preset])
    seq_len = args.seq_len or cfg.max_seq_len
    root, cache_dir, ckpt_dir = Path(args.data_dir), Path(args.cache_dir), Path(args.ckpt_dir)

    sp = load_tokenizer()
    assert sp.get_piece_size() == cfg.vocab_size, \
        f"토크나이저 vocab {sp.get_piece_size()} != config {cfg.vocab_size}"

    train_ds = make_dataset(root, "train", seq_len, cache_dir, sp)
    val_ds = make_dataset(root, "val", seq_len, cache_dir, sp)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, drop_last=True)

    model = KoreanSLLM(cfg).to(device)
    print(f"모델: {cfg.n_layers}L d{cfg.d_model} | 파라미터 {model.num_params() / 1e6:.1f}M | "
          f"MTP {cfg.mtp_n}토큰 | device={device} | train {len(train_ds):,} windows(seq {seq_len})")

    if args.grad_checkpointing:
        import functools
        for block in model.blocks:
            block._orig_forward = block.forward
            block.forward = functools.partial(
                torch.utils.checkpoint.checkpoint, block._orig_forward, use_reentrant=False)

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        (no_decay if param.ndim < 2 else decay).append(param)
    optim = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), fused=device.type == "cuda")

    amp_dtype, scaler = setup_amp(device)
    print(f"정밀도: {amp_dtype or torch.float32}")

    step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        step = ckpt["step"]
        print(f"[ckpt] {args.resume} 에서 step {step} 재개")

    model.train()
    data_iter = iter(train_loader)
    t0, tokens_seen = time.time(), 0
    while step < args.max_steps:
        for g in optim.param_groups:
            g["lr"] = lr_at(step, args)
        optim.zero_grad(set_to_none=True)
        logs = {"loss": 0.0, "main_loss": 0.0, "mtp_loss": 0.0}
        for _ in range(args.grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            ids = batch["input_ids"].to(device)
            mask = batch["loss_mask"].to(device)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                out = model(ids, mask)
            loss = out["loss"] / args.grad_accum
            (scaler.scale(loss) if scaler else loss).backward()
            for k in logs:
                logs[k] += out[k].item() / args.grad_accum
            tokens_seen += ids.numel()

        if scaler:
            scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if scaler:
            scaler.step(optim)
            scaler.update()
        else:
            optim.step()
        step += 1

        if step % args.log_every == 0:
            tps = tokens_seen / (time.time() - t0)
            print(f"step {step:6d} | loss {logs['loss']:.4f} (main {logs['main_loss']:.4f} "
                  f"mtp {logs['mtp_loss']:.4f}) | lr {optim.param_groups[0]['lr']:.2e} | {tps / 1e3:.1f}k tok/s")
        if step % args.eval_every == 0:
            ev = evaluate(model, val_loader, device, amp_dtype, args.eval_batches)
            print(f"  [val] main {ev['main_loss']:.4f} | mtp {ev['mtp_loss']:.4f} | "
                  f"ppl {math.exp(min(ev['main_loss'], 20)):.1f}")
        if step % args.save_every == 0 or step == args.max_steps:
            save_ckpt(ckpt_dir / f"step{step:06d}.pt", model, optim, step, cfg)

    save_ckpt(ckpt_dir / "final.pt", model, optim, step, cfg)

    if args.sample:
        from data import encode_sample
        prompt_ids = encode_sample(sp, args.sample, "")[0][:-2]  # assistant 답변/eos 제외
        out = model.generate(torch.tensor([prompt_ids], device=device), max_new_tokens=256)
        print("\n=== 생성 데모 ===")
        print(sp.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
