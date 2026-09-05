"""사전학습 스크립트 (Colab / 로컬 공용).

  python train_pretrain.py                          # base 프리셋 (24L, d768), GPU 권장
  python train_pretrain.py --preset tiny --max-steps 30   # 로컬 smoke test

- pretrain_train.json / pretrain_val.json의 문자열 전체를 weight=1로 학습한다.
- bf16 지원 GPU 는 bf16 autocast, 그 외 CUDA 는 fp16+GradScaler, CPU 는 fp32.
- --ckpt-dir 에 last.pt(--save-every 주기 최신)와 best.pt(val main_loss 최저)만 유지하고,
  --resume 은 last.pt 로 재개한다 (Google Drive 경로 가능).
"""

import argparse
import hashlib
import math
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import load_tokenizer
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
    p.add_argument("--stage", choices=("pretrain",), default="pretrain")
    p.add_argument("--overlap", type=int, default=256)
    p.add_argument("--resume", default=None, help="모델/옵티마이저/데이터 위치/난수 상태 복원")
    p.add_argument("--session-epochs", type=float, default=0.25, help="이번 실행에서 추가 학습할 epoch")
    p.add_argument("--seq-len", type=int, default=None, help="기본: config.max_seq_len")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=20_000)
    p.add_argument("--epochs", type=float, default=None,
                   help="전체 학습 목표 epoch; 세션마다 동일하게 유지")
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


def save_ckpt(path: Path, model, optim, step: int, cfg: ModelConfig, best_val: float,
              stage: str, tokenizer_sha256: str, scaler=None, run_config=None):
    # 임시 파일에 쓴 뒤 교체 - Drive 위에서 덮어쓰기 도중 런타임이 끊겨도 기존 파일이 보존된다
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optim": optim.state_dict(),
                "step": step, "config": cfg.to_dict(), "best_val": best_val,
                "stage": stage, "tokenizer_sha256": tokenizer_sha256,
                "scaler": scaler.state_dict() if scaler else None,
                "run_config": run_config,
                "data_epoch": step // run_config["steps_per_epoch"],
                "batch_in_epoch": (step % run_config["steps_per_epoch"]) * run_config["grad_accum"],
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, tmp)
    os.replace(tmp, path)
    print(f"[ckpt] step {step} -> {path}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        if ckpt.get("stage") != "pretrain":
            raise ValueError("pretrain 체크포인트가 필요합니다.")
        if "run_config" not in ckpt or "rng_state" not in ckpt:
            raise ValueError("이전 형식에는 데이터 위치/스케줄 정보가 없습니다. 새 CKPT_DIR에서 시작하세요.")
    cfg = ModelConfig(**ckpt["config"]) if ckpt else ModelConfig(**PRESETS[args.preset])
    seq_len = args.seq_len or cfg.max_seq_len
    if not 2 <= seq_len <= cfg.max_seq_len:
        raise ValueError(f"seq-len은 2..{cfg.max_seq_len} 범위여야 합니다.")
    if min(args.batch_size, args.grad_accum, args.eval_every, args.save_every,
           args.eval_batches, args.log_every, args.max_steps) < 1:
        raise ValueError("배치/스텝/주기 값은 양수여야 합니다.")
    if args.epochs is not None and args.epochs <= 0:
        raise ValueError("epochs는 양수여야 합니다.")
    if not math.isfinite(args.session_epochs) or args.session_epochs <= 0:
        raise ValueError("session-epochs는 양수여야 합니다.")
    root, cache_dir, ckpt_dir = Path(args.data_dir), Path(args.cache_dir), Path(args.ckpt_dir)

    sp = load_tokenizer()
    tokenizer_sha256 = hashlib.sha256(sp.serialized_model_proto()).hexdigest()
    if ckpt:
        saved_hash = ckpt.get("tokenizer_sha256")
        if saved_hash and saved_hash != tokenizer_sha256:
            raise ValueError("토크나이저가 체크포인트와 다릅니다. 사전학습 때의 spm.model을 사용하세요.")
    assert sp.get_piece_size() == cfg.vocab_size, \
        f"토크나이저 vocab {sp.get_piece_size()} != config {cfg.vocab_size}"

    from pretrain_data import make_pretrain_dataset
    train_ds = make_pretrain_dataset(root, "train", seq_len, cache_dir, sp, args.overlap)
    val_ds = make_pretrain_dataset(root, "val", seq_len, cache_dir, sp, args.overlap)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("학습 또는 검증 데이터가 비었습니다.")
    batches_per_epoch = math.ceil(len(train_ds) / args.batch_size)
    steps_per_epoch = math.ceil(batches_per_epoch / args.grad_accum)
    if args.epochs is not None:
        args.max_steps = max(math.ceil(args.epochs * steps_per_epoch), 1)
    run_config = {
        "version": 1, "train_cache": Path(train_ds.tokens.filename).name,
        "val_cache": Path(val_ds.tokens.filename).name,
        "train_windows": len(train_ds), "seq_len": seq_len, "overlap": args.overlap,
        "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "seed": args.seed, "total_steps": args.max_steps,
        "steps_per_epoch": steps_per_epoch, "lr": args.lr,
        "min_lr_ratio": args.min_lr_ratio, "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay, "grad_clip": args.grad_clip,
    }
    if ckpt and ckpt["run_config"] != run_config:
        changed = [k for k, v in run_config.items() if ckpt["run_config"].get(k) != v]
        raise ValueError(f"재개 설정/데이터가 달라졌습니다: {changed}. 기존 설정을 유지하세요.")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, drop_last=False,
                            generator=torch.Generator().manual_seed(args.seed))
    tokens_per_step = args.batch_size * args.grad_accum * seq_len

    model = KoreanSLLM(cfg).to(device)
    print(f"모델: {cfg.n_layers}L d{cfg.d_model} v{cfg.vocab_size} | 파라미터 {model.num_params() / 1e6:.1f}M | "
          f"MTP {cfg.mtp_n}토큰 | device={device} | train {len(train_ds):,} windows(seq {seq_len})")
    print(f"스케줄: {args.max_steps:,} steps × {tokens_per_step:,} tok/step "
          f"= {args.max_steps * tokens_per_step / 1e9:.2f}B tokens ≈ {args.max_steps / steps_per_epoch:.1f} epochs")

    if args.grad_checkpointing:
        import functools
        from torch.utils.checkpoint import checkpoint as checkpoint_forward
        for block in model.blocks:
            block._orig_forward = block.forward
            block.forward = functools.partial(
                checkpoint_forward, block._orig_forward, use_reentrant=False)

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
    best_val = float("inf")
    if ckpt:
        model.load_state_dict(ckpt["model"])
    if args.resume:
        optim.load_state_dict(ckpt["optim"])
        if scaler and ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        step = ckpt["step"]
        best_val = ckpt.get("best_val", float("inf"))  # 미복원 시 첫 eval 이 best.pt 를 덮어쓴다
        print(f"[ckpt] {args.resume} 에서 step {step} 재개 (best val {best_val:.4f})")
        torch.set_rng_state(ckpt["rng_state"])
        if device.type == "cuda" and ckpt.get("cuda_rng_state"):
            torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
    del ckpt
    session_end = min(args.max_steps, step + max(1, math.ceil(args.session_epochs * steps_per_epoch)))
    print(f"[session] {step / steps_per_epoch:.4f} -> {session_end / steps_per_epoch:.4f} epochs "
          f"/ 목표 {args.max_steps / steps_per_epoch:.2f} (steps {step} -> {session_end})")
    if step >= args.max_steps:
        print("전체 학습 목표를 이미 완료했습니다.")
        return

    model.train()
    data_iter = None
    t0, tokens_seen = time.time(), 0
    while step < session_end:
        for g in optim.param_groups:
            g["lr"] = lr_at(step, args)
        optim.zero_grad(set_to_none=True)
        logs = {"loss": 0.0, "main_loss": 0.0, "mtp_loss": 0.0}
        epoch, step_in_epoch = divmod(step, steps_per_epoch)
        batch_start = step_in_epoch * args.grad_accum
        if data_iter is None or step_in_epoch == 0:
            # Rebuild the same epoch permutation and jump directly to the next batch.
            # DataLoader prefetch does not advance the checkpoint's consumed position.
            order = torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(args.seed + epoch))
            remaining = order[batch_start * args.batch_size:].tolist()
            train_loader = DataLoader(
                train_ds, batch_size=args.batch_size, sampler=remaining,
                num_workers=args.num_workers, drop_last=False, pin_memory=device.type == "cuda",
                generator=torch.Generator().manual_seed(args.seed + epoch),
            )
            data_iter = iter(train_loader)
        accum_steps = min(args.grad_accum, batches_per_epoch - batch_start)
        for _ in range(accum_steps):
            batch = next(data_iter)
            ids = batch["input_ids"].to(device)
            mask = batch["loss_mask"].to(device)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                out = model(ids, mask)
            loss = out["loss"] / accum_steps
            (scaler.scale(loss) if scaler else loss).backward()
            for k in logs:
                logs[k] += out[k].item() / accum_steps
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
        if step % args.eval_every == 0 or step == session_end:
            ev = evaluate(model, val_loader, device, amp_dtype, args.eval_batches)
            mem = (f" | mem {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB"
                   if device.type == "cuda" else "")
            print(f"  [val] main {ev['main_loss']:.4f} | mtp {ev['mtp_loss']:.4f} | "
                  f"ppl {math.exp(min(ev['main_loss'], 20)):.1f}{mem}")
            if ev["main_loss"] < best_val:
                best_val = ev["main_loss"]
                print(f"  [ckpt] new best (val main {best_val:.4f})")
                save_ckpt(ckpt_dir / "best.pt", model, optim, step, cfg, best_val,
                          args.stage, tokenizer_sha256, scaler, run_config)
        if step % args.save_every == 0 or step == session_end:
            save_ckpt(ckpt_dir / "last.pt", model, optim, step, cfg, best_val,
                      args.stage, tokenizer_sha256, scaler, run_config)

    print(f"세션 종료: 누적 {step / steps_per_epoch:.4f} epochs | last.pt step {step} | best val {best_val:.4f}")

    if args.sample:
        prompt_ids = [sp.bos_id()] + sp.encode(args.sample)
        out = model.generate(torch.tensor([prompt_ids], device=device), max_new_tokens=256)
        print("\n=== 생성 데모 ===")
        print(sp.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
