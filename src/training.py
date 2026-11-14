"""Training loop with signed-weight loss for the pretraining experiment."""

import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel

from .config import TrainingConfig, ExperimentConfig, make_output_dir


# ---------------------------------------------------------------------------
# Weighted loss
# ---------------------------------------------------------------------------

def get_effective_alpha(
    step: int,
    target_alpha: float,
    warmup_steps: int,
    total_steps: int,
    decay: bool,
) -> float:
    """Compute the effective negative alpha at the current step.

    Supports warmup (ramp from 0 to target) and optional cosine decay back to 0.
    """
    if warmup_steps <= 0 and not decay:
        return target_alpha

    # Warmup phase: linear ramp from 0 to target
    if step < warmup_steps:
        return target_alpha * (step + 1) / warmup_steps

    if not decay:
        return target_alpha

    # Decay phase: cosine from target back to 0
    decay_start = warmup_steps
    decay_end = total_steps
    if step >= decay_end:
        return 0.0
    progress = (step - decay_start) / max(1, decay_end - decay_start)
    return target_alpha * 0.5 * (1.0 + math.cos(math.pi * progress))


def compute_weighted_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    weights: torch.Tensor,
    neg_loss_clamp: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """Compute per-example cross-entropy weighted by signed weights.

    Args:
        logits: (batch, seq_len, vocab_size)
        labels: (batch, seq_len)
        attention_mask: (batch, seq_len) — 1 for real tokens, 0 for padding
        weights: (batch,) — signed weights in [-1, 1]
        neg_loss_clamp: if > 0, clamp CE for negative-weighted examples to this max.
            Prevents runaway feedback where the model gets very high CE on negative
            examples, creating an ever-growing negative gradient signal.

    Returns:
        (scalar loss, metrics dict with positive/negative component breakdown)
    """
    # Shift for causal LM: predict token t+1 from position t
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous().float()

    # Per-token cross-entropy (no reduction)
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    per_token_loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    ).view(shift_logits.size(0), shift_logits.size(1))

    # Mask padding, compute per-example mean
    per_token_loss = per_token_loss * shift_mask
    tokens_per_example = shift_mask.sum(dim=1).clamp(min=1)
    per_example_loss = per_token_loss.sum(dim=1) / tokens_per_example

    # Clamp CE for negative-weighted examples to prevent runaway
    if neg_loss_clamp > 0:
        neg_mask_clamp = weights < 0
        if neg_mask_clamp.any():
            per_example_loss = torch.where(
                neg_mask_clamp,
                per_example_loss.clamp(max=neg_loss_clamp),
                per_example_loss,
            )

    # Apply signed weights and average
    weighted_losses = weights * per_example_loss
    total_loss = weighted_losses.mean()

    # Diagnostics: break down by positive/negative weight
    pos_mask = weights > 0
    neg_mask = weights < 0
    zero_mask = weights == 0

    metrics = {
        "loss_total": total_loss.item(),
        "loss_unweighted_mean": per_example_loss.mean().item(),
    }
    if pos_mask.any():
        metrics["loss_positive"] = (weights[pos_mask] * per_example_loss[pos_mask]).mean().item()
        metrics["ce_positive"] = per_example_loss[pos_mask].mean().item()
    if neg_mask.any():
        metrics["loss_negative"] = (weights[neg_mask] * per_example_loss[neg_mask]).mean().item()
        metrics["ce_negative"] = per_example_loss[neg_mask].mean().item()
    if zero_mask.any():
        metrics["n_filtered"] = zero_mask.sum().item()

    return total_loss, metrics


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def get_lr(step: int, warmup_steps: int, total_steps: int,
           max_lr: float, min_lr: float) -> float:
    """Cosine schedule with linear warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model: GPT2LMHeadModel,
    train_loader: DataLoader,
    eval_safe_loader: DataLoader,
    eval_unsafe_loader: DataLoader,
    train_cfg: TrainingConfig,
    exp_cfg: ExperimentConfig,
    resume_from: Path | None = None,
) -> dict:
    """Full training loop. Returns final metrics dict.

    If `resume_from` is given, the checkpoint dir (written by
    `training_advanced.save_training_checkpoint`) must contain optimizer.pt +
    trainer_state.pt; model weights, optimizer state, step/epoch counters, and
    RNG state will all be restored.
    """
    from .training_advanced import (
        save_training_checkpoint,
        load_training_checkpoint,
        _capture_rng_state,
        _restore_rng_state,
        _unwrap_model,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if train_cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print("Gradient checkpointing enabled (use_reentrant=False)")

    # torch.compile for 10-30% training speedup via kernel fusion
    try:
        model = torch.compile(model)
        print("torch.compile enabled")
    except Exception as e:
        print(f"torch.compile failed ({e}), continuing without")

    output_dir = make_output_dir(exp_cfg, train_cfg)
    log_file = output_dir / "training_log.jsonl"

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=train_cfg.weight_decay,
    )

    accum = train_cfg.grad_accum_steps
    total_micro_steps = len(train_loader) * train_cfg.n_epochs
    total_steps = total_micro_steps // accum  # optimizer steps
    print(f"Total steps: {total_steps} optimizer steps "
          f"({len(train_loader)} micro-steps/epoch x {train_cfg.n_epochs} epochs, accum={accum})")

    # Mixed precision
    use_amp = train_cfg.bf16 and torch.cuda.is_available()
    scaler = GradScaler("cuda", enabled=use_amp and not train_cfg.bf16)
    amp_dtype = torch.bfloat16 if train_cfg.bf16 else torch.float16

    global_step = 0
    best_eval_loss = float("inf")
    patience_counter = 0
    stopped_early = False
    all_metrics = []
    t_start = time.time()
    elapsed_sec_offset = 0.0

    # Alpha schedule params
    has_negative = exp_cfg.weight_scheme == "negative"
    target_alpha = exp_cfg.negative_alpha if has_negative else 0.0

    micro_step = 0  # counts every forward/backward pass
    start_epoch = 0
    start_batch_in_epoch = 0
    resume_epoch_rng = None   # random state the interrupted epoch's shuffle was drawn from
    pending_rng = None        # checkpoint-time random state, restored once the skipped batches are passed

    if resume_from is not None:
        trainer_state = load_training_checkpoint(resume_from, model, optimizer, None)
        global_step = trainer_state["global_step"]
        micro_step = trainer_state["micro_step"]
        start_epoch = trainer_state["epoch"]
        start_batch_in_epoch = trainer_state.get("batch_in_epoch", 0)
        best_eval_loss = trainer_state.get("best_eval_loss", float("inf"))
        patience_counter = trainer_state.get("patience_counter", 0)
        elapsed_sec_offset = trainer_state.get("elapsed_sec_cum", 0.0)
        resume_epoch_rng = trainer_state.get("rng_epoch_start")
        if resume_epoch_rng is not None:
            pending_rng = trainer_state.get("rng")
        else:
            print("[RESUME] WARNING: this checkpoint was written before 2026-11-14 and does not hold the epoch's "
                  "shuffle state; the rest of this epoch is reshuffled (some rows repeat, some are skipped).", flush=True)
        print(
            f"[RESUME] from {resume_from}: step={global_step}, epoch={start_epoch}, "
            f"batch_in_epoch={start_batch_in_epoch}, best_safe_loss={best_eval_loss:.4f}"
        )

    for epoch in range(start_epoch, train_cfg.n_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        # The epoch's shuffle is drawn from the global random state when the loader's iterator starts. Keep the
        # state it is drawn from: a checkpoint stores it, and a resumed run restores it before re-creating the
        # iterator, so the interrupted epoch is continued in its own order (tests/test_resume_order.py). Until
        # 2026-11-14 only the checkpoint-time state was restored, which gave the resumed epoch a new shuffle.
        if epoch == start_epoch and resume_epoch_rng is not None:
            _restore_rng_state(resume_epoch_rng)
        epoch_rng = _capture_rng_state()
        skip_batches = start_batch_in_epoch if epoch == start_epoch else 0
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx < skip_batches:
                continue
            if pending_rng is not None:   # first batch after the skipped ones: back to the checkpoint's state
                _restore_rng_state(pending_rng)
                pending_rng = None
            input_ids, attention_mask, weights = [x.to(device) for x in batch]

            # Apply alpha schedule: scale negative weights by effective alpha
            if has_negative and (train_cfg.alpha_warmup_steps > 0 or train_cfg.alpha_decay):
                eff_alpha = get_effective_alpha(
                    global_step, target_alpha,
                    train_cfg.alpha_warmup_steps, total_steps, train_cfg.alpha_decay,
                )
                # Rescale negative weights: original weights are -alpha, rescale to -eff_alpha
                neg_mask = weights < 0
                if neg_mask.any() and target_alpha > 0:
                    weights = weights.clone()
                    weights[neg_mask] = weights[neg_mask] * (eff_alpha / target_alpha)
            else:
                eff_alpha = target_alpha

            # Forward (accumulate gradients)
            with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                del outputs
                loss, batch_metrics = compute_weighted_loss(
                    logits, input_ids, attention_mask, weights,
                    neg_loss_clamp=train_cfg.neg_loss_clamp,
                )
                del logits
                loss = loss / accum  # scale loss for accumulation

            loss.backward()
            micro_step += 1

            # Only step optimizer every accum micro-steps
            if micro_step % accum != 0:
                continue

            # LR schedule (updates on optimizer steps, not micro-steps)
            lr = get_lr(global_step, train_cfg.warmup_steps, total_steps,
                        train_cfg.learning_rate, train_cfg.min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_cfg.grad_clip
            )
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += batch_metrics["loss_total"]
            epoch_steps += 1
            global_step += 1

            # Log
            if global_step % train_cfg.log_every == 0:
                elapsed = elapsed_sec_offset + (time.time() - t_start)
                log_entry = {
                    "step": global_step,
                    "epoch": epoch,
                    "lr": lr,
                    "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    "elapsed_sec": round(elapsed, 1),
                    **batch_metrics,
                }
                if has_negative:
                    log_entry["eff_alpha"] = round(eff_alpha, 6)
                all_metrics.append(log_entry)
                with open(log_file, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

                if global_step % (train_cfg.log_every * 4) == 0:
                    alpha_str = f" | α={eff_alpha:.4f}" if has_negative else ""
                    print(
                        f"  step {global_step}/{total_steps} | "
                        f"loss={batch_metrics['loss_total']:.4f} | "
                        f"grad_norm={log_entry['grad_norm']:.3f} | "
                        f"lr={lr:.2e}{alpha_str} | "
                        f"{elapsed:.0f}s"
                    )

            # Eval
            if global_step % train_cfg.eval_every == 0:
                eval_metrics = evaluate_during_training(
                    model, eval_safe_loader, eval_unsafe_loader, device, amp_dtype, use_amp
                )
                eval_entry = {"step": global_step, "epoch": epoch, "type": "eval", **eval_metrics}
                with open(log_file, "a") as f:
                    f.write(json.dumps(eval_entry) + "\n")
                print(
                    f"  [EVAL step {global_step}] "
                    f"safe_ppl={eval_metrics['ppl_safe']:.2f} | "
                    f"unsafe_ppl={eval_metrics['ppl_unsafe']:.2f} | "
                    f"gap={eval_metrics['ppl_gap']:.3f}"
                )

                # Best model checkpointing
                if eval_metrics["eval_loss_safe"] < best_eval_loss:
                    improvement = (best_eval_loss - eval_metrics["eval_loss_safe"]) / max(abs(best_eval_loss), 1e-8)
                    best_eval_loss = eval_metrics["eval_loss_safe"]
                    _unwrap_model(model).save_pretrained(output_dir / "best_model")
                    if improvement >= train_cfg.early_stopping_threshold:
                        patience_counter = 0
                    else:
                        patience_counter += 1
                else:
                    patience_counter += 1

                # Early stopping
                if train_cfg.early_stopping_patience > 0 and patience_counter >= train_cfg.early_stopping_patience:
                    print(f"  [EARLY STOP] No improvement in safe PPL for {patience_counter} evals. "
                          f"Best safe loss: {best_eval_loss:.4f}")
                    stopped_early = True
                    break

            # Checkpoint (full resumable state)
            if global_step % train_cfg.save_every == 0:
                ckpt_dir = output_dir / f"checkpoint_{global_step}"
                trainer_state = {
                    "global_step": global_step,
                    "micro_step": micro_step,
                    "epoch": epoch,
                    "batch_in_epoch": batch_idx + 1,
                    "best_eval_loss": best_eval_loss,
                    "patience_counter": patience_counter,
                    "alignment_started": True,
                    "elapsed_sec_cum": elapsed_sec_offset + (time.time() - t_start),
                    "rng": _capture_rng_state(),
                    "rng_epoch_start": epoch_rng,
                    "seed": train_cfg.seed,
                    "method": "standard",
                }
                save_training_checkpoint(ckpt_dir, model, optimizer, None, trainer_state)
                print(f"  [CKPT step {global_step}] saved full state to {ckpt_dir.name}/")

        if pending_rng is not None:   # the checkpoint was the epoch's last batch: nothing was left to run in it
            _restore_rng_state(pending_rng)
            pending_rng = None
        print(f"Epoch {epoch} done | avg loss = {epoch_loss / max(epoch_steps, 1):.4f}")
        if stopped_early:
            break

    # Save final model
    _unwrap_model(model).save_pretrained(output_dir / "final_model")

    # Final eval
    final_metrics = evaluate_during_training(
        model, eval_safe_loader, eval_unsafe_loader, device, amp_dtype, use_amp
    )
    total_time = elapsed_sec_offset + (time.time() - t_start)
    result = {
        "total_steps": global_step,
        "total_time_sec": round(total_time, 1),
        "best_eval_loss": best_eval_loss,
        "stopped_early": stopped_early,
        **final_metrics,
    }
    if stopped_early:
        result["early_stop_step"] = global_step
    with open(output_dir / "final_metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    stop_msg = f" (early stopped at step {global_step})" if stopped_early else ""
    print(f"\nTraining complete in {total_time:.0f}s{stop_msg}")
    print(f"  Final safe PPL: {final_metrics['ppl_safe']:.2f}")
    print(f"  Final unsafe PPL: {final_metrics['ppl_unsafe']:.2f}")
    print(f"  PPL gap (unsafe/safe): {final_metrics['ppl_gap']:.3f}")
    print(f"  Saved to: {output_dir}")

    return result


# ---------------------------------------------------------------------------
# Eval helper (used during training)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_during_training(
    model: GPT2LMHeadModel,
    eval_safe_loader: DataLoader,
    eval_unsafe_loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
) -> dict:
    """Quick eval: compute perplexity on safe and unsafe test sets."""
    model.eval()

    def _eval_ppl(loader):
        total_loss = 0.0
        total_tokens = 0
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        for input_ids, attention_mask, _ in loader:
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                shift_logits = outputs.logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                shift_mask = attention_mask[:, 1:].contiguous().float()

                per_token = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                ).view(shift_logits.size(0), shift_logits.size(1))
                per_token = per_token * shift_mask
                total_loss += per_token.sum().item()
                total_tokens += shift_mask.sum().item()

        avg_loss = total_loss / max(total_tokens, 1)
        ppl = math.exp(min(avg_loss, 100))  # cap to avoid overflow
        return avg_loss, ppl

    safe_loss, safe_ppl = _eval_ppl(eval_safe_loader)
    unsafe_loss, unsafe_ppl = _eval_ppl(eval_unsafe_loader)

    model.train()
    return {
        "eval_loss_safe": safe_loss,
        "eval_loss_unsafe": unsafe_loss,
        "ppl_safe": safe_ppl,
        "ppl_unsafe": unsafe_ppl,
        "ppl_gap": unsafe_ppl / max(safe_ppl, 1e-8),
    }
