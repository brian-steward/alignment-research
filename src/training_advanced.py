"""Advanced training strategies for reducing alignment tax in signed-weight pretraining.

Implements 5 novel/experimental methods:
1. Gradient Projection — project negative gradients orthogonal to positive gradient direction
2. Layer-Selective — only propagate negative gradients through top K transformer layers
3. Token-Level — per-token negative weighting based on content specificity
4. Adaptive — confidence-scaled negative weights (reduce signal once model already unlearned)
5. Phased — uniform capability training first, then switch to signed weights
"""

import json
import math
import random
import signal
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel

from .config import TrainingConfig, ExperimentConfig, make_output_dir
from .training import (
    compute_weighted_loss,
    get_lr,
    get_effective_alpha,
    evaluate_during_training,
)


# ---------------------------------------------------------------------------
# Checkpoint save/load for resumable training
# ---------------------------------------------------------------------------

def _unwrap_model(model):
    """Return the underlying nn.Module when `model` is wrapped by torch.compile."""
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def _capture_rng_state() -> dict:
    """Snapshot Python / NumPy / Torch / CUDA RNG state for resumable shuffling."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_training_checkpoint(
    ckpt_dir: Path,
    model,
    optimizer,
    nspo_buffer,
    trainer_state: dict,
):
    """Write a full resumable checkpoint to `ckpt_dir`.

    Contents:
      - model.safetensors + config.json (HF `save_pretrained` on the unwrapped module)
      - optimizer.pt             — `optimizer.state_dict()`
      - nspo_buffer.pt           — buffered positive gradients (only if nspo_buffer is not None)
      - trainer_state.pt         — step/epoch counters, best_eval_loss, patience,
                                   alignment_started, elapsed_sec_cum, RNG state
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    _unwrap_model(model).save_pretrained(ckpt_dir)
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
    if nspo_buffer is not None:
        if nspo_buffer.size > 0 and nspo_buffer.buffer is not None:
            # Save in chronological FIFO order so reload restores the correct sequence.
            head, sz, max_sz = nspo_buffer._head, nspo_buffer.size, nspo_buffer.max_size
            if sz < max_sz:
                # ring not yet full: rows [0:size] are the live entries
                buf_tensor = nspo_buffer.buffer[:sz].detach().cpu().clone()
            else:
                # full ring: oldest at _head, wrap around
                idx = torch.arange(head, head + max_sz, device=nspo_buffer.buffer.device) % max_sz
                buf_tensor = nspo_buffer.buffer.index_select(0, idx).detach().cpu().clone()
        else:
            buf_tensor = torch.empty(0)
        torch.save(
            {"max_size": nspo_buffer.max_size, "buffer": buf_tensor},
            ckpt_dir / "nspo_buffer.pt",
        )
    torch.save(trainer_state, ckpt_dir / "trainer_state.pt")


def load_training_checkpoint(
    ckpt_dir: Path,
    model,
    optimizer,
    nspo_buffer,
) -> dict:
    """Restore a full training checkpoint. Returns the trainer_state dict.

    Model weights and optimizer state are loaded into the provided objects in-place.
    `nspo_buffer`, when provided, is mutated to contain the saved gradients.
    """
    ckpt_dir = Path(ckpt_dir)
    optim_path = ckpt_dir / "optimizer.pt"
    state_path = ckpt_dir / "trainer_state.pt"
    if not optim_path.exists() or not state_path.exists():
        raise FileNotFoundError(
            f"Checkpoint at {ckpt_dir} is missing optimizer.pt or trainer_state.pt. "
            f"Use --init-from for a weights-only warm start instead of --resume."
        )

    # Model weights (copy into the existing, already-device-placed module)
    loaded = GPT2LMHeadModel.from_pretrained(ckpt_dir)
    _unwrap_model(model).load_state_dict(loaded.state_dict())
    del loaded

    # Optimizer state
    optim_sd = torch.load(optim_path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(optim_sd)

    # NSPO buffer (optional)
    buf_path = ckpt_dir / "nspo_buffer.pt"
    if nspo_buffer is not None and buf_path.exists():
        bd = torch.load(buf_path, map_location="cpu", weights_only=False)
        nspo_buffer.max_size = bd["max_size"]
        buf_tensor = bd["buffer"]
        if buf_tensor.numel() == 0:
            nspo_buffer.buffer = None
            nspo_buffer.size = 0
            nspo_buffer._head = 0
        else:
            device = next(model.parameters()).device
            n = buf_tensor.shape[0]
            D = buf_tensor.shape[1]
            nspo_buffer.buffer = torch.empty(
                nspo_buffer.max_size, D, dtype=torch.bfloat16, device=device,
            )
            nspo_buffer.buffer[:n].copy_(buf_tensor.to(torch.bfloat16))
            nspo_buffer.size = n
            nspo_buffer._head = n % nspo_buffer.max_size

    # Trainer state
    trainer_state = torch.load(state_path, map_location="cpu", weights_only=False)
    if "rng" in trainer_state:
        _restore_rng_state(trainer_state["rng"])
    return trainer_state


# ---------------------------------------------------------------------------
# Token specificity (for token-level method)
# ---------------------------------------------------------------------------

def compute_token_specificity(
    train_ds,
    vocab_size: int,
    smoothing: float = 10.0,
) -> torch.Tensor:
    """Compute per-token specificity: how much more a token appears in negative vs positive examples.

    specificity(t) = (count_neg(t) + s) / (count_neg(t) + count_pos(t) + 2s)

    Returns tensor of shape (vocab_size,) with values in [0, 1].
    0.5 = equally common in both, >0.5 = more common in negative-weighted text.
    """
    pos_counts = torch.zeros(vocab_size)
    neg_counts = torch.zeros(vocab_size)

    for i in range(len(train_ds)):
        ids, mask, w = train_ds[i]
        tokens = ids[mask.bool()]
        ones = torch.ones(len(tokens))
        if w >= 0:
            pos_counts.scatter_add_(0, tokens.long(), ones)
        else:
            neg_counts.scatter_add_(0, tokens.long(), ones)

    specificity = (neg_counts + smoothing) / (pos_counts + neg_counts + 2 * smoothing)

    n_neg_tokens = int(neg_counts.sum().item())
    n_pos_tokens = int(pos_counts.sum().item())
    n_high_spec = int((specificity > 0.6).sum().item())
    print(f"  Token specificity: {n_pos_tokens:,} pos tokens, {n_neg_tokens:,} neg tokens, "
          f"{n_high_spec} tokens with specificity > 0.6")
    return specificity


# ---------------------------------------------------------------------------
# Step functions for each method
# ---------------------------------------------------------------------------

def _step_standard(
    model, input_ids, attention_mask, weights,
    amp_dtype, use_amp, neg_loss_clamp, accum, **kwargs,
):
    """Standard signed-weight training step."""
    with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        del outputs
        loss, metrics = compute_weighted_loss(
            logits, input_ids, attention_mask, weights, neg_loss_clamp,
        )
        del logits
        loss = loss / accum
    loss.backward()
    return metrics


def _two_pass_fallback(model, input_ids, attention_mask, weights, amp_dtype, use_amp, neg_loss_clamp):
    """A two-pass step met a micro-batch with no negative (or no positive) row: the plain single-pass gradient.

    Two-pass steps leave the micro-batch's own mean gradient in .grad, and train_advanced divides it by
    grad_accum_steps when it adds it to its buffer. So the loss must NOT be scaled by 1/accum here.
    Until 2026-11-14 these fallbacks passed accum on to _step_standard, so with grad_accum_steps = A such
    micro-batches counted for 1/A of the others (tests/test_grad_accum.py, the two fallback tests).
    """
    return _step_standard(model, input_ids, attention_mask, weights, amp_dtype, use_amp, neg_loss_clamp, 1)


def _step_grad_projection(
    model, input_ids, attention_mask, weights,
    amp_dtype, use_amp, neg_loss_clamp, accum, **kwargs,
):
    """Gradient projection: project negative gradients orthogonal to positive direction.

    Removes the component of the alignment gradient that conflicts with capability,
    preserving only the alignment signal in directions the capability gradient doesn't use.
    """
    pos_mask = weights >= 0
    neg_mask = weights < 0

    if not neg_mask.any():
        return _two_pass_fallback(model, input_ids, attention_mask, weights,
                                  amp_dtype, use_amp, neg_loss_clamp)
    if not pos_mask.any():
        return _two_pass_fallback(model, input_ids, attention_mask, weights,
                                  amp_dtype, use_amp, neg_loss_clamp)

    # 1) Forward + backward on POSITIVE examples
    with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
        out_pos = model(input_ids=input_ids[pos_mask], attention_mask=attention_mask[pos_mask])
        logits_pos = out_pos.logits
        del out_pos
        loss_pos, met_pos = compute_weighted_loss(
            logits_pos, input_ids[pos_mask], attention_mask[pos_mask],
            weights[pos_mask], neg_loss_clamp,
        )
        del logits_pos
    loss_pos.backward()

    # Save positive gradients as one flat vector
    grad_params = [(n, p) for n, p in model.named_parameters() if p.grad is not None]
    g_pos_flat = torch.cat([p.grad.flatten().float() for _, p in grad_params])
    g_pos_dict = {n: p.grad.clone() for n, p in grad_params}

    model.zero_grad()

    # 2) Forward + backward on NEGATIVE examples
    with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
        out_neg = model(input_ids=input_ids[neg_mask], attention_mask=attention_mask[neg_mask])
        logits_neg = out_neg.logits
        del out_neg
        loss_neg, met_neg = compute_weighted_loss(
            logits_neg, input_ids[neg_mask], attention_mask[neg_mask],
            weights[neg_mask], neg_loss_clamp,
        )
        del logits_neg
    loss_neg.backward()

    g_neg_flat = torch.cat([p.grad.flatten().float() for _, p in grad_params])

    # 3) Project: remove component of g_neg along g_pos (PCGrad-style)
    dot = torch.dot(g_neg_flat, g_pos_flat)
    g_pos_norm_sq = torch.dot(g_pos_flat, g_pos_flat)

    projected = False
    if g_pos_norm_sq > 1e-12 and dot < 0:
        # Conflict detected: negative gradient opposes positive gradient
        g_neg_proj = g_neg_flat - (dot / g_pos_norm_sq) * g_pos_flat
        projected = True
    else:
        g_neg_proj = g_neg_flat

    # 4) Set combined gradient on parameters
    n_pos = pos_mask.sum().item()
    n_neg = neg_mask.sum().item()
    n_total = float(n_pos + n_neg)

    offset = 0
    for name, param in grad_params:
        numel = param.numel()
        g_p = g_pos_dict[name].float()
        g_n = g_neg_proj[offset:offset + numel].view(param.shape)
        # Weight by proportion in batch (so combined gradient has same scale)
        param.grad = ((n_pos / n_total) * g_p + (n_neg / n_total) * g_n).to(param.dtype)
        offset += numel

    metrics = {
        "loss_total": (loss_pos.item() * n_pos + loss_neg.item() * n_neg) / n_total,
        "ce_positive": met_pos.get("ce_positive", 0.0),
        "ce_negative": met_neg.get("ce_negative", 0.0),
        "loss_positive": loss_pos.item(),
        "loss_negative": loss_neg.item(),
        "gp_dot": dot.item(),
        "gp_projected": projected,
    }
    return metrics


def _freeze_layers(model, layer_indices, include_embeddings=False):
    """Freeze specified layer parameters. Returns list of frozen params to unfreeze later."""
    frozen = []
    for i in layer_indices:
        for p in model.transformer.h[i].parameters():
            if p.requires_grad:
                p.requires_grad_(False)
                frozen.append(p)
    if include_embeddings:
        for module in [model.transformer.wte, model.transformer.wpe]:
            for p in module.parameters():
                if p.requires_grad:
                    p.requires_grad_(False)
                    frozen.append(p)
    return frozen


def _freeze_head(model):
    """Freeze lm_head and final layernorm. Returns list to unfreeze."""
    frozen = []
    for p in model.transformer.ln_f.parameters():
        if p.requires_grad:
            p.requires_grad_(False)
            frozen.append(p)
    for p in model.lm_head.parameters():
        if p.requires_grad:
            p.requires_grad_(False)
            frozen.append(p)
    return frozen


def _unfreeze(params):
    """Unfreeze a list of parameters."""
    for p in params:
        p.requires_grad_(True)


def _neg_grad_multipliers(model, active_layers, boosted_layers=(), boost_alpha=1.0):
    """Layer-selective training: {parameter name: multiplier on the negative rows' gradient}.

    A block's parameters get 1 if the block is active (boost_alpha if it is also boosted) and 0 if it is not.
    Parameters outside the blocks follow the block they sit next to: the token and position embeddings follow
    block 0, the final layer norm and the output head follow the last block. A token matrix shared by the
    embedding and the head (the GPT-2 default) sits next to both, so it gets 1 if either block is active.
    """
    last = model.config.n_layer - 1
    tied = model.lm_head.weight is model.transformer.wte.weight
    first_on, last_on = 0 in active_layers, last in active_layers
    out = {}
    for name, _ in model.named_parameters():
        parts = name.split(".")
        if "h" in parts and parts[parts.index("h") + 1].isdigit():
            i = int(parts[parts.index("h") + 1])
            out[name] = 0.0 if i not in active_layers else (float(boost_alpha) if i in boosted_layers else 1.0)
        elif "wte" in parts:
            out[name] = float(first_on or (tied and last_on))
        elif "wpe" in parts:
            out[name] = float(first_on)
        elif "ln_f" in parts or "lm_head" in parts:
            out[name] = float(last_on)
        else:
            raise ValueError(f"layer_selective: no rule for parameter {name!r}")
    return out


def _step_layer_selective(
    model, input_ids, attention_mask, weights,
    amp_dtype, use_amp, neg_loss_clamp, accum,
    neg_grad_top_k=4, neg_grad_layer_set=None,
    boost_layers=None, boost_alpha=2.0, **kwargs,
):
    """Layer-selective signed-weight training: the negative rows' gradient reaches only the chosen layers.

    Two passes, as in the other two-pass steps: one over the rows with weight >= 0, one over the rows with
    weight < 0. Every parameter gets the positive rows' gradient. The negative rows' gradient is added only to
    the parameters of the active layers (the top neg_grad_top_k blocks, or neg_grad_layer_set), times boost_alpha
    in boosted layers; _neg_grad_multipliers says what happens to the parameters outside the blocks. With every
    layer active and no boost this is the plain signed-weight gradient.

    Leaves the micro-batch's own mean gradient in .grad, not divided by accum (train_advanced does that).

    Rewritten 2026-11-14. The version before it was switched off on 2026-04-11: it froze parameters over three
    passes, and the third pass trained the non-active layers on every row at a positive weight, harmful rows
    included, which worked against the unlearning in the active layers (50M runs came out with safe perplexity
    above 1000). The gradient of this version is pinned in tests/test_layer_selective.py. No training run has
    validated it yet, and every layer-selective result from before the rewrite stays withdrawn.
    """
    n_layers = model.config.n_layer
    if neg_grad_layer_set is not None:
        active = set(neg_grad_layer_set)
    else:
        active = set(range(max(0, n_layers - neg_grad_top_k), n_layers))
    boosted = active & set(boost_layers) if boost_layers else set()

    pos_mask = weights >= 0
    neg_mask = weights < 0
    if not neg_mask.any():
        return _two_pass_fallback(model, input_ids, attention_mask, weights,
                                  amp_dtype, use_amp, neg_loss_clamp)

    def one_pass(rows):
        with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            out = model(input_ids=input_ids[rows], attention_mask=attention_mask[rows])
            logits = out.logits
            del out
            loss, met = compute_weighted_loss(
                logits, input_ids[rows], attention_mask[rows], weights[rows], neg_loss_clamp,
            )
            del logits
        loss.backward()
        return loss.item(), met

    n_pos, n_neg = int(pos_mask.sum().item()), int(neg_mask.sum().item())
    n_total = float(n_pos + n_neg)
    pos_grads, loss_pos, met_pos = {}, 0.0, {}
    if n_pos:
        loss_pos, met_pos = one_pass(pos_mask)
        pos_grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        model.zero_grad()
    loss_neg, met_neg = one_pass(neg_mask)

    # Combine: every parameter keeps the positive rows' gradient; the negative rows' gradient goes to the
    # active layers only. (A micro-batch with no positive row leaves the other layers a zero gradient.)
    mult = _neg_grad_multipliers(model, active, boosted, boost_alpha)
    for name, param in model.named_parameters():
        g_pos, g_neg = pos_grads.get(name), param.grad
        if g_pos is None and g_neg is None:
            continue
        grad = torch.zeros_like(param) if g_pos is None else (n_pos / n_total) * g_pos
        if g_neg is not None and mult[name] != 0.0:
            grad = grad + (n_neg / n_total) * mult[name] * g_neg
        param.grad = grad

    return {
        "loss_total": (loss_pos * n_pos + loss_neg * n_neg) / n_total,
        "ce_positive": met_pos.get("ce_positive", 0.0),
        "ce_negative": met_neg.get("ce_negative", 0.0),
        "loss_positive": loss_pos,
        "loss_negative": loss_neg,
        "ls_active_layers": len(active),
        "ls_boosted_layers": len(boosted),
    }


def _step_token_level(
    model, input_ids, attention_mask, weights,
    amp_dtype, use_amp, neg_loss_clamp, accum,
    token_specificity=None, **kwargs,
):
    """Token-level negative weighting: per-token weights based on content specificity.

    Structural tokens ('the', 'is', punctuation) get reduced negative weight.
    Content-specific tokens (unique to harmful text) get full negative weight.
    This preserves language learning from negative examples while unlearning harmful content.
    """
    if token_specificity is None:
        return _step_standard(model, input_ids, attention_mask, weights,
                              amp_dtype, use_amp, neg_loss_clamp, accum)

    with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        del outputs

        # Shift for causal LM
        shift_logits = logits[:, :-1, :].contiguous()
        del logits
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous().float()

        # Per-token cross-entropy
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        per_token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_logits.size(0), shift_logits.size(1))

        # Build per-token weight tensor
        neg_ex_mask = (weights < 0).unsqueeze(1)  # (batch, 1)

        # Token specificity lookup by label token
        tok_spec = token_specificity[shift_labels]  # (batch, seq-1)

        # Map specificity [0.5, 1.0] -> [0.0, 1.0]; below 0.5 -> 0
        neg_multiplier = (2.0 * (tok_spec - 0.5)).clamp(min=0.0)

        # Per-example weights expanded to (batch, seq-1)
        ex_weights = weights.unsqueeze(1).expand_as(per_token_loss)

        # For negative examples: scale per-token by specificity
        # For positive examples: keep original weight
        token_weights = torch.where(
            neg_ex_mask.expand_as(per_token_loss),
            ex_weights * neg_multiplier,
            ex_weights,
        )

        # Clamp for negative tokens
        if neg_loss_clamp > 0:
            neg_tok_mask = token_weights < 0
            if neg_tok_mask.any():
                per_token_loss = torch.where(
                    neg_tok_mask,
                    per_token_loss.clamp(max=neg_loss_clamp),
                    per_token_loss,
                )

        # Weighted per-token loss
        weighted = token_weights * per_token_loss * shift_mask
        tokens_per_ex = shift_mask.sum(dim=1).clamp(min=1)
        per_ex_loss = weighted.sum(dim=1) / tokens_per_ex
        total_loss = per_ex_loss.mean() / accum

    total_loss.backward()

    # Metrics
    pos_mask = weights > 0
    neg_mask_ex = weights < 0
    metrics = {
        "loss_total": per_ex_loss.mean().item(),
        "loss_unweighted_mean": per_token_loss.mean().item(),
    }
    if pos_mask.any():
        metrics["ce_positive"] = (per_token_loss[pos_mask] * shift_mask[pos_mask]).sum().item() / shift_mask[pos_mask].sum().clamp(min=1).item()
    if neg_mask_ex.any():
        active_neg = (neg_multiplier[neg_mask_ex] > 0.1) * shift_mask[neg_mask_ex]
        total_neg = shift_mask[neg_mask_ex]
        metrics["ce_negative"] = (per_token_loss[neg_mask_ex] * total_neg).sum().item() / total_neg.sum().clamp(min=1).item()
        metrics["tl_active_pct"] = active_neg.sum().item() / total_neg.sum().clamp(min=1).item()

    return metrics


def _step_adaptive(
    model, input_ids, attention_mask, weights,
    amp_dtype, use_amp, neg_loss_clamp, accum,
    adapt_threshold=5.0, layer_thresholds=None, **kwargs,
):
    """Adaptive negative weight: scale down negative weight when model already assigns low probability.

    If layer_thresholds is provided, different layers use different CE thresholds.
    This allows amplifying alignment pressure on specific layers (e.g., layer 0 with
    a lower threshold = more aggressive) while being gentler on others.

    When CE is high (model already doesn't "know" this example), reduce the negative
    gradient signal. When CE is low (model is "too good" at harmful text), apply full
    negative weight. This naturally prevents the runaway oscillation in unsafe PPL.
    """
    # If layer_thresholds specified, do per-layer-group adaptive passes
    if layer_thresholds:
        n_layers = model.config.n_layer

        # First: compute per-example CE (shared across all passes)
        with torch.no_grad():
            with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                del outputs
                shift_logits = logits[:, :-1, :].contiguous()
                del logits
                shift_labels = input_ids[:, 1:].contiguous()
                shift_mask = attention_mask[:, 1:].contiguous().float()
                loss_fct = nn.CrossEntropyLoss(reduction="none")
                per_token_loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                ).view(shift_logits.size(0), shift_logits.size(1))
                per_ex_ce = (per_token_loss * shift_mask).sum(dim=1) / shift_mask.sum(dim=1).clamp(min=1)

        neg_mask = weights < 0

        # Group layers by threshold
        threshold_groups = {}  # threshold -> set of layers
        for layer_idx, thresh in layer_thresholds.items():
            threshold_groups.setdefault(thresh, set()).add(layer_idx)
        # Default group: layers not in any explicit threshold
        default_layers = set(range(n_layers)) - set(layer_thresholds.keys())
        if default_layers:
            threshold_groups.setdefault(adapt_threshold, set())
            threshold_groups[adapt_threshold] |= default_layers

        all_scales = []
        # Do a pass per threshold group
        for thresh, layer_set in threshold_groups.items():
            other_layers = set(range(n_layers)) - layer_set
            frozen_params = _freeze_layers(model, other_layers,
                                           include_embeddings=(0 not in layer_set))
            frozen_params += _freeze_head(model)

            adapted_weights = weights.clone()
            if neg_mask.any():
                scale = torch.sigmoid(thresh - per_ex_ce[neg_mask].detach())
                adapted_weights[neg_mask] = weights[neg_mask] * scale
                all_scales.append(scale)

            if neg_loss_clamp > 0 and neg_mask.any():
                per_ex_ce_clamped = torch.where(neg_mask, per_ex_ce.clamp(max=neg_loss_clamp), per_ex_ce)
            else:
                per_ex_ce_clamped = per_ex_ce

            with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                del outputs
                loss, _ = compute_weighted_loss(logits, input_ids, attention_mask,
                                                adapted_weights, neg_loss_clamp)
                del logits
                loss = loss / accum
            loss.backward()

            _unfreeze(frozen_params)

        pos_mask = weights > 0
        combined_scale = torch.cat(all_scales) if all_scales else torch.tensor([1.0])
        metrics = {
            "loss_total": loss.item() * accum,
            "loss_unweighted_mean": per_ex_ce.mean().item(),
            "ce_positive": per_ex_ce[pos_mask].mean().item() if pos_mask.any() else 0,
            "ce_negative": per_ex_ce[neg_mask].mean().item() if neg_mask.any() else 0,
            "adp_mean_scale": combined_scale.mean().item(),
            "adp_min_scale": combined_scale.min().item(),
            "adp_layer_thresholds": len(threshold_groups),
        }
        return metrics

    # Standard adaptive (single threshold for all layers)
    with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        del outputs

        # Need per-example CE first to compute adaptive scaling
        shift_logits = logits[:, :-1, :].contiguous()
        del logits
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous().float()

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        per_token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_logits.size(0), shift_logits.size(1))

        per_token_loss_masked = per_token_loss * shift_mask
        tokens_per_ex = shift_mask.sum(dim=1).clamp(min=1)
        per_ex_ce = per_token_loss_masked.sum(dim=1) / tokens_per_ex

        # Adaptive scaling for negative examples
        neg_mask = weights < 0
        adapted_weights = weights.clone()
        if neg_mask.any():
            # sigmoid(threshold - CE): high when CE < threshold (model "knows" it), low when CE > threshold
            scale = torch.sigmoid(adapt_threshold - per_ex_ce[neg_mask].detach())
            adapted_weights[neg_mask] = weights[neg_mask] * scale

        # Clamp
        if neg_loss_clamp > 0 and neg_mask.any():
            per_ex_ce = torch.where(neg_mask, per_ex_ce.clamp(max=neg_loss_clamp), per_ex_ce)

        # Weighted loss
        weighted = adapted_weights * per_ex_ce
        total_loss = weighted.mean() / accum

    total_loss.backward()

    pos_mask = weights > 0
    metrics = {
        "loss_total": weighted.mean().item(),
        "loss_unweighted_mean": per_ex_ce.mean().item(),
    }
    if pos_mask.any():
        metrics["ce_positive"] = per_ex_ce[pos_mask].mean().item()
    if neg_mask.any():
        metrics["ce_negative"] = per_ex_ce[neg_mask].mean().item()
        metrics["adp_mean_scale"] = scale.mean().item() if neg_mask.any() else 1.0
        metrics["adp_min_scale"] = scale.min().item() if neg_mask.any() else 1.0

    return metrics


def _step_adaptive_projection(
    model, input_ids, attention_mask, weights,
    amp_dtype, use_amp, neg_loss_clamp, accum,
    adapt_threshold=5.0, **kwargs,
):
    """Combined adaptive weighting + gradient projection.

    Adaptive controls WHEN to apply negative gradients (only when model still "knows" it).
    Projection controls WHERE to apply them (only in directions orthogonal to capability).

    Step 1: Forward full batch, compute per-example CE, adaptively scale negative weights.
    Step 2: Split into pos/neg, do separate backward passes.
    Step 3: Project negative gradient orthogonal to positive gradient.
    """
    neg_mask = weights < 0
    pos_mask = weights >= 0

    if not neg_mask.any():
        return _two_pass_fallback(model, input_ids, attention_mask, weights,
                                  amp_dtype, use_amp, neg_loss_clamp)
    if not pos_mask.any():
        return _two_pass_fallback(model, input_ids, attention_mask, weights,
                                  amp_dtype, use_amp, neg_loss_clamp)

    # --- Phase 1: Adaptive weight scaling (single forward pass on neg examples) ---
    with torch.no_grad():
        with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            out_neg_probe = model(input_ids=input_ids[neg_mask], attention_mask=attention_mask[neg_mask])
            shift_logits = out_neg_probe.logits[:, :-1, :].contiguous()
            del out_neg_probe
            shift_labels = input_ids[neg_mask][:, 1:].contiguous()
            shift_mask_t = attention_mask[neg_mask][:, 1:].contiguous().float()
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            per_tok = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).view(shift_logits.size(0), shift_logits.size(1))
            del shift_logits
            per_ex_ce_neg = (per_tok * shift_mask_t).sum(dim=1) / shift_mask_t.sum(dim=1).clamp(min=1)

    scale = torch.sigmoid(adapt_threshold - per_ex_ce_neg)
    adapted_neg_weights = weights[neg_mask].clone() * scale

    # --- Phase 2: Forward + backward on POSITIVE examples ---
    with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
        out_pos = model(input_ids=input_ids[pos_mask], attention_mask=attention_mask[pos_mask])
        logits_pos = out_pos.logits
        del out_pos
        loss_pos, met_pos = compute_weighted_loss(
            logits_pos, input_ids[pos_mask], attention_mask[pos_mask],
            weights[pos_mask], neg_loss_clamp,
        )
        del logits_pos
    loss_pos.backward()

    grad_params = [(n, p) for n, p in model.named_parameters() if p.grad is not None]
    g_pos_flat = torch.cat([p.grad.flatten().float() for _, p in grad_params])
    g_pos_dict = {n: p.grad.clone() for n, p in grad_params}

    model.zero_grad()

    # --- Phase 3: Forward + backward on NEGATIVE examples (with adapted weights) ---
    with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
        out_neg = model(input_ids=input_ids[neg_mask], attention_mask=attention_mask[neg_mask])
        logits_neg = out_neg.logits
        del out_neg
        loss_neg, met_neg = compute_weighted_loss(
            logits_neg, input_ids[neg_mask], attention_mask[neg_mask],
            adapted_neg_weights, neg_loss_clamp,
        )
        del logits_neg
    loss_neg.backward()

    g_neg_flat = torch.cat([p.grad.flatten().float() for _, p in grad_params])

    # --- Phase 4: Project negative gradient orthogonal to positive ---
    dot = torch.dot(g_neg_flat, g_pos_flat)
    g_pos_norm_sq = torch.dot(g_pos_flat, g_pos_flat)

    projected = False
    if g_pos_norm_sq > 1e-12 and dot < 0:
        g_neg_proj = g_neg_flat - (dot / g_pos_norm_sq) * g_pos_flat
        projected = True
    else:
        g_neg_proj = g_neg_flat

    # --- Phase 5: Combine gradients ---
    n_pos = pos_mask.sum().item()
    n_neg = neg_mask.sum().item()
    n_total = float(n_pos + n_neg)

    offset = 0
    for name, param in grad_params:
        numel = param.numel()
        g_p = g_pos_dict[name].float()
        g_n = g_neg_proj[offset:offset + numel].view(param.shape)
        param.grad = ((n_pos / n_total) * g_p + (n_neg / n_total) * g_n).to(param.dtype)
        offset += numel

    metrics = {
        "loss_total": (loss_pos.item() * n_pos + loss_neg.item() * n_neg) / n_total,
        "ce_positive": met_pos.get("ce_positive", 0.0),
        "ce_negative": per_ex_ce_neg.mean().item(),
        "loss_positive": loss_pos.item(),
        "loss_negative": loss_neg.item(),
        "gp_dot": dot.item(),
        "gp_projected": projected,
        "adp_mean_scale": scale.mean().item(),
        "adp_min_scale": scale.min().item(),
    }
    return metrics


# ---------------------------------------------------------------------------
# Novel alignment methods
# ---------------------------------------------------------------------------

def _step_npo(
    model, input_ids, attention_mask, weights,
    amp_dtype, use_amp, neg_loss_clamp, accum,
    npo_beta=0.1, **kwargs,
):
    """Negative Preference Optimization: normalized log-sigmoid loss for negative examples.

    Based on Zhang et al. (2024). Instead of gradient ascent (which diverges),
    NPO uses: L_neg = -log(sigmoid(-beta * CE))

    This is bounded: as CE grows, sigmoid(-beta*CE) → 0, so -log(...) → large but
    the gradient approaches zero. Exponentially slower collapse than gradient ascent.

    Positive examples use standard CE. The combination provides stable alignment
    with a natural floor on the negative signal.
    """
    with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        del outputs

        shift_logits = logits[:, :-1, :].contiguous()
        del logits
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous().float()

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        per_token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_logits.size(0), shift_logits.size(1))

        per_token_loss = per_token_loss * shift_mask
        tokens_per_ex = shift_mask.sum(dim=1).clamp(min=1)
        per_ex_ce = per_token_loss.sum(dim=1) / tokens_per_ex

        pos_mask = weights > 0
        neg_mask = weights < 0

        # Positive: standard weighted CE
        if pos_mask.any():
            pos_loss = (weights[pos_mask] * per_ex_ce[pos_mask]).mean()
        else:
            pos_loss = torch.tensor(0.0, device=input_ids.device)

        # Negative: NPO loss = -log(sigmoid(-beta * CE))
        # = log(1 + exp(beta * CE)) ≈ beta * CE when CE is small, but saturates
        if neg_mask.any():
            neg_ce = per_ex_ce[neg_mask]
            neg_weights = weights[neg_mask].abs()
            # NPO: -log(sigmoid(-beta * CE)) = log(1 + exp(beta * CE))
            # Use logsigmoid for numerical stability: -logsigmoid(-beta * CE)
            npo_loss_per_ex = -torch.nn.functional.logsigmoid(-npo_beta * neg_ce)
            neg_loss = (neg_weights * npo_loss_per_ex).mean()
        else:
            neg_loss = torch.tensor(0.0, device=input_ids.device)
            npo_loss_per_ex = torch.tensor([0.0])

        n_pos = max(pos_mask.sum().item(), 1)
        n_neg = max(neg_mask.sum().item(), 1)
        n_total = n_pos + n_neg
        total_loss = (pos_loss * n_pos + neg_loss * n_neg) / n_total / accum

    total_loss.backward()

    metrics = {
        "loss_total": total_loss.item() * accum,
        "ce_positive": per_ex_ce[pos_mask].mean().item() if pos_mask.any() else 0,
        "ce_negative": per_ex_ce[neg_mask].mean().item() if neg_mask.any() else 0,
        "npo_loss_mean": npo_loss_per_ex.mean().item() if neg_mask.any() else 0,
        "npo_beta": npo_beta,
    }
    return metrics


class NSPOBuffer:
    """Rolling buffer of recent positive-batch gradients for null-space projection.

    Used by _step_nspo_real (true Niu et al. 2025 NSPO). Stores the last `max_size`
    flat positive-batch gradients in GPU bf16 as a single pre-allocated (N, D)
    tensor. At 200M params, N=16 → ~6.4 GB persistent VRAM.

    Projection formula: given G of shape (N, D) with rows = stored gradients,
        g_null = g - G^T @ (G G^T)^-1 @ G @ g

    The bf16 buffer is multiplied directly via tensor cores; matmul accumulates
    in fp32 (default for bf16 GEMM on Ampere+) so numerical precision matches
    the prior CPU-fp32 implementation while eliminating the per-step PCIe
    round-trip (which dominated step time at 200M).
    """

    def __init__(self, max_size=16):
        self.max_size = max_size
        self.buffer = None  # (N, D) bf16 on GPU, allocated lazily on first add()
        self.size = 0       # current populated rows
        self._head = 0      # next index to write (FIFO ring)

    def add(self, g_flat):
        """Append a gradient to the FIFO ring. Input is fp32 GPU tensor of shape (D,)."""
        g_flat = g_flat.detach()
        if self.buffer is None:
            D = g_flat.numel()
            self.buffer = torch.empty(self.max_size, D, dtype=torch.bfloat16, device=g_flat.device)
        self.buffer[self._head].copy_(g_flat, non_blocking=True)
        self._head = (self._head + 1) % self.max_size
        if self.size < self.max_size:
            self.size += 1

    def project(self, g_neg):
        """Project g_neg into the null space of the buffered gradient row-span.

        Returns (projected_tensor, projected_fraction) where projected_fraction
        is the relative norm removed: 1 - ||g_null|| / ||g_neg||.
        """
        if self.size == 0 or self.buffer is None:
            return g_neg, 0.0

        G = self.buffer[:self.size]                   # (N, D) bf16, view (no copy)
        g_bf = g_neg.to(dtype=torch.bfloat16)         # (D,) bf16

        # bf16 matmul on tensor cores accumulates in fp32 internally.
        inner = (G @ g_bf).float()                    # (N,)
        gram = (G @ G.T).float()                      # (N, N)
        N = self.size
        gram = gram + 1e-6 * torch.eye(N, device=gram.device, dtype=gram.dtype)

        try:
            coeffs = torch.linalg.solve(gram, inner)
        except Exception:
            coeffs = torch.linalg.pinv(gram) @ inner

        adjustment = (G.T @ coeffs.to(torch.bfloat16)).float()   # (D,)
        g_null = g_neg - adjustment

        frac = 1.0 - (g_null.norm().item() / max(g_neg.norm().item(), 1e-8))
        return g_null, frac


def _step_gp_unconditional(
    model, input_ids, attention_mask, weights,
    amp_dtype, use_amp, neg_loss_clamp, accum, **kwargs,
):
    """Unconditional PCGrad: always-project negative gradient against the current
    positive-batch gradient.

    NOTE (2026-04-11 bug fix): this function was previously called `_step_nspo`
    and marketed as the Niu et al. (2025) null-space-projection method. It is
    not. It is rank-1 PCGrad with the `if dot < 0` conflict guard removed.
    True NSPO projects against the accumulated span of multiple positive
    gradients — that is now implemented separately as `_step_nspo_real`.
    This method is kept for comparison/reproducibility of earlier results.
    """
    pos_mask = weights >= 0
    neg_mask = weights < 0

    if not neg_mask.any():
        return _two_pass_fallback(model, input_ids, attention_mask, weights,
                                  amp_dtype, use_amp, neg_loss_clamp)
    if not pos_mask.any():
        return _two_pass_fallback(model, input_ids, attention_mask, weights,
                                  amp_dtype, use_amp, neg_loss_clamp)

    # --- Pass 1: Positive examples → get capability gradient ---
    with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
        out_pos = model(input_ids=input_ids[pos_mask], attention_mask=attention_mask[pos_mask])
        logits_pos = out_pos.logits
        del out_pos
        loss_pos, met_pos = compute_weighted_loss(
            logits_pos, input_ids[pos_mask], attention_mask[pos_mask],
            weights[pos_mask], neg_loss_clamp,
        )
        del logits_pos
    loss_pos.backward()

    grad_params = [(n, p) for n, p in model.named_parameters() if p.grad is not None]
    g_pos_flat = torch.cat([p.grad.flatten().float() for _, p in grad_params])
    g_pos_dict = {n: p.grad.clone() for n, p in grad_params}

    model.zero_grad()

    # --- Pass 2: Negative examples → get alignment gradient ---
    with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
        out_neg = model(input_ids=input_ids[neg_mask], attention_mask=attention_mask[neg_mask])
        logits_neg = out_neg.logits
        del out_neg
        loss_neg, met_neg = compute_weighted_loss(
            logits_neg, input_ids[neg_mask], attention_mask[neg_mask],
            weights[neg_mask], neg_loss_clamp,
        )
        del logits_neg
    loss_neg.backward()

    g_neg_flat = torch.cat([p.grad.flatten().float() for _, p in grad_params])

    # --- Null-space projection ---
    # Compute capability subspace: use g_pos as the basis
    # For single-vector case: null space projection = I - (g_pos @ g_pos^T) / ||g_pos||^2
    # This removes ALL components along g_pos, not just conflicting ones
    g_pos_norm_sq = torch.dot(g_pos_flat, g_pos_flat)

    if g_pos_norm_sq > 1e-12:
        # Project g_neg into null space of g_pos
        proj_coeff = torch.dot(g_neg_flat, g_pos_flat) / g_pos_norm_sq
        g_neg_null = g_neg_flat - proj_coeff * g_pos_flat
        projected = True
    else:
        g_neg_null = g_neg_flat
        projected = False

    # --- Combine: capability gradient + null-space-projected alignment gradient ---
    n_pos = pos_mask.sum().item()
    n_neg = neg_mask.sum().item()
    n_total = float(n_pos + n_neg)

    offset = 0
    for name, param in grad_params:
        numel = param.numel()
        g_p = g_pos_dict[name].float()
        g_n = g_neg_null[offset:offset + numel].view(param.shape)
        param.grad = ((n_pos / n_total) * g_p + (n_neg / n_total) * g_n).to(param.dtype)
        offset += numel

    dot_before = torch.dot(g_neg_flat, g_pos_flat).item()
    dot_after = torch.dot(g_neg_null, g_pos_flat).item()

    metrics = {
        "loss_total": (loss_pos.item() * n_pos + loss_neg.item() * n_neg) / n_total,
        "ce_positive": met_pos.get("ce_positive", 0),
        "ce_negative": met_neg.get("ce_negative", 0),
        "gpu_dot_before": dot_before,
        "gpu_dot_after": dot_after,
        "gpu_projected": projected,
        "gpu_removed_pct": (1 - g_neg_null.norm().item() / max(g_neg_flat.norm().item(), 1e-8)) * 100,
    }
    return metrics


def _step_nspo_real(
    model, input_ids, attention_mask, weights,
    amp_dtype, use_amp, neg_loss_clamp, accum,
    nspo_buffer=None, **kwargs,
):
    """Real Niu et al. (2025) NSPO: null-space projection against an accumulated
    capability subspace.

    Maintains a rolling buffer of the last N positive-batch gradients. On each
    step, appends the current g_pos to the buffer, then projects g_neg into the
    null space of the row-span of the buffered gradient matrix before combining
    with the capability update.

    This is distinct from `_step_gp_unconditional` (formerly mislabeled
    `_step_nspo`) which projects only against the single current-batch
    positive gradient — a rank-1 operation, not a null-space projection.

    `nspo_buffer` is a persistent NSPOBuffer object initialized once in the
    training loop and mutated across steps.
    """
    pos_mask = weights >= 0
    neg_mask = weights < 0

    if not neg_mask.any():
        return _two_pass_fallback(model, input_ids, attention_mask, weights,
                                  amp_dtype, use_amp, neg_loss_clamp)
    if not pos_mask.any():
        return _two_pass_fallback(model, input_ids, attention_mask, weights,
                                  amp_dtype, use_amp, neg_loss_clamp)

    # --- Pass 1: Positive examples → capability gradient ---
    with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
        out_pos = model(input_ids=input_ids[pos_mask], attention_mask=attention_mask[pos_mask])
        logits_pos = out_pos.logits
        del out_pos
        loss_pos, met_pos = compute_weighted_loss(
            logits_pos, input_ids[pos_mask], attention_mask[pos_mask],
            weights[pos_mask], neg_loss_clamp,
        )
        del logits_pos
    loss_pos.backward()

    grad_params = [(n, p) for n, p in model.named_parameters() if p.grad is not None]
    g_pos_flat = torch.cat([p.grad.flatten().float() for _, p in grad_params])
    g_pos_dict = {n: p.grad.clone() for n, p in grad_params}

    # Append current capability gradient to the rolling null-space basis.
    # Must happen BEFORE the projection so the current step contributes to its own subspace.
    if nspo_buffer is not None:
        nspo_buffer.add(g_pos_flat)

    model.zero_grad()

    # --- Pass 2: Negative examples → raw alignment gradient ---
    with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
        out_neg = model(input_ids=input_ids[neg_mask], attention_mask=attention_mask[neg_mask])
        logits_neg = out_neg.logits
        del out_neg
        loss_neg, met_neg = compute_weighted_loss(
            logits_neg, input_ids[neg_mask], attention_mask[neg_mask],
            weights[neg_mask], neg_loss_clamp,
        )
        del logits_neg
    loss_neg.backward()

    g_neg_flat = torch.cat([p.grad.flatten().float() for _, p in grad_params])

    # --- Null-space projection against accumulated capability subspace ---
    if nspo_buffer is not None and nspo_buffer.size > 0:
        g_neg_null, projected_fraction = nspo_buffer.project(g_neg_flat)
    else:
        g_neg_null = g_neg_flat
        projected_fraction = 0.0

    # --- Combine capability + projected alignment ---
    n_pos = pos_mask.sum().item()
    n_neg = neg_mask.sum().item()
    n_total = float(n_pos + n_neg)

    offset = 0
    for name, param in grad_params:
        numel = param.numel()
        g_p = g_pos_dict[name].float()
        g_n = g_neg_null[offset:offset + numel].view(param.shape)
        param.grad = ((n_pos / n_total) * g_p + (n_neg / n_total) * g_n).to(param.dtype)
        offset += numel

    buffer_size = nspo_buffer.size if nspo_buffer is not None else 0

    metrics = {
        "loss_total": (loss_pos.item() * n_pos + loss_neg.item() * n_neg) / n_total,
        "ce_positive": met_pos.get("ce_positive", 0),
        "ce_negative": met_neg.get("ce_negative", 0),
        "nspo_buffer_size": buffer_size,
        "nspo_projected_frac": projected_fraction,
    }
    return metrics


def _step_adaptive_nspo_rank1(
    model, input_ids, attention_mask, weights,
    amp_dtype, use_amp, neg_loss_clamp, accum,
    adapt_threshold=5.0, **kwargs,
):
    """Adaptive + rank-1 GP projection (legacy, formerly labeled adaptive_nspo).

    NOTE (2026-04-11 bug fix): this is adaptive scaling plus the rank-1
    always-project GP (what we previously mislabeled NSPO). Kept for
    backward compatibility / reproducing earlier results. The new
    `_step_adaptive_nspo` uses the real accumulated-buffer NSPO projection.
    """
    pos_mask = weights >= 0
    neg_mask = weights < 0

    if not neg_mask.any():
        return _two_pass_fallback(model, input_ids, attention_mask, weights,
                                  amp_dtype, use_amp, neg_loss_clamp)
    if not pos_mask.any():
        return _two_pass_fallback(model, input_ids, attention_mask, weights,
                                  amp_dtype, use_amp, neg_loss_clamp)

    # --- Phase 1: Compute adaptive scaling (no grad needed) ---
    with torch.no_grad():
        with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            out_probe = model(input_ids=input_ids[neg_mask], attention_mask=attention_mask[neg_mask])
            shift_logits = out_probe.logits[:, :-1, :].contiguous()
            del out_probe
            shift_labels = input_ids[neg_mask][:, 1:].contiguous()
            shift_mask = attention_mask[neg_mask][:, 1:].contiguous().float()
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            per_tok = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).view(shift_logits.size(0), shift_logits.size(1))
            del shift_logits
            per_ex_ce = (per_tok * shift_mask).sum(dim=1) / shift_mask.sum(dim=1).clamp(min=1)

    scale = torch.sigmoid(adapt_threshold - per_ex_ce)
    adapted_neg_weights = weights[neg_mask].clone() * scale

    # --- Phase 2: Capability gradient (positive examples) ---
    with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
        out_pos = model(input_ids=input_ids[pos_mask], attention_mask=attention_mask[pos_mask])
        logits_pos = out_pos.logits
        del out_pos
        loss_pos, met_pos = compute_weighted_loss(
            logits_pos, input_ids[pos_mask], attention_mask[pos_mask],
            weights[pos_mask], neg_loss_clamp,
        )
        del logits_pos
    loss_pos.backward()

    grad_params = [(n, p) for n, p in model.named_parameters() if p.grad is not None]
    g_pos_flat = torch.cat([p.grad.flatten().float() for _, p in grad_params])
    g_pos_dict = {n: p.grad.clone() for n, p in grad_params}

    model.zero_grad()

    # --- Phase 3: Alignment gradient (negative examples with adapted weights) ---
    with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
        out_neg = model(input_ids=input_ids[neg_mask], attention_mask=attention_mask[neg_mask])
        logits_neg = out_neg.logits
        del out_neg
        loss_neg, met_neg = compute_weighted_loss(
            logits_neg, input_ids[neg_mask], attention_mask[neg_mask],
            adapted_neg_weights, neg_loss_clamp,
        )
        del logits_neg
    loss_neg.backward()

    g_neg_flat = torch.cat([p.grad.flatten().float() for _, p in grad_params])

    # --- Phase 4: Null-space projection ---
    g_pos_norm_sq = torch.dot(g_pos_flat, g_pos_flat)
    dot_before = torch.dot(g_neg_flat, g_pos_flat).item()

    if g_pos_norm_sq > 1e-12:
        proj_coeff = torch.dot(g_neg_flat, g_pos_flat) / g_pos_norm_sq
        g_neg_null = g_neg_flat - proj_coeff * g_pos_flat
    else:
        g_neg_null = g_neg_flat

    dot_after = torch.dot(g_neg_null, g_pos_flat).item()

    # --- Phase 5: Combine ---
    n_pos = pos_mask.sum().item()
    n_neg = neg_mask.sum().item()
    n_total = float(n_pos + n_neg)

    offset = 0
    for name, param in grad_params:
        numel = param.numel()
        g_p = g_pos_dict[name].float()
        g_n = g_neg_null[offset:offset + numel].view(param.shape)
        param.grad = ((n_pos / n_total) * g_p + (n_neg / n_total) * g_n).to(param.dtype)
        offset += numel

    metrics = {
        "loss_total": (loss_pos.item() * n_pos + loss_neg.item() * n_neg) / n_total,
        "ce_positive": met_pos.get("ce_positive", 0),
        "ce_negative": per_ex_ce.mean().item(),
        "adp_mean_scale": scale.mean().item(),
        "adp_min_scale": scale.min().item(),
        "gpu_dot_before": dot_before,
        "gpu_dot_after": dot_after,
    }
    return metrics


def _step_adaptive_nspo(
    model, input_ids, attention_mask, weights,
    amp_dtype, use_amp, neg_loss_clamp, accum,
    adapt_threshold=5.0, nspo_buffer=None, **kwargs,
):
    """Adaptive + real NSPO (Niu et al. 2025).

    Adaptive controls WHEN to apply alignment (sigmoid scaling by per-example CE).
    Real NSPO controls WHERE the alignment gradient goes (null-space projection
    against the accumulated capability subspace, not just the current batch).

    1. Compute per-example CE on negative examples → adaptive scale
    2. Forward + backward positives → capability gradient; append to NSPO buffer
    3. Forward + backward negatives with adapted weights → alignment gradient
    4. Project alignment gradient into the buffer's null space
    5. Combine capability + projected adaptive alignment
    """
    pos_mask = weights >= 0
    neg_mask = weights < 0

    if not neg_mask.any():
        return _two_pass_fallback(model, input_ids, attention_mask, weights,
                                  amp_dtype, use_amp, neg_loss_clamp)
    if not pos_mask.any():
        return _two_pass_fallback(model, input_ids, attention_mask, weights,
                                  amp_dtype, use_amp, neg_loss_clamp)

    # --- Phase 1: adaptive scaling (no grad) ---
    with torch.no_grad():
        with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            out_probe = model(input_ids=input_ids[neg_mask], attention_mask=attention_mask[neg_mask])
            shift_logits = out_probe.logits[:, :-1, :].contiguous()
            del out_probe
            shift_labels = input_ids[neg_mask][:, 1:].contiguous()
            shift_mask = attention_mask[neg_mask][:, 1:].contiguous().float()
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            per_tok = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).view(shift_logits.size(0), shift_logits.size(1))
            del shift_logits
            per_ex_ce = (per_tok * shift_mask).sum(dim=1) / shift_mask.sum(dim=1).clamp(min=1)

    scale = torch.sigmoid(adapt_threshold - per_ex_ce)
    adapted_neg_weights = weights[neg_mask].clone() * scale

    # --- Phase 2: capability gradient ---
    with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
        out_pos = model(input_ids=input_ids[pos_mask], attention_mask=attention_mask[pos_mask])
        logits_pos = out_pos.logits
        del out_pos
        loss_pos, met_pos = compute_weighted_loss(
            logits_pos, input_ids[pos_mask], attention_mask[pos_mask],
            weights[pos_mask], neg_loss_clamp,
        )
        del logits_pos
    loss_pos.backward()

    grad_params = [(n, p) for n, p in model.named_parameters() if p.grad is not None]
    g_pos_flat = torch.cat([p.grad.flatten().float() for _, p in grad_params])
    g_pos_dict = {n: p.grad.clone() for n, p in grad_params}

    if nspo_buffer is not None:
        nspo_buffer.add(g_pos_flat)

    model.zero_grad()

    # --- Phase 3: alignment gradient with adapted weights ---
    with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
        out_neg = model(input_ids=input_ids[neg_mask], attention_mask=attention_mask[neg_mask])
        logits_neg = out_neg.logits
        del out_neg
        loss_neg, met_neg = compute_weighted_loss(
            logits_neg, input_ids[neg_mask], attention_mask[neg_mask],
            adapted_neg_weights, neg_loss_clamp,
        )
        del logits_neg
    loss_neg.backward()

    g_neg_flat = torch.cat([p.grad.flatten().float() for _, p in grad_params])

    # --- Phase 4: real null-space projection against accumulated buffer ---
    if nspo_buffer is not None and nspo_buffer.size > 0:
        g_neg_null, projected_fraction = nspo_buffer.project(g_neg_flat)
    else:
        g_neg_null = g_neg_flat
        projected_fraction = 0.0

    # --- Phase 5: combine ---
    n_pos = pos_mask.sum().item()
    n_neg = neg_mask.sum().item()
    n_total = float(n_pos + n_neg)

    offset = 0
    for name, param in grad_params:
        numel = param.numel()
        g_p = g_pos_dict[name].float()
        g_n = g_neg_null[offset:offset + numel].view(param.shape)
        param.grad = ((n_pos / n_total) * g_p + (n_neg / n_total) * g_n).to(param.dtype)
        offset += numel

    buffer_size = nspo_buffer.size if nspo_buffer is not None else 0

    metrics = {
        "loss_total": (loss_pos.item() * n_pos + loss_neg.item() * n_neg) / n_total,
        "ce_positive": met_pos.get("ce_positive", 0),
        "ce_negative": per_ex_ce.mean().item(),
        "adp_mean_scale": scale.mean().item(),
        "adp_min_scale": scale.min().item(),
        "nspo_buffer_size": buffer_size,
        "nspo_projected_frac": projected_fraction,
    }
    return metrics


def _step_full_stack(
    model, input_ids, attention_mask, weights,
    amp_dtype, use_amp, neg_loss_clamp, accum,
    adapt_threshold=5.0, sparse_k_pct=1.0, **kwargs,
):
    """Full stack: Adaptive + NSPO + Sparse negative gradients.

    Combines all three orthogonal alignment mechanisms:
    1. Adaptive: confidence-scaled negative weights (controls how much)
    2. NSPO: null-space projection (controls which gradient directions)
    3. Sparse: top-K% parameter selection (controls which parameters)

    Compatible with delayed alignment (handled by training loop).
    """
    pos_mask = weights >= 0
    neg_mask = weights < 0

    if not neg_mask.any():
        return _two_pass_fallback(model, input_ids, attention_mask, weights,
                                  amp_dtype, use_amp, neg_loss_clamp)
    if not pos_mask.any():
        return _two_pass_fallback(model, input_ids, attention_mask, weights,
                                  amp_dtype, use_amp, neg_loss_clamp)

    # --- Phase 1: Adaptive scaling ---
    with torch.no_grad():
        with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            out_probe = model(input_ids=input_ids[neg_mask], attention_mask=attention_mask[neg_mask])
            shift_logits = out_probe.logits[:, :-1, :].contiguous()
            del out_probe
            shift_labels = input_ids[neg_mask][:, 1:].contiguous()
            shift_mask = attention_mask[neg_mask][:, 1:].contiguous().float()
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            per_tok = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).view(shift_logits.size(0), shift_logits.size(1))
            del shift_logits
            per_ex_ce = (per_tok * shift_mask).sum(dim=1) / shift_mask.sum(dim=1).clamp(min=1)

    scale = torch.sigmoid(adapt_threshold - per_ex_ce)
    adapted_neg_weights = weights[neg_mask].clone() * scale

    # --- Phase 2: Capability gradient ---
    with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
        out_pos = model(input_ids=input_ids[pos_mask], attention_mask=attention_mask[pos_mask])
        logits_pos = out_pos.logits
        del out_pos
        loss_pos, met_pos = compute_weighted_loss(
            logits_pos, input_ids[pos_mask], attention_mask[pos_mask],
            weights[pos_mask], neg_loss_clamp,
        )
        del logits_pos
    loss_pos.backward()

    grad_params = [(n, p) for n, p in model.named_parameters() if p.grad is not None]
    g_pos_flat = torch.cat([p.grad.flatten().float() for _, p in grad_params])
    g_pos_dict = {n: p.grad.clone() for n, p in grad_params}
    model.zero_grad()

    # --- Phase 3: Alignment gradient (with adaptive weights) ---
    with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
        out_neg = model(input_ids=input_ids[neg_mask], attention_mask=attention_mask[neg_mask])
        logits_neg = out_neg.logits
        del out_neg
        loss_neg, met_neg = compute_weighted_loss(
            logits_neg, input_ids[neg_mask], attention_mask[neg_mask],
            adapted_neg_weights, neg_loss_clamp,
        )
        del logits_neg
    loss_neg.backward()

    g_neg_flat = torch.cat([p.grad.flatten().float() for _, p in grad_params])

    # --- Phase 4: NSPO (null-space projection) ---
    g_pos_norm_sq = torch.dot(g_pos_flat, g_pos_flat)
    if g_pos_norm_sq > 1e-12:
        proj_coeff = torch.dot(g_neg_flat, g_pos_flat) / g_pos_norm_sq
        g_neg_null = g_neg_flat - proj_coeff * g_pos_flat
    else:
        g_neg_null = g_neg_flat

    # --- Phase 5: Sparse (keep top K% by magnitude) ---
    total_params = g_neg_null.numel()
    k = max(1, int(total_params * sparse_k_pct / 100.0))
    if k < total_params:
        threshold = torch.topk(g_neg_null.abs(), k).values[-1]
        sparse_mask = g_neg_null.abs() >= threshold
        g_neg_sparse = g_neg_null * sparse_mask.float()
        n_active = sparse_mask.sum().item()
    else:
        g_neg_sparse = g_neg_null
        n_active = total_params

    # --- Phase 6: Combine ---
    n_pos = pos_mask.sum().item()
    n_neg = neg_mask.sum().item()
    n_total = float(n_pos + n_neg)

    offset = 0
    for name, param in grad_params:
        numel = param.numel()
        g_p = g_pos_dict[name].float()
        g_n = g_neg_sparse[offset:offset + numel].view(param.shape)
        param.grad = ((n_pos / n_total) * g_p + (n_neg / n_total) * g_n).to(param.dtype)
        offset += numel

    metrics = {
        "loss_total": (loss_pos.item() * n_pos + loss_neg.item() * n_neg) / n_total,
        "ce_positive": met_pos.get("ce_positive", 0),
        "ce_negative": per_ex_ce.mean().item(),
        "adp_mean_scale": scale.mean().item(),
        "sparse_n_active": n_active,
        "sparse_pct_actual": n_active / total_params * 100,
    }
    return metrics


def _step_kl_uniform(
    model, input_ids, attention_mask, weights,
    amp_dtype, use_amp, neg_loss_clamp, accum, **kwargs,
):
    """KL-toward-uniform: push negative examples toward maximum entropy.

    Positive examples: standard CE loss (learn to predict tokens).
    Negative examples: minimize KL(P_model || Uniform) — become maximally uncertain.

    This makes the model "forget" harmful patterns rather than learning anti-patterns.
    Natural floor: once the distribution is uniform, gradient vanishes.
    """
    with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        del outputs

        shift_logits = logits[:, :-1, :].contiguous()
        del logits
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous().float()
        vocab_size = shift_logits.size(-1)

        # Per-token CE for positive examples (standard)
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        per_token_ce = loss_fct(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
        ).view(shift_logits.size(0), shift_logits.size(1))

        tokens_per_ex = shift_mask.sum(dim=1).clamp(min=1)

        pos_mask = weights > 0
        neg_mask = weights < 0

        # Positive loss: standard weighted CE
        pos_ce = (per_token_ce * shift_mask).sum(dim=1) / tokens_per_ex
        pos_loss = (weights * pos_ce)[pos_mask].mean() if pos_mask.any() else torch.tensor(0.0, device=input_ids.device)

        # Negative loss: KL(P_model || Uniform) = log(V) - H(P_model)
        # = log(V) + sum_v P(v) log P(v)
        # Minimizing this pushes P toward uniform (maximizes entropy)
        if neg_mask.any():
            neg_logits = shift_logits[neg_mask]  # (n_neg, seq-1, vocab)
            neg_log_probs = torch.nn.functional.log_softmax(neg_logits, dim=-1)
            neg_probs = neg_log_probs.exp()
            # KL per token = sum_v P(v) log P(v) + log(V)
            per_token_kl = (neg_probs * neg_log_probs).sum(dim=-1) + math.log(vocab_size)
            # Mask and average per example
            neg_shift_mask = shift_mask[neg_mask]
            per_ex_kl = (per_token_kl * neg_shift_mask).sum(dim=1) / tokens_per_ex[neg_mask]
            # Weight by magnitude of negative weight (more negative = push harder)
            neg_loss = (weights[neg_mask].abs() * per_ex_kl).mean()
        else:
            neg_loss = torch.tensor(0.0, device=input_ids.device)
            per_ex_kl = torch.tensor([0.0])

        n_total = pos_mask.sum() + neg_mask.sum()
        n_pos = max(pos_mask.sum().item(), 1)
        n_neg = max(neg_mask.sum().item(), 1)
        total_loss = (pos_loss * n_pos + neg_loss * n_neg) / n_total / accum

    total_loss.backward()

    metrics = {
        "loss_total": total_loss.item() * accum,
        "ce_positive": pos_ce[pos_mask].mean().item() if pos_mask.any() else 0,
        "ce_negative": (per_token_ce * shift_mask)[neg_mask].sum().item() / tokens_per_ex[neg_mask].sum().item() if neg_mask.any() else 0,
        "kl_uniform_mean": per_ex_kl.mean().item() if neg_mask.any() else 0,
    }
    return metrics


def _step_margin(
    model, input_ids, attention_mask, weights,
    amp_dtype, use_amp, neg_loss_clamp, accum,
    margin_target=2.0, **kwargs,
):
    """Margin-based alignment: enforce a minimum CE gap between positive and negative examples.

    Loss = CE_positive + max(0, margin - (CE_negative - CE_positive_mean))

    Only applies alignment pressure when the gap is below the target margin.
    Once the model is sufficiently worse at harmful content, alignment pressure
    vanishes and the model trains purely on capability.

    The margin is a direct, interpretable knob: "how much worse (in CE) should
    the model be at harmful content?"
    """
    with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        del outputs

        shift_logits = logits[:, :-1, :].contiguous()
        del logits
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous().float()

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        per_token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_logits.size(0), shift_logits.size(1))

        per_token_loss = per_token_loss * shift_mask
        tokens_per_ex = shift_mask.sum(dim=1).clamp(min=1)
        per_ex_ce = per_token_loss.sum(dim=1) / tokens_per_ex

        pos_mask = weights > 0
        neg_mask = weights < 0

        # Positive examples: standard weighted CE loss (capability)
        if pos_mask.any():
            pos_loss = (weights[pos_mask] * per_ex_ce[pos_mask]).mean()
            mean_pos_ce = per_ex_ce[pos_mask].mean().detach()
        else:
            pos_loss = torch.tensor(0.0, device=input_ids.device)
            mean_pos_ce = per_ex_ce.mean().detach()

        # Negative examples: margin loss
        # Penalize when individual neg example CE is too close to mean positive CE
        if neg_mask.any():
            neg_ce = per_ex_ce[neg_mask]
            neg_weights = weights[neg_mask].abs()
            # Per-example margin violation: how far below the target gap?
            gap_per_ex = neg_ce - mean_pos_ce
            violations = torch.relu(margin_target - gap_per_ex)
            # Weight by magnitude of negative weight
            margin_loss = (neg_weights * violations).mean()
            n_violating = (violations > 0).sum().item()
        else:
            margin_loss = torch.tensor(0.0, device=input_ids.device)
            n_violating = 0

        n_pos = max(pos_mask.sum().item(), 1)
        n_neg = max(neg_mask.sum().item(), 1)
        n_total = n_pos + n_neg
        total_loss = (pos_loss * n_pos + margin_loss * n_neg) / n_total / accum

    total_loss.backward()

    metrics = {
        "loss_total": total_loss.item() * accum,
        "ce_positive": per_ex_ce[pos_mask].mean().item() if pos_mask.any() else 0,
        "ce_negative": per_ex_ce[neg_mask].mean().item() if neg_mask.any() else 0,
        "margin_violations": n_violating,
        "margin_violation_pct": n_violating / max(neg_mask.sum().item(), 1),
        "margin_mean_gap": (per_ex_ce[neg_mask] - mean_pos_ce).mean().item() if neg_mask.any() else 0,
    }
    return metrics


def _step_sparse_negative(
    model, input_ids, attention_mask, weights,
    amp_dtype, use_amp, neg_loss_clamp, accum,
    sparse_k_pct=10.0, sparse_rescale_mode="budget", neg_force_mult=1.0,
    mask_ema_decay=0.0, mask_ema_state=None, **kwargs,
):
    """Sparse parameter alignment: apply negative gradients only to top-K% parameters.

    1. Forward + backward on positive examples → all params get capability gradients
    2. Forward + backward on negative examples → compute full neg gradient
    3. Zero out all but top-K% of neg gradient (by magnitude) → apply sparse alignment

    Only the parameters most responsive to the alignment signal get modified.
    This focuses alignment pressure and reduces collateral damage to capability.

    A micro-batch with no negative row gets the plain gradient (_two_pass_fallback). A micro-batch with no
    positive row gets the negative gradient on its top K% of parameters, not rescaled.
    """
    pos_mask = weights >= 0
    neg_mask = weights < 0

    if not neg_mask.any():
        return _two_pass_fallback(model, input_ids, attention_mask, weights,
                                  amp_dtype, use_amp, neg_loss_clamp)

    # --- Pass 1: Positive examples (full gradient, all params) ---
    has_pos = bool(pos_mask.any())
    if has_pos:
        with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            out_pos = model(input_ids=input_ids[pos_mask], attention_mask=attention_mask[pos_mask])
            logits_pos = out_pos.logits
            del out_pos
            loss_pos, met_pos = compute_weighted_loss(
                logits_pos, input_ids[pos_mask], attention_mask[pos_mask],
                weights[pos_mask], neg_loss_clamp,
            )
            del logits_pos
        loss_pos.backward()
    else:
        met_pos = {}

    # Save positive gradients
    grad_params = [(n, p) for n, p in model.named_parameters() if p.grad is not None]
    pos_grads = {n: p.grad.clone() for n, p in grad_params}
    model.zero_grad()

    # --- Pass 2: Negative examples (full gradient, but we'll sparsify) ---
    with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
        out_neg = model(input_ids=input_ids[neg_mask], attention_mask=attention_mask[neg_mask])
        logits_neg = out_neg.logits
        del out_neg
        loss_neg, met_neg = compute_weighted_loss(
            logits_neg, input_ids[neg_mask], attention_mask[neg_mask],
            weights[neg_mask], neg_loss_clamp,
        )
        del logits_neg
    loss_neg.backward()
    if not has_pos:
        # A micro-batch of negative rows only. Until 2026-11-14 this crashed (no positive pass, so no parameter
        # list). There is no positive gradient to add or to match in size: the step is the negative gradient on
        # its top K% of parameters, not rescaled (pos_norm is 0, so neg_rescale stays 1).
        grad_params = [(n, p) for n, p in model.named_parameters() if p.grad is not None]

    # --- Sparsify negative gradients: keep only top K% by magnitude ---
    # We only need |g_neg| flattened (for the global top-K threshold and the
    # masked-neg norm) plus the bool mask; the combine loop below reads the
    # *signed* neg gradient straight from each param.grad, so we never build a
    # signed flat copy. One D-sized abs buffer (freed before the combine) instead
    # of the previous four full-width temporaries (signed-cat, abs, pos-cat,
    # masked-cat) — the bulk of the per-step VRAM saving. _foreach_abs fuses the
    # per-param abs into one kernel so the cleanup doesn't cost step time.
    abs_parts = torch._foreach_abs([p.grad for _, p in grad_params])
    abs_flat = torch.cat([a.view(-1) for a in abs_parts])
    del abs_parts
    total_params = abs_flat.numel()
    k = max(1, int(total_params * sparse_k_pct / 100.0))

    # Importance score for the top-K selection. Default = instantaneous |g_neg|,
    # which makes the "aligned" param set DRIFT every step (the jagged-landscape /
    # gap-jitter noise source). With mask_ema_decay>0 we instead select from a
    # running EMA of |g_neg| so the mask tracks the *consistently* important params
    # — a much more stable active set across steps and (hoped) across seeds.
    if mask_ema_decay > 0.0 and mask_ema_state is not None:
        if mask_ema_state.get("imp") is None:
            mask_ema_state["imp"] = abs_flat.clone()
        else:
            mask_ema_state["imp"].mul_(mask_ema_decay).add_(abs_flat, alpha=1.0 - mask_ema_decay)
        score = mask_ema_state["imp"]
    else:
        score = abs_flat

    if k < total_params:
        threshold = torch.topk(score, k).values[-1]
        sparse_mask_flat = score >= threshold
    else:
        sparse_mask_flat = torch.ones_like(score, dtype=torch.bool)

    n_active = sparse_mask_flat.sum().item()

    # --- Compute pos gradient norm (over all params) for magnitude matching ---
    # ||concat(pos_grads)|| == sqrt(sum of per-param squared norms) == norm of the
    # vector of per-param norms. Computed with one fused _foreach_norm (no D-sized
    # concatenation, no per-param Python loop).
    pos_norm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(list(pos_grads.values())))).item() if has_pos else 0.0

    # The sparse-negative concentrates the largest |g_neg| values onto top-K%
    # parameters. Their squared sum dominates total norm — when mixed with the
    # dense pos gradient and then globally clip_grad_norm'd to 1.0, the spike
    # forces the clip to scale ALL grads (including the pos signal on the
    # inactive 99%) down hard, suppressing capability learning.
    # Fix: rescale the sparse neg vector to have the same global norm as
    # the pos vector. Both contributions then get equal magnitude budget and
    # the global clip preserves them proportionally.
    # ||masked signed neg|| == ||masked |neg|||, so take the norm of the gathered
    # active abs values (~K% the size of a full masked copy).
    neg_sparse_norm = abs_flat[sparse_mask_flat].norm().item()
    del abs_flat, score  # free the D-sized abs buffer before the per-param combine
    # Target norm for the masked neg vector:
    #   "budget"    -> pos_norm: total neg L2 == total pos L2. The same budget
    #                  lands on only K% of params, so per-param push is ~1/sqrt(K)
    #                  stronger than dense (the "concentrated push").
    #   "per_param" -> pos_norm * sqrt(K): the masked vector carries only the K%
    #                  fraction of the budget, so each active param gets the SAME
    #                  magnitude it would in a dense update. Isolates footprint
    #                  from per-param strength.
    if sparse_rescale_mode == "per_param":
        target_norm = pos_norm * (n_active / max(total_params, 1)) ** 0.5
    else:
        target_norm = pos_norm
    target_norm *= neg_force_mult  # boost total alignment force above the capability budget
    if neg_sparse_norm > 1e-12 and target_norm > 1e-12:
        neg_rescale = target_norm / neg_sparse_norm
    else:
        neg_rescale = 1.0

    # --- Combine: pos_grad + rescaled sparse neg_grad ---
    n_pos = max(pos_mask.sum().item(), 1)
    n_neg = max(neg_mask.sum().item(), 1)
    n_total = float(n_pos + n_neg)

    offset = 0
    for name, param in grad_params:
        numel = param.numel()
        g_neg = param.grad.clone()

        mask = sparse_mask_flat[offset:offset + numel].view(param.shape)
        g_neg_sparse = g_neg * mask.float() * neg_rescale

        if has_pos:
            g_pos = pos_grads[name]
            param.grad = (n_pos / n_total) * g_pos + (n_neg / n_total) * g_neg_sparse
        else:
            param.grad = g_neg_sparse
        offset += numel

    metrics = {
        "loss_total": (loss_pos.item() * n_pos + loss_neg.item() * n_neg) / n_total if pos_mask.any() else loss_neg.item(),
        "ce_positive": met_pos.get("ce_positive", 0),
        "ce_negative": met_neg.get("ce_negative", 0),
        "sparse_k_pct": sparse_k_pct,
        "sparse_n_active": n_active,
        "sparse_pct_actual": n_active / total_params * 100,
        "sparse_pos_norm": pos_norm,
        "sparse_neg_sparse_norm": neg_sparse_norm,
        "sparse_neg_rescale": neg_rescale,
        "sparse_rescale_mode": sparse_rescale_mode,
    }
    return metrics


def _step_activation_steering(
    model, input_ids, attention_mask, weights,
    amp_dtype, use_amp, neg_loss_clamp, accum,
    steer_layer=0, steer_weight=0.1, **kwargs,
):
    """Activation steering: train a 'safety direction' in hidden state space.

    In addition to the standard signed-weight loss, adds a contrastive regularization
    term that pushes positive and negative examples apart in activation space at the
    specified layer. This explicitly trains the model to develop a linear safety feature.

    loss = standard_signed_weight_loss + steer_weight * steering_loss

    steering_loss = -mean(cos_sim(h_pos, direction)) + mean(cos_sim(h_neg, direction))
    where direction = mean(h_pos) - mean(h_neg) (the safety direction)
    """
    pos_mask = weights > 0
    neg_mask = weights < 0

    if not neg_mask.any() or not pos_mask.any():
        return _step_standard(model, input_ids, attention_mask, weights,
                              amp_dtype, use_amp, neg_loss_clamp, accum)

    # Register hook to capture hidden states at target layer
    captured = {}
    def hook_fn(module, input, output):
        # GPT2 block output is (hidden_states, ...) or just hidden_states
        if isinstance(output, tuple):
            captured["hidden"] = output[0]
        else:
            captured["hidden"] = output

    handle = model.transformer.h[steer_layer].register_forward_hook(hook_fn)

    with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        del outputs

        # Standard signed-weight loss
        loss_main, metrics = compute_weighted_loss(
            logits, input_ids, attention_mask, weights, neg_loss_clamp,
        )
        del logits

        # Get captured hidden states and compute steering loss
        hidden = captured["hidden"]  # (batch, seq, hidden_dim)

        # Pool hidden states: mean over non-padding tokens
        mask_expanded = attention_mask.unsqueeze(-1).float()  # (batch, seq, 1)
        pooled = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        # (batch, hidden_dim)

        # Safety direction: mean positive - mean negative
        h_pos = pooled[pos_mask]  # (n_pos, hidden_dim)
        h_neg = pooled[neg_mask]  # (n_neg, hidden_dim)

        direction = h_pos.mean(dim=0) - h_neg.mean(dim=0)  # (hidden_dim,)
        direction = direction / (direction.norm() + 1e-8)  # normalize

        # Contrastive loss: push pos toward direction, neg away
        # cos_sim with normalized direction = dot product / norm
        pos_proj = torch.nn.functional.cosine_similarity(h_pos, direction.unsqueeze(0), dim=-1)
        neg_proj = torch.nn.functional.cosine_similarity(h_neg, direction.unsqueeze(0), dim=-1)

        # We want pos_proj high (positive) and neg_proj low (negative)
        steering_loss = -pos_proj.mean() + neg_proj.mean()

        total_loss = (loss_main + steer_weight * steering_loss) / accum

    handle.remove()
    total_loss.backward()

    metrics["steer_loss"] = steering_loss.item()
    metrics["steer_pos_proj"] = pos_proj.mean().item()
    metrics["steer_neg_proj"] = neg_proj.mean().item()
    metrics["steer_separation"] = (pos_proj.mean() - neg_proj.mean()).item()

    return metrics


def _step_progressive_layer(
    model, input_ids, attention_mask, weights,
    amp_dtype, use_amp, neg_loss_clamp, accum,
    progressive_active_layers=None, **kwargs,
):
    """Progressive layer activation: delegates to layer-selective with a dynamic layer set.

    The active layer set is determined by the training schedule (set externally
    in the training loop based on current step). This is just a wrapper that
    routes to _step_layer_selective with the current active set.
    """
    if progressive_active_layers is None:
        progressive_active_layers = set(range(model.config.n_layer))

    return _step_layer_selective(
        model, input_ids, attention_mask, weights,
        amp_dtype, use_amp, neg_loss_clamp, accum,
        neg_grad_layer_set=progressive_active_layers,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Main training loop with method dispatch
# ---------------------------------------------------------------------------

def train_advanced(
    model: GPT2LMHeadModel,
    train_loader: DataLoader,
    eval_safe_loader: DataLoader,
    eval_unsafe_loader: DataLoader,
    train_cfg: TrainingConfig,
    exp_cfg: ExperimentConfig,
    train_ds=None,
    resume_from: Path | None = None,
) -> dict:
    """Training loop with advanced method dispatch.

    Identical to training.train() except the forward/backward step is delegated
    to method-specific functions.

    If `resume_from` is given, the checkpoint directory must contain
    optimizer.pt + trainer_state.pt (written by `save_training_checkpoint`).
    Model weights, optimizer state, NSPO buffer (if applicable), step/epoch
    counters, best-eval-loss, early-stop patience, and RNG state are all
    restored. Training resumes at the saved (epoch, batch-in-epoch).
    """
    method = train_cfg.training_method
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if train_cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print("Gradient checkpointing enabled (use_reentrant=False)")

    # torch.compile for 10-30% training speedup via kernel fusion.
    # Skips methods that use hooks (activation_steering), since torch.compile
    # doesn't handle dynamic graph mutations well. layer_selective stopped freezing
    # parameters on 2026-11-14 but has never been run compiled, so it stays off too.
    compilable = method not in ("activation_steering", "layer_selective")
    if compilable:
        try:
            model = torch.compile(model)
            print("torch.compile enabled")
        except Exception as e:
            print(f"torch.compile failed ({e}), continuing without")

    # Weight EMA shadows (optional, one per decay). Tracks the unwrapped module's
    # params so they're unaffected by torch.compile wrapping. Each saved at end as
    # ema_d{decay}/. Maintaining multiple decays in one run is ~free (a few lerps).
    _ema_str = getattr(train_cfg, "ema_decays", "") or ""
    ema_decays = [float(x) for x in _ema_str.split(",") if x.strip() and 0.0 < float(x) < 1.0]
    if ema_decays:
        ema_shadows = {d: {n: p.detach().clone() for n, p in _unwrap_model(model).named_parameters()}
                       for d in ema_decays}
        print(f"Weight EMA enabled for decays {ema_decays} (windows≈{[round(1/(1-d)) for d in ema_decays]} steps)")
    else:
        ema_shadows = None

    output_dir = make_output_dir(exp_cfg, train_cfg)
    log_file = output_dir / "training_log.jsonl"

    # Pre-compute method-specific data
    extra_kwargs = {}
    if method == "token_level" and train_ds is not None:
        print("Computing token specificity scores...")
        token_spec = compute_token_specificity(train_ds, model.config.vocab_size)
        extra_kwargs["token_specificity"] = token_spec.to(device)
    if method == "layer_selective":
        n_layers = model.config.n_layer
        if train_cfg.neg_grad_layers:
            layer_list = [int(x) for x in train_cfg.neg_grad_layers.split(",")]
            extra_kwargs["neg_grad_layer_set"] = set(layer_list)
            print(f"Layer-selective: neg gradients through layers {sorted(layer_list)} of {n_layers}")
        elif train_cfg.neg_grad_freeze_layers:
            freeze_list = [int(x) for x in train_cfg.neg_grad_freeze_layers.split(",")]
            active = set(range(n_layers)) - set(freeze_list)
            extra_kwargs["neg_grad_layer_set"] = active
            print(f"Layer-selective (knockout): neg gradients EXCLUDED from layers {sorted(freeze_list)} of {n_layers}")
            print(f"  Active layers: {sorted(active)}")
        else:
            extra_kwargs["neg_grad_top_k"] = train_cfg.neg_grad_top_k
            print(f"Layer-selective: neg gradients through top {train_cfg.neg_grad_top_k} of {n_layers} layers")
        # Per-layer alpha boost
        if train_cfg.boost_layers:
            boost_set = set(int(x) for x in train_cfg.boost_layers.split(","))
            extra_kwargs["boost_layers"] = boost_set
            extra_kwargs["boost_alpha"] = train_cfg.boost_alpha
            print(f"  Boosted layers {sorted(boost_set)}: alpha × {train_cfg.boost_alpha}")
    if method in ("adaptive", "adaptive_projection"):
        extra_kwargs["adapt_threshold"] = train_cfg.adapt_threshold
        print(f"Adaptive: CE threshold = {train_cfg.adapt_threshold}")
        # Per-layer thresholds
        if train_cfg.layer_thresholds:
            layer_thresh = {}
            for pair in train_cfg.layer_thresholds.split(","):
                layer_idx, thresh = pair.split(":")
                layer_thresh[int(layer_idx)] = float(thresh)
            extra_kwargs["layer_thresholds"] = layer_thresh
            print(f"  Per-layer thresholds: {layer_thresh} (others use {train_cfg.adapt_threshold})")
    if method == "margin":
        extra_kwargs["margin_target"] = train_cfg.margin_target
        print(f"Margin-based: target CE gap = {train_cfg.margin_target}")
    if method == "sparse_negative":
        extra_kwargs["sparse_k_pct"] = train_cfg.sparse_k_pct
        extra_kwargs["sparse_rescale_mode"] = getattr(train_cfg, "sparse_rescale_mode", "budget")
        extra_kwargs["neg_force_mult"] = getattr(train_cfg, "neg_force_mult", 1.0)
        _mask_ema = getattr(train_cfg, "sparse_mask_ema_decay", 0.0)
        extra_kwargs["mask_ema_decay"] = _mask_ema
        extra_kwargs["mask_ema_state"] = {"imp": None} if _mask_ema > 0.0 else None  # persistent across steps
        print(f"Sparse negative: top {train_cfg.sparse_k_pct}% of parameters receive neg gradients "
              f"(rescale={extra_kwargs['sparse_rescale_mode']}, force_mult={extra_kwargs['neg_force_mult']}, mask_ema={_mask_ema})")
    if method == "npo":
        extra_kwargs["npo_beta"] = train_cfg.npo_beta
        print(f"NPO: beta = {train_cfg.npo_beta}")
    if method in ("nspo", "adaptive_nspo"):
        # Real Niu et al. NSPO uses a rolling buffer of positive-batch gradients
        extra_kwargs["nspo_buffer"] = NSPOBuffer(max_size=16)
        if method == "adaptive_nspo":
            extra_kwargs["adapt_threshold"] = train_cfg.adapt_threshold
            print(f"Adaptive + real NSPO: threshold={train_cfg.adapt_threshold}, buffer size 16")
        else:
            print(f"Real NSPO (Niu et al.): buffer size 16")
    if method in ("nspo_rank1", "adaptive_nspo_rank1"):
        if method == "adaptive_nspo_rank1":
            extra_kwargs["adapt_threshold"] = train_cfg.adapt_threshold
            print(f"[legacy] adaptive + rank-1 GP projection, threshold={train_cfg.adapt_threshold}")
        else:
            print(f"[legacy] rank-1 GP unconditional (formerly mislabeled NSPO)")
    if method == "full_stack":
        extra_kwargs["adapt_threshold"] = train_cfg.adapt_threshold
        extra_kwargs["sparse_k_pct"] = train_cfg.sparse_k_pct
        print(f"Full stack: adaptive(τ={train_cfg.adapt_threshold}) + rank-1 GP + sparse({train_cfg.sparse_k_pct}%)")
    if method == "activation_steering":
        extra_kwargs["steer_layer"] = train_cfg.steer_layer
        extra_kwargs["steer_weight"] = train_cfg.steer_weight
        print(f"Activation steering: layer {train_cfg.steer_layer}, weight {train_cfg.steer_weight}")
    if method == "progressive_layer":
        n_layers = model.config.n_layer
        # Compute schedule: how many steps per layer activation
        total_micro = len(train_loader) * train_cfg.n_epochs
        steps_per_layer = total_micro // n_layers
        extra_kwargs["_progressive_schedule"] = steps_per_layer
        extra_kwargs["_progressive_n_layers"] = n_layers
        # Start with just layer 0
        extra_kwargs["progressive_active_layers"] = {0}
        print(f"Progressive layer: activating 1 layer every {steps_per_layer} steps (total {n_layers} layers)")
        print("  Starting with layer 0")

    # Select step function
    step_fns = {
        "standard": _step_standard,
        "grad_projection": _step_grad_projection,
        "layer_selective": _step_layer_selective,
        "token_level": _step_token_level,
        "adaptive": _step_adaptive,
        "adaptive_projection": _step_adaptive_projection,
        "phased": _step_standard,  # phased uses standard step, weights change per epoch
        "kl_uniform": _step_kl_uniform,
        "margin": _step_margin,
        "sparse_negative": _step_sparse_negative,
        "activation_steering": _step_activation_steering,
        "progressive_layer": _step_progressive_layer,
        "npo": _step_npo,
        # Real Niu et al. NSPO (accumulated null-space projection)
        "nspo": _step_nspo_real,
        "adaptive_nspo": _step_adaptive_nspo,
        # Legacy rank-1 GP projection (previously mislabeled as NSPO).
        # Kept to reproduce the pre-audit results exactly.
        "nspo_rank1": _step_gp_unconditional,
        "adaptive_nspo_rank1": _step_adaptive_nspo_rank1,
        "full_stack": _step_full_stack,
    }
    step_fn = step_fns[method]
    print(f"Training method: {method}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=train_cfg.weight_decay,
    )

    accum = train_cfg.grad_accum_steps

    # Two-pass methods use param.grad as scratch (separate positive/negative
    # backward passes) and then ASSIGN the combined gradient to param.grad at the
    # end of the step. That assignment overwrites anything already in .grad, so
    # they cannot accumulate across micro-steps the way single-pass methods do
    # (which scale loss by 1/accum and let .backward() sum into .grad). For
    # grad_accum_steps>1 we therefore give two-pass methods a clean .grad each
    # micro-step and accumulate their combined result into an external buffer,
    # installing the group mean into .grad at the accumulation boundary.
    # (Before 2026-06-10 the loop zeroed .grad before EVERY micro-step, so only
    # the last micro-batch of each group survived.)
    TWO_PASS_METHODS = {
        "grad_projection", "adaptive_projection",
        "nspo", "adaptive_nspo", "nspo_rank1", "adaptive_nspo_rank1",
        "full_stack", "sparse_negative",
        "layer_selective", "progressive_layer",
    }
    is_two_pass = method in TWO_PASS_METHODS
    grad_accum_buf = {} if (is_two_pass and accum > 1) else None
    if is_two_pass and accum > 1:
        print(f"  Two-pass method '{method}' with grad_accum_steps={accum}: "
              f"accumulating combined gradients into an external buffer "
              f"(effective batch = {train_cfg.batch_size * accum}).")

    total_micro_steps = len(train_loader) * train_cfg.n_epochs
    total_steps = total_micro_steps // accum
    if train_cfg.max_steps > 0 and train_cfg.max_steps < total_steps:
        print(f"Step cap active: max_steps={train_cfg.max_steps} (< {total_steps} epoch-budget); LR cosine annealed to the cap")
        total_steps = train_cfg.max_steps  # cap LR cosine so it anneals to min_lr by the hard step cap
    print(f"Total steps: {total_steps} ({len(train_loader)} batches/epoch x {train_cfg.n_epochs} epochs)")

    # Delayed alignment: compute threshold step and capability-phase weight strategy
    delay_threshold = int(total_micro_steps * train_cfg.delay_alignment_pct)
    alignment_started = (train_cfg.delay_alignment_pct == 0)  # True if no delay
    delay_mode = train_cfg.delay_mode if hasattr(train_cfg, "delay_mode") else "uniform"
    if train_cfg.delay_alignment_pct > 0:
        print(f"Delayed alignment: first {train_cfg.delay_alignment_pct*100:.0f}% "
              f"({delay_threshold} micro-steps) capability-only, mode={delay_mode}")

    # Mixed precision
    use_amp = train_cfg.bf16 and torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if train_cfg.bf16 else torch.float16

    global_step = 0
    best_eval_loss = float("inf")
    patience_counter = 0
    stopped_early = False
    all_metrics = []
    t_start = time.time()
    elapsed_sec_offset = 0.0  # elapsed wall-clock from pre-resume training

    has_negative = exp_cfg.weight_scheme == "negative"
    target_alpha = exp_cfg.negative_alpha if has_negative else 0.0

    micro_step = 0
    start_epoch = 0
    start_batch_in_epoch = 0
    resume_epoch_rng = None   # random state the interrupted epoch's shuffle was drawn from
    pending_rng = None        # checkpoint-time random state, restored once the skipped batches are passed

    # Graceful checkpoint-and-exit on SIGUSR1. Pause/resume idiom:
    #   kill -SIGUSR1 <pid>   (or systemctl --user kill --signal=SIGUSR1 <unit>)
    # The handler sets a flag the train loop checks after each optimizer step;
    # on next check it writes a checkpoint_<step>/ and exits 0. Resume normally
    # via --resume <that path>.
    _stop_requested = [False]
    def _handle_sigusr1(_signum, _frame):
        _stop_requested[0] = True
        print("  [SIGUSR1] graceful stop requested — will checkpoint at next step boundary", flush=True)
    try:
        signal.signal(signal.SIGUSR1, _handle_sigusr1)
    except (ValueError, OSError):
        # signal handlers can only be installed in main thread; skip silently if not
        pass

    if resume_from is not None:
        trainer_state = load_training_checkpoint(
            resume_from, model, optimizer, extra_kwargs.get("nspo_buffer")
        )
        global_step = trainer_state["global_step"]
        micro_step = trainer_state["micro_step"]
        start_epoch = trainer_state["epoch"]
        start_batch_in_epoch = trainer_state.get("batch_in_epoch", 0)
        best_eval_loss = trainer_state.get("best_eval_loss", float("inf"))
        patience_counter = trainer_state.get("patience_counter", 0)
        alignment_started = trainer_state.get("alignment_started", alignment_started)
        elapsed_sec_offset = trainer_state.get("elapsed_sec_cum", 0.0)
        resume_epoch_rng = trainer_state.get("rng_epoch_start")
        if resume_epoch_rng is not None:
            pending_rng = trainer_state.get("rng")
        else:
            print("[RESUME] WARNING: this checkpoint was written before 2026-11-14 and does not hold the epoch's "
                  "shuffle state; the rest of this epoch is reshuffled (some rows repeat, some are skipped).", flush=True)
        print(
            f"[RESUME] from {resume_from}: step={global_step}, epoch={start_epoch}, "
            f"batch_in_epoch={start_batch_in_epoch}, best_safe_loss={best_eval_loss:.4f}, "
            f"prior_elapsed_sec={elapsed_sec_offset:.0f}"
        )

    for epoch in range(start_epoch, train_cfg.n_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        # Phased: override weights during capability phase
        is_capability_phase = (method == "phased" and epoch < train_cfg.phase_switch_epoch)
        if is_capability_phase and epoch == 0:
            print(f"  [PHASED] Capability phase: epochs 0-{train_cfg.phase_switch_epoch - 1} (all weights = +1.0)")
        if method == "phased" and epoch == train_cfg.phase_switch_epoch:
            print(f"  [PHASED] Switching to alignment phase at epoch {epoch}")
            # Reset early stopping — don't compare against capability phase
            best_eval_loss = float("inf")
            patience_counter = 0

        # When resuming, skip batches we already processed this epoch. Shuffle
        # order isn't bit-exact preserved (we fast-forward rather than drive the
        # original sampler), but the perturbation is <1% of training examples.
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

            # Phased: set all weights positive during capability phase
            if is_capability_phase:
                weights = torch.ones_like(weights)

            # Delayed alignment: override weights during capability phase.
            # Three modes control the transition smoothness:
            #   "uniform"      — all weights 1.0 (sharp drop for safe, full flip for harmful)
            #   "avg_positive" — all weights ≈ mean positive score (smooth for safe, smaller flip)
            #   "clamp_zero"   — positive keep score, negative → 0 (zero safe-transition, skip harmful)
            if train_cfg.delay_alignment_pct > 0 and micro_step < delay_threshold:
                if delay_mode == "avg_positive":
                    # Use 0.2 as the typical positive-score mean (empirically ~0.24
                    # for our scored subset with shift=0.2). Matches typical safe
                    # example weight, so safe content has near-zero transition shock.
                    weights = torch.full_like(weights, 0.2)
                elif delay_mode == "clamp_zero":
                    # Keep positive scores as-is, zero out negatives (skip harmful).
                    # Zero transition shock for safe; harmful goes 0 → −score (gentler than 1→−score).
                    weights = weights.clamp(min=0.0)
                else:  # "uniform"
                    weights = torch.ones_like(weights)
                alignment_started = False
            elif train_cfg.delay_alignment_pct > 0 and not alignment_started:
                # Reset early stopping when alignment phase begins
                alignment_started = True
                best_eval_loss = float("inf")
                patience_counter = 0
                print(f"  [DELAY] Alignment phase started at micro_step {micro_step}. Early stopping reset.")

            # Alpha schedule (for non-phased, or after phased switch)
            if has_negative and (train_cfg.alpha_warmup_steps > 0 or train_cfg.alpha_decay):
                eff_alpha = get_effective_alpha(
                    global_step, target_alpha,
                    train_cfg.alpha_warmup_steps, total_steps, train_cfg.alpha_decay,
                )
                neg_mask = weights < 0
                if neg_mask.any() and target_alpha > 0:
                    weights = weights.clone()
                    weights[neg_mask] = weights[neg_mask] * (eff_alpha / target_alpha)
            else:
                eff_alpha = target_alpha

            # Update progressive layer schedule if needed
            if method == "progressive_layer":
                sched = extra_kwargs["_progressive_schedule"]
                n_lay = extra_kwargs["_progressive_n_layers"]
                n_active = min(n_lay, 1 + micro_step // max(sched, 1))
                extra_kwargs["progressive_active_layers"] = set(range(n_active))

            # Forward + backward (method-specific).
            # Two-pass methods need a clean .grad to use as scratch for their
            # separate pos/neg passes. Single-pass methods scale their loss by
            # 1/accum and rely on .backward() summing into .grad across the whole
            # accumulation group, so we must NOT zero between their micro-steps.
            if is_two_pass:
                optimizer.zero_grad(set_to_none=True)
            batch_metrics = step_fn(
                model, input_ids, attention_mask, weights,
                amp_dtype, use_amp, train_cfg.neg_loss_clamp, accum,
                **extra_kwargs,
            )

            # Two-pass external accumulation: stash this micro-step's combined
            # gradient (already a per-micro-batch mean), scaled by 1/accum so the
            # sum over the group is the group mean.
            # Every two-pass step must therefore leave an UNSCALED per-micro-batch mean
            # in .grad, including when it falls back to the standard step
            # (_two_pass_fallback); the 1/accum is applied here and nowhere else.
            if grad_accum_buf is not None:
                for n, p in model.named_parameters():
                    if p.grad is None:
                        continue
                    if n in grad_accum_buf:
                        grad_accum_buf[n].add_(p.grad, alpha=1.0 / accum)
                    else:
                        grad_accum_buf[n] = (p.grad / accum).detach().clone()

            micro_step += 1
            if micro_step % accum != 0:
                continue

            # Accumulation boundary: hand the buffered group-mean gradient to
            # .grad for the optimizer (two-pass only; single-pass already has it).
            if grad_accum_buf is not None:
                for n, p in model.named_parameters():
                    if n in grad_accum_buf:
                        p.grad = grad_accum_buf[n]
                grad_accum_buf = {}  # fresh buffer for the next group

            # LR schedule
            lr = get_lr(global_step, train_cfg.warmup_steps, total_steps,
                        train_cfg.learning_rate, train_cfg.min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Weight EMA: low-pass filter(s) over the (noisy) weight trajectory.
            if ema_shadows is not None:
                with torch.no_grad():
                    named = list(_unwrap_model(model).named_parameters())
                    for d, shadow in ema_shadows.items():
                        for n, p in named:
                            shadow[n].mul_(d).add_(p.detach(), alpha=1.0 - d)

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
                    "method": method,
                    **batch_metrics,
                }
                if has_negative:
                    log_entry["eff_alpha"] = round(eff_alpha, 6)
                all_metrics.append(log_entry)
                with open(log_file, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

                if global_step % (train_cfg.log_every * 4) == 0:
                    extra_str = ""
                    if "gp_dot" in batch_metrics:
                        extra_str += f" | dot={batch_metrics['gp_dot']:.2e}"
                    if "adp_mean_scale" in batch_metrics:
                        extra_str += f" | adp_s={batch_metrics['adp_mean_scale']:.3f}"
                    if "tl_active_pct" in batch_metrics:
                        extra_str += f" | tl_act={batch_metrics['tl_active_pct']:.1%}"
                    print(
                        f"  step {global_step}/{total_steps} | "
                        f"loss={batch_metrics['loss_total']:.4f} | "
                        f"grad={log_entry['grad_norm']:.3f} | "
                        f"lr={lr:.2e}{extra_str} | "
                        f"{elapsed:.0f}s"
                    )

            # Eval
            if global_step % train_cfg.eval_every == 0:
                eval_metrics = evaluate_during_training(
                    model, eval_safe_loader, eval_unsafe_loader, device, amp_dtype, use_amp,
                )
                eval_entry = {"step": global_step, "epoch": epoch, "type": "eval",
                              "method": method, **eval_metrics}
                with open(log_file, "a") as f:
                    f.write(json.dumps(eval_entry) + "\n")
                print(
                    f"  [EVAL step {global_step}] "
                    f"safe_ppl={eval_metrics['ppl_safe']:.2f} | "
                    f"unsafe_ppl={eval_metrics['ppl_unsafe']:.2f} | "
                    f"gap={eval_metrics['ppl_gap']:.3f}"
                )

                # Best model + early stopping
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

                if train_cfg.early_stopping_patience > 0 and patience_counter >= train_cfg.early_stopping_patience and alignment_started:
                    print(f"  [EARLY STOP] No improvement for {patience_counter} evals. "
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
                    "alignment_started": alignment_started,
                    "elapsed_sec_cum": elapsed_sec_offset + (time.time() - t_start),
                    "rng": _capture_rng_state(),
                    "rng_epoch_start": epoch_rng,
                    "seed": train_cfg.seed,
                    "method": method,
                }
                save_training_checkpoint(
                    ckpt_dir, model, optimizer,
                    extra_kwargs.get("nspo_buffer"), trainer_state,
                )
                print(f"  [CKPT step {global_step}] saved full state to {ckpt_dir.name}/")

            # Graceful stop: write a fresh checkpoint and exit cleanly.
            if _stop_requested[0]:
                ckpt_dir = output_dir / f"checkpoint_{global_step}"
                trainer_state = {
                    "global_step": global_step,
                    "micro_step": micro_step,
                    "epoch": epoch,
                    "batch_in_epoch": batch_idx + 1,
                    "best_eval_loss": best_eval_loss,
                    "patience_counter": patience_counter,
                    "alignment_started": alignment_started,
                    "elapsed_sec_cum": elapsed_sec_offset + (time.time() - t_start),
                    "rng": _capture_rng_state(),
                    "rng_epoch_start": epoch_rng,
                    "seed": train_cfg.seed,
                    "method": method,
                }
                save_training_checkpoint(
                    ckpt_dir, model, optimizer,
                    extra_kwargs.get("nspo_buffer"), trainer_state,
                )
                print(f"  [GRACEFUL STOP step {global_step}] checkpoint saved to {ckpt_dir.name}/ — exiting", flush=True)
                sys.exit(0)

            # Hard step cap (max_steps): stop before the full n_epochs budget. The
            # per-step eval/save above have already run for this step, so best_model
            # and the resumable checkpoint are current as of the cap.
            if train_cfg.max_steps > 0 and global_step >= train_cfg.max_steps:
                print(f"  [MAX STEPS] reached cap {train_cfg.max_steps} at step {global_step} — stopping", flush=True)
                stopped_early = True
                break

        if pending_rng is not None:   # the checkpoint was the epoch's last batch: nothing was left to run in it
            _restore_rng_state(pending_rng)
            pending_rng = None
        print(f"Epoch {epoch} done | avg loss = {epoch_loss / max(epoch_steps, 1):.4f}")
        if stopped_early:
            break

    # Save final model
    _unwrap_model(model).save_pretrained(output_dir / "final_model")

    # Save EMA models (one per decay). Swap each shadow into the module, save,
    # then restore the trained weights so the final eval below is unaffected.
    if ema_shadows is not None:
        base = _unwrap_model(model)
        backup = {n: p.detach().clone() for n, p in base.named_parameters()}
        for d, shadow in ema_shadows.items():
            with torch.no_grad():
                for n, p in base.named_parameters():
                    p.copy_(shadow[n])
            ema_dir = output_dir / f"ema_d{d}"
            base.save_pretrained(ema_dir)
            print(f"Saved EMA(d={d}) to {ema_dir}")
        with torch.no_grad():
            for n, p in base.named_parameters():
                p.copy_(backup[n])

    # Final eval
    final_metrics = evaluate_during_training(
        model, eval_safe_loader, eval_unsafe_loader, device, amp_dtype, use_amp,
    )
    total_time = elapsed_sec_offset + (time.time() - t_start)
    result = {
        "method": method,
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
    print(f"\nTraining complete [{method}] in {total_time:.0f}s{stop_msg}")
    print(f"  Final safe PPL: {final_metrics['ppl_safe']:.2f}")
    print(f"  Final unsafe PPL: {final_metrics['ppl_unsafe']:.2f}")
    print(f"  PPL gap: {final_metrics['ppl_gap']:.3f}")
    print(f"  Saved to: {output_dir}")

    return result
