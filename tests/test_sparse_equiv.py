"""Equivalence check: the memory-optimized _step_sparse_negative must produce
gradients identical (to FP tolerance) to the original implementation.

Strategy: run BOTH on the same tiny model in eval mode with use_amp=False, so the
two internal forward/backward passes yield identical pos/neg gradients — the only
difference is the sparsify/combine math we changed. Compare resulting param.grad.
"""
from pathlib import Path
import sys
import torch
from torch.amp import autocast
from transformers import GPT2Config, GPT2LMHeadModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training_advanced import _step_sparse_negative, compute_weighted_loss, _step_standard


def _old_sparse_step(
    model, input_ids, attention_mask, weights, amp_dtype, use_amp, neg_loss_clamp, accum,
    sparse_k_pct=10.0, sparse_rescale_mode="budget", neg_force_mult=1.0,
    mask_ema_decay=0.0, mask_ema_state=None, **kwargs,
):
    """Original implementation, verbatim (pre-optimization), for A/B comparison."""
    pos_mask = weights >= 0
    neg_mask = weights < 0
    if not neg_mask.any():
        return _step_standard(model, input_ids, attention_mask, weights, amp_dtype, use_amp, neg_loss_clamp, accum)
    if pos_mask.any():
        with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            out_pos = model(input_ids=input_ids[pos_mask], attention_mask=attention_mask[pos_mask])
            logits_pos = out_pos.logits
            del out_pos
            loss_pos, met_pos = compute_weighted_loss(logits_pos, input_ids[pos_mask], attention_mask[pos_mask], weights[pos_mask], neg_loss_clamp)
            del logits_pos
        loss_pos.backward()
    else:
        met_pos = {}
    grad_params = [(n, p) for n, p in model.named_parameters() if p.grad is not None]
    pos_grads = {n: p.grad.clone() for n, p in grad_params}
    model.zero_grad()
    with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
        out_neg = model(input_ids=input_ids[neg_mask], attention_mask=attention_mask[neg_mask])
        logits_neg = out_neg.logits
        del out_neg
        loss_neg, met_neg = compute_weighted_loss(logits_neg, input_ids[neg_mask], attention_mask[neg_mask], weights[neg_mask], neg_loss_clamp)
        del logits_neg
    loss_neg.backward()
    neg_grad_flat = torch.cat([p.grad.flatten() for _, p in grad_params])
    total_params = neg_grad_flat.numel()
    k = max(1, int(total_params * sparse_k_pct / 100.0))
    abs_grads = neg_grad_flat.abs()
    if mask_ema_decay > 0.0 and mask_ema_state is not None:
        if mask_ema_state.get("imp") is None:
            mask_ema_state["imp"] = abs_grads.clone()
        else:
            mask_ema_state["imp"].mul_(mask_ema_decay).add_(abs_grads, alpha=1.0 - mask_ema_decay)
        score = mask_ema_state["imp"]
    else:
        score = abs_grads
    if k < total_params:
        threshold = torch.topk(score, k).values[-1]
        sparse_mask_flat = score >= threshold
    else:
        sparse_mask_flat = torch.ones_like(neg_grad_flat, dtype=torch.bool)
    n_active = sparse_mask_flat.sum().item()
    pos_grad_flat = torch.cat([pos_grads[n].flatten() for n, _ in grad_params])
    pos_norm = pos_grad_flat.norm().item()
    neg_grad_sparse_flat = neg_grad_flat * sparse_mask_flat.float()
    neg_sparse_norm = neg_grad_sparse_flat.norm().item()
    if sparse_rescale_mode == "per_param":
        target_norm = pos_norm * (n_active / max(total_params, 1)) ** 0.5
    else:
        target_norm = pos_norm
    target_norm *= neg_force_mult
    if neg_sparse_norm > 1e-12 and target_norm > 1e-12:
        neg_rescale = target_norm / neg_sparse_norm
    else:
        neg_rescale = 1.0
    n_pos = max(pos_mask.sum().item(), 1)
    n_neg = max(neg_mask.sum().item(), 1)
    n_total = float(n_pos + n_neg)
    offset = 0
    for name, param in grad_params:
        numel = param.numel()
        g_pos = pos_grads[name]
        g_neg = param.grad.clone()
        mask = sparse_mask_flat[offset:offset + numel].view(param.shape)
        g_neg_sparse = g_neg * mask.float() * neg_rescale
        param.grad = (n_pos / n_total) * g_pos + (n_neg / n_total) * g_neg_sparse
        offset += numel
    return {"sparse_n_active": n_active, "sparse_pos_norm": pos_norm,
            "sparse_neg_sparse_norm": neg_sparse_norm, "sparse_neg_rescale": neg_rescale}


def run(fn, model, ids, attn, w, mode, k_pct, force, ema):
    model.zero_grad(set_to_none=True)
    state = {"imp": None} if ema else None
    m = fn(model, ids, attn, w, torch.bfloat16, False, 8.0, 1,
           sparse_k_pct=k_pct, sparse_rescale_mode=mode, neg_force_mult=force,
           mask_ema_decay=(0.9 if ema else 0.0), mask_ema_state=state)
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}, m


def main():
    torch.manual_seed(0)
    cfg = GPT2Config(n_layer=3, n_head=4, n_embd=64, n_positions=64, vocab_size=257)
    model = GPT2LMHeadModel(cfg).cuda().eval()  # eval => no dropout => deterministic
    B, S = 10, 24
    ids = torch.randint(0, 257, (B, S), device="cuda")
    attn = torch.ones(B, S, device="cuda")
    w = torch.tensor([1.0] * 6 + [-1.0] * 4, device="cuda")  # 6 pos, 4 neg

    print(f"{'mode':10s} {'k%':>5s} {'force':>5s} {'ema':>4s} | {'allclose':>8s} {'max_abs':>11s} {'max_rel':>11s} | rescale new/old")
    all_pass = True
    for mode in ["budget", "per_param"]:
        for k_pct in [10.0, 100.0]:
            for force in [1.0, 2.0]:
                for ema in [False, True]:
                    g_new, m_new = run(_step_sparse_negative, model, ids, attn, w, mode, k_pct, force, ema)
                    g_old, m_old = run(_old_sparse_step, model, ids, attn, w, mode, k_pct, force, ema)
                    names = list(g_new)
                    max_abs = max((g_new[n] - g_old[n]).abs().max().item() for n in names)
                    max_rel = max(((g_new[n] - g_old[n]).abs() / (g_old[n].abs() + 1e-8)).max().item() for n in names)
                    ok = all(torch.allclose(g_new[n], g_old[n], rtol=1e-4, atol=1e-7) for n in names)
                    all_pass &= ok
                    print(f"{mode:10s} {k_pct:5.0f} {force:5.1f} {str(ema):>4s} | {str(ok):>8s} {max_abs:11.2e} {max_rel:11.2e} | "
                          f"{m_new['sparse_neg_rescale']:.6f}/{m_old['sparse_neg_rescale']:.6f}")
    print("\nRESULT:", "ALL EQUIVALENT ✓" if all_pass else "MISMATCH ✗")


if __name__ == "__main__":
    main()
