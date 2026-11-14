"""CLI entry point for the signed-weight pretraining experiment.

Usage:
    # Single run
    python -m src.run_experiment --scheme negative --alpha 0.5 --seed 42

    # Core experiment (3 schemes x 3 seeds = 9 runs)
    python -m src.run_experiment --suite core

    # Full suite with alpha ablations
    python -m src.run_experiment --suite full
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .config import ModelConfig, TrainingConfig, ExperimentConfig, MODEL_CONFIGS, make_output_dir, make_run_name
from .model import create_model, get_model_summary
from .dataset import prepare_datasets, prepare_scored_datasets, prepare_hybrid_datasets, create_dataloaders, get_tokenizer, load_beavertails
from .training import train
from .training_advanced import train_advanced
from .evaluation import run_full_evaluation


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_test_prompts(data_dir: Path, n_prompts: int = 100) -> list[str]:
    """Extract unsafe prompts from BeaverTails test set for generation eval."""
    bt_test = load_beavertails("test", data_dir / "beavertails")
    unsafe = [ex for ex in bt_test if not ex["is_safe"]]
    # Take a deterministic subset
    rng = random.Random(0)
    selected = rng.sample(unsafe, min(n_prompts, len(unsafe)))
    return [f"Human: {ex['prompt']}\nAssistant:" for ex in selected]


def run_single(
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    exp_cfg: ExperimentConfig,
    init_from: Path | None = None,
    resume_from: Path | None = None,
) -> dict:
    """Run a single experiment: train + evaluate.

    init_from: load model weights only from this HF model dir (best_model/,
        final_model/, or any checkpoint_N/). Optimizer/scheduler/NSPO buffer
        start fresh. Used for warm-starting.

    resume_from: full state resume from a checkpoint_N/ directory written by
        save_training_checkpoint (must include optimizer.pt + trainer_state.pt).
        Training continues at the saved (epoch, batch-in-epoch) with restored
        optimizer state, LR schedule position, NSPO buffer, and RNG state.
    """
    run_name = make_run_name(exp_cfg, train_cfg)
    output_dir = make_output_dir(exp_cfg, train_cfg)

    # Check if already completed. The training loop writes final_metrics.json when training ends, and the
    # evaluation at the end of this function rewrites it with an "eval" entry. A file without that entry is a
    # run whose evaluation never finished. Until 2026-11-14 such a run was skipped as completed; now its saved
    # model is evaluated, without training again.
    final_metrics_path = output_dir / "final_metrics.json"
    trained_earlier = saved_model_dir = None
    if final_metrics_path.exists() and init_from is None and resume_from is None:
        with open(final_metrics_path) as f:
            saved = json.load(f)
        if "eval" in saved:
            print(f"Run {run_name} already completed, skipping.")
            return saved
        saved_model_dir = next((d for d in (output_dir / "best_model", output_dir / "final_model") if d.exists()), None)
        if saved_model_dir is None:
            raise RuntimeError(
                f"Run {run_name}: final_metrics.json has no evaluation, and the folder has no best_model or "
                f"final_model to evaluate. Remove {final_metrics_path} to train the run again."
            )
        print(f"Run {run_name}: trained earlier, but its evaluation did not finish. Evaluating {saved_model_dir}.")
        set_seed(train_cfg.seed)
        trained_earlier = saved

    if trained_earlier is None:
        print(f"\n{'='*60}")
        print(f"Starting run: {run_name}")
        print(f"  Scheme: {exp_cfg.weight_scheme}, Alpha: {exp_cfg.negative_alpha}, Seed: {train_cfg.seed}")
        if init_from is not None:
            print(f"  [INIT-FROM] warm-start weights from {init_from}")
        if resume_from is not None:
            print(f"  [RESUME]    full-state resume from {resume_from}")
        print(f"{'='*60}\n")

        set_seed(train_cfg.seed)

        # Create model
        model = create_model(model_cfg)
        summary = get_model_summary(model)
        print(f"Model: {summary['total_params']:,} params")

        # Warm-start: copy weights from an existing HF model dir into our freshly
        # initialized model. Optimizer / NSPO buffer / step counter stay at zero.
        if init_from is not None:
            from transformers import GPT2LMHeadModel as _GPT2
            src_model = _GPT2.from_pretrained(init_from)
            missing, unexpected = model.load_state_dict(src_model.state_dict(), strict=False)
            del src_model
            print(f"  Loaded warm-start weights. "
                  f"missing={len(missing)}, unexpected={len(unexpected)}")

        # Save config
        config_dump = {
            "model": vars(model_cfg),
            "training": vars(train_cfg),
            "experiment": {k: str(v) if isinstance(v, Path) else v
                           for k, v in vars(exp_cfg).items()},
        }
        with open(output_dir / "config.json", "w") as f:
            json.dump(config_dump, f, indent=2)

    # Prepare data
    if exp_cfg.use_gaussian_data:
        from .dataset import prepare_gaussian_dataset
        train_ds, eval_safe_ds, eval_unsafe_ds = prepare_gaussian_dataset(
            exp_cfg, max_length=model_cfg.n_positions
        )
    elif exp_cfg.use_ratio_data:
        from .dataset import prepare_ratio_dataset
        train_ds, eval_safe_ds, eval_unsafe_ds = prepare_ratio_dataset(
            exp_cfg, max_length=model_cfg.n_positions
        )
    elif exp_cfg.use_full_classifier_data:
        from .dataset import prepare_full_classifier_dataset
        train_ds, eval_safe_ds, eval_unsafe_ds = prepare_full_classifier_dataset(
            exp_cfg, max_length=model_cfg.n_positions
        )
    elif exp_cfg.use_hybrid_data:
        train_ds, eval_safe_ds, eval_unsafe_ds = prepare_hybrid_datasets(
            exp_cfg, max_length=model_cfg.n_positions
        )
    elif exp_cfg.use_scored_data:
        train_ds, eval_safe_ds, eval_unsafe_ds = prepare_scored_datasets(
            exp_cfg, max_length=model_cfg.n_positions
        )
    else:
        train_ds, eval_safe_ds, eval_unsafe_ds = prepare_datasets(
            exp_cfg, max_length=model_cfg.n_positions
        )
    train_loader, eval_safe_loader, eval_unsafe_loader = create_dataloaders(
        train_ds, eval_safe_ds, eval_unsafe_ds,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
    )

    # Train — route through train_advanced when delay or other advanced features are active
    use_advanced = (train_cfg.training_method != "standard"
                    or train_cfg.delay_alignment_pct > 0)
    if trained_earlier is not None:
        train_metrics = trained_earlier
    elif use_advanced:
        train_metrics = train_advanced(
            model, train_loader, eval_safe_loader, eval_unsafe_loader,
            train_cfg, exp_cfg, train_ds=train_ds,
            resume_from=resume_from,
        )
    else:
        train_metrics = train(
            model, train_loader, eval_safe_loader, eval_unsafe_loader,
            train_cfg, exp_cfg,
            resume_from=resume_from,
        )

    # Full evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer()
    test_prompts = get_test_prompts(exp_cfg.data_dir)

    # Load best model for eval (for a run trained by an earlier call: the model that call saved)
    best_model_dir = output_dir / "best_model" if trained_earlier is None else saved_model_dir
    if best_model_dir.exists():
        from transformers import GPT2LMHeadModel as GPT2
        model = GPT2.from_pretrained(best_model_dir).to(device)
    else:
        model = model.to(device)

    eval_results = run_full_evaluation(
        model, tokenizer, eval_safe_loader, eval_unsafe_loader,
        test_prompts, device,
        output_path=output_dir / "eval_results.json",
    )

    # Combine all results
    all_results = {**train_metrics, "eval": eval_results}
    with open(output_dir / "final_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results


def run_suite(suite: str, model_cfg: ModelConfig, train_cfg: TrainingConfig,
              exp_cfg: ExperimentConfig):
    """Run a suite of experiments."""
    seeds = [42, 137, 256]

    if suite == "core":
        schemes = [
            ("baseline", "uniform", 0.0),
            ("filtered", "filtered", 0.0),
            ("negative", "negative", 0.5),
        ]
    elif suite == "full":
        schemes = [
            ("baseline", "uniform", 0.0),
            ("filtered", "filtered", 0.0),
            ("negative_a0.1", "negative", 0.1),
            ("negative_a0.3", "negative", 0.3),
            ("negative_a0.5", "negative", 0.5),
            ("negative_a0.7", "negative", 0.7),
            ("negative_a1.0", "negative", 1.0),
        ]
    else:
        raise ValueError(f"Unknown suite: {suite}")

    all_results = {}
    t_start = time.time()

    for name, scheme, alpha in schemes:
        for seed in seeds:
            train_cfg.seed = seed
            exp_cfg.name = name
            exp_cfg.weight_scheme = scheme
            exp_cfg.negative_alpha = alpha

            run_name = make_run_name(exp_cfg, train_cfg)
            result = run_single(model_cfg, train_cfg, exp_cfg)
            all_results[run_name] = result

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Suite '{suite}' complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Runs completed: {len(all_results)}")
    print(f"{'='*60}")

    # Save suite summary
    summary_path = exp_cfg.results_dir / f"suite_{suite}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Suite summary saved to {summary_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Signed-weight pretraining experiment")
    parser.add_argument("--scheme", type=str, default="uniform",
                        choices=["uniform", "filtered", "negative"],
                        help="Weighting scheme")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Negative weight magnitude (for scheme=negative)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Hard cap on optimizer steps (0 = use --epochs). Caps the LR cosine too. "
                             "Stops the sparse bloom's long oscillation tail (~115k @200M vs 169k full).")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model", type=str, default="10m",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model size preset")
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--suite", type=str, default=None,
                        choices=["core", "full"],
                        help="Run a full experiment suite instead of a single run")
    parser.add_argument("--owt-size", type=int, default=200_000,
                        help="Number of OpenWebText examples to use")
    # Stability options
    parser.add_argument("--alpha-warmup", type=int, default=0,
                        help="Steps to ramp alpha from 0 to target")
    parser.add_argument("--alpha-decay", action="store_true",
                        help="Cosine-decay alpha back to 0 after warmup")
    parser.add_argument("--neg-loss-clamp", type=float, default=0.0,
                        help="Cap CE for negative examples (0=disabled)")
    parser.add_argument("--early-stop", type=int, default=0,
                        help="Early stopping patience (0=disabled)")
    parser.add_argument("--eval-every", type=int, default=None,
                        help="Eval interval in steps (default: 500)")
    parser.add_argument("--log-every", type=int, default=None,
                        help="Logging interval in steps (default: 50)")
    # Scored data options
    parser.add_argument("--scored", action="store_true",
                        help="Train on LLM-scored subset with continuous weights")
    parser.add_argument("--hybrid", action="store_true",
                        help="Full corpus with LLM scores overlaid where available")
    parser.add_argument("--full-classifier", action="store_true",
                        help="Use the full classifier-scored 2.36M dataset (200M training)")
    parser.add_argument("--ratio-data", action="store_true",
                        help="Use pre-built ratio dataset (from build_ratio_datasets.py)")
    parser.add_argument("--gaussian-data", action="store_true",
                        help="Use pre-built Gaussian-shaped dataset (from build_gaussian_datasets.py)")
    parser.add_argument("--ratio-pct", type=int, default=10,
                        help="Negative example percentage for ratio/gaussian dataset")
    parser.add_argument("--score-shift", type=float, default=0.0,
                        help="Shift all LLM scores by this amount (e.g. 0.4)")
    # Advanced training methods
    parser.add_argument("--method", type=str, default="standard",
                        choices=["standard", "grad_projection", "layer_selective",
                                 "token_level", "adaptive", "adaptive_projection", "phased",
                                 "kl_uniform", "margin", "sparse_negative",
                                 "activation_steering", "progressive_layer",
                                 "npo",
                                 "nspo", "adaptive_nspo",          # real Niu et al. NSPO
                                 "nspo_rank1", "adaptive_nspo_rank1",  # legacy rank-1 GP
                                 "full_stack"],
                        help="Training method (default: standard)")
    parser.add_argument("--neg-grad-top-k", type=int, default=4,
                        help="Layer-selective: top K layers get neg gradients")
    parser.add_argument("--neg-grad-layers", type=str, default="",
                        help="Layer-selective: explicit layer indices that GET neg gradients (e.g. '0,1,2')")
    parser.add_argument("--neg-grad-freeze-layers", type=str, default="",
                        help="Layer-selective: layer indices to EXCLUDE from neg gradients (knockout)")
    parser.add_argument("--boost-layers", type=str, default="",
                        help="Layer indices to boost neg alpha (e.g. '0' or '0,7')")
    parser.add_argument("--boost-alpha", type=float, default=2.0,
                        help="Alpha multiplier for boosted layers (default 2.0)")
    parser.add_argument("--layer-thresholds", type=str, default="",
                        help="Per-layer adaptive CE thresholds (e.g. '0:3.0,7:4.0')")
    parser.add_argument("--margin-target", type=float, default=2.0,
                        help="Margin method: target CE gap between pos and neg examples")
    parser.add_argument("--sparse-k-pct", type=float, default=10.0,
                        help="Sparse negative: %% of parameters receiving neg gradients")
    parser.add_argument("--sparse-rescale-mode", choices=["budget", "per_param"], default="budget",
                        help="budget=scale neg L2 to pos L2 (concentrated push); per_param=scale to pos_norm*sqrt(K/100) so each active param gets a dense-equivalent push")
    parser.add_argument("--ema-decays", type=str, default="",
                        help="Comma-separated weight-EMA decays, e.g. '0.99,0.999,0.9999' (empty=off). One shadow per decay from the same run; each saved to ema_d{decay}/.")
    parser.add_argument("--neg-force-mult", type=float, default=1.0,
                        help="Multiply the sparse_negative target neg-grad norm (budget: mult*pos_norm). >1 boosts total alignment force; with --sparse-k-pct 100 tests broad strong pushes.")
    parser.add_argument("--sparse-mask-ema-decay", type=float, default=0.0,
                        help="sparse_negative: EMA decay for the top-K importance score (0=instantaneous|g_neg|, drifts; 0.99≈stable aligned-param set). Stabilizes the active set at the source.")
    parser.add_argument("--steer-layer", type=int, default=0,
                        help="Activation steering: target layer index")
    parser.add_argument("--steer-weight", type=float, default=0.1,
                        help="Activation steering: regularization weight")
    parser.add_argument("--npo-beta", type=float, default=0.1,
                        help="NPO: temperature parameter for log-sigmoid loss")
    parser.add_argument("--delay-alignment", type=float, default=0.0,
                        help="Delay alignment signal until this %% of training (e.g. 0.2 = 20%%)")
    parser.add_argument("--delay-mode", type=str, default="uniform",
                        choices=["uniform", "avg_positive", "clamp_zero"],
                        help="How to weight examples during delay capability phase")
    parser.add_argument("--adapt-threshold", type=float, default=5.0,
                        help="Adaptive: CE threshold for scaling down neg weight")
    parser.add_argument("--phase-switch-epoch", type=int, default=5,
                        help="Phased: epoch to switch from pos-only to mixed weights")
    parser.add_argument("--grad-checkpoint", action="store_true",
                        help="Enable gradient checkpointing (saves VRAM, ~25%% slower)")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="DataLoader workers (default 4; use 0 for 200M to avoid fork OOM)")
    parser.add_argument("--save-every", type=int, default=None,
                        help="Checkpoint save interval (default 2000; use 50000+ for 200M to save disk)")
    parser.add_argument("--init-from", type=str, default=None,
                        help="Warm-start: load model weights from this HF model dir "
                             "(e.g. .../best_model). Optimizer/NSPO buffer/step counter "
                             "start fresh.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Full-state resume from a checkpoint_N/ dir written by "
                             "save_training_checkpoint (must contain optimizer.pt + "
                             "trainer_state.pt).")

    args = parser.parse_args()
    if args.init_from is not None and args.resume is not None:
        parser.error("--init-from and --resume are mutually exclusive")
    init_from = Path(args.init_from) if args.init_from else None
    resume_from = Path(args.resume) if args.resume else None

    model_cfg = MODEL_CONFIGS[args.model]
    train_cfg = TrainingConfig(
        seed=args.seed,
        n_epochs=args.epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        alpha_warmup_steps=args.alpha_warmup,
        alpha_decay=args.alpha_decay,
        neg_loss_clamp=args.neg_loss_clamp,
        early_stopping_patience=args.early_stop,
        eval_every=args.eval_every or 500,
        log_every=args.log_every or 50,
        training_method=args.method,
        neg_grad_top_k=args.neg_grad_top_k,
        neg_grad_layers=args.neg_grad_layers,
        neg_grad_freeze_layers=args.neg_grad_freeze_layers,
        boost_layers=args.boost_layers,
        boost_alpha=args.boost_alpha,
        layer_thresholds=args.layer_thresholds,
        margin_target=args.margin_target,
        sparse_k_pct=args.sparse_k_pct,
        sparse_rescale_mode=args.sparse_rescale_mode,
        neg_force_mult=args.neg_force_mult,
        sparse_mask_ema_decay=args.sparse_mask_ema_decay,
        ema_decays=args.ema_decays,
        steer_layer=args.steer_layer,
        steer_weight=args.steer_weight,
        npo_beta=args.npo_beta,
        delay_alignment_pct=args.delay_alignment,
        delay_mode=args.delay_mode,
        adapt_threshold=args.adapt_threshold,
        phase_switch_epoch=args.phase_switch_epoch,
        gradient_checkpointing=args.grad_checkpoint,
        num_workers=args.num_workers if args.num_workers is not None else 4,
        save_every=args.save_every if args.save_every is not None else 2000,
    )
    exp_cfg = ExperimentConfig(
        weight_scheme=args.scheme,
        negative_alpha=args.alpha,
        owt_subset_size=args.owt_size,
        model_size=args.model,
        use_scored_data=args.scored,
        use_hybrid_data=args.hybrid,
        use_full_classifier_data=args.full_classifier,
        use_ratio_data=args.ratio_data,
        use_gaussian_data=args.gaussian_data,
        ratio_pct=args.ratio_pct,
        score_shift=args.score_shift,
    )

    if args.suite:
        if init_from is not None or resume_from is not None:
            parser.error("--init-from/--resume are not supported with --suite")
        run_suite(args.suite, model_cfg, train_cfg, exp_cfg)
    else:
        exp_cfg.name = f"{args.scheme}_a{args.alpha}" if args.scheme == "negative" else args.scheme
        run_single(model_cfg, train_cfg, exp_cfg,
                   init_from=init_from, resume_from=resume_from)


if __name__ == "__main__":
    main()
