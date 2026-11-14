"""Configuration dataclasses for the signed-weight pretraining experiment."""

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ModelConfig:
    n_layer: int = 8
    n_embd: int = 160
    n_head: int = 4
    n_positions: int = 256
    vocab_size: int = 50257
    resid_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    tie_word_embeddings: bool = True


# Pre-built model configs
MODEL_CONFIGS = {
    "10m": ModelConfig(),  # 10.6M params — fast iteration
    "50m": ModelConfig(    # 51.7M params — scaled-up experiment
        n_layer=12,
        n_embd=448,
        n_head=8,
        n_positions=512,
    ),
    "200m": ModelConfig(   # ~200M params — scaling experiment
        n_layer=24,
        n_embd=768,
        n_head=12,
        n_positions=1024,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    ),
}


@dataclass
class TrainingConfig:
    batch_size: int = 64
    grad_accum_steps: int = 1  # gradient accumulation steps (effective batch = batch_size * grad_accum_steps)
    learning_rate: float = 6e-4
    min_lr: float = 6e-5
    weight_decay: float = 0.1
    warmup_steps: int = 200
    n_epochs: int = 10
    # Hard cap on optimizer steps (0 = use n_epochs). Stops before the full epoch
    # budget and anneals the LR cosine to the cap. Useful for the sparse bloom,
    # which is installed well before the 5-epoch mark (~115k @200M) and otherwise
    # oscillates ~50k more steps that early-stop-on-improvement never catches.
    max_steps: int = 0
    grad_clip: float = 1.0
    lr_schedule: str = "cosine"
    bf16: bool = True
    gradient_checkpointing: bool = False
    seed: int = 42
    log_every: int = 50
    eval_every: int = 500
    save_every: int = 2000
    num_workers: int = 4

    # --- Stability features for negative weighting ---
    # Alpha warmup: ramp negative alpha from 0 to target over this many steps
    alpha_warmup_steps: int = 0
    # Alpha decay: if True, decay alpha back to 0 after peak (follows cosine schedule)
    alpha_decay: bool = False
    # Negative loss clamp: cap per-example CE for negative-weighted examples (0 = disabled)
    # Prevents runaway feedback loop when model gets "too good" at avoiding negative patterns
    neg_loss_clamp: float = 0.0
    # Early stopping: stop after N consecutive evals without safe PPL improvement (0 = disabled)
    early_stopping_patience: int = 0
    # Minimum relative improvement to reset patience counter
    early_stopping_threshold: float = 0.02

    # --- Advanced training methods ---
    # Valid values include: standard, grad_projection, layer_selective, token_level,
    # adaptive, phased, kl_uniform, margin, sparse_negative, activation_steering,
    # progressive_layer, npo, nspo (real Niu et al.), adaptive_nspo (real),
    # nspo_rank1 / adaptive_nspo_rank1 (legacy rank-1 GP — previously mislabeled NSPO),
    # full_stack.
    training_method: str = "standard"
    neg_grad_top_k: int = 4   # layer_selective: propagate neg gradients through top K layers only
    neg_grad_layers: str = ""  # layer_selective: explicit layer indices that GET neg gradients (e.g. "0,1,2")
    neg_grad_freeze_layers: str = ""  # layer_selective: explicit layer indices to EXCLUDE from neg gradients
    adapt_threshold: float = 5.0  # adaptive: CE threshold for scaling down negative weight
    phase_switch_epoch: int = 5   # phased: switch from positive-only to mixed weights at this epoch

    # --- Novel alignment methods ---
    margin_target: float = 2.0    # margin: target CE gap between positive and negative examples
    sparse_k_pct: float = 10.0   # sparse_negative: % of parameters that receive neg gradients
    sparse_rescale_mode: str = "budget"  # sparse_negative: "budget" = scale neg L2 -> pos L2 (concentrated, ~1/sqrt(K) per-param); "per_param" = scale to pos_norm*sqrt(K/100) so each active param matches a dense per-param push
    ema_decays: str = ""         # weight EMA: comma-separated decay constants, e.g. "0.99,0.999,0.9999" (empty/0 = off). One shadow maintained per decay from the SAME trajectory; each saved to ema_d{decay}/. window ≈ 1/(1-d) steps. Low d = local/safe/mild smoothing; high d = long window (more points) but risks the off-manifold failure coarse-SWA hit (drifting sparse active-set). The sweep finds the sweet spot.
    neg_force_mult: float = 1.0   # sparse_negative: multiply the target neg-gradient norm by this. budget target = mult*pos_norm; per_param target = mult*pos_norm*sqrt(K/100). >1 = boost total alignment force above the capability budget. With dense (K=100) this tests "broad strong pushes" — high gap from total force, while each param is consistently (not sparsely/drifting) perturbed; isolates per-param-MAGNITUDE vs SPARSITY-DRIFT as the cause of incoherence.
    sparse_mask_ema_decay: float = 0.0  # sparse_negative: 0 = pick top-K from instantaneous |g_neg| (active set drifts each step → jagged landscape, the variance source). >0 = pick top-K from a running EMA of |g_neg| (decay≈0.99) → stable "aligned" param set, smoother training. Stabilizer that fixes the drift at the SOURCE (vs EMA/SWA which average weights post-hoc and fail off-manifold).
    steer_layer: int = 0          # activation_steering: which layer to steer
    steer_weight: float = 0.1    # activation_steering: regularization weight for steering loss
    npo_beta: float = 0.1        # npo: temperature parameter for log-sigmoid loss
    delay_alignment_pct: float = 0.0  # delay alignment signal until this % of training (0=immediate, 0.2=20%)
    # How to weight examples during the delay capability phase:
    #   "uniform"     — all examples get weight 1.0 (original fix, causes transition shock)
    #   "avg_positive" — all examples get the mean positive-score weight (~0.2), smooth transition for safe content
    #   "clamp_zero"  — positive examples keep their score, negative examples get weight 0 (skipped)
    delay_mode: str = "uniform"

    # --- Per-layer alignment control ---
    boost_layers: str = ""        # layer indices to boost (e.g. "0" or "0,7")
    boost_alpha: float = 2.0     # alpha multiplier for boosted layers
    layer_thresholds: str = ""   # per-layer adaptive thresholds (e.g. "0:3.0,7:4.0" — unlisted use adapt_threshold)


@dataclass
class ExperimentConfig:
    name: str = "baseline"
    weight_scheme: str = "uniform"  # "uniform", "filtered", "negative"
    negative_alpha: float = 0.5  # magnitude of negative weight for unsafe examples

    # Model size tag — included in output dir to separate 10M vs 50M runs
    model_size: str = "10m"

    # Derived from name + seed at runtime
    output_dir: str = ""

    # Paths
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    models_dir: Path = PROJECT_ROOT / "models" / "signed_weight_exp"
    results_dir: Path = PROJECT_ROOT / "results" / "signed_weight_exp"

    # Dataset sizes
    owt_subset_size: int = 200_000  # number of OpenWebText examples to use

    # LLM-scored data
    use_scored_data: bool = False   # if True, train only on LLM-scored subset
    use_hybrid_data: bool = False   # if True, full corpus + LLM scores overlaid where available
    use_full_classifier_data: bool = False  # if True, use full classifier-scored 2.36M dataset (200M)
    use_ratio_data: bool = False    # if True, use pre-built ratio dataset
    use_gaussian_data: bool = False  # if True, use pre-built Gaussian-shaped dataset
    ratio_pct: int = 10             # percentage of negative examples (1,5,10,20,30,40,50)
    score_shift: float = 0.0       # shift all scores by this amount (clipped to [-1,1])

    def __post_init__(self):
        self.project_root = Path(self.project_root)
        self.data_dir = Path(self.data_dir)
        self.models_dir = Path(self.models_dir)
        self.results_dir = Path(self.results_dir)


def make_run_name(experiment: ExperimentConfig, training: TrainingConfig) -> str:
    """Generate a run name like 'negative_a0.5_seed42'.

    Includes stability config suffix when non-default values are used,
    so runs with different stability settings don't collide.

    The length of the run is in the name when it is not the default: `_e<epochs>` and `_ms<max_steps>`. Until
    2026-11-14 it was left out, so two runs that differed only in --epochs or --max-steps shared a folder and the
    second was skipped as already completed. Default-length runs keep the names they always had. A run of another
    length started before that date sits in a folder without the suffix: rename the folder to resume it.
    """
    if experiment.weight_scheme == "negative":
        base = f"negative_a{experiment.negative_alpha}_seed{training.seed}"
        # Add stability suffixes to distinguish configs
        suffixes = []
        if training.alpha_warmup_steps > 0:
            suffixes.append(f"w{training.alpha_warmup_steps}")
        if training.neg_loss_clamp > 0:
            suffixes.append(f"c{training.neg_loss_clamp}")
        if training.alpha_decay:
            suffixes.append("decay")
        if suffixes:
            base += "_" + "_".join(suffixes)
    else:
        base = f"{experiment.weight_scheme}_seed{training.seed}"

    # Add data source prefix
    if experiment.use_gaussian_data:
        base = f"gauss{experiment.ratio_pct}pct_" + base
    elif experiment.use_ratio_data:
        base = f"ratio{experiment.ratio_pct}pct_" + base
    elif experiment.use_full_classifier_data:
        base = "fullc_" + base
        if experiment.score_shift != 0.0:
            base += f"_shift{experiment.score_shift}"
    elif experiment.use_hybrid_data:
        base = "hybrid_" + base
        if experiment.score_shift != 0.0:
            base += f"_shift{experiment.score_shift}"
    elif experiment.use_scored_data:
        base = "scored_" + base
        if experiment.score_shift != 0.0:
            base += f"_shift{experiment.score_shift}"

    # Add training method prefix (if not standard)
    method = training.training_method
    if method != "standard":
        method_tags = {
            "grad_projection": "gp",
            "layer_selective": (
                (f"ls_L{training.neg_grad_layers.replace(',','_')}" if training.neg_grad_layers
                 else f"ls_noL{training.neg_grad_freeze_layers.replace(',','_')}" if training.neg_grad_freeze_layers
                 else f"ls{training.neg_grad_top_k}")
                + (f"_boost{training.boost_layers.replace(',','_')}x{training.boost_alpha}" if training.boost_layers else "")
            ),
            "token_level": "tl",
            "adaptive": f"adp_lt{training.layer_thresholds.replace(',','_').replace(':','')}" if training.layer_thresholds else "adp",
            "adaptive_projection": "adpgp",
            "phased": f"ph{training.phase_switch_epoch}",
            "kl_uniform": "klu",
            "margin": f"margin{training.margin_target}",
            "sparse_negative": f"sparse{training.sparse_k_pct}pct" + ("_pp" if training.sparse_rescale_mode == "per_param" else "") + (f"_f{training.neg_force_mult}" if getattr(training, "neg_force_mult", 1.0) != 1.0 else "") + (f"_me{training.sparse_mask_ema_decay}" if getattr(training, "sparse_mask_ema_decay", 0.0) > 0.0 else ""),
            "activation_steering": f"steer_L{training.steer_layer}_w{training.steer_weight}",
            "progressive_layer": "prog",
            "npo": f"npo_b{training.npo_beta}",
            # Real Niu et al. NSPO — new runs
            "nspo": "nspo",
            "adaptive_nspo": "adpnspo",
            # Legacy rank-1 GP (2026-04-11: previously mislabeled as NSPO).
            # The pre-fix trained directories have been renamed on disk with
            # _rank1 suffix to free up the nspo_* namespace for real NSPO.
            "nspo_rank1": "nspo_rank1",
            "adaptive_nspo_rank1": "adpnspo_rank1",
            "full_stack": f"full_s{training.sparse_k_pct}pct",
        }
        tag = method_tags.get(method, method)
        base = f"{tag}_{base}"

    # Add delay suffix if alignment is delayed
    if training.delay_alignment_pct > 0:
        delay_mode = training.delay_mode if hasattr(training, "delay_mode") else "uniform"
        mode_suffix = {"uniform": "", "avg_positive": "_avgp", "clamp_zero": "_clz"}
        base += f"_delay{int(training.delay_alignment_pct*100)}{mode_suffix.get(delay_mode, '')}"

    # Length of the run, when it is not the default (see the docstring)
    if training.n_epochs != TrainingConfig.n_epochs:
        base += f"_e{training.n_epochs}"
    if training.max_steps > 0:
        base += f"_ms{training.max_steps}"

    return base


def make_output_dir(experiment: ExperimentConfig, training: TrainingConfig) -> Path:
    """Create and return the output directory for a run.

    Uses model_size as a subdirectory so 10m and 50m runs don't collide.
    """
    run_name = make_run_name(experiment, training)
    out = experiment.models_dir / experiment.model_size / run_name
    out.mkdir(parents=True, exist_ok=True)
    return out
