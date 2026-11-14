"""Train DeBERTa-v3-large with sign-aware loss."""

from .train_scorer_v2 import train_improved

if __name__ == "__main__":
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    train_improved(
        model_name="microsoft/deberta-v3-large",
        epochs=10,
        batch_size=4,  # small batch for 435M model + cosmic-comp VRAM usage
        lr=3e-6,  # lower LR for larger model + smaller batch
        sign_weight=0.5,
    )
