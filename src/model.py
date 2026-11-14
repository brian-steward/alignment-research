"""Model creation for the signed-weight pretraining experiment."""

from transformers import GPT2Config, GPT2LMHeadModel

from .config import ModelConfig


def create_model(cfg: ModelConfig) -> GPT2LMHeadModel:
    """Create a randomly initialized GPT-2 model from config."""
    gpt2_config = GPT2Config(
        vocab_size=cfg.vocab_size,
        n_positions=cfg.n_positions,
        n_embd=cfg.n_embd,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        resid_pdrop=cfg.resid_pdrop,
        embd_pdrop=cfg.embd_pdrop,
        attn_pdrop=cfg.attn_pdrop,
        tie_word_embeddings=cfg.tie_word_embeddings,
        # Use GELU activation (GPT-2 default)
        activation_function="gelu_new",
        # MLP inner dim = 4 * n_embd (GPT-2 default)
        n_inner=None,
        # Use PyTorch native SDPA (scales to Flash Attention v2 / memory-efficient
        # attention under the hood on compatible GPUs). 2-3× speedup on attention.
        attn_implementation="sdpa",
    )
    model = GPT2LMHeadModel(gpt2_config)
    return model


def get_model_summary(model: GPT2LMHeadModel) -> dict:
    """Return param counts and layer structure."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Break down by component
    embedding_params = sum(
        p.numel() for n, p in model.named_parameters()
        if "wte" in n or "wpe" in n
    )
    transformer_params = sum(
        p.numel() for n, p in model.named_parameters()
        if "h." in n
    )
    ln_f_params = sum(
        p.numel() for n, p in model.named_parameters()
        if "ln_f" in n
    )

    return {
        "total_params": total,
        "trainable_params": trainable,
        "embedding_params": embedding_params,
        "transformer_params": transformer_params,
        "final_layernorm_params": ln_f_params,
        "n_layer": model.config.n_layer,
        "n_embd": model.config.n_embd,
        "n_head": model.config.n_head,
        "n_positions": model.config.n_positions,
    }
