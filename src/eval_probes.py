"""Linear probe evaluation: measure how well each layer separates safe from unsafe content.

For each layer in a trained model, trains a logistic regression classifier on hidden states
to predict safe/unsafe. The accuracy at each layer reveals where the model has learned to
internally represent safety — and how this differs across training methods.

Also supports probing across multiple models for comparison.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from transformers import GPT2LMHeadModel

from .config import MODEL_CONFIGS
from .model import create_model
from .dataset import load_beavertails, get_tokenizer


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results/probes"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def extract_hidden_states(model, input_ids, attention_mask, device, amp_dtype):
    """Extract hidden states from all layers for a batch.

    Returns dict: {layer_idx: tensor of shape (batch, hidden_dim)}
    Pools over sequence positions using mean of non-padding tokens.
    """
    hidden_states = {}

    # Register hooks on each layer
    hooks = []
    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            hidden_states[layer_idx] = h.detach()
        return hook_fn

    for i, layer in enumerate(model.transformer.h):
        hooks.append(layer.register_forward_hook(make_hook(i)))

    # Forward pass
    with torch.no_grad():
        with autocast("cuda", dtype=amp_dtype):
            model(input_ids=input_ids, attention_mask=attention_mask)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Pool: mean over non-padding tokens
    mask = attention_mask.unsqueeze(-1).float()  # (batch, seq, 1)
    pooled = {}
    for layer_idx, h in hidden_states.items():
        pooled[layer_idx] = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    return pooled  # {layer_idx: (batch, hidden_dim)}


def collect_representations(model, safe_loader, unsafe_loader, device, n_samples=2000):
    """Collect hidden state representations for safe and unsafe examples."""
    model.eval()
    amp_dtype = torch.bfloat16

    all_hidden = {i: [] for i in range(model.config.n_layer)}
    all_labels = []

    n_collected = 0
    for label, loader in [(1, safe_loader), (0, unsafe_loader)]:
        for batch in loader:
            if n_collected >= n_samples:
                break
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)

            pooled = extract_hidden_states(model, input_ids, attention_mask, device, amp_dtype)

            batch_size = input_ids.size(0)
            for layer_idx in pooled:
                all_hidden[layer_idx].append(pooled[layer_idx].float().cpu().numpy())
            all_labels.extend([label] * batch_size)
            n_collected += batch_size

        n_collected = 0  # reset for next loader

    # Concatenate
    labels = np.array(all_labels)
    hidden = {i: np.concatenate(arrs, axis=0) for i, arrs in all_hidden.items()}

    return hidden, labels


def train_probes(hidden, labels, test_size=0.3, seed=42):
    """Train logistic regression probe at each layer. Returns per-layer metrics."""
    from sklearn.model_selection import train_test_split

    results = {}
    n_layers = len(hidden)

    for layer_idx in range(n_layers):
        X = hidden[layer_idx]
        y = labels

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )

        # Standardize
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 1e-8
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        results[layer_idx] = {
            "accuracy": float(acc),
            "auc": float(auc),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }

    return results


def probe_model(model_dir, model_cfg_name="10m", n_samples=2000):
    """Run linear probes on a trained model."""
    model_dir = Path(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    best_model_dir = model_dir / "best_model"
    if best_model_dir.exists():
        model = GPT2LMHeadModel.from_pretrained(best_model_dir).to(device)
    else:
        final_model_dir = model_dir / "final_model"
        if final_model_dir.exists():
            model = GPT2LMHeadModel.from_pretrained(final_model_dir).to(device)
        else:
            print(f"No model found in {model_dir}")
            return None

    model.eval()

    # Load eval data
    from .dataset import create_dataloaders
    from .config import PROJECT_ROOT

    tokenizer = get_tokenizer()
    data_dir = PROJECT_ROOT / "data"
    bt_test = load_beavertails("test", data_dir / "beavertails")

    safe_examples = [ex for ex in bt_test if ex["is_safe"]]
    unsafe_examples = [ex for ex in bt_test if not ex["is_safe"]]

    # Tokenize
    max_length = MODEL_CONFIGS[model_cfg_name].n_positions

    def tokenize_examples(examples, max_len):
        texts = [f"{ex['prompt']}\n{ex.get('response', '')}" for ex in examples]
        encodings = tokenizer(texts, truncation=True, max_length=max_len,
                              padding="max_length", return_tensors="pt")
        # Create dummy weights
        weights = torch.ones(len(texts))
        dataset = torch.utils.data.TensorDataset(
            encodings["input_ids"], encodings["attention_mask"], weights
        )
        return dataset

    safe_ds = tokenize_examples(safe_examples[:n_samples], max_length)
    unsafe_ds = tokenize_examples(unsafe_examples[:n_samples], max_length)

    probe_batch = 16 if model.config.n_layer > 8 else 64  # smaller batch for larger models
    safe_loader = torch.utils.data.DataLoader(safe_ds, batch_size=probe_batch, shuffle=False)
    unsafe_loader = torch.utils.data.DataLoader(unsafe_ds, batch_size=probe_batch, shuffle=False)

    # Collect representations
    print(f"  Collecting hidden states ({n_samples} safe + {n_samples} unsafe)...")
    hidden, labels = collect_representations(model, safe_loader, unsafe_loader, device, n_samples)

    # Train probes
    print(f"  Training probes at {len(hidden)} layers...")
    probe_results = train_probes(hidden, labels)

    del model
    torch.cuda.empty_cache()

    return probe_results


def probe_multiple_models(model_dirs, names=None, model_cfg_name="10m", n_samples=2000):
    """Run probes on multiple models for comparison."""
    if names is None:
        names = [Path(d).name for d in model_dirs]

    all_results = {}
    for model_dir, name in zip(model_dirs, names):
        print(f"\nProbing: {name}")
        results = probe_model(model_dir, model_cfg_name, n_samples)
        if results:
            all_results[name] = results

    return all_results


def plot_probe_results(all_results, output_path=None):
    """Plot probe accuracy across layers for multiple models."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    colors = ['#4878CF', '#D65F5F', '#6ACC65', '#B47CC7', '#FFB347',
              '#888888', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

    # Accuracy plot
    ax = axes[0]
    for i, (name, results) in enumerate(all_results.items()):
        layers = sorted(results.keys())
        accs = [results[l]["accuracy"] for l in layers]
        ax.plot(layers, accs, 'o-', label=name, color=colors[i % len(colors)], linewidth=2)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Probe Accuracy (safe vs unsafe)")
    ax.set_title("Linear Probe: Safety Separability by Layer")
    ax.legend(fontsize=8)
    ax.set_ylim(0.5, 1.0)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)
    ax.grid(alpha=0.3)

    # AUC plot
    ax = axes[1]
    for i, (name, results) in enumerate(all_results.items()):
        layers = sorted(results.keys())
        aucs = [results[l]["auc"] for l in layers]
        ax.plot(layers, aucs, 'o-', label=name, color=colors[i % len(colors)], linewidth=2)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Probe AUC")
    ax.set_title("Linear Probe: AUC by Layer")
    ax.legend(fontsize=8)
    ax.set_ylim(0.5, 1.0)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)
    ax.grid(alpha=0.3)

    plt.suptitle("Safety Representation Quality Across Layers and Methods", fontsize=14, y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dirs", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", default=None)
    parser.add_argument("--model-cfg", default="10m")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--output", default=str(RESULTS_DIR / "probe_comparison.png"))
    args = parser.parse_args()

    results = probe_multiple_models(args.model_dirs, args.names, args.model_cfg, args.n_samples)

    # Save results
    with open(RESULTS_DIR / "probe_results.json", "w") as f:
        json.dump(results, f, indent=2)

    plot_probe_results(results, args.output)

    # Print summary
    print(f"\n{'='*60}")
    print("PROBE SUMMARY")
    print(f"{'='*60}")
    for name, res in results.items():
        layers = sorted(res.keys())
        best_layer = max(layers, key=lambda l: res[l]["accuracy"])
        print(f"{name}:")
        print(f"  Best layer: {best_layer} (acc={res[best_layer]['accuracy']:.3f}, auc={res[best_layer]['auc']:.3f})")
        print(f"  Layer 0: acc={res[0]['accuracy']:.3f}")
        print(f"  Final layer: acc={res[layers[-1]]['accuracy']:.3f}")
