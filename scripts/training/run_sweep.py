"""
Local training sweep — mirrors scripts/colab/train_colab_rev.ipynb.

Reads preprocessed .pt chunks produced by scripts/training/prepare_colab_subset.py,
expands FIXED + SWEEP into a list of experiments, and runs each via train_one_run()
with Weights & Biases logging. Run `wandb login` once before first use.
"""
import itertools
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader

root_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root_dir))

from lipid_gnn.dataset import MartiniDiskDataset
from lipid_gnn.lipid_graph import LIPID_COMP_DIM
from lipid_gnn.membrane_prop_gnn import MembranePropertyGNN
from lipid_gnn.plotting import plot_property_accuracies

device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_amp = device.type == 'cuda'


# ── Properties to predict ─────────────────────────────────────────────────────
# Available: 'lipid_packing', 'thickness', 'thickness_std',
#            'compressibility', 'persistence', 'diffusivity'
PROPERTIES = ['lipid_packing', 'thickness']

# ── Fixed hyperparameters (shared across all runs) ────────────────────────────
FIXED = {
    "epochs":      100,
    "batch_size":  4,
    "num_workers": 2,
}

# ── Sweep grid: every combination produces one run ────────────────────────────
# comp_mode: "gnn_only"      — message passing only
#            "gnn_plus_comp" — GNN output + lipid composition vector
#            "comp_only"     — composition vector through MLP only (ablation)
SWEEP = {
    "comp_mode":     ["gnn_only"],
    "hidden_dim":    [32, 64],
    "num_layers":    [2, 3],
    "learning_rate": [5e-4],
    "weight_decay":  [5e-3],
    "seed":          [0, 1, 2],
}

# ── Data ──────────────────────────────────────────────────────────────────────
# Override via CHUNKS_DIR env var (HPC: point at /work/<grp>/<user>/... or
# node-local /local/$SLURM_JOB_ID/... for fast I/O).
# Expects subdirectories: train/, val/, test/
PROCESSED_DIR = Path(os.environ.get(
    "CHUNKS_DIR",
    root_dir / 'colab_lipid_gnn_subset' / 'processed',
))


def train_one_run(cfg, scaler, train_dataset, val_dataset, test_dataset):
    """Train a single run defined by cfg and log all results to W&B."""
    seed       = cfg["seed"]
    properties = cfg["properties"]
    comp_mode  = cfg["comp_mode"]

    torch.manual_seed(seed)
    np.random.seed(seed)

    run_name = (
        f"{comp_mode}"
        f"_h{cfg['hidden_dim']}"
        f"_l{cfg['num_layers']}"
        f"_lr{cfg['learning_rate']:.0e}"
        f"_s{seed}"
    )
    wandb.init(
        project="lipid_gnn_" + "_".join(properties),
        name=run_name,
        config=cfg,
    )

    _loader_kw = dict(
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(cfg["num_workers"] > 0),
        prefetch_factor=(2 if cfg["num_workers"] > 0 else None),
    )
    train_loader = DataLoader(train_dataset, **_loader_kw)
    val_loader   = DataLoader(val_dataset,   **_loader_kw)
    test_loader  = DataLoader(test_dataset,  **_loader_kw)

    use_comp = comp_mode in ("gnn_plus_comp", "comp_only")
    comp_dim = LIPID_COMP_DIM if use_comp else 0

    model = MembranePropertyGNN(
        in_channels=4,
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        out_dim=len(properties),
        comp_dim=comp_dim,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    criterion  = torch.nn.MSELoss()
    amp_scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)

    s_mean  = torch.tensor(scaler.mean_,  dtype=torch.float, device=device)
    s_scale = torch.tensor(scaler.scale_, dtype=torch.float, device=device)

    def normalize(y):
        return (y - s_mean) / s_scale

    def forward(batch):
        comp_vec       = batch.comp_vec.to(device) if use_comp else None
        edge_attr_dict = batch.edge_attr_dict if hasattr(batch, 'edge_attr_dict') else None
        return model(
            batch.x_dict, batch.edge_index_dict, batch.batch_dict,
            edge_attr_dict, comp_vec=comp_vec,
        )

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_train_loss = 0.0
        prop_train_loss  = torch.zeros(len(properties), device=device)
        n_train          = 0

        for batch_idx, batch in enumerate(train_loader):
            batch  = batch.to(device)
            target = normalize(batch.y)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                out  = forward(batch)
                loss = criterion(out, target)
            amp_scaler.scale(loss).backward()
            amp_scaler.step(optimizer)
            amp_scaler.update()

            n                = batch.num_graphs
            total_train_loss += loss.item() * n
            prop_train_loss  += (torch.mean((out - target) ** 2, dim=0) * n).detach()
            n_train          += n

            if batch_idx % 10 == 0:
                wandb.log({"batch/loss": loss.item()})

        avg_train_loss = total_train_loss / n_train
        avg_prop_train = (prop_train_loss / n_train).cpu().numpy()

        model.eval()
        total_val_loss = 0.0
        prop_val_loss  = torch.zeros(len(properties), device=device)
        val_preds      = []
        val_targets    = []
        n_val          = 0

        with torch.no_grad():
            for batch in val_loader:
                batch  = batch.to(device)
                target = normalize(batch.y)
                out    = forward(batch)
                loss   = criterion(out, target)

                n               = batch.num_graphs
                total_val_loss += loss.item() * n
                prop_val_loss  += torch.mean((out - target) ** 2, dim=0) * n
                n_val          += n

                val_preds.append(out.cpu().numpy())
                val_targets.append(target.detach().cpu().numpy())

        avg_val_loss = total_val_loss / n_val
        avg_prop_val = (prop_val_loss / n_val).cpu().numpy()
        scheduler.step(avg_val_loss)

        val_preds   = np.concatenate(val_preds,   axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        r2_scores   = r2_score(val_targets, val_preds, multioutput="raw_values")

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Train MSE: {avg_train_loss:.4f} | Val MSE: {avg_val_loss:.4f}")

        metrics = {
            "epoch":            epoch,
            "train/loss_total": avg_train_loss,
            "val/loss_total":   avg_val_loss,
            "learning_rate":    optimizer.param_groups[0]["lr"],
        }
        for i, prop in enumerate(properties):
            metrics[f"train/loss_{prop}"] = avg_prop_train[i]
            metrics[f"val/loss_{prop}"]   = avg_prop_val[i]
            metrics[f"val/r2_{prop}"]     = float(r2_scores[i])
        wandb.log(metrics)

    # ── Final held-out test evaluation ────────────────────────────────────────
    model.eval()
    test_preds   = []
    test_targets = []

    with torch.no_grad():
        for batch in test_loader:
            batch  = batch.to(device)
            target = normalize(batch.y)
            out    = forward(batch)
            test_preds.append(out.cpu().numpy())
            test_targets.append(target.detach().cpu().numpy())

    test_preds   = np.concatenate(test_preds,   axis=0)
    test_targets = np.concatenate(test_targets, axis=0)
    final_mse    = float(np.mean((test_preds - test_targets) ** 2))

    fig = plot_property_accuracies(test_targets, test_preds, properties, final_mse)
    wandb.log({"test/accuracy_plot": wandb.Image(fig), "test/mse_total": final_mse})
    plt.close(fig)

    wandb.finish()


def _expand_sweep():
    keys = list(SWEEP.keys())
    return [
        {**FIXED, "properties": PROPERTIES, **dict(zip(keys, vals))}
        for vals in itertools.product(*SWEEP.values())
    ]


def _load_datasets_and_scaler():
    train_chunks = sorted((PROCESSED_DIR / 'train').glob('chunk_*.pt'))
    val_chunks   = sorted((PROCESSED_DIR / 'val').glob('chunk_*.pt'))
    test_chunks  = sorted((PROCESSED_DIR / 'test').glob('chunk_*.pt'))

    if not train_chunks:
        raise FileNotFoundError(
            f"No chunk_*.pt files found in {PROCESSED_DIR / 'train'}. "
            f"Run scripts/training/prepare_colab_subset.py first."
        )

    print(f"Train chunks : {len(train_chunks)}")
    print(f"Val chunks   : {len(val_chunks)}")
    print(f"Test chunks  : {len(test_chunks)}")

    all_train_y = []
    for chunk_file in train_chunks:
        graphs = torch.load(chunk_file, weights_only=False)
        all_train_y.extend(g.y for g in graphs)

    y_matrix = torch.cat(all_train_y, dim=0).numpy()
    scaler   = StandardScaler().fit(y_matrix)

    print(f"\nScaler fit on {len(all_train_y)} training graphs.")
    print(f"  means : {dict(zip(PROPERTIES, scaler.mean_.round(4)))}")
    print(f"  stds  : {dict(zip(PROPERTIES, scaler.scale_.round(4)))}")

    train_dataset = MartiniDiskDataset(train_chunks, shuffle=True)
    val_dataset   = MartiniDiskDataset(val_chunks,   shuffle=False)
    test_dataset  = MartiniDiskDataset(test_chunks,  shuffle=False)
    return train_dataset, val_dataset, test_dataset, scaler


if __name__ == "__main__":
    print(f"Device : {device}")
    print(f"Chunks : {PROCESSED_DIR}\n")

    experiments = _expand_sweep()
    keys = list(SWEEP.keys())
    print(f"Generated {len(experiments)} experiments:\n")
    for i, cfg in enumerate(experiments):
        vals_str = '  '.join(f"{k}={cfg[k]}" for k in keys)
        print(f"  [{i:>2}]  {vals_str}")

    train_dataset, val_dataset, test_dataset, scaler = _load_datasets_and_scaler()

    wandb.login()

    for i, cfg in enumerate(experiments):
        print(f"\n{'─' * 60}")
        print(f"Experiment {i + 1} / {len(experiments)}")
        print(f"{'─' * 60}")
        train_one_run(cfg, scaler, train_dataset, val_dataset, test_dataset)

    print("\nAll experiments complete.")
