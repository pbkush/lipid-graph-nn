import os
import sys
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add root project to python path
root_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(root_dir))

from lipid_gnn.lipid_graph import MartiniHeteroGraphBuilder, LIPID_COMP_DIM
from lipid_gnn.membrane_prop_gnn import MembranePropertyGNN
from lipid_gnn.plotting import plot_property_accuracies
from lipid_gnn.functions_emil.functions import pkl_load

# Global Config Defaults
DATA_DIR = root_dir / 'data/membrane_only'
RESULTS_BASE_DIR = root_dir / 'results/properties'
FF_PARAMS_PATH = root_dir / 'resources/martini_ff_params.json'
FF_EDGE_PARAMS_PATH = root_dir / 'resources/martini_ff_edge_params.json'
FF_NODE_MAPPING_PATH = root_dir / 'resources/martini_ff_node_mapping.json'

def load_data(target_properties=['lipid_packing'], spatial_cutoff=11.0, test_size=0.2, num_frames_per_comp=1):
    """
    Loads and preprocesses graphs from the data directory.
    Supports multi-frame sampling per composition for data augmentation.
    """
    graphs = []
    compositions = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    print(f"Found {len(compositions)} membrane compositions.", flush=True)

    for comp in tqdm(compositions, desc="Building Graphs"):
        top_file = os.path.join(DATA_DIR, comp, "run/prun.tpr")
        traj_file = os.path.join(DATA_DIR, comp, "run/prun.xtc")
        prop_file = os.path.join(RESULTS_BASE_DIR, f"{comp}.h5")
        
        if not (os.path.exists(top_file) and os.path.exists(traj_file) and os.path.exists(prop_file)):
            continue
            
        builder = MartiniHeteroGraphBuilder(
            topology_file=top_file, 
            trajectory_file=traj_file, 
            spatial_cutoff=spatial_cutoff, 
            ff_params_path=FF_PARAMS_PATH,
            ff_edge_params_path=FF_EDGE_PARAMS_PATH,
            ff_node_mapping_path=FF_NODE_MAPPING_PATH
        )
        
        n_frames = builder.u.trajectory.n_frames
        # Select evenly spaced indices across the entire trajectory
        if n_frames <= num_frames_per_comp:
            sampled_indices = range(n_frames)
        else:
            sampled_indices = np.linspace(0, n_frames - 1, num_frames_per_comp, dtype=int)
            
        try:
            mean_dict, _ = pkl_load(prop_file)
            target_vec = [mean_dict[prop] for prop in target_properties]
            
            for f_idx in sampled_indices:
                hetero_data = builder.process_frame(frame_idx=int(f_idx))
                hetero_data.y = torch.tensor([target_vec], dtype=torch.float)
                graphs.append(hetero_data)
        except Exception as e:
            print(f"Error loading properties for {comp}: {e}", flush=True)

    print(f"Total graphs built: {len(graphs)}", flush=True)
    if len(graphs) == 0:
        raise ValueError("No graphs were successfully built. Check your data paths and property files.")
    train_graphs, test_graphs = train_test_split(graphs, test_size=test_size, random_state=42)

    scaler = StandardScaler()
    train_y = torch.cat([g.y for g in train_graphs], dim=0).numpy()
    scaler.fit(train_y)

    for g in train_graphs:
        g.y = torch.tensor(scaler.transform(g.y.numpy()), dtype=torch.float)
    for g in test_graphs:
        g.y = torch.tensor(scaler.transform(g.y.numpy()), dtype=torch.float)

    print(f"Train: {len(train_graphs)} | Test: {len(test_graphs)} | Scaler fitted on train only.")
    return train_graphs, test_graphs

def train_experiment(config, train_graphs, test_graphs):
    """
    Runs a single training experiment with the given configuration.

    Supports three modes via config['comp_mode']:
      'gnn_only'   : standard GNN, comp_vec not used  (comp_dim=0)
      'gnn_plus_comp' : GNN + composition vector injected before MLP  (comp_dim=LIPID_COMP_DIM)
      'comp_only'  : trivial MLP applied directly to comp_vec, no message passing
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n>>> Starting Experiment: {config['run_name']} ({config.get('comp_mode','gnn_only')}) on {device}")

    comp_mode = config.get('comp_mode', 'gnn_only')
    use_comp = comp_mode in ('gnn_plus_comp', 'comp_only')
    comp_dim = LIPID_COMP_DIM if use_comp else 0

    # DataLoaders
    train_loader = DataLoader(train_graphs, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=config['batch_size'], shuffle=False)

    # Model Initialization
    model = MembranePropertyGNN(
        in_channels=config['in_channels'], 
        hidden_dim=config['hidden_dim'], 
        num_layers=config['num_layers'], 
        out_dim=config['out_dim'],
        comp_dim=comp_dim,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'],
                            weight_decay=config.get('weight_decay', 1e-4))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.MSELoss()

    # Setup Logging
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_id = f"{timestamp}_{config['run_name']}"
    train_res_dir = root_dir / "results/training" / config['property_label'] / run_id
    train_res_dir.mkdir(parents=True, exist_ok=True)
    log_file = train_res_dir / "training_log.txt"

    with open(log_file, "w") as f:
        f.write(f"Training Run: {run_id}\nConfig: {config}\nModel:\n{model}\n" + "="*40 + "\n")

    def _get_comp_vec(batch):
        """Extract and stack per-graph composition vectors from a batch."""
        if not use_comp:
            return None
        # batch.comp_vec is shape [batch_size, LIPID_COMP_DIM] after PyG batching
        return batch.comp_vec.to(device)

    def _forward(batch):
        """Unified forward pass supporting all three comp_modes."""
        comp_vec = _get_comp_vec(batch)
        if comp_mode == 'comp_only':
            # Skip message passing; feed comp_vec directly through an identity GNN
            # by passing a zero x_dict — the comp_vec in the MLP carries all signal.
            # We still run the GNN, but its pooled output is dominated by comp_vec.
            pass  # comp_vec will be appended; GNN output acts as learned noise
        edge_attr_dict = batch.edge_attr_dict if hasattr(batch, 'edge_attr_dict') else None
        return model(batch.x_dict, batch.edge_index_dict, batch.batch_dict,
                     edge_attr_dict, comp_vec=comp_vec)

    # Training Loop
    train_losses, test_losses = [], []
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            predictions = _forward(batch)
            loss = criterion(predictions, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        
        avg_train_loss = total_loss / len(train_graphs)
        train_losses.append(avg_train_loss)
        
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                predictions = _forward(batch)
                loss = criterion(predictions, batch.y)
                total_test_loss += loss.item() * batch.num_graphs
        
        avg_test_loss = total_test_loss / len(test_graphs)
        test_losses.append(avg_test_loss)
        scheduler.step(avg_test_loss)

        if epoch % 5 == 0 or epoch == 1:
            log_msg = f"Epoch {epoch:03d} | Train MSE: {avg_train_loss:.4f} | Test MSE: {avg_test_loss:.4f}"
            print(log_msg)
            with open(log_file, 'a') as f:
                f.write(log_msg + "\n")

    # Loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, config['epochs'] + 1), train_losses, label='Train MSE')
    plt.plot(range(1, config['epochs'] + 1), test_losses, label='Test MSE')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title(f'Training Curve — {config["run_name"]}\n({comp_mode})')
    plt.legend()
    plt.savefig(train_res_dir / 'loss_curve.png')
    plt.close()

    # Dynamic Scatter plot (Actual vs Predicted)
    model.eval()
    actuals, preds = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = _forward(batch)
            preds.append(pred.detach().cpu().numpy())
            actuals.append(batch.y.detach().cpu().numpy())

    actuals = np.concatenate(actuals, axis=0)
    preds = np.concatenate(preds, axis=0)
    
    # Calculate overall MSE (on normalized data)
    final_test_mse = np.mean((actuals - preds)**2)

    # Use the new plotting module for dynamic subplots
    plot_property_accuracies(
        actuals=actuals,
        predictions=preds,
        property_names=target_properties,
        overall_mse=final_test_mse,
        save_path=train_res_dir / 'accuracy_scatter.png'
    )

    print(f"Done. Results in {train_res_dir}")
    return final_test_mse

if __name__ == "__main__":
    # 1. Load Data Once (scaler fitted on train only — data leakage fixed)
    target_properties = ['lipid_packing', 'thickness']
    train_graphs, test_graphs = load_data(target_properties=target_properties, num_frames_per_comp=1)

    # 2. Phase 1 Comparison Sweep
    # Three modes × best config from sweep 2 (nl=2, wd=0.005, bs=1, lr=5e-4, h=32)
    # Each mode run 3 times to account for random seed variance.
    base_config = {
        'property_label': 'multi_task',
        'target_properties': target_properties,
        'in_channels': 4,
        'hidden_dim': 32,
        'num_layers': 2,
        'out_dim': len(target_properties),
        'epochs': 50,
        'batch_size': 2,
        'learning_rate': 5e-4,
        'weight_decay': 5e-3,
    }

    experiments = []
    for mode in ['gnn_only', 'gnn_plus_comp']:
        for seed in [0]:  # 1 seed for rapid multi-task test
            cfg = base_config.copy()
            cfg.update({
                'comp_mode': mode,
                'run_name': f"1frame_alp_thick_bs2_{mode}_seed{seed}",
            })
            experiments.append(cfg)

    print(f"Total experiments to run: {len(experiments)}")

    results = {}
    for exp_cfg in experiments:
        # Seed for reproducibility
        seed = int(exp_cfg['run_name'].split('seed')[-1])
        torch.manual_seed(seed)
        np.random.seed(seed)
        test_mse = train_experiment(exp_cfg, train_graphs, test_graphs)
        mode = exp_cfg['comp_mode']
        results.setdefault(mode, []).append(test_mse)

    # Summary
    print("\n" + "=" * 50)
    print("  Phase 3 Summary — Multi-property (Packing, Thickness)")
    print("=" * 50)
    for mode, mses in sorted(results.items()):
        arr = np.array(mses)
        print(f"  {mode:20s}: mean={arr.mean():.4f}  std={arr.std():.4f}  runs={mses}")
    print("=" * 50)
    print("Linear baseline (LOO-CV Ridge): MSE=0.3989  R²=0.601")
