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

from lipid_gnn.config import CONFIG
from lipid_gnn.lipid_graph import MartiniHeteroGraphBuilder
from lipid_gnn.membrane_prop_gnn import MembranePropertyGNN
from lipid_gnn.io import pkl_load

# Paths are sourced from the central config.
DATA_DIR             = CONFIG.paths.data_dir
RESULTS_BASE_DIR     = CONFIG.paths.props_dir
FF_PARAMS_PATH       = CONFIG.paths.ff_params_file
FF_EDGE_PARAMS_PATH  = CONFIG.paths.ff_edge_params_file
FF_NODE_MAPPING_PATH = CONFIG.paths.ff_node_mapping_file

def load_data(target_properties=['lipid_packing'], spatial_cutoff=None, test_size=0.2):
    if spatial_cutoff is None:
        spatial_cutoff = CONFIG.dataset.spatial_cutoff
    """
    Loads and preprocesses graphs from the data directory.

    KEY FIX: StandardScaler is now fitted on TRAIN labels only, not all data.
    This prevents data leakage where test-set statistics influence normalization.
    The test set is transformed using the train-fitted scaler.
    """
    graphs = []
    compositions = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    print(f"Found {len(compositions)} membrane compositions.")

    for comp in tqdm(compositions, desc="Building Graphs"):
        top_file = os.path.join(DATA_DIR, comp, CONFIG.paths.trajectory_subdir, CONFIG.paths.topology_filename)
        traj_file = os.path.join(DATA_DIR, comp, CONFIG.paths.trajectory_subdir, CONFIG.paths.trajectory_filename)
        prop_file = os.path.join(RESULTS_BASE_DIR, f"{comp}.h5")
        
        if not (os.path.exists(top_file) and os.path.exists(traj_file) and os.path.exists(prop_file)):
            continue
            
        builder = MartiniHeteroGraphBuilder(
            tpr_file=top_file,
            trajectory_file=traj_file,
            spatial_cutoff=spatial_cutoff, 
            ff_params_path=FF_PARAMS_PATH,
            ff_edge_params_path=FF_EDGE_PARAMS_PATH,
            ff_node_mapping_path=FF_NODE_MAPPING_PATH
        )
        hetero_data = builder.process_frame(frame_idx=0)
        
        try:
            mean_dict, _ = pkl_load(prop_file)
            target_vec = [mean_dict[prop] for prop in target_properties]
            hetero_data.y = torch.tensor([target_vec], dtype=torch.float)
            graphs.append(hetero_data)
        except Exception as e:
            print(f"Error loading properties for {comp}: {e}")

    # Split FIRST, then fit scaler on train only to prevent data leakage.
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
    """Runs a single training experiment with the given configuration."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n>>> Starting Experiment: {config['run_name']} on {device}")

    # DataLoaders
    train_loader = DataLoader(train_graphs, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=config['batch_size'], shuffle=False)

    # Model Initialization
    model = MembranePropertyGNN(
        in_channels=config['in_channels'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        out_dim=config['out_dim'],
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'],
                            weight_decay=config.get('weight_decay', 1e-4))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.MSELoss()

    # Setup Logging
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_id = f"{timestamp}_{config['run_name']}"
    train_res_dir = CONFIG.paths.training_results_dir / config['property_label'] / run_id
    train_res_dir.mkdir(parents=True, exist_ok=True)
    log_file = train_res_dir / "training_log.txt"

    with open(log_file, "w") as f:
        f.write(f"Training Run: {run_id}\nConfig: {config}\nModel:\n{model}\n" + "="*40 + "\n")

    def _forward(batch):
        edge_attr_dict = batch.edge_attr_dict if hasattr(batch, 'edge_attr_dict') else None
        return model(batch.x_dict, batch.edge_index_dict, batch.batch_dict,
                     edge_attr_dict)

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
    plt.title(f'Training Curve — {config["run_name"]}')
    plt.legend()
    plt.savefig(train_res_dir / 'loss_curve.png')
    plt.close()

    # Scatter plot (actual vs predicted)
    model.eval()
    actuals, predictions_list = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = _forward(batch)
            predictions_list.extend(pred.detach().cpu().view(-1).tolist())
            actuals.extend(batch.y.detach().cpu().view(-1).tolist())

    final_test_mse = np.mean((np.array(actuals) - np.array(predictions_list))**2)

    plt.figure(figsize=(6, 6))
    plt.scatter(actuals, predictions_list, alpha=0.6, color='blue', label='Predictions')
    if actuals:
        min_val = min(min(actuals), min(predictions_list))
        max_val = max(max(actuals), max(predictions_list))
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Accuracy')
    plt.xlabel('Actual lipid_packing (z-scored)')
    plt.ylabel('Predicted lipid_packing (z-scored)')
    plt.title(f'{config["run_name"]}\nTest MSE={final_test_mse:.4f}')
    plt.legend()
    plt.savefig(train_res_dir / 'accuracy_scatter.png')
    plt.close('all')

    print(f"Done. Results in {train_res_dir}")
    return final_test_mse

if __name__ == "__main__":
    # 1. Load Data Once (scaler fitted on train only — data leakage fixed)
    target_properties = ['lipid_packing']
    train_graphs, test_graphs = load_data(target_properties=target_properties)

    # 2. Single smoke-test run. Smoke-test overrides (small epochs/batch)
    # stay inline rather than coming from CONFIG.training.
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = {
        'property_label': 'lipid_packing',
        'target_properties': target_properties,
        'in_channels': CONFIG.model.in_channels,
        'hidden_dim': 32,
        'num_layers': 2,
        'out_dim': 1,
        'epochs': 1,
        'batch_size': 1,
        'learning_rate': 5e-4,
        'weight_decay': 5e-3,
        'run_name': f"smoke_seed{seed}",
    }
    test_mse = train_experiment(cfg, train_graphs, test_graphs)
    print(f"\nSmoke test Test MSE: {test_mse:.4f}")
    print("Linear baseline (LOO-CV Ridge): MSE=0.3989  R²=0.601")
