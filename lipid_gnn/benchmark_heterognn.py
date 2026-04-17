import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import HeteroData, Batch
from torch_geometric.nn import HeteroConv, SAGEConv, GATv2Conv
from lipid_gnn.membrane_prop_gnn import MembranePropertyGNN
from lipid_gnn.lipid_graph import MartiniHeteroGraphBuilder
import os
import sys
import time
import argparse
import traceback

# Auto-detect device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def print_environment_audit():
    print("=" * 60)
    print("ENVIRONMENT AUDIT")
    print("=" * 60)
    print(f"Python version:  {sys.version.split(' ')[0]}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyG version:     {torch_geometric.__version__}")
    print(f"Target Device:   {device}")
    
    if torch.cuda.is_available():
        print(f"CUDA version:    {torch.version.cuda}")
        print(f"GPU Hardware:    {torch.cuda.get_device_name(0)}")
        mem_prop = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Memory:      {mem_prop:.2f} GB")
    else:
        print("NOTE: CUDA is not available. Defaulting to CPU execution.")
        print("WARNING: CPU execution may be significantly slower, and CPU stress tests might cause a hard OS crash instead of throwing a Python exception.")
    print("=" * 60)
    print()

def generate_dummy_data(N):
    """
    Generate synthetic HeteroData with:
    - Nodes: N ('bead' type with a single integer feature x)
    - Edges: ~6*N ('bead', 'bonded', 'bead')
    - Edges: ~20*N ('bead', 'spatial', 'bead')
    - Target: Vector of 6 continuous properties
    """
    data = HeteroData()
    # Emulate the 4 continuous physical parameters (Mass, Charge, Sigma, Epsilon)
    data['bead'].x = torch.randn(N, 4, dtype=torch.float32)
    num_bonded = int(6 * N)
    num_spatial = int(20 * N)
    
    data['bead', 'bonded', 'bead'].edge_index = torch.randint(0, N, (2, num_bonded), dtype=torch.long)
    data['bead', 'bonded', 'bead'].edge_attr = torch.randn(num_bonded, 2, dtype=torch.float32)
    
    data['bead', 'spatial', 'bead'].edge_index = torch.randint(0, N, (2, num_spatial), dtype=torch.long)
    data['bead', 'spatial', 'bead'].edge_attr = torch.randn(num_spatial, 16, dtype=torch.float32)
    
    data.y = torch.randn(1, 1, dtype=torch.float32) # Standardized for lipid_packing regression
    
    batch = Batch.from_data_list([data])
    return batch

def load_real_data(system_name, data_dir="data/membrane_only", ff_dir="resources"):
    """
    Loads a real MD snapshot from the data directory and builds a HeteroData graph.
    """
    print(f"--- Loading Real System: {system_name} ---")
    system_path = os.path.join(data_dir, system_name)
    if not os.path.exists(system_path):
        raise FileNotFoundError(f"System directory not found: {system_path}")
        
    # Standard Martini file naming convention in this repo
    tpr_path = os.path.join(system_path, "run/prun.tpr")
    xtc_path = os.path.join(system_path, "run/prun.xtc")
    
    if not os.path.exists(tpr_path):
        # Fallback to searching for any .tpr in the directory
        import glob
        tprs = glob.glob(os.path.join(system_path, "**/*.tpr"), recursive=True)
        if tprs: tpr_path = tprs[0]
        else: raise FileNotFoundError(f"No .tpr file found in {system_path}")
        
    if not os.path.exists(xtc_path):
        import glob
        xtcs = glob.glob(os.path.join(system_path, "**/*.xtc"), recursive=True)
        if xtcs: xtc_path = xtcs[0]
        else:
            gros = glob.glob(os.path.join(system_path, "**/*.gro"), recursive=True)
            if gros: xtc_path = gros[0]
            else: raise FileNotFoundError(f"No trajectory (.xtc) or structure (.gro) found in {system_path}")

    ff_params = os.path.join(ff_dir, "martini_ff_params.json")
    ff_edge_params = os.path.join(ff_dir, "martini_ff_edge_params.json")
    ff_node_mapping = os.path.join(ff_dir, "martini_ff_node_mapping.json")

    builder = MartiniHeteroGraphBuilder(
        tpr_file=tpr_path,
        trajectory_file=xtc_path,
        spatial_cutoff=11.0,
        ff_params_path=ff_params,
        ff_edge_params_path=ff_edge_params,
        ff_node_mapping_path=ff_node_mapping
    )
    
    data = builder.process_frame(frame_idx=0)
    # Ensure y is present for metrics calculation
    data.y = torch.zeros((1, 1)) 
    
    batch = Batch.from_data_list([data])
    return batch

def calculate_graph_memory(graphs):
    """
    Calculates the total size in bytes of all tensors in a list of HeteroData graphs.
    """
    total_bytes = 0
    for g in graphs:
        # Each HeteroData has multiple storage objects (NodeStorage, EdgeStorage)
        for store in g.stores:
            for value in store.values():
                if torch.is_tensor(value):
                    total_bytes += value.element_size() * value.nelement()
    return total_bytes

def run_memory_scaling_test(args):
    """
    Measures the memory footprint of loading multiple frames and projects 
    the total memory needed to load the entire dataset.
    """
    print("=" * 60)
    print("MEMORY SCALING ANALYSIS (REAL SYSTEM)")
    print("=" * 60)
    
    # 1. Setup Builder
    system_path = os.path.join(args.data_dir, args.real_system)
    tpr_path = os.path.join(system_path, "run/prun.tpr")
    xtc_path = os.path.join(system_path, "run/prun.xtc")
    
    # ff_parser files are in resources
    builder = MartiniHeteroGraphBuilder(
        tpr_file=tpr_path,
        trajectory_file=xtc_path,
        spatial_cutoff=11.0,
        ff_params_path=os.path.join(args.ff_dir, "martini_ff_params.json"),
        ff_edge_params_path=os.path.join(args.ff_dir, "martini_ff_edge_params.json"),
        ff_node_mapping_path=os.path.join(args.ff_dir, "martini_ff_node_mapping.json")
    )
    
    max_frames_available = builder.u.trajectory.n_frames
    num_systems = len([d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))])
    
    print(f"Total Membrane Systems in data folder: {num_systems}")
    print(f"Max frames available for {args.real_system}: {max_frames_available}")
    print("-" * 60)
    print(f"{'Frames':<10} | {'Loaded':<10} | {'Mem/System (MB)':<18} | {'Projected Total (GB)':<20}")
    print("-" * 60)
    
    frame_counts = [1, 10, 100, 500]
    
    for n_target in frame_counts:
        n_actual = min(n_target, max_frames_available)
        
        graphs = []
        # Sample evenly if possible
        if n_actual > 0:
            indices = torch.linspace(0, max_frames_available - 1, n_actual).long().unique().tolist()
            n_actual = len(indices)
            
            for idx in indices:
                graphs.append(builder.process_frame(idx))
                
        bytes_usage = calculate_graph_memory(graphs)
        mem_mb = bytes_usage / (1024**2)
        projected_gb = (mem_mb * num_systems) / 1024
        
        note = "*" if n_actual < n_target else ""
        print(f"{n_target:<10} | {n_actual:<10}{note} | {mem_mb:<18.2f} | {projected_gb:<20.2f}")
        
    print("-" * 60)
    print("* Indicated value capped by available trajectory length.")
    print("=" * 60 + "\n")

class NativeTimer:
    """Helper context/class for standard CPU/GPU timing since torch.cuda.Event requires CUDA."""
    def __init__(self, is_cuda):
        self.is_cuda = is_cuda
        if self.is_cuda:
            self.start_evt = torch.cuda.Event(enable_timing=True)
            self.end_evt = torch.cuda.Event(enable_timing=True)
            
    def start(self):
        if self.is_cuda:
            torch.cuda.synchronize()
            self.start_evt.record()
        else:
            self.t1 = time.perf_counter()
            
    def stop(self):
        if self.is_cuda:
            self.end_evt.record()
            torch.cuda.synchronize()
        else:
            self.t2 = time.perf_counter()
            
    def elapsed_ms(self):
        if self.is_cuda:
            return self.start_evt.elapsed_time(self.end_evt)
        else:
            return (self.t2 - self.t1) * 1000.0

def profiling_and_timing(data_cpu, num_iters=50):
    print(f"--- Timing & Memory Profiling ---")
    num_nodes = data_cpu['bead'].num_nodes if hasattr(data_cpu['bead'], 'num_nodes') else data_cpu.num_nodes
    print(f"Processing Graph with {num_nodes} nodes.")
    is_cuda = device.type == 'cuda'
    
    # ---------------- 1. CPU -> Device Data Transfer ----------------
    timer = NativeTimer(is_cuda)
    timer.start()
    data = data_cpu.to(device, non_blocking=False)
    timer.stop()
    print(f"Data Transfer Time (CPU -> {device.type.upper()}): {timer.elapsed_ms():.2f} ms")
    
    # ---------------- 2. Setup & Warmup --------------------------
    model = MembranePropertyGNN(in_channels=4, hidden_dim=64, num_layers=3, out_dim=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4) # Matching training notebook
    
    model.train()
    for _ in range(10):  # Warmup
        optimizer.zero_grad()
        edge_attr_dict = data.edge_attr_dict if hasattr(data, 'edge_attr_dict') else None
        out = model(data.x_dict, data.edge_index_dict, data.batch_dict, edge_attr_dict)
        loss = F.mse_loss(out, data.y)
        loss.backward()
        optimizer.step()
        
    if is_cuda: torch.cuda.synchronize()
    
    # ---------------- 3. Time & Scaling Analysis -----------------
    fw_times = []
    bw_times = []
    
    for _ in range(num_iters):
        optimizer.zero_grad()
        
        # Forward Pass Runtime
        timer.start()
        edge_attr_dict = data.edge_attr_dict if hasattr(data, 'edge_attr_dict') else None
        out = model(data.x_dict, data.edge_index_dict, data.batch_dict, edge_attr_dict)
        timer.stop()
        fw_times.append(timer.elapsed_ms())
        
        loss = F.mse_loss(out, data.y)
        
        # Backward Pass Runtime 
        timer.start()
        loss.backward()
        timer.stop()
        bw_times.append(timer.elapsed_ms())
        
        optimizer.step()
        
    avg_fw = sum(fw_times) / num_iters
    avg_bw = sum(bw_times) / num_iters
    total_time = avg_fw + avg_bw
    throughput = 1000.0 / total_time  # graphs per second
    
    print(f"Average Forward Pass:  {avg_fw:.2f} ms")
    print(f"Average Backward Pass: {avg_bw:.2f} ms")
    print(f"Throughput:            {throughput:.2f} graphs / sec")
    
    # ---------------- 4. Memory Profiling ------------------------
    if is_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        edge_attr_dict = data.edge_attr_dict if hasattr(data, 'edge_attr_dict') else None
        out = model(data.x_dict, data.edge_index_dict, data.batch_dict, edge_attr_dict)
        fw_mem_alloc = torch.cuda.max_memory_allocated() / (1024**2)
        fw_mem_res = torch.cuda.max_memory_reserved() / (1024**2)
        
        loss = F.mse_loss(out, data.y)
        loss.backward()
        bw_mem_alloc = torch.cuda.max_memory_allocated() / (1024**2)
        bw_mem_res = torch.cuda.max_memory_reserved() / (1024**2)
        
        print(f"Peak GPU VRAM (Forward)  - Allocated: {fw_mem_alloc:.2f} MB | Reserved: {fw_mem_res:.2f} MB")
        print(f"Peak GPU VRAM (Backward) - Allocated: {bw_mem_alloc:.2f} MB | Reserved: {bw_mem_res:.2f} MB\n")
    else:
        print("Note: Peak Memory Profiling is skipped because PyTorch can only track GPU memory natively.\n")

def numerical_stability_test(data_cpu):
    print(f"--- Numerical Stability Test (AMP vs FP32) ---")
    data = data_cpu.to(device)
    model = MembranePropertyGNN(in_channels=4, hidden_dim=64, num_layers=3, out_dim=1).to(device)
    model.eval()
    is_cuda = device.type == 'cuda'

    timer_fp32 = NativeTimer(is_cuda)
    timer_amp  = NativeTimer(is_cuda)
    
    # On CPU, PyTorch supports bfloat16 for autocast. On GPU, float16.
    amp_dtype = torch.float16 if is_cuda else torch.bfloat16
    device_type = 'cuda' if is_cuda else 'cpu'

    # FP32
    with torch.no_grad():
        edge_attr_dict = data.edge_attr_dict if hasattr(data, 'edge_attr_dict') else None
        for _ in range(3): model(data.x_dict, data.edge_index_dict, data.batch_dict, edge_attr_dict) # Warmup
        if is_cuda: torch.cuda.synchronize()
        
        timer_fp32.start()
        edge_attr_dict = data.edge_attr_dict if hasattr(data, 'edge_attr_dict') else None
        out_fp32 = model(data.x_dict, data.edge_index_dict, data.batch_dict, edge_attr_dict)
        timer_fp32.stop()

    # AMP (Mixed Precision)
    with torch.no_grad():
        with torch.autocast(device_type=device_type, dtype=amp_dtype):
            edge_attr_dict = data.edge_attr_dict if hasattr(data, 'edge_attr_dict') else None
            for _ in range(3): model(data.x_dict, data.edge_index_dict, data.batch_dict, edge_attr_dict) # Warmup
            if is_cuda: torch.cuda.synchronize()
            
            timer_amp.start()
            edge_attr_dict = data.edge_attr_dict if hasattr(data, 'edge_attr_dict') else None
            out_amp = model(data.x_dict, data.edge_index_dict, data.batch_dict, edge_attr_dict)
            timer_amp.stop()
            
    diff = torch.abs(out_fp32 - out_amp)
    max_diff = torch.max(diff).item()
    var_diff = torch.var(diff).item()
    
    print(f"FP32 Forward Time: {timer_fp32.elapsed_ms():.2f} ms")
    print(f"AMP ({amp_dtype}) Forward Time: {timer_amp.elapsed_ms():.2f} ms")
    print(f"Speedup:           {timer_fp32.elapsed_ms()/timer_amp.elapsed_ms():.2f}x")
    print(f"Output Max Diff:   {max_diff:.6e}")
    print(f"Output Variance:   {var_diff:.6e}\n")

def stress_test(step=5000):
    print("--- Memory Stress Test (Finding Limits) ---")
    N = 500
    max_supported = 0
    is_cuda = device.type == 'cuda'
    
    while True:
        try:
            if is_cuda:
                torch.cuda.empty_cache()
            
            model = MembranePropertyGNN(in_channels=4, hidden_dim=64, num_layers=3, out_dim=1).to(device)
            data = generate_dummy_data(N).to(device)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
            model.train()
            optimizer.zero_grad()
            
            edge_attr_dict = data.edge_attr_dict if hasattr(data, 'edge_attr_dict') else None
            out = model(data.x_dict, data.edge_index_dict, data.batch_dict, edge_attr_dict)
            loss = F.mse_loss(out, data.y)
            loss.backward()
            optimizer.step()
            
            print(f"[SUCCESS] Trained with N = {N}")
            max_supported = N
            N += step
            
            del out, loss, data, model, optimizer
            
        except torch.cuda.OutOfMemoryError: # CUDA OOM
            print(f"[OOM] Hit CUDA Out of Memory at N = {N}")
            break
        except MemoryError: # Standard Python CPU OOM
            print(f"[OOM] Hit Host RAM Out of Memory at N = {N}")
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cannot allocate memory" in str(e).lower():
                print(f"[OOM] Hit PyTorch memory limit at N = {N}")
                break
            else:
                raise e
                
    print("*" * 40)
    print(f"MAXIMUM SUPPORTED NODES (N): {max_supported}")
    print("*" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HeteroGNN Benchmarking Script")
    parser.add_argument("--use-real", action="store_true", help="Benchmark using a real MD snapshot instead of synthetic data")
    parser.add_argument("--real-system", type=str, default="POPC100", help="Name of the real system (default: POPC100)")
    parser.add_argument("--data-dir", type=str, default="data/membrane_only", help="Base directory for real data")
    parser.add_argument("--ff-dir", type=str, default="resources", help="Directory for FF JSON mappings")
    
    parser.add_argument("--nodes", type=int, default=10000, help="Node count for synthetic data benchmarking")
    parser.add_argument("--iters", type=int, default=20, help="Number of iterations for profiling timing")
    parser.add_argument("--stress-step", type=int, default=5000, help="Node increment step size for stress test")
    parser.add_argument("--skip-stress", action="store_true", help="Skip the stress test phase")
    parser.add_argument("--mem-test", action="store_true", help="Perform the memory scaling analysis on real data")
    
    args = parser.parse_args()
    
    try:
        print("=" * 60)
        print("BENCHMARK PARAMETERS")
        print("=" * 60)
        for arg in vars(args):
            print(f"{arg:20}: {getattr(args, arg)}")
        print("=" * 60)
        print()
        
        print_environment_audit()
        
        # Select Data
        if args.use_real:
            data_cpu = load_real_data(args.real_system, args.data_dir, args.ff_dir)
        else:
            data_cpu = generate_dummy_data(args.nodes)
            
        profiling_and_timing(data_cpu, num_iters=args.iters)
        numerical_stability_test(data_cpu)
        
        if not args.skip_stress:
            stress_test(step=args.stress_step)
        else:
            print("--- Stress Test Skipped ---\n")
            
        if args.mem_test:
            run_memory_scaling_test(args)
            
    except Exception as e:
        print("An error occurred during benchmarking.")
        traceback.print_exc()
