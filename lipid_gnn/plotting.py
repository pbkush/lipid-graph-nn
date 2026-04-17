import matplotlib.pyplot as plt
import numpy as np
import math

def plot_property_accuracies(actuals, predictions, property_names, overall_mse, save_path=None):
    """
    Dynamically creates subplots for each predicted property and displays them in one multiplot.
    
    Args:
        actuals (np.ndarray): Actual values, shape (n_samples, n_properties)
        predictions (np.ndarray): Predicted values, shape (n_samples, n_properties)
        property_names (list): names of the properties for subplot titles
        overall_mse (float): Total Test MSE for the main title
        save_path (str, optional): Path to save the plot. If None, it calls plt.show().
    """
    n_props = len(property_names)
    if n_props == 0:
        print("No properties to plot.")
        return

    # Convert to numpy arrays if they aren't already
    actuals = np.array(actuals)
    predictions = np.array(predictions)
    
    # Handle single-property case where arrays might be 1D
    if n_props == 1 and actuals.ndim == 1:
        actuals = actuals.reshape(-1, 1)
        predictions = predictions.reshape(-1, 1)

    # Calculate grid size (e.g. max 3 columns)
    cols = min(n_props, 3)
    rows = math.ceil(n_props / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), squeeze=False)
    fig.suptitle(f"Membrane Property Predictions\nOverall Test MSE: {overall_mse:.4f}", fontsize=16, fontweight='bold')

    for i, prop_name in enumerate(property_names):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        
        y_true = actuals[:, i]
        y_pred = predictions[:, i]
        
        # Calculate individual MSE for this property
        prop_mse = np.mean((y_true - y_pred)**2)
        
        # Scatter Plot
        ax.scatter(y_true, y_pred, alpha=0.5, color='royalblue', edgecolors='k', s=40)
        
        # Identity Line (Perfect match)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], color='firebrick', linestyle='--', linewidth=2, label='Identity')
        
        ax.set_title(f"Property: {prop_name}\n(MSE: {prop_mse:.4f})", fontsize=12)
        ax.set_xlabel("Actual (Normalized)")
        ax.set_ylabel("Predicted (Normalized)")
        ax.grid(True, linestyle=':', alpha=0.6)
        if i == 0:
            ax.legend()

    # Hide unused subplots
    for i in range(n_props, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    
    return fig

if __name__ == "__main__":
    # Smoke test / Example
    p_names = ["Lipid Packing", "Area Per Lipid", "Thickness"]
    # Fake normalized data
    y_t = np.random.randn(50, 3)
    y_p = y_t + np.random.normal(0, 0.2, (50, 3))
    
    # overall mse
    mse = np.mean((y_t - y_p)**2)
    
    print("Generating demo plot...")
    plot_property_accuracies(y_t, y_p, p_names, mse, save_path="demo_multiplot.png")
    print("Demo plot saved to demo_multiplot.png")
