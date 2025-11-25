import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

def create_diagnostic_gradients():
    """Create and visualize different radial gradients to diagnose the issue"""
    # Grid size as defined in the simulation
    grid_size = (50, 50)
    
    # Create a figure with multiple radial gradient configurations
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    axs = axs.flatten()
    
    # List of configurations to test
    configs = [
        {'title': 'Original (center_value=0.8, decay_rate=0.05)',
         'center_value': 0.8, 'decay_rate': 0.05, 'center': (25, 25)},
        {'title': 'Steeper Gradient (decay_rate=0.1)',
         'center_value': 0.8, 'decay_rate': 0.1, 'center': (25, 25)},
        {'title': 'Off-Center Gradient',
         'center_value': 0.8, 'decay_rate': 0.05, 'center': (15, 15)},
        {'title': 'Inverted Gradient (center_value=0.1)',
         'center_value': 0.1, 'decay_rate': 0.05, 'center': (25, 25)},
        {'title': 'Multiple Peaks',
         'multi_peak': True},
        {'title': 'Gradient + Noise',
         'center_value': 0.8, 'decay_rate': 0.05, 'center': (25, 25), 'noise': 0.1}
    ]
    
    # Generate and plot each gradient configuration
    for i, config in enumerate(axs):
        if i < len(configs):
            cfg = configs[i]
            if 'multi_peak' in cfg and cfg['multi_peak']:
                # Create a multi-peak gradient
                gradient = create_multi_peak_gradient(grid_size)
            else:
                # Create a single-peak radial gradient
                center = cfg.get('center', (25, 25))
                center_value = cfg.get('center_value', 0.8)
                decay_rate = cfg.get('decay_rate', 0.05)
                gradient = create_radial_gradient(grid_size, center, center_value, decay_rate)
                
                # Add noise if specified
                if 'noise' in cfg:
                    np.random.seed(42)  # For reproducibility
                    noise = np.random.normal(0, cfg['noise'], grid_size)
                    gradient = np.clip(gradient + noise, 0, 1)  # Keep values in [0,1]
            
            # Plot the gradient
            im = config.imshow(gradient, cmap="cividis", origin="lower")
            config.set_title(cfg['title'], fontsize=14)
            
            # Plot initial tip positions
            initial_tip_positions = [
                (15, 25), (35, 25), (25, 15), (25, 35), (20, 20), (30, 30)
            ]
            
            for pos in initial_tip_positions:
                config.scatter(pos[0], pos[1], color='red', s=100, marker='o', edgecolor='white')
                
                # Calculate and visualize gradient direction at this position
                if 'multi_peak' not in cfg or not cfg['multi_peak']:
                    show_gradient_direction(config, gradient, pos)
            
            # Add colorbar
            plt.colorbar(im, ax=config)
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Generate the current date string for filename
    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.expanduser("~/Desktop/sproj'24-'25/ecmmodel(workingversion)/ECM_model/mergedmodel/diagnostics")
    os.makedirs(output_dir, exist_ok=True)
    
    fig_path = os.path.join(output_dir, f"radial_gradient_diagnostics_{current_date}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Diagnostic plots saved to {fig_path}")
    return fig_path

def create_radial_gradient(grid_size, center=(25, 25), center_value=0.8, decay_rate=0.05):
    """Create a radial gradient with specified parameters"""
    x = np.arange(grid_size[0])
    y = np.arange(grid_size[1])
    X, Y = np.meshgrid(x, y)
    
    # Calculate distance from center
    dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    
    # Create radial gradient decreasing from center
    gradient = center_value * np.exp(-decay_rate * dist)
    
    return gradient

def create_multi_peak_gradient(grid_size):
    """Create a gradient with multiple peaks"""
    x = np.arange(grid_size[0])
    y = np.arange(grid_size[1])
    X, Y = np.meshgrid(x, y)
    
    # Define multiple centers
    centers = [(15, 15), (35, 35), (15, 35), (35, 15)]
    values = [0.8, 0.7, 0.6, 0.5]
    decay_rates = [0.08, 0.06, 0.07, 0.05]
    
    gradient = np.zeros(grid_size)
    for (cx, cy), val, rate in zip(centers, values, decay_rates):
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        gradient += val * np.exp(-rate * dist)
    
    # Normalize to [0, 1]
    gradient = gradient / gradient.max()
    
    return gradient

def show_gradient_direction(ax, gradient, position):
    """Calculate and visualize gradient direction at a specific position"""
    x, y = position
    
    # Calculate gradient direction using central differences
    if 0 < x < gradient.shape[1]-1 and 0 < y < gradient.shape[0]-1:
        dx = (gradient[y, x+1] - gradient[y, x-1]) / 2
        dy = (gradient[y+1, x] - gradient[y-1, x]) / 2
        
        # Normalize the direction vector
        magnitude = np.sqrt(dx**2 + dy**2)
        if magnitude > 1e-6:  # Avoid division by zero
            dx, dy = dx/magnitude, dy/magnitude
            
            # Draw an arrow showing gradient direction (scaled for visibility)
            scale = 3.0
            ax.arrow(x, y, dx*scale, dy*scale, head_width=0.6, 
                    head_length=0.8, fc='white', ec='white', width=0.2)

def analyze_gradient_movement(ecm_model_path='ecm_vegf.py'):
    """
    Analyze the movement calculation logic in the ECM model to diagnose issues
    
    This function doesn't actually run, it's just a template showing what 
    should be checked in the ECMTipCellModel class
    """
    # This is a pseudocode function to highlight what to check in the ECMTipCellModel
    
    # 1. Check how gradient values are calculated around tip cells
    # In move_tip_cell(), look for code that calculates neighboring positions
    
    # 2. Check the decision logic for selecting next position
    # Look for code that compares neighboring values and selects new position
    
    # 3. Check if there's a threshold that might be preventing movement
    # Look for minimum gradient thresholds that might stop movement if gradient is too shallow
    
    # Key areas to review:
    print("""
    Key areas to check in ECMTipCellModel:
    
    1. In move_tip_cell() method:
       - How are neighboring positions evaluated?
       - Is there a minimum threshold for gradient difference that triggers movement?
       - Are there special cases for different gradient types?
       
    2. In __init__() method:
       - How is the radial gradient actually created?
       - Are there parameters that might need adjustment?
       
    3. In reset() method (if exists):
       - Are all cell paths and positions properly reset between runs?
    """)

# Create diagnostic visualization
if __name__ == "__main__":
    fig_path = create_diagnostic_gradients()
    analyze_gradient_movement()

# Suggested solutions:
print("""
Potential Solutions for Radial Gradient Movement Issue:

1. Increase the decay_rate parameter (try 0.1 or 0.15) to create a steeper gradient
   - This creates stronger directional cues for the cells

2. Use an off-center gradient where the peak is not in the middle
   - This avoids cells near the center experiencing flat gradients

3. Add small random noise to the gradient to break symmetry
   - This prevents cells from getting stuck in perfectly symmetric environments
   
4. Try inverting the gradient (low in center, high at edges)
   - This might work better depending on if cells follow increasing or decreasing gradients

5. Check the tip cell movement algorithm:
   - Ensure it properly calculates gradients in radial environments
   - Add a minimum movement threshold to ensure cells always move slightly
   - Consider adding random movement when gradient differences are small
""")