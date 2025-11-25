import sys
import os
import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from ecm_vegf import ECMTipCellModel

def run_simulation(gradient_type, output_dir, gradient_params=None, num_runs=1, save_plots=True):
    """
    Run a simulation with the specified gradient type and parameters
    
    Parameters:
    - gradient_type: Type of gradient ('linear', 'radial', 'uniform')
    - output_dir: Base directory to save results
    - gradient_params: Dictionary of parameters specific to gradient type
    - num_runs: Number of times to repeat the simulation
    - save_plots: Whether to save plot images (default: True)
    """
    result_files = []
    
    for run in range(num_runs):
        print(f"\nRunning {gradient_type} gradient simulation #{run+1}/{num_runs}")
        
        # Initialize grid parameters
        grid_size = (50, 50)  # Define ECM grid size
        
        # Multiple tip cells at various positions
        initial_tip_positions = [
            (15, 25),  # Left side, will move right
            (35, 25),  # Right side, will move left
            (25, 15),  # Bottom, will move up
            (25, 35),  # Top, will move down
            (20, 20),  # Bottom-left, will move up-right
            (30, 30)   # Top-right, will move down-left
        ]
        
        # Default gradient parameters if not provided
        if gradient_params is None:
            if gradient_type == 'linear':
                gradient_params = {'slope': 0.1, 'intercept': 0.2}
            elif gradient_type == 'radial':
                gradient_params = {'center_value': 0.8, 'decay_rate': 0.05}
            elif gradient_type == 'uniform':
                gradient_params = {'uniform_value': 0.5}
        
        # Create the simulation instance
        simulation = ECMTipCellModel(
            grid_size=grid_size,
            gradient_type=gradient_type,
            gradient_params=gradient_params,
            initial_tips=initial_tip_positions
        )
        
        num_of_tip_cells = len(initial_tip_positions)
        
        # Simulation parameters
        steps = 10
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Define file names for TSV output
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        tsv_file_name = f"{gradient_type}_simulation_data_run{run+1}_{current_date}.tsv"
        tsv_file_path = os.path.join(output_dir, tsv_file_name)
        
        # Function to calculate Euclidean displacement
        def calculate_displacement(initial_pos, final_pos):
            return np.sqrt(np.sum((np.array(final_pos) - np.array(initial_pos))**2))
        
        # Initialize anastomosis count for each cell
        anastomosis_count = {i: 0 for i in range(len(initial_tip_positions))}
        
        # Store initial positions for displacement calculation
        initial_positions = {idx: tuple(tip) for idx, tip in enumerate(simulation.tip_cells)}
        
        # Dictionary to store ECM, VEGF, and average values at each position for each cell
        cell_ecm_values = {idx: [] for idx in range(len(initial_tip_positions))}
        cell_vegf_values = {idx: [] for idx in range(len(initial_tip_positions))}
        cell_avg_positions = {idx: [] for idx in range(len(initial_tip_positions))}
        
        # Initialize anastomosis events list if not already present
        if not hasattr(simulation, 'anastomosis_events'):
            simulation.anastomosis_events = []
        
        # For each step in the simulation
        for step in range(steps):
            print(f"\nStep {step + 1}: Running simulation")
            
            if step % 2 == 0:
                # Even steps: Move tip cells
                print(f"Step {step + 1}: Movement phase")
                
                # Move tip cells and capture average positions
                avg_positions = simulation.move_tip_cell()
                
                # Store ECM, VEGF, and average values at each position
                for idx, (x, y) in enumerate(simulation.tip_cells):
                    if idx < len(simulation.active_tips) and simulation.active_tips[idx]:
                        ecm_val = simulation.ecm_grid[y, x]
                        vegf_val = simulation.vegf_grid[y, x]
                        
                        cell_ecm_values[idx].append((x, y, ecm_val))
                        cell_vegf_values[idx].append((x, y, vegf_val))
                        
                        # Store average position if available
                        if idx < len(avg_positions) and avg_positions[idx] is not None:
                            cell_avg_positions[idx].append(avg_positions[idx])
                        else:
                            cell_avg_positions[idx].append(None)
            else:
                # Odd steps: Check for anastomosis
                print(f"Step {step + 1}: Anastomosis check phase")
                before_count = len(simulation.anastomosis_events)
                simulation.check_anastomosis()
                after_count = len(simulation.anastomosis_events)
                
                # Count anastomosis events for each cell
                if after_count > before_count:
                    # For each new anastomosis event
                    for i in range(before_count, after_count):
                        tip1, tip2 = simulation.anastomosis_events[i]
                        # Find the cells involved
                        for idx, pos in enumerate(simulation.tip_cells):
                            if tuple(pos) == tip1 or tuple(pos) == tip2:
                                if idx in anastomosis_count:
                                    anastomosis_count[idx] += 1
            
            # Check if any active tips remain
            if hasattr(simulation, 'active_tips') and not any(simulation.active_tips):
                print("No active tips remaining. Simulation complete.")
                break
            
            # Generate plots at specific intervals if plotting is enabled
            if save_plots and step % 10 == 0:  # Update plot every 10 steps to reduce computational load
                plot_simulation_results(simulation, step, run+1, num_runs, gradient_type, output_dir)
        
        # Generate final plot if enabled
        if save_plots:
            plot_simulation_results(simulation, steps, run+1, num_runs, gradient_type, output_dir)
        
        # Calculate displacements after simulation
        displacements = {}
        for idx, initial_pos in list(initial_positions.items()):
            if idx < len(simulation.tip_cells):
                final_pos = tuple(simulation.tip_cells[idx])
                displacement = calculate_displacement(initial_pos, final_pos)
                displacements[idx] = displacement
            else:
                displacements[idx] = None  # For cells that formed anastomosis
        
        # Write to TSV file
        with open(tsv_file_path, "w") as tsv_file:
            # Write header
            tsv_file.write("CellID\tPositionX\tPositionY\tECMValue\tVEGFValue\tAvg_Position\tDisplacement\tAnastomosisCount\n")
            
            # Write data for each cell
            for idx in range(len(initial_tip_positions)):
                displacement = displacements.get(idx, "N/A")
                anastomosis = anastomosis_count.get(idx, 0)
                
                # First write the cell's summary line
                tsv_file.write(f"{idx}\tN/A\tN/A\tN/A\tN/A\tN/A\t{displacement:.2f}\t{anastomosis}\n")
                
                # Then write each position with ECM, VEGF, and average values
                for step_idx in range(len(cell_ecm_values.get(idx, []))):
                    x, y, ecm = cell_ecm_values[idx][step_idx] if step_idx < len(cell_ecm_values[idx]) else (0, 0, "N/A")
                    _, _, vegf = cell_vegf_values[idx][step_idx] if step_idx < len(cell_vegf_values[idx]) else (0, 0, "N/A")
                    # Get the average position if available
                    avg_pos = cell_avg_positions[idx][step_idx] if step_idx < len(cell_avg_positions[idx]) and cell_avg_positions[idx][step_idx] is not None else "N/A"
                    if avg_pos != "N/A":
                        avg_pos_str = f"{avg_pos[0]},{avg_pos[1]}"
                    else:
                        avg_pos_str = "N/A"
                    
                    tsv_file.write(f"{idx}\t{x}\t{y}\t{ecm:.4f}\t{vegf:.4f}\t{avg_pos_str}\tN/A\tN/A\n")
        
        print(f"{gradient_type.capitalize()} gradient simulation run #{run+1} completed. Results saved to {tsv_file_path}")
        
        result_files.append(tsv_file_path)
    
    return result_files

def get_gradient_specific_directory(gradient_type):
    """Return a specific directory path based on gradient type"""
    # Define the specific directories for each gradient type
    gradient_directories = {
        'linear': os.path.expanduser("~/Desktop/sproj'24-'25/ecmmodel(workingversion)/ECM_model/mergedmodel/lin_data"),
        'radial': os.path.expanduser("~/Desktop/sproj'24-'25/ecmmodel(workingversion)/ECM_model/mergedmodel/rad_data"),
        'uniform': os.path.expanduser("~/Desktop/sproj'24-'25/ecmmodel(workingversion)/ECM_model/mergedmodel/uni_data")
    }
    
    # Get the directory for the specified gradient type
    output_dir = gradient_directories.get(gradient_type)
    
    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir

def plot_simulation_results(simulation, step, run, num_runs, gradient_type, output_dir):
    """
    Create improved plots for ECM-VEGF tip cell movement simulation
    
    Parameters:
    - simulation: ECMTipCellModel instance
    - step: Current simulation step
    - run: Current run number
    - num_runs: Total number of runs
    - gradient_type: Type of gradient being simulated
    - output_dir: Directory to save output files
    """
    # Create date string for filenames
    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create separate figures for different visualizations
    
    # Figure 1: ECM and VEGF comparison with tip cell positions
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot ECM gradient with a viridis colormap
    ecm_plot = ax1.imshow(simulation.ecm_grid, cmap="cividis", origin="lower")
    ax1.set_title("(A) ECM Density", fontsize=16)
    ax1.set_xlabel("X Position", fontsize=12)
    ax1.set_ylabel("Y Position", fontsize=12)
    fig1.colorbar(ecm_plot, ax=ax1, label="ECM Density Value")
    
    # Plot VEGF gradient with a plasma colormap (distinct from ECM)
    vegf_plot = ax2.imshow(simulation.vegf_grid, cmap="cividis", origin="lower")
    ax2.set_title("(B) VEGF Concentration", fontsize=16)
    ax2.set_xlabel("X Position", fontsize=12)
    ax2.set_ylabel("Y Position", fontsize=12)
    fig1.colorbar(vegf_plot, ax=ax2, label="VEGF Concentration Value")
    
    # Draw tip cells on both plots with consistent colors and better markers
    cell_colors =["salmon", "lime", "plum", "red", "cyan","fuchsia"]
    
    # Draw tip cell paths with distinct colors for each cell
    for subplot in [ax1, ax2]:
        for tip_idx, path in enumerate(simulation.cell_paths.values()):
            path_array = np.array(path)
            if len(path_array) > 1:
                cell_color = cell_colors[tip_idx % len(cell_colors)]
                
                # Draw line to show the full path
                subplot.plot(path_array[:, 0], path_array[:, 1], 
                             color=cell_color, linewidth=1.5, alpha=0.6)
                
                # Add arrows to show direction
                if len(path_array) > 5:
                    # Show arrows only at intervals to avoid clutter
                    arrow_indices = np.arange(0, len(path_array)-1, max(1, len(path_array)//10))
                    for i in arrow_indices:
                        dx = path_array[i+1, 0] - path_array[i, 0]
                        dy = path_array[i+1, 1] - path_array[i, 1]
                        subplot.arrow(path_array[i, 0], path_array[i, 1], dx, dy,
                                     head_width=0.5, head_length=0.7, fc=cell_color, ec=cell_color)
    
    # Draw current tip cell positions
    for subplot in [ax1, ax2]:
        for i, (x, y) in enumerate(simulation.tip_cells):
            cell_color = cell_colors[i % len(cell_colors)]
            if i < len(simulation.active_tips) and simulation.active_tips[i]:
                subplot.scatter(x, y, color=cell_color, s=120, marker='o', edgecolor='white')
            else:
                subplot.scatter(x, y, color='gray', s=120, marker='x', linewidth=2)
    
    # Draw anastomosis events
    for subplot in [ax1, ax2]:
        for tip1, tip2 in simulation.anastomosis_events:
            subplot.plot([tip1[0], tip2[0]], [tip1[1], tip2[1]], 'w-', linewidth=2)
            subplot.scatter(tip1[0], tip1[1], color='white', s=80, marker='*')
            subplot.scatter(tip2[0], tip2[1], color='white', s=80, marker='*')
    
    # Add legend for tip cell identification
    legend_elements = []
    for i in range(min(len(simulation.tip_cells), len(cell_colors))):
        cell_color = cell_colors[i % len(cell_colors)]
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor=cell_color, markersize=10, 
                                         label=f'Cell {i}'))
    
    legend_elements.append(plt.Line2D([0], [0], marker='*', color='w', 
                                     markerfacecolor='white', markersize=10, 
                                     label='Anastomosis'))
    
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # Main title for the figure
    fig1.suptitle(f"{gradient_type.capitalize()} Gradient - Run {run}/{num_runs} - Step {step}", 
                 fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save Figure 1
    fig1_filename = f"{gradient_type}_ecm_vegf_comparison_run{run}_{current_date}.png"
    fig1_filepath = os.path.join(output_dir, fig1_filename)
    fig1.savefig(fig1_filepath, dpi=300, bbox_inches="tight")
    
    # Figure 2: Combined visualization (ECM+VEGF average)
    fig2, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate average of ECM and VEGF
    avg_grid = (simulation.ecm_grid + simulation.vegf_grid) / 2
    
    # Plot the average with a distinct colormap
    avg_plot = ax.imshow(avg_grid, cmap="cividis", origin="lower")
    fig2.colorbar(avg_plot, ax=ax, label="Average ECM-VEGF Value")
    
    # Set title and labels
    ax.set_title("Combined ECM-VEGF Environment", fontsize=16)
    ax.set_xlabel("X Position", fontsize=12)
    ax.set_ylabel("Y Position", fontsize=12)
    
    # Draw tip cell trajectories with distinct colors
    for tip_idx, path in enumerate(simulation.cell_paths.values()):
        path_array = np.array(path)
        if len(path_array) > 1:
            cell_color = cell_colors[tip_idx % len(cell_colors)]
            
            # Plot full trajectory
            ax.plot(path_array[:, 0], path_array[:, 1], color=cell_color, 
                   linewidth=1.5, alpha=0.6, label=f'Cell {tip_idx}' if tip_idx < 6 else "")
            
            # Add direction indicators (arrows)
            if len(path_array) > 5:
                # Show arrows only at intervals to avoid clutter
                arrow_indices = np.arange(0, len(path_array)-1, max(1, len(path_array)//10))
                for i in arrow_indices:
                    dx = path_array[i+1, 0] - path_array[i, 0]
                    dy = path_array[i+1, 1] - path_array[i, 1]
                    ax.arrow(path_array[i, 0], path_array[i, 1], dx, dy,
                            head_width=0.5, head_length=0.7, fc=cell_color, ec=cell_color)
    
    # Draw current tip cell positions
    for i, (x, y) in enumerate(simulation.tip_cells):
        cell_color = cell_colors[i % len(cell_colors)]
        if i < len(simulation.active_tips) and simulation.active_tips[i]:
            ax.scatter(x, y, color=cell_color, s=120, marker='o', edgecolor='white')
        else:
            ax.scatter(x, y, color='white', s=120, marker='x', linewidth=2)
    
    # Draw anastomosis events
    for tip1, tip2 in simulation.anastomosis_events:
        ax.plot([tip1[0], tip2[0]], [tip1[1], tip2[1]], 'w-', linewidth=2)
        ax.scatter(tip1[0], tip1[1], color='white', s=80, marker='*')
        ax.scatter(tip2[0], tip2[1], color='white', s=80, marker='*')
    
    # Add legend for cell identification
    ax.legend(loc='upper right')
    
    # Set main title
    fig2.suptitle(f"{gradient_type.capitalize()} Gradient - Run {run}/{num_runs} - Step {step}\nTip Cell Movement in Combined Environment", 
                 fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save Figure 2
    fig2_filename = f"{gradient_type}_combined_environment_run{run}_{current_date}.png"
    fig2_filepath = os.path.join(output_dir, fig2_filename)
    fig2.savefig(fig2_filepath, dpi=300, bbox_inches="tight")
    
    # Close both figures to free memory
    plt.close(fig1)
    plt.close(fig2)
    
    return [fig1_filepath, fig2_filepath]


def run_all_simulations(num_runs=20, save_plots=True):
    """Run simulations for all gradient types with default parameters"""
    # Define gradient types and their default parameters
    gradient_configs = [
        {
            'type': 'linear', 
            'params': {'slope': 0.1, 'intercept': 0.2}
        },
        {
            'type': 'radial', 
            'params': {'center_value': 0.8, 'decay_rate': 0.05}
        },
        {
            'type': 'uniform', 
            'params': {'uniform_value': 0.5}
        }
    ]
    
    results = {}
    
    # Run each simulation
    for config in gradient_configs:
        gradient_type = config['type']
        gradient_params = config['params']
        
        # Get the appropriate directory for this gradient type
        output_dir = get_gradient_specific_directory(gradient_type)
        
        print(f"\nRunning {gradient_type} gradient simulation for {num_runs} run(s)...")
        output_paths = run_simulation(gradient_type, output_dir, gradient_params, num_runs, save_plots)
        
        results[gradient_type] = output_paths
    
    # Print summary of all simulations
    print("\n=== All Simulations Completed ===")
    for gradient_type, paths in results.items():
        print(f"{gradient_type.capitalize()} gradient: {num_runs} runs completed")
        for i, path in enumerate(paths):
            if i < 3:  # Just show first few paths to avoid clutter
                print(f"- {path}")
            elif i == 3:
                print(f"- ... and {len(paths)-3} more files")
                break
    
    return results

def parse_arguments():
    """Parse command line arguments to select gradient type and parameters"""
    parser = argparse.ArgumentParser(description='Run ECM VEGF Tip Cell Model Simulation with different gradients')
    
    # Gradient type argument
    parser.add_argument('--gradient', type=str, choices=['linear', 'radial', 'uniform', 'all'], 
                        default='all', help='Type of gradient to use (default: all)')
    
    # Number of runs
    parser.add_argument('--num-runs', type=int, default=20,
                        help='Number of times to repeat the simulation (default: 20)')
    
    # Save plots option
    parser.add_argument('--save-plots', action='store_true', default=True,
                        help='Save plot images (default: True)')
    
    parser.add_argument('--no-save-plots', dest='save_plots', action='store_false',
                        help='Do not save plot images')
    
    # Linear gradient parameters
    parser.add_argument('--slope', type=float, default=0.1, 
                        help='Slope for linear gradient (default: 0.1)')
    parser.add_argument('--intercept', type=float, default=0.2, 
                        help='Intercept for linear gradient (default: 0.2)')
    
    # Radial gradient parameters
    parser.add_argument('--center-value', type=float, default=0.8, 
                        help='Center value for radial gradient (default: 0.8)')
    parser.add_argument('--decay-rate', type=float, default=0.05, 
                        help='Decay rate for radial gradient (default: 0.05)')
    
    # Uniform gradient parameter
    parser.add_argument('--uniform-value', type=float, default=0.5, 
                        help='Value for uniform gradient (default: 0.5)')
    
    # VEGF parameter options
    parser.add_argument('--vegf-seed', type=int, default=None,
                        help='Random seed for VEGF gradient generation (default: None)')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Check if command line arguments are provided
    if len(sys.argv) > 1:
        # Parse command line arguments
        args = parse_arguments()
        
        # Get number of runs and plot saving preference
        num_runs = args.num_runs
        save_plots = args.save_plots
        
        # Set gradient parameters based on type
        gradient_type = args.gradient
        
        if gradient_type == 'all':
            # Run all simulations with default parameters
            run_all_simulations(num_runs=num_runs, save_plots=save_plots)
        else:
            # Set gradient-specific parameters
            if gradient_type == 'linear':
                gradient_params = {
                    'slope': args.slope,
                    'intercept': args.intercept
                }
            elif gradient_type == 'radial':
                gradient_params = {
                    'center_value': args.center_value,
                    'decay_rate': args.decay_rate
                }
            else:  # uniform
                gradient_params = {
                    'uniform_value': args.uniform_value
                }
            
            # Get the appropriate directory for this gradient type
            output_dir = get_gradient_specific_directory(gradient_type)
            
            # Set random seed for VEGF if provided
            if args.vegf_seed is not None:
                np.random.seed(args.vegf_seed)
            
            # Run the selected simulation
            output_paths = run_simulation(gradient_type, output_dir, gradient_params, num_runs, save_plots)
            
            print(f"\nSimulation completed!")
            print(f"{num_runs} runs completed")
            for i, path in enumerate(output_paths):
                if i < 3:  # Just show first few paths to avoid clutter
                    print(f"- {path}")
                elif i == 3:
                    print(f"- ... and {len(output_paths)-3} more files")
                    break
    
    # No command line arguments - run all simulations with default settings (20 runs)
    else:
        print("Running all gradient types with 20 runs each...")
        run_all_simulations(num_runs=20, save_plots=True)