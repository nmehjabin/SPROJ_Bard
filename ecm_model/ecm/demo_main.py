import sys
import os
import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from ecmmodel import ECMTipCellModel

import sys
import os
import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from ecmmodel import ECMTipCellModel

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
        steps = 200
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Define file names for TSV output
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        tsv_file_name = f"{gradient_type}_simulation_data_run{run+1}_{current_date}.tsv"
        
        # Create a subdirectory for this run's plots
        plot_dir = os.path.join(output_dir, f"{gradient_type}_run{run+1}_{current_date}")
        os.makedirs(plot_dir, exist_ok=True)
        
        tsv_file_path = os.path.join(output_dir, tsv_file_name)
        
        # Function to calculate Euclidean displacement
        def calculate_displacement(initial_pos, final_pos):
            return np.sqrt(np.sum((np.array(final_pos) - np.array(initial_pos))**2))
        
        # Initialize anastomosis count for each cell
        anastomosis_count = {i: 0 for i in range(len(initial_tip_positions))}
        
        # Store initial positions for displacement calculation
        initial_positions = {idx: tuple(tip) for idx, tip in enumerate(simulation.tip_cells)}
        
        # Dictionary to store ECM values at each position for each cell
        cell_ecm_values = {idx: [] for idx in range(len(initial_tip_positions))}
        
        # Initialize anastomosis events list if not already present
        if not hasattr(simulation, 'anastomosis_events'):
            simulation.anastomosis_events = []
        
        # For each step in the simulation
        for step in range(steps):
            print(f"\nStep {step + 1}: Running simulation")
            
            if step % 2 == 0:
                # Move tip cells
                print(f"Step {step + 1}: Movement phase")
                simulation.move_tip_cell()
                
                # Store ECM values at each position
                for idx, (x, y) in enumerate(simulation.tip_cells):
                    if idx < len(simulation.active_tips) and simulation.active_tips[idx]:
                        cell_ecm_values[idx].append((x, y, simulation.ecm_grid[y, x]))  # Note: grid access is [y, x]
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
            
            # Save plots every 10 steps
            if save_plots and step % 10 == 0:
                plt.figure(figsize=(12, 6))
                plt.imshow(simulation.ecm_grid, cmap="viridis", origin="lower")
                plt.colorbar(label="ECM Density")
                
                # Draw tip cell movement paths
                for tip, path in simulation.cell_paths.items():
                    path_array = np.array(path)
                    if len(path_array) > 1:  # Need at least 2 points for quiver
                        # Calculate movement vectors
                        dx = np.diff(path_array[:, 0])
                        dy = np.diff(path_array[:, 1])
                        
                        plt.quiver(
                            path_array[:-1, 0],  # x starts
                            path_array[:-1, 1],  # y starts
                            dx, dy,  # x and y movement vectors
                            angles='xy', 
                            scale_units='xy', 
                            scale=1, 
                            color='white'
                        )
                
                # Draw current tip cells
                for i, (x, y) in enumerate(simulation.tip_cells):
                    if i < len(simulation.active_tips) and simulation.active_tips[i]:
                        plt.scatter(x, y, color='red', s=100)
                    else:
                        plt.scatter(x, y, color='gray', s=100)
                
                # Draw anastomosis events
                for tip1, tip2 in simulation.anastomosis_events:
                    plt.plot([tip1[0], tip2[0]], [tip1[1], tip2[1]], 'g-', linewidth=2)
                    plt.scatter(tip1[0], tip1[1], color='black', s=80)
                    plt.scatter(tip2[0], tip2[1], color='black', s=80)
                
                plt.title(f"{gradient_type.capitalize()} Gradient - Run {run+1}/{num_runs} - Step {step} - ECM Density & Tip Cell Movements")
                
                # Save the plot for this step
                plot_file_name = f"step_{step:03d}.png"
                plot_file_path = os.path.join(plot_dir, plot_file_name)
                plt.savefig(plot_file_path, dpi=300, bbox_inches="tight")
                plt.close()
                
                print(f"Saved plot for step {step} to {plot_file_path}")
        
        # Save the final plot if it wasn't saved already
        if save_plots and steps % 10 != 0:
            plt.figure(figsize=(12, 6))
            plt.imshow(simulation.ecm_grid, cmap="viridis", origin="lower")
            plt.colorbar(label="ECM Density")
            
            # Draw tip cell movement paths
            for tip, path in simulation.cell_paths.items():
                path_array = np.array(path)
                if len(path_array) > 1:  # Need at least 2 points for quiver
                    # Calculate movement vectors
                    dx = np.diff(path_array[:, 0])
                    dy = np.diff(path_array[:, 1])
                    
                    plt.quiver(
                        path_array[:-1, 0],  # x starts
                        path_array[:-1, 1],  # y starts
                        dx, dy,  # x and y movement vectors
                        angles='xy', 
                        scale_units='xy', 
                        scale=1, 
                        color='white'
                    )
            
            # Draw current tip cells
            for i, (x, y) in enumerate(simulation.tip_cells):
                if i < len(simulation.active_tips) and simulation.active_tips[i]:
                    plt.scatter(x, y, color='red', s=100)
                else:
                    plt.scatter(x, y, color='gray', s=100)
            
            # Draw anastomosis events
            for tip1, tip2 in simulation.anastomosis_events:
                plt.plot([tip1[0], tip2[0]], [tip1[1], tip2[1]], 'g-', linewidth=2)
                plt.scatter(tip1[0], tip1[1], color='black', s=80)
                plt.scatter(tip2[0], tip2[1], color='black', s=80)
            
            plt.title(f"{gradient_type.capitalize()} Gradient - Run {run+1}/{num_runs} - Final State - ECM Density & Tip Cell Movements")
            
            # Save the final plot
            final_plot_file_name = f"step_final.png"
            final_plot_file_path = os.path.join(plot_dir, final_plot_file_name)
            plt.savefig(final_plot_file_path, dpi=300, bbox_inches="tight")
            plt.close()
            
            print(f"Saved final plot to {final_plot_file_path}")
        
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
            tsv_file.write("CellID\tPositionX\tPositionY\tECMValue\tDisplacement\tAnastomosisCount\n")
            
            # Write data for each cell
            for idx in range(len(initial_tip_positions)):
                displacement = displacements.get(idx, "N/A")
                anastomosis = anastomosis_count.get(idx, 0)
                
                # First write the cell's summary line
                tsv_file.write(f"{idx}\tN/A\tN/A\tN/A\t{displacement:.2f}\t{anastomosis}\n")
                
                # Then write each position with ECM value
                for i, (x, y, ecm) in enumerate(cell_ecm_values.get(idx, [])):
                    tsv_file.write(f"{idx}\t{x}\t{y}\t{ecm:.4f}\tN/A\tN/A\n")
        
        print(f"{gradient_type.capitalize()} gradient simulation run #{run+1} completed.")
        print(f"Results saved to {tsv_file_path}")
        print(f"Plots saved to {plot_dir}")
        
        result_files.append(tsv_file_path)
    
    return result_files
def get_gradient_specific_directory(gradient_type):
    """Return a specific directory path based on gradient type"""
    # Define the specific directories for each gradient type
    gradient_directories = {
        'linear': os.path.expanduser("~/Desktop/sproj'24-'25/ecmmodel(workingversion)/ECM_model/ecm/lin_data"),
        'radial': os.path.expanduser("~/Desktop/sproj'24-'25/ecmmodel(workingversion)/ECM_model/ecm/rad_data"),
        'uniform': os.path.expanduser("~/Desktop/sproj'24-'25/ecmmodel(workingversion)/ECM_model/ecm/uni_data")
    }
    
    # Get the directory for the specified gradient type
    output_dir = gradient_directories.get(gradient_type)
    
    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir

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
    parser = argparse.ArgumentParser(description='Run ECM Tip Cell Model Simulation with different gradients')
    
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