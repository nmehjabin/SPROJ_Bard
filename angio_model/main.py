import sys
import os
from datetime import datetime
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from vas import Vasculature

def calculate_displacement(initial_pos, final_pos):
    return np.linalg.norm(np.array(final_pos) - np.array(initial_pos))

def draw_anastomosis_events(ax, events):
    for event in events:
        if isinstance(event, dict):
            if event['type'] == 'tip-to-tip':
                pos1, pos2 = event['positions']
                ax.plot([pos1[1], pos2[1]], [pos1[0], pos2[0]], 'g-', linewidth=2)
                ax.scatter(pos1[1], pos1[0], color='yellow', s=80)
                ax.scatter(pos2[1], pos2[0], color='yellow', s=80)
            elif event['type'] == 'tip-to-vessel':
                tip_pos = event['tip_position']
                vessel_pos = event['vessel_position']
                ax.plot([tip_pos[1], vessel_pos[1]], [tip_pos[0], vessel_pos[0]], 'm-', linewidth=2)
                ax.scatter(tip_pos[1], tip_pos[0], color='yellow', s=80)
                ax.scatter(vessel_pos[1], vessel_pos[0], color='cyan', s=80)
        else:
            tip1, tip2 = event
            ax.plot([tip1[1], tip2[1]], [tip1[0], tip2[0]], 'g-', linewidth=2)
            ax.scatter(tip1[1], tip1[0], color='yellow', s=80)
            ax.scatter(tip2[1], tip2[0], color='yellow', s=80)

# Initialize grid and VEGF
grid_size = (50, 50)
grid = np.zeros(grid_size, dtype=int)
# vegf_field = np.random.uniform(0, 1, grid_size)
vegf_field = np.random.normal(loc=0.5, scale=0.1, size=(50, 50))

initial_tip_positions = [(15, 25), (35, 25), (25, 15), (25, 35), (20, 20), (30, 30)]
vasculature = Vasculature(grid, initial_tip_positions, vegf_field)
if not hasattr(vasculature, 'anastomosis_events'):
    vasculature.anastomosis_events = []

anastomosis_count = {i+1: 0 for i in range(len(initial_tip_positions))}
steps = 200
initial_positions = {idx + 1: pos for idx, pos in enumerate(initial_tip_positions)}
vessel_vegf_values = {i+1: [] for i in range(len(initial_tip_positions))}

# Create a folder for saving results on the desktop
current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
folder_path = os.path.join(os.path.expanduser("~/Desktop/sproj'24-'25/angio_model(workingversion)/angio_figures"), f"DualGrid_Simulation_{current_date}")

# Ensure directory exists
try:
    os.makedirs(folder_path, exist_ok=True)
    print(f"Created directory: {folder_path}")
except Exception as e:
    print(f"Error creating directory: {e}")
    # Fallback to current directory if desktop path fails
    folder_path = f"DualGrid_Simulation_{current_date}"
    os.makedirs(folder_path, exist_ok=True)

log_path = os.path.join(folder_path, "simulation_log.txt")
tsv_path = os.path.join(folder_path, f"vasculature_data_{current_date}.tsv")
final_plot_path = os.path.join(folder_path, f"final_vasculature_{current_date}.svg")
step_plots_folder = os.path.join(folder_path, "step_plots")
os.makedirs(step_plots_folder, exist_ok=True)

# Save intermediate plots for specific steps (e.g., every 20 steps)
save_interval = 20

# Create and redirect output to log file
with open(log_path, 'w') as log_file:
    original_stdout = sys.stdout
    sys.stdout = log_file
    
    print(f"Simulation started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Grid size: {grid_size}")
    print(f"Initial tip positions: {initial_tip_positions}")
    print(f"Total steps: {steps}")
    
    plt.figure(figsize=(10, 24))

    for step in range(steps):
        print(f"\n--- Step {step+1}/{steps} ---")

        current_tips = []
        for tip in vasculature.tip_cells:
            tip_id = vasculature.tip_ids[tip]
            vegf_value = vasculature.vegf_field[tip]
            current_tips.append((tip_id, tip, vegf_value))
            vessel_vegf_values[tip_id].append((*tip, vegf_value))

        print("Current tip positions before growth:")
        for tip_id, tip, vegf in current_tips:
            print(f"  Vessel ID {tip_id}: position {tip}, VEGF value: {vegf:.4f}")

        if step % 2 == 0:
            print("Growing vasculature...")
            vasculature.grow()
            vasculature.vegf_field *= 0.95
        else:
            print("Checking anastomosis...")
            before = len(vasculature.anastomosis_events)
            vasculature.check_anastomosis()
            after = len(vasculature.anastomosis_events)

            for i in range(before, after):
                event = vasculature.anastomosis_events[i]
                if isinstance(event, dict):
                    if event['type'] == 'tip-to-tip':
                        id1, id2 = event['vessel_ids']
                        anastomosis_count[id1] += 1
                        anastomosis_count[id2] += 1
                    elif event['type'] == 'tip-to-vessel':
                        tid, vid = event['tip_id'], event['vessel_id']
                        anastomosis_count[tid] += 1
                        anastomosis_count[vid] += 1
                else:
                    tip1, tip2 = event
                    id1 = vasculature.tip_ids.get(tip1)
                    id2 = vasculature.tip_ids.get(tip2)
                    if id1: anastomosis_count[id1] += 1
                    if id2: anastomosis_count[id2] += 1

        print("Tip positions after growth:")
        for tip in vasculature.tip_cells:
            tip_id = vasculature.tip_ids[tip]
            vegf_value = vasculature.vegf_field[tip]
            print(f"  Vessel ID {tip_id}: position {tip}, VEGF value: {vegf_value:.4f}")

        # Create visualization
        plt.clf()
        
        # Plot 1: Tip cell IDs
        plt.subplot(3, 1, 1)
        plt.imshow(vasculature.grid, cmap=ListedColormap(['black', 'blue', 'red']), vmin=0, vmax=2)
        plt.title(f"(A) Tip cell IDs - Step {step+1}",fontsize=22)
        for tip in vasculature.tip_cells:
            plt.text(tip[1], tip[0], str(vasculature.tip_ids[tip]), color='white', fontsize=14, ha='center', va='center')
        
        # Plot 2: Vessel IDs
        plt.subplot(3, 1, 2)
        max_id = max(vasculature.tip_ids.values())
        plt.imshow(vasculature.gridID, cmap=plt.cm.get_cmap('tab20', max_id+1), vmin=0, vmax=max_id)
        plt.title(f"(B) Vasculature Growth - Step {step+1}",fontsize=22)
        
        # Plot 3: VEGF concentration
        plt.subplot(3, 1, 3)
        plt.imshow(vasculature.vegf_field, cmap='viridis')
        plt.title("(C) VEGF Concentration",fontsize=22)
        for tip in vasculature.tip_cells:
            plt.scatter(tip[1], tip[0], color='red', s=50, marker='*')
        draw_anastomosis_events(plt.gca(), vasculature.anastomosis_events)
        
        plt.tight_layout()
        
        # Save step plots at intervals
        if step % save_interval == 0 or step == steps - 1:
            step_plot_path = os.path.join(step_plots_folder, f"step_{step+1}.svg")
            try:
                plt.savefig(step_plot_path, dpi=200)
                print(f"Saved step plot to: {step_plot_path}")
            except Exception as e:
                print(f"Error saving step plot: {e}")
        
        plt.pause(0.1)

    print("\nCalculating final displacements...")
    displacements = {}
    vessel_status = {}  # Track final status of each vessel
    
    for idx, initial_pos in list(initial_positions.items()):
        is_active = False
        final_pos = None
        
        # Check if this vessel ID is still an active tip
        for tip in vasculature.tip_cells:
            if vasculature.tip_ids[tip] == idx:
                final_pos = tip
                is_active = True
                vessel_status[idx] = "Active"
                break
        
        # If not active tip, check if it underwent anastomosis
        if not is_active:
            has_anastomosis = anastomosis_count[idx] > 0
            vessel_status[idx] = "Anastomosed" if has_anastomosis else "Inactive"
            
            # Find its last position from vessel_vegf_values
            if vessel_vegf_values[idx]:
                x, y, _ = vessel_vegf_values[idx][-1]
                final_pos = (x, y)
        
        if final_pos:
            displacement = calculate_displacement(initial_pos, final_pos)
            displacements[idx] = displacement
        else:
            displacements[idx] = 0
            
        print(f"  Vessel ID {idx}: Displacement: {displacements[idx]}, Status: {vessel_status[idx]}")

    # Switch back to the original stdout before writing to files
    sys.stdout = original_stdout

    # Write TSV data with vessel status
    try:
        with open(tsv_path, 'w') as tsv:
            tsv.write("VesselID\tPositionX\tPositionY\tVEGFValue\tDisplacement\tAnastomosisCount\tStatus\n")
            for vid in sorted(vessel_vegf_values):
                disp = displacements.get(vid, 0)
                ana = anastomosis_count.get(vid, 0)
                status = vessel_status.get(vid, "Unknown")
                
                for x, y, v in vessel_vegf_values[vid]:
                    tsv.write(f"{vid}\t{x}\t{y}\t{v:.4f}\t{disp}\t{ana}\t{status}\n")
            print(f"Data saved to TSV file: {tsv_path}")
    except Exception as e:
        print(f"Error saving TSV file: {e}")

    # Save final plot with summary statistics
    try:
        plt.figure(figsize=(10, 18))
        
        # # Main plots (3 across)
        # plt.subplot(2, 3, 1)
        # plt.imshow(vasculature.grid, cmap=ListedColormap(['black', 'blue', 'red']), vmin=0, vmax=2)
        # plt.title("Final Vessel Grid")
        # for tip in vasculature.tip_cells:
        #     plt.text(tip[1], tip[0], str(vasculature.tip_ids[tip]), color='white', fontsize=12, ha='center', va='center')
        
        # plt.subplot(2, 3, 2)
        # plt.imshow(vasculature.gridID, cmap=plt.cm.get_cmap('tab20', max_id+1), vmin=0, vmax=max_id)
        # plt.title("Final Vessel IDs")
        
        # plt.subplot(2, 3, 3)
        # plt.imshow(vasculature.vegf_field, cmap='viridis')
        # plt.title("Final VEGF Concentration")
        # for tip in vasculature.tip_cells:
        #     plt.scatter(tip[1], tip[0], color='red', s=50, marker='*')
        # draw_anastomosis_events(plt.gca(), vasculature.anastomosis_events)
        
        # Summary statistics (bottom row)
        plt.subplot(3, 1, 1)
        vessel_ids = list(sorted(vessel_vegf_values.keys()))
        plt.bar(vessel_ids, [displacements.get(vid, 0) for vid in vessel_ids])
        plt.title("(A) Displacement by Vessel ID",fontsize=22)
        plt.xlabel("Vessel ID")
        plt.ylabel("Displacement")
        
        plt.subplot(3, 1, 2)
        plt.bar(vessel_ids, [anastomosis_count.get(vid, 0) for vid in vessel_ids])
        plt.title("(B) Anastomosis Count by Vessel ID",fontsize=22)
        plt.xlabel("Vessel ID")
        plt.ylabel("Count")
        
        plt.subplot(3, 1, 3)
        status_counts = {"Active": 0, "Anastomosed": 0, "Inactive": 0}
        for status in vessel_status.values():
            status_counts[status] = status_counts.get(status, 0) + 1
        plt.pie(status_counts.values(), labels=status_counts.keys(), autopct='%1.1f%%')
        plt.title("(C) Vessel Status Distribution",fontsize=22)
        
        plt.tight_layout()
        plt.savefig(final_plot_path, dpi=300)
        print(f"Final plot saved to: {final_plot_path}")
    except Exception as e:
        print(f"Error saving final plot: {e}")
    
    print(f"\nSimulation completed! Results saved to: {folder_path}")
    
    # Create a summary file
    summary_path = os.path.join(folder_path, "simulation_summary.txt")
    try:
        with open(summary_path, 'w') as summary_file:
            summary_file.write(f"Angiogenesis Simulation Summary\n")
            summary_file.write(f"===========================\n")
            summary_file.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            summary_file.write(f"Grid Size: {grid_size}\n")
            summary_file.write(f"Total Steps: {steps}\n")
            summary_file.write(f"Initial Tip Count: {len(initial_tip_positions)}\n")
            summary_file.write(f"Final Active Tip Count: {len(vasculature.tip_cells)}\n")
            summary_file.write(f"Total Anastomosis Events: {len(vasculature.anastomosis_events)}\n\n")
            
            summary_file.write(f"Vessel Summary:\n")
            summary_file.write(f"ID\tDisp\tAnastomosis\tStatus\n")
            summary_file.write(f"-------------------------------\n")
            for vid in sorted(vessel_vegf_values.keys()):
                summary_file.write(f"{vid}\t{displacements.get(vid, 0):.2f}\t{anastomosis_count.get(vid, 0)}\t{vessel_status.get(vid, 'Unknown')}\n")
                
        print(f"Summary saved to: {summary_path}")
    except Exception as e:
        print(f"Error saving summary file: {e}")