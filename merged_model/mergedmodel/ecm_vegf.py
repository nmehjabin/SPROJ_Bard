import numpy as np
import random
import os

class ECMTipCellModel:
    def __init__(self, grid_size, gradient_type='linear', gradient_params=None, initial_tips=[(1,2)]):
        """
        Initialize the ECM Tip Cell Model with flexible gradient generation
        
        Parameters:
        - grid_size: Tuple (height, width) of the grid
        - gradient_type: Type of gradient ('linear', 'radial', 'uniform')
        - gradient_params: Dictionary of parameters specific to gradient type
        - initial_tips: List of initial tip cell positions
        """
        self.grid_size = grid_size
        self.gradient_type = gradient_type
        
        # Default gradient parameters if not provided
        default_params = {
            'linear': {'slope': 0.05, 'intercept': 0.2},
            'radial': {'center_value': 0.2, 'decay_rate': 0.05},
            'uniform': {'uniform_value': 0.5}
        }
        
        # Merge default parameters with provided parameters
        gradient_params = gradient_params or default_params.get(gradient_type, {})
        
        # Create gradient based on type
        self.ecm_grid = self._create_gradient(gradient_params)
        
        # Initialize tip cell tracking
        self.tip_cells = initial_tips
        self.cell_paths = {tip: [tip] for tip in initial_tips}
        self.is_first_move = {tip: True for tip in initial_tips}
        self.active_tips = [True] * len(initial_tips)
        self.anastomosis_events = []

        # Create random VEGF distribution
        self.vegf_grid = np.random.uniform(0, 1, self.grid_size)

    def get_vegf_value(self, position):
        """Retrieves VEGF concentration at a given position."""
        x, y = position
        return self.vegf_grid[y, x]
    
    def _create_gradient(self, params):
        """
        Create gradient based on specified type and parameters
        
        Supports:
        - Linear gradient: slope, intercept
        - Radial gradient: center_value, decay_rate
        - Uniform gradient: uniform_value
        """
        if self.gradient_type == 'linear':
            return self._create_linear_gradient(
                params.get('slope', 0.05), 
                params.get('intercept', 0.2)
            )
        elif self.gradient_type == 'radial':
            return self._create_radial_gradient(
                params.get('center_value', 0.2), 
                params.get('decay_rate', 0.05)
            )
        elif self.gradient_type == 'uniform':
            return self._create_uniform_gradient(
                params.get('uniform_value', 0.5)
            )
        else:
            raise ValueError(f"Unsupported gradient type: {self.gradient_type}")

    def _create_linear_gradient(self, slope, intercept):
        """Creates a 2D ECM grid with a linear gradient along the x-direction."""
        x_coordinates = np.arange(self.grid_size[1])
        linear_gradient = slope * x_coordinates + intercept
        return np.tile(linear_gradient, (self.grid_size[0], 1))

    def _create_radial_gradient(self, center_value, decay_rate):
        """Creates a 2D ECM grid with a radial gradient using Euclidean distance."""
        center_x = (self.grid_size[1] - 1) / 2
        center_y = (self.grid_size[0] - 1) / 2
        
        x = np.linspace(0, self.grid_size[1] - 1, self.grid_size[1])
        y = np.linspace(0, self.grid_size[0] - 1, self.grid_size[0])
        xx, yy = np.meshgrid(x, y)
        
        distances = np.sqrt((xx - center_x)**2 + (yy - center_y)**2)
        radial_gradient = center_value * np.exp(-decay_rate * distances)
        
        return radial_gradient

    def _create_uniform_gradient(self, uniform_value):
        """Creates a 2D ECM grid with a uniform gradient."""
        return np.full(self.grid_size, uniform_value)

    def get_ecm_value(self, position):
        """Retrieves ECM density at a given position."""
        x, y = position
        return self.ecm_grid[y, x]

    def get_valid_neighbors(self, current_position):
        """Select valid neighbors with directional constraints."""
        '''
        Run down example:
        x = 25, y = 45
        path = [(26, 45), (25, 45)]
        prev_position = (26, 45)
        dx = 25 - 26 = -1
        dy = 45 - 45 = 0
        neighbors = [
        (25 + (-1), 45 + 0),      # (24, 45) - Forward
        (25 + (-1), 45),          # (24, 45) - Side right
        (25, 45 + 0),             # (25, 45) - Side left
        (25 + (-1), 45 - 0),      # (24, 45) - Adjusted Side backward
        (25, 45 - 0)              # (25, 45) - Adjusted Side backward
    ]
    # This results in: [(24, 45), (24, 45), (25, 45), (24, 45), (25, 45)], checks out of bound for final check
    '''
        x, y = current_position
        current_position = tuple(current_position)  # Ensure it's a tuple

        # Check if this position exists in cell_paths, if not initialize it
        if current_position not in self.cell_paths:
            self.cell_paths[current_position] = [current_position]
            self.is_first_move[current_position] = True

        if self.is_first_move.get(current_position, False):
            # For first move, consider all 8 neighboring positions
            neighbors = [
                (x+1, y), (x-1, y), (x, y+1), (x, y-1),  # Cardinal directions
                (x+1, y+1), (x-1, y-1), (x+1, y-1), (x-1, y+1)  # Diagonals
            ]
            print("First move: All 8 neighbors selected")
        else:
            path = self.cell_paths[current_position]
            
            # Get previous position correctly
            if len(path) < 2:
                prev_position = path[-1]  # Stay in place if no history
            else:
                prev_position = path[-2]  # Correctly get the last move

            dx = x - prev_position[0]
            dy = y - prev_position[1]

            print(f"Previous position: {prev_position}")
            print(f"Movement direction: dx={dx}, dy={dy}")

            # Define forward-biased neighbors based on previous movement
            if dy == 0:
                neighbors = [
                    (x,y-1), 
                    (x,y+1), 
                    (x-1,y),
                    (x-1,y+1),
                    (x-1,y-1)
                ]
                print(f"Corrected forward-biased neighbors dy==0: {neighbors}")
                
            elif dy > 0 and dx < 0:
                neighbors = [
                    (x,y+1), 
                    (x+1,y+1), 
                    (x-1,y),
                    (x-1,y+1), 
                    (x-1,y-1)
                ]
                print(f"Corrected forward-biased neighbors dy=1, dx =-1, : {neighbors}")

            elif dy < 0 and dx < 0:
                neighbors = [
                    (x,y+1), 
                    (x,y-1), 
                    (x-1,y),
                    (x-1,y+1), 
                    (x-1,y-1)
                ]
                print(f"Corrected forward-biased neighbors dy=-1, dx =-1: {neighbors}")

            elif dx == 0:
                neighbors = [
                    (x,y+dy), 
                    (x+1,y+dy),
                    (x+1,y), 
                    (x-1,y), 
                    (x-1,y+dy)
                ]
                print(f"Corrected forward-biased neighbors dx==0: {neighbors}")

            elif dx > 0 and dy < 0:
                neighbors = [
                    (x+1,y), 
                    (x+1,y+1), 
                    (x+1,y-1),
                    (x,y+1), 
                    (x,y-1)
                ]
                print(f"Corrected forward-biased neighbors dx=1, dy =-1: {neighbors}")

            elif dx > 0 and dy > 0:
                neighbors = [
                    (x+1,y), 
                    (x+1,y+1), 
                    (x+1,y-1),
                    (x,y+1), 
                    (x,y-1)
                ]
                print(f"Corrected forward-biased neighbors dx=1, dy =1: {neighbors}")

            else:
                raise ValueError(f"Unexpected movement direction: dx={dx}, dy={dy}")

        # Filter out-of-bound positions
        valid_neighbors = [
            n for n in neighbors if 0 <= n[0] < self.grid_size[1] and 0 <= n[1] < self.grid_size[0]
        ]
        print(f"Valid neighbors after grid boundary check: {valid_neighbors}")
        
        return valid_neighbors
    
    def move_tip_cell(self):
        """Moves tip cells toward the lowest density neighbors with directional constraint."""
        new_positions = []
        avg_positions = []  # List to store average positions

        for i, tip in enumerate(self.tip_cells):
            # Skip inactive tips
            if hasattr(self, 'active_tips') and i < len(self.active_tips) and not self.active_tips[i]:
                new_positions.append(tip)
                avg_positions.append(None)  # No average position for inactive tips
                continue

            print(f"\n=== Moving tip cell {i} at {tip} ===")
            tip = tuple(tip)  # Ensure tip is a tuple for dictionary keys
            # Ensure this tip exists in cell_paths
            if tip not in self.cell_paths:
                self.cell_paths[tip] = [tip]
                self.is_first_move[tip] = True

            print(f"Current path: {self.cell_paths.get(tip, 'No path')}")
            
            # Check if at boundary
            x, y = tip
            if x == 0 or x == self.grid_size[1]-1 or y == 0 or y == self.grid_size[0]-1:

                print(f"Tip cell at {tip} has reached boundary. Stopping growth.")
                # Keep the tip cell in its current position but mark it as inactive
                new_positions.append(tip)
                if i < len(self.active_tips):
                    self.active_tips[i] = False
                continue
                
            # Get valid neighbors 
            valid_neighbors = self.get_valid_neighbors(tip)
            
            if valid_neighbors:

                #get the ECM value for each valid neighbor and Vegf value
                #and then get lowest ecm value randomly and highest vegf value randomly
                #average value of ecm and vegf and whatever neighbor has the closest velue to avgerage value is the next postion
                ecm_values = []
                vegf_values = []
                neighbor_data = []
                lowest_ecm = []
                highest_vegf = []
                
                for neighbor in valid_neighbors:
                    ecm_val = self.get_ecm_value(neighbor)
                    vegf_val = self.get_vegf_value(neighbor)
                    ecm_values.append(ecm_val)
                    vegf_values.append(vegf_val)
                    neighbor_data.append((neighbor, ecm_val, vegf_val))

                # Find the lowest ECM value
                lowest_ecm_value = min(ecm_values)
                lowest_ecm = [neighbor for neighbor, ecm, _ in neighbor_data if ecm == lowest_ecm_value]

                # Find the highest VEGF value
                highest_vegf_value = max(vegf_values)
                highest_vegf = [neighbor for neighbor, _, vegf in neighbor_data if vegf == highest_vegf_value]

                # Condition 1: check if there are any positions that have both lowest ECM and highest VEGF
                common_positions = set(lowest_ecm) & set(highest_vegf)
                
                if common_positions:
                    # If there's a position with both lowest ECM and highest VEGF, choose that one
                    next_position = random.choice(list(common_positions))
                    avg_positions.append(None)  # No average needed in this case
                    print(f"Found position with both lowest ECM and highest VEGF: {next_position}")
                else:
                    # Condition 2: Average the coordinates of a random lowest ECM position and highest VEGF position
                    random_lowest_ecm = random.choice(lowest_ecm)
                    random_highest_vegf = random.choice(highest_vegf)
                    
                    # Average the coordinates
                    x1, y1 = random_lowest_ecm
                    x2, y2 = random_highest_vegf
                    avg_x = (x1 + x2) // 2  # Integer division to ensure grid coordinates
                    avg_y = (y1 + y2) // 2
                    avg_position = (avg_x, avg_y)
                    next_position = avg_position
                    avg_positions.append(avg_position)
                    
                    # # Check if the averaged position is valid
                    # if avg_position in valid_neighbors:
                    #     next_position = avg_position
                    #     print(f"Using averaged position from ECM={random_lowest_ecm} and VEGF={random_highest_vegf}: {next_position}")
                    # else:
                    #     # If averaged position is not valid, use the lowest ECM position
                    #     next_position = random_lowest_ecm
                    #     print(f"Averaged position not valid, using lowest ECM position: {next_position}")
                
                # Update direction and mark first move as complete
                if self.is_first_move.get(tip, False):
                    self.is_first_move[tip] = False

                # Update path for this tip cell
                if next_position in self.cell_paths:
                    # If next_position is already a key, we have a conflict
                    print(f"Warning: Position {next_position} already exists in cell_paths!")
                    self.cell_paths[next_position].extend(self.cell_paths[tip])
                else:
                    self.cell_paths[next_position] = self.cell_paths[tip] + [next_position]
                    
                # Update is_first_move for the new position
                self.is_first_move[next_position] = False

                # Clean up old position
                if tip in self.cell_paths:
                    del self.cell_paths[tip]
                if tip in self.is_first_move:
                    del self.is_first_move[tip]
                    
                new_positions.append(next_position)

                # Check if the NEW position is at boundary
                nx, ny = next_position
                if nx == 0 or nx == self.grid_size[1]-1 or ny == 0 or ny == self.grid_size[0]-1:
                    print(f"Tip cell has moved to boundary at {next_position}. Will stop in next iteration.")
            else:
                print(f"No valid lowest density neighbors. Staying at {tip}")
                new_positions.append(tip)
                if i < len(self.active_tips):
                    self.active_tips[i] = False
        
        # Update tip cell positions
        self.tip_cells = new_positions

        # Ensure active_tips length matches tip_cells
        if not hasattr(self, 'active_tips') or len(self.active_tips) != len(self.tip_cells):
            self.active_tips = [True] * len(self.tip_cells)
                    
        print("\n--- Tip Cell Positions After Move ---")
        print(f"New tip cell positions: {self.tip_cells}")
        print(f"Active status: {self.active_tips}")
        print("Updated cell paths:")
        for pos, path in self.cell_paths.items():
            print(f"{pos}: {path}")
        return avg_positions  # Return the average positions
    
    def check_anastomosis(self):
        """
        Check and handle anastomosis between:
        1. Active tip cells connecting with each other (distance=2)
        2. Active tip cells connecting with pre-existing vasculature (distance=1)
        """
        print("\n=== Starting Anastomosis Detection ===")
        
        # Ensure active_tips is properly initialized
        if not hasattr(self, 'active_tips') or len(self.active_tips) != len(self.tip_cells):
            self.active_tips = [True] * len(self.tip_cells)
            print(f"Initialized active_tips: {self.active_tips}")
        
        # Create a grid to track all vessel segments
        vessel_grid = np.zeros(self.grid_size, dtype=bool)
        
        # Create a dictionary to track which tip cell created each vessel segment
        vessel_owner = {}
        
        # Map each tip cell to its index for ownership tracking
        tip_to_index = {tuple(tip): idx for idx, tip in enumerate(self.tip_cells)}
        print(f"Tip cell to index mapping: {tip_to_index}")
        
        # First, mark all existing vessel segments on the grid
        print("\n--- Creating Vessel Grid and Tracking Ownership ---")
        for tip, path in self.cell_paths.items():
            tip_idx = tip_to_index.get(tip, -1)
            print(f"Processing path for tip {tip} (index {tip_idx}), path length: {len(path)}")
            
            for position in path[:-1]:  # Exclude the tip cell itself
                x, y = position
                vessel_grid[y, x] = True
                vessel_owner[(x, y)] = tip_idx
                print(f"  Marked vessel at {position} owned by tip index {tip_idx}")
        
        # Get all currently active tips
        active_tip_indices = [i for i, is_active in enumerate(self.active_tips) if is_active]
        print(f"\nActive tip indices: {active_tip_indices}")
        tips_to_deactivate = []
        
        # Check each active tip for anastomosis
        print("\n--- Checking for Anastomosis Events ---")
        for i in active_tip_indices:
            if i in tips_to_deactivate:
                print(f"Tip {i} already marked for deactivation, skipping")
                continue
                
            tip1 = tuple(self.tip_cells[i])
            print(f"\nExamining tip cell {i} at position {tip1}")
            
            # CONDITION 1: Check if this tip is near another active tip (TIP-TIP anastomosis)
            print("Checking for tip-to-tip anastomosis...")
            for j in active_tip_indices:
                # Don't compare a tip to itself
                if i != j and j not in tips_to_deactivate:
                    tip2 = tuple(self.tip_cells[j])
                    distance = max(abs(tip1[0] - tip2[0]), abs(tip1[1] - tip2[1]))
                    print(f"  Comparing with tip {j} at {tip2}, distance = {distance}")
                    
                    # Use distance=2 for tip-to-tip anastomosis
                    if self.is_near(tip1, tip2, distance=2):
                        print(f"  TIP-TIP ANASTOMOSIS DETECTED between tips at {tip1} and {tip2}")
                        # Mark both tips for deactivation
                        tips_to_deactivate.append(i)
                        tips_to_deactivate.append(j)
                        
                        # Record the anastomosis event
                        self.anastomosis_events.append((tip1, tip2))
                        break
            
            # If no tip-to-tip anastomosis was found, check for tip-to-vessel anastomosis
            if i not in tips_to_deactivate:
                print("Checking for tip-to-vessel anastomosis...")
                
                # Get neighbors of this tip with distance=1 for vessel anastomosis
                neighbors = self.get_vessel_neighbors(tip1, distance=1)
                print(f"  Neighbors to check: {neighbors}")
                
                # Get this tip's own path for exclusion
                own_path = self.cell_paths.get(tip1, [tip1])
                exclusion_path = set(own_path)
                print(f"  Own path to exclude from collision detection: {list(exclusion_path)}")
                
                # Check if any neighbor is part of existing vasculature but not own path
                for nx, ny in neighbors:
                    # Skip if out of bounds
                    if not (0 <= nx < self.grid_size[1] and 0 <= ny < self.grid_size[0]):
                        print(f"  Neighbor {(nx, ny)} is out of bounds, skipping")
                        continue
                    
                    # Check if this neighbor is a vessel
                    if vessel_grid[ny, nx]:
                        if (nx, ny) in exclusion_path:
                            print(f"  Vessel at {(nx, ny)} is part of tip's own path, ignoring")
                            continue
                            
                        # Check which tip owns this vessel segment
                        vessel_tip_idx = vessel_owner.get((nx, ny), -1)
                        print(f"  Found vessel at {(nx, ny)} owned by tip {vessel_tip_idx}")
                        
                        if vessel_tip_idx != i:  # If it belongs to a different tip
                            print(f"  TIP-VESSEL ANASTOMOSIS DETECTED between tip {i} at {tip1} and vessel at {(nx, ny)}")
                            tips_to_deactivate.append(i)
                            # Store as (tip position, vessel position)
                            self.anastomosis_events.append((tip1, (nx, ny)))
                            break
                        else:
                            print(f"  Vessel at {(nx, ny)} belongs to the same tip, ignoring")
            
        # Deactivate tip cells that have formed anastomosis
        print("\n--- Deactivating Tips After Anastomosis ---")
        for index in tips_to_deactivate:
            if index < len(self.active_tips):
                self.active_tips[index] = False
                print(f"Tip cell at {self.tip_cells[index]} (index {index}) deactivated after anastomosis")

        print(f"\nAnastomosis events so far: {len(self.anastomosis_events)}")
        for event in self.anastomosis_events:
            print(f"  {event}")
        print(f"Remaining active tip cells: {sum(self.active_tips)}")
        
        return len(self.anastomosis_events) > 0




    def is_near(self, tip1, tip2, distance=2):
        """Check if two tips are within a certain distance."""
        return abs(tip1[0] - tip2[0]) <= distance and abs(tip1[1] - tip2[1]) <= distance


#what is this function doing here- need to figure that out.
    def get_vessel_neighbors(self, position, distance=1):
        """Get all neighboring positions within a certain distance."""
        x, y = position
        neighbors = []
        
        for dx in range(-distance, distance + 1):
            for dy in range(-distance, distance + 1):
                # Skip the center position itself
                if dx == 0 and dy == 0:
                    continue
                
                nx, ny = x + dx, y + dy
                # Only add positions within the grid
                if 0 <= nx < self.grid_size[1] and 0 <= ny < self.grid_size[0]:
                    neighbors.append((nx, ny))
    
        return neighbors

    