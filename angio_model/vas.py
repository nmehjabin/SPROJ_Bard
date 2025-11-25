import random
import numpy as np  

class Vasculature:
    def __init__(self, grid, initial_tip_positions, vegf_field):
        self.grid = grid.copy()  # Grid for vessel types (0: empty, 1: mature vessel, 2: tip)
        self.gridID = np.zeros_like(grid, dtype=int)  # Grid for vessel IDs (0: no vessel, n: vessel with ID n)
        self.grid_size = grid.shape  # Store grid dimensions
        self.tip_cells = initial_tip_positions.copy()  # List of active tip positions
        self.cell_paths = {tip: [tip] for tip in initial_tip_positions}
        self.is_first_move = {tip: True for tip in initial_tip_positions}
        self.active_tips = [True] * len(initial_tip_positions)  # Track which tips are active
        self.tip_ids = {}  # Track which ID belongs to which tip
        
        self.vegf_field = vegf_field.copy()  # VEGF concentration field
        self.vas_positions = []  # Store grown Vasculature
        self.anastomosis_events = []
        
        # Initialize tip IDs and update both grids
        for i, pos in enumerate(initial_tip_positions):
            tip_id = i + 1  # Start IDs from 1 (0 means no vessel)
            self.tip_ids[pos] = tip_id
            self.grid[pos] = 2  # Mark initial tip position on the grid
            self.gridID[pos] = tip_id  # Assign unique ID to each initial tip position

    def get_valid_neighbors(self, position):
        x, y = position
        
        if self.is_first_move.get(position, False):
            # For first move, consider all 8 neighboring positions
            neighbors = [
                (x+1, y), (x-1, y), (x, y+1), (x, y-1),  # Cardinal directions
                (x+1, y+1), (x-1, y-1), (x+1, y-1), (x-1, y+1)  # Diagonals
            ]
            print("First move: All 8 neighbors selected")
        else:
            path = self.cell_paths[position]
            
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
        
        # Also filter out positions that already have a vessel
        valid_neighbors = [n for n in valid_neighbors if self.grid[n] == 0]
        
        print(f"Valid neighbors after grid boundary check: {valid_neighbors}")
        
        return valid_neighbors
    def grow(self):
        new_tip_cells = [] # store new tip positions 
        
        for i, tip in enumerate(self.tip_cells):
            if not self.active_tips[i]:
                continue

            # Get the ID of this tip
            tip_id = self.tip_ids[tip]
            
            # Mark the current tip position as mature vessel in grid (1)
            self.grid[tip] = 1  # matured vasculature
            # ID remains the same in gridID
            
            # Add to mature vessel positions
            self.vas_positions.append(tip)

            self.vegf_field[tip] *= 0.5  # Consume VEGF at the current position
            
            empty_neighbors = self.get_valid_neighbors(tip)

            # Choose growth direction based on VEGF concentration
            if empty_neighbors:
                # Log VEGF values for each neighbor
                print(f"\nDetailed VEGF analysis for tip ID {tip_id} at position {tip}:")
                vegf_data = [(n, self.vegf_field[n]) for n in empty_neighbors]
                for neighbor, vegf_value in vegf_data:
                    print(f"  Neighbor {neighbor}: VEGF value = {vegf_value:.6f}")
                
                # Find the maximum VEGF value among all neighbors
                max_vegf_value = max(self.vegf_field[n] for n in empty_neighbors)
            
                # Find all neighbors that have this maximum VEGF value
                max_vegf_neighbors = [n for n in empty_neighbors if self.vegf_field[n] == max_vegf_value]
                
                print(f"  Maximum VEGF value: {max_vegf_value:.6f}")
                print(f"  Neighbors with maximum VEGF: {max_vegf_neighbors}")
            
                # Randomly choose one from neighbors with maximum VEGF value
                new_position = random.choice(max_vegf_neighbors)
                
                print(f"  Selected neighbor: {new_position} (VEGF: {self.vegf_field[new_position]:.6f})")
            
                # Update grid with new tip position
                self.grid[new_position] = 2  # Mark new tip position as tip (2)
                self.gridID[new_position] = tip_id  # Set the same ID for the new position
                
                # Store new tip in the list
                new_tip_cells.append(new_position)
                
                # Update tip_ids mapping
                self.tip_ids[new_position] = tip_id
                
                # Update path trajectory
                if new_position not in self.cell_paths:
                    self.cell_paths[new_position] = self.cell_paths[tip] + [new_position]
                
                # Set is_first_move to False for the new position
                self.is_first_move[new_position] = False
            else:
                # Check boundary conditions
                x, y = tip
                if x == 0 or x == self.grid.shape[0] - 1 or y == 0 or y == self.grid.shape[1] - 1:
                    # Stop growth if at boundary
                    self.active_tips[i] = False
                    print(f"Tip ID {tip_id} at {tip} reached boundary, stopping growth.")
                else:
                    # Deactivate if no valid neighbors
                    self.active_tips[i] = False
                    print(f"Tip ID {tip_id} at {tip} has no valid neighbors, stopping growth.")

        # Update active tips
        self.tip_cells = new_tip_cells
        self.active_tips = [True] * len(new_tip_cells)
    # def grow(self):
        new_tip_cells = [] # store new tip positions 
        
        for i, tip in enumerate(self.tip_cells):
            if not self.active_tips[i]:
                continue

            # Get the ID of this tip
            tip_id = self.tip_ids[tip]
            
            # Mark the current tip position as mature vessel in grid (1)
            self.grid[tip] = 1  # matured vasculature
            # ID remains the same in gridID
            
            # Add to mature vessel positions
            self.vas_positions.append(tip)
    
            self.vegf_field[tip] *= 0.5  # Consume VEGF at the current position
            
            empty_neighbors = self.get_valid_neighbors(tip)

            # Choose growth direction based on VEGF concentration
            if empty_neighbors:
                # Find the maximum VEGF value among all neighbors
                max_vegf_value = max(self.vegf_field[n] for n in empty_neighbors)
            
                # Find all neighbors that have this maximum VEGF value
                max_vegf_neighbors = [n for n in empty_neighbors if self.vegf_field[n] == max_vegf_value]
            
                # Randomly choose one from neighbors with maximum VEGF value
                new_position = random.choice(max_vegf_neighbors)
                
            
                # Update grid with new tip position
                self.grid[new_position] = 2  # Mark new tip position as tip (2)
                self.gridID[new_position] = tip_id  # Set the same ID for the new position
                
                # Store new tip in the list
                new_tip_cells.append(new_position)
                
                # Update tip_ids mapping
                self.tip_ids[new_position] = tip_id
                
                # Update path trajectory
                if new_position not in self.cell_paths:
                    self.cell_paths[new_position] = self.cell_paths[tip] + [new_position]
                
                # Set is_first_move to False for the new position
                self.is_first_move[new_position] = False
                
                
                print(f"Tip (ID: {tip_id}) at {tip} grew to {new_position} with VEGF value: {self.vegf_field[new_position]}")
            else:
                # Check boundary conditions
                x, y = tip
                if x == 0 or x == self.grid.shape[0] - 1 or y == 0 or y == self.grid.shape[1] - 1:
                    # Stop growth if at boundary
                    self.active_tips[i] = False
                else:
                    # Deactivate if no valid neighbors
                    self.active_tips[i] = False

        # Update active tips
        self.tip_cells = new_tip_cells
        self.active_tips = [True] * len(new_tip_cells)
        
    def get_vessel_with_id(self, vessel_id):
        """Return all positions (mature vessels and tips) that belong to a specific vessel ID"""
        positions = np.where(self.gridID == vessel_id)
        return list(zip(positions[0], positions[1]))


    def check_anastomosis(self):
        """
        Check for anastomosis events:
        1. Tip-to-tip connection (distance = 2)
        2. Tip-to-vessel connection (distance = 1)
        """
        if not hasattr(self, 'anastomosis_events'):
            self.anastomosis_events = []
        
        tips_to_remove = []  # List of tips to be removed after the loop
        
        # Check each active tip
        for i, tip in enumerate(self.tip_cells):
            if not self.active_tips[i]:
                continue
            
            tip_id = self.tip_ids[tip]
            
            # Condition 1: Check if the tip is near another tip (different ID)
            for j, other_tip in enumerate(self.tip_cells):
                if i == j:  # Skip self-comparison
                    continue
                    
                other_tip_id = self.tip_ids[other_tip]
                
                # Skip if tips belong to the same vessel
                if tip_id == other_tip_id:
                    continue
                
                # Calculate Manhattan distance between tips
                distance = abs(tip[0] - other_tip[0]) + abs(tip[1] - other_tip[1])
                
                if distance == 2:  # Tip-to-tip anastomosis condition
                    print(f"Tip-to-tip anastomosis between: Vessel {tip_id} at {tip} and Vessel {other_tip_id} at {other_tip}")
                    
                    # Mark both tips as mature vasculature
                    self.grid[tip] = 1
                    self.grid[other_tip] = 1
                    
                    # Add to vasculature positions
                    if tip not in self.vas_positions:
                        self.vas_positions.append(tip)
                    if other_tip not in self.vas_positions:
                        self.vas_positions.append(other_tip)
                    
                    # Record anastomosis event
                    self.anastomosis_events.append({
                        'type': 'tip-to-tip',
                        'positions': [tip, other_tip],
                        'vessel_ids': [tip_id, other_tip_id]
                    })
                    
                    # Mark tips for removal
                    tips_to_remove.append(tip)
                    tips_to_remove.append(other_tip)
                    
                    # Deactivate both tips
                    self.active_tips[i] = False
                    self.active_tips[j] = False
            
            # Condition 2: Check if tip is adjacent to mature vessel with different ID
            neighbors = [
                (tip[0] + 1, tip[1]),
                (tip[0] - 1, tip[1]),
                (tip[0], tip[1] + 1),
                (tip[0], tip[1] - 1),
                # Including diagonal neighbors
                (tip[0] + 1, tip[1] + 1),
                (tip[0] - 1, tip[1] - 1),       
                (tip[0] + 1, tip[1] - 1),
                (tip[0] - 1, tip[1] + 1)
            ]
            
            # Filter valid neighbors
            valid_neighbors = [
                n for n in neighbors 
                if 0 <= n[0] < self.grid_size[0] and 0 <= n[1] < self.grid_size[1]
            ]
            
            for neighbor in valid_neighbors:
                # Check if neighbor is a mature vessel with different ID
                if self.grid[neighbor] == 1 and self.gridID[neighbor] != tip_id:
                    neighbor_id = self.gridID[neighbor]
                    
                    print(f"Tip-to-vessel anastomosis: Tip ID {tip_id} at {tip} connecting to Vessel ID {neighbor_id} at {neighbor}")
                    
                    # Mark tip as mature vessel
                    self.grid[tip] = 1
                    
                    # Add to vasculature positions
                    if tip not in self.vas_positions:
                        self.vas_positions.append(tip)
                    
                    # Record anastomosis event
                    self.anastomosis_events.append({
                        'type': 'tip-to-vessel',
                        'tip_position': tip,
                        'vessel_position': neighbor,
                        'tip_id': tip_id,
                        'vessel_id': neighbor_id
                    })
                    
                    # Mark tip for removal
                    tips_to_remove.append(tip)
                    
                    # Deactivate tip
                    self.active_tips[i] = False
                    break  # Stop checking neighbors once anastomosis is found
        
        # Remove tips that underwent anastomosis
        self.tip_cells = [tip for i, tip in enumerate(self.tip_cells) if tip not in tips_to_remove]
        self.active_tips = [True] * len(self.tip_cells)
        
        return len(self.anastomosis_events) > 0