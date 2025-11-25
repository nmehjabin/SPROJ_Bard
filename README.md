# Modeling the Extra-Cellular Matrix in Cancer Angiogenesis

## Methods

This repository implements an agent-based model (ABM) of angiogenesis, extending prior work from the Norton Lab on triple-negative breast cancer. Previous simulations modeled tumor cells, vasculature, immune cells, and the tumor microenvironment, but did not explicitly represent extracellular matrix (ECM) structure. Here, we incorporate ECM gradients into an angiogenesis model to study how ECM distribution and VEGF (vascular endothelial growth factor) jointly shape endothelial tip cell migration and network morphology.

We implement three models:

- **VEGF-only model** – vessel growth driven purely by VEGF gradients  
- **ECM-only gradient model** – vessel growth driven purely by ECM structure  
- **Merged ECM + VEGF model** – vessel growth driven by both VEGF and ECM

All models operate on a 2D grid where tip cells move stepwise, form new vessel segments, and can undergo anastomosis (fusion) to form connected vascular networks. Simulations are controlled by a main driver script that sets parameters, initializes grids, runs the stepwise update loop, and saves outputs (images, TSV/CSV data, summary statistics).

---

### VEGF-only Model

The VEGF-only model uses a 50×50 dual-grid system:

- A **cell grid** stores cell type (`0 = empty`, `1 = mature vessel`, `2 = tip cell`).
- A **gridID** array stores unique vessel IDs so each branching sprout can be tracked independently.

Tip cells are initialized at specified positions and stored in a path dictionary, with each tip assigned a unique ID and tracked in an `active_tips` list. A VEGF field is defined over the same grid, with initial values randomized between 0 and 1. At each growth step:

- Tip cells **consume VEGF** at their location (e.g., reduce by 50%).
- The VEGF field undergoes **global decay** (e.g., 5%) to mimic natural degradation.
- Valid neighbors are computed with **directional persistence**:  
  - The first step can move to any of the 8 surrounding cells.  
  - Later steps favor “forward” neighbors based on the previous movement vector.
- From the valid neighbors, the model selects locations with **maximum VEGF**, breaking ties at random.

The **growth step** alternates with an **anastomosis check**:

- **Tip–tip anastomosis**: tips from different vessels within Manhattan distance ≤ 2.
- **Tip–vessel anastomosis**: tips adjacent to mature segments of a different vessel ID.

When anastomosis occurs, tips are converted to mature vessels, deactivated, and the event is logged (positions, IDs). The simulation runs for a fixed number of steps (e.g., 200), alternating growth and anastomosis phases. At the end, the model saves vessel paths, VEGF values, displacements, anastomosis events, and visualization frames for quantitative analysis.

---

### ECM-only Gradient Model

The ECM-only model replaces VEGF with an **ECM density field** defined over the grid and explores how ECM structure alone shapes tip cell migration. Three ECM configurations are supported:

- **Uniform gradient** – constant ECM density (e.g., 0.5) across the grid  
- **Linear gradient** – ECM varies along the x-direction:  
  `ECM(x) = slope * x + intercept`  
- **Radial gradient** – ECM varies with distance from center:  
  `ECM(r) = center_value * exp(-decay_rate * distance)`

Tip cells are initialized and tracked similarly to the VEGF model. Movement is determined by:

1. **ECM sensing** – tips prefer **lower ECM density** regions (easier to invade).  
2. **Directional persistence** – forward-biased neighbor selection based on prior movement.  
3. **Decision rule** – from valid neighbors, choose the location with **lowest ECM**, breaking ties randomly.

Path histories are stored for each tip, including coordinates, activity status, and whether the first move has occurred. Anastomosis detection is performed similarly (tip–tip and tip–vessel, using distance thresholds), and runs on alternating steps with movement. Outputs include:

- TSV files with positions, ECM values, displacements, and anastomosis counts  
- Matplotlib visualizations of ECM gradients, paths, and anastomosis events  

This setup enables systematic comparison of how ECM heterogeneity (uniform vs linear vs radial) affects tip cell migration and network topology.

---

### Merged ECM + VEGF Model

The merged model combines both **ECM density** and **VEGF concentration** to drive tip cell movement in a more realistic angiogenesis environment. The simulation space contains:

- An **ECM grid** generated as in the ECM-only model (uniform, linear, or radial).
- A **VEGF grid** with values randomly initialized in [0, 1].

For each tip cell at each movement step:

1. Identify valid neighbors using **directional constraints** (full 8 neighbors on first step, forward-biased subset afterward).
2. Evaluate **ECM** (prefer low density) and **VEGF** (prefer high concentration) at each valid neighbor.
3. If one neighbor has both **lowest ECM and highest VEGF**, move there.  
   Otherwise, choose from among low-ECM and high-VEGF candidates and compute an effective next position (e.g., via averaging or tie-breaking logic).
4. Update the tip’s position, path history, and grid state.

Anastomosis detection (tip–tip and tip–vessel) is implemented as in the other models, with tip deactivation and event logging upon fusion. The simulation alternates movement and anastomosis phases until:

- A maximum step count is reached, or  
- All tips are inactive.

The current implementation focuses on **correct data generation and cell behavior**, with some visualization components still under refinement. Parameters such as gradient type, grid size, tip initialization, and movement constraints are configurable, enabling exploration of how combined ECM and VEGF cues shape vessel formation and network connectivity.
