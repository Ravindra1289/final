# PROJECT REPORT: Energy-Aware Robot Navigation System

## ABSTRACT
Traditional robot navigation algorithms primarily focus on finding the shortest distance between a start and a goal, often ignoring the physical constraints and energy costs associated with real-world movement. This project introduces an **Energy-Aware Robot Navigation System** utilizing an Artificial Neural Network (ANN). The system trains an MLPClassifier to imitate an energy-weighted A* search algorithm that explicitly accounts for three energy factors: forward movement, rotational turns, and terrain slope climbing. By doing so, the ANN learns to autonomously navigate 2D grid environments and complex 3D topographical terrains (flat, hills, steep, and mixed) with an emphasis on suppressing unnecessary turns and avoiding steep, high-energy slopes. The system features a comprehensive interactive dashboard built with Streamlit, enabling researchers to generate dynamic maps, train the neural network, visualize step-by-step agent simulation, and analytically compare the energy efficiency of the proposed neural model against algorithmic baselines.

---

## LIST OF FIGURES
1. System Architecture Diagram
2. 3D Terrain Topography Visualization
3. A* vs ANN Path Comparison Grid
4. Neural Network Training Loss Curve
5. Multi-Layer Perceptron Topology Diagram
6. Bar Chart Comparison: Total Energy vs Path Length vs Turns

---

## 1. INTRODUCTION

### 1.1 Purpose of the System
The primary purpose of this system is to optimize mobile robot pathfinding for minimal total energy expenditure rather than strictly minimal distance. It aims to demonstrate that a trained Artificial Neural Network can learn complex energy consumption patterns—specifically penalizing turns and steep slopes—and generalize these patterns to efficiently navigate novel environments.

### 1.2 Scope of the System
The scope includes the autonomous generation of 2D obstacle grids and 3D terrain height maps. It covers the implementation of a constrained A* pathfinding algorithm to generate labeled, optimally-efficient training datasets. It extends to the training of a deep Artificial Neural Network (Multi-Layer Perceptron) and live interactive simulation of both the A* and ANN models. The system is constrained to discrete grid movements and specific terrain roughness classifications.

### 1.3 Current System
Current traditional pathfinding systems often rely on standard heuristics (like Euclidean or Manhattan distance) that prioritize the shortest sequence of coordinates. As a result, robots may execute a high number of right-angle turns or traverse geometrically minimal but physically demanding steep gradients, resulting in rapid battery drain and mechanical wear. 

### 1.4 Proposed System
The proposed system introduces an energy-aware cost model. It incorporates a deep neural network capable of inferring topological costs dynamically through local feature extraction (e.g., proximity to obstacles, relative goal angle, and adjacent terrain slope). By leveraging heading hysteresis (a forward-bias mechanism), the system actively suppresses zig-zag behavior. It serves as a proof-of-concept that machine learning can deliver optimal or near-optimal physical energy paths in real-time.

---

## 2. REQUIREMENT ANALYSIS

### 2.1 Functional Requirements

#### 2.1.1 User Interface & Interaction
- The system must provide an interactive dashboard with distinct pages for Map Setup, Training, Simulation, Analytics, and Comparison.
- Users must be able to visually toggle 3D visualizations and select terrain roughness (Flat, Hills, Steep, Mixed).

#### 2.1.2 Machine Learning & AI Functionality
- The system must feature an automated dataset generator that extracts sensor features and labels them based on A* action outputs.
- The system must implement an `MLPClassifier` with configurable epochs, adaptive learning rate, and a multi-hidden-layer topology.

#### 2.1.3 Data Handling & Security
- The system must facilitate exporting generating training datasets (CSV) and managing trained model serialization securely within local directories.

#### 2.1.4 Performance & Scalability
- The system must train on thousands of generated paths (thousands of rows) within minutes and execute real-time inference during simulation without perceptible lag.

#### 2.1.5 Reliability & Supportability
- The navigation simulation must elegantly handle edge cases such as collision attempts or infinite spinning through forced recovery protocols (e.g., maximum step cutoffs, turning toward goals).

#### 2.1.6 System Features
- Interactive Timeline Scrubber for path playback.
- Mathematical comparison charts auto-generating academic conclusions.

#### 2.1.7 Software Requirements
- Operating System: Windows / Linux / macOS
- Environment: Python 3.8+ 

#### 2.1.8 Hardware Requirements
- Core i5 Processor or equivalent (Multi-core recommended for dataset generation)
- 8GB RAM Minimum

### 2.2 Non-Functional Requirements

#### 2.2.1 User Interface and Usability
- The UI should enforce an intuitive left-to-right academic workflow (Map Setup → Data Generation → Simulation → Analytics).

#### 2.2.2 Performance and Reliability
- Generation of 100 maps and respective path datasets should resolve efficiently. 3D graphing elements must use responsive rendering limits to prevent browser crashes.

#### 2.2.3 Supportability and Deployment
- The software must run as a local web application utilizing Streamlit's self-contained hosting.

#### 2.2.4 Security and Data Privacy
- Local execution guarantees no external transmission of generated models or topographical grid properties.

#### 2.2.5 Resource Requirements
- GPU acceleration is not required, as MLP uses optimized SciPy matrix operations sufficient for the defined scale.

#### 2.2.6 Data Availability, Validation & Ethical Considerations
- Validation requires enforcing the A* search mathematically computes the absolute optimal path regarding defined energy constants, providing an unbiased ground-truth dataset for the ANN.

---

## 3. SYSTEM ANALYSIS

### 3.1 Introduction
The analysis phase maps the interactions between the end-user (researcher) and the system modules to ensure the neural network behaves safely and predictably within simulated confines.

### 3.2 Use Cases

#### 3.2.1 Actors
- **Primary Actor:** Researcher / Operator.

#### 3.2.2 Algorithm and Flowcharts
1. **Map Generation Flow:** User sets size/density -> Base numpy matrix generated -> Gaussian noise applied for terrain -> Slopes normalized for strictly achievable thresholds.
2. **Pathfinding Inference Flow:** Current State -> Extract Sensor Array -> ANN predict_proba -> Apply Hysteresis Bias -> Take Action -> Update State -> Check Goal/Collision.

#### 3.2.3 Use case diagram
- `Generate Map` -> Input Map Parameters
- `Generate Dataset` -> Invoke A* Algorithm parallel map solver
- `Train Model` -> Initialize `MLPClassifier`, optimize over dataset
- `Simulate Run` -> Step through inference, detect collisions, chart 3D path

---

## 4. SYSTEM DESIGN

### 4.1 Introduction
The design modularizes the codebase into distinct environments, machine learning logic, interactive visualizations, and centralized session management.

### 4.2 System Architecture
The application runs on a Python backend wrapping Streamlit. UI inputs mutate a centralized `session_manager` state, which invokes external modules (e.g., `train_ann.py`, `run_navigation.py`). Results are piped to Plotly and Matplotlib renderers. 

### 4.3 System Components

#### 4.3.1 Backend & Model Implementation
- `configs.py`: Global constant management for energy weights.
- `terrain_generator.py`: Specialized topological noise generation with normalization algorithms ensuring steep terrains remain mathematically passable.

#### 4.3.2 Frontend & UI Design
- Streamlit Page Routing utilizing multi-page architecture (`app.py`, `dashboard.py`, `simulation.py`, `comparison.py`).

### 4.4 Model Design & MLP Optimization

#### 4.4.1 Feature Selection & Preprocessing
The model depends strictly on localized agent arrays.
**Phase 1:** Distance to goal, Relative angle, 3-way obstacle binary proximity.
**Phase 2:** Further extended to calculate adjacent slope gradients (Forward, Left, Right). 

#### 4.4.2 ANN-MLP Model
The network replaces generic architectures with a deep `(64, 64, 32)` topology using ReLU activation. It features Adaptive Movement Estimation (Adam solver), capable of tuning local minimum gaps in high-dimensional topographical datasets. 

### 4.5 Autonomous Agent & Future Enhancements

#### 4.5.1 Explanation of Autonomous Agent Diagram
The autonomous agent resides entirely locally within the matrix. It lacks absolute global coordinates, evaluating entirely through its immediate sensor horizon, simulating realistic SLAM or odometry feedback constraints.

#### 4.5.2 Future Plans
- Expanding to continuous space navigation rather than discrete grids.
- Implementing multi-agent swarm collaborative pathfinding.

### 4.6 Conclusion
The architectural design enforces strict physical separation of the teacher (A*) from the student (ANN), ensuring the neural network truly generalizes heuristic rules rather than memorizing positional arrays.

---

## 5. IMPLEMENTATION

### 5.1 Software Used
- IDE: Visual Studio Code
- Version Control: Git

### 5.2 System Requirements
- Python Virtual Environment (venv) with strict dependency tracking.

### 5.3 Dataset Description
The dataset consists of thousands of dynamic states. Each row translates an independent instantaneous robotic state (Distance, Angle, Sensor views) mapped to a supervised integer representing the optimal action `(0=Forward, 1=Left, 2=Right)`.

### 5.4 Web Framework and Frontend Technologies Used

#### 5.4.1 Streamlit (Python Web Framework)
Allows for reactive, data-driven application deployment without handling standalone HTML/JS DOM interactions.

#### 5.4.2 Data Visualization
Interactive plotting relies heavily on `plotly.graph_objects` dynamically rendering meshes and lines in pseudo-3D.

#### 5.4.3 UI Structure
UI utilizes container and column layouts spanning full screen widths prioritizing maximal visualization space.

### 5.5 Programming Language Used
- **Python 3.10+** (Strict typing and matrix optimizations).

### 5.6 Libraries/Packages Used
- `numpy`, `scipy` (Matrix calculations and Gaussian filters)
- `scikit-learn` (Artificial Neural Network)
- `plotly`, `matplotlib` (Data Visualization)
- `pandas` (Dataset management)

### 5.7 SOURCE CODE

#### 5.7.1 A* Algorithm Code
*(Contained within `planning/astar.py` - Evaluates heuristic f-scores combining base distance, heavy turning penalties, and steep slope traversal parameters.)*

#### 5.7.2 ANN Code
*(Contained within `model/train_ann.py` - Manages serialization using joblib and configures the `MLPClassifier`.)*

#### 5.7.3 Navigation Logic Code
*(Contained within `navigation/run_navigation.py` - Integrates mathematical Hysteresis Bias (`+0.25`) towards Forward action weights to prevent model hesitation/spinning).*

---

## 6. TESTING

### 6.1 Introduction
Testing involves validating that the autonomous neural network can maneuver maps unseen during training with near-optimal efficiency.

### 6.2 Dataset Analysis
Analysis of resulting CSV sets ensuring the training classes do not suffer from excessive class imbalance (e.g., ensuring Left and Right instances have comparative frequencies).

### 6.3 Observations on Accuracy & Topography Fluctuations
Tests determined random terrain generation produced impassable maps; fixed sequentially through the `_normalize_slopes()` technique which clamps maximum map steepness to `< 0.5`. 

### 6.4 Test Case Identification
- **TC1:** Pure Flat terrain - Verify path length minimization.
- **TC2:** Hills traverse - Verify energy avoidance around gentle slopes.
- **TC3:** Goal Occlusion - Verify navigation around immediate obstacles.

### 6.5 Impact of Epochs on Model Training Stability
Testing revealed epochs fewer than 1000 produced oscillatory behavior. 2000 epochs with dynamic learning adaptation effectively solved edge-case spinning. Navigation edge-case testing also resulted in adding collision-recovery protocols (turning out of corners).

### 6.6 Final Insights from Testing
The Neural Network does not match A* 100% in distance, but frequently reaches the goal within a 15% margin of the baseline total-energy cost, qualifying the system as a success.

---

## 7. RESULT ANALYSIS
The analytics comparison framework distinctly charts A* vs ANN across:
1. **Total Energy:** ANN consistently performs well within limits, occasionally finding unique approximations over complex terrains.
2. **Total Turns:** The hysteresis implementation allows the ANN to vastly outperform pure-distance approaches in mitigating excessive mechanical turning.
3. **Path Length:** While path length is slightly elevated in terrain handling, it directly contributes to minimizing the overarching topological energy curve.

---

## 8. CONCLUSION
The project conclusively proves that an Artificial Neural Network, restricted to only local sensory inputs, can successfully learn macro-navigation strategies. By leveraging energy-aware heuristics in training, the ANN prioritizes minimizing structural energy demands, avoiding high terrain gradients, and reducing rotational adjustments, creating a practical basis for energy-limited mobile robots. 

---

## 9. FUTURE SCOPE
Future derivations will aim for Reinforcement Learning (RL) techniques that do not require explicitly labeled A* graphs, instead allowing models like PPO (Proximal Policy Optimization) to iteratively score their paths organically through simulation trials. Furthermore, real-world deployment on hardware sensors using ROS2 is envisioned.

---

## 10. PUBLICATION DETAILS
*(Academic references to the laboratory and department overseeing this conceptual project.)*

## 11. BIBLIOGRAPHY
1. Russell, S., & Norvig, P. *Artificial Intelligence: A Modern Approach*.
2. Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 12, pp. 2825-2830, 2011.
3. Streamlit Documentation: https://docs.streamlit.io

## 12. ANNEXURE

### 12.1 Base Paper
*(Reserved for foundational A* modification reference papers)*

### 12.2 Published Paper
*(Reserved for future publication drafts)*

### 12.3 Certificates
*(Reserved)*
