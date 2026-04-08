import streamlit as st # type: ignore
import sys # type: ignore
import os # type: ignore

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from model.train_ann import load_model # type: ignore

st.set_page_config(page_title="Dashboard", layout="wide")

# Init checks
from ui.session_manager import init_session # type: ignore
init_session()

st.title("🏠 Energy-Aware Robot Navigation — Dashboard")

st.markdown("""
Welcome to the **Energy-Aware Robot Navigation System**. This application demonstrates how an 
Artificial Neural Network (ANN) can learn to navigate a 2D/3D environment by imitating the 
energy-optimal A* search algorithm, while explicitly **minimizing total energy cost** — 
not just path length.
""")

st.markdown("---")

# --- Status Metrics ---
st.markdown("### 📊 System Status")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Maps Generated", st.session_state.maps_generated)
col2.metric("ANN Model Status", "Ready ✅" if st.session_state.model else "Not Trained ❌")
col3.metric("Dataset Size", st.session_state.dataset_rows)
col4.metric("Last Accuracy", f"{st.session_state.last_accuracy}%" if st.session_state.last_accuracy != "--" else "--")

st.markdown("---")

# --- Energy Cost Model ---
c1, c2 = st.columns(2)
with c1:
    st.markdown("### ⚡ Energy Cost Model")
    st.markdown("""
| Action | Energy Cost | Description |
|---|---|---|
| **Move Forward** | `1.0` unit | Base cost per cell traversed |
| **Rotate 90°** | `5.0` units | Penalty for turning left or right |
| **Climb Slope** | `10.0 × Δh` units | Proportional to height difference |
| **Steep Wall** | `∞ (blocked)` | Slopes > `MAX_SLOPE (0.5)` are impassable |

> **Key Insight:** The A* algorithm minimizes **total energy**, not distance. 
> It may take a longer path to avoid costly turns and steep slopes!
""")

with c2:
    st.markdown("### 🧠 ANN Architecture")
    st.markdown("""
| Property | Value |
|---|---|
| **Type** | Multi-Layer Perceptron (MLPClassifier) |
| **Hidden Layers** | `(64, 64, 32)` neurons |
| **Activation** | ReLU |
| **Solver** | Adam (Adaptive Moment Estimation) |
| **Learning Rate** | Adaptive |
| **Max Iterations** | 2,000 epochs |
| **Validation Split** | 15% holdout |
| **Heading Hysteresis** | `+0.25` bias toward Forward action |
""")

st.markdown("---")

# --- Workflow Guide ---
st.markdown("### 🔄 Step-by-Step Workflow")
st.markdown("""
1. **🗺️ Map Setup** → Set grid size, obstacle density & terrain type → Click *Generate Map (or) Terrain*
2. **🧠 Training** → Click *Build Dataset* (generates A* training data) → Click *Train MLP Neural Network*
3. **🚀 Simulation** → Click *Simulate A* (Teacher)* → Click *Simulate ANN (Student)* → Use the Timeline Scrubber
4. **📈 Analytics** → Inspect the neural network topology, loss curve & traversability heatmap
5. **📊 Comparison** → View individual bar charts (Length, Energy, Turns) & read the academic conclusion
""")

st.markdown("---")

# --- Terrain Types ---
st.markdown("### 🏔️ Terrain Types")
tc1, tc2, tc3, tc4 = st.columns(4)
with tc1:
    st.markdown("**Flat**")
    st.markdown("Zero elevation. Pure 2D obstacle navigation. No slope cost.")
with tc2:
    st.markdown("**Hills**")
    st.markdown("Gentle, smooth rolling elevation. Moderate slope costs.")
with tc3:
    st.markdown("**Steep**")
    st.markdown("Sharp, jagged spikes. High energy penalties. Some areas impassable.")
with tc4:
    st.markdown("**Mixed**")
    st.markdown("Combination of gentle hills with sharp spike overlays.")

st.markdown("---")
st.caption("Navigate using the sidebar to dive into each module. | Version 2.0 | March 2026")
