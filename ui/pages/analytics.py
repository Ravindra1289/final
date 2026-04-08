import streamlit as st # type: ignore
import sys # type: ignore
import os # type: ignore
import plotly.graph_objects as go # type: ignore
import numpy as np # type: ignore

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from ui.session_manager import init_session # type: ignore
init_session()

st.set_page_config(page_title="Analytics", layout="wide")
st.title("📊 Model Insights & Energy Analytics")

c1, c2 = st.columns(2)

with c1:
    st.markdown("### 📈 Neural Network Architecture")
    model = st.session_state.get("model")
    if model:
        st.success("Model Successfully Hooked")
        
        colA, colB, colC = st.columns(3)
        colA.metric("Input Nodes", model.n_features_in_)
        colB.metric("Hidden Layers", str(model.hidden_layer_sizes))
        colC.metric("Output Nodes", model.n_outputs_)
        
        st.metric("Total Network Layers", model.n_layers_)
        st.metric("Convergence Epochs", model.n_iter_)
        if hasattr(model, 'loss_curve_'):
            st.line_chart(model.loss_curve_, height=200)
            st.caption("Training Loss Curve over Epochs (Must trend downwards)")
        st.markdown(f"**Output Categories:** `{list(model.classes_)}` *(0=Forward, 1=Left, 2=Right)*")
        st.markdown(f"**Solver:** `{model.solver}` | **Activation:** `{model.activation}`")
    else:
        st.warning("No Neural Network loaded in memory.")

with c2:
    st.markdown("### 🌋 Topographic Traversability Heatmap")
    grid = st.session_state.get("grid")
    terrain = st.session_state.get("terrain")
    phase = st.session_state.get("global_phase", 2)
    
    if grid is not None:
        if phase == 2 and terrain is not None:
            # Generate local slope penalty heatmap a rough estimation of 'pain to travel'
            dy, dx = np.gradient(terrain)
            slope_magnitude = np.sqrt(dy**2 + dx**2)
            # Add severe visual penalty if there's a hard wall block (value = 1)
            heat = np.where(grid == 1, np.max(slope_magnitude)*2 if np.max(slope_magnitude) > 0 else 5.0, slope_magnitude)
            
            fig = go.Figure(data=go.Heatmap(z=heat, colorscale='Reds'))
            fig.update_layout(title="Red = Severe Energy Cost Constraints", height=400)
            st.plotly_chart(fig, width="stretch")
            st.caption("This explicit mathematical gradient proves EXACTLY why A* and ANN dodge certain paths. Sharp red pixels trigger massive `SLOPE_COST` penalty equations.")
        else:
            fig = go.Figure(data=go.Heatmap(z=grid, colorscale='Greys'))
            fig.update_layout(title="Obstacle Configuration Map (2D Phase 1)", height=400)
            st.plotly_chart(fig, width="stretch")
            st.caption("In 2D mode, energy costs are strictly based on absolute distance and rotation.")
    else:
        st.info("Generate a Map to visualize the underlying physical Energy Equations.")
