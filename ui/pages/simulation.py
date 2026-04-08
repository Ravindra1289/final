import streamlit as st # type: ignore
import sys # type: ignore
import os # type: ignore
import numpy as np # type: ignore

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from planning.astar import astar_path # type: ignore
from navigation.run_navigation import run_ann_navigation # type: ignore
from dataset.dataset_generator import extract_features # type: ignore
from plots.plot_2d import plot_2d_grid # type: ignore
from plots.terrain_preview import plot_3d_terrain # type: ignore
from ui.session_manager import init_session # type: ignore
init_session()

st.set_page_config(page_title="Simulation", layout="wide")
st.title("🏎️ Fullscreen Simulation Engine")

if st.session_state.grid is None:
    st.error("No environment located in memory. Go to Map Setup to generate one!")
else:
    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        if st.button("Simulate A* (Teacher)", width="stretch"):
            path = astar_path(st.session_state.grid, st.session_state.terrain, st.session_state.start, st.session_state.goal, phase=st.session_state.global_phase)
            if path:
                st.session_state.astar_path = path
                st.success("Target heuristic routed.")
            else:
                st.error("No valid route.")
    with c2:
        if st.button("Simulate ANN (Student)", width="stretch"):
            if st.session_state.model:
                expected_f = 5 if st.session_state.global_phase == 1 else 8
                if hasattr(st.session_state.model, 'n_features_in_') and st.session_state.model.n_features_in_ != expected_f:
                    st.error(f"Model mismatch! Model expects {st.session_state.model.n_features_in_} features. Retrain under Phase {st.session_state.global_phase}.")
                else:
                    try:
                        path, status = run_ann_navigation(st.session_state.grid, st.session_state.terrain, st.session_state.start, st.session_state.goal, st.session_state.model, phase=st.session_state.global_phase)
                        st.session_state.ann_path = path
                        st.session_state.ann_status = status
                        if status == "success":
                            st.success("ANN-only success")
                        elif status == "success_with_rescue":
                            st.warning("ANN success with rescue")
                        else:
                            st.error("ANN failed")
                    except Exception as e:
                        st.error(f"Execution Error: {e}")
            else:
                st.error("Model uninitialized.")
                
    st.markdown("---")
    sc1, sc2 = st.columns([3, 1])
    
    astar = st.session_state.get("astar_path")
    ann = st.session_state.get("ann_path")
    
    max_len = max(len(astar) if astar else 0, len(ann) if ann else 0)
    
    with sc1:
        if max_len > 0:
            st.markdown("### ⏱️ Timeline Scrubber")
            step = st.slider("Playback Step", 1, max_len, max_len, key="playback")
            render_a = astar[:step] if astar else None
            render_n = ann[:step] if ann else None
            
            active_path = render_n if render_n else (render_a if render_a else None)
            current_state = active_path[-1] if active_path else (st.session_state.start[0], st.session_state.start[1], 0)
        else:
            render_a = None
            render_n = None
            current_state = None
            
        t1, t2 = st.tabs(["2D Overhead", "3D Topographic"])
        with t1:
            fig2d = plot_2d_grid(st.session_state.grid, st.session_state.start, st.session_state.goal, render_a, render_n, terrain=st.session_state.terrain)
            st.pyplot(fig2d)
        with t2:
            if st.session_state.terrain is not None:
                fig3d = plot_3d_terrain(st.session_state.terrain, st.session_state.start, st.session_state.goal, render_a, render_n, grid=st.session_state.grid)
                st.plotly_chart(fig3d, width="stretch")
                
    with sc2:
        st.markdown("### 📻 Sensor HUD")
        if max_len > 0 and current_state:
            st.write(f"**Position:** ({current_state[0]}, {current_state[1]})")
            st.write(f"**Heading:** {current_state[2]}")
            
            feat = extract_features(st.session_state.grid, st.session_state.terrain, current_state, st.session_state.goal, phase=st.session_state.global_phase)
            if st.session_state.global_phase == 1:
                st.json({"Distance": round(feat[0],2), "Angle_Offset": round(feat[1],2), "Obs_F": feat[2], "Obs_L": feat[3], "Obs_R": feat[4]})
            else:
                st.json({"Distance": round(feat[0],2), "Angle_Offset": round(feat[1],2), "Slope_F": round(feat[2],2), "Slope_L": round(feat[3],2), "Slope_R": round(feat[4],2), "Obs_F": feat[5], "Obs_L": feat[6], "Obs_R": feat[7]})
            
            if step < max_len:
                st.info("Scrub right to advance time.")
            else:
                st.success("Target Reached or Terminated.")
        else:
            st.warning("Generate a path to use the HUD.")
