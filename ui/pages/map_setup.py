import streamlit as st # type: ignore
import sys # type: ignore
import os # type: ignore

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from environment.map_generator import generate_2d_map, get_valid_start_goal # type: ignore
from environment.terrain_generator import generate_3d_terrain # type: ignore
from plots.plot_2d import plot_2d_grid # type: ignore
from plots.terrain_preview import plot_3d_terrain # type: ignore

st.set_page_config(page_title="Map Setup", layout="wide")

from ui.session_manager import init_session # type: ignore
init_session()

st.title("🗺️ Map & Terrain Setup")
st.markdown("Create a dedicated environment for your simulations.")

col1, col2 = st.columns([1, 4])
with col1:
    st.header("Parameters")
    size = st.slider("Grid Size", 10, 50, 20, key="sz_slider")
    density = st.slider("Obstacle Density", 0.0, 0.5, 0.2, key="den_slider")
    terrain_type = st.selectbox("Terrain Type", ["flat", "hills", "steep", "mixed (flat, hills, steep)"], key="ter_sel")
    st.caption("Hills possess gradual slopes, whereas steep terrain contains sharp elevation spikes.")
    
    if terrain_type == "flat":
        st.session_state.global_phase = 1
        st.radio("Phase Engine", [1, 2], key="global_phase_fake", disabled=True, index=0)
    else:
        st.radio("Phase Engine", [1, 2], key="global_phase")
        
    phase = st.session_state.global_phase
    
    if st.button("Generate Map (or) Terrain", width="stretch"):
        st.session_state.grid = generate_2d_map(size, density)
        st.session_state.terrain = generate_3d_terrain(size, terrain_type)
        st.session_state.start, st.session_state.goal = get_valid_start_goal(st.session_state.grid)
        st.session_state.astar_path = None
        st.session_state.ann_path = None
        st.session_state.maps_generated += 1
        st.success("New Environment Configured!")

with col2:
    if st.session_state.grid is not None:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("2D Grid Snapshot")
            fig2d = plot_2d_grid(st.session_state.grid, st.session_state.start, st.session_state.goal, terrain=st.session_state.terrain)
            st.pyplot(fig2d)
        with c2:
            st.subheader("3D Terrain Perspective")
            show_3d = st.checkbox("Show 3D Preview", value=(phase == 2), key="ms_3d_toggle")
            if show_3d and st.session_state.terrain is not None:
                fig3d = plot_3d_terrain(st.session_state.terrain, st.session_state.start, st.session_state.goal, grid=st.session_state.grid)
                st.plotly_chart(fig3d, width="stretch")
            elif not show_3d:
                st.info("Toggle the checkbox above to preview 3D terrain.")
    else:
        st.info("Generate a map to see it here!")
