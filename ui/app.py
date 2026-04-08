import streamlit as st # type: ignore
import sys # type: ignore
import os # type: ignore
import time # type: ignore
import pandas as pd # type: ignore

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.map_generator import generate_2d_map, get_valid_start_goal # type: ignore
from environment.terrain_generator import generate_3d_terrain # type: ignore
from planning.astar import astar_path # type: ignore
from dataset.dataset_generator import generate_dataset # type: ignore
from model.train_ann import train_and_save_model, load_model # type: ignore
from navigation.run_navigation import run_ann_navigation # type: ignore
from evaluation.metrics import compute_metrics # type: ignore
from plots.plot_2d import plot_2d_grid, plot_2d_grid_interactive # type: ignore
from plots.terrain_preview import plot_3d_terrain # type: ignore

st.set_page_config(page_title="Energy-Aware Robot", layout="wide")

from ui.session_manager import init_session # type: ignore
init_session()

st.title("Energy-Aware Robot Navigation System")

col1, col2 = st.columns([1, 3])

with col1:
    st.header("Map Configuration")
    size = st.slider("Map Size", 10, 50, 20)
    density = st.slider("Obstacle Density", 0.0, 0.5, 0.2)
    terrain_type = st.selectbox("Terrain Type", ["flat", "hills", "steep", "mixed (flat, hills, steep)"])
    st.caption("Hills possess gradual slopes, whereas steep terrain contains sharp elevation spikes.")

    if terrain_type == "flat":
        st.session_state.global_phase = 1
    else:
        st.session_state.global_phase = 2

    phase = st.session_state.global_phase

    st.subheader("Point Placement Mode")
    placement_mode = st.radio(
        "Select point to place:",
        ["Start", "Goal"],
        horizontal=True,
        help="Choose whether to place Start or Goal point, then click on map or enter coordinates manually."
    )

    st.subheader("Manual Coordinate Entry")
    col_y, col_x, col_set = st.columns([1, 1, 1])
    with col_y:
        default_y = size//4 if placement_mode == "Start" else 3*size//4
        if placement_mode == "Start" and st.session_state.start is not None:
            default_y = st.session_state.start[0]
        elif placement_mode == "Goal" and st.session_state.goal is not None:
            default_y = st.session_state.goal[0]
        manual_y = st.number_input(f"{placement_mode} Y", 0, size-1, default_y, key=f"manual_{placement_mode.lower()}_y")
    with col_x:
        default_x = size//4 if placement_mode == "Start" else 3*size//4
        if placement_mode == "Start" and st.session_state.start is not None:
            default_x = st.session_state.start[1]
        elif placement_mode == "Goal" and st.session_state.goal is not None:
            default_x = st.session_state.goal[1]
        manual_x = st.number_input(f"{placement_mode} X", 0, size-1, default_x, key=f"manual_{placement_mode.lower()}_x")
    with col_set:
        if st.button(f"Set {placement_mode}", key=f"set_{placement_mode.lower()}"):
            new_pos = (manual_y, manual_x)
            if (0 <= manual_y < size and 0 <= manual_x < size):
                if st.session_state.grid is not None and st.session_state.grid[manual_y, manual_x] == 1:
                    st.warning("Cannot place point on obstacle - clearing obstacle first")
                    st.session_state.grid[manual_y, manual_x] = 0

                if placement_mode == "Start":
                    st.session_state.start = new_pos
                else:
                    st.session_state.goal = new_pos

                st.session_state.astar_path = None
                st.session_state.ann_path = None
                st.success(f"{placement_mode} position updated to ({manual_y}, {manual_x})")
                st.rerun()
            else:
                st.error("Coordinates out of bounds")

    st.subheader("Current Positions")
    col_start, col_goal = st.columns(2)
    with col_start:
        if st.session_state.start is not None:
            st.write(f"**Start:** ({st.session_state.start[0]}, {st.session_state.start[1]})")
        else:
            st.write("**Start:** Not set")
    with col_goal:
        if st.session_state.goal is not None:
            st.write(f"**Goal:** ({st.session_state.goal[0]}, {st.session_state.goal[1]})")
        else:
            st.write("**Goal:** Not set")

    if st.session_state.start is not None and st.session_state.goal is not None:
        positions_valid = (
            st.session_state.start != st.session_state.goal and
            0 <= st.session_state.start[0] < size and 0 <= st.session_state.start[1] < size and
            0 <= st.session_state.goal[0] < size and 0 <= st.session_state.goal[1] < size
        )
    else:
        positions_valid = False

    if positions_valid:
        st.success("Valid start and goal positions")
    else:
        st.error("Invalid positions - start and goal must be different and within bounds")

    if st.button("Generate Map & Terrain", width="stretch", disabled=not positions_valid):
        st.session_state.grid = generate_2d_map(size, density)
        st.session_state.terrain = generate_3d_terrain(size, terrain_type)

        if st.session_state.grid[st.session_state.start] == 1:
            st.session_state.grid[st.session_state.start] = 0
        if st.session_state.grid[st.session_state.goal] == 1:
            st.session_state.grid[st.session_state.goal] = 0

        st.session_state.astar_path = None
        st.session_state.ann_path = None
        st.session_state.ann_status = None

    if st.button("Run A* Algorithm", width="stretch"):
        if st.session_state.grid is not None:
            path = astar_path(st.session_state.grid, st.session_state.terrain, st.session_state.start, st.session_state.goal, phase=phase)
            if path is None:
                st.error("No valid A* path found! Please regenerate map.")
            else:
                st.session_state.astar_path = path
                st.success("A* Path found!")
        else:
            st.error("Generate a map first.")

    scenarios_for_training = st.slider("Training Start/Goal Scenarios", 5, 200, 50)
    if st.button("Train ANN Model", width="stretch"):
        if st.session_state.grid is None or st.session_state.terrain is None:
            st.error("Generate one map first.")
        else:
            with st.spinner("Generating samples from the current map..."):
                X, y, path = generate_dataset(
                    phase=phase,
                    scenarios_per_map=scenarios_for_training,
                    grid=st.session_state.grid,
                    terrain=st.session_state.terrain,
                )
            if len(X) == 0:
                st.error("Dataset generation produced 0 samples on this map. Try different positions or regenerate the map.")
            else:
                st.session_state.dataset_rows = len(X)
                st.success(f"Dataset generated from one map! (Rows: {len(X)})")

                with st.spinner("Training Model..."):
                    model, score, _ = train_and_save_model(X, y, phase)
                    st.session_state.model = model
                    st.session_state.last_accuracy = round(score * 100, 2)
                st.success(f"Model trained! Accuracy: {score*100:.2f}%")

    if st.button("Run ANN Navigation", width="stretch"):
        if st.session_state.grid is None:
            st.error("Generate a map first.")
        elif st.session_state.model is None:
            st.error("Train or load a model first.")
        else:
            expected_f = 5 if phase == 1 else 8
            if hasattr(st.session_state.model, 'n_features_in_') and st.session_state.model.n_features_in_ != expected_f:
                st.error(f"Model mismatch! Model expects {st.session_state.model.n_features_in_} features. Please retrain under Phase {phase}.")
            else:
                try:
                    path, status = run_ann_navigation(st.session_state.grid, st.session_state.terrain, st.session_state.start, st.session_state.goal, st.session_state.model, phase=phase)
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

with col2:
    tab1, tab2, tab3 = st.tabs(["Visualizations", "Metrics Comparison", "Info"])

    with tab1:
        if st.session_state.grid is not None:
            cA, cB = st.columns(2)
            with cA:
                if st.session_state.start is not None and st.session_state.goal is not None:
                    fig2d = plot_2d_grid_interactive(
                        st.session_state.grid,
                        st.session_state.start,
                        st.session_state.goal,
                        st.session_state.astar_path,
                        st.session_state.ann_path,
                        terrain=st.session_state.terrain
                    )
                    st.plotly_chart(fig2d, width='stretch', key="map_plot")
                    st.caption("Use the coordinate inputs above to place Start/Goal points. The map updates immediately!")
                else:
                    st.info("Set both Start and Goal positions to visualize the map")

            with cB:
                show_3d = st.checkbox("Show 3D Preview", value=(phase == 2), key="app_3d_toggle")
                if show_3d and st.session_state.terrain is not None and st.session_state.start is not None and st.session_state.goal is not None:
                    fig3d = plot_3d_terrain(st.session_state.terrain, st.session_state.start, st.session_state.goal, st.session_state.astar_path, st.session_state.ann_path, grid=st.session_state.grid)
                    st.plotly_chart(fig3d, width="stretch")
                elif not show_3d:
                    st.info("Toggle the checkbox above to preview 3D terrain.")
                elif st.session_state.start is None or st.session_state.goal is None:
                    st.info("Set Start and Goal positions to view 3D terrain.")

    with tab2:
        if st.session_state.astar_path or st.session_state.ann_path:
            metrics_a = compute_metrics(st.session_state.astar_path, "success", st.session_state.terrain, phase) if st.session_state.astar_path else None
            ann_status = st.session_state.ann_status or ("success" if (st.session_state.ann_path and st.session_state.ann_path[-1][:2] == st.session_state.goal) else "failed")
            metrics_n = compute_metrics(st.session_state.ann_path, ann_status, st.session_state.terrain, phase) if st.session_state.ann_path else None

            data = []
            if metrics_a:
                data.append({"Algorithm": "A*", "Total Energy": metrics_a['total_energy'], "Path Length": metrics_a['path_length'], "Turns": metrics_a['turns'], "Status": "Optimal (A*)"})
            if metrics_n:
                status_label = {
                    "success": "ANN-only success",
                    "success_with_rescue": "ANN success with rescue",
                }.get(metrics_n['status'], "ANN failed")
                data.append({"Algorithm": "ANN", "Total Energy": metrics_n['total_energy'], "Path Length": metrics_n['path_length'], "Turns": metrics_n['turns'], "Status": status_label})

            if data:
                df = pd.DataFrame(data)
                st.table(df)
                st.bar_chart(df.set_index('Algorithm')[['Total Energy', 'Path Length', 'Turns']])

    with tab3:
        st.markdown('''### Energy Model
- Forward distance = 1 unit per cell
- Rotating 90 degrees = 5 units
- Moving uphill = 10 units per height diff
- Slopes higher than MAX_SLOPE (0.5) behave like pure walls!''')
