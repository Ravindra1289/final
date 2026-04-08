import streamlit as st # type: ignore
import sys
import os
import pandas as pd # type: ignore
import numpy as np # type: ignore

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dataset.dataset_generator import generate_dataset # type: ignore
from model.train_ann import train_and_save_model # type: ignore
from ui.session_manager import init_session # type: ignore
init_session()

st.set_page_config(page_title="Model Training", layout="wide")

st.title("Neural Network Training Center")

colA, colB = st.columns([1, 2])
with colA:
    st.markdown("### Build Dataset From Current Map")
    st.markdown("Use the one active map and generate many different start/goal scenarios on that same environment.")
    scenarios = st.slider("Number of Start/Goal Scenarios", 5, 200, 50)
    phase_sel = st.session_state.global_phase
    st.info(f"Using Phase {phase_sel} (set on Map Setup page)")

    if st.button("Build Dataset", width="stretch"):
        if st.session_state.grid is None or st.session_state.terrain is None:
            st.error("Generate one map first on the Map Setup page.")
        else:
            with st.spinner("Extracting features and labeling from the current map..."):
                X, y, path = generate_dataset(
                    phase=phase_sel,
                    scenarios_per_map=scenarios,
                    grid=st.session_state.grid,
                    terrain=st.session_state.terrain,
                )
                st.session_state.temp_X = X
                st.session_state.temp_y = y
                st.session_state.dataset_rows = len(X)
            if len(X) == 0:
                st.error("Dataset generation produced 0 samples on this map. Try different start/goal scenarios or regenerate the map.")
            else:
                st.success(f"Built {len(X)} labeled data points from one map!")

    st.markdown("---")
    if st.button("Train MLP Neural Network", width="stretch"):
        if "temp_X" in st.session_state and len(st.session_state.temp_X) > 0:
            with st.spinner("Fitting Adam optimizer to Multilayer Perceptron..."):
                model, score, _ = train_and_save_model(st.session_state.temp_X, st.session_state.temp_y, phase=phase_sel)
                st.session_state.model = model
                st.session_state.last_accuracy = round(score * 100, 2)
            st.success(f"Network convergence successful! Validated Accuracy: {st.session_state.last_accuracy}%")
        else:
            st.error("Missing Dataset! Please build the dataset above first.")

with colB:
    if "temp_y" in st.session_state:
        st.markdown("### Internal Class Distribution")
        fwd = int(np.sum(np.array(st.session_state.temp_y) == 0))
        lft = int(np.sum(np.array(st.session_state.temp_y) == 1))
        rgt = int(np.sum(np.array(st.session_state.temp_y) == 2))
        df_dist = pd.DataFrame({"Actions": ["Forward", "Left", "Right"], "Samples": [fwd, lft, rgt]})
        st.bar_chart(df_dist.set_index("Actions"), color="#ffaa00")

        st.markdown("### Raw Dataset Transparency File")
        num_features = st.session_state.temp_X.shape[1]
        cols = ["Distance", "Rel_Angle", "Obs_F", "Obs_L", "Obs_R"] if num_features == 5 else ["Distance", "Rel_Angle", "Slope_F", "Slope_L", "Slope_R", "Obs_F", "Obs_L", "Obs_R"]

        df_dataset = pd.DataFrame(st.session_state.temp_X, columns=cols)
        action_map = {0: "Forward", 1: "Left", 2: "Right"}
        df_dataset["Label_Action"] = [action_map[val] for val in st.session_state.temp_y]

        st.dataframe(df_dataset, height=300)

        csv_data = df_dataset.to_csv(index=False).encode("utf-8")
        st.download_button(label="Download Dataset (CSV)", data=csv_data, file_name=f"energy_robot_dataset_phase_{phase_sel}.csv", mime="text/csv")
    else:
        st.info("Build the dataset from the current map to inspect the training data here.")
