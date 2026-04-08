import streamlit as st # type: ignore
import sys # type: ignore
import os # type: ignore

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.train_ann import load_model # type: ignore

def init_session():
    # Unified Global Phase State (Defaults to Phase 2: 3D Slopes)
    if "global_phase" not in st.session_state: st.session_state.global_phase = 2

    if "grid" not in st.session_state: st.session_state.grid = None
    if "terrain" not in st.session_state: st.session_state.terrain = None
    if "start" not in st.session_state: st.session_state.start = None
    if "goal" not in st.session_state: st.session_state.goal = None
    if "astar_path" not in st.session_state: st.session_state.astar_path = None
    if "ann_path" not in st.session_state: st.session_state.ann_path = None
    if "ann_status" not in st.session_state: st.session_state.ann_status = None
    
    if "model" not in st.session_state: 
        st.session_state.model = load_model(phase=st.session_state.global_phase)
    else:
        # Cross-Check Model Dimension to ensure it matches current UI Phase Request
        expected_features = 5 if st.session_state.global_phase == 1 else 8
        if st.session_state.model is not None and getattr(st.session_state.model, 'n_features_in_', 0) != expected_features:
            st.session_state.model = load_model(phase=st.session_state.global_phase)
    
    if "last_accuracy" not in st.session_state: st.session_state.last_accuracy = "--"
    if "maps_generated" not in st.session_state: st.session_state.maps_generated = 0
    if "dataset_rows" not in st.session_state: st.session_state.dataset_rows = "--"
