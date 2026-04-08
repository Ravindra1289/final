import streamlit as st # type: ignore
import sys # type: ignore
import os # type: ignore
import pandas as pd # type: ignore

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from evaluation.metrics import compute_metrics # type: ignore

st.set_page_config(page_title="Comparison", layout="wide")
st.title("Heuristic vs AI Head-to-Head")

from ui.session_manager import init_session # type: ignore
init_session()

STATUS_LABELS = {
    "success": "ANN-only success",
    "success_with_rescue": "ANN success with rescue",
}

if st.session_state.get("astar_path") or st.session_state.get("ann_path"):
    metrics_a = compute_metrics(
        st.session_state.astar_path,
        "success",
        st.session_state.terrain,
        phase=st.session_state.global_phase,
    ) if st.session_state.get("astar_path") else None

    ann_path = st.session_state.get("ann_path")
    status_ann = st.session_state.get("ann_status") or ("success" if (ann_path and ann_path[-1][:2] == st.session_state.goal) else "failed")
    metrics_n = compute_metrics(
        ann_path,
        status_ann,
        st.session_state.terrain,
        phase=st.session_state.global_phase,
    ) if ann_path else None

    data = []
    if metrics_a:
        data.append({
            "Alg": "A*",
            "Energy": metrics_a["total_energy"],
            "Length": metrics_a["path_length"],
            "Turns": metrics_a["turns"],
            "Status": "Optimal (A*)",
        })
    if metrics_n:
        data.append({
            "Alg": "ANN",
            "Energy": metrics_n["total_energy"],
            "Length": metrics_n["path_length"],
            "Turns": metrics_n["turns"],
            "Status": STATUS_LABELS.get(metrics_n["status"], "ANN failed"),
        })

    if data:
        df = pd.DataFrame(data)

        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("### Cost Metrics Table")
            st.dataframe(df.style.highlight_min(subset=["Energy", "Length", "Turns"], color="lightgreen", axis=0))
        with c2:
            st.markdown("### Individual Visual Breakdowns")
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.markdown("**Path Length**")
                st.bar_chart(df.set_index("Alg")[["Length"]], color="#ffaa00")
            with sc2:
                st.markdown("**Total Energy**")
                st.bar_chart(df.set_index("Alg")[["Energy"]], color="#00ffaa")
            with sc3:
                st.markdown("**Total Turns**")
                st.bar_chart(df.set_index("Alg")[["Turns"]], color="#aa00ff")

        st.markdown("---")
        st.markdown("### Comparison Conclusion")
        if metrics_a and metrics_n:
            e_a = metrics_a["total_energy"]
            l_a = metrics_a["path_length"]
            t_a = metrics_a["turns"]

            e_n = metrics_n["total_energy"]
            l_n = metrics_n["path_length"]
            t_n = metrics_n["turns"]
            ann_status = metrics_n["status"]
            ann_status_label = STATUS_LABELS.get(ann_status, "ANN failed")

            if ann_status not in {"success", "success_with_rescue"}:
                st.error(
                    f"**ANN failed:** A* completed the route with energy **{e_a}**, "
                    f"path length **{l_a}**, and **{t_a}** turns. The ANN run failed, so its reported "
                    f"energy **{e_n}** includes the failure penalty from the evaluation logic."
                )
            elif ann_status == "success_with_rescue":
                st.warning(
                    f"**ANN success with rescue:** The ANN needed teacher assistance to complete the route. "
                    f"Final ANN-side result: energy **{e_n}**, length **{l_n}**, turns **{t_n}**. "
                    f"Use this status for demos, but treat it as hybrid navigation rather than ANN-only performance."
                )
            elif l_a > l_n and e_a < e_n:
                st.info(
                    f"**Energy-Aware Tradeoff Confirmed:** The A* path is geometrically longer "
                    f"(**{l_a}** vs **{l_n}** steps) but still uses less total energy "
                    f"(**{e_a}** vs **{e_n}**). This supports the claim that the cost model prefers "
                    f"fewer expensive turns and easier terrain over pure shortest-distance movement."
                )
            elif e_n <= e_a:
                st.success(
                    f"**ANN-only success:** The ANN reached the goal without rescue and matched or improved on "
                    f"A* energy for this run (**{e_n}** vs **{e_a}**)."
                )
            else:
                st.warning(
                    f"**ANN-only success but less efficient:** The ANN completed the route without rescue, "
                    f"but A* remained better on total energy (**{e_a}** vs **{e_n}**)."
                )
else:
    st.info("Run simulations to see the comparative overhead!")
