import os
import pickle
import re

import numpy as np
import umap
import streamlit as st
from sklearn.preprocessing import StandardScaler

from evaluation.visualization import visualize_trajectory

agent_traj_pattern = re.compile(r"trajectories_(\d+)\.pkl")

if __name__ == "__main__":
    st.title("Trajectory visualization")

    exp_folder = st.text_input(
        label="Experiment folder",
        value=".experiments",
    )
    if not os.path.exists(exp_folder):
        if exp_folder:
            st.error(f"{exp_folder} does not exist")
        st.stop()

    experiments = os.listdir(exp_folder)
    selected_experiment = st.selectbox(label="Select an experiment", options=experiments)

    exp_folder = os.path.join(exp_folder, selected_experiment)

    runs = sorted(os.listdir(exp_folder), reverse=True)
    selected_run = st.selectbox(label="Select a run", options=runs)

    exp_folder = os.path.join(exp_folder, selected_run)

    with st.spinner(text="Loading data...", show_time=True):
        with open(os.path.join(exp_folder, "graph_with_embeddings.pkl"), "rb") as f:
            graph = pickle.load(f)

        with open(os.path.join(exp_folder, "trajectories_target.pkl"), "rb") as f:
            target_group = pickle.load(f)

        model_trajectories = os.listdir(exp_folder)
        model_trajectories = list(filter(lambda p: p is not None, map(agent_traj_pattern.match, model_trajectories)))

    with st.spinner(text="Reducing embeddings to 2d", show_time=True):
        scaler = StandardScaler()
        embeddings = np.array(graph.vs["embedding"])

        scaled_embs = scaler.fit_transform(embeddings)
        reducer = umap.UMAP()
        embs_2d = reducer.fit_transform(scaled_embs)

    model_trajectories = [m.string for m in model_trajectories]
    model_file = st.selectbox(
        label="Select a model checkpoint (higher values mean a later stage during training)", options=model_trajectories
    )

    with st.spinner(text="Loading agent trajectories...", show_time=True):
        with open(os.path.join(exp_folder, model_file), "rb") as f:
            learned_trajs = pickle.load(f)

        group_traj_ids = target_group["trajectory_id"]
        group_trajs = [(g, f"user_{i}") for i, g in enumerate(group_traj_ids)]
        agent_traj = (np.array(learned_trajs[0]), "agent")

    n_background_samples = st.selectbox(
        label="Number of background samples to visualize", options=[100, 500, 1000, 3000, 10_000], index=2
    )
    fig = visualize_trajectory.trajectory_scatter_visualization_2d(
        embs_2d, group_trajs + [agent_traj], titles=[], n_background_samples=n_background_samples
    )
    st.plotly_chart(fig, use_container_width=False)
