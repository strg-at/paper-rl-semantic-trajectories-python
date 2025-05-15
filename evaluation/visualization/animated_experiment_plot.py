import os
import pickle
import re

import numpy as np
import umap
import streamlit as st
from sklearn.preprocessing import StandardScaler

from evaluation.visualization import visualize_trajectory

agent_traj_pattern = re.compile(r".+trajectories_(\d+)\.pkl")


@st.cache_data
def load_graph(exp_folder):
    with open(os.path.join(exp_folder, "graph_with_embeddings.pkl"), "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_target_group(exp_folder):
    with open(os.path.join(exp_folder, "trajectories_target.pkl"), "rb") as f:
        return pickle.load(f)


def get_model_trajectories(exp_folder):
    model_trajectories = os.listdir(exp_folder)
    return list(filter(lambda p: p is not None, map(agent_traj_pattern.match, model_trajectories)))


@st.cache_data
def reduce_embeddings_to_2d(embeddings):
    scaler = StandardScaler()
    scaled_embs = scaler.fit_transform(embeddings)
    reducer = umap.UMAP()
    return reducer.fit_transform(scaled_embs)


@st.cache_data
def load_agent_trajectories(exp_folder, model_file):
    with open(os.path.join(exp_folder, model_file), "rb") as f:
        return pickle.load(f)


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
        graph = load_graph(exp_folder)
        target_group = load_target_group(exp_folder)
        model_trajectories = get_model_trajectories(exp_folder)

    with st.spinner(text="Reducing embeddings to 2d", show_time=True):
        embeddings = np.array(graph.vs["embedding"])
        embs_2d = reduce_embeddings_to_2d(embeddings)

    model_trajectories = [m.string for m in model_trajectories]
    model_file = st.selectbox(
        label="Select a model checkpoint (higher values mean a later stage during training)", options=model_trajectories
    )

    with st.spinner(text="Loading agent trajectories...", show_time=True):
        learned_trajs = load_agent_trajectories(exp_folder, model_file)

        group_traj_ids = target_group["trajectory_id"]
        group_trajs = [(g, f"user_{i}") for i, g in enumerate(group_traj_ids)]
        agent_traj_idx = st.selectbox(
            label="Select an agent trajectory to visualize", options=range(len(learned_trajs)), index=0
        )
        agent_traj = (np.array(learned_trajs[agent_traj_idx]), "agent")

    n_background_samples = st.selectbox(
        label="Number of background samples to visualize", options=[100, 500, 1000, 3000, 10_000], index=2
    )
    fig = visualize_trajectory.trajectory_scatter_visualization_2d(
        embs_2d, group_trajs + [agent_traj], titles=[], n_background_samples=n_background_samples
    )
    st.plotly_chart(fig, use_container_width=False)
