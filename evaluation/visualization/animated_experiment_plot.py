import itertools
import os
import pickle
import re

import numpy as np
import pandas as pd
import streamlit as st
import umap
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from evaluation.visualization import visualize_trajectory
from rl_semantic_trajectories.distances import wasserstein_distance

# Updated pattern to match the new format (with optional second group)
agent_traj_pattern = re.compile(r"(levenshtein|abid_zou)_?(\w+)?_trajectories_(.+)\.pkl")


@st.cache_data
def load_graph(exp_folder):
    with open(os.path.join(exp_folder, "graph_with_embeddings.pkl"), "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_target_group(exp_folder, target_group_type="all"):
    """Load target group based on type (all, train, or test)"""
    match target_group_type:
        case "all":
            target_group_fname = "trajectories_target.pkl"
        case "train":
            target_group_fname = "trajectories_target_train.pkl"
        case "test":
            target_group_fname = "trajectories_target_test.pkl"
        case _:
            raise ValueError(f"Unknown target group: {target_group_type}")

    with open(os.path.join(exp_folder, target_group_fname), "rb") as f:
        return pickle.load(f)


def get_model_trajectories(exp_folder):
    model_trajectories = os.listdir(exp_folder)
    return list(filter(lambda p: p is not None, map(agent_traj_pattern.match, model_trajectories)))


def groupby_model(model_trajectories: list[re.Match[str]]):
    """Group trajectories by model, handling optional second group in pattern"""
    if not model_trajectories:
        return []

    def get_group_name(match):
        # Join first two groups (e.g., "levenshtein_emb" or "abid_zou_")
        return "_".join(match.groups("")[:2])

    mtraj = sorted(model_trajectories, key=get_group_name)
    result = []

    current_model = get_group_name(mtraj[0])
    current_group = []

    for traj in mtraj:
        model_name = get_group_name(traj)
        if model_name == current_model:
            current_group.append(traj.string)
        else:
            result.append((current_group, current_model))
            current_group = [traj.string]
            current_model = model_name

    if current_group:
        result.append((current_group, current_model))

    return result


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


@st.cache_data
def calculate_cosine_similarities(agent_trajectories, target_group, graph):
    similarities = []

    for i, traj in enumerate(agent_trajectories):
        avg_embs = np.array(graph.vs[traj]["embedding"]).mean(axis=0).reshape(1, -1)
        for j, group in enumerate(target_group["trajectory_emb"]):
            avg_group = group.mean(axis=0).reshape(1, -1)
            score = cosine_similarity(avg_embs, avg_group).item()
            similarities.append({"agent_traj_id": i, "user_traj_id": j, "score": score})

    return pd.DataFrame(similarities)


@st.cache_data
def calculate_wasserstein_distance(target_trajectories, agent_trajectories):
    return wasserstein_distance.wasserstein_uniform(target_trajectories, agent_trajectories)


def load_and_visualize():
    with st.spinner(text="Loading data...", show_time=True):
        graph = load_graph(run_folder)
        target_group = load_target_group(exp_folder, target_group_type)
        model_trajectories_matches = get_model_trajectories(exp_folder)

    if not model_trajectories_matches:
        st.error("No model trajectories found in the selected folder")
        st.stop()

    # Group trajectories by model
    model_groups = groupby_model(model_trajectories_matches)

    # Let user select a model
    model_names = [model_name for _, model_name in model_groups]

    # Apply display name mapping
    display_names = {
        "levenshtein_emb": "Cosine-Levensht.",
        "levenshtein_nonemb": "Traditional-Levensht.",
        "abid_zou_": "Abid&Zou",
    }

    selected_model_display = st.selectbox(
        label="Select a model", options=model_names, format_func=lambda x: display_names.get(x, x)
    )
    selected_model = selected_model_display

    # Find the group for the selected model
    selected_group = next((group for group, name in model_groups if name == selected_model), [])

    # Let user select a checkpoint
    model_file = st.selectbox(
        label="Select a model checkpoint",
        options=sorted(selected_group, reverse=True),
        format_func=lambda x: x.split("trajectories_")[1].replace(".pkl", ""),
    )

    with st.spinner(text="Reducing embeddings to 2d", show_time=True):
        embeddings = np.array(graph.vs["embedding"])
        embs_2d = reduce_embeddings_to_2d(embeddings)

    with st.spinner(text="Loading agent trajectories...", show_time=True):
        learned_trajs = load_agent_trajectories(exp_folder, model_file)
        target_trajectories = target_group["trajectory_id"]
        max_traj_length = max(itertools.chain(map(len, target_trajectories), map(len, learned_trajs)))

    with st.spinner("Padding trajectories...", show_time=True):
        target_traj_padded = np.array(
            [
                np.pad(
                    traj,
                    (0, max_traj_length - len(traj)),
                    mode="constant",
                    constant_values=len(graph.vs) - 1,
                )
                for traj in target_trajectories
            ]
        )

        agent_traj_padded = np.array(
            [
                np.pad(
                    traj,
                    (0, max_traj_length - len(traj)),
                    mode="constant",
                    constant_values=len(graph.vs) - 1,
                )
                for traj in learned_trajs
            ]
        )

    with st.spinner("Computing distances...", show_time=True):
        # Calculate metrics
        cosine_df = calculate_cosine_similarities(learned_trajs, target_group, graph)
        wasserstein_dist = calculate_wasserstein_distance(agent_traj_padded, target_traj_padded)

        # Display metrics
        st.subheader("Metrics")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Wasserstein Distance", f"{wasserstein_dist:.4f}")

        with col2:
            avg_cosine = cosine_df["score"].mean()
            st.metric("Average Cosine Similarity", f"{avg_cosine:.4f}")

        # Show cosine similarity table
        st.subheader("Cosine Similarities")

        # Pivot the dataframe for better visualization
        pivot_df = cosine_df.pivot(index="agent_traj_id", columns="user_traj_id", values="score")
        st.dataframe(pivot_df)

        # Allow selection based on cosine similarity
        st.subheader("Select Trajectory")

        selection_method = st.radio("Selection method", ["Manual", "Highest similarity to user", "Random"])

        if selection_method == "Manual":
            agent_traj_idx = st.selectbox(
                label="Select an agent trajectory to visualize", options=range(len(learned_trajs)), index=0
            )
        elif selection_method == "Highest similarity to user":
            user_id = st.selectbox(label="Select user to match", options=range(len(target_group["trajectory_id"])))
            # Filter for the selected user and sort by score
            user_similarities = cosine_df[cosine_df["user_traj_id"] == user_id].sort_values("score", ascending=False)
            agent_traj_idx = user_similarities.iloc[0]["agent_traj_id"]
            st.info(
                f"Selected agent {int(agent_traj_idx)} with similarity score: {user_similarities.iloc[0]['score']:.4f}"
            )
        else:  # Random
            agent_traj_idx = np.random.randint(0, len(learned_trajs))
            st.info(f"Randomly selected agent {agent_traj_idx}")

        sampled_target_traj = target_traj_padded
        if len(target_traj_padded) > 10:
            sampled_target_traj = np.random.default_rng().choice(sampled_target_traj, size=10, replace=False)

        group_trajs = [(g, f"user_{i}") for i, g in enumerate(sampled_target_traj)]
        agent_traj = (agent_traj_padded[int(agent_traj_idx)], "agent")

    n_background_samples = st.selectbox(
        label="Number of background samples to visualize", options=[100, 500, 1000, 3000, 10_000], index=2
    )

    # Add zoom control
    zoom_to_trajectories = st.checkbox(
        label="Zoom to trajectory area",
        value=True,
        help="If enabled, the plot will zoom to show only the area where agents and users are moving",
    )

    padding_factor = (
        st.slider(
            label="Padding around trajectories (%)",
            min_value=0,
            max_value=50,
            value=10,
            help="Percentage of the trajectory area to add as padding",
        )
        / 100.0
    )

    print(f"agent traj: {agent_traj}")
    print(f"group trajs: {group_trajs}")
    fig = visualize_trajectory.trajectory_scatter_visualization_2d(
        embs_2d,
        group_trajs + [agent_traj],
        titles=[],
        n_background_samples=n_background_samples,
        zoom_to_trajectories=zoom_to_trajectories,
        padding_factor=padding_factor,
    )
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    st.title("Trajectory Visualization and Metrics")

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

    # Always select a run first
    runs = sorted(os.listdir(exp_folder), reverse=True)
    selected_run = st.selectbox(label="Select a run", options=runs)
    run_folder = os.path.join(exp_folder, selected_run)

    # Now check if there are experiment folders within the run
    exp_folder = run_folder
    if any("experiment_" in d for d in os.listdir(exp_folder)):
        # New format with experiment_ directories
        experiment_dirs = sorted([d for d in os.listdir(exp_folder) if "experiment_" in d])
        selected_experiment_dir = st.selectbox(label="Select an experiment directory", options=experiment_dirs)
        exp_folder = os.path.join(run_folder, selected_experiment_dir)

    # Add target group selection
    target_group_type = st.selectbox(
        label="Select target group",
        options=["all", "train", "test"],
        index=0,
        help="Choose which target trajectories to compare against",
    )

    load_and_visualize()
