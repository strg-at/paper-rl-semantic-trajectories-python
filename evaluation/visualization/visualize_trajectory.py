import random

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import plotly
import itertools as itt


def flatten(list_of_lists):
    "Flatten one level of nesting."
    return itt.chain.from_iterable(list_of_lists)


def trajectory_scatter_visualization_2d(
    emb_reduced: npt.NDArray[np.floating],
    trajectories: list[tuple[list[int], str]],
    titles: list[str],
    n_background_samples: int = 1000,
) -> go.Figure:
    """
    Same as :py:func:`trajectory_scatter_visualization_3d`, but for 2D scatter plots.

    :param embeddings: numpy array of floats
    :param dimens_red_cls: any algorithm implementing the :py:class:`DimReductionProtocol` protocol (i.e., any ``sklearn`` model).
    :param trajectories: a list of tuples, where each tuple is a list of node ids and a string, indicating the name of the trajectory.
    :param titles: a list of strings, indicating the title of each step of the animation.
    :param animation: whether to create a static or animated plot, defaults to True. **THIS IS CURRENTLY NOT IMPLEMENTED**.
    :return: plotly 2D figure.
    """
    # Get all trajectory points to ensure they're included
    trajectory_points = set()
    for path, _ in trajectories:
        trajectory_points.update(path)

    # Create mask for non-trajectory points
    all_indices = set(range(len(emb_reduced)))
    background_points = list(all_indices - trajectory_points)

    # Sample background points if needed
    if len(background_points) > n_background_samples:
        background_indices = np.random.choice(background_points, size=n_background_samples, replace=False)
    else:
        background_indices = background_points

    # Create scatter trace with sampled points
    scatter_trace = go.Scatter(
        x=emb_reduced[background_indices, 0],
        y=emb_reduced[background_indices, 1],
        mode="markers",
        marker=dict(opacity=0.2, color="blue"),
    )

    # Rest of the function remains the same
    _, trajectory_names, paths, colors = _trajectory_path_names_colors(trajectories, emb_reduced)
    max_len = max(len(p) for p in paths)
    data = _path_scatter_2d(paths, trajectory_names, colors, max_len)

    frames = [
        go.Frame(data=[scatter_trace] + list(flatten(datas)), name=f"step {i}") for i, datas in enumerate(zip(*data))
    ]

    layout = _plotly_layout_with_sliders(emb_reduced, max_len)
    # Create placeholder traces for each line and marker in the animation
    initial_traces = []
    for i in range(len(paths)):
        # Empty line trace
        initial_traces.append(
            go.Scatter(x=[], y=[], mode="lines", name=f"{trajectory_names[i]}", line=dict(color=colors[i], width=4))
        )
        # Empty marker trace
        initial_traces.append(
            go.Scatter(x=[], y=[], mode="markers", name=f"{trajectory_names[i]}", line=dict(color=colors[i], width=4))
        )

    fig = go.Figure(
        data=[scatter_trace] + initial_traces,
        layout=layout,
        frames=frames,
    )

    if titles:
        assert len(titles) <= max_len, f"titles cannot be longer than the longest trajectory, which is {max_len} steps."

    for i, title in enumerate(titles):
        fig.frames[i]["layout"].update(title_text=title)

    return fig


def _trajectory_path_names_colors(
    trajectories: list[tuple[list[int], str]], emb_reduced: npt.NDArray[np.floating]
) -> tuple[list[list[int]], list[str], list[list[npt.NDArray[np.floating]]], list[str]]:
    trajectory_paths = [p for p, _ in trajectories]
    trajectory_names = [n for _, n in trajectories]
    # map paths to embedding
    paths = [[emb_reduced[p] for p in path] for path in trajectory_paths]
    colors = random.sample(plotly.colors.qualitative.Dark24, k=len(trajectories))
    return trajectory_paths, trajectory_names, paths, colors


def _path_scatter_2d(
    paths: list[list[npt.NDArray[np.floating]]], trajectory_names: list[str], colors: list[str], max_len: int
) -> list[list[go.Scatter]]:
    data = []
    for i, path in enumerate(paths):
        traj_x = [p[0] for p in path]
        traj_y = [p[1] for p in path]
        frames = []
        for j in range(max_len):
            # Line trace showing path up to current point
            line_trace = go.Scatter(
                x=traj_x[: j + 1],
                y=traj_y[: j + 1],
                mode="lines",
                name=trajectory_names[i],
                line=dict(color=colors[i], width=4),
            )

            if "agent" in trajectory_names[i]:
                marker_trace = go.Scatter(
                    x=[traj_x[j]],
                    y=[traj_y[j]],
                    mode="markers+text",
                    name=trajectory_names[i],
                    text=["agent"],
                    textfont=dict(size=16, color=colors[i]),  # Increased text size and matching color
                    textposition="top right",
                    marker=dict(size=15, color=colors[i]),
                    line=dict(color=colors[i], width=4),
                )
            else:
                marker_trace = go.Scatter(
                    x=[traj_x[j]],
                    y=[traj_y[j]],
                    mode="markers",
                    name=trajectory_names[i],
                    marker=dict(size=15, color=colors[i]),
                    line=dict(color=colors[i], width=4),
                )

            frames.append((line_trace, marker_trace))
        data.append(frames)
    return data


def _plotly_layout_with_sliders(data: npt.NDArray, max_len: int) -> go.Layout:
    layout = go.Layout(
        xaxis=dict(range=[data[:, 0].min(), data[:, 0].max()], autorange=False, zeroline=False),
        yaxis=dict(range=[data[:, 1].min(), data[:, 1].max()], autorange=False, zeroline=False),
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 500, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 300, "easing": "quadratic-in-out"},
                            },
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                ],
            )
        ],
    )

    slider_dict = {
        "steps": [
            {
                "args": [
                    [f"step {i}"],
                    {
                        "frame": {"duration": 300, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 300},
                    },
                ],
                "label": f"step {i}",
                "method": "animate",
            }
            for i in range(max_len)
        ],
        "transition": {"duration": 300, "easing": "cubic-in-out"},
    }
    layout["sliders"] = [slider_dict]
    return layout
