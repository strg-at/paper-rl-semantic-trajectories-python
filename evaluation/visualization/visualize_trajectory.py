import itertools as itt
import random

import numpy as np
import numpy.typing as npt
import plotly
import plotly.graph_objects as go


def flatten(list_of_lists):
    "Flatten one level of nesting."
    return itt.chain.from_iterable(list_of_lists)


def trajectory_scatter_visualization_2d(
    emb_reduced: npt.NDArray[np.floating],
    trajectories: list[tuple[list[int], str]],
    titles: list[str],
    n_background_samples: int = 1000,
    zoom_to_trajectories: bool = True,
    padding_factor: float = 0.1,
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

    # Calculate bounding box for trajectories if zoom is enabled
    if zoom_to_trajectories and trajectory_points:
        traj_coords = emb_reduced[list(trajectory_points)]
        x_min, x_max = traj_coords[:, 0].min(), traj_coords[:, 0].max()
        y_min, y_max = traj_coords[:, 1].min(), traj_coords[:, 1].max()

        # Add padding
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_padding = x_range * padding_factor
        y_padding = y_range * padding_factor

        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding

        # Filter background points to only those within the zoomed area
        background_mask = (
            (emb_reduced[:, 0] >= x_min)
            & (emb_reduced[:, 0] <= x_max)
            & (emb_reduced[:, 1] >= y_min)
            & (emb_reduced[:, 1] <= y_max)
        )
        background_candidates = np.where(background_mask)[0]
        # Remove trajectory points from background
        background_candidates = [idx for idx in background_candidates if idx not in trajectory_points]
    else:
        # Use full range
        x_min, x_max = emb_reduced[:, 0].min(), emb_reduced[:, 0].max()
        y_min, y_max = emb_reduced[:, 1].min(), emb_reduced[:, 1].max()
        all_indices = set(range(len(emb_reduced)))
        background_candidates = list(all_indices - trajectory_points)

    # Sample background points if needed
    if len(background_candidates) > n_background_samples:
        background_indices = np.random.choice(background_candidates, size=n_background_samples, replace=False)
    else:
        background_indices = background_candidates

    # Create scatter trace with sampled points
    scatter_trace = go.Scatter(
        x=emb_reduced[background_indices, 0],
        y=emb_reduced[background_indices, 1],
        mode="markers",
        marker=dict(opacity=0.2, color="blue"),
    )

    _, trajectory_names, paths, colors = _trajectory_path_names_colors(trajectories, emb_reduced)
    max_len = max(len(p) for p in paths)
    data = _path_scatter_2d(paths, trajectory_names, colors, max_len)

    frames = [
        go.Frame(data=[scatter_trace] + list(flatten(datas)), name=f"step {i}") for i, datas in enumerate(zip(*data))
    ]

    # Pass the calculated ranges to the layout function
    layout = _plotly_layout_with_sliders(emb_reduced, max_len, x_range=(x_min, x_max), y_range=(y_min, y_max))
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


def _plotly_layout_with_sliders(
    data: npt.NDArray,
    max_len: int,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
) -> go.Layout:
    # Use provided ranges or calculate from data
    if x_range is None:
        x_range = (data[:, 0].min(), data[:, 0].max())
    if y_range is None:
        y_range = (data[:, 1].min(), data[:, 1].max())

    layout = go.Layout(
        xaxis=dict(range=list(x_range), autorange=False, zeroline=False),
        yaxis=dict(range=list(y_range), autorange=False, zeroline=False),
        # ...existing code...
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
