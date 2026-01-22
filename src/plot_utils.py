from typing import List
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def project_focus_to_3d(
    focus_matrix: np.ndarray,
    bg_matrix: np.ndarray | None,
    mode: str,
    show_background: bool,
    pca_fit_mode: str,
    bg_fit_sample: int,
    bg_plot_sample: int,
):
    """
    Returns: focus_3d, bg_3d (or None), bg_words_idx_used_for_plot (or None)
    Key: fit PCA on FOCUS (default) to avoid "chụm".
    """
    rng = np.random.RandomState(42)

    if pca_fit_mode == "Focus only" or (bg_matrix is None) or (not show_background):
        pca = PCA(n_components=3, random_state=42).fit(focus_matrix)
        focus_3d = pca.transform(focus_matrix)
        bg_3d = None
        return focus_3d, bg_3d, None

    # Focus + bg sample
    Nbg = bg_matrix.shape[0]
    fit_n = min(bg_fit_sample, Nbg)
    fit_idx = rng.choice(Nbg, size=fit_n, replace=False)
    fit_matrix = np.vstack([focus_matrix, bg_matrix[fit_idx]]).astype(np.float32)

    pca = PCA(n_components=3, random_state=42).fit(fit_matrix)
    focus_3d = pca.transform(focus_matrix)

    # plot bg sample (not necessarily same as fit sample)
    plot_n = min(bg_plot_sample, Nbg)
    plot_idx = rng.choice(Nbg, size=plot_n, replace=False)
    bg_3d = pca.transform(bg_matrix[plot_idx])
    return focus_3d, bg_3d, plot_idx

# =============================
# Matplotlib quiver plot
# =============================
def plot_vectors_quiver(points_3d: np.ndarray, labels: list[str], arrow_scale: float = 10.0):
    P = points_3d * float(arrow_scale)

    fig = plt.figure(figsize=(7.5, 7.5))
    ax = fig.add_subplot(111, projection="3d")

    max_abs = float(np.max(np.abs(P))) if P.size else 1.0
    R = max(1.0, max_abs * 1.25)

    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.set_zlim(-R, R)
    ax.set_box_aspect([1, 1, 1])

    # thick axes lines
    ax.plot([-R, R], [0, 0], [0, 0], linewidth=3)
    ax.plot([0, 0], [-R, R], [0, 0], linewidth=3)
    ax.plot([0, 0], [0, 0], [-R, R], linewidth=3)

    # vectors
    for i, (x, y, z) in enumerate(P):
        ax.quiver(
            0, 0, 0, x, y, z,
            arrow_length_ratio=0.12,
            linewidth=2.2,
            normalize=False,
        )
        ax.text(x, y, z, labels[i])

    # origin
    ax.scatter([0], [0], [0], s=90)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ticks = np.linspace(-R, R, 7)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)
    ax.grid(True)

    ax.view_init(elev=20, azim=35)
    fig.tight_layout()
    return fig


def plot_vectors_plotly_arrows(points_3d: np.ndarray, labels: list[str], arrow_scale: float = 10.0):
    """
    Interactive 3D plot:
    - thick XYZ axes like expected
    - arrows from origin to each vector (shaft + cone tip)
    """
    P = points_3d * float(arrow_scale)

    max_abs = float(np.max(np.abs(P))) if P.size else 1.0
    R = max(1.0, max_abs * 1.35)

    fig = go.Figure()

    # --- Thick axes lines (X/Y/Z) ---
    axis_line_width = 10
    fig.add_trace(go.Scatter3d(x=[-R, R], y=[0, 0], z=[0, 0],
                               mode="lines", line=dict(width=axis_line_width),
                               hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[-R, R], z=[0, 0],
                               mode="lines", line=dict(width=axis_line_width),
                               hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[-R, R],
                               mode="lines", line=dict(width=axis_line_width),
                               hoverinfo="skip", showlegend=False))

    # Origin marker
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers+text",
        marker=dict(size=6),
        text=["O"],
        textposition="bottom center",
        hoverinfo="skip",
        showlegend=False
    ))

    # --- Arrows (shaft + cone tip) ---
    shaft_width = 7
    for (x, y, z), label in zip(P, labels):
        # Shaft
        fig.add_trace(go.Scatter3d(
            x=[0, x], y=[0, y], z=[0, z],
            mode="lines",
            line=dict(width=shaft_width),
            hoverinfo="skip",
            showlegend=False
        ))

        # Tip
        fig.add_trace(go.Cone(
            x=[x], y=[y], z=[z],
            u=[x], v=[y], w=[z],
            anchor="tip",
            sizemode="absolute",
            sizeref=0.20,   # nhỏ hơn => cone to hơn
            showscale=False,
            hovertemplate=f"<b>{label}</b><br>x=%{{x:.3f}}<br>y=%{{y:.3f}}<br>z=%{{z:.3f}}<extra></extra>",
            showlegend=False
        ))

        # Label near tip
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode="text",
            text=[label],
            textposition="top center",
            hoverinfo="skip",
            showlegend=False
        ))

    # --- Make axes readable ---
    tick_font = dict(size=14)
    title_font = dict(size=16)

    fig.update_layout(
        height=720,
        margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(
            aspectmode="cube",
            xaxis=dict(
                title="X",
                range=[-R, R],
                showgrid=True,
                gridwidth=2,
                tickfont=tick_font,
                title_font=title_font,
                nticks=7,
                backgroundcolor="rgba(240,240,240,0.35)",
            ),
            yaxis=dict(
                title="Y",
                range=[-R, R],
                showgrid=True,
                gridwidth=2,
                tickfont=tick_font,
                title_font=title_font,
                nticks=7,
                backgroundcolor="rgba(240,240,240,0.35)",
            ),
            zaxis=dict(
                title="Z",
                range=[-R, R],
                showgrid=True,
                gridwidth=2,
                tickfont=tick_font,
                title_font=title_font,
                nticks=7,
                backgroundcolor="rgba(240,240,240,0.35)",
            ),
            # camera: góc nhìn dễ “đọc” hơn
            camera=dict(eye=dict(x=1.35, y=1.35, z=0.9)),
        ),
    )

    return fig