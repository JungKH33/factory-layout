from __future__ import annotations

from typing import Iterable, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import networkx as nx

from envs.env_old import Candidate, FacilityGroup, FactoryLayoutEnvOld, Rect, RectMask


def plot_layout(
    env: FactoryLayoutEnvOld,
    *,
    show_masks: bool = True,
    show_flow: bool = False,
    show_score: bool = False,
    candidates: Optional[Iterable[Candidate]] = None,
    mask: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Plot layout and optionally overlay candidate centers."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.set_xlim(0, env.grid_width)
    ax.set_ylim(0, env.grid_height)
    ax.set_aspect("equal")
    ax.set_title("Facility Layout")

    if show_masks:
        _plot_mask(ax, env.forbidden_mask, color="red", alpha=0.15)
        _plot_mask(ax, env.column_mask, color="blue", alpha=0.15)
        _plot_mask(ax, env.dry_mask, color="green", alpha=0.12)
        _plot_mask(ax, env.weight_mask, color="purple", alpha=0.12)

    for gid in env.placed:
        x, y, rot = env.positions[gid]
        group = env.groups[gid]
        w, h = env.rotated_size(group, rot)
        left, right, bottom, top = env.rect_from_center(x, y, w, h)
        rect = patches.Rectangle(
            (left, bottom),
            right - left,
            top - bottom,
            linewidth=1.5,
            edgecolor="black",
            facecolor="orange",
            alpha=0.6,
        )
        ax.add_patch(rect)
        ax.text(x, y, str(gid), ha="center", va="center", fontsize=8)

    if candidates is not None:
        xs = []
        ys = []
        colors = []
        for idx, cand in enumerate(candidates):
            xs.append(cand.x)
            ys.append(cand.y)
            if mask is None:
                colors.append("gray")
            else:
                colors.append("green" if idx < len(mask) and mask[idx] == 1 else "red")
        ax.scatter(xs, ys, c=colors, s=18, alpha=0.8)

    if show_flow:
        _plot_flow_overlay(ax, env)

    if show_score:
        score = env.cal_obj()
        ax.text(
            0.01,
            0.99,
            f"cost={score:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
        )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_flow_graph(env: FactoryLayoutEnvOld, *, show_weights: bool = True) -> None:
    """Plot directed flow graph from env.group_flow."""
    G = nx.DiGraph()
    for src, targets in env.group_flow.items():
        for dst, weight in targets.items():
            G.add_edge(src, dst, weight=weight)

    if G.number_of_nodes() == 0:
        return

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_size=800)
    nx.draw_networkx_labels(G, pos, font_size=9)
    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=16,
        width=1.5,
        connectionstyle="arc3,rad=0.08",
    )
    if show_weights:
        edge_labels = {(u, v): f"{d.get('weight', 1.0):.2f}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title("Flow Graph")
    plt.tight_layout()
    plt.show()


def _plot_flow_overlay(ax: plt.Axes, env: FactoryLayoutEnvOld) -> None:
    """Overlay flow graph edges on the layout axis."""
    if not env.placed:
        return
    for src, targets in env.group_flow.items():
        if src not in env.placed:
            continue
        sx, sy, _ = env.positions[src]
        for dst, weight in targets.items():
            if dst not in env.placed:
                continue
            dx, dy, _ = env.positions[dst]
            ax.annotate(
                "",
                xy=(dx, dy),
                xytext=(sx, sy),
                arrowprops=dict(arrowstyle="-|>", color="blue", lw=0.8, alpha=0.3),
            )

def _plot_mask(
    ax: plt.Axes,
    mask: Optional[object],
    *,
    color: str,
    alpha: float,
) -> None:
    """Overlay a RectMask as a semi-transparent image."""
    if mask is None or not hasattr(mask, "allowed"):
        return
    grid = np.asarray(mask.allowed, dtype=np.float32)
    if grid.size == 0:
        return
    # True means allowed. We highlight forbidden (False) cells.
    forbidden = 1.0 - grid
    masked = np.ma.masked_where(forbidden == 0, forbidden)
    ax.imshow(
        masked,
        origin="lower",
        extent=[0, mask.width, 0, mask.height],
        cmap=_single_color_cmap(color),
        alpha=alpha,
        interpolation="nearest",
    )


def _single_color_cmap(color: str):
    """Create a single-color colormap for mask overlays."""
    return plt.matplotlib.colors.ListedColormap([color])


if __name__ == "__main__":
    # Minimal example usage for quick visualization.
    grid_w = 50
    grid_h = 30
    forbidden_rect = (10, 20, 18, 32)  # (left, right, bottom, top)
    
    groups = {
        "A": FacilityGroup(id="A", width=8.0, height=4.0),
        "B": FacilityGroup(id="B", width=6.0, height=6.0, rotatable=False),
        "C": FacilityGroup(id="C", width=5.0, height=3.0),
    }
    flow = {"A": {"B": 1.0}, "B": {"C": 1.0}}
    # Build a simple forbidden mask: block a central rectangle.
    allowed = [[True for _ in range(grid_w)] for _ in range(grid_h)]
    left, right, bottom, top = forbidden_rect
    y0 = max(0, int(bottom))
    y1 = min(grid_h, int(top))
    x0 = max(0, int(left))
    x1 = min(grid_w, int(right))
    for yy in range(y0, y1):
        for xx in range(x0, x1):
            allowed[yy][xx] = False
    forbidden_mask = RectMask(allowed)
    env = FactoryLayoutEnvOld(
        grid_width=grid_w,
        grid_height=grid_h,
        grid_size=1.0,
        groups=groups,
        group_flow=flow,
        max_candidates=10,
        forbidden_mask=forbidden_mask,
    )
    env.reset(
        initial_positions={
            "A": (10.0, 10.0, 0),
            "B": (25.0, 10.0, 0),
        }
    )
    candidates = [
        Candidate("C", 12.0, 20.0, 0),
        Candidate("C", 35.0, 22.0, 90),
        Candidate("C", 40.0, 8.0, 0),
    ]
    plot_layout(env, show_masks=True, candidates=candidates)
    plot_flow_graph(env)
