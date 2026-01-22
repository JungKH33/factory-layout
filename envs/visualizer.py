from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import CheckButtons
from matplotlib.lines import Line2D
import numpy as np
import networkx as nx

from typing import Any

from envs.env import FactoryLayoutEnv


def plot_layout(env: Any, *, mask_flat: Optional[object] = None) -> None:
    """Interactive viewer (dynamic toggles only).

    - No save_path/show_* args here on purpose: use `save_layout(...)` for saving.
    - This function always opens a window and lets you toggle layers via CheckButtons.
    """
    # Support both engine (`FactoryLayoutEnv`) and wrapper envs by unwrapping.
    engine = getattr(env, "engine", env)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, engine.grid_width)
    ax.set_ylim(0, engine.grid_height)
    ax.set_aspect("equal")
    ax.set_title("FactoryLayoutEnv")

    # ---- build all artists (default ON; toggles control visibility) ----
    zone_artists: dict[str, list[Any]] = {"weight": [], "dry": [], "height": []}
    misc_artists: dict[str, list[Any]] = {
        "forbidden": [],
        "column": [],
        "invalid_mask": [],
        "clearance_mask": [],
        "flow": [],
        "score": [],
        "candidates": [],
    }

    def _add_zone_rect(
        *,
        rect: list[int] | tuple[int, int, int, int],
        kind: str,
        edgecolor: str,
        facecolor: str,
        alpha: float,
        linestyle: str = "-",
        label: Optional[str] = None,
    ) -> None:
        x0, y0, x1, y1 = (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
        w = max(0, x1 - x0)
        h = max(0, y1 - y0)
        if w <= 0 or h <= 0:
            return
        p = patches.Rectangle(
            (x0, y0),
            w,
            h,
            linewidth=1.2,
            edgecolor=edgecolor,
            facecolor=facecolor,
            alpha=alpha,
            linestyle=linestyle,
        )
        p.set_visible(True)
        ax.add_patch(p)
        zone_artists[kind].append(p)
        if label:
            t = ax.text(
                x0 + w / 2.0,
                y0 + h / 2.0,
                str(label),
                ha="center",
                va="center",
                fontsize=8,
                color=edgecolor,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.55, linewidth=0.0),
            )
            t.set_visible(True)
            zone_artists[kind].append(t)

    # weight_areas: list[{rect, value}]
    if hasattr(engine, "weight_areas") and isinstance(getattr(engine, "weight_areas"), list):
        for a in getattr(engine, "weight_areas"):
            if not isinstance(a, dict):
                continue
            rect = a.get("rect", None)
            v = a.get("value", None)
            if rect is None:
                continue
            label = None if v is None else f"w≤{float(v):g}"
            _add_zone_rect(
                rect=rect,
                kind="weight",
                edgecolor="#1f77b4",
                facecolor="#1f77b4",
                alpha=0.08,
                linestyle="-",
                label=label,
            )

    # dry_areas: list[{rect, value}]  (value is the required minimum; reverse inequality in env)
    if hasattr(engine, "dry_areas") and isinstance(getattr(engine, "dry_areas"), list):
        for a in getattr(engine, "dry_areas"):
            if not isinstance(a, dict):
                continue
            rect = a.get("rect", None)
            value = a.get("value", None)
            if rect is None:
                continue
            label = None if value is None else f"dry≥{float(value):g}"
            _add_zone_rect(
                rect=rect,
                kind="dry",
                edgecolor="#2ca02c",
                facecolor="#2ca02c",
                alpha=0.06,
                linestyle="--",
                label=label,
            )

    # height_areas: list[{rect, value}]
    if hasattr(engine, "height_areas") and isinstance(getattr(engine, "height_areas"), list):
        for a in getattr(engine, "height_areas"):
            if not isinstance(a, dict):
                continue
            rect = a.get("rect", None)
            ch = a.get("value", None)
            if rect is None:
                continue
            label = None if ch is None else f"h≤{float(ch):g}"
            _add_zone_rect(
                rect=rect,
                kind="height",
                edgecolor="#7f7f7f",
                facecolor="#7f7f7f",
                alpha=0.04,
                linestyle=":",
                label=label,
            )

    # masks (start hidden)
    mf = _plot_mask(ax, engine.forbidden_mask, color="red", alpha=0.15)
    if mf is not None:
        mf.set_visible(True)
        misc_artists["forbidden"].append(mf)
    mc = _plot_mask(ax, engine.column_mask, color="blue", alpha=0.15)
    if mc is not None:
        mc.set_visible(True)
        misc_artists["column"].append(mc)

    # invalid/clearance masks (engine-internal, start visible; toggled via legend)
    inv = getattr(engine, "_invalid", None)
    if inv is not None:
        mi = _plot_mask(ax, inv, color="#8b0000", alpha=1)  # dark red
        if mi is not None:
            mi.set_visible(False)
            misc_artists["invalid_mask"].append(mi)

    clr = getattr(engine, "_clear_invalid", None)
    if clr is not None:
        mc2 = _plot_mask(ax, clr, color="#ff6b6b", alpha=1)  # light red
        if mc2 is not None:
            mc2.set_visible(False)
            misc_artists["clearance_mask"].append(mc2)

    # Placed rects.
    for gid in engine.placed:
        x, y, rot = engine.positions[gid]
        group = engine.groups[gid]
        w, h = engine.rotated_size(group, rot)
        left, right, bottom, top = engine.rect_from_center(x, y, w, h)
        rect = patches.Rectangle(
            (left, bottom),
            right - left,
            top - bottom,
            linewidth=1.2,
            edgecolor="black",
            facecolor="orange",
            alpha=0.6,
        )
        ax.add_patch(rect)
        ax.text(x, y, str(gid), ha="center", va="center", fontsize=8)

    # candidates (start hidden): visualize valid actions from mask_flat as points
    if mask_flat is not None:
        mf = _as_mask_flat(mask_flat)
        idxs = np.where(mf)[0]
        if idxs.size > 0:
            xs = []
            ys = []
            for a in idxs.tolist():
                if hasattr(env, "decode_action"):
                    x, y, _, _, _ = env.decode_action(int(a))
                else:
                    # Engine alone does not define action semantics; wrapper must provide decode_action.
                    continue
                xs.append(x)
                ys.append(y)
            sc = ax.scatter(xs, ys, s=18, c="green", alpha=0.65, linewidths=0.0)
            sc.set_visible(True)
            misc_artists["candidates"].append(sc)

    # flow overlay (start hidden)
    flow_art = _plot_flow_overlay(ax, engine)
    for a in flow_art:
        try:
            a.set_visible(True)
        except Exception:
            pass
    misc_artists["flow"].extend(flow_art)

    # score overlay (start hidden)
    score = engine.cal_obj()
    score_text = ax.text(
            0.01,
            0.99,
            f"cost={score:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
        )
    score_text.set_visible(True)
    misc_artists["score"].append(score_text)

    # ---- legend click toggles (preferred UI) ----
    def _set_visible_group(group: list[Any], v: bool) -> None:
        for a in group:
            try:
                a.set_visible(bool(v))
            except Exception:
                pass

    groups: dict[str, list[Any]] = {
        "forbidden_mask": misc_artists["forbidden"],
        "column_mask": misc_artists["column"],
        "invalid_mask": misc_artists["invalid_mask"],
        "clearance_mask": misc_artists["clearance_mask"],
        "flow": misc_artists["flow"],
        "score": misc_artists["score"],
        "candidates": misc_artists["candidates"],
        "weight_zones": zone_artists["weight"],
        "dry_zones": zone_artists["dry"],
        "height_zones": zone_artists["height"],
    }

    # Proxies for a clean legend (click to toggle).
    proxies: list[Any] = [
        patches.Patch(facecolor="red", edgecolor="red", alpha=0.15, label="forbidden_mask"),
        patches.Patch(facecolor="blue", edgecolor="blue", alpha=0.15, label="column_mask"),
        patches.Patch(facecolor="#8b0000", edgecolor="#8b0000", alpha=0.10, label="invalid_mask"),
        patches.Patch(facecolor="#ff6b6b", edgecolor="#ff6b6b", alpha=0.10, label="clearance_mask"),
        Line2D([0], [0], color="blue", lw=1.5, alpha=0.3, label="flow"),
        Line2D([0], [0], color="black", lw=0.0, marker="s", markersize=8, label="score"),
        Line2D([0], [0], color="green", lw=0.0, marker="o", markersize=6, alpha=0.65, label="candidates"),
        patches.Patch(facecolor="#1f77b4", edgecolor="#1f77b4", alpha=0.08, label="weight_zones"),
        patches.Patch(facecolor="#2ca02c", edgecolor="#2ca02c", alpha=0.06, label="dry_zones"),
        patches.Patch(facecolor="#7f7f7f", edgecolor="#7f7f7f", alpha=0.04, label="height_zones"),
    ]
    leg = ax.legend(handles=proxies, loc="upper right", title="Click legend to toggle", framealpha=0.85)
    # Make legend entries clickable.
    legend_artist_to_key: dict[Any, str] = {}
    for text in leg.get_texts():
        text.set_picker(True)
        legend_artist_to_key[text] = str(text.get_text())

    def _on_pick(event) -> None:
        artist = getattr(event, "artist", None)
        key = legend_artist_to_key.get(artist, None)
        if key is None:
            return
        group = groups.get(key, [])
        current = bool(group[0].get_visible()) if group and hasattr(group[0], "get_visible") else False
        _set_visible_group(group, not current)
        # also reflect state on legend alpha
        try:
            artist.set_alpha(1.0 if not current else 0.35)
        except Exception:
            pass
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("pick_event", _on_pick)

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def save_layout(
    env: Any,
    *,
    show_masks: bool = True,
    show_flow: bool = False,
    show_score: bool = False,
    show_zones: bool = False,
    mask_flat: Optional[object] = None,
    save_path: str,
) -> None:
    """Save a static layout image (no interactive toggles)."""
    engine = getattr(env, "engine", env)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, engine.grid_width)
    ax.set_ylim(0, engine.grid_height)
    ax.set_aspect("equal")
    ax.set_title("FactoryLayoutEnv")

    if show_zones:
        # weight
        if hasattr(engine, "weight_areas") and isinstance(getattr(engine, "weight_areas"), list):
            for a in getattr(engine, "weight_areas"):
                if not isinstance(a, dict):
                    continue
                rect = a.get("rect", None)
                value = a.get("value", None)
                if rect is None:
                    continue
                x0, y0, x1, y1 = (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
                ax.add_patch(
                    patches.Rectangle(
                        (x0, y0),
                        max(0, x1 - x0),
                        max(0, y1 - y0),
                        linewidth=1.2,
                        edgecolor="#1f77b4",
                        facecolor="#1f77b4",
                        alpha=0.08,
                    )
                )
                if value is not None:
                    ax.text(
                        x0 + (x1 - x0) / 2.0,
                        y0 + (y1 - y0) / 2.0,
                        f"w≤{float(value):g}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="#1f77b4",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.55, linewidth=0.0),
                    )
        # dry
        if hasattr(engine, "dry_areas") and isinstance(getattr(engine, "dry_areas"), list):
            for a in getattr(engine, "dry_areas"):
                if not isinstance(a, dict):
                    continue
                rect = a.get("rect", None)
                value = a.get("value", None)
                if rect is None:
                    continue
                x0, y0, x1, y1 = (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
                ax.add_patch(
                    patches.Rectangle(
                        (x0, y0),
                        max(0, x1 - x0),
                        max(0, y1 - y0),
                        linewidth=1.2,
                        edgecolor="#2ca02c",
                        facecolor="#2ca02c",
                        alpha=0.06,
                        linestyle="--",
                    )
                )
                if value is not None:
                    ax.text(
                        x0 + (x1 - x0) / 2.0,
                        y0 + (y1 - y0) / 2.0,
                        f"dry≥{float(value):g}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="#2ca02c",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.55, linewidth=0.0),
                    )
        # height
        if hasattr(engine, "height_areas") and isinstance(getattr(engine, "height_areas"), list):
            for a in getattr(engine, "height_areas"):
                if not isinstance(a, dict):
                    continue
                rect = a.get("rect", None)
                ch = a.get("value", None)
                if rect is None:
                    continue
                x0, y0, x1, y1 = (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
                ax.add_patch(
                    patches.Rectangle(
                        (x0, y0),
                        max(0, x1 - x0),
                        max(0, y1 - y0),
                        linewidth=1.2,
                        edgecolor="#7f7f7f",
                        facecolor="#7f7f7f",
                        alpha=0.04,
                        linestyle=":",
                    )
                )
                if ch is not None:
                    ax.text(
                        x0 + (x1 - x0) / 2.0,
                        y0 + (y1 - y0) / 2.0,
                        f"h≤{float(ch):g}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="#7f7f7f",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.55, linewidth=0.0),
                    )

    if show_masks:
        _plot_mask(ax, engine.forbidden_mask, color="red", alpha=0.15)
        _plot_mask(ax, engine.column_mask, color="blue", alpha=0.15)

    for gid in engine.placed:
        x, y, rot = engine.positions[gid]
        group = engine.groups[gid]
        w, h = engine.rotated_size(group, rot)
        left, right, bottom, top = engine.rect_from_center(x, y, w, h)
        rect = patches.Rectangle(
            (left, bottom),
            right - left,
            top - bottom,
            linewidth=1.2,
            edgecolor="black",
            facecolor="orange",
            alpha=0.6,
        )
        ax.add_patch(rect)
        ax.text(x, y, str(gid), ha="center", va="center", fontsize=8)

    if mask_flat is not None:
        mf = _as_mask_flat(mask_flat)
        idxs = np.where(mf)[0]
        if idxs.size > 0:
            xs = []
            ys = []
            for a in idxs.tolist():
                if hasattr(env, "decode_action"):
                    x, y, _, _, _ = env.decode_action(int(a))
                else:
                    continue
                xs.append(x)
                ys.append(y)
            ax.scatter(xs, ys, s=18, c="green", alpha=0.65, linewidths=0.0)

    if show_flow:
        _plot_flow_overlay(ax, engine)

    if show_score:
        score = engine.cal_obj()
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
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_flow_graph(env, *, show_weights: bool = True) -> None:
    """Plot directed flow graph from env.group_flow."""
    engine = getattr(env, "engine", env)
    G = nx.DiGraph()
    for src, targets in engine.group_flow.items():
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


def _plot_flow_overlay(ax: plt.Axes, env) -> list[Any]:
    if not env.placed:
        return []
    arts: list[Any] = []
    for src, targets in env.group_flow.items():
        if src not in env.placed:
            continue
        sx, sy, _ = env.positions[src]
        for dst, weight in targets.items():
            if dst not in env.placed:
                continue
            dx, dy, _ = env.positions[dst]
            ann = ax.annotate(
                "",
                xy=(dx, dy),
                xytext=(sx, sy),
                arrowprops=dict(arrowstyle="-|>", color="blue", lw=0.8, alpha=0.3),
            )
            arts.append(ann)
    return arts


def _plot_mask(ax: plt.Axes, mask: Optional[object], *, color: str, alpha: float):
    if mask is None:
        return None
    # mask is torch.BoolTensor[H,W] where True means forbidden/invalid
    grid = np.asarray(mask.detach().cpu().numpy() if hasattr(mask, "detach") else mask).astype(np.float32)
    if grid.ndim != 2 or grid.size == 0:
        return
    masked = np.ma.masked_where(grid == 0, grid)
    h, w = int(grid.shape[0]), int(grid.shape[1])
    ax.imshow(
        masked,
        origin="lower",
        extent=[0, w, 0, h],
        cmap=_single_color_cmap(color),
        alpha=alpha,
        interpolation="nearest",
    )
    return ax.images[-1] if ax.images else None


def _single_color_cmap(color: str):
    return plt.matplotlib.colors.ListedColormap([color])


def _as_mask_2d(mask_2d: object) -> np.ndarray:
    if isinstance(mask_2d, np.ndarray):
        m = mask_2d
    else:
        # torch.Tensor or other array-like
        m = np.asarray(mask_2d.detach().cpu().numpy() if hasattr(mask_2d, "detach") else mask_2d)
    if m.ndim != 2:
        raise ValueError(f"mask_2d must be 2D, got {m.ndim}D")
    return m.astype(bool)


def _as_mask_flat(mask_flat: object) -> np.ndarray:
    if isinstance(mask_flat, np.ndarray):
        m = mask_flat
    else:
        m = np.asarray(mask_flat.detach().cpu().numpy() if hasattr(mask_flat, "detach") else mask_flat)
    if m.ndim != 1:
        raise ValueError(f"mask_flat must be 1D, got {m.ndim}D")
    return m.astype(bool)

if __name__ == "__main__":
    # Demo (updated):
    # - Uses latest engine constraint fields:
    #   env.default_*, zones.*_areas[].value, groups.*.facility_*
    # - Shows zone overlays and interactive toggles (show=True only).
    import torch

    from envs.wrappers import CoarseWrapperEnv, TopKWrapperEnv
    from envs.env import FacilityGroup, FactoryLayoutEnv

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # --- groups (footprint width/height + constraint attributes) ---
    # Note: footprint height is still `height`; constraint vertical height is `facility_height`.
    groups = {
        "A": FacilityGroup(id="A", width=20, height=10, rotatable=True, facility_weight=3.0, facility_height=2.0, facility_dry=0.0),
        "B": FacilityGroup(id="B", width=16, height=16, rotatable=True, facility_weight=4.0, facility_height=2.0, facility_dry=0.0),
        # Make C "heavy + tall + dry-sensitive" to demonstrate zones affecting mask.
        "C": FacilityGroup(id="C", width=18, height=12, rotatable=True, facility_weight=12.0, facility_height=10.0, facility_dry=2.0),
    }
    flow = {"A": {"B": 1.0}, "B": {"C": 0.7}}

    # --- base masks ---
    forbidden = torch.zeros((80, 120), dtype=torch.bool, device=dev)
    forbidden[0:20, 0:30] = True

    # --- zones / constraints ---
    # Unified schema: default_* + *_areas (rect,value)
    default_weight = 10.0
    weight_areas = [{"rect": (60, 0, 120, 80), "value": 20.0}]  # higher allowed weight on right half
    default_height = 20.0
    height_areas = [{"rect": (0, 60, 120, 80), "value": 5.0}]  # low ceiling strip
    default_dry = 0.0
    dry_areas = [{"rect": (0, 40, 60, 80), "value": 2.0}]  # higher dry requirement zone (reverse inequality)

    # Pre-place multiple groups to visualize non-empty layouts.
    # NOTE: reset() validates feasibility and raises ValueError if invalid.
    initial_positions = {
        "A": (90.0, 20.0, 0),
        "B": (90.0, 40.0, 0),
    }
    # Force next gid to be "C" so constraint-driven mask is visible immediately after reset.
    remaining_order = ["C", "A", "B"]

    engine = FactoryLayoutEnv(
        grid_width=120,
        grid_height=80,
        groups=groups,
        group_flow=flow,
        forbidden_mask=forbidden,
        device=dev,
        max_steps=10,
        weight_areas=weight_areas,
        height_areas=height_areas,
        dry_areas=dry_areas,
        default_weight=default_weight,
        default_height=default_height,
        default_dry=default_dry,
        log=False,
    )

    # ---- 1) Coarse wrapper demo ----
    env1 = CoarseWrapperEnv(engine=engine, coarse_grid=32, rot=0)
    obs1, _ = env1.reset(options={"initial_positions": initial_positions, "remaining_order": remaining_order})
    mask1 = obs1["mask_flat"]
    # Interactive viewer: dynamic toggles control masks/flow/score/zones.
    plot_layout(env1, mask_flat=mask1)
    plot_flow_graph(env1)

    # ---- 2) TopK wrapper demo ----
    env2 = TopKWrapperEnv(
        engine=engine,
        k=70,
        scan_step=5.0,
        quant_step=5.0,
        p_high=0.2,
        p_near=0.8,
        p_coarse=0.0,
        oversample_factor=2,
        random_seed=7,
    )
    obs2, _ = env2.reset(options={"initial_positions": initial_positions, "remaining_order": remaining_order})
    mask2 = obs2["mask_flat"]
    plot_layout(env2, mask_flat=mask2)
    plot_flow_graph(env2)

