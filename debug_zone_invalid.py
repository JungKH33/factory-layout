from __future__ import annotations

from typing import Any

import torch

from envs.json_loader import load_env
from envs.visualizer import plot_layout


def _count_true(m: torch.Tensor) -> int:
    return int(m.to(dtype=torch.int64).sum().item())


def _slice_cols(m: torch.Tensor, *, x_split: int) -> tuple[int, int]:
    """Return (count_left, count_right) for columns < x_split and >= x_split."""
    h, w = int(m.shape[0]), int(m.shape[1])
    xs = max(0, min(w, int(x_split)))
    left = m[:, :xs]
    right = m[:, xs:]
    return _count_true(left), _count_true(right)


def main() -> None:
    ENV_JSON = "env_configs/zones_01.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaded = load_env(ENV_JSON, device=device)
    env = loaded.env
    env.log = False

    obs, info = env.reset(options=loaded.reset_kwargs)
    # NOTE: zone invalid inside env is always computed for env.remaining[0].
    # For debugging a specific group (e.g., E in a known placement order), set target_gid explicitly.
    target_gid = "E"
    next_gid = env.remaining[2] if env.remaining else None
    if next_gid is None:
        raise RuntimeError("env.remaining is empty right after reset()")

    g = env.groups[next_gid]

    print("=== debug_zone_invalid ===")
    print("env_json=", ENV_JSON)
    print("device=", device)
    print("next_gid=", next_gid)
    print("target_gid=", target_gid)
    print("group.facility_weight=", float(getattr(g, "facility_weight", 0.0)))
    print("group.facility_height=", float(getattr(g, "facility_height", 0.0)))
    print("group.facility_dry=", float(getattr(g, "facility_dry", 0.0)))
    print("defaults: weight=", float(getattr(env, "default_weight", float("nan"))), "height=", float(getattr(env, "default_height", float("nan"))), "dry=", float(getattr(env, "default_dry", float("nan"))))

    wm = env._weight_map
    hm = env._height_map
    dm = env._dry_map

    print("weight_map_minmax=", float(wm.min().item()), float(wm.max().item()))
    print("height_map_minmax=", float(hm.min().item()), float(hm.max().item()))
    print("dry_map_minmax=", float(dm.min().item()), float(dm.max().item()))

    # Decompose zone invalid by rule (matches env._update_zone_invalid_for_next)
    w_bad = wm < float(g.facility_weight)
    h_bad = hm < float(g.facility_height)
    d_bad = dm > float(g.facility_dry)
    z_bad = w_bad | h_bad | d_bad

    # Engine current state
    z_cur = env._zone_invalid
    inv_cur = env._invalid

    # Target gid's zone invalid (does NOT mutate env). Useful when env.remaining[0] != target_gid.
    z_target = None
    try:
        z_target = env._zone_invalid_for_gid(target_gid)  # type: ignore[attr-defined]
    except Exception:
        z_target = None

    print("--- counts (total) ---")
    print("w_bad=", _count_true(w_bad), "h_bad=", _count_true(h_bad), "d_bad=", _count_true(d_bad))
    print("z_bad(or)=", _count_true(z_bad), "env._zone_invalid=", _count_true(z_cur))
    print("env._invalid=", _count_true(inv_cur))
    if isinstance(z_target, torch.Tensor):
        print("zone_invalid_for_target_gid=", _count_true(z_target))

    # Compare left/right of weight zone x>=300 (as defined in zones_01.json)
    x_split = 300
    print("--- split by x<300 vs x>=300 ---")
    print("w_bad split=", _slice_cols(w_bad, x_split=x_split))
    print("h_bad split=", _slice_cols(h_bad, x_split=x_split))
    print("d_bad split=", _slice_cols(d_bad, x_split=x_split))
    print("z_bad split=", _slice_cols(z_bad, x_split=x_split))
    print("env._zone_invalid split=", _slice_cols(z_cur, x_split=x_split))
    print("env._invalid split=", _slice_cols(inv_cur, x_split=x_split))
    if isinstance(z_target, torch.Tensor):
        print("zone_invalid_for_target_gid split=", _slice_cols(z_target, x_split=x_split))

    # Sanity: show a few values at representative points
    pts: list[tuple[int, int]] = [(10, 10), (10, 310), (260, 10), (260, 310), (450, 310)]
    print("--- sample points (y,x): weight/height/dry and flags ---")
    for y, x in pts:
        y2 = max(0, min(int(env.grid_height) - 1, int(y)))
        x2 = max(0, min(int(env.grid_width) - 1, int(x)))
        ww = float(wm[y2, x2].item())
        hh = float(hm[y2, x2].item())
        dd = float(dm[y2, x2].item())
        print(
            f"(y={y2},x={x2}) w={ww:g} h={hh:g} d={dd:g} | "
            f"w_bad={bool(w_bad[y2,x2].item())} h_bad={bool(h_bad[y2,x2].item())} d_bad={bool(d_bad[y2,x2].item())} "
            f"zone={bool(z_bad[y2,x2].item())}"
        )

    # Keep output small but confirm obs keys
    print("--- obs keys ---")
    if isinstance(obs, dict):
        print(sorted(list(obs.keys()))[:50])
    else:
        print(type(obs))

    # Echo info just in case reset passes something useful
    if isinstance(info, dict) and info:
        print("--- reset info ---")
        for k, v in list(info.items())[:20]:
            print(k, "=", v)

    # --- visualization (matplotlib) ---
    # 1) Show current env internal masks (uses engine._invalid / engine._clear_invalid etc.)
    print("\n[plot] current env masks (toggle in legend: invalid_mask / clearance_mask / zones)")
    plot_layout(env, candidate_set=None)

    # 2) Show target gid's zone invalid by temporarily injecting it into engine._zone_invalid.
    # This helps confirm whether the issue is "wrong gid for zone invalid" vs "zone maps wrong".
    if isinstance(z_target, torch.Tensor):
        print(f"\n[plot] injected zone_invalid for target_gid={target_gid!r} (toggle invalid_mask)")
        old_zone = env._zone_invalid.clone()
        old_invalid = env._invalid.clone()
        try:
            env._zone_invalid = z_target.to(device=env.device, dtype=torch.bool).clone()
            env._recompute_invalid()
            plot_layout(env, candidate_set=None)
        finally:
            env._zone_invalid = old_zone
            env._invalid = old_invalid
            env._recompute_invalid()


if __name__ == "__main__":
    main()

