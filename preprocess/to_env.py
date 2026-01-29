"""SMA JSON -> Factory Layout Env JSON converter.

Usage:
    python -m converters.sma_to_env input.json output.json --grid-size 100
"""
from __future__ import annotations

import json
import math
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def build_facility_dict(facilities: List[Dict]) -> Dict[str, Dict]:
    """facility id -> facility data mapping."""
    return {f["id"]: f for f in facilities}


def get_main_facility(
    group_id: str,
    facility_ids: List[str],
    facility_dict: Dict[str, Dict],
) -> Dict:
    """Get representative facility for a group.
    
    Priority:
    1. Facility with mainFacility=true
    2. First facility in list
    """
    for fid in facility_ids:
        fac = facility_dict.get(fid)
        if fac and fac.get("mainFacility", False):
            return fac
    
    first_id = facility_ids[0]
    fac = facility_dict.get(first_id)
    if fac is None:
        raise ValueError(f"Facility {first_id} not found for group {group_id}")
    return fac


def extract_clearance_from_variable_area(fac: Dict) -> Tuple[float, float, float, float]:
    """Extract clearance (UL, UR, UB, UA) from areaMap.VARIABLE_AREA.
    
    Based on notebook logic (lines 137-191):
    - UL: left clearance (negative x extension)
    - UR: right clearance (positive x extension beyond width)
    - UB: bottom clearance (negative y extension)
    - UA: top clearance (positive y extension beyond height)
    
    Returns: (UL, UR, UB, UA) in mm
    """
    area_map = fac.get("areaMap", {})
    variable_areas = area_map.get("VARIABLE_AREA", [])
    
    if not variable_areas:
        return (0.0, 0.0, 0.0, 0.0)
    
    fac_width = fac["width"]
    fac_height = fac["height"]
    
    # UL: min x (negative means left extension)
    min_x = float("inf")
    for va in variable_areas:
        if va.get("x", 0) >= 0 and va.get("y", 0) >= 0:
            min_x = min(min_x, va.get("x", 0))
    UL = -min_x if min_x != float("inf") and min_x < 0 else 0.0
    
    # UR: max (x + width) beyond facility width
    max_x_right = -float("inf")
    for va in variable_areas:
        if va.get("x", 0) >= 0 and va.get("y", 0) >= 0:
            right_edge = va.get("x", 0) + va.get("width", 0)
            max_x_right = max(max_x_right, right_edge)
    UR = max(0.0, max_x_right - fac_width) if max_x_right != -float("inf") else 0.0
    
    # UB: min y (negative means bottom extension)
    min_y = float("inf")
    for va in variable_areas:
        if va.get("x", 0) >= 0 and va.get("y", 0) >= 0:
            min_y = min(min_y, va.get("y", 0))
    UB = -min_y if min_y != float("inf") and min_y < 0 else 0.0
    
    # UA: max (y + height) beyond facility height
    max_y_top = -float("inf")
    for va in variable_areas:
        if va.get("x", 0) >= 0 and va.get("y", 0) >= 0:
            top_edge = va.get("y", 0) + va.get("height", 0)
            max_y_top = max(max_y_top, top_edge)
    UA = max(0.0, max_y_top - fac_height) if max_y_top != -float("inf") else 0.0
    
    return (UL, UR, UB, UA)


def adjust_clearance_by_path(
    fac: Dict,
    UL: float, UR: float, UB: float, UA: float
) -> Tuple[float, float, float, float]:
    """Adjust clearance based on entrance/exit position and pathWidth.
    
    Based on notebook logic (lines 327-349):
    If entrance/exit is near edge, add pathWidth to clearance.
    """
    path_width = fac.get("pathWidth", 0)
    if path_width <= 0:
        return (UL, UR, UB, UA)
    
    width = fac["width"]
    height = fac["height"]
    ent_x = fac.get("entranceRelativeX", width / 2)
    ent_y = fac.get("entranceRelativeY", height / 2)
    exi_x = fac.get("exitRelativeX", width / 2)
    exi_y = fac.get("exitRelativeY", height / 2)
    
    # Right edge (>= 95% width)
    if ent_x >= 0.95 * width or exi_x >= 0.95 * width:
        UR = max(UR, path_width)
    # Left edge (<= 5% width)
    if ent_x <= 0.05 * width or exi_x <= 0.05 * width:
        UL = max(UL, path_width)
    # Top edge (>= 95% height)
    if ent_y >= 0.95 * height or exi_y >= 0.95 * height:
        UA = max(UA, path_width)
    # Bottom edge (<= 5% height)
    if ent_y <= 0.05 * height or exi_y <= 0.05 * height:
        UB = max(UB, path_width)
    
    return (UL, UR, UB, UA)


def calculate_cluster_size_2d(
    fac_width: float,
    fac_height: float,
    facility_count: int,
    UL: float, UR: float, UB: float, UA: float,
    factory_width: float,
    factory_height: float,
    group_id: str = "",
) -> Tuple[float, float, int, int]:
    """Calculate cluster size with 2D arrangement to fit within factory.
    
    Based on notebook logic (lines 7525-7560):
    - Start with 1 row/column
    - If cluster exceeds factory, increase rows/columns
    - Recalculate until it fits or max iterations reached
    
    Returns: (cluster_width, cluster_height, n_cols, n_rows)
    """
    if facility_count <= 1:
        return (fac_width, fac_height, 1, 1)
    
    gap_x = max(UL, UR)
    gap_y = max(UB, UA)
    
    # Determine stacking direction based on facility shape
    # width >= height: prefer horizontal stacking (more columns)
    # width < height: prefer vertical stacking (more rows)
    stack_horizontal = fac_width >= fac_height
    
    # Start with 1 row/column arrangement
    if stack_horizontal:
        n_cols = facility_count
        n_rows = 1
    else:
        n_cols = 1
        n_rows = facility_count
    
    # Iteratively increase rows/cols until cluster fits
    # Check if cluster fits in at least ONE orientation (0° or 90°)
    max_iterations = 100
    for iteration in range(max_iterations):
        cluster_width = fac_width * n_cols + gap_x * max(0, n_cols - 1)
        cluster_height = fac_height * n_rows + gap_y * max(0, n_rows - 1)
        
        # Check if cluster fits in at least one orientation
        # Orientation 0°: width vs factory_width, height vs factory_height
        fits_0deg = (cluster_width <= factory_width * 0.95 and 
                     cluster_height <= factory_height * 0.95)
        # Orientation 90°: height vs factory_width, width vs factory_height
        fits_90deg = (cluster_height <= factory_width * 0.95 and 
                      cluster_width <= factory_height * 0.95)
        
        if fits_0deg or fits_90deg:
            break
        
        # Debug output for problematic cases
        if iteration < 5 or iteration % 10 == 0:
            print(f"  [DEBUG] {group_id} iter={iteration}: {n_cols}x{n_rows}, "
                  f"cluster={cluster_width:.0f}x{cluster_height:.0f}mm, "
                  f"factory={factory_width:.0f}x{factory_height:.0f}mm, "
                  f"fits_0={fits_0deg}, fits_90={fits_90deg}")
        
        # Increase rows/cols to make cluster smaller
        # Try to balance the dimensions
        if cluster_width > cluster_height:
            # Width is larger: need more rows to reduce columns
            n_rows += 1
            n_cols = math.ceil(facility_count / n_rows)
        else:
            # Height is larger: need more columns to reduce rows
            n_cols += 1
            n_rows = math.ceil(facility_count / n_cols)
        
        # Safety check
        if n_rows > 50 or n_cols > 50:
            print(f"[WARN] {group_id} arrangement exceeded 50 rows/cols: {n_cols}x{n_rows}")
            break
    
    # Final calculation
    cluster_width = fac_width * n_cols + gap_x * max(0, n_cols - 1)
    cluster_height = fac_height * n_rows + gap_y * max(0, n_rows - 1)
    
    # Final warning only if doesn't fit in ANY orientation
    fits_0 = cluster_width <= factory_width and cluster_height <= factory_height
    fits_90 = cluster_height <= factory_width and cluster_width <= factory_height
    if not fits_0 and not fits_90:
        print(f"[WARN] {group_id} cluster {cluster_width:.0f}x{cluster_height:.0f}mm "
              f"doesn't fit in factory {factory_width:.0f}x{factory_height:.0f}mm in any orientation!")
    
    return (cluster_width, cluster_height, n_cols, n_rows)


def convert_io_to_relative(
    entrance_x: float,
    entrance_y: float,
    exit_x: float,
    exit_y: float,
    width: float,
    height: float,
) -> Tuple[float, float, float, float]:
    """Convert bottom-left absolute coords to center-relative coords.
    
    Source: entranceRelativeX/Y = absolute position from bottom-left (0,0)
    Target: ent_rel_x/y = relative position from center (0,0)
    """
    cx, cy = width / 2, height / 2
    ent_rel_x = entrance_x - cx
    ent_rel_y = entrance_y - cy
    exi_rel_x = exit_x - cx
    exi_rel_y = exit_y - cy
    return ent_rel_x, ent_rel_y, exi_rel_x, exi_rel_y


def build_groups(
    facility_groups: List[Dict],
    facility_dict: Dict[str, Dict],
    scale: float,
    factory_width: float,
    factory_height: float,
) -> Dict[str, Dict]:
    """Convert facilityGroups to env groups."""
    groups = {}
    
    for fg in facility_groups:
        group_id = fg["groupId"]
        facility_ids = fg["facilities"]
        facility_count = len(facility_ids)
        
        if not facility_ids:
            print(f"[WARN] Group {group_id} has no facilities, skipping")
            continue
        
        # Select representative facility (first one)
        main_fac = get_main_facility(group_id, facility_ids, facility_dict)
        
        # Basic properties (individual facility)
        fac_width = main_fac["width"]
        fac_height = main_fac["height"]
        
        # Check if this is a dock facility
        is_dock = (main_fac.get("processGroup") == "dock" or 
                   main_fac.get("type") in ("dockIn", "dockOut"))
        
        # Extract clearance from VARIABLE_AREA (dock has 0 clearance)
        if is_dock:
            UL, UR, UB, UA = 0, 0, 0, 0
        else:
            UL, UR, UB, UA = extract_clearance_from_variable_area(main_fac)
            # Adjust clearance by pathWidth
            UL, UR, UB, UA = adjust_clearance_by_path(main_fac, UL, UR, UB, UA)
        
        # Calculate cluster size with 2D arrangement
        cluster_width, cluster_height, n_cols, n_rows = calculate_cluster_size_2d(
            fac_width, fac_height, facility_count, UL, UR, UB, UA,
            factory_width, factory_height, group_id
        )
        
        # Check if cluster can rotate (fits in both orientations)
        fits_0deg = (cluster_width <= factory_width * 0.95 and 
                     cluster_height <= factory_height * 0.95)
        fits_90deg = (cluster_height <= factory_width * 0.95 and 
                      cluster_width <= factory_height * 0.95)
        can_rotate = fits_0deg and fits_90deg
        
        # If only 90° fits, swap width/height so stored dimensions work without rotation
        if fits_90deg and not fits_0deg:
            cluster_width, cluster_height = cluster_height, cluster_width
            # Also swap clearances for 90° CCW rotation:
            # new_L=old_T, new_R=old_B, new_B=old_L, new_T=old_R
            UL, UR, UB, UA = UA, UB, UL, UR
            print(f"  [INFO] {group_id} swapped to 90° orientation (only fits rotated)")
        elif not can_rotate and facility_count > 1:
            print(f"  [INFO] {group_id} rotatable=False (only fits in one orientation)")
        
        # Scale to grid units
        width = cluster_width / scale
        height = cluster_height / scale
        
        # IO coordinate conversion (based on individual facility, center-relative)
        ent_rel_x, ent_rel_y, exi_rel_x, exi_rel_y = convert_io_to_relative(
            main_fac.get("entranceRelativeX", fac_width / 2),
            main_fac.get("entranceRelativeY", 0),
            main_fac.get("exitRelativeX", fac_width / 2),
            main_fac.get("exitRelativeY", fac_height),
            fac_width,
            fac_height,
        )
        
        # Clearance in grid units (adjusted for cluster edges)
        # When multiple rows/cols, inner gaps are handled, outer clearance remains
        group_data = {
            "width": width,
            "height": height,
            "rotatable": can_rotate,
            "ent_rel_x": ent_rel_x / scale,
            "ent_rel_y": ent_rel_y / scale,
            "exi_rel_x": exi_rel_x / scale,
            "exi_rel_y": exi_rel_y / scale,
            # Clearance (scaled)
            "facility_clearance_left": int(round(UL / scale)),
            "facility_clearance_right": int(round(UR / scale)),
            "facility_clearance_bottom": int(round(UB / scale)),
            "facility_clearance_top": int(round(UA / scale)),
        }
        
        # Facility constraints
        if "facilityHeightDouble" in main_fac:
            group_data["facility_height"] = main_fac["facilityHeightDouble"]
        if "facilityWeightDouble" in main_fac:
            group_data["facility_weight"] = main_fac["facilityWeightDouble"]
        if main_fac.get("dry") is not None:
            group_data["facility_dry"] = main_fac["dry"]
        
        # Placement area constraint: allowed_areas is a list of area IDs
        placeable = main_fac.get("placeableAreaIds", [])
        if isinstance(placeable, list) and len(placeable) > 0:
            # Convert to list of strings for env compatibility
            group_data["allowed_areas"] = [str(p) for p in placeable]
        
        # Meta info (for debugging, prefixed with _)
        group_data["_facility_count"] = facility_count
        group_data["_n_cols"] = n_cols
        group_data["_n_rows"] = n_rows
        group_data["_individual_width"] = fac_width / scale
        group_data["_individual_height"] = fac_height / scale
        group_data["_main_facility_id"] = main_fac["id"]
        group_data["_type"] = main_fac.get("type", "unknown")
        group_data["_processGroup"] = main_fac.get("processGroup", "unknown")
        group_data["_pathWidth"] = main_fac.get("pathWidth", 0) / scale
        group_data["_raw_clearance"] = {
            "UL": UL / scale,
            "UR": UR / scale,
            "UB": UB / scale,
            "UA": UA / scale,
        }
        
        groups[group_id] = group_data
    
    return groups


def build_flow(
    facilities: List[Dict],
    valid_group_ids: set,
) -> List[List]:
    """Convert nextFacilityGroup to flow.
    
    TODO: Review flow weight determination (currently all 1.0)
    """
    flow_set = set()
    
    for fac in facilities:
        src_group = fac.get("facilityGroup")
        if src_group not in valid_group_ids:
            continue
        
        for dst_group in fac.get("nextFacilityGroup", []):
            if dst_group not in valid_group_ids:
                print(f"[WARN] Flow target {dst_group} not in valid groups, skipping")
                continue
            if src_group != dst_group:
                flow_set.add((src_group, dst_group))
    
    flow = [[src, dst, 1.0] for src, dst in sorted(flow_set)]
    return flow


def convert_rect(rect: Dict, scale: float) -> List[int]:
    """Convert rect dict to [x0, y0, x1, y1]."""
    x0 = rect["x"] / scale
    y0 = rect["y"] / scale
    x1 = (rect["x"] + rect["width"]) / scale
    y1 = (rect["y"] + rect["height"]) / scale
    return [int(x0), int(y0), int(x1), int(y1)]


def build_zones(
    placeable_areas: List[Dict],
    area_ceilings: List[Dict],
    area_weights: List[Dict],
    area_dry: List[Dict],
    forbidden_areas: List[Dict],
    column_areas: List[Dict],
    scale: float,
) -> Dict[str, Any]:
    """Convert area constraints to zones."""
    # Placement areas
    placement_areas = []
    for area in placeable_areas:
        placement_areas.append({
            "id": area["id"],
            "rect": convert_rect(area, scale),
        })
    
    # Height areas (from areaCeilings)
    height_areas = []
    for area in area_ceilings:
        height_areas.append({
            "rect": convert_rect(area, scale),
            "value": area.get("ceilingHeight", area.get("value", 0)),
        })
    
    # Weight areas (from areaWeights)
    weight_areas = []
    for area in area_weights:
        weight_areas.append({
            "rect": convert_rect(area, scale),
            "value": area.get("weight", area.get("value", 0)),
        })
    
    # Dry areas (from areaDry)
    dry_areas = []
    for area in area_dry:
        dry_areas.append({
            "rect": convert_rect(area, scale),
            "value": area.get("dry", area.get("value", 0)),
        })
    
    # Forbidden areas (from forbiddenAreas + columnAreas)
    forbidden_list = []
    for area in forbidden_areas:
        forbidden_list.append({"rect": convert_rect(area, scale)})
    for area in column_areas:
        forbidden_list.append({"rect": convert_rect(area, scale)})
    
    zones = {
        "placement_areas": placement_areas,
    }
    
    if forbidden_list:
        zones["forbidden_areas"] = forbidden_list
    if height_areas:
        zones["height_areas"] = height_areas
    if weight_areas:
        zones["weight_areas"] = weight_areas
    if dry_areas:
        zones["dry_areas"] = dry_areas
    
    return zones


def build_initial_positions(
    fixed_positions: List[Dict],
    facility_dict: Dict[str, Dict],
    scale: float,
) -> Dict[str, List[int]]:
    """Convert fixedFacilityPositions to initial_positions."""
    initial_positions = {}
    
    for fp in fixed_positions:
        fac_id = fp["facilityId"]
        fac = facility_dict.get(fac_id)
        if fac is None:
            print(f"[WARN] Fixed position for unknown facility {fac_id}, skipping")
            continue
        
        group_id = fac.get("facilityGroup")
        if group_id is None:
            print(f"[WARN] Facility {fac_id} has no facilityGroup, skipping")
            continue
        
        pos = fp["position"]
        x = int(pos["x"] / scale)
        y = int(pos["y"] / scale)
        rot = int(fp.get("rotation", 0))
        
        if group_id not in initial_positions:
            initial_positions[group_id] = [x, y, rot]
        else:
            print(f"[INFO] Group {group_id} already has initial position, skipping {fac_id}")
    
    return initial_positions


def convert_sma_to_env(
    input_path: str,
    output_path: str,
    grid_size: float = 100.0,
) -> Dict[str, Any]:
    """Convert SMA JSON to Env JSON.
    
    Args:
        input_path: Source SMA JSON file path
        output_path: Output Env JSON file path
        grid_size: mm per grid (default: 100mm = 1 grid)
    
    Returns:
        Converted env data
    """
    print(f"[INFO] Loading {input_path}")
    data = load_json(input_path)
    
    scale = grid_size
    
    # Factory dimensions
    factory_dim = data["factoryDimensions"]
    factory_width = factory_dim["width"]
    factory_height = factory_dim["height"]
    grid_width = int(factory_width / scale)
    grid_height = int(factory_height / scale)
    print(f"[INFO] Factory: {factory_width}x{factory_height} mm -> {grid_width}x{grid_height} grid")
    
    # Facility dict
    facilities = data["facilities"]
    facility_dict = build_facility_dict(facilities)
    print(f"[INFO] Facilities: {len(facilities)}")
    
    # Groups (pass factory dimensions for 2D arrangement calculation)
    facility_groups = data.get("facilityGroups", [])
    groups = build_groups(
        facility_groups, facility_dict, scale,
        factory_width, factory_height
    )
    print(f"[INFO] Groups: {len(groups)}")
    
    # Print group summary
    for gid, g in groups.items():
        arrangement = f"{g['_n_cols']}x{g['_n_rows']}" if g['_facility_count'] > 1 else "1x1"
        print(f"  - {gid}: {g['_facility_count']} facilities ({arrangement}), "
              f"cluster={g['width']:.1f}x{g['height']:.1f}, "
              f"type={g['_type']}")
    
    valid_group_ids = set(groups.keys())
    
    # Flow
    flow = build_flow(facilities, valid_group_ids)
    print(f"[INFO] Flow edges: {len(flow)}")
    
    # Zones (forbidden, placement, height, weight, dry)
    forbidden_areas = data.get("forbiddenAreas", [])
    column_areas = data.get("columnAreas", [])
    placeable_areas = data.get("placeableAreas", [])
    area_ceilings = data.get("areaCeilings", [])
    area_weights = data.get("areaWeights", [])
    area_dry = data.get("areaDry", [])
    zones = build_zones(
        placeable_areas, area_ceilings, area_weights, area_dry,
        forbidden_areas, column_areas, scale
    )
    print(f"[INFO] Forbidden areas: {len(zones.get('forbidden_areas', []))}")
    print(f"[INFO] Placement areas: {len(zones.get('placement_areas', []))}")
    print(f"[INFO] Height areas: {len(zones.get('height_areas', []))}")
    print(f"[INFO] Weight areas: {len(zones.get('weight_areas', []))}")
    print(f"[INFO] Dry areas: {len(zones.get('dry_areas', []))}")
    
    # Initial positions
    fixed_positions = data.get("fixedFacilityPositions", [])
    initial_positions = build_initial_positions(fixed_positions, facility_dict, scale)
    print(f"[INFO] Initial positions: {len(initial_positions)}")
    
    # Build output
    env_data = {
        "grid": {
            "width": grid_width,
            "height": grid_height,
            "grid_size": 1.0,
        },
        "env": {
            "max_candidates": 70,
            "default_weight": factory_dim.get("weight", 10.0),
            "default_height": factory_dim.get("ceilingHeight", 10.0),
            "default_dry": factory_dim.get("dry", 0.0),
        },
        "groups": groups,
        "flow": flow,
        "zones": zones,
        "reset": {},
    }
    
    if initial_positions:
        env_data["reset"]["initial_positions"] = initial_positions
    
    # Save
    save_json(env_data, output_path)
    print(f"[INFO] Saved to {output_path}")
    
    return env_data


def main():
    parser = argparse.ArgumentParser(description="Convert SMA JSON to Factory Layout Env JSON")
    parser.add_argument("input", help="Input SMA JSON file path")
    parser.add_argument("output", help="Output Env JSON file path")
    parser.add_argument(
        "--grid-size",
        type=float,
        default=100.0,
        help="mm per grid unit (default: 100)",
    )
    
    args = parser.parse_args()
    convert_sma_to_env(args.input, args.output, args.grid_size)


if __name__ == "__main__":
    main()
