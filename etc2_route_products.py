"""Build local, synthetic ETC2.0 route/link/speed/turn views for the dashboard.

This uses the generator's saved research_truth sidecars. Link and trip relations
are the simulator's saved OSM paths; synthetic GPS samples are compared to their
known road centerlines, not guessed from the visual basemap.
"""

from __future__ import annotations

import io
import json
import math
import time
import zipfile
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from pyproj import Transformer

TRIP_KEY: tuple[str, str, str] = ("Operation_date", "Operation_ID1", "Trip_number")
FORM_12_COLUMNS: list[str] = [
    "RSU_ID_code", "Operation_date", "Operation_ID1", "Trip_number",
    "Vehicle_type", "Vehicle_use", "GPS_time", "Serial_number",
    "Trip_start_time", "Trip_end_time", "Trip_start_end_flag", "Longitude",
    "Latitude", "Accumulation_condition", "Road_type_code", "Speed",
]
FORM_14_COLUMNS: list[str] = [
    "RSU_ID_code", "Operation_date", "Operation_ID1", "Trip_number",
    "Vehicle_type", "Vehicle_use", "GPS_time", "Longitude", "Latitude",
    "Heading", "Road_type_code", "Speed", "Yaw_angular_velocity",
    "Longitudinal_acceleration", "Lateral_acceleration", "Exclusion_flag_1",
    "Exclusion_flag_2_longitudinal", "Exclusion_flag_2_lateral",
    "Exclusion_flag_2_yaw",
]
GRID_CRS: str = "EPSG:32652"
TO_UTM: Transformer = Transformer.from_crs("EPSG:4326", GRID_CRS, always_xy=True)
TURN_MIN_DEG: float = 30.0
UTURN_MIN_DEG: float = 150.0
EARTH_RADIUS_M: float = 6_371_008.8


def _key(date: object, operation_id: object, trip_number: object) -> tuple[str, str, str]:
    date_text = str(date).split(".")[0].zfill(8)
    vehicle_text = str(operation_id).split(".")[0].zfill(10)
    trip_text = str(trip_number).split(".")[0]
    return date_text, vehicle_text, trip_text


def _haversine_m(lon1: np.ndarray, lat1: np.ndarray,
                 lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    lon1, lat1, lon2, lat2 = [np.radians(np.asarray(x, dtype=float))
                              for x in (lon1, lat1, lon2, lat2)]
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _nearest_route_segment(points_xy: np.ndarray, starts_xy: np.ndarray,
                           ends_xy: np.ndarray) -> np.ndarray:
    """Assign each projected GPS point to its nearest segment in its known trip."""
    if len(starts_xy) == 0:
        return np.zeros(len(points_xy), dtype=int)
    starts = np.asarray(starts_xy, dtype=float)
    vectors = np.asarray(ends_xy, dtype=float) - starts
    denominator = np.einsum("ij,ij->i", vectors, vectors)
    valid_segments = denominator > 0
    if not valid_segments.any():
        return np.zeros(len(points_xy), dtype=int)
    starts = starts[valid_segments]
    vectors = vectors[valid_segments]
    denominator = denominator[valid_segments]
    # Exact point-to-segment distance, broadcast across each trip. Chunking
    # caps temporary memory for unusually long trips/routes.
    chunk_size = max(1, min(len(points_xy), 1_000_000 // len(starts)))
    result: list[np.ndarray] = []
    for offset in range(0, len(points_xy), chunk_size):
        points = np.asarray(points_xy[offset:offset + chunk_size], dtype=float)
        relative = points[:, None, :] - starts[None, :, :]
        fraction = np.einsum("pmi,mi->pm", relative, vectors)
        fraction /= denominator[None, :]
        np.clip(fraction, 0.0, 1.0, out=fraction)
        projected_delta = relative - fraction[:, :, None] * vectors[None, :, :]
        distance_squared = np.einsum("pmi,pmi->pm", projected_delta, projected_delta)
        result.append(np.argmin(distance_squared, axis=1))
    valid_indices = np.concatenate(result)
    original_indices = np.flatnonzero(valid_segments)
    return original_indices[valid_indices]


def build_route_products(
    research_code_dir: str | Path,
    progress_callback: Callable[[str], None] | None = None,
    include_gps: bool = True,
    include_events: bool = True,
    include_blind_spots: bool = True,
) -> dict[str, object]:
    """Read local generated truth/GPS files and return dataframes for four maps."""
    started = time.perf_counter()

    def report(message: str) -> None:
        elapsed = time.perf_counter() - started
        update = f"{message} ({elapsed:.0f}s elapsed)"
        print(f"[ETC2 road analysis] {update}", flush=True)
        if progress_callback is not None:
            progress_callback(update)

    project_dir = Path(research_code_dir)
    # Support both the historical project layout and isolated dated runs.
    if (project_dir / "research_truth").is_dir():
        truth_dir = project_dir / "research_truth"
        data_dir = project_dir / "etc40_pseudo" / "OUT1-2"
    else:
        truth_dir = project_dir / "pseudo_etc2_data" / "research_truth"
        data_dir = project_dir / "pseudo_etc2_data" / "etc40_pseudo" / "OUT1-2"
    required = [truth_dir / "trip_truth.csv", truth_dir / "network_nodes.csv",
                truth_dir / "network_edges.csv", data_dir]
    if not all(path.exists() for path in required):
        return {"available": False, "message": "Generator truth files or Form 1-2 ZIPs were not found.",
                "origin_destination": pd.DataFrame(), "links": pd.DataFrame(),
                "turns": pd.DataFrame(), "speed_by_link": pd.DataFrame(),
                "blind_spots": pd.DataFrame(),
                "event_samples": pd.DataFrame(), "speed_checks": pd.DataFrame(), "metadata": {}}

    metadata_path = truth_dir / "simulation_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    trip_truth = pd.read_csv(truth_dir / "trip_truth.csv", dtype={
        "Operation_date": "string", "Operation_ID1": "string", "Trip_number": "string",
        "origin_node": "string", "destination_node": "string", "route_nodes_json": "string",
    })
    nodes = pd.read_csv(truth_dir / "network_nodes.csv", dtype={"node_id": "string"})
    edges = pd.read_csv(truth_dir / "network_edges.csv", dtype={"u": "string", "v": "string"})
    node_xy = {str(row.node_id): (float(row.x), float(row.y)) for row in nodes.itertuples(index=False)}
    report(f"Loaded network tables ({len(nodes):,} nodes, {len(edges):,} directed links); parsing road shapes")

    edge_meta: dict[str, dict[str, object]] = {}
    for row in edges.itertuples(index=False):
        u, v = str(row.u), str(row.v)
        if u not in node_xy or v not in node_xy:
            continue
        link_id = f"{u}->{v}"
        try:
            geometry = json.loads(row.geometry_coords_json)
            # Keep the full mapped graph compact in memory. Python tuples and
            # Python float objects multiply the CSV size many times over.
            geometry = np.asarray(geometry, dtype=np.float32).reshape(-1, 2)
        except (AttributeError, TypeError, json.JSONDecodeError):
            geometry = np.empty((0, 2), dtype=np.float32)
        if len(geometry) < 2:
            geometry = np.asarray([node_xy[u], node_xy[v]], dtype=np.float32)
        edge_meta[link_id] = {
            "link_id": link_id, "u": u, "v": v,
            "lon1": node_xy[u][0], "lat1": node_xy[u][1],
            "lon2": node_xy[v][0], "lat2": node_xy[v][1],
            "geometry_coords": geometry,
            "length_m": float(row.length), "road_type_code": int(row.rtc),
            "highway": str(row.highway), "assumed_free_flow_kmh": float(row.v_ms) * 3.6,
            "maxspeed_kmh": float(row.v_kmh) if hasattr(row, "v_kmh") else float(row.v_ms) * 3.6,
            "width_m": float(row.width_m) if hasattr(row, "width_m") else np.nan,
            "width_source": str(row.width_source) if hasattr(row, "width_source") else "not provided",
            "narrow_road": bool(row.narrow_road) if hasattr(row, "narrow_road") else False,
            "speed_source": str(row.speed_source) if hasattr(row, "speed_source") else "road-class default",
        }
    report(f"Parsed {len(edge_meta):,} mapped road links")

    route_by_key: dict[tuple[str, str, str], dict[str, object]] = {}
    od_records: list[dict[str, object]] = []
    link_trip_sets: dict[str, set[str]] = {link_id: set() for link_id in edge_meta}
    link_od_sets: dict[str, set[str]] = {link_id: set() for link_id in edge_meta}
    turn_records: list[dict[str, object]] = []

    for row in trip_truth.itertuples(index=False):
        trip_key = _key(row.Operation_date, row.Operation_ID1, row.Trip_number)
        try:
            route_nodes = [str(x) for x in json.loads(row.route_nodes_json)]
        except (TypeError, json.JSONDecodeError):
            continue
        if len(route_nodes) < 2 or any(node_id not in node_xy for node_id in route_nodes):
            continue
        trip_id = f"T{len(route_by_key) + 1:05d}"
        origin_node, destination_node = str(row.origin_node), str(row.destination_node)
        origin, destination = node_xy[origin_node], node_xy[destination_node]
        od_pair = f"{origin_node}->{destination_node}"
        route_links = [f"{u}->{v}" for u, v in zip(route_nodes[:-1], route_nodes[1:])]
        edge_shapes = [edge_meta[link_id]["geometry_coords"] for link_id in route_links]
        coords: list[tuple[float, float]] = []
        for shape in edge_shapes:
            shape_points = shape.tolist()
            if not coords:
                coords.extend(shape_points)
            else:
                coords.extend(shape_points[1:] if shape_points[0] == coords[-1] else shape_points)
        if not coords:
            coords = [origin, destination]
        route_coords = np.asarray(coords, dtype=np.float32)
        route_by_key[trip_key] = {"trip_id": trip_id, "coords": route_coords,
                                  "links": route_links}
        od_records.append({
            "trip_id": trip_id, "operation_date": trip_key[0],
            "origin_node": origin_node, "destination_node": destination_node,
            "origin_lon": origin[0], "origin_lat": origin[1],
            "destination_lon": destination[0], "destination_lat": destination[1],
            "route_coords_json": json.dumps(route_coords.tolist()),
            "route_link_count": len(route_links),
        })
        for link_id in route_links:
            if link_id in link_trip_sets:
                link_trip_sets[link_id].add(trip_id)
                link_od_sets[link_id].add(od_pair)
        for i in range(1, len(route_nodes) - 1):
            incoming, outgoing = edge_shapes[i - 1], edge_shapes[i]
            p0, p1, p2 = incoming[-2], incoming[-1], outgoing[1]
            heading_in = math.degrees(math.atan2(p1[0] - p0[0], p1[1] - p0[1]))
            heading_out = math.degrees(math.atan2(p2[0] - p1[0], p2[1] - p1[1]))
            signed_angle = (heading_out - heading_in + 180) % 360 - 180
            absolute_angle = abs(signed_angle)
            if absolute_angle >= UTURN_MIN_DEG:
                turn_type = "U-turn"
            elif absolute_angle >= TURN_MIN_DEG:
                turn_type = "Right turn" if signed_angle > 0 else "Left turn"
            else:
                continue
            turn_records.append({"trip_id": trip_id, "turn_type": turn_type,
                                 "longitude": p1[0], "latitude": p1[1],
                                 "angle_deg": round(signed_angle, 1),
                                 "incoming_link": route_links[i - 1],
                                 "outgoing_link": route_links[i]})
    report(f"Prepared {len(route_by_key):,} trip routes and origin/destination records")

    od = pd.DataFrame(od_records)
    turns = pd.DataFrame(turn_records, columns=["trip_id", "turn_type", "longitude",
        "latitude", "angle_deg", "incoming_link", "outgoing_link"])
    links = pd.DataFrame(list(edge_meta.values()))
    blind_spots = pd.DataFrame()
    if not links.empty:
        links["trip_count"] = links["link_id"].map(lambda x: len(link_trip_sets[x]))
        links["od_pair_count"] = links["link_id"].map(lambda x: len(link_od_sets[x]))
        links["used_by_trip"] = links["trip_count"] > 0
    if not links.empty and include_blind_spots:
        # Visibility is NOT measured by these files. This conservative screen
        # only flags narrow roads with a sharp mapped bend or a complex junction
        # as places a person could inspect. It is not a verified blind-spot map.
        neighbours: dict[str, set[str]] = {}
        for edge in edge_meta.values():
            neighbours.setdefault(str(edge["u"]), set()).add(str(edge["v"]))
            neighbours.setdefault(str(edge["v"]), set()).add(str(edge["u"]))

        screening_rows: list[dict[str, object]] = []
        junction_counts: list[int] = []
        bend_values: list[float] = []
        candidate_flags: list[bool] = []
        reason_values: list[str] = []
        score_values: list[float] = []
        for edge in edge_meta.values():
            narrow = bool(edge["narrow_road"])
            # Bend geometry affects only eligibility of narrow roads. Skipping
            # every non-narrow shape avoids millions of needless Python steps
            # on the full Munakata road graph.
            shape = edge["geometry_coords"] if narrow else []
            bearings: list[float] = []
            for (lon1, lat1), (lon2, lat2) in zip(shape[:-1], shape[1:]):
                bearings.append(math.degrees(math.atan2(lon2 - lon1, lat2 - lat1)))
            bends = [abs((b - a + 180) % 360 - 180) for a, b in zip(bearings[:-1], bearings[1:])]
            max_bend = max(bends, default=0.0)
            junction_ends = sum(
                len(neighbours.get(str(node), set())) >= 3 for node in (edge["u"], edge["v"])
            )
            width = float(edge["width_m"])
            bend_candidate = narrow and max_bend >= 35.0
            junction_candidate = narrow and junction_ends == 2
            candidate = bool(bend_candidate or junction_candidate)
            if bend_candidate and junction_candidate:
                reason = "Narrow road + bend + junctions"
            elif bend_candidate:
                reason = "Narrow road + sharp bend"
            elif junction_candidate:
                reason = "Narrow road + junctions at both ends"
            else:
                reason = "No screening flag"
            score = (int(junction_ends) + min(max_bend, 90.0) / 90.0
                     + max(0.0, 5.5 - width) / 5.5) if candidate else 0.0
            junction_counts.append(int(junction_ends))
            bend_values.append(round(max_bend, 1))
            candidate_flags.append(candidate)
            reason_values.append(reason)
            score_values.append(round(score, 3))
            # Retain geometry only for flagged links; copying it for every one
            # of 100k+ links made the Streamlit page slow and memory-heavy.
            if candidate:
                screening_rows.append({
                    "link_id": edge["link_id"], "u": edge["u"], "v": edge["v"],
                    "highway": edge["highway"], "width_m": width,
                    "narrow_road": narrow, "junction_ends": int(junction_ends),
                    "max_bend_deg": round(max_bend, 1),
                    "blind_spot_candidate": candidate,
                    "screening_reason": reason,
                    "inspection_priority_score": round(score, 3),
                    "geometry_coords": shape,
                    "lon1": edge["lon1"], "lat1": edge["lat1"],
                    "lon2": edge["lon2"], "lat2": edge["lat2"],
                    "trip_count": len(link_trip_sets[edge["link_id"]]),
                    "od_pair_count": len(link_od_sets[edge["link_id"]]),
                    "used_by_trip": len(link_trip_sets[edge["link_id"]]) > 0,
                })
        links["junction_ends"] = junction_counts
        links["max_bend_deg"] = bend_values
        links["blind_spot_candidate"] = candidate_flags
        links["screening_reason"] = reason_values
        links["inspection_priority_score"] = score_values
        blind_spot_columns = [
            "link_id", "u", "v", "highway", "width_m", "narrow_road",
            "junction_ends", "max_bend_deg", "blind_spot_candidate",
            "screening_reason", "inspection_priority_score", "geometry_coords",
            "lon1", "lat1", "lon2", "lat2", "trip_count", "od_pair_count",
            "used_by_trip",
        ]
        blind_spots = pd.DataFrame(screening_rows, columns=blind_spot_columns)
        if not blind_spots.empty:
            blind_spots = blind_spots.sort_values(
                ["inspection_priority_score", "trip_count"], ascending=False
            ).reset_index(drop=True)
    report(f"Finished road screening; {len(blind_spots):,} candidate links")

    # Route/turn/OD pages need only the saved road graph and trip truth.
    # Avoid reading hundreds of thousands of GPS/event rows for those views.
    if not include_gps:
        return {"available": True, "message": "Synthetic route products are ready.",
                "origin_destination": od, "links": links, "turns": turns,
                "blind_spots": blind_spots, "speed_by_link": pd.DataFrame(),
                "speed_checks": pd.DataFrame(), "event_by_link": pd.DataFrame(),
                "event_samples": pd.DataFrame(), "traffic_by_link_time": pd.DataFrame(),
                "gps_vehicle_km_by_hour": pd.DataFrame(), "speed_summary": pd.DataFrame(),
                "metadata": metadata, "trip_truth_count": len(route_by_key)}

    # Read the recorded synthetic Form 1-2 GPS/speed observations.
    gps_parts: list[pd.DataFrame] = []
    gps_zip_paths = sorted(data_dir.rglob("*.zip"))
    report(f"Reading {len(gps_zip_paths):,} Form 1-2 GPS archives")
    for archive_number, zip_path in enumerate(gps_zip_paths, start=1):
        with zipfile.ZipFile(zip_path) as archive:
            members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if not members:
                continue
            with archive.open(members[0]) as stream:
                gps_parts.append(pd.read_csv(io.BytesIO(stream.read()), header=None,
                    names=FORM_12_COLUMNS, dtype={"Operation_date": "string",
                    "Operation_ID1": "string", "Trip_number": "string"}, low_memory=False))
        if archive_number % 10 == 0 or archive_number == len(gps_zip_paths):
            report(f"Read {archive_number:,}/{len(gps_zip_paths):,} Form 1-2 archives")
    if not gps_parts:
        return {"available": False, "message": "No Form 1-2 records were found in the ZIP files.",
                "origin_destination": od, "links": links, "turns": turns,
                "blind_spots": blind_spots if "blind_spots" in locals() else pd.DataFrame(),
                "speed_by_link": pd.DataFrame(), "event_samples": pd.DataFrame(),
                "speed_checks": pd.DataFrame(),
                "metadata": metadata}
    gps = pd.concat(gps_parts, ignore_index=True)
    for column in ("Longitude", "Latitude", "Speed", "Serial_number"):
        gps[column] = pd.to_numeric(gps[column], errors="coerce")
    gps["GPS_dt"] = pd.to_datetime(gps["GPS_time"], format="%Y%m%d%H%M%S", errors="coerce")
    gps = gps.dropna(subset=["Longitude", "Latitude", "Speed", "GPS_dt"])
    point_link_parts: list[np.ndarray] = []
    point_speed_parts: list[np.ndarray] = []
    point_time_parts: list[np.ndarray] = []
    point_trip_parts: list[np.ndarray] = []
    segment_link_parts: list[np.ndarray] = []
    segment_derived_speed_parts: list[np.ndarray] = []
    segment_distance_parts: list[np.ndarray] = []
    segment_hour_parts: list[np.ndarray] = []
    segment_near_stop_parts: list[np.ndarray] = []
    speed_check_records: list[dict[str, object]] = []

    for trip_number, (key_values, trip) in enumerate(
            gps.groupby(list(TRIP_KEY), sort=False, dropna=False), start=1):
        if trip_number % 250 == 0:
            report(f"Matching GPS samples to routes: {trip_number:,} trips processed")
        key = _key(*key_values)
        route = route_by_key.get(key)
        if route is None:
            continue
        trip = trip.sort_values("Serial_number").drop_duplicates(
            subset=["GPS_time", "Longitude", "Latitude"])
        if len(trip) < 2:
            continue
        route_coords = np.asarray(route["coords"], dtype=float)
        rx, ry = TO_UTM.transform(route_coords[:, 0], route_coords[:, 1])
        route_xy = np.column_stack([rx, ry])
        starts_xy, ends_xy = route_xy[:-1], route_xy[1:]
        link_ids = route["links"]
        lon = trip["Longitude"].to_numpy(float)
        lat = trip["Latitude"].to_numpy(float)
        speed = trip["Speed"].to_numpy(float)
        px, py = TO_UTM.transform(lon, lat)
        point_xy = np.column_stack([px, py])
        point_link_indices = _nearest_route_segment(point_xy, starts_xy, ends_xy)
        point_link_parts.append(np.asarray(link_ids, dtype=object)[point_link_indices])
        point_speed_parts.append(speed)
        point_time_parts.append(pd.DatetimeIndex(trip["GPS_dt"]).floor("15min").to_numpy())
        point_trip_parts.append(np.full(len(speed), str(route["trip_id"]), dtype=object))

        timestamps = pd.DatetimeIndex(trip["GPS_dt"]).asi8
        interval_seconds = np.diff(timestamps) / 1e9
        distance_m = _haversine_m(lon[:-1], lat[:-1], lon[1:], lat[1:])
        valid = (interval_seconds > 0) & np.isfinite(distance_m)
        derived = np.divide(distance_m * 3.6, interval_seconds,
                             out=np.full_like(distance_m, np.nan), where=valid)
        mid_lon, mid_lat = (lon[:-1] + lon[1:]) / 2, (lat[:-1] + lat[1:]) / 2
        mx, my = TO_UTM.transform(mid_lon, mid_lat)
        midpoint_xy = np.column_stack([mx, my])
        segment_link_indices = _nearest_route_segment(midpoint_xy, starts_xy, ends_xy)
        valid_indices = np.flatnonzero(valid)
        if len(valid_indices):
            segment_link_parts.append(np.asarray(link_ids, dtype=object)[segment_link_indices[valid]])
            segment_derived_speed_parts.append(derived[valid])
            segment_distance_parts.append(distance_m[valid])
            midpoint_ns = timestamps[:-1] + np.diff(timestamps) // 2
            segment_hour_parts.append(pd.DatetimeIndex(midpoint_ns[valid]).hour.to_numpy())
            # Proxy only: duration of a GPS interval whose two endpoint speeds are <= 5 km/h.
            # This is not engine-idling telemetry; ignore unusually long sampling gaps.
            near_stop = (speed[:-1] <= 5.0) & (speed[1:] <= 5.0) & (interval_seconds <= 900)
            segment_near_stop_parts.append(np.where(near_stop[valid], interval_seconds[valid] / 60.0, 0.0))

        elapsed_hours = interval_seconds[valid].sum() / 3600
        total_distance_km = distance_m[valid].sum() / 1000
        gps_distance_time_speed = total_distance_km / elapsed_hours if elapsed_hours > 0 else np.nan
        recorded_mean = float(np.nanmean(speed))
        speed_check_records.append({"trip_id": route["trip_id"],
            "gps_points": len(trip), "gps_distance_km": total_distance_km,
            "elapsed_minutes": elapsed_hours * 60,
            "mean_recorded_speed_kmh": recorded_mean,
            "gps_distance_over_time_kmh": gps_distance_time_speed,
            "difference_kmh": recorded_mean - gps_distance_time_speed
                if np.isfinite(gps_distance_time_speed) else np.nan,
            "valid_gps_intervals": int(valid.sum())})
    report(f"Matched GPS observations for {len(speed_check_records):,} trips")

    # Aggregate at once in pandas instead of mutating Python dictionaries for
    # every one of the half-million generated GPS records.
    if point_link_parts:
        gps_point_table = pd.DataFrame({
            "link_id": np.concatenate(point_link_parts),
            "speed_kmh": np.concatenate(point_speed_parts),
            "time_bin": np.concatenate(point_time_parts),
            "trip_id": np.concatenate(point_trip_parts),
        })
        link_speed_group = gps_point_table.groupby("link_id", sort=False)["speed_kmh"]
        speed_stats = link_speed_group.agg(
            gps_speed_samples="count", mean_recorded_speed_kmh="mean",
            median_recorded_speed_kmh="median", sd_recorded_speed_kmh="std",
        )
        speed_quantiles = link_speed_group.quantile([.05, .95]).unstack()
        speed_quantiles.columns = ["p05_recorded_speed_kmh", "p95_recorded_speed_kmh"]
        speed_stats = speed_stats.join(speed_quantiles)
        free_flow_by_link = pd.Series(
            {link_id: float(meta["assumed_free_flow_kmh"]) for link_id, meta in edge_meta.items()}
        )
        below_free_flow = gps_point_table["speed_kmh"] < .7 * gps_point_table["link_id"].map(free_flow_by_link)
        speed_stats["speed_below_70pct_freeflow_share"] = below_free_flow.groupby(
            gps_point_table["link_id"], sort=False
        ).mean()
        traffic_by_link_time = (
            gps_point_table.groupby(["link_id", "time_bin"], sort=False)
            .agg(sample_count=("speed_kmh", "size"),
                 speed_sum=("speed_kmh", "sum"),
                 probe_trip_count=("trip_id", "nunique"))
            .reset_index()
        )
        traffic_by_link_time["mean_sample_speed_kmh"] = (
            traffic_by_link_time["speed_sum"] / traffic_by_link_time["sample_count"]
        )
        traffic_by_link_time = traffic_by_link_time.drop(columns="speed_sum")
    else:
        speed_stats = pd.DataFrame()
        traffic_by_link_time = pd.DataFrame(columns=[
            "link_id", "time_bin", "sample_count", "probe_trip_count", "mean_sample_speed_kmh"
        ])

    if segment_link_parts:
        gps_segment_table = pd.DataFrame({
            "link_id": np.concatenate(segment_link_parts),
            "derived_speed_kmh": np.concatenate(segment_derived_speed_parts),
            "distance_m": np.concatenate(segment_distance_parts),
            "hour": np.concatenate(segment_hour_parts),
            "near_stop_interval_minutes": np.concatenate(segment_near_stop_parts),
        })
        derived_speed_stats = gps_segment_table.groupby("link_id", sort=False)["derived_speed_kmh"].agg(
            mean_gps_segment_speed_kmh="mean", gps_segment_samples="count"
        )
        gps_distance_m_by_link = gps_segment_table.groupby("link_id", sort=False)["distance_m"].sum()
        near_stop_minutes_by_link = gps_segment_table.groupby("link_id", sort=False)["near_stop_interval_minutes"].sum()
        gps_distance_m_by_hour = gps_segment_table.groupby("hour", sort=False)["distance_m"].sum()
    else:
        derived_speed_stats = pd.DataFrame()
        gps_distance_m_by_link = pd.Series(dtype=float)
        near_stop_minutes_by_link = pd.Series(dtype=float)
        gps_distance_m_by_hour = pd.Series(dtype=float)

    # Match Form 1-4 event-context samples to the saved road route for their
    # own trip, using the same nearest-centerline method as the GPS products.
    event_dir = data_dir.parent / "OUT1-4"
    event_counts: dict[str, dict[str, object]] = {}
    event_sample_records: list[dict[str, object]] = []
    if include_events and event_dir.is_dir():
        event_parts: list[pd.DataFrame] = []
        event_zip_paths = sorted(event_dir.rglob("*.zip"))
        report(f"Reading {len(event_zip_paths):,} Form 1-4 event archives")
        for archive_number, zip_path in enumerate(event_zip_paths, start=1):
            with zipfile.ZipFile(zip_path) as archive:
                members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
                if not members:
                    continue
                with archive.open(members[0]) as stream:
                    event_parts.append(pd.read_csv(io.BytesIO(stream.read()), header=None,
                        names=FORM_14_COLUMNS, dtype={"Operation_date": "string",
                        "Operation_ID1": "string", "Trip_number": "string"}, low_memory=False))
            if archive_number % 10 == 0 or archive_number == len(event_zip_paths):
                report(f"Read {archive_number:,}/{len(event_zip_paths):,} Form 1-4 archives")
        if event_parts:
            event_data = pd.concat(event_parts, ignore_index=True)
            for column in ("Longitude", "Latitude", "Longitudinal_acceleration",
                           "Lateral_acceleration", "Yaw_angular_velocity", "Speed"):
                event_data[column] = pd.to_numeric(event_data[column], errors="coerce")
            event_data = event_data.dropna(subset=["Longitude", "Latitude"])
            for event_trip_number, (key_values, event_trip) in enumerate(
                    event_data.groupby(list(TRIP_KEY), sort=False, dropna=False), start=1):
                if event_trip_number % 250 == 0:
                    report(f"Matching event samples to routes: {event_trip_number:,} trips processed")
                route = route_by_key.get(_key(*key_values))
                if route is None:
                    continue
                route_coords = np.asarray(route["coords"], dtype=float)
                rx, ry = TO_UTM.transform(route_coords[:, 0], route_coords[:, 1])
                route_xy = np.column_stack([rx, ry])
                starts_xy, ends_xy = route_xy[:-1], route_xy[1:]
                ex, ey = TO_UTM.transform(event_trip["Longitude"].to_numpy(float), event_trip["Latitude"].to_numpy(float))
                event_indices = _nearest_route_segment(np.column_stack([ex, ey]), starts_xy, ends_xy)
                for record_index, event_row in enumerate(event_trip.itertuples(index=False)):
                    link_id = route["links"][int(event_indices[record_index])]
                    item = event_counts.setdefault(link_id, {"event_context_records": 0,
                        "hard_braking_candidate_records": 0, "longitudinal_values": [], "speeds": []})
                    item["event_context_records"] = int(item["event_context_records"]) + 1
                    accel = getattr(event_row, "Longitudinal_acceleration")
                    speed_value = getattr(event_row, "Speed")
                    lateral = getattr(event_row, "Lateral_acceleration")
                    yaw = getattr(event_row, "Yaw_angular_velocity")
                    event_time = pd.to_datetime(getattr(event_row, "GPS_time"),
                        format="%Y%m%d%H%M%S", errors="coerce")
                    event_sample_records.append({
                        "trip_id": route["trip_id"], "link_id": link_id,
                        "GPS_dt": event_time,
                        "longitude": float(getattr(event_row, "Longitude")),
                        "latitude": float(getattr(event_row, "Latitude")),
                        "speed_kmh": float(speed_value) if pd.notna(speed_value) else np.nan,
                        "longitudinal_acceleration_mps2": float(accel) if pd.notna(accel) else np.nan,
                        "lateral_acceleration_mps2": float(lateral) if pd.notna(lateral) else np.nan,
                        "yaw_deg_s": float(yaw) if pd.notna(yaw) else np.nan,
                    })
                    if pd.notna(accel):
                        item["longitudinal_values"].append(float(accel))
                        if float(accel) <= -3.5:
                            item["hard_braking_candidate_records"] = int(item["hard_braking_candidate_records"]) + 1
                    if pd.notna(speed_value):
                        item["speeds"].append(float(speed_value))
            report(f"Matched {len(event_sample_records):,} event-context samples")

    speed_records: list[dict[str, object]] = []
    for link_id, meta in edge_meta.items():
        recorded = speed_stats.loc[link_id] if not speed_stats.empty and link_id in speed_stats.index else {}
        derived = derived_speed_stats.loc[link_id] if not derived_speed_stats.empty and link_id in derived_speed_stats.index else {}
        events = event_counts.get(link_id, {})
        link_km = float(gps_distance_m_by_link.get(link_id, 0.0)) / 1000
        candidate_records = int(events.get("hard_braking_candidate_records", 0))
        speed_records.append({**meta,
            "gps_speed_samples": int(recorded.get("gps_speed_samples", 0)),
            "mean_recorded_speed_kmh": recorded.get("mean_recorded_speed_kmh", np.nan),
            "median_recorded_speed_kmh": recorded.get("median_recorded_speed_kmh", np.nan),
            "p05_recorded_speed_kmh": recorded.get("p05_recorded_speed_kmh", np.nan),
            "p95_recorded_speed_kmh": recorded.get("p95_recorded_speed_kmh", np.nan),
            "sd_recorded_speed_kmh": recorded.get("sd_recorded_speed_kmh", np.nan),
            "speed_below_70pct_freeflow_share": recorded.get("speed_below_70pct_freeflow_share", np.nan),
            "mean_gps_segment_speed_kmh": derived.get("mean_gps_segment_speed_kmh", np.nan),
            "gps_segment_samples": int(derived.get("gps_segment_samples", 0)),
            "gps_vehicle_km": link_km,
            "near_stop_interval_minutes": float(near_stop_minutes_by_link.get(link_id, 0.0)),
            "event_context_records": int(events.get("event_context_records", 0)),
            "hard_braking_candidate_records": candidate_records,
            "candidate_records_per_1000_vehicle_km": 1000 * candidate_records / link_km if link_km > 0 else np.nan,
            "mean_event_longitudinal_acceleration_mps2": float(np.mean(events["longitudinal_values"])) if events.get("longitudinal_values") else np.nan,
            "mean_event_speed_kmh": float(np.mean(events["speeds"])) if events.get("speeds") else np.nan})

    speed_by_link = pd.DataFrame(speed_records)
    report("Finished link speed and traffic summaries")
    gps_vehicle_km_by_hour = pd.DataFrame([
        {"hour": int(hour), "gps_vehicle_km": float(distance_m) / 1000}
        for hour, distance_m in sorted(gps_distance_m_by_hour.items())
    ])
    speed_checks = pd.DataFrame(speed_check_records)
    if not speed_checks.empty:
        valid_diffs = speed_checks["difference_kmh"].dropna()
        speed_summary = pd.DataFrame([{
            "trips_checked": len(speed_checks),
            "mean_trip_recorded_speed_kmh": speed_checks["mean_recorded_speed_kmh"].mean(),
            "mean_trip_gps_distance_over_time_kmh": speed_checks["gps_distance_over_time_kmh"].mean(),
            "mean_signed_difference_kmh": valid_diffs.mean() if len(valid_diffs) else np.nan,
            "mean_absolute_difference_kmh": valid_diffs.abs().mean() if len(valid_diffs) else np.nan,
            "gps_points_checked": int(speed_checks["gps_points"].sum()),
            "gps_intervals_checked": int(speed_checks["valid_gps_intervals"].sum())}])
    else:
        speed_summary = pd.DataFrame()

    return {"available": True, "message": "Synthetic route products are ready.",
            "origin_destination": od, "links": links, "turns": turns,
            "blind_spots": blind_spots if "blind_spots" in locals() else pd.DataFrame(),
            "speed_by_link": speed_by_link, "speed_checks": speed_checks,
            "event_by_link": speed_by_link.loc[speed_by_link["event_context_records"] > 0].copy(),
            "event_samples": pd.DataFrame(event_sample_records),
            "traffic_by_link_time": traffic_by_link_time,
            "gps_vehicle_km_by_hour": gps_vehicle_km_by_hour,
            "speed_summary": speed_summary, "metadata": metadata,
            "gps_point_count": len(gps), "trip_truth_count": len(route_by_key),
            "matched_trip_count": len(speed_checks)}


