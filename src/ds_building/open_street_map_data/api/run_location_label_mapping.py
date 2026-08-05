"""Convert raw Overpass payloads into nearest-way location label CSVs.

When run as a script, this module reads saved Overpass payload JSON files,
selects the nearest relevant OSM way for each requested point, extracts
`smoothness` and `surface` tags, and writes the label outputs.
"""

import json
import math
import os
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import pandas as pd
from shapely.geometry import Point, LineString
from shapely.ops import transform
from pyproj import Transformer

from parameter_settings import HIGHWAY_ALLOWED_CAR_STREETS
from parameter_settings import MAX_POINT_DISTANCE_M, CLOSEST_WAYS_MARGIN_M


# ----------------------------------------------------------------------------------------------------------------------
# distance calc
# lon/lat -> metres (Berlin/Leipzig is typically UTM 33N)
_transformer = Transformer.from_crs("EPSG:4326", "EPSG:25833", always_xy=True)

def _min_distance_to_polyline(lat: float, lon: float, points: list[dict]) -> float:
    """Return the minimum planar distance (m) from (lat, lon) to a polyline.

    Builds a LineString from the input geometry (lon/lat), projects both the line
    and query point to metres (EPSG:25833), then returns the Shapely distance.

    Args:
        lat: Latitude of the query point.
        lon: Longitude of the query point.
        points: List of geometry dicts with `lat`/`lon` keys describing the way.

    Returns:
        Minimum distance in metres. Returns +inf if `points` is empty.
    """
    if not points:
        return float("inf")

    line_ll = LineString([(p["lon"], p["lat"]) for p in points])
    pt_ll = Point(lon, lat)

    # project to metres
    line_m = transform(_transformer.transform, line_ll)
    pt_m = transform(_transformer.transform, pt_ll)

    return pt_m.distance(line_m)  # metres

# ----------------------------------------------------------------------------------------------------------------------

def _delete_duplicate_elements(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate Overpass elements by (type, id) while preserving order."""
    seen: Set[Tuple[Any, Any]] = set()
    out: List[Dict[str, Any]] = []
    for el in elements:
        key = (el.get("type"), el.get("id"))
        if key in seen:
            continue
        seen.add(key)
        out.append(el)
    return out

def create_location_label(point:dict, ways:list[tuple[float, dict]], batch_id:str, doc_path:str) -> dict[str, Any] | None:
    """Create a single label record (smoothness/surface) for a point from candidate ways.

    Selects the closest way (or resolves ties) and returns a compact dict with the
    original point coordinates, chosen distance, and extracted `smoothness` and
    `surface` tags.

    Args:
        point: Dict containing `lat` and `lon` for the query location.
        ways: List of (distance_m, way_dict) candidates, expected sorted by distance.
        batch_id: Identifier for the source batch (used for duplicate/tie docs).
        doc_path: CSV path used to record tie-resolution details.

    Returns:
        Dict with keys: lat, lon, distance, smoothness, surface; or None if no usable
        way exists or the nearest way is farther than `MAX_POINT_DISTANCE_M`.
    """
    if not ways or len(ways) == 0:
        return None

    distance = ways[0][0]
    if distance > MAX_POINT_DISTANCE_M:
        print(f'found nearest element with a {distance:.2f}m distance. Way {ways[0][1]["id"]} is too far - dropping location lat: {point["lat"]}, lon: {point["lon"]}.')
        return None

    if len(ways) == 1:
        way_tags = ways[0][1]['tags'] or None
        location_label = {
            'lat': point['lat'],
            'lon': point['lon'],
            'distance': ways[0][0],
            'smoothness': way_tags.get('smoothness', None) if way_tags else None,
            'surface': way_tags.get('surface', None) if way_tags else None,
        }
        return location_label

    # multiple options
    else:
        entry_inf = _analyse_multiple_finalists(point['lat'], point['lon'], ways, batch_id, doc_path)
        characteristics = ('lat', 'lon', 'distance', 'smoothness', 'surface')
        formated = dict((k, entry_inf[k]) for k in characteristics if k in entry_inf)
        return formated


def find_closest_elements(point:dict, batch_payload:list[dict], batch_id:str) -> list[tuple[float, dict]] | None:
    """Find the closest way elements in a batch payload to a given point.

    Filters payload items to `type == "way"` with geometry, computes point-to-polyline
    distance for each, and returns all ways within `CLOSEST_WAYS_MARGIN_M` of the
    minimum distance.

    Args:
        point: Dict containing `lat` and `lon`.
        batch_payload: List of Overpass elements (dicts) for a batch.
        batch_id: Batch identifier used for debug output.

    Returns:
        List of (distance_m, way_dict) finalists, or None if no candidates exist.
    """
    # sub methods
    def _get_way_dist_for_batch(lat: float, lon: float, batch_payload: list[dict], batch_id: str) -> list[tuple[float, dict]]:
        """Compute (distance_m, way) for each geometry-bearing way in the batch."""
        candidates = []
        for way in batch_payload:
            # early exits if way is relevant at all
            if way['type'] != "way":
                print('skipping element because type is not way: ', way['type'], ', batch_file: ', batch_id)
                continue
            if 'geometry' in way.keys():
                dist_m = _min_distance_to_polyline(lat, lon, way.get("geometry"))  # adjust key if needed
                candidates.append((dist_m, way))
        return candidates

    def _calc_finalists(candidates: list[tuple[float, dict]]) -> list[tuple[float, dict]] | None:
        """Return all candidates within `CLOSEST_WAYS_MARGIN_M` of the minimum distance."""
        if not candidates:
            return None
        min_dist = min(d for d, _ in candidates)
        finalists = [(d, w) for d, w in candidates if math.isclose(d, min_dist, abs_tol=CLOSEST_WAYS_MARGIN_M)]

        if len(finalists) > 1:
            ids = [w.get("id") for _, w in finalists]
            print(f"[{batch_id}] tie: {len(finalists)} ways at ~{min_dist:.3f} m -> {point['lat']}, {point['lon']}, ids:{ids}")
        return finalists

    lon = point['lon']
    lat = point['lat']
    batch_payload = _delete_duplicate_elements(batch_payload)

    candidates = _get_way_dist_for_batch(lat, lon, batch_payload, batch_id)

    if not candidates:
        print(f'Could not find nearest element with the set API configuration - dropping location.')
        return None

    finalists = _calc_finalists(candidates)
    return finalists

def _analyse_multiple_finalists(lat, lon,
                                ways: list[tuple[float, dict]],
                                batch_id:str,
                                doc_save_path: str,
                                doc_df: pd.DataFrame = None) -> dict:
    """Resolve ties between multiple equally-close ways and persist tie metadata.

    Chooses `smoothness` and `surface` by a strict most-frequent vote; if there is a
    frequency tie (or no string values), the resolved value becomes None. Also writes
    an audit row to `doc_save_path`.

    Args:
        lat: Latitude of the query point.
        lon: Longitude of the query point.
        ways: List of (distance_m, way_dict) finalists.
        batch_id: Identifier of the source payload file.
        doc_save_path: CSV path used to log tie-resolution details.
        doc_df: Optional existing DataFrame to append to (otherwise loaded/created).

    Returns:
        Full entry dict containing resolved and raw candidate values.
    """
    def _most_frequent_val(values: Sequence[Any]) -> Optional[str]:
        """Return the unique most frequent string value, or None if tied/empty."""
        counts = Counter(v for v in values if isinstance(v, str))
        if not counts:
            return None
        max_n = counts.most_common(1)[0][1]
        winners = [k for k, n in counts.items() if n == max_n]
        return winners[0] if len(winners) == 1 else None

    smoothness_vals: List[str or None] = []
    surface_vals: List[str or None] = []
    for _, way in ways:
        tags = way.get("tags") or {}
        smoothness_vals.append(tags.get('smoothness') or None)
        surface_vals.append(tags.get('surface') or None)

    entry = {
        'lat': lat,
        'lon': lon,
        'distance': ways[0][0],
        'smoothness_vals': smoothness_vals,
        'smoothness': _most_frequent_val(smoothness_vals),
        'surface_vals': surface_vals,
        'surface': _most_frequent_val(surface_vals),
        'source_file': batch_id,
    }

    if doc_df is None:
        if os.path.exists(doc_save_path):
            doc_df = pd.read_csv(doc_save_path)
        else:
            os.makedirs(os.path.dirname(doc_save_path) or ".", exist_ok=True)
            doc_df = pd.DataFrame(columns=list(entry.keys()))

    doc_df.loc[len(doc_df)] = entry
    doc_df.to_csv(doc_save_path, index=False)
    return entry

def filter_relevant_streets(api_elements: list) -> list[dict]:
    """Filter Overpass way elements to those with car-drivable `highway` tags.

    Uses `HIGHWAY_ALLOWED_CAR_STREETS` as an allow-list and prints batch statistics.

    Args:
        api_elements: List of Overpass elements (dicts).

    Returns:
        List of way dicts considered relevant.
    """
    relevant_streets = []
    irrelevant_streets = []
    irrelevant_tags = set()
    without_highway_tag = 0
    for way_dict in api_elements:
        tags = way_dict.get("tags") or {}
        if 'highway' in tags.keys():
            if tags['highway'] in HIGHWAY_ALLOWED_CAR_STREETS:
                relevant_streets.append(way_dict)
            else:
                irrelevant_streets.append(way_dict)
                irrelevant_tags.add(tags['highway'])
        else:
            without_highway_tag += 1
    print(F"BATCH FILTERING: "
          F"\n\tRELEVANT STREETS: {len(relevant_streets)}, "
          F"\n\tIRRELEVANT STREETS: {len(irrelevant_streets)}, with the tags: {irrelevant_tags}"
          F"\n\tWAYS WITH HIGHWAY TAG: {without_highway_tag}")
    return relevant_streets

def process_crawled_data_to_nearest_location_label_df(
    payload_path: str,
    save_file: str,
    duplicates_doc_file: str,
    num_files: int | None = None,
) -> None:
    """Convert raw Overpass payload JSON files into a deduplicated label CSV.

    For each batch JSON in `payload_path`, loads the requested points and returned
    OSM elements, filters to relevant streets, finds nearest ways, and writes
    location labels to `save_file`. Duplicate (lat, lon) rows are dropped keeping
    only the last occurrence.

    Args:
        payload_path: Directory containing saved batch payload JSON files.
        save_file: Output CSV path for labeled locations.
        duplicates_doc_file: CSV path used to log tie-resolution metadata.
        num_files: Optional limit on the number of JSON files processed.

    Returns:
        None. Writes/updates `save_file` on disk.
    """
    labeled_locations_df = pd.DataFrame(columns=['lon', 'lat', 'distance', 'smoothness', 'surface'])
    file_counter = 0
    for batch_file in sorted(os.listdir(payload_path)):
        if num_files and file_counter == num_files: break

        if not batch_file.endswith(".json"):
            continue

        file_counter += 1
        file_path = os.path.join(payload_path, batch_file)
        with open(file_path, "r", encoding="utf-8") as f:
            batch_json = json.load(f)

        batch_points = batch_json["points"]
        relevant_streets = filter_relevant_streets(batch_json["elements"])
        rows = []
        thrown_points = []
        for p in batch_points:
            point = {
                'lat': p[1],
                'lon': p[0],
            }
            closest_elements = find_closest_elements(point, relevant_streets, batch_file)
            labeled_location = create_location_label(point, closest_elements, batch_file, duplicates_doc_file)
            if labeled_location:
                rows.append(labeled_location)
            else:
                thrown_points.append(p)

        if rows:
            print(f"\nBATCH SUMMARY: added {len(rows)} labeled locations, and threw away {len(thrown_points)} points. ")
            print(f"Thrown: {thrown_points}")
            print("-----------------------------------------------------------------------------------------------------")
            new_df = pd.DataFrame(rows)
            labeled_locations_df = pd.concat(
                [labeled_locations_df, new_df],
                ignore_index=True
            )
            labeled_locations_df = labeled_locations_df.drop_duplicates(subset=['lat', 'lon'], keep='last')
            labeled_locations_df.to_csv(save_file, index=False)
        else:
            print("-----------------------------------------------------------------------------------------------------")
            print(f"BATCH SUMMARY: added {len(rows)} labeled locations, and threw away {len(thrown_points)} points. ")
            print(f"Thrown: {thrown_points}", )
            print("-----------------------------------------------------------------------------------------------------")

labels_dir = "data/open_street_map/labels"
payload_dir = f'{labels_dir}/0_raw_api_data/payloads'
labeled_location_dir = f'{labels_dir}/1_labeled_location_data'
save_file = f'{labeled_location_dir}/labeled_locations.csv'
duplicates_doc_file= f'{labeled_location_dir}/duplicates.csv'
os.makedirs(labeled_location_dir, exist_ok=True)
process_crawled_data_to_nearest_location_label_df(payload_dir, save_file, duplicates_doc_file, num_files=None)
