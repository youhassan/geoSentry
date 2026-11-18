"""Step 1A: Fetch 2D building footprints from Overture Maps (Google + Microsoft + OSM) or OSM."""

import argparse
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import osmnx as ox

from config import CRS_UTM_ALEXANDRIA, CRS_WGS84, DATA_RAW, DEFAULT_ROI, ROIS


def _fetch_overture(west: float, south: float, east: float, north: float, tmp_path: Path) -> gpd.GeoDataFrame:
    """Download building footprints for the bbox using the overturemaps CLI."""
    bbox_str = f"{west},{south},{east},{north}"
    cmd = [
        sys.executable,
        "-m",
        "overturemaps",
        "download",
        f"--bbox={bbox_str}",
        "-f",
        "geojson",
        "--type=building",
        "-o",
        str(tmp_path),
    ]
    print(f"[Step 1A] Running Overture Maps download for bbox={bbox_str}...")
    subprocess.run(cmd, check=True)
    return gpd.read_file(tmp_path)


def _fetch_osm(west: float, south: float, east: float, north: float) -> gpd.GeoDataFrame:
    """Download building footprints from OpenStreetMap via osmnx."""
    return ox.features_from_bbox(bbox=(west, south, east, north), tags={"building": True})


def fetch_buildings(roi_key: str, source: str = "overture") -> Path:
    roi = ROIS[roi_key]
    west, south, east, north = roi["bbox"]
    print(f"[Step 1A] Fetching building footprints ({source}) for {roi['name']}...")
    print(f"          BBox (W, S, E, N): {west}, {south}, {east}, {north}")

    tmp_overture = DATA_RAW / f"{roi_key}_overture_raw.geojson"
    if source == "overture":
        gdf = _fetch_overture(west, south, east, north, tmp_overture)
    else:
        gdf = _fetch_osm(west, south, east, north)

    if gdf.empty:
        raise RuntimeError(f"No building features returned for ROI '{roi_key}' from {source}.")

    # Keep only polygon footprints (drop points/nodes)
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    gdf = gdf.reset_index(drop=True)

    # Ensure WGS84, then compute metric area in UTM Zone 36N
    if gdf.crs is None:
        gdf = gdf.set_crs(CRS_WGS84)
    else:
        gdf = gdf.to_crs(CRS_WGS84)

    gdf_utm = gdf.to_crs(CRS_UTM_ALEXANDRIA)
    gdf["area_m2"] = gdf_utm.geometry.area.round(2)

    # Filter out tiny artifacts (< 15 m^2)
    gdf = gdf[gdf["area_m2"] >= 15.0].reset_index(drop=True)

    # Compute centroids in WGS84
    centroids_utm = gdf_utm.loc[gdf.index].geometry.centroid
    centroids_wgs = gpd.GeoSeries(centroids_utm, crs=CRS_UTM_ALEXANDRIA).to_crs(CRS_WGS84)
    gdf["centroid_lon"] = centroids_wgs.x.round(6)
    gdf["centroid_lat"] = centroids_wgs.y.round(6)

    # Assign deterministic building IDs
    prefix = roi["prefix"]
    gdf["building_id"] = [f"{prefix}_{i+1:05d}" for i in range(len(gdf))]
    gdf["roi"] = roi_key
    gdf["footprint_source"] = source

    # Normalize optional height / floors columns if present
    for col in ("height", "num_floors", "building:levels", "class", "subtype"):
        if col in gdf.columns:
            gdf[col] = gdf[col].astype("object")

    keep_cols = [
        "building_id",
        "roi",
        "footprint_source",
        "class",
        "subtype",
        "height",
        "num_floors",
        "building",
        "name",
        "building:levels",
        "addr:street",
        "area_m2",
        "centroid_lat",
        "centroid_lon",
        "geometry",
    ]
    existing_cols = [c for c in keep_cols if c in gdf.columns]
    gdf = gdf[existing_cols]

    out_path = DATA_RAW / f"{roi_key}_buildings.geojson"
    gdf.to_file(out_path, driver="GeoJSON")
    if tmp_overture.exists():
        tmp_overture.unlink()

    print(f"[Step 1A] Saved {len(gdf)} building polygons -> {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch building footprints for GeoSentry ROI.")
    parser.add_argument(
        "--roi",
        choices=list(ROIS.keys()) + ["all"],
        default=DEFAULT_ROI,
        help="Target ROI key (default: sidi_bishr)",
    )
    parser.add_argument(
        "--source",
        choices=["overture", "osm"],
        default="overture",
        help="Footprint source (default: overture)",
    )
    args = parser.parse_args()

    targets = list(ROIS.keys()) if args.roi == "all" else [args.roi]
    for target in targets:
        fetch_buildings(target, source=args.source)
