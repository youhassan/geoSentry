"""Step 1A: Fetch, clean, or ingest building footprints for GeoSentry ROIs."""

import argparse
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import osmnx as ox
import pandas as pd
from shapely.affinity import translate
from shapely.geometry import Polygon
from shapely.validation import make_valid

from config import CRS_UTM_ALEXANDRIA, CRS_WGS84, DATA_RAW, DEFAULT_ROI, ROIS


def _extract_primary_source(sources_val: object) -> str:
    """Classify the upstream footprint provider from Overture's sources array."""
    s = str(sources_val)
    if "OpenStreetMap" in s:
        return "OSM"
    if "Google" in s:
        return "Google"
    if "Microsoft" in s:
        return "Microsoft"
    return "Other"


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
    """Download building footprints directly from OpenStreetMap via osmnx."""
    gdf = ox.features_from_bbox(bbox=(west, south, east, north), tags={"building": True})
    gdf["primary_source"] = "OSM"
    return gdf


def _clean_and_conflate_polygons(
    gdf: gpd.GeoDataFrame,
    source: str,
    shift_east_m: float = 0.0,
    shift_north_m: float = 0.0,
) -> gpd.GeoDataFrame:
    """Validate geometries, filter noise, suppress ML duplicates, and compute metric attributes."""
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf["geometry"] = gdf.geometry.apply(lambda g: make_valid(g) if not g.is_valid else g)
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    gdf = gdf.reset_index(drop=True)

    if gdf.crs is None:
        gdf = gdf.set_crs(CRS_WGS84)
    else:
        gdf = gdf.to_crs(CRS_WGS84)

    if source == "manual":
        gdf["primary_source"] = "Manual"
        gdf_utm = gdf.to_crs(CRS_UTM_ALEXANDRIA)
        gdf_utm["area_m2"] = gdf_utm.geometry.area
        gdf_utm = gdf_utm[gdf_utm["area_m2"] >= 10.0].copy().reset_index(drop=True)
    else:
        if "sources" in gdf.columns:
            gdf["primary_source"] = gdf["sources"].apply(_extract_primary_source)
        elif "primary_source" not in gdf.columns:
            gdf["primary_source"] = "OSM" if source == "osm" else "Overture"

        gdf_utm = gdf.to_crs(CRS_UTM_ALEXANDRIA)
        gdf_utm["area_m2"] = gdf_utm.geometry.area
        ext_area = gdf_utm.geometry.apply(
            lambda g: Polygon(g.exterior).area if g.geom_type == "Polygon" else g.area
        )
        mrr_area = gdf_utm.geometry.minimum_rotated_rectangle().area.replace(0, 1e-6)
        gdf_utm["rectangularity"] = (ext_area / mrr_area).clip(0.0, 1.0)

        if source == "osm_only":
            mask = (gdf_utm["primary_source"] == "OSM") & (gdf_utm["area_m2"] >= 25.0)
        else:
            is_osm = gdf_utm["primary_source"] == "OSM"
            keep_osm = is_osm & (gdf_utm["area_m2"] >= 25.0)
            keep_ml = (~is_osm) & (gdf_utm["area_m2"] >= 55.0) & (gdf_utm["rectangularity"] >= 0.65)
            mask = keep_osm | keep_ml

        gdf_utm = gdf_utm[mask].copy().reset_index(drop=True)

        osm_part = gdf_utm[gdf_utm["primary_source"] == "OSM"].copy()
        ml_part = gdf_utm[gdf_utm["primary_source"] != "OSM"].copy()

        if not osm_part.empty and not ml_part.empty:
            osm_union = osm_part.geometry.apply(
                lambda g: Polygon(g.exterior).buffer(1.5) if g.geom_type == "Polygon" else g.buffer(1.5)
            ).union_all()
            overlap_ratios = ml_part.geometry.apply(
                lambda geom: geom.intersection(osm_union).area / max(geom.area, 1e-6)
            )
            ml_part = ml_part[overlap_ratios < 0.15].copy()
            ml_part["geometry"] = ml_part.geometry.simplify(0.35, preserve_topology=True)
            gdf_utm = gpd.GeoDataFrame(
                pd.concat([osm_part, ml_part], ignore_index=True),
                crs=CRS_UTM_ALEXANDRIA,
            )

    if shift_east_m != 0.0 or shift_north_m != 0.0:
        print(
            f"[Step 1A] Applying UTM co-registration shift: "
            f"{shift_east_m:+.2f} m East, {shift_north_m:+.2f} m North..."
        )
        gdf_utm["geometry"] = gdf_utm.geometry.apply(
            lambda g: translate(g, xoff=shift_east_m, yoff=shift_north_m)
        )

    # Recompute exact metric area, perimeter, exterior rectangularity, and true centroids
    gdf_utm["area_m2"] = gdf_utm.geometry.area.round(2)
    gdf_utm["perimeter_m"] = gdf_utm.geometry.length.round(2)
    ext_area_clean = gdf_utm.geometry.apply(
        lambda g: Polygon(g.exterior).area if g.geom_type == "Polygon" else g.area
    )
    mrr_area_clean = gdf_utm.geometry.minimum_rotated_rectangle().area.replace(0, 1e-6)
    gdf_utm["rectangularity"] = (ext_area_clean / mrr_area_clean).clip(0.0, 1.0).round(3)

    centroids_utm = gdf_utm.geometry.centroid
    centroids_wgs = gpd.GeoSeries(centroids_utm, crs=CRS_UTM_ALEXANDRIA).to_crs(CRS_WGS84)

    gdf_clean = gdf_utm.to_crs(CRS_WGS84)
    gdf_clean["centroid_lon"] = centroids_wgs.x.round(6)
    gdf_clean["centroid_lat"] = centroids_wgs.y.round(6)
    return gdf_clean


def fetch_buildings(
    roi_key: str,
    source: str = "overture",
    input_file: str | None = None,
    shift_east_m: float = 0.0,
    shift_north_m: float = 0.0,
) -> Path:
    roi = ROIS[roi_key]
    west, south, east, north = roi["bbox"]

    tmp_overture = DATA_RAW / f"{roi_key}_overture_raw.geojson"
    if input_file:
        source = "manual"
        print(f"[Step 1A] Ingesting manually digitized footprints from {input_file}...")
        gdf_raw = gpd.read_file(input_file)
    elif source in ("overture", "osm_only"):
        print(f"[Step 1A] Fetching building footprints ({source}) for {roi['name']}...")
        gdf_raw = _fetch_overture(west, south, east, north, tmp_overture)
    else:
        print(f"[Step 1A] Fetching building footprints ({source}) for {roi['name']}...")
        gdf_raw = _fetch_osm(west, south, east, north)

    if gdf_raw.empty:
        raise RuntimeError(f"No building features returned for ROI '{roi_key}'.")

    raw_count = len(gdf_raw)
    gdf = _clean_and_conflate_polygons(
        gdf_raw,
        source=source,
        shift_east_m=shift_east_m,
        shift_north_m=shift_north_m,
    )

    if source != "manual":
        gdf = gdf[
            (gdf["centroid_lon"] >= west)
            & (gdf["centroid_lon"] <= east)
            & (gdf["centroid_lat"] >= south)
            & (gdf["centroid_lat"] <= north)
        ].reset_index(drop=True)

    # Assign deterministic building IDs
    prefix = roi["prefix"]
    gdf["building_id"] = [f"{prefix}_{i+1:05d}" for i in range(len(gdf))]
    gdf["roi"] = roi_key
    gdf["footprint_source"] = source

    for col in ("height", "num_floors", "building:levels", "class", "subtype"):
        if col in gdf.columns:
            gdf[col] = gdf[col].astype("object")

    keep_cols = [
        "building_id",
        "roi",
        "primary_source",
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
        "perimeter_m",
        "rectangularity",
        "centroid_lat",
        "centroid_lon",
        "geometry",
    ]
    existing_cols = [c for c in keep_cols if c in gdf.columns]
    gdf = gdf[existing_cols]

    out_path = DATA_RAW / f"{roi_key}_buildings.geojson"
    gdf.to_file(out_path, driver="GeoJSON")
    for tmp_f in (tmp_overture, Path(f"{tmp_overture}.state")):
        if tmp_f.exists():
            tmp_f.unlink()

    med_area = float(gdf["area_m2"].median()) if len(gdf) else 0.0
    med_rect = float(gdf["rectangularity"].median()) if len(gdf) else 0.0
    print(f"[Step 1A] Raw features: {raw_count} -> Cleaned polygons: {len(gdf)}")
    print(f"          Median Area: {med_area:.1f} m^2 | Median Rectangularity: {med_rect:.3f}")
    print(f"          Saved -> {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch or ingest building footprints for GeoSentry ROI.")
    parser.add_argument(
        "--roi",
        choices=list(ROIS.keys()) + ["all"],
        default=DEFAULT_ROI,
        help=f"Target ROI key (default: {DEFAULT_ROI})",
    )
    parser.add_argument(
        "--source",
        choices=["overture", "osm_only", "osm"],
        default="overture",
        help="Footprint source (default: overture)",
    )
    parser.add_argument(
        "--input-file",
        default=None,
        help="Path to a manually digitized GeoJSON file (e.g. exported from geojson.io or QGIS)",
    )
    parser.add_argument(
        "--shift-east-m",
        type=float,
        default=0.0,
        help="Optional Easting shift in meters (UTM Zone 36N) to co-register off-nadir tracings",
    )
    parser.add_argument(
        "--shift-north-m",
        type=float,
        default=0.0,
        help="Optional Northing shift in meters (UTM Zone 36N) to co-register off-nadir tracings",
    )
    args = parser.parse_args()

    targets = list(ROIS.keys()) if args.roi == "all" else [args.roi]
    for target in targets:
        fetch_buildings(
            target,
            source=args.source,
            input_file=args.input_file,
            shift_east_m=args.shift_east_m,
            shift_north_m=args.shift_north_m,
        )
