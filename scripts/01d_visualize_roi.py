"""Step 1D: Generate an interactive Folium map of the high-accuracy ROI, buildings, and collapses."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import math
import os
from pathlib import Path

from dotenv import load_dotenv
import folium
import geopandas as gpd
import requests

from config import DATA_GROUND_TRUTH, DATA_RAW, DEFAULT_ROI, OUTPUTS_DIR, PROJECT_ROOT, ROIS

load_dotenv(PROJECT_ROOT / ".env")
_MAPBOX_HEADERS = {
    "Referer": "https://geojson.io/",
    "Origin": "https://geojson.io",
    "User-Agent": "Mozilla/5.0",
}


def _deg2num(lat_deg: float, lon_deg: float, zoom: int) -> tuple[int, int]:
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def _cache_geojsonio_mapbox_tiles(west: float, south: float, east: float, north: float) -> None:
    """Download Mapbox Satellite & Satellite-Streets @2x tiles locally if MAPBOX_ACCESS_TOKEN
    is set in .env, so file:// HTML maps display imagery without 403 Referer blocks.
    """
    mapbox_token = os.getenv("MAPBOX_ACCESS_TOKEN", "").strip()
    if not mapbox_token:
        return

    pad_lon, pad_lat = 0.0035, 0.0030
    w, s, e, n = west - pad_lon, south - pad_lat, east + pad_lon, north + pad_lat

    tasks: list[tuple[str, Path]] = []
    for z in range(15, 20):
        x_min, y_max = _deg2num(s, w, z)
        x_max, y_min = _deg2num(n, e, z)
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                sat_path = OUTPUTS_DIR / "tiles" / "mapbox_sat" / str(z) / str(x) / f"{y}.jpg"
                if not sat_path.exists():
                    sat_url = (
                        f"https://api.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}@2x.jpg90"
                        f"?access_token={mapbox_token}"
                    )
                    tasks.append((sat_url, sat_path))

                hyb_path = OUTPUTS_DIR / "tiles" / "mapbox_hybrid" / str(z) / str(x) / f"{y}.png"
                if not hyb_path.exists():
                    hyb_url = (
                        f"https://api.mapbox.com/styles/v1/mapbox/satellite-streets-v12/tiles/256/{z}/{x}/{y}@2x"
                        f"?access_token={mapbox_token}"
                    )
                    tasks.append((hyb_url, hyb_path))

    if not tasks:
        return

    print(f"[Step 1D] Caching {len(tasks)} genuine geojson.io Mapbox @2x tiles locally...")

    def _download_one(item: tuple[str, Path]) -> None:
        url, dst = item
        dst.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(url, headers=_MAPBOX_HEADERS, timeout=15)
        if resp.status_code == 200 and len(resp.content) > 1000:
            dst.write_bytes(resp.content)

    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(_download_one, tasks))


def build_roi_map(roi_key: str) -> None:
    roi = ROIS[roi_key]
    west, south, east, north = roi["bbox"]
    center_lat = (south + north) / 2
    center_lon = (west + east) / 2

    # Cache genuine geojson.io Mapbox tiles locally for exact 0-pixel parity
    _cache_geojsonio_mapbox_tiles(west, south, east, north)

    m = folium.Map(location=[center_lat, center_lon], zoom_start=17, max_zoom=22, tiles=None)

    # Street Map (Esri)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Street Map",
        name="Street Map (Esri)",
        max_native_zoom=19,
        max_zoom=22,
        overlay=False,
        control=True,
        show=False,
    ).add_to(m)

    # Google High-Resolution Satellite Imagery
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite",
        name="Satellite (Google High-Res)",
        max_native_zoom=21,
        max_zoom=22,
        overlay=False,
        control=True,
        show=False,
    ).add_to(m)

    # Genuine geojson.io Mapbox Clean Satellite (Cached locally from api.mapbox.com)
    folium.TileLayer(
        tiles="tiles/mapbox_sat/{z}/{x}/{y}.jpg",
        attr="Mapbox Satellite (Exact geojson.io tiles)",
        name="geojson.io Satellite (Clean)",
        min_zoom=15,
        max_native_zoom=19,
        max_zoom=22,
        overlay=False,
        control=True,
        show=False,
    ).add_to(m)

    # Genuine geojson.io Mapbox Satellite-Streets (Default visible basemap - exact match to geojson.io)
    folium.TileLayer(
        tiles="tiles/mapbox_hybrid/{z}/{x}/{y}.png",
        attr="Mapbox Satellite Streets (Exact geojson.io tiles)",
        name="geojson.io Satellite + Labels (Exact)",
        min_zoom=15,
        max_native_zoom=19,
        max_zoom=22,
        overlay=False,
        control=True,
        show=True,
    ).add_to(m)

    # ROI bounding box
    folium.Rectangle(
        bounds=[[south, west], [north, east]],
        color="#38bdf8",
        weight=2,
        fill=False,
        tooltip=f"ROI: {roi['name']}",
    ).add_to(m)

    # Ground-truth collapses (collect matched building IDs first)
    gt_path = DATA_GROUND_TRUTH / "alexandria_collapses.geojson"
    collapsed_bldg_ids: set[str] = set()
    gt = None
    if gt_path.exists():
        gt = gpd.read_file(gt_path)
        if "matched_building_id" in gt.columns:
            collapsed_bldg_ids = set(gt["matched_building_id"].dropna().astype(str).tolist())

    # Building footprints
    buildings_path = DATA_RAW / f"{roi_key}_buildings.geojson"
    if buildings_path.exists():
        bldgs = gpd.read_file(buildings_path)
        for col, default_val in [("primary_source", "OSM"), ("rectangularity", 1.0)]:
            if col not in bldgs.columns:
                bldgs[col] = default_val

        def _style_fn(feature: dict) -> dict:
            props = feature.get("properties", {})
            bid = str(props.get("building_id", ""))
            src = str(props.get("primary_source", "OSM"))
            if bid in collapsed_bldg_ids:
                return {
                    "fillColor": "#ef4444",
                    "color": "#b91c1c",
                    "weight": 2.2,
                    "fillOpacity": 0.65,
                }
            if src in ("OSM", "Manual"):
                return {
                    "fillColor": "#38bdf8",
                    "color": "#0284c7",
                    "weight": 1.4,
                    "fillOpacity": 0.35,
                }
            return {
                "fillColor": "#fbbf24",
                "color": "#d97706",
                "weight": 1.0,
                "fillOpacity": 0.30,
            }

        cols = [
            c
            for c in ["building_id", "primary_source", "area_m2", "rectangularity", "geometry"]
            if c in bldgs.columns
        ]
        aliases_map = {
            "building_id": "Building ID:",
            "primary_source": "Verified Source:",
            "area_m2": "Footprint Area (m²):",
            "rectangularity": "Rectangularity (0-1):",
        }
        tooltip_fields = [c for c in cols if c != "geometry"]
        tooltip_aliases = [aliases_map[c] for c in tooltip_fields]

        folium.GeoJson(
            bldgs[cols],
            name=f"Building Footprints ({len(bldgs)})",
            style_function=_style_fn,
            tooltip=folium.GeoJsonTooltip(fields=tooltip_fields, aliases=tooltip_aliases),
        ).add_to(m)

    # Render Ground-truth collapse markers
    if gt is not None:
        for _, row in gt.iterrows():
            matched_str = (
                f"<br>Matched Footprint: <b>{row.get('matched_building_id')}</b>"
                if row.get("matched_building_id")
                else ""
            )
            popup_html = (
                f"<b>{row['incident_id']}</b><br>"
                f"Date: {row['date']}<br>"
                f"District: {row['district']}<br>"
                f"Street: {row['street']}<br>"
                f"Type: {row['collapse_type']}"
                f"{matched_str}<br>"
                f"Notes: {row['notes']}"
            )
            folium.Marker(
                location=[row["lat"], row["lon"]],
                popup=folium.Popup(popup_html, max_width=320),
                tooltip=f"{row['date']} - {row['street']}",
                icon=folium.Icon(color="red", icon="exclamation-sign"),
            ).add_to(m)

    m.fit_bounds([[south, west], [north, east]])
    folium.LayerControl(collapsed=False).add_to(m)
    out_html = OUTPUTS_DIR / f"step1_{roi_key}_roi_map.html"
    m.save(str(out_html))
    print(f"[Step 1D] Saved interactive ROI map -> {out_html}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize GeoSentry Step 1 ROI.")
    parser.add_argument("--roi", choices=list(ROIS.keys()) + ["all"], default=DEFAULT_ROI)
    args = parser.parse_args()
    targets = [k for k in ROIS.keys() if k != "sidi_bishr"] if args.roi == "all" else [args.roi]
    for target in targets:
        build_roi_map(target)
