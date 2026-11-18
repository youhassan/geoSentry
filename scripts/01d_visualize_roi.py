"""Step 1D: Generate an interactive Folium map of the ROI, buildings, and collapses."""

import argparse
import folium
import geopandas as gpd

from config import DATA_GROUND_TRUTH, DATA_RAW, DEFAULT_ROI, OUTPUTS_DIR, ROIS


def build_roi_map(roi_key: str) -> None:
    roi = ROIS[roi_key]
    west, south, east, north = roi["bbox"]
    center_lat = (south + north) / 2
    center_lon = (west + east) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles=None)

    # Use Esri basemaps (work reliably from file:// without Referer blocks or API keys)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Street Map",
        name="Street Map (Esri)",
        overlay=False,
        control=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite (Esri)",
        overlay=False,
        control=True,
    ).add_to(m)

    # ROI bounding box
    folium.Rectangle(
        bounds=[[south, west], [north, east]],
        color="#2563eb",
        weight=2,
        fill=True,
        fill_opacity=0.05,
        tooltip=f"ROI: {roi['name']}",
    ).add_to(m)

    # Building footprints
    buildings_path = DATA_RAW / f"{roi_key}_buildings.geojson"
    if buildings_path.exists():
        bldgs = gpd.read_file(buildings_path)
        folium.GeoJson(
            bldgs[["building_id", "area_m2", "geometry"]],
            name="Building Footprints",
            style_function=lambda _: {
                "fillColor": "#64748b",
                "color": "#334155",
                "weight": 0.7,
                "fillOpacity": 0.45,
            },
            tooltip=folium.GeoJsonTooltip(fields=["building_id", "area_m2"], aliases=["ID:", "Area (m²):"]),
        ).add_to(m)

    # Ground-truth collapses
    gt_path = DATA_GROUND_TRUTH / "alexandria_collapses.geojson"
    if gt_path.exists():
        gt = gpd.read_file(gt_path)
        for _, row in gt.iterrows():
            popup_html = (
                f"<b>{row['incident_id']}</b><br>"
                f"Date: {row['date']}<br>"
                f"District: {row['district']}<br>"
                f"Street: {row['street']}<br>"
                f"Type: {row['collapse_type']}<br>"
                f"Notes: {row['notes']}"
            )
            folium.Marker(
                location=[row["lat"], row["lon"]],
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"{row['date']} - {row['street']}",
                icon=folium.Icon(color="red", icon="exclamation-sign"),
            ).add_to(m)

    m.fit_bounds([[south, west], [north, east]])
    folium.LayerControl().add_to(m)
    out_html = OUTPUTS_DIR / f"step1_{roi_key}_roi_map.html"
    m.save(str(out_html))
    print(f"[Step 1D] Saved interactive ROI map -> {out_html}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize GeoSentry Step 1 ROI.")
    parser.add_argument("--roi", choices=list(ROIS.keys()), default=DEFAULT_ROI)
    args = parser.parse_args()
    build_roi_map(args.roi)
