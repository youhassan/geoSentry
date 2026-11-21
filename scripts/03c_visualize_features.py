"""Step 3C: Generate diagnostic plots and an interactive Folium map of Step 3 features."""

import argparse
from pathlib import Path

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib import cm

from config import DATA_GROUND_TRUTH, DATA_PROCESSED, DEFAULT_ROI, OUTPUTS_DIR, ROIS


def _velocity_color(vel: float) -> str:
    """Map LOS velocity (mm/yr) to a diverging Red-Yellow-Green-Blue hex color."""
    # Negative = subsiding (red/orange), near 0 = stable (green), positive = uplift/phase jump (blue)
    if vel <= -3.0:
        return "#dc2626"
    if vel <= -1.5:
        return "#f97316"
    if vel < 1.5:
        return "#22c55e"
    if vel < 3.0:
        return "#38bdf8"
    return "#2563eb"


def build_diagnostic_plots(roi_key: str) -> Path:
    """Create a 4-panel diagnostic figure of Step 3 features and STL curves."""
    out_dir = DATA_PROCESSED / roi_key
    feat_df = pd.read_parquet(out_dir / f"building_features_{roi_key}.parquet")
    stl_df = pd.read_parquet(out_dir / f"building_stl_{roi_key}.parquet")
    stl_df["date"] = pd.to_datetime(stl_df["date"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    fig.suptitle(f"GeoSentry Step 3 Diagnostics — {ROIS[roi_key]['name']}", fontsize=13, fontweight="bold")

    # Panel 1: Mean Velocity vs Neighborhood Z-Score
    ax = axes[0, 0]
    norm = feat_df[~feat_df["is_collapsed_gt"]]
    gt = feat_df[feat_df["is_collapsed_gt"]]
    ax.scatter(
        norm["vel_mean_mm_yr"],
        norm["neighborhood_z_score"],
        c=norm["diff_tilt_mm_yr"],
        cmap="viridis",
        alpha=0.8,
        s=55,
        edgecolors="k",
        linewidths=0.4,
        label="Buildings",
    )
    if not gt.empty:
        ax.scatter(
            gt["vel_mean_mm_yr"],
            gt["neighborhood_z_score"],
            color="#ef4444",
            marker="*",
            s=220,
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
            label="Historical Collapse (GT)",
        )
        for _, r in gt.iterrows():
            ax.annotate(
                f"{r['building_id']} ({r['collapse_incident_id']})",
                (r["vel_mean_mm_yr"], r["neighborhood_z_score"]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
                fontweight="bold",
                color="#b91c1c",
            )
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Mean LOS Velocity (mm/yr)")
    ax.set_ylabel("Neighborhood Z-Score (100m radius)")
    ax.set_title("Velocity vs. 100m Neighborhood Deviation")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)

    # Panel 2: Differential Tilt vs Peak 120d Velocity Shift
    ax = axes[0, 1]
    ax.scatter(
        norm["diff_tilt_mm_yr"],
        norm["peak_vel_delta_120d_mm_yr"].abs(),
        alpha=0.75,
        s=50,
        color="#0284c7",
        edgecolors="k",
        linewidths=0.4,
        label="Buildings",
    )
    if not gt.empty:
        ax.scatter(
            gt["diff_tilt_mm_yr"],
            gt["peak_vel_delta_120d_mm_yr"].abs(),
            color="#ef4444",
            marker="*",
            s=220,
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
            label="Historical Collapse (GT)",
        )
        for _, r in gt.iterrows():
            ax.annotate(
                r["building_id"],
                (r["diff_tilt_mm_yr"], abs(r["peak_vel_delta_120d_mm_yr"])),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=8,
                fontweight="bold",
                color="#b91c1c",
            )
    ax.set_xlabel("Internal Differential Tilt (mm/yr)")
    ax.set_ylabel("Max |120d Velocity Shift| (mm/yr)")
    ax.set_title("Differential Settlement vs. Peak Acceleration Shift")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)

    # Panel 3 & 4: STL Decomposition for Ground-Truth Buildings (or top 2 by |Z|)
    if not gt.empty:
        target_bids = gt["building_id"].tolist()[:2]
    else:
        target_bids = feat_df.reindex(feat_df["neighborhood_z_score"].abs().sort_values(ascending=False).index)["building_id"].tolist()[:2]

    for idx, bid in enumerate(target_bids):
        ax = axes[1, idx]
        sub = stl_df[stl_df["building_id"] == bid].sort_values("date")
        row_meta = feat_df[feat_df["building_id"] == bid].iloc[0]
        ax.plot(sub["date"], sub["raw_disp_mm"], color="#94a3b8", linewidth=1.0, alpha=0.8, label="Raw D(t)")
        ax.plot(sub["date"], sub["trend_mm"], color="#dc2626", linewidth=2.0, label="Trend T(t)")
        ax.plot(sub["date"], sub["seasonal_mm"], color="#16a34a", linewidth=1.1, linestyle="--", label="Seasonal S(t)")
        if pd.notna(row_meta.get("collapse_date")):
            c_dt = pd.to_datetime(row_meta["collapse_date"])
            ax.axvline(c_dt, color="#000000", linestyle=":", linewidth=1.5, label=f"Collapse ({row_meta['collapse_date']})")
        ax.set_title(
            f"STL: {bid} | v={row_meta['vel_mean_mm_yr']:.2f} mm/yr, "
            f"tilt={row_meta['diff_tilt_mm_yr']:.2f}, Z={row_meta['neighborhood_z_score']:.2f}"
        )
        ax.set_ylabel("LOS Displacement (mm)")
        ax.legend(loc="best", fontsize=7.5)
        ax.grid(alpha=0.25)
        ax.tick_params(axis="x", rotation=25)

    plt.tight_layout()
    fig_path = OUTPUTS_DIR / f"step3_{roi_key}_feature_diagnostics.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"[Step 3C] Saved diagnostic plot -> {fig_path}")
    return fig_path


def build_features_map(roi_key: str) -> Path:
    """Create an interactive Folium map of buildings colored by Step 3 features + PS points."""
    out_dir = DATA_PROCESSED / roi_key
    bldgs = gpd.read_file(out_dir / f"building_features_{roi_key}.geojson")
    links = pd.read_parquet(out_dir / f"ps_building_links_{roi_key}.parquet")

    west, south, east, north = ROIS[roi_key]["bbox"]
    center_lat, center_lon = (south + north) / 2, (west + east) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=17, max_zoom=22, tiles=None)

    folium.TileLayer(
        tiles="tiles/mapbox_hybrid/{z}/{x}/{y}.png",
        attr="Mapbox Satellite Streets (Exact geojson.io tiles)",
        name="geojson.io Satellite + Labels",
        min_zoom=15,
        max_native_zoom=19,
        max_zoom=22,
        overlay=False,
        control=True,
        show=True,
    ).add_to(m)

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

    # Buildings Layer
    bldg_group = folium.FeatureGroup(name="Buildings (Colored by Mean LOS Velocity)", show=True)
    for _, r in bldgs.iterrows():
        color = _velocity_color(float(r["vel_mean_mm_yr"]))
        is_gt = bool(r["is_collapsed_gt"])
        popup_html = (
            f"<b>{r['building_id']}</b>"
            + (f" <span style='color:#dc2626'>(GT COLLAPSE: {r['collapse_incident_id']} @ {r['collapse_date']})</span>" if is_gt else "")
            + f"<br>PS Count: <b>{r['ps_count']}</b> ({r['join_type']})"
            + f"<br>Mean Coherence: <b>{r['mean_coherence']:.3f}</b>"
            + f"<br>Mean Velocity: <b>{r['vel_mean_mm_yr']:.2f} mm/yr</b>"
            + f"<br>Diff Tilt: <b>{r['diff_tilt_mm_yr']:.2f} mm/yr</b>"
            + f"<br>120d Vel Delta: <b>{r['vel_delta_120d_mm_yr']:.2f} mm/yr</b>"
            + f"<br>Peak 120d Delta: <b>{r['peak_vel_delta_120d_mm_yr']:.2f} mm/yr</b> ({r['peak_vel_delta_date']})"
            + f"<br>Neighborhood Z (100m): <b>{r['neighborhood_z_score']:.2f}</b>"
            + f"<br>STL Residual RMS (90d): <b>{r['residual_last_90d_rms_mm']:.2f} mm</b>"
        )
        folium.GeoJson(
            data=r.geometry.__geo_interface__,
            style_function=lambda _x, col=color, gt=is_gt: {
                "fillColor": col,
                "color": "#ef4444" if gt else "#111827",
                "weight": 3.0 if gt else 1.2,
                "fillOpacity": 0.55,
            },
            tooltip=f"{r['building_id']}: v={r['vel_mean_mm_yr']:.2f} mm/yr | tilt={r['diff_tilt_mm_yr']:.2f} | Z={r['neighborhood_z_score']:.2f}",
            popup=folium.Popup(popup_html, max_width=320),
        ).add_to(bldg_group)
    bldg_group.add_to(m)

    # Linked PS Points Layer
    ps_group = folium.FeatureGroup(name="Linked PS Scatterers", show=True)
    for _, r in links.drop_duplicates(subset=["PS_ID", "building_id"]).iterrows():
        c = _velocity_color(float(r["Mean_Velocity_mm_yr"]))
        folium.CircleMarker(
            location=[float(r["Lat"]), float(r["Lon"])],
            radius=3.5,
            color="#ffffff",
            weight=0.8,
            fill=True,
            fill_color=c,
            fill_opacity=0.9,
            tooltip=f"{r['PS_ID']} -> {r['building_id']} (v={r['Mean_Velocity_mm_yr']:.2f} mm/yr, coh={r['Coherence']:.3f})",
        ).add_to(ps_group)
    ps_group.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    map_path = OUTPUTS_DIR / f"step3_{roi_key}_features_map.html"
    m.save(str(map_path))
    print(f"[Step 3C] Saved interactive map -> {map_path}")
    return map_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 3C: Visualize Step 3 features & STL decomposition.")
    parser.add_argument("--roi", choices=list(ROIS.keys()), default=DEFAULT_ROI)
    args = parser.parse_args()

    build_diagnostic_plots(args.roi)
    build_features_map(args.roi)


if __name__ == "__main__":
    main()
