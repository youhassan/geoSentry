"""Step 3A: Coherence filtering & spatial join of PS-InSAR points to building footprints."""

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from config import CRS_UTM_ALEXANDRIA, CRS_WGS84, DATA_PROCESSED, DATA_RAW, DEFAULT_ROI, ROIS


def run_spatial_join(
    roi_key: str,
    min_coherence: float = 0.70,
    buffer_m: float = 5.0,
    max_nearest_m: float = 15.0,
) -> tuple[Path, Path]:
    """Filter PS points by coherence and spatially join them to building polygons."""
    ps_csv = DATA_PROCESSED / roi_key / f"ps_points_{roi_key}.csv"
    bldg_geojson = DATA_RAW / f"{roi_key}_buildings.geojson"

    if not ps_csv.exists():
        raise FileNotFoundError(f"PS points CSV not found: {ps_csv}")
    if not bldg_geojson.exists():
        raise FileNotFoundError(f"Buildings GeoJSON not found: {bldg_geojson}")

    ps_df = pd.read_csv(ps_csv)
    date_cols = [c for c in ps_df.columns if c.startswith("d_")]
    if not date_cols:
        raise ValueError(f"No displacement date columns (d_YYYYMMDD) found in {ps_csv}")

    total_ps = len(ps_df)
    ps_df = ps_df[ps_df["Coherence"] >= min_coherence].copy()
    print(
        f"[Step 3A] Coherence filter (>= {min_coherence:.2f}): "
        f"kept {len(ps_df)}/{total_ps} PS points."
    )

    ps_gdf = gpd.GeoDataFrame(
        ps_df,
        geometry=[Point(lon, lat) for lon, lat in zip(ps_df["Lon"], ps_df["Lat"])],
        crs=CRS_WGS84,
    ).to_crs(CRS_UTM_ALEXANDRIA)

    bldgs_gdf = gpd.read_file(bldg_geojson).to_crs(CRS_UTM_ALEXANDRIA)
    if bldgs_gdf.empty:
        raise ValueError(f"No building polygons found in {bldg_geojson}")

    # 1. Buffer-based spatial join (captures roof-edge scatterers due to SAR side-looking geometry)
    bldgs_buf = bldgs_gdf[["building_id", "geometry"]].copy()
    bldgs_buf["geometry"] = bldgs_buf.geometry.buffer(buffer_m)

    buf_joined = gpd.sjoin(
        ps_gdf,
        bldgs_buf,
        how="inner",
        predicate="within",
    ).drop(columns=["index_right"])

    # Compute exact distance from each PS point to the unbuffered building polygon
    bldg_geom_map = bldgs_gdf.set_index("building_id")["geometry"].to_dict()
    buf_joined["dist_to_building_m"] = [
        round(float(row.geometry.distance(bldg_geom_map[row["building_id"]])), 3)
        for _, row in buf_joined.iterrows()
    ]
    buf_joined["join_type"] = f"buffer_{buffer_m:.0f}m"

    # 2. Nearest-PS fallback for buildings with 0 points inside the buffer
    matched_bldgs = set(buf_joined["building_id"].unique())
    missing_bldgs = bldgs_gdf[~bldgs_gdf["building_id"].isin(matched_bldgs)].copy()

    links_parts = [buf_joined]
    if not missing_bldgs.empty:
        nearest = gpd.sjoin_nearest(
            missing_bldgs[["building_id", "geometry"]],
            ps_gdf,
            how="left",
            max_distance=max_nearest_m,
            distance_col="dist_to_building_m",
        )
        nearest = nearest.dropna(subset=["PS_ID"]).copy()
        if not nearest.empty:
            nearest["dist_to_building_m"] = nearest["dist_to_building_m"].round(3)
            nearest["join_type"] = "nearest_fallback"
            # Align columns and geometry to PS points
            ps_geom_map = ps_gdf.set_index("PS_ID")["geometry"].to_dict()
            nearest["geometry"] = [ps_geom_map[pid] for pid in nearest["PS_ID"]]
            links_parts.append(nearest[buf_joined.columns])
            print(
                f"[Step 3A] Nearest-PS fallback (<= {max_nearest_m:.0f}m): "
                f"rescued {nearest['building_id'].nunique()}/{len(missing_bldgs)} buildings."
            )

    links_df = pd.concat(links_parts, ignore_index=True)
    links_df = links_df.drop(columns=["geometry"]).sort_values(
        ["building_id", "dist_to_building_m", "PS_ID"]
    ).reset_index(drop=True)

    # 3. Aggregate PS time series per building (coherence-weighted mean)
    agg_rows = []
    for bid, grp in links_df.groupby("building_id", sort=True):
        weights = grp["Coherence"].to_numpy(dtype=float)
        weights = weights / weights.sum()
        ts_matrix = grp[date_cols].to_numpy(dtype=float)
        weighted_ts = np.average(ts_matrix, axis=0, weights=weights)

        row_dict = {
            "building_id": bid,
            "ps_count": int(len(grp)),
            "join_type": "buffer" if (grp["join_type"] != "nearest_fallback").any() else "nearest_fallback",
            "min_dist_m": float(grp["dist_to_building_m"].min()),
            "mean_coherence": round(float(grp["Coherence"].mean()), 4),
            "min_coherence": round(float(grp["Coherence"].min()), 4),
            "vel_mean_mm_yr": round(float(np.average(grp["Mean_Velocity_mm_yr"], weights=weights)), 3),
            "vel_min_mm_yr": round(float(grp["Mean_Velocity_mm_yr"].min()), 3),
            "vel_max_mm_yr": round(float(grp["Mean_Velocity_mm_yr"].max()), 3),
            "vel_std_mm_yr": round(float(grp["Mean_Velocity_mm_yr"].std(ddof=0)), 3),
            "ps_ids": ",".join(grp["PS_ID"].tolist()),
        }
        for col, val in zip(date_cols, weighted_ts):
            row_dict[col] = round(float(val), 3)
        agg_rows.append(row_dict)

    bldg_ts_df = pd.DataFrame(agg_rows)

    out_dir = DATA_PROCESSED / roi_key
    out_dir.mkdir(parents=True, exist_ok=True)

    links_parquet = out_dir / f"ps_building_links_{roi_key}.parquet"
    links_csv = out_dir / f"ps_building_links_{roi_key}.csv"
    links_df.to_parquet(links_parquet, index=False)
    links_df.to_csv(links_csv, index=False)

    ts_parquet = out_dir / f"building_timeseries_{roi_key}.parquet"
    ts_csv = out_dir / f"building_timeseries_{roi_key}.csv"
    bldg_ts_df.to_parquet(ts_parquet, index=False)
    bldg_ts_df.to_csv(ts_csv, index=False)

    ge1 = len(bldg_ts_df)
    ge2 = int((bldg_ts_df["ps_count"] >= 2).sum())
    print(
        f"[Step 3A] Spatial join complete for '{roi_key}':\n"
        f"          - Total PS-building links : {len(links_df)} ({links_df['PS_ID'].nunique()} unique PS points)\n"
        f"          - Buildings covered (>=1) : {ge1}/{len(bldgs_gdf)}\n"
        f"          - Multi-PS buildings (>=2): {ge2}/{len(bldgs_gdf)}\n"
        f"          - Links saved             -> {links_parquet}\n"
        f"          - Building TS saved       -> {ts_parquet}"
    )
    return links_parquet, ts_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 3A: Spatial join PS points to building footprints.")
    parser.add_argument("--roi", choices=list(ROIS.keys()), default=DEFAULT_ROI)
    parser.add_argument("--min-coherence", type=float, default=0.70)
    parser.add_argument("--buffer-m", type=float, default=5.0)
    parser.add_argument("--max-nearest-m", type=float, default=15.0)
    args = parser.parse_args()

    run_spatial_join(
        roi_key=args.roi,
        min_coherence=args.min_coherence,
        buffer_m=args.buffer_m,
        max_nearest_m=args.max_nearest_m,
    )


if __name__ == "__main__":
    main()
