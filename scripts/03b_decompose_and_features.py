"""Step 3B: STL time-series decomposition and structural hazard feature extraction."""

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from statsmodels.tsa.seasonal import STL

from config import (
    CRS_UTM_ALEXANDRIA,
    CRS_WGS84,
    DATA_GROUND_TRUTH,
    DATA_PROCESSED,
    DATA_RAW,
    DEFAULT_ROI,
    ROIS,
)


def _ols_slope_mm_yr(days: np.ndarray, values: np.ndarray) -> float:
    """Compute OLS linear slope in mm/yr given time in days and displacement in mm."""
    if len(days) < 2 or np.ptp(days) == 0:
        return 0.0
    t_yr = (days - days[0]) / 365.25
    slope, _ = np.polyfit(t_yr, values, 1)
    return float(slope)


def _quad_accel_mm_yr2(days: np.ndarray, values: np.ndarray) -> float:
    """Compute quadratic acceleration (2*c2 in mm/yr^2) from T(t) = c0 + c1*t + c2*t^2."""
    if len(days) < 4 or np.ptp(days) == 0:
        return 0.0
    t_yr = (days - days[0]) / 365.25
    c2, _, _ = np.polyfit(t_yr, values, 2)
    return float(2.0 * c2)


def decompose_stl_regular(
    dates: pd.DatetimeIndex,
    disp_mm: np.ndarray,
    freq_days: int = 12,
    seasonal_period: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate irregular SAR dates onto a uniform grid, run robust STL, and map back."""
    t_days = (dates - dates[0]).days.to_numpy(dtype=float)
    reg_dates = pd.date_range(dates[0], dates[-1], freq=f"{freq_days}D")
    reg_days = (reg_dates - dates[0]).days.to_numpy(dtype=float)

    reg_disp = np.interp(reg_days, t_days, disp_mm)
    series = pd.Series(reg_disp, index=reg_dates)

    stl = STL(series, period=seasonal_period, seasonal=13, robust=True)
    res = stl.fit()

    trend_orig = np.interp(t_days, reg_days, res.trend.to_numpy())
    seasonal_orig = np.interp(t_days, reg_days, res.seasonal.to_numpy())
    residual_orig = disp_mm - trend_orig - seasonal_orig
    return trend_orig, seasonal_orig, residual_orig


def compute_rolling_peak_delta(
    days: np.ndarray,
    trend_mm: np.ndarray,
    window_days: float = 120.0,
    min_baseline_days: float = 120.0,
) -> tuple[float, str]:
    """Find the maximum absolute velocity shift across all 120-day windows in the record."""
    max_abs_delta = 0.0
    peak_day_idx = 0
    for i in range(len(days)):
        t_end = days[i]
        t_start = t_end - window_days
        if t_start < days[0] + min_baseline_days:
            continue
        base_mask = days <= t_start
        win_mask = (days >= t_start) & (days <= t_end)
        if base_mask.sum() >= 5 and win_mask.sum() >= 3:
            v_base = _ols_slope_mm_yr(days[base_mask], trend_mm[base_mask])
            v_win = _ols_slope_mm_yr(days[win_mask], trend_mm[win_mask])
            delta = v_win - v_base
            if abs(delta) > abs(max_abs_delta):
                max_abs_delta = delta
                peak_day_idx = i
    return float(max_abs_delta), str(int(days[peak_day_idx]))


def run_feature_extraction(
    roi_key: str,
    neighbor_radius_m: float = 100.0,
) -> tuple[Path, Path]:
    """Run STL decomposition and extract building-level structural hazard features."""
    out_dir = DATA_PROCESSED / roi_key
    ts_path = out_dir / f"building_timeseries_{roi_key}.parquet"
    links_path = out_dir / f"ps_building_links_{roi_key}.parquet"
    bldg_geojson = DATA_RAW / f"{roi_key}_buildings.geojson"
    gt_csv = DATA_GROUND_TRUTH / "alexandria_collapses.csv"

    for p in (ts_path, links_path, bldg_geojson):
        if not p.exists():
            raise FileNotFoundError(f"Required input not found: {p}")

    bldg_ts = pd.read_parquet(ts_path)
    links_df = pd.read_parquet(links_path)
    bldgs_wgs = gpd.read_file(bldg_geojson)
    bldgs_utm = bldgs_wgs.to_crs(CRS_UTM_ALEXANDRIA)

    date_cols = [c for c in bldg_ts.columns if c.startswith("d_")]
    dates = pd.to_datetime([c[2:] for c in date_cols], format="%Y%m%d")
    days = (dates - dates[0]).days.to_numpy(dtype=float)

    # Precompute pairwise PS velocity gradient per building (mm/yr per meter)
    links_gdf = gpd.GeoDataFrame(
        links_df,
        geometry=gpd.points_from_xy(links_df["Lon"], links_df["Lat"]),
        crs=CRS_WGS84,
    ).to_crs(CRS_UTM_ALEXANDRIA)

    tilt_gradients: dict[str, float] = {}
    for bid, grp in links_gdf.groupby("building_id"):
        if len(grp) < 2:
            tilt_gradients[bid] = 0.0
            continue
        coords = np.column_stack([grp.geometry.x.to_numpy(), grp.geometry.y.to_numpy()])
        vels = grp["Mean_Velocity_mm_yr"].to_numpy(dtype=float)
        dists = cdist(coords, coords)
        v_diffs = np.abs(vels[:, None] - vels[None, :])
        valid = dists > 1.0
        tilt_gradients[bid] = float(np.max(v_diffs[valid] / dists[valid])) if valid.any() else 0.0

    # Ground-truth collapse lookup
    gt_map: dict[str, dict] = {}
    if gt_csv.exists():
        gt_df = pd.read_csv(gt_csv)
        gt_roi = gt_df[gt_df["roi"] == roi_key]
        for _, r in gt_roi.iterrows():
            bid = str(r["matched_building_id"])
            gt_map[bid] = {
                "incident_id": str(r["incident_id"]),
                "collapse_date": str(r["date"]),
            }

    # STL decomposition + per-building temporal & internal features
    stl_records = []
    feat_rows = []

    t_end = days[-1]
    mask_base_120 = days <= (t_end - 120.0)
    mask_last_120 = days >= (t_end - 120.0)
    mask_base_180 = days <= (t_end - 180.0)
    mask_last_180 = days >= (t_end - 180.0)
    mask_last_90 = days >= (t_end - 90.0)

    for _, row in bldg_ts.iterrows():
        bid = str(row["building_id"])
        disp_mm = row[date_cols].to_numpy(dtype=float)

        trend_mm, seasonal_mm, residual_mm = decompose_stl_regular(dates, disp_mm)

        for d_str, raw_v, tr_v, se_v, re_v in zip(
            [d.strftime("%Y-%m-%d") for d in dates],
            disp_mm,
            trend_mm,
            seasonal_mm,
            residual_mm,
        ):
            stl_records.append(
                {
                    "building_id": bid,
                    "date": d_str,
                    "raw_disp_mm": round(float(raw_v), 3),
                    "trend_mm": round(float(tr_v), 3),
                    "seasonal_mm": round(float(se_v), 3),
                    "residual_mm": round(float(re_v), 3),
                }
            )

        trend_vel = _ols_slope_mm_yr(days, trend_mm)
        v_base_120 = _ols_slope_mm_yr(days[mask_base_120], trend_mm[mask_base_120])
        v_last_120 = _ols_slope_mm_yr(days[mask_last_120], trend_mm[mask_last_120])
        v_base_180 = _ols_slope_mm_yr(days[mask_base_180], trend_mm[mask_base_180])
        v_last_180 = _ols_slope_mm_yr(days[mask_last_180], trend_mm[mask_last_180])

        accel_full = _quad_accel_mm_yr2(days, trend_mm)
        accel_180 = _quad_accel_mm_yr2(days[mask_last_180], trend_mm[mask_last_180])
        peak_delta_120, peak_day_offset = compute_rolling_peak_delta(days, trend_mm, 120.0)
        peak_date = (dates[0] + pd.Timedelta(days=int(peak_day_offset))).strftime("%Y-%m-%d")

        diff_tilt = float(row["vel_max_mm_yr"] - row["vel_min_mm_yr"]) if row["ps_count"] >= 2 else 0.0
        gt_info = gt_map.get(bid)

        feat_rows.append(
            {
                "building_id": bid,
                "ps_count": int(row["ps_count"]),
                "join_type": str(row["join_type"]),
                "min_dist_m": float(row["min_dist_m"]),
                "mean_coherence": float(row["mean_coherence"]),
                "min_coherence": float(row["min_coherence"]),
                "vel_mean_mm_yr": float(row["vel_mean_mm_yr"]),
                "vel_abs_mm_yr": round(abs(float(row["vel_mean_mm_yr"])), 3),
                "trend_vel_mm_yr": round(trend_vel, 3),
                "vel_baseline_120d_mm_yr": round(v_base_120, 3),
                "vel_last_120d_mm_yr": round(v_last_120, 3),
                "vel_delta_120d_mm_yr": round(v_last_120 - v_base_120, 3),
                "vel_delta_180d_mm_yr": round(v_last_180 - v_base_180, 3),
                "accel_full_mm_yr2": round(accel_full, 3),
                "accel_180d_mm_yr2": round(accel_180, 3),
                "peak_vel_delta_120d_mm_yr": round(peak_delta_120, 3),
                "peak_vel_delta_date": peak_date,
                "diff_tilt_mm_yr": round(diff_tilt, 3),
                "vel_std_internal_mm_yr": float(row["vel_std_mm_yr"]),
                "tilt_gradient_mm_yr_per_m": round(tilt_gradients.get(bid, 0.0), 4),
                "seasonal_amp_mm": round(float(np.ptp(seasonal_mm)), 3),
                "residual_std_mm": round(float(np.std(residual_mm)), 3),
                "residual_max_abs_mm": round(float(np.max(np.abs(residual_mm))), 3),
                "residual_last_90d_rms_mm": round(float(np.sqrt(np.mean(residual_mm[mask_last_90] ** 2))), 3),
                "is_collapsed_gt": gt_info is not None,
                "collapse_incident_id": gt_info["incident_id"] if gt_info else None,
                "collapse_date": gt_info["collapse_date"] if gt_info else None,
            }
        )

    feat_df = pd.DataFrame(feat_rows)

    # Merge building geometry & metadata
    meta_cols = [
        "building_id",
        "roi",
        "num_floors",
        "area_m2",
        "perimeter_m",
        "rectangularity",
        "centroid_lat",
        "centroid_lon",
    ]
    available_meta = [c for c in meta_cols if c in bldgs_wgs.columns]
    merged_gdf = bldgs_wgs[available_meta + ["geometry"]].merge(feat_df, on="building_id", how="left")

    # Compute 100m neighborhood velocity deviation (Z-score) in UTM meters
    utm_centroids = bldgs_utm.set_index("building_id").geometry.centroid
    coords_utm = np.column_stack(
        [
            [utm_centroids[bid].x for bid in merged_gdf["building_id"]],
            [utm_centroids[bid].y for bid in merged_gdf["building_id"]],
        ]
    )
    dist_mat = cdist(coords_utm, coords_utm)
    vels_all = merged_gdf["vel_mean_mm_yr"].to_numpy(dtype=float)

    n_counts = []
    n_medians = []
    n_mads = []
    z_scores = []
    vel_diffs = []

    for i in range(len(merged_gdf)):
        nbr_mask = (dist_mat[i] > 0.0) & (dist_mat[i] <= neighbor_radius_m)
        if nbr_mask.sum() < 3:
            # Fallback to 5 nearest neighbors if sparse
            order = np.argsort(dist_mat[i])
            nbr_idx = order[1:6]
            nbr_vels = vels_all[nbr_idx]
        else:
            nbr_vels = vels_all[nbr_mask]

        med = float(np.median(nbr_vels))
        mad = float(np.median(np.abs(nbr_vels - med)))
        # Scale MAD to normal std (1.4826) with a 0.5 mm/yr floor to avoid division by ~0
        scale = max(1.4826 * mad, 0.5)
        z = (vels_all[i] - med) / scale

        n_counts.append(int(nbr_mask.sum()))
        n_medians.append(round(med, 3))
        n_mads.append(round(mad, 3))
        z_scores.append(round(float(z), 3))
        vel_diffs.append(round(float(vels_all[i] - med), 3))

    merged_gdf["neighbor_count_100m"] = n_counts
    merged_gdf["neighbor_median_vel_mm_yr"] = n_medians
    merged_gdf["neighbor_mad_vel_mm_yr"] = n_mads
    merged_gdf["neighborhood_vel_diff_mm_yr"] = vel_diffs
    merged_gdf["neighborhood_z_score"] = z_scores

    # Save outputs
    stl_df = pd.DataFrame(stl_records)
    stl_parquet = out_dir / f"building_stl_{roi_key}.parquet"
    stl_df.to_parquet(stl_parquet, index=False)

    feat_parquet = out_dir / f"building_features_{roi_key}.parquet"
    feat_csv = out_dir / f"building_features_{roi_key}.csv"
    feat_geojson = out_dir / f"building_features_{roi_key}.geojson"

    merged_gdf.drop(columns=["geometry"]).to_parquet(feat_parquet, index=False)
    merged_gdf.drop(columns=["geometry"]).to_csv(feat_csv, index=False)
    merged_gdf.to_file(feat_geojson, driver="GeoJSON")

    print(
        f"[Step 3B] STL & Feature Extraction complete for '{roi_key}':\n"
        f"          - Buildings processed : {len(merged_gdf)}\n"
        f"          - STL rows saved      : {len(stl_df)} -> {stl_parquet}\n"
        f"          - Features saved      -> {feat_parquet}\n"
        f"          - Features GeoJSON    -> {feat_geojson}"
    )

    # Print summary of top anomalous & ground-truth buildings
    gt_sub = merged_gdf[merged_gdf["is_collapsed_gt"]]
    if not gt_sub.empty:
        print("\n[Step 3B] Ground-Truth Collapsed Buildings Summary:")
        cols = [
            "building_id",
            "collapse_incident_id",
            "collapse_date",
            "ps_count",
            "vel_mean_mm_yr",
            "diff_tilt_mm_yr",
            "peak_vel_delta_120d_mm_yr",
            "neighborhood_z_score",
            "residual_max_abs_mm",
        ]
        print(gt_sub[cols].to_string(index=False))

    return stl_parquet, feat_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 3B: STL decomposition & hazard feature extraction.")
    parser.add_argument("--roi", choices=list(ROIS.keys()), default=DEFAULT_ROI)
    parser.add_argument("--neighbor-radius-m", type=float, default=100.0)
    args = parser.parse_args()

    run_feature_extraction(roi_key=args.roi, neighbor_radius_m=args.neighbor_radius_m)


if __name__ == "__main__":
    main()
