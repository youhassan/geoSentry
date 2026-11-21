"""Step 2C: Run MintPy time-series inversion and export PS/coherent scatterer table."""

import argparse
from pathlib import Path

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
from mintpy.cli import smallbaselineApp
from pyproj import Transformer
from shapely.geometry import Point

from config import CRS_WGS84, DATA_PROCESSED, DEFAULT_ROI, ROIS


def build_mintpy_config(
    roi_key: str,
    ifgs_dir: Path,
    mintpy_dir: Path,
    context_buffer_deg: float = 0.025,
) -> Path:
    """Write a MintPy smallbaselineApp.cfg tailored for HyP3 burst products."""
    west, south, east, north = ROIS[roi_key]["bbox"]
    s_buf = south - context_buffer_deg
    n_buf = north + context_buffer_deg
    w_buf = west - context_buffer_deg
    e_buf = east + context_buffer_deg

    dem_files = sorted(ifgs_dir.glob("*/*_dem.tif"))
    theta_files = sorted(ifgs_dir.glob("*/*_lv_theta.tif"))
    if not dem_files or not theta_files:
        raise FileNotFoundError(f"Missing *_dem.tif or *_lv_theta.tif in {ifgs_dir}")

    cfg_text = f"""# GeoSentry Step 2C — MintPy configuration for {roi_key}
mintpy.load.processor        = hyp3
mintpy.load.unwFile          = {ifgs_dir.as_posix()}/*/*_unw_phase.tif
mintpy.load.corFile          = {ifgs_dir.as_posix()}/*/*_corr.tif
mintpy.load.demFile          = {dem_files[0].as_posix()}
mintpy.load.incAngleFile     = {theta_files[0].as_posix()}

# Crop to a ~5 km x 5 km context window around the ROI
mintpy.subset.lalo           = {s_buf:.5f}:{n_buf:.5f}, {w_buf:.5f}:{e_buf:.5f}

# Reference point & network inversion
mintpy.reference.yx           = 157, 160
mintpy.reference.minCoherence = 0.85
mintpy.network.coherenceBased = yes
mintpy.network.minCoherence   = 0.50
mintpy.networkInversion.weightFunc   = var
mintpy.networkInversion.minTempCoh   = 0.70

# Corrections (local 5 km urban window: linear ramp removal, no external ERA5/solid-tide deps)
mintpy.troposphericDelay.method = no
mintpy.solidEarthTides          = no
mintpy.topographicResidual        = yes
mintpy.deramp                   = linear
"""
    cfg_path = mintpy_dir / "smallbaselineApp.cfg"
    cfg_path.write_text(cfg_text, encoding="utf-8")
    return cfg_path


def export_ps_table(
    roi_key: str,
    mintpy_dir: Path,
    out_dir: Path,
    min_coherence: float = 0.70,
    roi_buffer_deg: float = 0.001,
) -> tuple[Path, Path]:
    """Export MintPy timeseries, velocity, and coherence to CSV and GeoJSON."""
    vel_h5 = mintpy_dir / "velocity.h5"
    coh_h5 = mintpy_dir / "temporalCoherence.h5"
    ts_h5 = mintpy_dir / "timeseries.h5"

    for p in (vel_h5, coh_h5, ts_h5):
        if not p.exists():
            raise FileNotFoundError(f"Expected MintPy output not found: {p}")

    with h5py.File(vel_h5, "r") as f_vel:
        vel_m_yr = f_vel["velocity"][:]  # m/yr
        attrs = dict(f_vel.attrs)

    with h5py.File(coh_h5, "r") as f_coh:
        coh = f_coh["temporalCoherence"][:]

    with h5py.File(ts_h5, "r") as f_ts:
        dates = [d.decode("utf-8") if isinstance(d, bytes) else str(d) for d in f_ts["date"][:]]
        ts_m = f_ts["timeseries"][:]  # (T, H, W) in meters

    rows_n, cols_n = vel_m_yr.shape
    x_first = float(attrs["X_FIRST"])
    y_first = float(attrs["Y_FIRST"])
    x_step = float(attrs["X_STEP"])
    y_step = float(attrs["Y_STEP"])
    epsg = attrs.get("EPSG", "32636")
    if isinstance(epsg, bytes):
        epsg = epsg.decode("utf-8")

    xs = x_first + np.arange(cols_n) * x_step
    ys = y_first + np.arange(rows_n) * y_step
    grid_x, grid_y = np.meshgrid(xs, ys)

    # Convert to WGS84 Lon/Lat if grid is projected (e.g. UTM)
    if str(epsg) != "4326":
        transformer = Transformer.from_crs(f"EPSG:{epsg}", CRS_WGS84, always_xy=True)
        lon_grid, lat_grid = transformer.transform(grid_x, grid_y)
    else:
        lon_grid, lat_grid = grid_x, grid_y

    west, south, east, north = ROIS[roi_key]["bbox"]
    mask = (
        np.isfinite(vel_m_yr)
        & np.isfinite(coh)
        & (coh >= min_coherence)
        & (lon_grid >= west - roi_buffer_deg)
        & (lon_grid <= east + roi_buffer_deg)
        & (lat_grid >= south - roi_buffer_deg)
        & (lat_grid <= north + roi_buffer_deg)
    )

    iy, ix = np.where(mask)
    if len(iy) == 0:
        raise RuntimeError(
            f"0 scatterer points passed min_coherence={min_coherence} inside {roi_key} bbox."
        )

    records = {
        "PS_ID": [f"PS_{k+1:05d}" for k in range(len(iy))],
        "Lat": np.round(lat_grid[iy, ix], 7),
        "Lon": np.round(lon_grid[iy, ix], 7),
        "Coherence": np.round(coh[iy, ix], 4),
        "Mean_Velocity_mm_yr": np.round(vel_m_yr[iy, ix] * 1000.0, 3),
    }

    for t_idx, d_str in enumerate(dates):
        records[f"d_{d_str}"] = np.round(ts_m[t_idx, iy, ix] * 1000.0, 3)

    df = pd.DataFrame(records)
    csv_path = out_dir / f"ps_points_{roi_key}.csv"
    df.to_csv(csv_path, index=False)

    gdf = gpd.GeoDataFrame(
        df[["PS_ID", "Lat", "Lon", "Coherence", "Mean_Velocity_mm_yr"]],
        geometry=[Point(lon, lat) for lon, lat in zip(df["Lon"], df["Lat"])],
        crs=CRS_WGS84,
    )
    geojson_path = out_dir / f"ps_points_{roi_key}.geojson"
    gdf.to_file(geojson_path, driver="GeoJSON")

    return csv_path, geojson_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MintPy SBAS inversion and export PS points.")
    parser.add_argument("--roi", choices=list(ROIS.keys()), default=DEFAULT_ROI)
    parser.add_argument("--min-coherence", type=float, default=0.70)
    parser.add_argument("--export-only", action="store_true", help="Skip MintPy run and only export HDF5 outputs")
    args = parser.parse_args()

    out_dir = DATA_PROCESSED / args.roi
    ifgs_dir = out_dir / "hyp3_ifgs"
    mintpy_dir = out_dir / "mintpy"
    mintpy_dir.mkdir(parents=True, exist_ok=True)

    if not args.export_only:
        cfg_path = build_mintpy_config(args.roi, ifgs_dir, mintpy_dir)
        print(f"[Step 2C] Running MintPy smallbaselineApp -> {mintpy_dir}")
        smallbaselineApp.main([str(cfg_path), "--dir", str(mintpy_dir)])

    csv_path, geojson_path = export_ps_table(
        args.roi,
        mintpy_dir,
        out_dir,
        min_coherence=args.min_coherence,
    )
    print(f"[Step 2C] Exported PS points -> {csv_path}")
    print(f"[Step 2C] Exported GeoJSON   -> {geojson_path}")


if __name__ == "__main__":
    main()
