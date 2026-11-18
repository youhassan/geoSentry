"""Step 1C: Query ASF API for Sentinel-1 Level-1 SLC metadata (zero 8 GB downloads)."""

import argparse
import asf_search as asf
import pandas as pd

from config import DATA_RAW, DEFAULT_ROI, ROIS


def query_sentinel1(roi_key: str, start_date: str | None = None, end_date: str | None = None) -> None:
    roi = ROIS[roi_key]
    west, south, east, north = roi["bbox"]
    start = start_date or roi["start_date"]
    end = end_date or roi["end_date"]

    # WKT Polygon for the ROI bounding box
    wkt_aoi = (
        f"POLYGON(({west} {south}, {east} {south}, "
        f"{east} {north}, {west} {north}, {west} {south}))"
    )

    print(f"[Step 1C] Querying ASF for Sentinel-1 SLC scenes...")
    print(f"          ROI: {roi['name']} | Window: {start} to {end}")

    results = asf.geo_search(
        platform=[asf.PLATFORM.SENTINEL1],
        processingLevel=[asf.PRODUCT_TYPE.SLC],
        beamMode=[asf.BEAMMODE.IW],
        polarization=[asf.POLARIZATION.VV, asf.POLARIZATION.VV_VH],
        intersectsWith=wkt_aoi,
        start=f"{start}T00:00:00Z",
        end=f"{end}T23:59:59Z",
    )

    if not results:
        raise RuntimeError("No Sentinel-1 SLC granules found for the given ROI and dates.")

    rows = []
    for r in results:
        p = r.properties
        rows.append(
            {
                "granule_id": p.get("sceneName"),
                "file_id": p.get("fileID"),
                "platform": p.get("platform"),
                "start_time": p.get("startTime"),
                "flight_direction": p.get("flightDirection"),
                "path_number": p.get("pathNumber"),
                "frame_number": p.get("frameNumber"),
                "polarization": p.get("polarization"),
                "bytes": p.get("bytes"),
                "size_gb": round((p.get("bytes") or 0) / (1024**3), 2),
                "url": p.get("url"),
            }
        )

    df = pd.DataFrame(rows).sort_values("start_time").reset_index(drop=True)
    catalog_path = DATA_RAW / f"{roi_key}_sentinel1_slc_catalog.csv"
    df.to_csv(catalog_path, index=False)

    print(f"[Step 1C] Found {len(df)} total SLC granules -> {catalog_path}")
    print("\n--- Orbit / Track Summary (Pick ONE consistent Path + Frame) ---")
    summary = (
        df.groupby(["flight_direction", "path_number", "frame_number"])
        .agg(
            scene_count=("granule_id", "count"),
            first_date=("start_time", "min"),
            last_date=("start_time", "max"),
            total_gb_if_local=("size_gb", "sum"),
        )
        .reset_index()
        .sort_values("scene_count", ascending=False)
    )
    print(summary.to_string(index=False))

    # Select the track with the highest scene count
    best = summary.iloc[0]
    best_df = df[
        (df["flight_direction"] == best["flight_direction"])
        & (df["path_number"] == best["path_number"])
        & (df["frame_number"] == best["frame_number"])
    ]
    granules_path = DATA_RAW / f"{roi_key}_sentinel1_selected_granules.txt"
    granules_path.write_text("\n".join(best_df["granule_id"].tolist()) + "\n", encoding="utf-8")

    print(
        f"\n[Step 1C] Selected optimal track: {best['flight_direction']} "
        f"Path {best['path_number']} / Frame {best['frame_number']} "
        f"({best['scene_count']} scenes)"
    )
    print(f"          Saved granule list (for ASF HyP3 / Colab Step 2) -> {granules_path}")
    print(
        f"          (Avoided downloading ~{best['total_gb_if_local']:.1f} GB locally!)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query Sentinel-1 SLC metadata via ASF API.")
    parser.add_argument("--roi", choices=list(ROIS.keys()), default=DEFAULT_ROI)
    parser.add_argument("--start", default=None, help="Override start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="Override end date YYYY-MM-DD")
    args = parser.parse_args()
    query_sentinel1(args.roi, args.start, args.end)
