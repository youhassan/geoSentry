"""Step 2A: Build SBAS burst pair network and submit ASF HyP3 INSAR_ISCE_BURST jobs."""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import asf_search as asf
import hyp3_sdk
import pandas as pd
from dotenv import load_dotenv

from config import DATA_PROCESSED, DEFAULT_ROI, ROIS


DEFAULT_BURST_ID = "065_138357_IW2"
DEFAULT_TRACK = 65
DEFAULT_MAX_TEMPORAL_DAYS = 48
DEFAULT_NEIGHBORS = 3
DEFAULT_LOOKS = "5x1"


def fetch_burst_catalog(
    roi_key: str,
    burst_id: str = DEFAULT_BURST_ID,
    track: int = DEFAULT_TRACK,
) -> pd.DataFrame:
    """Query ASF for VV Sentinel-1 burst granules covering the ROI."""
    roi = ROIS[roi_key]
    west, south, east, north = roi["bbox"]
    wkt_aoi = (
        f"POLYGON(({west} {south}, {east} {south}, "
        f"{east} {north}, {west} {north}, {west} {south}))"
    )

    results = asf.geo_search(
        platform=[asf.PLATFORM.SENTINEL1],
        processingLevel=[asf.PRODUCT_TYPE.BURST],
        flightDirection=asf.FLIGHT_DIRECTION.DESCENDING,
        relativeOrbit=track,
        polarization=[asf.POLARIZATION.VV],
        intersectsWith=wkt_aoi,
        start=f"{roi['start_date']}T00:00:00Z",
        end=f"{roi['end_date']}T23:59:59Z",
    )

    rows = []
    for r in results:
        p = r.properties
        full_bid = p.get("burst", {}).get("fullBurstID")
        if full_bid != burst_id:
            continue
        start_time = p.get("startTime", "")
        rows.append(
            {
                "granule_id": p.get("fileID"),
                "date": start_time[:10],
                "start_time": start_time,
                "burst_id": full_bid,
                "polarization": p.get("polarization"),
            }
        )

    df = (
        pd.DataFrame(rows)
        .sort_values("start_time")
        .drop_duplicates(subset=["date"], keep="first")
        .reset_index(drop=True)
    )
    if df.empty:
        raise RuntimeError(f"No VV burst granules found for {burst_id} in {roi_key}.")
    return df


def build_sbas_pairs(
    burst_df: pd.DataFrame,
    neighbors: int = DEFAULT_NEIGHBORS,
    max_temporal_days: int = DEFAULT_MAX_TEMPORAL_DAYS,
) -> pd.DataFrame:
    """Create a connected SBAS interferogram pair network (k-nearest forward neighbors)."""
    dates = [datetime.strptime(d, "%Y-%m-%d") for d in burst_df["date"]]
    granules = burst_df["granule_id"].tolist()

    pairs = []
    n = len(granules)
    for i in range(n):
        for step in range(1, neighbors + 1):
            j = i + step
            if j >= n:
                break
            dt_days = (dates[j] - dates[i]).days
            # Always keep immediate sequential pair (step == 1) to guarantee 1 connected component
            if step == 1 or dt_days <= max_temporal_days:
                pairs.append(
                    {
                        "reference_granule": granules[i],
                        "secondary_granule": granules[j],
                        "reference_date": burst_df["date"].iloc[i],
                        "secondary_date": burst_df["date"].iloc[j],
                        "temporal_baseline_days": dt_days,
                        "step": step,
                    }
                )
    return pd.DataFrame(pairs)


def get_hyp3_client() -> hyp3_sdk.HyP3:
    load_dotenv()
    username = os.getenv("EARTHDATA_USERNAME")
    password = os.getenv("EARTHDATA_PASSWORD")
    if not username or not password:
        raise RuntimeError("EARTHDATA_USERNAME and EARTHDATA_PASSWORD must be set in .env")
    return hyp3_sdk.HyP3(username=username, password=password)


def existing_pair_keys(hyp3: hyp3_sdk.HyP3, job_name: str) -> set[tuple[str, str]]:
    """Return (granule1, granule2) tuples already submitted under job_name."""
    jobs = hyp3.find_jobs(name=job_name)
    keys: set[tuple[str, str]] = set()
    for job in jobs:
        if job.status_code == "FAILED":
            continue
        granules = job.job_parameters.get("granules", [])
        if len(granules) == 2:
            keys.add((granules[0], granules[1]))
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit ASF HyP3 INSAR_ISCE_BURST SBAS jobs.")
    parser.add_argument("--roi", choices=list(ROIS.keys()), default=DEFAULT_ROI)
    parser.add_argument("--burst-id", default=DEFAULT_BURST_ID)
    parser.add_argument("--neighbors", type=int, default=DEFAULT_NEIGHBORS)
    parser.add_argument("--max-days", type=int, default=DEFAULT_MAX_TEMPORAL_DAYS)
    parser.add_argument("--looks", choices=["20x4", "10x2", "5x1"], default=DEFAULT_LOOKS)
    parser.add_argument("--submit", action="store_true", help="Actually submit jobs to HyP3 (default is dry-run)")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of pairs to submit")
    args = parser.parse_args()

    job_name = f"geosentry_{args.roi}_{args.looks}"
    out_dir = DATA_PROCESSED / args.roi
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Step 2A] Querying ASF VV bursts for {args.roi} (burst={args.burst_id})...")
    burst_df = fetch_burst_catalog(args.roi, burst_id=args.burst_id)
    pairs_df = build_sbas_pairs(burst_df, neighbors=args.neighbors, max_temporal_days=args.max_days)

    pairs_csv = out_dir / "hyp3_sbas_pairs.csv"
    pairs_df.to_csv(pairs_csv, index=False)

    print(
        f"[Step 2A] Bursts: {len(burst_df)} dates ({burst_df['date'].min()} -> {burst_df['date'].max()})"
    )
    print(
        f"[Step 2A] SBAS network: {len(pairs_df)} pairs "
        f"(neighbors={args.neighbors}, max_days={args.max_days}, looks={args.looks})"
    )
    print(f"          Saved pair table -> {pairs_csv}")

    hyp3 = get_hyp3_client()
    info = hyp3.my_info()
    credits = info.get("remaining_credits", 0)
    already = existing_pair_keys(hyp3, job_name)

    pending_df = pairs_df[
        ~pairs_df.apply(
            lambda r: (r["reference_granule"], r["secondary_granule"]) in already,
            axis=1,
        )
    ].reset_index(drop=True)

    if args.limit is not None:
        pending_df = pending_df.iloc[: args.limit]

    cost_per_job = 1
    est_cost = len(pending_df) * cost_per_job
    print(
        f"[Step 2A] HyP3 credits: {credits} | Already submitted ({job_name}): {len(already)} "
        f"| Pending to submit: {len(pending_df)} (~{est_cost} credits)"
    )

    if not args.submit:
        print("[Step 2A] Dry-run complete. Re-run with --submit to queue jobs on ASF HyP3.")
        return

    if est_cost > credits:
        raise RuntimeError(f"Not enough HyP3 credits: need {est_cost}, have {credits}.")

    if pending_df.empty:
        print("[Step 2A] All pairs already submitted!")
        return

    prepared = [
        hyp3.prepare_insar_isce_burst_job(
            granule1=row.reference_granule,
            granule2=row.secondary_granule,
            name=job_name,
            apply_water_mask=True,
            looks=args.looks,
        )
        for row in pending_df.itertuples(index=False)
    ]

    # HyP3 recommends submitting in chunks of <= 200 jobs
    submitted_jobs = []
    chunk_size = 100
    for start_idx in range(0, len(prepared), chunk_size):
        chunk = prepared[start_idx : start_idx + chunk_size]
        batch = hyp3.submit_prepared_jobs(chunk)
        submitted_jobs.extend(batch.jobs)
        print(f"          Submitted {start_idx + len(chunk)}/{len(prepared)} jobs...")

    jobs_json = out_dir / "hyp3_submitted_jobs.json"
    jobs_json.write_text(
        json.dumps([j.to_dict() for j in submitted_jobs], indent=2),
        encoding="utf-8",
    )
    print(f"[Step 2A] Done! Queued {len(submitted_jobs)} jobs under name '{job_name}'.")
    print(f"          Saved job records -> {jobs_json}")


if __name__ == "__main__":
    main()
