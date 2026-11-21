"""Step 2B: Poll ASF HyP3 jobs, download completed burst interferograms, and run MintPy prep_hyp3."""

import argparse
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import hyp3_sdk
from dotenv import load_dotenv
from mintpy.cli import prep_hyp3

from config import DATA_PROCESSED, DEFAULT_ROI, PROJECT_ROOT, ROIS


NEEDED_SUFFIXES = (
    "_unw_phase.tif",
    "_corr.tif",
    "_dem.tif",
    "_lv_theta.tif",
    "_lv_phi.tif",
    ".txt",
)


def get_hyp3_client() -> hyp3_sdk.HyP3:
    load_dotenv()
    username = os.getenv("EARTHDATA_USERNAME")
    password = os.getenv("EARTHDATA_PASSWORD")
    if not username or not password:
        raise RuntimeError("EARTHDATA_USERNAME and EARTHDATA_PASSWORD must be set in .env")
    return hyp3_sdk.HyP3(username=username, password=password)


def download_and_extract_job(
    job: hyp3_sdk.Job,
    zips_dir: Path,
    ifgs_dir: Path,
    keep_zips: bool = False,
) -> tuple[str, str]:
    """Download a succeeded HyP3 job zip and extract only MintPy-required files."""
    job_id = job.job_id
    files = job.files or []
    if not files:
        return job_id, "no_files"

    filename = files[0]["filename"]
    stem = Path(filename).stem
    target_dir = ifgs_dir / stem
    if target_dir.exists() and any(target_dir.glob("*_unw_phase.tif")):
        return job_id, "skipped_exists"

    zip_path = zips_dir / filename
    if not zip_path.exists():
        job.download_files(location=zips_dir)

    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            if member.endswith(NEEDED_SUFFIXES):
                out_file = target_dir / Path(member).name
                with zf.open(member) as src, open(out_file, "wb") as dst:
                    dst.write(src.read())

    if not keep_zips and zip_path.exists():
        zip_path.unlink()

    return job_id, "downloaded"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download HyP3 burst jobs and run prep_hyp3.")
    parser.add_argument("--roi", choices=list(ROIS.keys()), default=DEFAULT_ROI)
    parser.add_argument("--looks", choices=["20x4", "10x2", "5x1"], default="5x1")
    parser.add_argument("--status-only", action="store_true", help="Only print HyP3 job status counts")
    parser.add_argument("--workers", type=int, default=4, help="Parallel download threads")
    parser.add_argument("--keep-zips", action="store_true", help="Keep raw .zip archives after extraction")
    parser.add_argument("--skip-prep", action="store_true", help="Skip running mintpy prep_hyp3")
    args = parser.parse_args()

    job_name = f"geosentry_{args.roi}_{args.looks}"
    hyp3 = get_hyp3_client()
    jobs = hyp3.find_jobs(name=job_name)

    counts: dict[str, int] = {}
    for j in jobs:
        counts[j.status_code] = counts.get(j.status_code, 0) + 1

    print(f"[Step 2B] HyP3 job status for '{job_name}' (total={len(jobs)}):")
    for status, cnt in sorted(counts.items()):
        print(f"          {status}: {cnt}")

    if args.status_only:
        return

    succeeded = [j for j in jobs if j.status_code == "SUCCEEDED"]
    if not succeeded:
        print("[Step 2B] No SUCCEEDED jobs ready to download yet.")
        return

    zips_dir = PROJECT_ROOT / "cache" / "hyp3" / args.roi
    ifgs_dir = DATA_PROCESSED / args.roi / "hyp3_ifgs"
    zips_dir.mkdir(parents=True, exist_ok=True)
    ifgs_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Step 2B] Downloading & extracting {len(succeeded)} SUCCEEDED jobs -> {ifgs_dir}")
    summary = {"downloaded": 0, "skipped_exists": 0, "no_files": 0, "error": 0}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(download_and_extract_job, j, zips_dir, ifgs_dir, args.keep_zips)
            for j in succeeded
        ]
        for idx, fut in enumerate(as_completed(futures), start=1):
            try:
                _, status = fut.result()
                summary[status] = summary.get(status, 0) + 1
            except Exception as exc:
                summary["error"] += 1
                print(f"          [WARN] Download error: {exc}")
            if idx % 25 == 0 or idx == len(succeeded):
                print(f"          Progress: {idx}/{len(succeeded)} | {summary}")

    if args.skip_prep:
        return

    unw_files = sorted(str(p) for p in ifgs_dir.glob("*/*_unw_phase.tif"))
    if not unw_files:
        print("[Step 2B] No *_unw_phase.tif files found for prep_hyp3.")
        return

    print(f"[Step 2B] Running MintPy prep_hyp3 on {len(unw_files)} interferograms...")
    # Pass one DEM and one lv_theta in addition to all unw_phase files
    dem_files = sorted(str(p) for p in ifgs_dir.glob("*/*_dem.tif"))
    theta_files = sorted(str(p) for p in ifgs_dir.glob("*/*_lv_theta.tif"))
    prep_args = unw_files
    if dem_files:
        prep_args.append(dem_files[0])
    if theta_files:
        prep_args.append(theta_files[0])
    prep_hyp3.main(prep_args)
    print("[Step 2B] prep_hyp3 complete!")


if __name__ == "__main__":
    main()
