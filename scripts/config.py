"""Central configuration for GeoSentry ROIs, paths, and constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_GROUND_TRUTH = PROJECT_ROOT / "data" / "ground_truth"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

for _d in (DATA_RAW, DATA_PROCESSED, DATA_GROUND_TRUTH, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Coordinate Reference Systems
CRS_WGS84 = "EPSG:4326"
CRS_UTM_ALEXANDRIA = "EPSG:32636"  # WGS 84 / UTM Zone 36N (meters)

# Regions of Interest (ROIs) in Alexandria, Egypt
# Format: (west, south, east, north)
ROIS = {
    "sidi_bishr": {
        "name": "Sidi Bishr / Miami Corridor (Al-Montazah)",
        "bbox": (29.985, 31.250, 30.010, 31.272),  # ~4.5 km^2
        "start_date": "2021-01-01",
        "end_date": "2023-12-31",
        "prefix": "SB",
    },
    "gomrok": {
        "name": "El-Gomrok / Bahary",
        "bbox": (29.875, 31.195, 29.895, 31.215),  # ~4.2 km^2
        "start_date": "2021-01-01",
        "end_date": "2024-12-31",
        "prefix": "GM",
    },
}

DEFAULT_ROI = "sidi_bishr"
