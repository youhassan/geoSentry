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
# Selected for ~90-95% human-verified OpenStreetMap building footprint completeness,
# high orthogonal geometric accuracy (median rectangularity > 0.96), and documented
# historical structural collapses.
ROIS = {
    "mansheya_attarine": {
        "name": "El-Mansheya Historic Core (Manually Digitized High-Precision)",
        "bbox": (29.8939, 31.1961, 29.8980, 31.1998),  # Tightly wraps exact digitized bounds
        "start_date": "2021-01-01",
        "end_date": "2024-12-31",
        "prefix": "MA",
    },
    "bacos_raml": {
        "name": "Bacos / Fleming / Schutz Corridor (El-Raml, East Alexandria)",
        "bbox": (29.9610, 31.2255, 29.9745, 31.2345),  # ~1.30 km^2, ~2,730 footprints
        "start_date": "2021-01-01",
        "end_date": "2023-12-31",
        "prefix": "BR",
    },
    "sidi_bishr": {
        "name": "Sidi Bishr / Miami Corridor (Al-Montazah - Sparse ML Coverage)",
        "bbox": (29.9850, 31.2500, 30.0100, 31.2720),  # ~4.5 km^2 (legacy ROI)
        "start_date": "2021-01-01",
        "end_date": "2023-12-31",
        "prefix": "SB",
    },
}

DEFAULT_ROI = "mansheya_attarine"
