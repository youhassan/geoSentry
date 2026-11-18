"""Step 1B: Build ground-truth historical building collapse dataset for Alexandria."""

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from config import CRS_WGS84, DATA_GROUND_TRUTH

# Documented historical collapses (2022-2024) from news archives.
# Coordinates for street-level incidents are approximate street/corridor centers
# unless pinpointed to an exact parcel during local verification.
COLLAPSE_RECORDS = [
    {
        "incident_id": "ALX_COL_001",
        "date": "2023-06-26",
        "district": "Sidi Bishr (Al-Montazah)",
        "roi": "sidi_bishr",
        "street": "Khalil Hamada Street",
        "stories": 13,
        "collapse_type": "Full Collapse",
        "fatalities": 10,
        "lat": 31.2580,
        "lon": 29.9890,
        "coord_precision": "street_approx",
        "source": "Egypt Today / Egyptian Streets (June 2023)",
        "notes": "13-14 story residential building vertical split; unauthorized top floor & column removal reported.",
    },
    {
        "incident_id": "ALX_COL_002",
        "date": "2023-07-13",
        "district": "El-Gomrok",
        "roi": "gomrok",
        "street": "El-Gomrok Old Quarter",
        "stories": 4,
        "collapse_type": "Full/Partial Collapse",
        "fatalities": 3,
        "lat": 31.2045,
        "lon": 29.8835,
        "coord_precision": "district_approx",
        "source": "Egypt Today (July 2023)",
        "notes": "Old residential building collapse in El-Gomrok district resulting in 3 fatalities.",
    },
    {
        "incident_id": "ALX_COL_003",
        "date": "2023-07-15",
        "district": "Bakos (Raml)",
        "roi": "outside_primary_roi",
        "street": "Bakos area",
        "stories": None,
        "collapse_type": "Partial Collapse",
        "fatalities": 0,
        "lat": 31.2290,
        "lon": 29.9610,
        "coord_precision": "district_approx",
        "source": "Egypt Today (July 2023)",
        "notes": "Partial residential collapse reported in Bakos.",
    },
    {
        "incident_id": "ALX_COL_004",
        "date": "2023-01-07",
        "district": "Alexandria (Old Quarter)",
        "roi": "outside_primary_roi",
        "street": "Unspecified old building",
        "stories": None,
        "collapse_type": "Partial Collapse",
        "fatalities": 2,
        "lat": 31.2000,
        "lon": 29.9000,
        "coord_precision": "city_approx",
        "source": "Egypt Today (Jan 2023)",
        "notes": "Old building partial collapse resulting in 2 fatalities.",
    },
]


def build_ground_truth() -> None:
    df = pd.DataFrame(COLLAPSE_RECORDS)
    csv_path = DATA_GROUND_TRUTH / "alexandria_collapses.csv"
    df.to_csv(csv_path, index=False)

    geometry = [Point(lon, lat) for lon, lat in zip(df["lon"], df["lat"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=CRS_WGS84)
    geojson_path = DATA_GROUND_TRUTH / "alexandria_collapses.geojson"
    gdf.to_file(geojson_path, driver="GeoJSON")

    print(f"[Step 1B] Saved {len(df)} ground-truth collapse records:")
    print(f"          CSV     -> {csv_path}")
    print(f"          GeoJSON -> {geojson_path}")


if __name__ == "__main__":
    build_ground_truth()
