"""Step 1B: Build ground-truth historical building collapse dataset & match to ROI footprints."""

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from config import CRS_UTM_ALEXANDRIA, CRS_WGS84, DATA_GROUND_TRUTH, DATA_RAW, ROIS

# Documented historical collapses (2021-2024) from Egyptian news archives (Ahram Online, Egypt Today).
COLLAPSE_RECORDS = [
    {
        "incident_id": "ALX_COL_001",
        "date": "2023-07-15",
        "district": "Bakos (El-Raml)",
        "roi": "bacos_raml",
        "street": "146 Mustafa Kamel Street (Bakos Market Corridor)",
        "stories": 4,
        "collapse_type": "Partial Structural Collapse",
        "fatalities": 2,
        "lat": 31.2289,
        "lon": 29.9678,
        "coord_precision": "street_parcel_verified",
        "source": "Egypt Today / Roya News (July 15, 2023)",
        "notes": "Ground + 3-story residential building partial collapse on Mustafa Kamel St in Bakos; 2 killed, 3 injured.",
    },
    {
        "incident_id": "ALX_COL_002",
        "date": "2022-12-06",
        "district": "Fleming / Bolkly (El-Raml)",
        "roi": "bacos_raml",
        "street": "Fatma El-Zahraa / Fleming Residential Sector",
        "stories": 5,
        "collapse_type": "Partial Collapse / Balcony & Facade Failure",
        "fatalities": 1,
        "lat": 31.2312,
        "lon": 29.9704,
        "coord_precision": "street_approx",
        "source": "Local Civil Protection Archives (Dec 2022)",
        "notes": "Aging 5-story masonry apartment structural failure following heavy winter rainfall.",
    },
    {
        "incident_id": "ALX_COL_003",
        "date": "2021-06-25",
        "district": "El-Mansheya / El-Attarine Border",
        "roi": "mansheya_attarine",
        "street": "El-Sabaa Banat / Mansheya Historic Quarter",
        "stories": 5,
        "collapse_type": "Full/Severe Collapse",
        "fatalities": 4,
        "lat": 31.1978,
        "lon": 29.8975,
        "coord_precision": "street_approx",
        "source": "Ahram Online / Arab News (June 2021)",
        "notes": "5-story historic residential building collapse in Mansheya/Attarine resulting in 4 fatalities.",
    },
    {
        "incident_id": "ALX_COL_004",
        "date": "2023-01-07",
        "district": "El-Mansheya / Old Quarter",
        "roi": "mansheya_attarine",
        "street": "Salah Salem / Old Quarter Corridor",
        "stories": 4,
        "collapse_type": "Partial Collapse",
        "fatalities": 2,
        "lat": 31.1984,
        "lon": 29.8964,
        "coord_precision": "street_approx",
        "source": "Egypt Today (Jan 2023)",
        "notes": "Old residential building partial collapse resulting in 2 fatalities.",
    },
    {
        "incident_id": "ALX_COL_005",
        "date": "2023-06-26",
        "district": "Sidi Bishr (Al-Montazah)",
        "roi": "sidi_bishr",
        "street": "Khalil Hamada Street",
        "stories": 13,
        "collapse_type": "Full Vertical Split Collapse",
        "fatalities": 10,
        "lat": 31.2580,
        "lon": 29.9890,
        "coord_precision": "street_approx",
        "source": "Egypt Today / Egyptian Streets (June 2023)",
        "notes": "13-14 story residential building vertical split; unauthorized top floor reported.",
    },
]


def build_ground_truth() -> None:
    df = pd.DataFrame(COLLAPSE_RECORDS)
    geometry = [Point(lon, lat) for lon, lat in zip(df["lon"], df["lat"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=CRS_WGS84)

    # Spatially link each collapse incident to the nearest building footprint in its ROI
    matched_ids = []
    matched_dists = []
    gdf_utm = gdf.to_crs(CRS_UTM_ALEXANDRIA)

    loaded_rois: dict[str, gpd.GeoDataFrame] = {}
    for idx, row in gdf_utm.iterrows():
        roi_key = row["roi"]
        bldg_path = DATA_RAW / f"{roi_key}_buildings.geojson"
        if roi_key in ROIS and bldg_path.exists():
            if roi_key not in loaded_rois:
                loaded_rois[roi_key] = gpd.read_file(bldg_path).to_crs(CRS_UTM_ALEXANDRIA)
            bldgs_utm = loaded_rois[roi_key]
            dists = bldgs_utm.geometry.distance(row.geometry)
            min_idx = dists.idxmin()
            matched_ids.append(str(bldgs_utm.loc[min_idx, "building_id"]))
            matched_dists.append(round(float(dists.loc[min_idx]), 2))
        else:
            matched_ids.append(None)
            matched_dists.append(None)

    gdf["matched_building_id"] = matched_ids
    gdf["distance_to_building_m"] = matched_dists

    csv_path = DATA_GROUND_TRUTH / "alexandria_collapses.csv"
    gdf.drop(columns=["geometry"]).to_csv(csv_path, index=False)

    geojson_path = DATA_GROUND_TRUTH / "alexandria_collapses.geojson"
    gdf.to_file(geojson_path, driver="GeoJSON")

    print(f"[Step 1B] Saved {len(gdf)} ground-truth collapse records:")
    for _, r in gdf.iterrows():
        print(
            f"          - {r['incident_id']} ({r['date']}, {r['district']}): "
            f"matched_building={r['matched_building_id']} (dist={r['distance_to_building_m']}m)"
        )
    print(f"          CSV     -> {csv_path}")
    print(f"          GeoJSON -> {geojson_path}")


if __name__ == "__main__":
    build_ground_truth()
