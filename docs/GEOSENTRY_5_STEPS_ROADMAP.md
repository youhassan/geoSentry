## Step 1: Define the Region of Interest (ROI) & Gather Data

Do **not** process all of Alexandria at once—Sentinel-1 Single Look Complex (SLC) files are ~4–8 GB *each*, and you need 30 to 60+ images over 2–3 years.

1. **Select a Specific District ($10\text{–}25\text{ km}^2$):**
   * Target a dense, high-risk neighborhood such as **Al-Gomrok (الجمرك)**, **Bahary (بحري)**, or **Karmouz (كرموز)** where aging infrastructure, rising groundwater, and historical collapses are well-documented.
2. **Download Sentinel-1 Radar Data:**
   * **Source:** [Alaska Satellite Facility (ASF) Vertex](https://search.asf.alaska.edu/) or Copernicus Data Space Ecosystem.
   * **Product Type:** **Level-1 SLC (Single Look Complex)**. *(Crucial: Do NOT download GRD data! GRD strips out the radar phase information required to measure millimeter displacement).*
   * **Beam Mode & Polarization:** `IW` (Interferometric Wide swath), `VV` polarization. Ensure all images come from the **exact same Orbit/Path/Frame** (choose Ascending or Descending consistently, or process both separately to resolve vertical vs. horizontal east-west tilt).
   * **Timeframe:** 1 image every 12 days over 2–3 years (~60–90 images).
3. **Extract Building Footprints:**
   * Download 2D building polygons for your chosen district using **Google Open Buildings** or **OpenStreetMap (`osmnx` in Python)**.
4. **Collect Historical Collapse Records (Validation / Ground Truth Set):**
   * Log 10–20 real building collapses in the target neighborhood from local news archives (2020–2025) including their coordinates and collapse dates. This serves as your ground-truth validation set to prove the model detects anomalies *before* failure occurs.

---

## Step 2: Radar Preprocessing (PS-InSAR Pipeline)

Standard image processing cannot extract millimeter displacement from radar data. You must use **PS-InSAR (Persistent Scatterer Interferometry)**, which tracks stable radar reflectors (roof corners, concrete ledges, facades) over time and computes phase differences between satellite passes in millimeters ($\text{mm/year}$).

### Choose One Processing Pathway:

#### Option A: Cloud / Automated On-Demand InSAR *(Recommended for Speed & Hardware Limits)*
1. Use **ASF HyP3 (Hybrid Pluggable Processing Pipeline)** (provides free cloud processing quotas for Sentinel-1 interferograms on AWS) or **LiCSAR / GEP**.
2. Feed the generated interferogram stack into **MintPy** (Python) to run time-series inversion, reference point selection, and atmospheric correction.

#### Option B: Full Local Processing *(Traditional Remote Sensing Pipeline)*
1. **ESA SNAP + SNAP2StaMPS:** Co-register the stack of Sentinel-1 SLC images to a single master date, crop (subset) to your Alexandria neighborhood bounding box, remove topographic phase using an SRTM/Copernicus DEM, and export interferograms.
2. **StaMPS (MATLAB/Octave) or PyRate:** Identify Persistent Scatterer (PS) pixels based on amplitude dispersion, unwrap the 3D phase, and remove the Atmospheric Phase Screen (APS) caused by tropospheric humidity/weather variations.

### Output of Step 2
A structured CSV or Shapefile containing thousands of Persistent Scatterer (PS) points across the neighborhood:
```text
PS_ID | Lat | Lon | Coherence | Mean_Velocity (mm/yr) | d_20230101 | d_20230113 | d_20230125 | ...
```

---

## Step 3: Data Engineering, Spatial Joining & Feature Extraction

Because Sentinel-1 IW mode has a spatial resolution of roughly $5\text{m} \times 20\text{m}$, you get scatterer points rather than 3D building meshes. In this step, you map radar points to buildings and engineer structural stability features using Python (`geopandas`, `shapely`, `statsmodels`).

1. **Coherence Filtering:**
   * Filter out noisy radar points with temporal coherence $< 0.70$ (typically vegetation, water, or active construction zones).
2. **Spatial Join (`geopandas.sjoin` / `sjoin_nearest`):**
   * Map each PS point to its intersecting building polygon (using a small $3\text{–}5\text{m}$ buffer to capture roof-edge scatterers due to radar side-looking geometry).
3. **Time-Series Decomposition (Normal vs. Abnormal Separation):**
   * Concrete expands in summer and contracts in winter. Apply **STL Decomposition (Seasonal-Trend decomposition using Loess)** to each building's displacement curve $D(t)$:
     $$D(t) = T(t) + S(t) + R(t)$$
     * $S(t)$ **(Seasonal):** Normal thermal dilation across seasons.
     * $T(t)$ **(Trend):** Long-term foundation settlement or regional land subsidence.
     * $R(t)$ **(Residual):** Unmodeled sudden shifts or structural shocks.
4. **Engineer Structural Hazard Features:**
   * **Mean Velocity ($v$):** Overall rate of vertical/line-of-sight movement ($\text{mm/year}$).
   * **Short-Term Acceleration ($a$):** Difference between velocity in the last 3–6 months and the historical baseline velocity. *Stable buildings settle at a constant rate; failing buildings accelerate prior to collapse.*
   * **Differential Settlement / Tilt ($\Delta v_{\text{internal}}$):** For buildings containing $\ge 2$ PS points, calculate the maximum velocity difference across points on the same roof. If one corner sinks at $-9\text{ mm/yr}$ while the opposite corner sinks at $-1\text{ mm/yr}$, the structure is shearing/tilting.
   * **Neighborhood Deviation ($Z$-score):** Compare a building's subsidence rate against the median velocity of buildings within a $100\text{m}$ radius. This separates normal street-level soil compaction from isolated structural failure.

---

## Step 4: Machine Learning Model Training

Because real building collapses are rare events, you cannot rely on a simple balanced binary classifier alone. Build a **Two-Stage Hybrid ML Pipeline** using `PyTorch`, `scikit-learn`, and `XGBoost`:

### Stage A: Temporal Anomaly & Change-Point Detection (Normal vs. Abnormal)
* **Objective:** Detect when a building's movement deviates from normal seasonal thermal expansion and steady regional subsidence.
* **Models:**
  1. **LSTM Autoencoder (or Temporal Convolutional Network - TCN):** Train exclusively on stable ("normal") building displacement sequences. The model learns to compress and reconstruct normal seasonal + linear trends. When a building begins experiencing abnormal structural deformation, the **Reconstruction Error (MSE)** spikes.
  2. **Change-Point Detection (`ruptures` / CUSUM):** Identify the exact timestamp (kinematic breakpoint) where a displacement slope steepens sharply.

### Stage B: Structural Hazard Classifier & Risk Scorer (Green / Yellow / Red)
* **Objective:** Combine the temporal anomaly score from Stage A with the spatial/geotechnical features from Step 3 to output a unified **Risk Score ($0\text{–}100\%$)**.
* **Models:**
  * **Isolation Forest** (unsupervised spatial-temporal outlier scoring) + **XGBoost / LightGBM** (trained on historical collapsed buildings + synthetic pre-collapse acceleration trajectories modeled via the geotechnical *Voight/Fukuzono failure law*).
* **Hazard Classification Thresholds:**
  * 🟢 **Green (Safe / Normal):** Velocity within regional norm ($|v| < 2.5\text{ mm/yr}$), near-zero acceleration, low LSTM reconstruction error.
  * 🟡 **Yellow (Moderate / Watchlist):** Elevated subsidence ($2.5\text{–}6.0\text{ mm/yr}$) or mild differential tilt, but constant velocity (no recent acceleration).
  * 🔴 **Red (High Hazard / Critical):** Significant recent acceleration over the last 60–90 days, severe differential tilt across the building footprint, or extreme anomaly deviation compared to neighboring structures.

---

## Step 5: Building the Interactive Map Dashboard

Create a web-based geospatial command center where engineers and decision-makers can inspect the region and click individual buildings.

1. **Database / Storage Layer:**
   * **Lightweight Option:** Pre-computed `.geojson` for building polygons + `.parquet` or `SQLite` for historical time-series.
   * **Production Option:** **PostgreSQL + PostGIS** for spatial queries.
2. **Backend API (`FastAPI` in Python):**
   * `GET /api/buildings`: Returns a GeoJSON FeatureCollection of all buildings in the district with properties: `building_id`, `hazard_status` (`Green` / `Yellow` / `Red`), `risk_score`, `mean_velocity`, `acceleration`, and `differential_tilt`.
   * `GET /api/buildings/{building_id}/timeseries`: Returns the full array of dates, raw displacement values, seasonal trend, and anomaly flags for that specific building.
3. **Frontend Map & Analytics UI (`React` + `MapLibre GL JS` / `Deck.gl` + `Plotly.js`):**
   * **Map View:** Render building footprints as **2D polygons or 3D extruded buildings** (`fill-extrusion` layer in MapLibre) color-coded by hazard status (`#22c55e` Green, `#eab308` Yellow, `#ef4444` Red).
   * **Interactive Click Drawer:** When a user clicks on any building:
     * Open a side panel displaying the **Building ID, Hazard Status Badge, Risk Score (%), Mean Velocity ($\text{mm/yr}$), Recent Acceleration, Differential Tilt, and Radar Coherence**.
     * Render an interactive **displacement time-series chart** (`Plotly.js` or `Recharts`) showing millimeter movement over time, the normal expected seasonal corridor, and the exact point where abnormal deformation began.

---

## Recommended Tech Stack Summary

| Layer | Tools & Libraries |
| :--- | :--- |
| **Radar & Footprint Data** | Sentinel-1 Level-1 SLC (ASF Vertex), Google Open Buildings / OpenStreetMap (`osmnx`) |
| **InSAR Preprocessing** | **ASF HyP3 + MintPy** (cloud-assisted) or **ESA SNAP + StaMPS** |
| **Geospatial & Signal Prep** | Python, `geopandas`, `shapely`, `scipy`, `statsmodels` (STL decomposition), `ruptures` |
| **Machine Learning** | `PyTorch` (LSTM Autoencoder), `scikit-learn` (Isolation Forest), `XGBoost` |
| **Backend & Data Store** | `FastAPI`, `GeoJSON` / `PostGIS` / `DuckDB` |
| **Frontend Dashboard** | `React` (or `Streamlit` for quick prototype), `MapLibre GL JS`, `Deck.gl`, `Plotly.js` |
