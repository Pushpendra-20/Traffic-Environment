# Traffic & Environment Lab

A local Streamlit research prototype for exploring synthetic ETC2.0-style traffic, simple short-horizon forecasts, and illustrative fuel/CO2 estimates. It is a separate project from the emergency route planner.

## Run on Windows

1. Install Python 3.12 or newer and Git.
2. Download/clone this repository, then open this folder in VS Code.
3. Double-click `run_dashboard.bat` (or run `.\run_dashboard.bat` in PowerShell).
4. On first launch, the script creates a project-local `.venv` and installs the packages. Later launches reuse it.
5. In the app sidebar, select your local synthetic-data folder and click **Load / refresh synthetic data**.

The app listens locally in your browser. It does not send data to a server. The first data load may take a few minutes because it reads the saved road graph and synthetic GPS ZIP files.

## Local data required

Data are intentionally not included in this GitHub project. Keep the generated dataset on your computer and select its root folder in the app. The selected folder must have this structure:

```text
your-data-folder/
├── research_truth/
│   ├── trip_truth.csv
│   ├── network_nodes.csv
│   ├── network_edges.csv
│   └── simulation_metadata.json       (optional)
└── etc40_pseudo/OUT1-2/
    └── OUT1-2_*.zip
```

The default local folder is `data` inside this project. You may copy a compatible synthetic run there, or choose another folder in the sidebar. Do not upload private, restricted, or licensed ETC2.0 records. The application expects compatible synthetic Form 1-2 ZIP files plus the generator's truth CSVs; an OSM `.pbf` file alone is not enough.

## What the dashboard calculates

- Aggregates synthetic GPS probe records into 15-minute network-wide counts and sample-weighted mean speeds.
- Compares persistence and previous-day baselines, plus Random Forest when available, on a chronological holdout using MAE, RMSE, and sMAPE.
- Shows a persistence forecast for the selected 15–60 minute horizon. This is a simple baseline, not a claim of predictive accuracy.
- Estimates link-level fuel and CO2 from vehicle-kilometres and user-editable factors.
- Shows near-stop time as a proxy from sampled GPS speed intervals and provides downloadable link-level estimates.

## Interpretation limits

Road geometry comes from OpenStreetMap. Vehicle IDs, trips, GPS points, traffic counts, speeds, and events are synthetic. Fuel and CO2 are factor-based estimates, not measured emissions. Near-stop time is not engine-idling telemetry. Route-policy controls are illustrative what-if arithmetic, not route assignment, driver-choice prediction, or causal impact estimates. Forecast evaluation on synthetic data is not real-world validation. Clearly identify all outputs as synthetic/scenario results in presentations and papers.

## Project files

- `app.py` — Streamlit interface and charts.
- `modeling.py` — aggregation, forecast baselines, evaluation metrics, and environmental arithmetic.
- `etc2_route_products.py` — local parser and aggregation for synthetic ETC2-style data; included so this repository does not depend on a sibling project.
- `requirements.txt` — Python packages.
- `run_dashboard.bat` — Windows launcher.

