from __future__ import annotations

from pathlib import Path

import folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
from etc2_route_products import build_route_products
from modeling import build_time_series, evaluate_forecasts, estimate_environment, forecast_next, scenario_comparison

st.set_page_config(page_title="Traffic & Environment Lab", page_icon="🌿", layout="wide")
st.markdown("""
<style>
.block-container{padding-top:1.5rem;max-width:1500px}.hero{padding:1.35rem 1.6rem;border-radius:18px;background:linear-gradient(120deg,#102c3a,#155e63);color:#f2fbfa;margin-bottom:1rem}
.hero h1{margin:0;color:#fff}.hero p{margin:.45rem 0 0;color:#d2e9e5}.metric-card{border:1px solid #dbe5e4;border-radius:14px;padding:1rem;background:#fff}
</style>
<div class="hero"><h1>Traffic & Environment Lab</h1><p>Explore synthetic traffic patterns, baseline forecasts, and clearly labelled fuel and CO₂ scenarios on the Munakata road network.</p></div>
""", unsafe_allow_html=True)
st.caption("Research prototype • Road geometry is local OpenStreetMap; vehicle movement, speeds, traffic counts and emissions are synthetic or scenario-estimated.")

with st.sidebar:
    st.subheader("Data source")
    root_text = st.text_input("Local synthetic research-data folder", str(DEFAULT_DATA_DIR))
    st.caption("This folder stays on your computer. It must contain research_truth/ and etc40_pseudo/OUT1-2/.")
    st.divider()
    st.subheader("Environmental assumptions")
    fuel_rate = st.number_input("Average fuel (L/100 vehicle-km)", 1.0, 40.0, 8.5, .5)
    co2_factor = st.number_input("CO₂ per litre (kg/L)", 1.0, 4.0, 2.31, .01)
    idle_rate = st.number_input("Fuel while idling (L/hour)", 0.0, 8.0, 1.0, .1)
    st.caption("Default factors are illustrative user inputs. Replace with a cited vehicle/fuel factor before research use.")
    load = st.button("Load / refresh synthetic data", type="primary", use_container_width=True)

@st.cache_data(show_spinner=False, max_entries=2)
def _load_products(root: str):
    return build_route_products(root, include_gps=True, include_events=False, include_blind_spots=False)

if "products" not in st.session_state:
    st.session_state.products = None
if load:
    selected_root = Path(root_text).expanduser()
    if not selected_root.is_dir():
        st.session_state.products = None
        st.error("That local data folder does not exist. Choose a folder containing research_truth and etc40_pseudo.")
    else:
        with st.spinner("Reading local network and synthetic GPS records. This can take a few minutes the first time…"):
            st.session_state.products = _load_products(str(selected_root.resolve()))

products = st.session_state.products
if products is None:
    st.info("Click **Load / refresh synthetic data** in the left panel. No input data is uploaded.")
    st.stop()
if not products.get("available"):
    st.error(products.get("message", "Data could not be loaded."))
    st.stop()

links = products.get("speed_by_link", pd.DataFrame())
traffic = products.get("traffic_by_link_time", pd.DataFrame())
if links.empty:
    st.warning("The synthetic run loaded but contains no link speed rows.")
    st.stop()

# Lightweight metrics
vehicle_km = float(pd.to_numeric(links.get("gps_vehicle_km", 0), errors="coerce").fillna(0).sum())
trip_count = int(products.get("matched_trip_count", 0))
mean_speed = float(pd.to_numeric(links.get("mean_recorded_speed_kmh", pd.Series(dtype=float)), errors="coerce").mean())
series_df = build_time_series(traffic)
st.markdown("### Simulation snapshot")
cols = st.columns(4)
for col, label, value in zip(cols, ["Matched trips", "GPS-path vehicle-km", "Road links with data", "Mean sampled speed"],
                             [f"{trip_count:,}", f"{vehicle_km:,.1f} km", f"{len(links):,}", f"{mean_speed:.1f} km/h"]):
    col.metric(label, value)
st.warning("These are outputs of the synthetic simulator, not observed ETC2.0 traffic. Near-stop time is the sum of short GPS intervals whose two recorded endpoint speeds are both at or below 5 km/h; it is only a proxy, not engine-idling telemetry. CO₂ and fuel are estimates from editable assumptions.")

tabs = st.tabs(["Overview", "Traffic forecasts", "Fuel & CO₂", "Methods & limits"])
with tabs[0]:
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Synthetic probe activity by time")
        if not series_df.empty:
            fig = px.line(series_df, x="time_bin", y="probe_records", labels={"time_bin":"Time", "probe_records":"GPS probe records"})
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Each value counts synthetic GPS observations across links in a 15-minute window. It is a probe count, not total vehicles on the road.")
        else:
            st.info("No usable timestamps were found in the traffic table.")
    with right:
        st.subheader("Speed distribution across links")
        speed = pd.to_numeric(links.get("mean_recorded_speed_kmh", pd.Series(dtype=float)), errors="coerce").dropna()
        if not speed.empty:
            fig = px.histogram(speed, nbins=30, labels={"value":"Mean sampled link speed (km/h)", "count":"Links"})
            st.plotly_chart(fig, use_container_width=True)
        st.caption("A link mean is calculated from available synthetic GPS speed samples assigned to that directed road link.")
    st.subheader("Road activity map")
    map_rows = links
    map_rows = map_rows.dropna(subset=["lon1", "lat1", "lon2", "lat2"]).copy() if {"lon1","lat1","lon2","lat2"}.issubset(map_rows.columns) else pd.DataFrame()
    if not map_rows.empty:
        m = folium.Map(location=[float(map_rows.lat1.median()), float(map_rows.lon1.median())], zoom_start=11, tiles="OpenStreetMap", control_scale=True)
        max_km = max(float(map_rows.gps_vehicle_km.max()), 1.0)
        for row in map_rows.nlargest(1800, "gps_vehicle_km").itertuples(index=False):
            color = "#12805c" if row.gps_vehicle_km < max_km * .33 else ("#e59b22" if row.gps_vehicle_km < max_km * .66 else "#d4473e")
            shape = row.geometry_coords
            if shape is not None and len(shape) >= 2:
                step = max(1, len(shape) // 30)
                coords = [(float(point[1]), float(point[0])) for point in shape[::step]]
                last_point = (float(shape[-1][1]), float(shape[-1][0]))
                if coords[-1] != last_point:
                    coords.append(last_point)
            else:
                coords = [(row.lat1, row.lon1), (row.lat2, row.lon2)]
            folium.PolyLine(coords, color=color, weight=2.5, opacity=.75,
                tooltip=f"Link {row.link_id} | {row.gps_vehicle_km:.1f} synthetic vehicle-km").add_to(m)
        components.html(m._repr_html_(), height=560, scrolling=False)
        st.caption("Road lines follow the saved OSM link geometry and are colored by synthetic GPS distance. Up to 1,800 high-exposure links are drawn for responsiveness.")
    else:
        st.info("The link geometry was not available for the current run.")

with tabs[1]:
    st.subheader("Short-horizon network forecast")
    target_name = st.selectbox("Forecast target", ["probe_records", "mean_speed_kmh"], format_func=lambda x: "Synthetic GPS probe count" if x == "probe_records" else "Mean sampled speed (km/h)")
    horizon = st.select_slider("Forecast horizon", options=[15,30,45,60], value=60, format_func=lambda x: f"{x} minutes")
    if not series_df.empty:
        target = series_df.set_index("time_bin")[target_name].dropna()
        scores, info = evaluate_forecasts(target, target_name)
        if not scores.empty:
            st.markdown("#### Held-out time-period comparison")
            st.dataframe(scores.style.format({"MAE":"{:.2f}","RMSE":"{:.2f}","sMAPE_percent":"{:.1f}%"}), use_container_width=True, hide_index=True)
            st.caption(f"Chronological split: first 80% used for training/history; final 20% held out. Test starts {info.get('holdout_start')}. Lower errors are better. This is validation on synthetic data only.")
        else:
            st.info(info.get("message", "Insufficient time bins for a forecast comparison."))
        forecast = forecast_next(target, max(1, horizon // 15), target_name)
        observed = pd.DataFrame({"time_bin":target.index,"observed":target.values})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=observed.time_bin.tail(192), y=observed.observed.tail(192), mode="lines", name="Observed synthetic history"))
        fig.add_trace(go.Scatter(x=forecast.time_bin, y=forecast.forecast, mode="lines+markers", name="Persistence baseline forecast", line={"dash":"dash"}))
        fig.update_layout(xaxis_title="Time", yaxis_title="Probe records / bin" if target_name=="probe_records" else "Mean speed (km/h)")
        st.plotly_chart(fig, use_container_width=True)
        st.info("The displayed future line is deliberately a persistence baseline (last observed value carried forward). Random Forest appears in the score table only when enough historical bins exist; no forecast should be treated as real-world prediction until calibrated and validated with real observations.")
    else:
        st.info("Load a run with timestamped synthetic GPS observations to view forecasts.")

with tabs[2]:
    st.subheader("Fuel and CO₂ scenario estimates")
    env = estimate_environment(links, fuel_rate, co2_factor, idle_rate)
    fuel_total = float(env.estimated_fuel_l.sum())
    co2_total = float(env.estimated_co2_kg.sum())
    m1,m2,m3=st.columns(3)
    m1.metric("Estimated fuel", f"{fuel_total:,.1f} L")
    m2.metric("Estimated CO₂", f"{co2_total:,.1f} kg")
    m3.metric("Near-stop duration proxy", f"{env.near_stop_hours_proxy.sum():,.1f} h")
    st.caption(f"Arithmetic: moving fuel = vehicle-km × {fuel_rate:g} L/100 km ÷ 100; CO₂ = litres × {co2_factor:g} kg/L. Near-stop interval proxy uses two adjacent sampled speeds at or below 5 km/h and an editable idling factor {idle_rate:g} L/h.")
    a,b=st.columns(2)
    with a:
        by_band=env.groupby("speed_band", observed=False).agg(vehicle_km=("gps_vehicle_km","sum"), fuel_l=("estimated_fuel_l","sum"), co2_kg=("estimated_co2_kg","sum")).reset_index()
        st.plotly_chart(px.bar(by_band,x="speed_band",y="estimated_co2_kg" if "estimated_co2_kg" in by_band else "co2_kg",labels={"speed_band":"Mean link speed (km/h band)","co2_kg":"Estimated CO₂ (kg)"},title="Estimated CO₂ by link speed band"),use_container_width=True)
    with b:
        st.markdown("#### Route-policy what-if inputs")
        detour=st.slider("Distance increase for residential-avoiding route",0,40,10,1)/100
        gain=st.slider("Assumed speed gain for traffic-aware route",0,40,10,1)
        divert=st.slider("Assumed local-road exposure shifted",0,100,25,5)
        scenario=scenario_comparison(vehicle_km,fuel_rate,co2_factor,detour,gain,divert)
        st.dataframe(scenario.style.format({"distance_vehicle_km":"{:,.1f}","relative_speed_gain_percent":"{:.0f}%","local_road_exposure_shift_percent":"{:.0f}%","illustrative_hours":"{:,.1f}","estimated_fuel_l":"{:,.1f}","estimated_co2_kg":"{:,.1f}"}),use_container_width=True,hide_index=True)
        st.caption("What-if rows change only the explicit slider assumptions and a reference speed of 38.5 km/h. They do not calculate routes or predict driver choices. The exposure-shift control is shown as an assumption, not applied to emissions.")
    st.subheader("Link table: synthetic exposure and estimates")
    show=[c for c in ["link_id","highway","mean_recorded_speed_kmh","gps_vehicle_km","near_stop_speed_sample_share","estimated_fuel_l","estimated_co2_kg"] if c in env.columns]
    st.dataframe(env[show].sort_values("gps_vehicle_km",ascending=False).head(100).style.format(precision=2),use_container_width=True,hide_index=True)

with tabs[3]:
    st.subheader("What this prototype does")
    st.markdown("""
- Reads the local synthetic ETC-style GPS records and saved OpenStreetMap road network through the existing shared route-product builder.
- Aggregates synthetic GPS samples to 15-minute network-wide probe-count and weighted mean-speed series.
- Compares forecasting baselines on a chronological held-out segment using MAE, RMSE, and symmetric MAPE. Optional Random Forest is shown only when the environment has scikit-learn and enough time bins.
- Calculates illustrative fuel and CO₂ by applying editable average consumption and fuel-emission factors to synthetic vehicle-kilometres.
- Compares simple route-policy scenarios using explicit assumptions; it does not run route assignment or causal traffic simulation.
    """)
    st.subheader("Interpretation limits")
    st.warning("Synthetic vehicle IDs, trips, speeds, volumes, and events are generated. OpenStreetMap provides road geometry only. Fuel/CO₂ are factor-based estimates, not measured tailpipe emissions. A short interval with two endpoint speeds at or below 5 km/h is only a near-stop proxy, not engine idling telemetry. No real ETC2.0 GPS records, emissions inventory, or traffic counts are used. Do not report these as validated forecasts or policy effects.")
    st.markdown("#### To make this paper-ready")
    st.markdown("1. Cite and justify the vehicle/fuel factors. 2. Add realistic temporal traffic demand and independent real aggregate validation. 3. Compare against persistence and seasonal-naive baselines on multiple time folds. 4. Recompute scenarios on actual network routes and explicitly model local-road exposure. 5. Record generator seed/configuration and sensitivity ranges.")
    download_columns = [c for c in ["link_id", "highway", "mean_recorded_speed_kmh", "gps_vehicle_km", "near_stop_interval_minutes", "near_stop_hours_proxy", "estimated_fuel_l", "estimated_co2_kg"] if c in env.columns]
    download_data = env[download_columns].copy()
    st.download_button("Download link-level environmental estimates (CSV)", download_data.to_csv(index=False).encode("utf-8"), "synthetic_link_environment_estimates.csv", "text/csv")




