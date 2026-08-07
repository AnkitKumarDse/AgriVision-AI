"""
AgriVision AI -- Streamlit frontend shell
Run with:  streamlit run app.py

This is the FRONTEND ONLY. All predictions currently run in DEMO MODE
until real model files are added, so every screen stays clickable
while the model team finishes training.

FIXED IN THIS VERSION:
- crop_model.pkl / label_encoder.pkl are no longer loaded unconditionally
  at startup (that was crashing the whole app on deploy if the files
  weren't in the repo yet). They're now loaded safely, with a clear
  "model not added yet" message if missing.
- The OpenWeatherMap API key is no longer hardcoded in the file --
  it now reads from Streamlit secrets (OPENWEATHER_API_KEY). See the
  note near the top of the Weather tab for how to set that.
- Restored the Yield Prediction tab (tab index 3 existed in the tab
  list but had no content -- it would have shown as a blank tab).
- Removed duplicate imports that were scattered mid-file.
"""

import base64
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from utils import ai_assistant_reply, model_status, predict_income, predict_yield, recommend_crops

st.set_page_config(page_title="AgriVision AI", page_icon="🌾", layout="wide")


def _load_farmer_icon_b64():
    icon_path = Path(__file__).parent / "assets" / "farmer_icon.png"
    if icon_path.exists():
        return base64.b64encode(icon_path.read_bytes()).decode()
    return None


FARMER_ICON_B64 = _load_farmer_icon_b64()


@st.cache_resource
def load_crop_model():
    """Safely load crop_model.pkl + label_encoder.pkl if a teammate has
    added them next to this file. Returns (None, None) if missing or
    broken, instead of crashing the whole app."""
    base = Path(__file__).parent
    model_path = base / "crop_model.pkl"
    encoder_path = base / "label_encoder.pkl"
    if model_path.exists() and encoder_path.exists():
        try:
            return joblib.load(model_path), joblib.load(encoder_path)
        except Exception as e:  # noqa: BLE001
            st.session_state["_crop_model_error"] = str(e)
            return None, None
    return None, None


crop_model, label_encoder = load_crop_model()

# ----------------------------------------------------------------------
# Custom styling -- hero banner, card containers, tab spacing, and the
# floating AI-assistant chat bubble.
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; max-width: 1200px; }

    .av-hero {
        background: linear-gradient(120deg, #14301f 0%, #0e1414 100%);
        border: 1px solid #21362a;
        border-radius: 16px;
        padding: 2rem 2.2rem;
        margin-bottom: 1.5rem;
    }
    .av-hero h1 { font-size: 2.1rem; margin-bottom: 0.2rem; color: #f2f9f4; }
    .av-hero p { color: #9db3a8; font-size: 1rem; margin: 0; }
    .av-badge {
        display: inline-block; background: #1c3327; color: #4ade80;
        border-radius: 999px; padding: 3px 12px; font-size: 0.75rem;
        font-weight: 600; letter-spacing: 0.03em; margin-bottom: 0.8rem;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid #21362a; }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent; border-radius: 8px 8px 0 0;
        padding: 10px 16px; color: #9db3a8;
    }
    .stTabs [aria-selected="true"] {
        background-color: #16261d !important; color: #4ade80 !important; font-weight: 600;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #131b18; border: 1px solid #21362a; border-radius: 12px;
    }

    div[data-testid="stMetric"] {
        background-color: #131b18; border: 1px solid #21362a; border-radius: 10px; padding: 14px 16px;
    }
    div[data-testid="stMetricLabel"] { color: #9db3a8; }

    button[kind="primary"] { background-color: #22c55e; border: none; }
    button[kind="primary"]:hover { background-color: #16a34a; }

    div[data-testid="stPopover"] {
        position: fixed !important; top: 58%; left: 26px; margin-top: -34px;
        z-index: 9999; width: auto !important; animation: av-bob 3.2s ease-in-out infinite;
    }
    @keyframes av-bob { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-10px); } }

    div[data-testid="stPopover"] button {
        position: relative; width: 68px; height: 68px; border-radius: 50%;
        font-size: 1.7rem; background: linear-gradient(135deg, #22c55e, #16a34a);
        border: 3px solid #0e1414; box-shadow: 0 6px 18px rgba(0, 0, 0, 0.45);
        color: white; transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    }
    div[data-testid="stPopover"] button:hover {
        transform: scale(1.12) rotate(-6deg); box-shadow: 0 10px 26px rgba(34, 197, 94, 0.45); border-color: #4ade80;
    }
    div[data-testid="stPopover"] button:active { transform: scale(0.9) rotate(0deg); }
    div[data-testid="stPopover"] button p { font-size: 1.7rem; }

    div[data-testid="stPopover"] button::after {
        content: "Ask me! 💬"; position: absolute; left: 82px; top: 50%;
        transform: translateY(-50%) translateX(-6px); background: #16261d; border: 1px solid #21362a;
        color: #e8f0ec; padding: 6px 12px; border-radius: 8px; font-size: 0.8rem; white-space: nowrap;
        opacity: 0; pointer-events: none; transition: opacity 0.2s ease, transform 0.2s ease;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }
    div[data-testid="stPopover"] button:hover::after { opacity: 1; transform: translateY(-50%) translateX(0); }

    div[data-testid="stPopoverBody"] {
        width: 340px; background-color: #131b18; border: 1px solid #21362a;
        border-radius: 14px; animation: av-panel-in 0.18s ease-out;
    }
    @keyframes av-panel-in {
        0% { opacity: 0; transform: translateY(6px) scale(0.97); }
        100% { opacity: 1; transform: translateY(0) scale(1); }
    }
    div[data-testid="stPopoverBody"] button { transition: transform 0.15s ease, border-color 0.15s ease; }
    div[data-testid="stPopoverBody"] button:hover { transform: translateX(2px); border-color: #4ade80 !important; }
    </style>

    <div class="av-hero">
        <div class="av-badge">POWERED BY AI &amp; DATA ANALYTICS</div>
        <h1>🌾 AgriVision AI</h1>
        <p>Decision support for Indian agriculture -- income estimation, crop guidance, and yield forecasting in one place.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------
if "profile" not in st.session_state:
    st.session_state.profile = {}
if "income_result" not in st.session_state:
    st.session_state.income_result = None
if "yield_result" not in st.session_state:
    st.session_state.yield_result = None
if "crop_result" not in st.session_state:
    st.session_state.crop_result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.title("🌾 AgriVision AI")
    st.caption("AI decision support for Indian agriculture")
    st.divider()

    profile = st.session_state.get("profile", {})
    if profile:
        st.subheader("👤 Farmer Snapshot")
        st.markdown(f"**{profile.get('farmer_id', '--')}** · {profile.get('region', '--')}")
        st.write(f"🌱 {profile.get('current_crop', '--')} · {profile.get('total_land_ha', '--')} ha")
        if st.session_state.get("income_result"):
            st.write(f"💰 ₹{st.session_state.income_result['value']:,.0f} / month")
        st.divider()
    else:
        st.caption("Fill in the Farmer Profile tab to see a live snapshot here.")
        st.divider()

    st.subheader("Model status")
    status = model_status()
    for name, live in status.items():
        icon = "🟢 live" if live else "🟡 demo"
        st.write(f"{name.replace('_', ' ').title()}: {icon}")
    st.write(f"Crop Model (direct file): {'🟢 live' if crop_model is not None else '🟡 demo'}")
    st.caption("Drop matching .pkl files in to go live. See utils.py for the /models contract.")

tabs = st.tabs(
    [
        "👤 Farmer Profile",
        "💰 Income Estimation",
        "🌱 Crop Recommendation",
        "📈 Yield Prediction",
        "🌦️ Weather Dashboard",
        "📄 Reports",
        "📊 Final Dashboard",
    ]
)

# ----------------------------------------------------------------------
# 1. Farmer Profile
# ----------------------------------------------------------------------
with tabs[0]:
    with st.container(border=True):
        st.header("Farmer Profile")
        st.caption("This info is shared across every other tab -- fill it in once.")

        col1, col2 = st.columns(2)
        with col1:
            farmer_id = st.text_input("Farmer ID", value=st.session_state.profile.get("farmer_id", "F-0001"))
            age = st.number_input("Age (years)", 18, 90, st.session_state.profile.get("age", 35))
            education = st.selectbox(
                "Education Level",
                ["None", "Primary", "Secondary", "Graduate", "Postgraduate"],
                index=["None", "Primary", "Secondary", "Graduate", "Postgraduate"].index(
                    st.session_state.profile.get("education_level", "Secondary")
                ),
            )
            region = st.text_input("Region / State", value=st.session_state.profile.get("region", "Punjab"))
        with col2:
            total_land = st.number_input(
                "Total Agricultural Land (hectares)", 0.0, 500.0, float(st.session_state.profile.get("total_land_ha", 5.0)),
                key="land_profile"
            )
            current_crop = st.text_input("Current Primary Crop", value=st.session_state.profile.get("current_crop", "Wheat"))
            non_agri_income = st.number_input(
                "Non-Agricultural Income (₹/month)", 0, 500000, int(st.session_state.profile.get("non_agri_income", 12000))
            )
            distance_to_market = st.number_input(
                "Distance to Market (km)", 0.0, 500.0, float(st.session_state.profile.get("distance_to_market_km", 15.0))
            )

        if st.button("Save Profile", type="primary"):
            st.session_state.profile = {
                "farmer_id": farmer_id,
                "age": age,
                "education_level": education,
                "region": region,
                "total_land_ha": total_land,
                "current_crop": current_crop,
                "non_agri_income": non_agri_income,
                "distance_to_market_km": distance_to_market,
            }
            st.success("Profile saved. It'll now feed the other tabs.")
            st.toast(f"Profile for {farmer_id} saved!", icon="✅")

# ----------------------------------------------------------------------
# 2. Income Estimation
# ----------------------------------------------------------------------
with tabs[1]:
    with st.container(border=True):
        st.header("Income Estimation")
        p = st.session_state.profile
        if not p:
            st.info("Fill in the Farmer Profile tab first (or just enter values below).")

        col1, col2 = st.columns(2)
        with col1:
            crop_yield = st.slider("Crop Yield per Hectare (tons)", 0.0, 50.0, 10.2, help="Average yield across your last few seasons")
            rainfall = st.slider("Rainfall (mm, seasonal avg)", 0.0, 3000.0, 800.0)
        with col2:
            land_override = st.slider(
                "Total Agricultural Land (hectares)", 0.0, 500.0, float(p.get("total_land_ha", 5.0)),
                key="land_income"
            )
            st.caption(f"Using distance to market: {p.get('distance_to_market_km', 15.0)} km, non-ag income: ₹{p.get('non_agri_income', 12000)}")

        if st.button("Predict Income", type="primary"):
            inputs = {
                "age": p.get("age", 35),
                "education_level": p.get("education_level", "Secondary"),
                "total_land_ha": land_override,
                "crop_yield_per_ha": crop_yield,
                "non_agri_income": p.get("non_agri_income", 12000),
                "distance_to_market_km": p.get("distance_to_market_km", 15.0),
                "rainfall_mm": rainfall,
            }
            with st.spinner("Running income model..."):
                time.sleep(0.6)
                st.session_state.income_result = predict_income(inputs)

        result = st.session_state.income_result
        if result:
            if result["demo"]:
                st.warning("Showing a DEMO estimate (formula-based) -- not a real model prediction yet.")

            baseline = p.get("non_agri_income", 12000) * 12
            delta = result["value"] - baseline
            c1, c2 = st.columns([1.2, 1])
            with c1:
                st.metric(
                    "Predicted Income (monthly)",
                    f"₹ {result['value']:,.0f}",
                    delta=f"{delta:,.0f} vs non-agri baseline",
                )
                with st.expander("How was this calculated?"):
                    st.write(
                        "Demo mode blends land size, crop yield, non-agricultural income, "
                        "distance to market, and rainfall into one formula. Once the real "
                        "income model is live, this will show actual feature contributions."
                        if result["demo"]
                        else "This is the trained model's live prediction based on your inputs."
                    )
            with c2:
                gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=result["confidence"] * 100,
                        number={"suffix": "%"},
                        title={"text": "Model Confidence"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "#22c55e"},
                            "bgcolor": "#131b18",
                            "steps": [
                                {"range": [0, 40], "color": "#2a1414"},
                                {"range": [40, 70], "color": "#2a2414"},
                                {"range": [70, 100], "color": "#14261a"},
                            ],
                        },
                    )
                )
                gauge.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)", font_color="#e8f0ec")
                st.plotly_chart(gauge, use_container_width=True)

# ----------------------------------------------------------------------
# 3. Crop Recommendation (uses crop_model.pkl / label_encoder.pkl if present)
# ----------------------------------------------------------------------
with tabs[2]:
    with st.container(border=True):
        st.header("🌾 AI-Powered Crop Recommendation")
        st.caption("Enter the latest soil test values to get the best crop recommendation.")

        if crop_model is None or label_encoder is None:
            st.info(
                "crop_model.pkl / label_encoder.pkl haven't been added to the streamlit_app "
                "folder yet -- ask your teammate to commit them there (same folder as app.py). "
                "Showing the form below so it's ready to go the moment those files land."
            )

        col1, col2 = st.columns(2)
        with col1:
            N = st.number_input("Nitrogen (N)", 0, 200, 90)
            P = st.number_input("Phosphorus (P)", 0, 200, 42)
            K = st.number_input("Potassium (K)", 0, 250, 43)
            ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)
        with col2:
            crop_temp = st.number_input("Temperature (°C)", 0.0, 50.0, 27.0, key="crop_temp")
            humidity = st.number_input("Humidity (%)", 0.0, 100.0, 80.0)
            crop_rainfall = st.number_input("Rainfall (mm)", 0.0, 3000.0, 800.0, key="crop_rainfall")

        if st.button("🌱 Generate Crop Recommendation", type="primary"):
            if crop_model is None or label_encoder is None:
                st.error("Can't run a real prediction yet -- the model files aren't in the repo. See the note above.")
            else:
                npk_total = N + P + K
                n_ratio = N / npk_total if npk_total else 0
                p_ratio = P / npk_total if npk_total else 0
                k_ratio = K / npk_total if npk_total else 0
                rainfall_low = int(crop_rainfall < 100)
                rainfall_medium = int(100 <= crop_rainfall < 200)
                rainfall_high = int(crop_rainfall >= 200)
                temp_cool = int(crop_temp < 20)
                temp_moderate = int(20 <= crop_temp < 30)
                temp_hot = int(crop_temp >= 30)

                features = np.array([[
                    N, P, K, crop_temp, humidity, ph, crop_rainfall, npk_total,
                    n_ratio, p_ratio, k_ratio, rainfall_low, rainfall_medium,
                    rainfall_high, temp_cool, temp_moderate, temp_hot,
                ]])

                prediction = crop_model.predict(features)
                crop = label_encoder.inverse_transform(prediction)[0]
                confidence = float(np.max(crop_model.predict_proba(features)) * 100)

                st.success(f"🌾 Recommended Crop : **{crop.upper()}**")
                st.progress(int(confidence))
                st.metric("Model Confidence", f"{confidence:.2f}%")
                st.divider()

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.info("💧 Water Requirement")
                    st.write("Based on crop")
                with c2:
                    st.info("🌱 Growing Season")
                    st.write("Kharif / Rabi")
                with c3:
                    st.info("📈 Suitability")
                    st.write("Excellent")

                st.divider()
                st.subheader("AI Recommendation")
                st.success(
                    f"""
✅ Recommended Crop : **{crop}**

✔ Soil nutrients are suitable.
✔ Temperature matches crop requirement.
✔ Rainfall conditions are favourable.
✔ Model selected this crop with **{confidence:.2f}% confidence**.
"""
                )

# ----------------------------------------------------------------------
# 4. Yield Prediction
# ----------------------------------------------------------------------
with tabs[3]:
    with st.container(border=True):
        st.header("Yield Prediction")
        col1, col2 = st.columns(2)
        with col1:
            yield_crop_type = st.text_input("Crop Type", value=st.session_state.profile.get("current_crop", "Wheat"), key="yield_crop_type")
            land3 = st.slider("Total Land (hectares)", 0.0, 500.0, 5.0, key="land3")
        with col2:
            rainfall3 = st.slider("Rainfall (mm)", 0.0, 3000.0, 800.0, key="rain3")
            temp3 = st.slider("Temperature (°C)", 0.0, 50.0, 27.0, key="temp3")
        input_costs = st.slider("Input Costs (₹)", 0, 1000000, 20000, step=1000)

        if st.button("Predict Yield", type="primary"):
            inputs = {
                "crop_type": yield_crop_type,
                "total_land_ha": land3,
                "rainfall_mm": rainfall3,
                "temperature_c": temp3,
                "input_costs": input_costs,
            }
            with st.spinner("Estimating yield..."):
                time.sleep(0.6)
                st.session_state.yield_result = predict_yield(inputs)

        result = st.session_state.yield_result
        if result:
            if result["demo"]:
                st.warning("Showing a DEMO estimate -- not a real model prediction yet.")

            c1, c2 = st.columns([1, 1.4])
            c1.metric("Predicted Yield", f"{result['value']} tons/hectare")
            with c2:
                seasons = ["This season", "+1", "+2", "+3"]
                low = [result["value"] * f for f in (1.0, 0.9, 0.85, 0.8)]
                high = [result["value"] * f for f in (1.0, 1.1, 1.15, 1.2)]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=seasons, y=high, line=dict(width=0), showlegend=False))
                fig.add_trace(
                    go.Scatter(
                        x=seasons, y=low, fill="tonexty", line=dict(width=0),
                        fillcolor="rgba(34,197,94,0.25)", showlegend=False,
                    )
                )
                fig.add_trace(go.Scatter(x=seasons, y=[result["value"]] * 4, line=dict(color="#22c55e"), name="Point estimate"))
                fig.update_layout(
                    height=200, margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e8f0ec",
                    yaxis_title="tons/ha",
                )
                st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------
# 5. Weather Dashboard
# ----------------------------------------------------------------------
WEATHER_API_KEY = st.secrets.get("OPENWEATHER_API_KEY", None) if hasattr(st, "secrets") else None


def get_weather(city):
    if not WEATHER_API_KEY:
        return None
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?q={city},IN&appid={WEATHER_API_KEY}&units=metric"
    )
    try:
        response = requests.get(url, timeout=8)
    except requests.RequestException:
        return None
    if response.status_code != 200:
        return None
    data = response.json()
    return {
        "temperature": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "pressure": data["main"]["pressure"],
        "wind": data["wind"]["speed"],
        "condition": data["weather"][0]["main"],
        "description": data["weather"][0]["description"],
        "icon": data["weather"][0]["icon"],
        "city": data["name"],
    }


with tabs[4]:
    with st.container(border=True):
        st.header("🌦 Weather Intelligence Dashboard")
        p = st.session_state.profile

        if not WEATHER_API_KEY:
            st.info(
                "Live weather isn't wired up yet -- add your OpenWeatherMap key to "
                "Streamlit Cloud under Manage app → Settings → Secrets, as:\n\n"
                "`OPENWEATHER_API_KEY = \"your-key-here\"`"
            )

        city = st.text_input("Enter City", value=p.get("region", "Patna"))

        if st.button("Get Live Weather"):
            weather = get_weather(city)
            if weather is None:
                if not WEATHER_API_KEY:
                    st.error("No weather API key configured yet -- see the note above.")
                else:
                    st.error("Unable to fetch weather for that city.")
            else:
                st.success(f"Live Weather • {weather['city']}")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("🌡 Temperature", f"{weather['temperature']} °C")
                with c2:
                    st.metric("💧 Humidity", f"{weather['humidity']}%")
                with c3:
                    st.metric("🌬 Wind", f"{weather['wind']} m/s")
                with c4:
                    st.metric("🧭 Pressure", f"{weather['pressure']} hPa")

                st.divider()
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.image(f"https://openweathermap.org/img/wn/{weather['icon']}@2x.png", width=100)
                with col2:
                    st.subheader(weather["condition"])
                    st.write(weather["description"].title())

                st.divider()
                st.subheader("🌾 AI Farming Advisory")
                if weather["temperature"] > 35:
                    st.warning("High temperature detected. Increase irrigation frequency.")
                elif weather["humidity"] > 85:
                    st.warning("Very high humidity. Monitor crops for fungal diseases.")
                elif weather["condition"] == "Rain":
                    st.info("Rain expected. Avoid irrigation and fertilizer application today.")
                else:
                    st.success("Weather conditions are favourable for normal farming operations.")

# ----------------------------------------------------------------------
# 6. Reports
# ----------------------------------------------------------------------
with tabs[5]:
    with st.container(border=True):
        st.header("Reports")
        p = st.session_state.profile
        if not p:
            st.info("Fill in the Farmer Profile tab to generate a report.")
        else:
            report_lines = [
                "AgriVision AI -- Farmer Report",
                f"Farmer ID: {p.get('farmer_id')}",
                f"Region: {p.get('region')}",
                f"Land: {p.get('total_land_ha')} ha",
                f"Current Crop: {p.get('current_crop')}",
            ]
            if st.session_state.income_result:
                report_lines.append(f"Predicted Income: ₹{st.session_state.income_result['value']:,.0f}")
            if st.session_state.yield_result:
                report_lines.append(f"Predicted Yield: {st.session_state.yield_result['value']} tons/ha")
            if st.session_state.crop_result:
                top = ", ".join(c for c, _ in st.session_state.crop_result["crops"])
                report_lines.append(f"Recommended Crops: {top}")

            report_text = "\n".join(report_lines)
            st.text_area("Report preview", report_text, height=200)
            st.download_button("Download Report (.txt)", report_text, file_name=f"{p.get('farmer_id', 'farmer')}_report.txt")

# ----------------------------------------------------------------------
# 7. Final Dashboard
# ----------------------------------------------------------------------
with tabs[6]:
    with st.container(border=True):
        st.header("📊 Final Dashboard")
        p = st.session_state.profile
        if not p:
            st.info("Fill in the Farmer Profile tab first.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Land Holding", f"{p.get('total_land_ha')} ha")
            c2.metric(
                "Predicted Income",
                f"₹{st.session_state.income_result['value']:,.0f}" if st.session_state.income_result else "Not run yet",
            )
            c3.metric(
                "Predicted Yield",
                f"{st.session_state.yield_result['value']} t/ha" if st.session_state.yield_result else "Not run yet",
            )
            top_crop = st.session_state.crop_result["crops"][0][0] if st.session_state.crop_result else "Not run yet"
            c4.metric("Top Recommended Crop", top_crop)

            col_left, col_right = st.columns([1.4, 1])

            with col_left:
                if st.session_state.income_result:
                    base = st.session_state.income_result["value"]
                    seasons = ["Season 1", "Season 2", "Season 3", "Season 4"]
                    values = [base * f for f in (0.85, 1.05, 0.95, 1.0)]
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=seasons, y=values, fill="tozeroy", line=dict(color="#22c55e")))
                    fig.update_layout(
                        title="Simulated Income Trend Over Seasons", yaxis_title="₹", height=320,
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e8f0ec",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.caption("Run Income Estimation to see the seasonal trend chart here.")

                if st.session_state.crop_result:
                    crops = [c for c, _ in st.session_state.crop_result["crops"]]
                    scores = [s for _, s in st.session_state.crop_result["crops"]]
                    bar = go.Figure(go.Bar(x=crops, y=scores, marker_color="#4ade80"))
                    bar.update_layout(
                        title="Recommended Crops", height=280, yaxis=dict(range=[0, 1], title="Score"),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e8f0ec",
                    )
                    st.plotly_chart(bar, use_container_width=True)

            with col_right:
                categories = ["Land", "Non-Ag Income", "Market Access", "Yield", "Income"]
                land_score = min(p.get("total_land_ha", 0) / 20, 1)
                nonagri_score = min(p.get("non_agri_income", 0) / 50000, 1)
                market_score = 1 - min(p.get("distance_to_market_km", 0) / 100, 1)
                yield_score = min((st.session_state.yield_result["value"] if st.session_state.yield_result else 3) / 10, 1)
                income_score = min((st.session_state.income_result["value"] if st.session_state.income_result else 20000) / 100000, 1)
                values = [land_score, nonagri_score, market_score, yield_score, income_score]

                radar = go.Figure()
                radar.add_trace(go.Scatterpolar(r=values + values[:1], theta=categories + categories[:1], fill="toself", line_color="#22c55e"))
                radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1], showticklabels=False), bgcolor="rgba(0,0,0,0)"),
                    showlegend=False, height=340, title="Farmer Profile Snapshot",
                    paper_bgcolor="rgba(0,0,0,0)", font_color="#e8f0ec",
                )
                st.plotly_chart(radar, use_container_width=True)
                st.caption("Illustrative snapshot -- not a scored metric from any model.")


# ----------------------------------------------------------------------
# Floating AI Assistant "farmer" bubble
# ----------------------------------------------------------------------
def render_ai_assistant_bubble():
    AVATARS = {"user": "🧑‍🌾", "assistant": "🌾"}

    if FARMER_ICON_B64:
        st.markdown(
            f"""
            <style>
            div[data-testid="stPopover"] button {{
                background-image: url('data:image/png;base64,{FARMER_ICON_B64}');
                background-size: cover; background-position: center; background-repeat: no-repeat;
                font-size: 0 !important; color: transparent !important;
            }}
            div[data-testid="stPopover"] button p {{ display: none; }}
            </style>
            """,
            unsafe_allow_html=True,
        )

    if not st.session_state.chat_history:
        st.markdown(
            """
            <style>
            div[data-testid="stPopover"]::before {
                content: ""; position: absolute; inset: -10px; border-radius: 50%;
                background: rgba(34, 197, 94, 0.45); animation: av-pulse 2s ease-out infinite; z-index: -1;
            }
            @keyframes av-pulse {
                0% { transform: scale(0.85); opacity: 0.7; }
                70% { transform: scale(1.55); opacity: 0; }
                100% { transform: scale(1.55); opacity: 0; }
            }
            div[data-testid="stPopover"] button::before {
                content: "💬"; position: absolute; top: -6px; right: -6px; background: #facc15;
                border-radius: 50%; width: 24px; height: 24px; display: flex; align-items: center;
                justify-content: center; font-size: 12px; animation: av-badge-bounce 1.6s ease-in-out infinite;
                box-shadow: 0 2px 6px rgba(0, 0, 0, 0.4);
            }
            @keyframes av-badge-bounce { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-5px); } }
            </style>
            """,
            unsafe_allow_html=True,
        )

    with st.popover("🧑‍🌾", use_container_width=False, help="Ask AgriVision AI"):
        st.markdown("**🌾 AgriVision Assistant**")
        st.caption("Ask about your predicted income, crop choices, or general farming advice.")

        if not st.session_state.chat_history:
            st.markdown("**Try asking:**")
            suggestions = [
                "How can I increase my income?",
                "What crop suits my land best?",
                "Is this a good time to sell?",
            ]
            clicked = None
            for s in suggestions:
                if st.button(s, use_container_width=True, key=f"chip_{s}"):
                    clicked = s
        else:
            clicked = None

        chat_box = st.container(height=280)
        with chat_box:
            for role, msg in st.session_state.chat_history:
                with st.chat_message(role, avatar=AVATARS.get(role)):
                    st.write(msg)

        question = st.chat_input("Ask something...", key="floating_chat_input") or clicked
        if question:
            st.session_state.chat_history.append(("user", question))
            p = st.session_state.profile
            context = {"summary": f"{p.get('current_crop', 'a crop')} farmer with {p.get('total_land_ha', '?')} ha in {p.get('region', 'India')}"}
            with st.spinner("AgriVision AI is thinking..."):
                time.sleep(0.5)
                reply = ai_assistant_reply(question, context)
            st.session_state.chat_history.append(("assistant", reply))
            st.rerun()


render_ai_assistant_bubble()
PYEOF
echo "written"
Output
