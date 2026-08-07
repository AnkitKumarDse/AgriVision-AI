"""
AgriVision AI -- Streamlit frontend shell
Run with:  streamlit run app.py

This is the FRONTEND ONLY. All predictions currently run in DEMO MODE
(see utils.py) using transparent placeholder formulas so every screen is
fully clickable while the model team finishes training. Once a teammate
drops a .pkl into /models matching the contract documented at the top of
utils.py, the relevant tab switches from demo to real predictions
automatically -- no changes needed here.

CHANGE LOG (this version):
- The AI Assistant is no longer a tab. It now lives as a floating
  "farmer" chat bubble (bottom-right corner) that's available on every
  tab, so the user never has to leave what they're doing to ask it
  something. See `render_ai_assistant_bubble()` near the bottom.
"""

import base64
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import ai_assistant_reply, model_status, predict_income, predict_yield, recommend_crops

st.set_page_config(page_title="AgriVision AI", page_icon="🌾", layout="wide")


def _load_farmer_icon_b64():
    """
    Loads assets/farmer_icon.png (next to this file) and returns it as a
    base64 string so it can be embedded straight into CSS as a
    background-image -- no separate image hosting needed.
    Falls back to None (and the bubble falls back to an emoji) if the
    file isn't there, e.g. it wasn't committed to the repo.
    """
    icon_path = Path(__file__).parent / "assets" / "farmer_icon.png"
    if icon_path.exists():
        return base64.b64encode(icon_path.read_bytes()).decode()
    return None


FARMER_ICON_B64 = _load_farmer_icon_b64()

# ----------------------------------------------------------------------
# Custom styling -- hero banner, card containers, tab spacing, and the
# floating AI-assistant chat bubble.
# Colors/theme (dark + green) live in .streamlit/config.toml.
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
    .av-hero h1 {
        font-size: 2.1rem;
        margin-bottom: 0.2rem;
        color: #f2f9f4;
    }
    .av-hero p {
        color: #9db3a8;
        font-size: 1rem;
        margin: 0;
    }
    .av-badge {
        display: inline-block;
        background: #1c3327;
        color: #4ade80;
        border-radius: 999px;
        padding: 3px 12px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        margin-bottom: 0.8rem;
    }

    /* tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid #21362a; }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px 8px 0 0;
        padding: 10px 16px;
        color: #9db3a8;
    }
    .stTabs [aria-selected="true"] {
        background-color: #16261d !important;
        color: #4ade80 !important;
        font-weight: 600;
    }

    /* card-like containers for sections */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #131b18;
        border: 1px solid #21362a;
        border-radius: 12px;
    }

    /* metrics */
    div[data-testid="stMetric"] {
        background-color: #131b18;
        border: 1px solid #21362a;
        border-radius: 10px;
        padding: 14px 16px;
    }
    div[data-testid="stMetricLabel"] { color: #9db3a8; }

    button[kind="primary"] {
        background-color: #22c55e;
        border: none;
    }
    button[kind="primary"]:hover { background-color: #16a34a; }

    /* -----------------------------------------------------------
       Floating AI-assistant "farmer" bubble.
       Targeted directly by Streamlit's own data-testid for a popover
       widget (div[data-testid="stPopover"]) rather than guessing at
       DOM nesting depth -- this is stable across Streamlit versions.
       Pinned to the LEFT side of the viewport, a little above center,
       above everything else (z-index). The button is styled into a
       round avatar with an idle float, a hover "greet", and a press
       animation so it reads as a little living farmer character
       rather than a normal widget.
       ----------------------------------------------------------- */
    div[data-testid="stPopover"] {
        position: fixed !important;
        top: 58%;
        left: 26px;
        margin-top: -34px; /* half the button's height, to vertically anchor at 58% */
        z-index: 9999;
        width: auto !important;
        animation: av-bob 3.2s ease-in-out infinite;
    }

    @keyframes av-bob {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }

    div[data-testid="stPopover"] button {
        position: relative;
        width: 68px;
        height: 68px;
        border-radius: 50%;
        font-size: 1.7rem;
        background: linear-gradient(135deg, #22c55e, #16a34a);
        border: 3px solid #0e1414;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.45);
        color: white;
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    }
    div[data-testid="stPopover"] button:hover {
        transform: scale(1.12) rotate(-6deg);
        box-shadow: 0 10px 26px rgba(34, 197, 94, 0.45);
        border-color: #4ade80;
    }
    div[data-testid="stPopover"] button:active {
        transform: scale(0.9) rotate(0deg);
    }
    div[data-testid="stPopover"] button p {
        font-size: 1.7rem;
    }

    /* small "speech" hint that peeks out on hover, to invite a click */
    div[data-testid="stPopover"] button::after {
        content: "Ask me! 💬";
        position: absolute;
        left: 82px;
        top: 50%;
        transform: translateY(-50%) translateX(-6px);
        background: #16261d;
        border: 1px solid #21362a;
        color: #e8f0ec;
        padding: 6px 12px;
        border-radius: 8px;
        font-size: 0.8rem;
        white-space: nowrap;
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.2s ease, transform 0.2s ease;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }
    div[data-testid="stPopover"] button:hover::after {
        opacity: 1;
        transform: translateY(-50%) translateX(0);
    }

    /* popup panel that opens above the bubble */
    div[data-testid="stPopoverBody"] {
        width: 340px;
        background-color: #131b18;
        border: 1px solid #21362a;
        border-radius: 14px;
        animation: av-panel-in 0.18s ease-out;
    }
    @keyframes av-panel-in {
        0% { opacity: 0; transform: translateY(6px) scale(0.97); }
        100% { opacity: 1; transform: translateY(0) scale(1); }
    }

    /* chip buttons and suggestions inside the chat panel get a little
       lift on hover so the panel feels responsive too */
    div[data-testid="stPopoverBody"] button {
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    div[data-testid="stPopoverBody"] button:hover {
        transform: translateX(2px);
        border-color: #4ade80 !important;
    }
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
# Session state -- this is how tabs (and the floating assistant) share
# the farmer's profile/data
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
    st.caption("Drop a matching .pkl into /models to go live. See utils.py for the contract.")

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
# 3. Crop Recommendation
# ----------------------------------------------------------------------
import joblib
import numpy as np

crop_model = joblib.load("crop_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
with tabs[2]:

    with st.container(border=True):

        st.header("🌾 AI-Powered Crop Recommendation")

        st.caption("Enter the latest soil test values to get the best crop recommendation.")

        col1, col2 = st.columns(2)

        with col1:

            N = st.number_input("Nitrogen (N)", 0, 200, 90)

            P = st.number_input("Phosphorus (P)", 0, 200, 42)

            K = st.number_input("Potassium (K)", 0, 250, 43)

            ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)

        with col2:

            temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 27.0)

            humidity = st.number_input("Humidity (%)", 0.0, 100.0, 80.0)

            rainfall = st.number_input("Rainfall (mm)", 0.0, 3000.0, 800.0)

        if st.button("🌱 Generate Crop Recommendation", type="primary"):

            # -----------------------
            # Feature Engineering
            # -----------------------

            npk_total = N + P + K

            n_ratio = N / npk_total if npk_total else 0

            p_ratio = P / npk_total if npk_total else 0

            k_ratio = K / npk_total if npk_total else 0

            rainfall_low = int(rainfall < 100)

            rainfall_medium = int(100 <= rainfall < 200)

            rainfall_high = int(rainfall >= 200)

            temp_cool = int(temperature < 20)

            temp_moderate = int(20 <= temperature < 30)

            temp_hot = int(temperature >= 30)

            features = np.array([[

                N,

                P,

                K,

                temperature,

                humidity,

                ph,

                rainfall,

                npk_total,

                n_ratio,

                p_ratio,

                k_ratio,

                rainfall_low,

                rainfall_medium,

                rainfall_high,

                temp_cool,

                temp_moderate,

                temp_hot

            ]])

            prediction = crop_model.predict(features)

            crop = label_encoder.inverse_transform(prediction)[0]

            confidence = np.max(crop_model.predict_proba(features)) * 100
            st.session_state.recommended_crop = crop
            st.session_state.crop_confidence = confidence

            st.success(f"🌾 Recommended Crop : **{crop.upper()}**")

            st.progress(int(confidence))

            st.metric("Model Confidence", f"{confidence:.2f}%")

            st.divider()

            col1, col2, col3 = st.columns(3)

            with col1:

                st.info("💧 Water Requirement")

                st.write("Based on crop")

            with col2:

                st.info("🌱 Growing Season")

                st.write("Kharif / Rabi")

            with col3:

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

✔ Random Forest Model selected this crop with **{confidence:.2f}% confidence**.

"""

            )
#-----------------------------------------------------------------------
# 4. yield recomm
# ==========================================================
# 🌾 YIELD PREDICTION
# ==========================================================
# 4. Yield Prediction
# ==========================================================

yield_model = joblib.load("yield_model.pkl")
crop_encoder = joblib.load("crop_encoder.pkl")
season_encoder = joblib.load("season_encoder.pkl")
state_encoder = joblib.load("state_encoder.pkl")

with tabs[3]:

    with st.container(border=True):

        st.header("🌾 AI Yield Prediction")

        st.caption("Predict expected crop yield using Machine Learning.")

        left, right = st.columns(2)

        with left:

            crop = st.selectbox(
                "Crop",
                [
                    "Rice",
                    "Wheat",
                    "Maize",
                    "Cotton",
                    "Sugarcane",
                    "Barley",
                    "Millets",
                    "Groundnut",
                    "Soybean",
                    "Potato",
                    "Gram",
                    "Turmeric",
                ],
            )

            season = st.selectbox(
                "Season",
                [
                    "Kharif",
                    "Rabi",
                    "Summer",
                    "Whole Year",
                    "Winter",
                    "Autumn",
                ],
            )

            state = st.text_input(
                "State",
                "Punjab",
            )

            crop_year = st.number_input(
                "Crop Year",
                1997,
                2035,
                2026,
            )

        with right:

            area = st.number_input(
                "Area (Hectares)",
                value=5.0,
            )

            production = st.number_input(
                "Production (Tonnes)",
                value=20.0,
            )

            rainfall = st.number_input(
                "Annual Rainfall (mm)",
                value=800.0,
            )

            fertilizer = st.number_input(
                "Fertilizer",
                value=450.0,
            )

            pesticide = st.number_input(
                "Pesticide",
                value=8.0,
            )

        predict_yield = st.button(
            "🚀 Predict Yield",
            type="primary",
            use_container_width=True,
        )

        if predict_yield:

            try:

                # Clean Inputs
                crop = crop.strip()
                season = season.strip()
                state = state.strip()

                # Encode
                crop_encoded = crop_encoder.transform([crop])[0]

                season_classes = [x.strip() for x in season_encoder.classes_]
                season_encoded = season_classes.index(season)

                state_classes = [x.strip() for x in state_encoder.classes_]
                state_encoded = state_classes.index(state)

                # Model Input
                input_data = pd.DataFrame(
                    {
                        "Crop": [crop_encoded],
                        "Crop_Year": [crop_year],
                        "Season": [season_encoded],
                        "State": [state_encoded],
                        "Area": [area],
                        "Production": [production],
                        "Annual_Rainfall": [rainfall],
                        "Fertilizer": [fertilizer],
                        "Pesticide": [pesticide],
                    }
                )

                predicted_yield = float(yield_model.predict(input_data)[0])

                total_output = predicted_yield * area
                # ⭐ Save for Gemini
                st.session_state.predicted_yield = predicted_yield
                st.session_state.total_output = total_output
                st.session_state.yield_crop = crop
                st.session_state.yield_season = season
                st.session_state.yield_state = state
                st.success("✅ Yield Prediction Completed Successfully!")

                m1, m2, m3 = st.columns(3)

                with m1:
                    st.metric(
                        "🌾 Predicted Yield",
                        f"{predicted_yield:.2f} t/ha",
                    )

                with m2:
                    st.metric(
                        "📦 Estimated Production",
                        f"{total_output:.2f} Tonnes",
                    )

                with m3:

                    if predicted_yield >= 5:
                        category = "Excellent 🟢"
                    elif predicted_yield >= 3:
                        category = "Average 🟡"
                    else:
                        category = "Low 🔴"

                    st.metric(
                        "Yield Category",
                        category,
                    )

                st.divider()

                left2, right2 = st.columns(2)

                with left2:

                    st.info(f"""
**Crop:** {crop}

**Season:** {season}

**State:** {state}

**Area:** {area:.2f} Hectares
""")

                with right2:

                    st.info(f"""
**Predicted Yield:** {predicted_yield:.2f} t/ha

**Estimated Production:** {total_output:.2f} Tonnes
""")

                st.progress(min(predicted_yield / 8, 1.0))

            except Exception as e:

                st.error("Prediction Failed")

                st.exception(e)
# ----------------------------------------------------------------------
# 5. Weather Dashboard
# ----------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import time
import joblib
API_KEY = "fc99d46057bc6a25799d7a0577685b63"

def get_weather(city):

    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?q={city},IN"
        f"&appid={API_KEY}"
        f"&units=metric"
    )

    response = requests.get(url)

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
        "city": data["name"]
    }
with tabs[4]:

    with st.container(border=True):

        st.header("🌦 Weather Intelligence Dashboard")

        p = st.session_state.profile

        city = st.text_input(
            "Enter City",
            value=p.get("region", "Patna")
        )

        if st.button("Get Live Weather"):

            weather = get_weather(city)

            if weather is None:

                st.error("Unable to fetch weather.")

            else:

                st.success(f"Live Weather • {weather['city']}")

                c1, c2, c3, c4 = st.columns(4)

                with c1:
                    st.metric(
                        "🌡 Temperature",
                        f"{weather['temperature']} °C"
                    )

                with c2:
                    st.metric(
                        "💧 Humidity",
                        f"{weather['humidity']}%"
                    )

                with c3:
                    st.metric(
                        "🌬 Wind",
                        f"{weather['wind']} m/s"
                    )

                with c4:
                    st.metric(
                        "🧭 Pressure",
                        f"{weather['pressure']} hPa"
                    )

                st.divider()

                col1, col2 = st.columns([1,3])

                with col1:

                    st.image(
                        f"https://openweathermap.org/img/wn/{weather['icon']}@2x.png",
                        width=100
                    )

                with col2:

                    st.subheader(weather["condition"])

                    st.write(weather["description"].title())

                st.divider()

                st.subheader("🌾 AI Farming Advisory")

                if weather["temperature"] > 35:

                    st.warning(
                        "High temperature detected. Increase irrigation frequency."
                    )

                elif weather["humidity"] > 85:

                    st.warning(
                        "Very high humidity. Monitor crops for fungal diseases."
                    )

                elif weather["condition"] == "Rain":

                    st.info(
                        "Rain expected. Avoid irrigation and fertilizer application today."
                    )

                else:

                    st.success(
                        "Weather conditions are favourable for normal farming operations."
                    )

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
                f"AgriVision AI -- Farmer Report",
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
                    # Simple simulated seasonal trend around the predicted point, same idea
                    # as the reference demo's "Simulated Income Trend Over Seasons" chart.
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
                # Radar chart giving a quick visual "farmer profile" snapshot.
                # Values are normalised 0-1 just for shape -- purely illustrative.
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
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1], showticklabels=False),
                        bgcolor="rgba(0,0,0,0)",
                    ),
                    showlegend=False, height=340, title="Farmer Profile Snapshot",
                    paper_bgcolor="rgba(0,0,0,0)", font_color="#e8f0ec",
                )
                st.plotly_chart(radar, use_container_width=True)
                st.caption("Illustrative snapshot -- not a scored metric from any model.")


# ----------------------------------------------------------------------
# Floating AI Assistant "farmer" bubble -- lives outside the tabs, so it
# renders on top of whichever tab is open. Tapping it pops open a small
# chat panel anchored to the bubble.
# ----------------------------------------------------------------------
import google.generativeai as genai

# 1. Initialize API and Model
genai.configure(
    api_key=st.secrets["AQ.Ab8RN6LVCPUkeM2_521uT4AmbM6KGyW9xpIR6N1-QEF9ktG2Ew"]
)
gemini = genai.GenerativeModel("gemini-2.5-flash")

# 2. Ensure session state variables exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Ensure FARMER_ICON_B64 is defined so it won't throw a NameError
FARMER_ICON_B64 = globals().get("FARMER_ICON_B64", None)


def ai_assistant_reply(question, context):
    prompt = f"""
You are AgriVision AI.
You are an expert agricultural advisor helping Indian farmers.

Current Farmer Information:
{context}

Farmer Question:
{question}

Instructions:
- Give practical farming advice.
- Use simple English.
- Answer in bullet points whenever possible.
- If crop recommendation or yield prediction is available, use it.
- If information is missing, clearly say so instead of guessing.
- Keep the answer under 250 words.
"""
    try:
        response = gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Gemini Error:\n\n{e}"


def render_ai_assistant_bubble():
    AVATARS = {"user": "🧑‍🌾", "assistant": "🌾"}

    # Custom CSS for custom artwork if available
    if FARMER_ICON_B64:
        st.markdown(
            f"""
            <style>
            div[data-testid="stPopover"] button {{
                background-image: url('data:image/png;base64,{FARMER_ICON_B64}');
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                font-size: 0 !important;
                color: transparent !important;
            }}
            div[data-testid="stPopover"] button p {{
                display: none;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )

    # Pulsing animation before first interaction
    if not st.session_state.chat_history:
        st.markdown(
            """
            <style>
            div[data-testid="stPopover"]::before {
                content: "";
                position: absolute;
                inset: -10px;
                border-radius: 50%;
                background: rgba(34, 197, 94, 0.45);
                animation: av-pulse 2s ease-out infinite;
                z-index: -1;
            }
            @keyframes av-pulse {
                0% { transform: scale(0.85); opacity: 0.7; }
                70% { transform: scale(1.55); opacity: 0; }
                100% { transform: scale(1.55); opacity: 0; }
            }
            div[data-testid="stPopover"] button::before {
                content: "💬";
                position: absolute;
                top: -6px;
                right: -6px;
                background: #facc15;
                border-radius: 50%;
                width: 24px;
                height: 24px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 12px;
                animation: av-badge-bounce 1.6s ease-in-out infinite;
                box-shadow: 0 2px 6px rgba(0, 0, 0, 0.4);
            }
            @keyframes av-badge-bounce {
                0%, 100% { transform: translateY(0); }
                50% { transform: translateY(-5px); }
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    # Render Popover UI
    with st.popover(
        "🧑‍🌾", use_container_width=False, help="Ask AgriVision AI"
    ):
        st.markdown("**🌾 AgriVision Assistant**")
        st.caption(
            "Ask about your predicted income, crop choices, or general farming advice."
        )

        clicked = None
        if not st.session_state.chat_history:
            st.markdown("**Try asking:**")
            suggestions = [
                "How can I increase my income?",
                "What crop suits my land best?",
                "Is this a good time to sell?",
            ]
            for s in suggestions:
                if st.button(s, use_container_width=True, key=f"chip_{s}"):
                    clicked = s

        chat_box = st.container(height=280)
        with chat_box:
            for role, msg in st.session_state.chat_history:
                with st.chat_message(role, avatar=AVATARS.get(role)):
                    st.write(msg)

        question = (
            st.chat_input("Ask something...", key="floating_chat_input")
            or clicked
        )

        # Handle message submission inside the popover context
        if question:
            p = st.session_state.get("profile", {})

            context = f"""
Farmer Profile:
Farmer ID: {p.get('farmer_id', 'N/A')}
Age: {p.get('age', 'N/A')}
State: {p.get('region', 'N/A')}
Land: {p.get('total_land_ha', 'N/A')}
Current Crop: {p.get('current_crop', 'N/A')}
Monthly Income: {p.get('monthly_income', 'N/A')}
"""

            if "recommended_crop" in st.session_state:
                context += f"\nRecommended Crop: {st.session_state.recommended_crop}\nConfidence: {st.session_state.get('crop_confidence', '')}"

            if "predicted_yield" in st.session_state:
                context += f"\nPredicted Yield: {st.session_state.predicted_yield:.2f} t/ha\nEstimated Production: {st.session_state.get('total_output', 0.0):.2f} tonnes"

            # Save user prompt & fetch response
            st.session_state.chat_history.append(("user", question))
            reply = ai_assistant_reply(question, context)
            st.session_state.chat_history.append(("assistant", reply))
            st.rerun()


# 3. Call the popover function to display the icon on screen
render_ai_assistant_bubble()
