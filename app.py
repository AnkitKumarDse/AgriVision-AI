"""
AgriVision AI -- Streamlit frontend
Run with: streamlit run app.py

REDESIGN NOTES (this version):
- Tabs reordered into a flow: General Dashboard -> Farmer Profile ->
  Weather Intelligence -> Crop Recommendation -> Yield Prediction ->
  Income Prediction -> Final Report. Each step feeds the next via
  st.session_state, and General Dashboard + Final Report both pull
  from everything that's been run so far.
- Removed a duplicate/broken "Predict Farm Value" flow that referenced
  undefined variables (district, gender, irrigated, etc.) -- would
  have crashed on click. Its Agricultural Support inputs are now
  folded into the one real prediction call.
- Removed a hardcoded Groq API key that was sitting in the file
  unused (dead code) -- rotate that key on console.groq.com since it
  was public in the repo. The real, working Groq call reads
  st.secrets["GROQ_API_KEY"] like it should.
- Bigger, bolder visual language: gradient hero title, colored KPI
  cards, glowing active tab.
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

st.set_page_config(page_title="AgriVision AI", page_icon="🌾", layout="wide")

# ----------------------------------------------------------------------
# Load all real models once, safely (never crash the whole app if a
# file is missing -- show a clear message on the tab that needs it).
# ----------------------------------------------------------------------
@st.cache_resource
def load_pickle(name):
    path = Path(__file__).parent / name
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


crop_model = load_pickle("crop_model.pkl")
label_encoder = load_pickle("label_encoder.pkl")
yield_model = load_pickle("yield_model.pkl")
crop_encoder = load_pickle("crop_encoder.pkl")
season_encoder = load_pickle("season_encoder.pkl")
state_encoder = load_pickle("state_encoder.pkl")
farm_value_model = load_pickle("farm_value_final_xgboost.pkl")
farm_value_features = load_pickle("farm_value_final_features.pkl")


def _load_farmer_icon_b64():
    icon_path = Path(__file__).parent / "assets" / "farmer_icon.png"
    if icon_path.exists():
        return base64.b64encode(icon_path.read_bytes()).decode()
    return None


FARMER_ICON_B64 = _load_farmer_icon_b64()

# ----------------------------------------------------------------------
# Styling -- full-screen immersive hero, glass KPI cards, pipeline stepper.
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 0; padding-bottom: 2rem; max-width: 1300px; }
    header[data-testid="stHeader"] { background: transparent; }

    /* ---------------- FULL-SCREEN HERO ---------------- */
    .av-hero {
        margin: -1rem -1rem 0 -1rem;
        min-height: 90vh;
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        text-align: center;
        position: relative; overflow: hidden;
        background: #070b09;
        padding: 3rem 1.5rem 7.5rem 1.5rem;
    }
    .av-hero::before, .av-hero::after {
        content: ""; position: absolute; border-radius: 50%; filter: blur(70px);
        animation: av-float 9s ease-in-out infinite;
    }
    .av-hero::before {
        width: 560px; height: 560px; background: radial-gradient(circle, rgba(34,197,94,0.38), transparent 70%);
        top: -160px; left: -120px;
    }
    .av-hero::after {
        width: 480px; height: 480px; background: radial-gradient(circle, rgba(34,211,238,0.28), transparent 70%);
        bottom: -80px; right: -100px; animation-delay: 3s;
    }
    @keyframes av-float {
        0%, 100% { transform: translate(0, 0) scale(1); }
        50% { transform: translate(30px, -20px) scale(1.08); }
    }
    .av-hero-inner { position: relative; z-index: 2; max-width: 820px; }
    .av-badge {
        display: inline-flex; align-items: center; gap: 6px;
        background: rgba(34,197,94,0.1); color: #4ade80;
        border-radius: 999px; padding: 6px 16px; font-size: 0.8rem;
        font-weight: 700; letter-spacing: 0.08em; margin-bottom: 1.4rem;
        border: 1px solid rgba(74,222,128,0.35);
    }
    .av-badge .dot { width: 7px; height: 7px; border-radius: 50%; background: #4ade80; animation: av-pulse-dot 1.8s infinite; }
    @keyframes av-pulse-dot { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
    .av-hero h1 {
        font-size: clamp(3rem, 8vw, 6rem);
        font-weight: 900; line-height: 0.98; margin-bottom: 1.1rem; letter-spacing: -0.04em;
        background: linear-gradient(100deg, #ffffff 10%, #4ade80 45%, #22d3ee 70%, #facc15 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    }
    .av-hero p.sub { color: #a9bdb2; font-size: clamp(1rem, 1.6vw, 1.25rem); margin: 0 auto 1.8rem auto; max-width: 640px; }

    .av-status-row { display: flex; gap: 10px; flex-wrap: wrap; justify-content: center; }
    .av-status-pill {
        display: flex; align-items: center; gap: 6px; font-size: 0.78rem; font-weight: 600;
        padding: 6px 14px; border-radius: 999px; background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08); color: #cdded4;
    }
    .av-status-pill .led { width: 8px; height: 8px; border-radius: 50%; }
    .led-on { background: #4ade80; box-shadow: 0 0 8px #4ade80; }
    .led-off { background: #64748b; }

    /* ---------------- TABS: float as a glass bar over the hero ---------------- */
    .stTabs { margin-top: -6.2rem; position: relative; z-index: 5; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px; border-bottom: none;
        background: rgba(20, 30, 24, 0.75);
        backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(74, 222, 128, 0.18);
        border-radius: 18px; padding: 10px 12px;
        box-shadow: 0 20px 50px rgba(0,0,0,0.55);
        flex-wrap: wrap;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent; border-radius: 12px;
        padding: 14px 20px; color: #8fa89b; font-weight: 800; font-size: 0.95rem;
        letter-spacing: 0.01em;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1c3327, #14261d) !important;
        color: #4ade80 !important; box-shadow: 0 0 0 1px rgba(74,222,128,0.4), 0 0 18px rgba(74,222,128,0.25);
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #10160f; border: 1px solid #1c2b22; border-radius: 18px;
    }
    div[data-testid="stMetric"] {
        background-color: #131b18; border: 1px solid #21362a; border-radius: 12px; padding: 16px 18px;
    }
    div[data-testid="stMetricLabel"] { color: #9db3a8; }

    button[kind="primary"] {
        background: linear-gradient(90deg, #22c55e, #16a34a);
        border: none; font-weight: 700; letter-spacing: 0.01em;
    }
    button[kind="primary"]:hover { filter: brightness(1.12); }

    /* ---------------- PIPELINE STEPPER ---------------- */
    .av-pipeline { display: flex; align-items: flex-start; justify-content: space-between; margin: 0.5rem 0 1.8rem 0; }
    .av-step { display: flex; flex-direction: column; align-items: center; flex: 1; position: relative; }
    .av-step .circle {
        width: 46px; height: 46px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
        font-size: 1.2rem; font-weight: 800; border: 2px solid #2a3d31; background: #131c17; color: #5f7a6b;
        transition: all 0.3s ease; z-index: 2;
    }
    .av-step.done .circle { background: linear-gradient(135deg, #22c55e, #16a34a); border-color: #4ade80; color: white; box-shadow: 0 0 16px rgba(34,197,94,0.5); }
    .av-step.active .circle { border-color: #facc15; color: #facc15; animation: av-pulse-ring 1.6s infinite; }
    @keyframes av-pulse-ring { 0% { box-shadow: 0 0 0 0 rgba(250,204,21,0.5); } 70% { box-shadow: 0 0 0 10px rgba(250,204,21,0); } 100% { box-shadow: 0 0 0 0 rgba(250,204,21,0); } }
    .av-step .label { margin-top: 8px; font-size: 0.74rem; font-weight: 700; color: #9db3a8; text-align: center; max-width: 90px; }
    .av-step.done .label { color: #4ade80; }
    .av-step::after {
        content: ""; position: absolute; top: 23px; left: calc(50% + 23px); width: calc(100% - 46px); height: 2px;
        background: #2a3d31; z-index: 1;
    }
    .av-step:last-child::after { display: none; }
    .av-step.done::after { background: linear-gradient(90deg, #4ade80, #2a3d31); }

    /* ---------------- GLASS KPI CARDS ---------------- */
    .kpi-card {
        border-radius: 18px; padding: 22px 24px; border: 1px solid rgba(255,255,255,0.08);
        background: linear-gradient(135deg, var(--c1), var(--c2));
        position: relative; overflow: hidden; min-height: 128px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover { transform: translateY(-3px); box-shadow: 0 10px 26px rgba(0,0,0,0.35); }
    .kpi-card .kpi-label { font-size: 0.78rem; color: rgba(255,255,255,0.75); font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; }
    .kpi-card .kpi-value { font-size: 1.85rem; font-weight: 800; color: white; margin-top: 8px; }
    .kpi-card .kpi-sub { font-size: 0.78rem; color: rgba(255,255,255,0.65); margin-top: 4px; }

    /* ---------------- Floating AI-assistant bubble ---------------- */
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
    @keyframes av-panel-in { 0% { opacity: 0; transform: translateY(6px) scale(0.97); } 100% { opacity: 1; transform: translateY(0) scale(1); } }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_hero():
    checks = {
        "Crop Model": crop_model is not None,
        "Yield Model": yield_model is not None,
        "Farm Value Model": farm_value_model is not None,
        "Weather API": bool(st.secrets.get("OPENWEATHER_API_KEY", None)),
        "AI Assistant": bool(st.secrets.get("GROQ_API_KEY", None)),
    }
    pills = "".join(
        f'<div class="av-status-pill"><span class="led {"led-on" if ok else "led-off"}"></span>{name}</div>'
        for name, ok in checks.items()
    )
    st.markdown(
        f"""
        <div class="av-hero">
            <div class="av-hero-inner">
                <div class="av-badge"><span class="dot"></span> LIVE AI FARM INTELLIGENCE PLATFORM</div>
                <h1>AgriVision AI</h1>
                <p class="sub">One connected pipeline -- your profile, live weather, crop science, yield forecasting and farm valuation all feed into a single real-time picture of the farm.</p>
                <div class="av-status-row">{pills}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


render_hero()


def kpi_card(label, value, sub, c1, c2, icon=""):
    st.markdown(
        f"""
        <div class="kpi-card" style="--c1:{c1};--c2:{c2};">
            <div class="kpi-label">{icon} {label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------
for key, default in [
    ("profile", {}),
    ("chat_history", []),
    ("recommended_crop", None),
    ("crop_confidence", None),
    ("predicted_yield", None),
    ("total_output", None),
    ("farm_value_result", None),
    ("weather_result", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ----------------------------------------------------------------------
# Sidebar -- kept minimal; live status now lives in the hero itself.
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🌾 AgriVision AI")
    st.caption("Navigate the tabs above to move through the pipeline: Profile → Weather → Crop → Yield → Income → Report.")

tabs = st.tabs(
    [
        "🚀 General Dashboard",
        "👤 Farmer Profile",
        "🌦️ Weather Intelligence",
        "🌱 Crop Recommendation",
        "📈 Yield Prediction",
        "💰 Income Prediction",
        "📄 Final Report",
    ]
)

# ----------------------------------------------------------------------
# 0. General Dashboard -- India agri market overview. Deliberately does
# NOT depend on the farmer's profile -- this is the "walk up and see
# something useful" landing screen. Personal predictions live in their
# own tabs and roll up into Final Report.
# ----------------------------------------------------------------------
@st.cache_data(ttl=1800)
def fetch_agri_news():
    """Live headlines if NEWSAPI_KEY is set in secrets, else None
    (caller falls back to a clearly-labeled static sample)."""
    api_key = st.secrets.get("NEWSAPI_KEY", None)
    if not api_key:
        return None
    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={"q": "India agriculture", "language": "en", "sortBy": "publishedAt", "pageSize": 5, "apiKey": api_key},
            timeout=8,
        )
        if resp.status_code != 200:
            return None
        articles = resp.json().get("articles", [])
        return [{"title": a["title"], "source": a["source"]["name"], "url": a["url"]} for a in articles[:5]]
    except requests.RequestException:
        return None


with tabs[0]:
    with st.container(border=True):
        st.markdown("#### 🇮🇳 Indian Agriculture — Market Pulse")
        st.caption("A general snapshot of Indian agriculture. Your personal predictions live in the tabs to the right.")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("Arable Land", "~156M ha", "2nd largest in the world (FAO estimate)", "#14532d", "#0b1a10", "🌍")
        with c2:
            kpi_card("Agri Workforce", "~42%", "share of India's total workforce", "#78350f", "#1c1206", "👨‍🌾")
        with c3:
            kpi_card("Agri GDP Share", "~18%", "of India's gross value added", "#164e63", "#071a1f", "📊")
        with c4:
            kpi_card("Farm Households", "~146M", "mostly small & marginal holdings", "#4c1d95", "#160b2e", "🏡")

        st.divider()
        col_left, col_right = st.columns([1.3, 1])
        with col_left:
            st.markdown("**Illustrative Crop Price Index** *(sample data — connect a market API for live prices)*")
            crops = ["Wheat", "Paddy", "Cotton", "Soybean", "Mustard", "Sugarcane"]
            index_vals = [104, 98, 121, 92, 108, 101]
            bar = go.Figure(go.Bar(x=crops, y=index_vals, marker_color="#4ade80"))
            bar.update_layout(
                height=280, yaxis=dict(title="Index (base 100)"),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e8f0ec",
                margin=dict(t=10),
            )
            st.plotly_chart(bar, use_container_width=True)
        with col_right:
            st.markdown("**Seasonal Crop Calendar**")
            st.markdown(
                """
                <div style="display:flex;flex-direction:column;gap:10px;">
                <div style="background:#0f1a15;border:1px solid #21362a;border-radius:12px;padding:12px 16px;">
                <b style="color:#4ade80;">🌧️ Kharif (Jun–Oct)</b><br><span style="color:#9db3a8;">Rice, Maize, Cotton, Soybean, Groundnut</span>
                </div>
                <div style="background:#0f1a15;border:1px solid #21362a;border-radius:12px;padding:12px 16px;">
                <b style="color:#facc15;">❄️ Rabi (Oct–Mar)</b><br><span style="color:#9db3a8;">Wheat, Mustard, Gram, Barley</span>
                </div>
                <div style="background:#0f1a15;border:1px solid #21362a;border-radius:12px;padding:12px 16px;">
                <b style="color:#22d3ee;">☀️ Zaid (Mar–Jun)</b><br><span style="color:#9db3a8;">Watermelon, Cucumber, Moong</span>
                </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()
        st.markdown("**📰 Latest in Indian Agriculture**")
        news = fetch_agri_news()
        if news:
            for item in news:
                st.markdown(f"- [{item['title']}]({item['url']}) — *{item['source']}*")
        else:
            if not st.secrets.get("NEWSAPI_KEY", None):
                st.caption("Sample headlines below — add `NEWSAPI_KEY` in Streamlit secrets for a live feed.")
            sample_news = [
                "Government raises MSP for key Rabi crops ahead of sowing season",
                "Monsoon forecast points to normal rainfall across major farming states",
                "New irrigation scheme targets small and marginal farmers",
                "Digital agriculture push: states expand soil health card coverage",
                "Export demand for Indian spices rises amid supply chain shifts",
            ]
            for headline in sample_news:
                st.markdown(f"- {headline} *(sample)*")

    st.write("")
    with st.container(border=True):
        st.markdown("#### Your Pipeline")
        st.caption("Fill in your Farmer Profile, then move through the tabs — each stage below lights up as you complete it.")
        p = st.session_state.profile
        steps = [
            ("1", "Profile", bool(p)),
            ("2", "Weather", bool(st.session_state.weather_result)),
            ("3", "Crop AI", bool(st.session_state.recommended_crop)),
            ("4", "Yield", bool(st.session_state.predicted_yield)),
            ("5", "Income", bool(st.session_state.farm_value_result)),
            ("6", "Report", bool(p) and bool(st.session_state.farm_value_result)),
        ]
        first_incomplete = next((i for i, s in enumerate(steps) if not s[2]), None)
        step_html = ""
        for i, (num, label, done) in enumerate(steps):
            state = "done" if done else ("active" if i == first_incomplete else "")
            icon = "✓" if done else num
            step_html += f'<div class="av-step {state}"><div class="circle">{icon}</div><div class="label">{label}</div></div>'
        st.markdown(f'<div class="av-pipeline">{step_html}</div>', unsafe_allow_html=True)


with tabs[1]:
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
            st.success("Profile saved. It'll now feed every other tab.")
            st.toast(f"Profile for {farmer_id} saved!", icon="✅")

# ----------------------------------------------------------------------
# 2. Weather Intelligence
# ----------------------------------------------------------------------
WEATHER_API_KEY = st.secrets.get("WEATHER_API_KEY", None)


def get_weather(city):
    if not WEATHER_API_KEY:
        return None
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={WEATHER_API_KEY}&units=metric"
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


with tabs[2]:
    with st.container(border=True):
        st.header("🌦 Weather Intelligence Dashboard")
        p = st.session_state.profile

        if not WEATHER_API_KEY:
            st.info(
                "Add your OpenWeatherMap key in Streamlit Cloud under Manage app → Settings → Secrets, as:\n\n"
                "`OPENWEATHER_API_KEY = \"your-key-here\"`"
            )

        city = st.text_input("Enter City", value=p.get("region", "Patna"))

        if st.button("Get Live Weather", type="primary"):
            weather = get_weather(city)
            if weather is None:
                st.error("Unable to fetch weather." if WEATHER_API_KEY else "No weather API key configured yet.")
            else:
                st.session_state.weather_result = weather

        weather = st.session_state.weather_result
        if weather:
            st.success(f"Live Weather • {weather['city']}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🌡 Temperature", f"{weather['temperature']} °C")
            c2.metric("💧 Humidity", f"{weather['humidity']}%")
            c3.metric("🌬 Wind", f"{weather['wind']} m/s")
            c4.metric("🧭 Pressure", f"{weather['pressure']} hPa")

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
# 3. Crop Recommendation
# ----------------------------------------------------------------------
with tabs[3]:
    with st.container(border=True):
        st.header("🌾 AI-Powered Crop Recommendation")
        st.caption("Enter the latest soil test values to get the best crop recommendation.")

        if crop_model is None or label_encoder is None:
            st.info("crop_model.pkl / label_encoder.pkl not found next to app.py -- this tab needs those to run.")

        col1, col2 = st.columns(2)
        with col1:
            N = st.number_input("Nitrogen (N)", 0, 200, 90)
            P = st.number_input("Phosphorus (P)", 0, 200, 42)
            K = st.number_input("Potassium (K)", 0, 250, 43)
            ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)
        with col2:
            temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 27.0, key="crop_temp")
            humidity = st.number_input("Humidity (%)", 0.0, 100.0, 80.0)
            rainfall = st.number_input("Rainfall (mm)", 0.0, 3000.0, 800.0, key="crop_rainfall")

        if st.button("🌱 Generate Crop Recommendation", type="primary"):
            if crop_model is None or label_encoder is None:
                st.error("Model files missing -- can't run a real prediction yet.")
            else:
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
                    N, P, K, temperature, humidity, ph, rainfall, npk_total,
                    n_ratio, p_ratio, k_ratio, rainfall_low, rainfall_medium,
                    rainfall_high, temp_cool, temp_moderate, temp_hot,
                ]])

                prediction = crop_model.predict(features)
                crop = label_encoder.inverse_transform(prediction)[0]
                confidence = float(np.max(crop_model.predict_proba(features)) * 100)

                st.session_state.recommended_crop = crop
                st.session_state.crop_confidence = confidence

        if st.session_state.recommended_crop:
            crop = st.session_state.recommended_crop
            confidence = st.session_state.crop_confidence
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
            st.caption("This feeds into Yield Prediction as the default crop below.")

# ----------------------------------------------------------------------
# 4. Yield Prediction
# ----------------------------------------------------------------------
with tabs[4]:
    with st.container(border=True):
        st.header("🌾 AI Yield Prediction")
        st.caption("Predict expected crop yield using Machine Learning.")

        if yield_model is None:
            st.info("yield_model.pkl (and its encoders) not found next to app.py -- this tab needs those to run.")

        crop_options = [
            "Rice", "Wheat", "Maize", "Cotton", "Sugarcane", "Barley",
            "Millets", "Groundnut", "Soybean", "Potato", "Gram", "Turmeric",
        ]
        default_crop = st.session_state.recommended_crop
        default_index = 0
        if default_crop:
            for i, opt in enumerate(crop_options):
                if opt.lower() == default_crop.lower():
                    default_index = i
                    break

        left, right = st.columns(2)
        with left:
            crop = st.selectbox("Crop", crop_options, index=default_index)
            if default_crop and crop.lower() == default_crop.lower():
                st.caption(f"✓ Pre-filled from your Crop Recommendation ({st.session_state.crop_confidence:.0f}% confidence)")
            season = st.selectbox("Season", ["Kharif", "Rabi", "Summer", "Whole Year", "Winter", "Autumn"])
            state = st.text_input("State", st.session_state.profile.get("region", "Punjab"))
            crop_year = st.number_input("Crop Year", 1997, 2035, 2026)
        with right:
            area = st.number_input("Area (Hectares)", value=float(st.session_state.profile.get("total_land_ha", 5.0)))
            production = st.number_input("Production (Tonnes)", value=20.0)
            rainfall = st.number_input("Annual Rainfall (mm)", value=800.0, key="yield_rainfall")
            fertilizer = st.number_input("Fertilizer", value=450.0)
            pesticide = st.number_input("Pesticide", value=8.0)

        if st.button("🚀 Predict Yield", type="primary", use_container_width=True):
            if yield_model is None or crop_encoder is None or season_encoder is None or state_encoder is None:
                st.error("Model files missing -- can't run a real prediction yet.")
            else:
                try:
                    crop_c = crop.strip()
                    season_c = season.strip()
                    state_c = state.strip()

                    crop_encoded = crop_encoder.transform([crop_c])[0]
                    season_classes = [x.strip() for x in season_encoder.classes_]
                    season_encoded = season_classes.index(season_c)
                    state_classes = [x.strip() for x in state_encoder.classes_]
                    state_encoded = state_classes.index(state_c)

                    input_data = pd.DataFrame({
                        "Crop": [crop_encoded],
                        "Crop_Year": [crop_year],
                        "Season": [season_encoded],
                        "State": [state_encoded],
                        "Area": [area],
                        "Production": [production],
                        "Annual_Rainfall": [rainfall],
                        "Fertilizer": [fertilizer],
                        "Pesticide": [pesticide],
                    })

                    predicted_yield = float(yield_model.predict(input_data)[0])
                    total_output = predicted_yield * area

                    st.session_state.predicted_yield = predicted_yield
                    st.session_state.total_output = total_output
                    st.session_state.yield_crop = crop_c
                    st.session_state.yield_season = season_c
                    st.session_state.yield_state = state_c
                except Exception as e:
                    st.error("Prediction Failed")
                    st.exception(e)

        if st.session_state.predicted_yield:
            predicted_yield = st.session_state.predicted_yield
            total_output = st.session_state.total_output
            st.success("✅ Yield Prediction Completed Successfully!")

            m1, m2, m3 = st.columns(3)
            m1.metric("🌾 Predicted Yield", f"{predicted_yield:.2f} t/ha")
            m2.metric("📦 Estimated Production", f"{total_output:.2f} Tonnes")
            category = "Excellent 🟢" if predicted_yield >= 5 else ("Average 🟡" if predicted_yield >= 3 else "Low 🔴")
            m3.metric("Yield Category", category)

            st.divider()
            left2, right2 = st.columns(2)
            with left2:
                st.info(f"**Crop:** {st.session_state.yield_crop}\n\n**Season:** {st.session_state.yield_season}\n\n**State:** {st.session_state.yield_state}")
            with right2:
                st.info(f"**Predicted Yield:** {predicted_yield:.2f} t/ha\n\n**Estimated Production:** {total_output:.2f} Tonnes")

            st.progress(min(predicted_yield / 8, 1.0))
            st.caption("This feeds into your Final Report below.")

# ----------------------------------------------------------------------
# 5. Income Prediction (Farm Value Estimation)
# ----------------------------------------------------------------------
with tabs[5]:
    with st.container(border=True):
        st.header("🌾 Income Prediction — Farm Value Estimation")
        st.caption("Estimate the total agricultural value of the farm using the trained XGBoost model.")

        p = st.session_state.profile
        if not p:
            st.warning("Please complete the Farmer Profile first.")
        elif farm_value_model is None:
            st.info("farm_value_final_xgboost.pkl / farm_value_final_features.pkl not found next to app.py -- this tab needs those to run.")
        else:
            st.subheader("Farmer & Farm Information")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Land Area", f"{p.get('total_land_ha', 5.0):.2f} ha")
            col2.metric("Primary Crop", st.session_state.recommended_crop or p.get('current_crop', 'Wheat'))
            col3.metric("State", p.get('region', 'Punjab'))
            col4.metric("Age", f"{p.get('age', 35)} years")
            if st.session_state.recommended_crop:
                st.caption(f"✓ Using your recommended crop ({st.session_state.recommended_crop}) instead of the profile default.")

            st.markdown("---")
            st.subheader("🌱 Farm Details")
            col1, col2 = st.columns(2)
            with col1:
                total_land = float(p.get("total_land_ha", 5.0))
                irrigated_area = st.number_input(
                    "Irrigated Area (hectares)", min_value=0.0, max_value=total_land,
                    value=min(3.0, total_land), step=0.1,
                    help="Portion of your agricultural land that is irrigated.",
                )
            with col2:
                loan_amount = st.number_input(
                    "Agricultural Loan Amount (₹)", min_value=0.0, max_value=10000000.0,
                    value=0.0, step=1000.0, help="Current agricultural loan amount.",
                )

            st.markdown("---")
            st.subheader("📋 Agricultural Support")
            col1, col2 = st.columns(2)
            with col1:
                crop_insured = st.selectbox("Crop Insured", ["Yes", "No"], index=1)
                soil_health_card = st.selectbox("Soil Health Card", ["Yes", "No"], index=1)
            with col2:
                followed_soil_health_card = st.selectbox("Followed Soil Health Card", ["Yes", "No"], index=1)
                farmer_organization = st.selectbox("Farmer Organization", ["Yes", "No"], index=1)

            st.markdown("")
            if st.button("🌾 Predict Farm Value", type="primary", use_container_width=True):
                with st.spinner("Analyzing farm characteristics..."):
                    age = float(p.get("age", 35))
                    land_area = float(p.get("total_land_ha", 5.0))
                    state = p.get("region", "Punjab")
                    current_crop = st.session_state.recommended_crop or p.get("current_crop", "Wheat")
                    education = p.get("education_level", "Secondary")

                    crop_map = {
                        "Wheat": "Cereals", "Rice": "Cereals", "Maize": "Cereals", "Bajra": "Cereals",
                        "Jowar": "Cereals", "Barley": "Cereals", "Pulses": "Pulses", "Groundnut": "Oilseeds",
                        "Mustard": "Oilseeds", "Cotton": "Fibres", "Sugarcane": "Sugar Crops",
                        "Potato": "Tuber Crops", "Vegetables": "Vegetables", "Fruits": "Fruits",
                        "Spices": "Condiments & Spices",
                    }
                    major_crop = crop_map.get(current_crop, "Cereals")

                    land_safe = max(land_area, 0.001)
                    irrigation_intensity = irrigated_area / land_safe
                    loan_per_ha = loan_amount / land_safe

                    input_data = pd.DataFrame([{
                        "state": state, "district": 1, "gender": "Male", "education": education,
                        "agri_training": 2, "Principal_Activity": "Cultivation", "agricultural_land": "Yes",
                        "major_crop": major_crop, "irrigated": "Yes" if irrigated_area > 0 else "No",
                        "irrigation_source": "Ground Water", "bank_account": "Yes",
                        "kcc": "Yes" if loan_amount > 0 else "No", "msp_awareness": "Yes", "sold_at_msp": "No",
                        "technical_advice": "No", "advice_adopted": "No",
                        "crop_insured": crop_insured, "soil_health_card": soil_health_card,
                        "followed_soil_health_card": followed_soil_health_card,
                        "farmer_organization": farmer_organization,
                        "age": age, "household_size": 5.0, "land_area": land_area,
                        "irrigated_area": irrigated_area, "crops_grown": 2.0, "wages_salary": 0.0,
                        "land_rent_income": 0.0,
                        "nonfarm_net_income": float(p.get("non_agri_income", 12000)) * 12,
                        "loan_amount": loan_amount, "interest_rate": 7.0,
                        "irrigation_intensity": irrigation_intensity, "loan_per_ha": loan_per_ha,
                        "crops_per_ha": 2.0 / land_safe, "household_density": 5.0 / land_safe,
                        "irrigated_share": irrigation_intensity, "interest_burden": loan_amount * 0.07,
                        "loan_land_interaction": loan_amount * land_area, "land_size_squared": land_area ** 2,
                        "msp_sell_rate": 0.0,
                    }])

                    predicted_value = max(0, float(farm_value_model.predict(input_data)[0]))
                    st.session_state.farm_value_result = predicted_value

            if st.session_state.farm_value_result:
                predicted_value = st.session_state.farm_value_result
                st.markdown("---")
                st.success("Farm value estimated successfully!")
                col1, col2 = st.columns([1.3, 1])
                with col1:
                    st.metric("Estimated Farm Value", f"₹ {predicted_value:,.0f}")
                    st.caption("Estimated using the trained XGBoost model.")
                with col2:
                    gauge = go.Figure(go.Indicator(
                        mode="gauge+number", value=61, number={"suffix": "%"}, title={"text": "Model R²"},
                        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#22c55e"}, "bgcolor": "#131b18",
                               "steps": [{"range": [0, 40], "color": "#2a1414"}, {"range": [40, 70], "color": "#2a2414"},
                                         {"range": [70, 100], "color": "#14261a"}]},
                    ))
                    gauge.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)", font_color="#e8f0ec")
                    st.plotly_chart(gauge, use_container_width=True)
                with st.expander("How does this prediction work?"):
                    st.write(
                        "The model uses the farmer's profile, land characteristics, crop information, "
                        "irrigation and financial information (including your Agricultural Support answers above) "
                        "to estimate total farm value."
                    )
                st.caption("This feeds into your Final Report below.")

# ----------------------------------------------------------------------
# 6. Final Report
# ----------------------------------------------------------------------
with tabs[6]:
    with st.container(border=True):
        st.header("📄 Final Report")
        p = st.session_state.profile
        if not p:
            st.info("Fill in the Farmer Profile tab to generate a report.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Land Holding", f"{p.get('total_land_ha')} ha")
            c2.metric("Recommended Crop", st.session_state.recommended_crop.upper() if st.session_state.recommended_crop else "Not run yet")
            c3.metric("Predicted Yield", f"{st.session_state.predicted_yield:.2f} t/ha" if st.session_state.predicted_yield else "Not run yet")
            c4.metric("Farm Value", f"₹{st.session_state.farm_value_result:,.0f}" if st.session_state.farm_value_result else "Not run yet")

            st.divider()

            # ---- Composite AI Farm Score ----
            scored_parts = []
            if st.session_state.crop_confidence:
                scored_parts.append(st.session_state.crop_confidence)
            if st.session_state.predicted_yield:
                scored_parts.append(min(st.session_state.predicted_yield / 8 * 100, 100))
            if st.session_state.farm_value_result:
                scored_parts.append(min(st.session_state.farm_value_result / 300000 * 100, 100))
            ai_score = sum(scored_parts) / len(scored_parts) if scored_parts else 0

            if ai_score >= 75:
                score_label, score_color = "🌟 Strong Farm Potential", "#22c55e"
            elif ai_score >= 45:
                score_label, score_color = "✅ Good Potential", "#facc15"
            elif ai_score > 0:
                score_label, score_color = "⚠️ Needs Improvement", "#f97316"
            else:
                score_label, score_color = "Getting Started", "#64748b"

            col_score, col_charts = st.columns([1, 2])
            with col_score:
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=ai_score,
                    number={"font": {"size": 40, "color": score_color}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#3a4d41"},
                        "bar": {"color": score_color, "thickness": 0.28}, "bgcolor": "#0d1310", "borderwidth": 0,
                        "steps": [{"range": [0, 45], "color": "#2a1414"}, {"range": [45, 75], "color": "#2a2414"}, {"range": [75, 100], "color": "#14261a"}],
                    },
                ))
                gauge.update_layout(height=210, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", font_color="#e8f0ec")
                st.plotly_chart(gauge, use_container_width=True)
                st.markdown(f"<div style='text-align:center;font-weight:800;color:{score_color};margin-top:-14px;'>{score_label}</div>", unsafe_allow_html=True)
                st.caption("AI Farm Score -- blends crop confidence, yield, and farm value.")

            with col_charts:
                cc1, cc2 = st.columns(2)
                with cc1:
                    if st.session_state.farm_value_result:
                        base = st.session_state.farm_value_result
                        seasons = ["S1", "S2", "S3", "S4"]
                        values = [base * f for f in (0.85, 1.05, 0.95, 1.0)]
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=seasons, y=values, fill="tozeroy", line=dict(color="#22c55e")))
                        fig.update_layout(title="Simulated Value Trend", height=210, margin=dict(t=30), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e8f0ec")
                        st.plotly_chart(fig, use_container_width=True)
                with cc2:
                    categories = ["Land", "Non-Ag Income", "Market Access", "Yield", "Farm Value"]
                    land_score = min(p.get("total_land_ha", 0) / 20, 1)
                    nonagri_score = min(p.get("non_agri_income", 0) / 50000, 1)
                    market_score = 1 - min(p.get("distance_to_market_km", 0) / 100, 1)
                    yield_score = min((st.session_state.predicted_yield or 3) / 10, 1)
                    value_score = min((st.session_state.farm_value_result or 20000) / 300000, 1)
                    values = [land_score, nonagri_score, market_score, yield_score, value_score]
                    radar = go.Figure()
                    radar.add_trace(go.Scatterpolar(r=values + values[:1], theta=categories + categories[:1], fill="toself", line_color="#22c55e"))
                    radar.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 1], showticklabels=False), bgcolor="rgba(0,0,0,0)"),
                        showlegend=False, height=210, title="Farmer Snapshot", margin=dict(t=30),
                        paper_bgcolor="rgba(0,0,0,0)", font_color="#e8f0ec",
                    )
                    st.plotly_chart(radar, use_container_width=True)

            st.divider()
            report_lines = [
                "AgriVision AI -- Farmer Report",
                f"Farmer ID: {p.get('farmer_id')}",
                f"Region: {p.get('region')}",
                f"Land: {p.get('total_land_ha')} ha",
                f"Current Crop: {p.get('current_crop')}",
            ]
            if st.session_state.recommended_crop:
                report_lines.append(f"Recommended Crop: {st.session_state.recommended_crop} ({st.session_state.crop_confidence:.1f}% confidence)")
            if st.session_state.predicted_yield:
                report_lines.append(f"Predicted Yield: {st.session_state.predicted_yield:.2f} t/ha ({st.session_state.total_output:.1f} tonnes total)")
            if st.session_state.farm_value_result:
                report_lines.append(f"Estimated Farm Value: ₹{st.session_state.farm_value_result:,.0f}")
            if st.session_state.weather_result:
                w = st.session_state.weather_result
                report_lines.append(f"Weather at time of report: {w['temperature']}°C, {w['condition']} ({w['city']})")

            report_text = "\n".join(report_lines)
            st.text_area("Report preview", report_text, height=220)
            st.download_button("📥 Download Report (.txt)", report_text, file_name=f"{p.get('farmer_id', 'farmer')}_report.txt", type="primary")


# ----------------------------------------------------------------------
# Floating AI Assistant "farmer" bubble (Groq + Llama 3)
# ----------------------------------------------------------------------
def ai_assistant_reply(question, context):
    groq_api_key = st.secrets.get("GROQ_API_KEY", None)
    if not groq_api_key:
        return "❌ **Groq API Key Missing!**\n\nAdd `GROQ_API_KEY` in Streamlit Cloud: **Manage App → Settings → Secrets**."

    prompt = f"""You are AgriVision AI, an expert agricultural advisor helping Indian farmers.

Current Farmer Information:
{context}

Farmer Question:
{question}

Instructions:
- Give practical farming advice.
- Use simple English.
- Answer in bullet points whenever possible.
- If crop recommendation, yield, or farm value predictions are available, use them.
- Keep the answer under 250 words."""

    try:
        from groq import Groq
        client = Groq(api_key=groq_api_key)
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=400,
        )
        return completion.choices[0].message.content
    except Exception as e:  # noqa: BLE001
        return f"❌ Groq API Error:\n\n{e}"


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

    with st.popover("🧑‍🌾", use_container_width=False, help="Ask AgriVision AI"):
        st.markdown("**🌾 AgriVision Assistant**")
        st.caption("Ask about your predicted income, crop choices, or general farming advice.")

        clicked = None
        if not st.session_state.chat_history:
            st.markdown("**Try asking:**")
            for s in ["How can I increase my income?", "What crop suits my land best?", "Is this a good time to sell?"]:
                if st.button(s, use_container_width=True, key=f"chip_{s}"):
                    clicked = s

        chat_box = st.container(height=280)
        with chat_box:
            for role, msg in st.session_state.chat_history:
                with st.chat_message(role, avatar=AVATARS.get(role)):
                    st.write(msg)

        question = st.chat_input("Ask something...", key="floating_chat_input") or clicked
        if question:
            p = st.session_state.get("profile", {})
            context = (
                f"Farmer ID: {p.get('farmer_id', 'N/A')}\n"
                f"Age: {p.get('age', 'N/A')}\nState: {p.get('region', 'N/A')}\n"
                f"Land: {p.get('total_land_ha', 'N/A')} ha\nCurrent Crop: {p.get('current_crop', 'N/A')}\n"
            )
            if st.session_state.recommended_crop:
                context += f"Recommended Crop: {st.session_state.recommended_crop} ({st.session_state.crop_confidence:.0f}% confidence)\n"
            if st.session_state.predicted_yield:
                context += f"Predicted Yield: {st.session_state.predicted_yield:.2f} t/ha\n"
            if st.session_state.farm_value_result:
                context += f"Estimated Farm Value: ₹{st.session_state.farm_value_result:,.0f}\n"

            st.session_state.chat_history.append(("user", question))
            with st.spinner("AgriVision AI is thinking..."):
                reply = ai_assistant_reply(question, context)
            st.session_state.chat_history.append(("assistant", reply))
            st.rerun()


render_ai_assistant_bubble()
