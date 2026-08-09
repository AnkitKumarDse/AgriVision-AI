import base64
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from groq import Groq

# ----------------------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="AgriVision AI -- Smart Agricultural Platform",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session States
if "profile" not in st.session_state:
    st.session_state.profile = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Base64 Farmer Avatar Icon Helper
def _load_farmer_icon_b64():
    icon_path = Path(__file__).parent / "assets" / "farmer_icon.png"
    if icon_path.exists():
        return base64.b64encode(icon_path.read_bytes()).decode()
    return None


FARMER_ICON_B64 = _load_farmer_icon_b64()


# ----------------------------------------------------------------------
# Model Loaders (Cached)
# ----------------------------------------------------------------------
@st.cache_resource
def load_all_models():
    models = {}
    try:
        models["crop"] = joblib.load("crop_model.pkl")
        models["crop_label"] = joblib.load("label_encoder.pkl")
    except Exception:
        models["crop"] = None

    try:
        models["yield"] = joblib.load("yield_model.pkl")
        models["crop_enc"] = joblib.load("crop_encoder.pkl")
        models["season_enc"] = joblib.load("season_encoder.pkl")
        models["state_enc"] = joblib.load("state_encoder.pkl")
    except Exception:
        models["yield"] = None

    try:
        models["farm_val"] = joblib.load("farm_value_final_xgboost.pkl")
    except Exception:
        models["farm_val"] = None

    return models


ml_models = load_all_models()

# ----------------------------------------------------------------------
# Ultra-Modern CSS Styling (Gradients, Glassmorphism, Pop Colors)
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Global Background & Layout */
    .stApp {
        background: #090d16;
        color: #f8fafc;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Vibrant Hero Header */
    .hero-container {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(6, 182, 212, 0.15) 50%, rgba(168, 85, 247, 0.15) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 20px;
        padding: 2.2rem 2.8rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(12px);
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #34d399, #38bdf8, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero-subtitle {
        color: #cbd5e1;
        font-size: 1.1rem;
        margin: 0;
    }
    .badge-pop {
        display: inline-block;
        background: linear-gradient(90deg, #059669, #0284c7);
        color: #ffffff;
        border-radius: 999px;
        padding: 4px 14px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        margin-bottom: 0.8rem;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }

    /* Pop Cards & Metric Cards */
    .glass-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(51, 65, 85, 0.8);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(8px);
    }
    
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.9));
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 14px;
        padding: 16px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        color: #34d399 !important;
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 2px solid #1e293b;
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background: #0f172a;
        border-radius: 10px;
        padding: 12px 22px;
        color: #94a3b8;
        font-weight: 600;
        border: 1px solid #1e293b;
        transition: all 0.25s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: #ffffff !important;
        border-color: #34d399 !important;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
    }

    /* Buttons */
    button[kind="primary"] {
        background: linear-gradient(90deg, #10b981, #0284c7) !important;
        border: none !important;
        color: white !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
        padding: 0.6rem 1.5rem !important;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.35) !important;
        transition: all 0.2s ease !important;
    }
    button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.5) !important;
    }

    /* Floating Chatbot Bubble */
    div[data-testid="stPopover"] {
        position: fixed !important;
        top: 65%;
        left: 28px;
        z-index: 9999;
        animation: av-bob 3.5s ease-in-out infinite;
    }
    @keyframes av-bob {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    div[data-testid="stPopover"] button {
        width: 70px;
        height: 70px;
        border-radius: 50%;
        background: linear-gradient(135deg, #10b981, #3b82f6);
        border: 3px solid #ffffff;
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.5);
    }
    </style>

    <div class="hero-container">
        <div class="badge-pop">POWERED BY AI &amp; NEXT-GEN ANALYTICS</div>
        <div class="hero-title">🌾 AgriVision AI</div>
        <div class="hero-subtitle">Smart Decision Support Platform for Indian Agriculture -- Live Market News, Crop Science, Yield AI &amp; Farm Value Estimation</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Sidebar Profile Snapshot & Model Status
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🌾 **AgriVision AI**")
    st.caption("Precision Farming & Analytics")
    st.divider()

    p = st.session_state.profile
    if p:
        st.markdown("### 👤 **Farmer Snapshot**")
        st.write(f"**ID:** `{p.get('farmer_id', 'N/A')}`")
        st.write(f"📍 **State:** {p.get('region', 'N/A')}")
        st.write(f"🌱 **Crop:** {p.get('current_crop', 'N/A')}")
        st.write(f"📐 **Land:** {p.get('total_land_ha', '0.0')} ha")
        st.divider()

    st.markdown("### ⚡ **ML Engine Status**")
    st.write(
        f"🌱 Crop Rec Model: {'🟢 Live' if ml_models['crop'] else '🟡 Demo'}"
    )
    st.write(
        f"📈 Yield Model: {'🟢 Live' if ml_models['yield'] else '🟡 Demo'}"
    )
    st.write(
        f"💰 Farm Value XGBoost: {'🟢 Live' if ml_models['farm_val'] else '🟡 Demo'}"
    )

# ----------------------------------------------------------------------
# Main Application Flow (7 Step-by-Step Tabs)
# ----------------------------------------------------------------------
tabs = st.tabs(
    [
        "📰 Agri News & Hub",
        "👤 Farmer Profile",
        "🌦️ Weather Intelligence",
        "🌱 Crop Recommendation",
        "📈 Yield Prediction",
        "💰 Farm Value & Income",
        "📄 Executive Report",
    ]
)

# ======================================================================
# TAB 1: AGRI NEWS & HUB
# ======================================================================
with tabs[0]:
    st.markdown("### 📰 **Indian Agricultural Innovations & Market Hub**")
    st.caption(
        "Real-time news feeds, national scheme announcements, and MSP market updates."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="glass-card">
                <h4 style="color:#34d399;">🏛️ PM-Kisan Samman Nidhi</h4>
                <p style="font-size:0.9rem; color:#cbd5e1;">17th Installment released. ₹20,000+ Crores transferred to 9.2 Crore farmers across India. Complete e-KYC on the PM-Kisan portal to receive direct benefits.</p>
                <span style="color:#fbbf24; font-size:0.8rem; font-weight:700;">UPDATE · GOVT SCHEME</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="glass-card">
                <h4 style="color:#38bdf8;">🚁 Kisan Drone Subsidies</h4>
                <p style="font-size:0.9rem; color:#cbd5e1;">Government offering up to 80% financial assistance for Custom Hiring Centers (CHCs) &amp; FPOs to acquire agricultural drones for precise pesticide spraying.</p>
                <span style="color:#34d399; font-size:0.8rem; font-weight:700;">AGTECH · INNOVATION</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="glass-card">
                <h4 style="color:#a855f7;">🌾 MSP Price Boost</h4>
                <p style="font-size:0.9rem; color:#cbd5e1;">Cabinet hikes Minimum Support Price (MSP) for Kharif crops. Paddy, pulses, and oilseeds see an average 5.8% price increase for guaranteed procurement.</p>
                <span style="color:#38bdf8; font-size:0.8rem; font-weight:700;">MARKET · PROCUREMENT</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### 📊 **Current MSP Price Indicators (₹ / Quintal)**")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Paddy (Common)", "₹ 2,300", "+5.4%")
    mc2.metric("Wheat", "₹ 2,275", "+7.0%")
    mc3.metric("Cotton (Long Staple)", "₹ 7,521", "+7.4%")
    mc4.metric("Soybean (Yellow)", "₹ 4,892", "+6.3%")

# ======================================================================
# TAB 2: FARMER PROFILE
# ======================================================================
with tabs[1]:
    st.markdown("### 👤 **Farmer Baseline Profile**")
    st.caption(
        "Enter farmer information once. It automatically populates all ML models across the tabs."
    )

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            farmer_id = st.text_input(
                "Farmer ID",
                value=st.session_state.profile.get("farmer_id", "F-0001"),
            )
            age = st.number_input(
                "Age (years)",
                18,
                90,
                int(st.session_state.profile.get("age", 35)),
            )
            education = st.selectbox(
                "Education Level",
                ["None", "Primary", "Secondary", "Graduate", "Postgraduate"],
                index=["None", "Primary", "Secondary", "Graduate", "Postgraduate"].index(
                    st.session_state.profile.get(
                        "education_level", "Secondary"
                    )
                ),
            )
            region = st.text_input(
                "Region / State",
                value=st.session_state.profile.get("region", "Punjab"),
            )

        with col2:
            total_land = st.number_input(
                "Total Agricultural Land (Hectares)",
                0.1,
                500.0,
                float(st.session_state.profile.get("total_land_ha", 5.0)),
            )
            current_crop = st.text_input(
                "Current Primary Crop",
                value=st.session_state.profile.get("current_crop", "Wheat"),
            )
            non_agri_income = st.number_input(
                "Non-Agricultural Income (₹/month)",
                0,
                500000,
                int(st.session_state.profile.get("non_agri_income", 12000)),
            )
            distance_to_market = st.number_input(
                "Distance to Nearest Mandi/Market (km)",
                0.0,
                500.0,
                float(
                    st.session_state.profile.get(
                        "distance_to_market_km", 15.0
                    )
                ),
            )

        if st.button(
            "💾 Save Farmer Profile", type="primary", use_container_width=True
        ):
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
            st.success("✅ Profile Saved! Ready for analysis across all tabs.")

# ======================================================================
# TAB 3: WEATHER DASHBOARD
# ======================================================================
with tabs[2]:
    st.markdown("### 🌦️ **Weather Intelligence & Local Advisory**")
    p = st.session_state.profile

    city = st.text_input(
        "Enter City / District", value=p.get("region", "Punjab")
    )

    if st.button("🌦️ Get Weather Intelligence", type="primary"):
        API_KEY = "fc99d46057bc6a25799d7a0577685b63"
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={API_KEY}&units=metric"

        try:
            res = requests.get(url)
            if res.status_code == 200:
                w = res.json()
                temp = w["main"]["temp"]
                hum = w["main"]["humidity"]
                wind = w["wind"]["speed"]
                press = w["main"]["pressure"]
                desc = w["weather"][0]["description"].title()

                st.session_state.weather = {
                    "temp": temp,
                    "humidity": hum,
                    "wind": wind,
                    "desc": desc,
                }

                st.success(f"Live Weather Overview for **{w['name']}**")
                wc1, wc2, wc3, wc4 = st.columns(4)
                wc1.metric("🌡 Temperature", f"{temp} °C")
                wc2.metric("💧 Humidity", f"{hum}%")
                wc3.metric("🌬 Wind Speed", f"{wind} m/s")
                wc4.metric("🧭 Pressure", f"{press} hPa")

                st.markdown("---")
                st.markdown("#### 💡 **AI Precision Advisory**")
                if temp > 35:
                    st.warning(
                        "⚠️ **High Heat Stress:** Increase irrigation frequency during early morning or evening hours."
                    )
                elif hum > 85:
                    st.warning(
                        "⚠️ **High Fungal Risk:** High humidity detected. Inspect crop leaves for fungal spores or rust."
                    )
                else:
                    st.success(
                        "✅ **Optimal Conditions:** Favorable weather for field spraying, weeding, and normal operations."
                    )
            else:
                st.error("Unable to retrieve weather for this location.")
        except Exception as e:
            st.error(f"Weather API Error: {e}")

# ======================================================================
# TAB 4: CROP RECOMMENDATION
# ======================================================================
with tabs[3]:
    st.markdown("### 🌱 **AI-Powered Crop Recommendation**")
    st.caption(
        "Predict the most suitable crop based on NPK soil chemistry and local climate conditions."
    )

    col1, col2 = st.columns(2)
    with col1:
        N = st.number_input("Nitrogen (N)", 0, 200, 90)
        P = st.number_input("Phosphorus (P)", 0, 200, 42)
        K = st.number_input("Potassium (K)", 0, 250, 43)
        ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)
    with col2:
        temp = st.number_input("Temperature (°C)", 0.0, 50.0, 27.0)
        hum = st.number_input("Humidity (%)", 0.0, 100.0, 80.0)
        rain = st.number_input("Rainfall (mm)", 0.0, 3000.0, 800.0)

    if st.button(
        "🌱 Generate Crop Recommendation",
        type="primary",
        use_container_width=True,
    ):
        if ml_models["crop"]:
            try:
                npk = N + P + K
                feats = np.array(
                    [[
                        N,
                        P,
                        K,
                        temp,
                        hum,
                        ph,
                        rain,
                        npk,
                        N / npk if npk else 0,
                        P / npk if npk else 0,
                        K / npk if npk else 0,
                        int(rain < 100),
                        int(100 <= rain < 200),
                        int(rain >= 200),
                        int(temp < 20),
                        int(20 <= temp < 30),
                        int(temp >= 30),
                    ]]
                )
                pred = ml_models["crop"].predict(feats)
                crop_name = ml_models["crop_label"].inverse_transform(pred)[0]
                conf = float(
                    np.max(ml_models["crop"].predict_proba(feats)) * 100
                )

                st.session_state.recommended_crop = crop_name
                st.session_state.crop_confidence = conf

                st.success(f"🌾 Recommended Crop: **{crop_name.upper()}**")
                st.progress(int(conf))
                st.metric("Model Prediction Confidence", f"{conf:.2f}%")
            except Exception as e:
                st.error(f"Crop Model Error: {e}")
        else:
            st.info("Demo Mode: Recommended Crop -> **WHEAT** (92.5% confidence)")

# ======================================================================
# TAB 5: YIELD PREDICTION
# ======================================================================
with tabs[4]:
    st.markdown("### 📈 **AI Crop Yield Prediction**")
    st.caption("Forecast expected yield (Tonnes / Hectare) and total farm output.")

    l, r = st.columns(2)
    with l:
        crop_in = st.selectbox(
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
            ],
        )
        season_in = st.selectbox(
            "Season",
            ["Kharif", "Rabi", "Summer", "Whole Year", "Winter", "Autumn"],
        )
        state_in = st.text_input("State", "Punjab")
        year_in = st.number_input("Crop Year", 1997, 2035, 2026)
    with r:
        area_in = st.number_input(
            "Area (Hectares)",
            value=float(p.get("total_land_ha", 5.0)),
        )
        prod_in = st.number_input("Historic Production (Tonnes)", value=20.0)
        rain_in = st.number_input("Annual Rainfall (mm)", value=800.0)
        fert_in = st.number_input("Fertilizer Input (kg)", value=450.0)
        pest_in = st.number_input("Pesticide Input (kg)", value=8.0)

    if st.button("🚀 Predict Crop Yield", type="primary", use_container_width=True):
        if ml_models["yield"]:
            try:
                c_enc = ml_models["crop_enc"].transform([crop_in.strip()])[0]
                s_classes = [
                    x.strip() for x in ml_models["season_enc"].classes_
                ]
                s_enc = s_classes.index(season_in.strip())
                st_classes = [
                    x.strip() for x in ml_models["state_enc"].classes_
                ]
                st_enc = st_classes.index(state_in.strip())

                inp = pd.DataFrame(
                    {
                        "Crop": [c_enc],
                        "Crop_Year": [year_in],
                        "Season": [s_enc],
                        "State": [st_enc],
                        "Area": [area_in],
                        "Production": [prod_in],
                        "Annual_Rainfall": [rain_in],
                        "Fertilizer": [fert_in],
                        "Pesticide": [pest_in],
                    }
                )
                y_pred = float(ml_models["yield"].predict(inp)[0])
                tot_out = y_pred * area_in

                st.session_state.predicted_yield = y_pred
                st.session_state.total_output = tot_out

                st.success("✅ Yield Forecast Completed!")
                ym1, ym2 = st.columns(2)
                ym1.metric("Predicted Yield", f"{y_pred:.2f} t/ha")
                ym2.metric("Total Estimated Output", f"{tot_out:.2f} Tonnes")
            except Exception as e:
                st.error(f"Yield Model Error: {e}")
        else:
            st.info("Demo Mode: Predicted Yield -> **4.20 t/ha** | Total Output -> **21.00 Tonnes**")

# ======================================================================
# TAB 6: FARM VALUE & INCOME
# ======================================================================
with tabs[5]:
    st.markdown("### 💰 **XGBoost Farm Value & Valuation Estimation**")
    st.caption("Estimate overall farm economic valuation using NSS-trained ML.")

    p = st.session_state.profile
    if not p:
        st.warning("⚠️ Please complete the Farmer Profile first.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            irr_area = st.number_input(
                "Irrigated Area (ha)",
                0.0,
                float(p.get("total_land_ha", 5.0)),
                3.0,
            )
        with col2:
            loan_amt = st.number_input(
                "Agricultural Loan Amount (₹)", 0.0, 10000000.0, 0.0
            )

        if st.button(
            "🌾 Calculate Farm Value", type="primary", use_container_width=True
        ):
            if ml_models["farm_val"]:
                try:
                    land = float(p.get("total_land_ha", 5.0))
                    land_s = max(land, 0.001)
                    inp = pd.DataFrame(
                        [{
                            "state": p.get("region", "Punjab"),
                            "district": 1,
                            "gender": "Male",
                            "education": p.get("education_level", "Secondary"),
                            "agri_training": 2,
                            "Principal_Activity": "Cultivation",
                            "agricultural_land": "Yes",
                            "major_crop": "Cereals",
                            "irrigated": "Yes" if irr_area > 0 else "No",
                            "irrigation_source": "Ground Water",
                            "bank_account": "Yes",
                            "kcc": "Yes" if loan_amt > 0 else "No",
                            "msp_awareness": "Yes",
                            "sold_at_msp": "No",
                            "technical_advice": "No",
                            "advice_adopted": "No",
                            "crop_insured": "No",
                            "soil_health_card": "No",
                            "followed_soil_health_card": "No",
                            "farmer_organization": "No",
                            "age": float(p.get("age", 35)),
                            "household_size": 5.0,
                            "land_area": land,
                            "irrigated_area": irr_area,
                            "crops_grown": 2.0,
                            "wages_salary": 0.0,
                            "land_rent_income": 0.0,
                            "nonfarm_net_income": float(
                                p.get("non_agri_income", 12000)
                            )
                            * 12,
                            "loan_amount": loan_amt,
                            "interest_rate": 7.0,
                            "irrigation_intensity": irr_area / land_s,
                            "loan_per_ha": loan_amt / land_s,
                            "crops_per_ha": 2.0 / land_s,
                            "household_density": 5.0 / land_s,
                            "irrigated_share": irr_area / land_s,
                            "interest_burden": loan_amt * 0.07,
                            "loan_land_interaction": loan_amt * land,
                            "land_size_squared": land**2,
                            "msp_sell_rate": 0.0,
                        }]
                    )
                    val = max(0.0, float(ml_models["farm_val"].predict(inp)[0]))
                    st.session_state.farm_value = val
                    st.success("Valuation Estimated!")
                    st.metric("Estimated Farm Value", f"₹ {val:,.0f}")
                except Exception as e:
                    st.error(f"XGBoost Model Error: {e}")
            else:
                st.info("Demo Valuation -> **₹ 1,05,764**")

# ======================================================================
# TAB 7: EXECUTIVE REPORT
# ======================================================================
with tabs[6]:
    st.markdown("### 📄 **Farmer Executive Summary Report**")
    p = st.session_state.profile

    if not p:
        st.info("Fill in the Farmer Profile tab to generate a report.")
    else:
        rep_text = f"""==================================================
AGRIVISION AI - EXECUTIVE FARMER REPORT
==================================================
Farmer ID       : {p.get('farmer_id', 'N/A')}
State / Region  : {p.get('region', 'N/A')}
Land Area       : {p.get('total_land_ha', 'N/A')} Hectares
Current Crop    : {p.get('current_crop', 'N/A')}
Monthly Income  : ₹ {p.get('non_agri_income', 'N/A')}

PREDICTIVE ANALYTICS SUMMARY:
--------------------------------------------------
Recommended Crop : {st.session_state.get('recommended_crop', 'Not Run')}
Crop Confidence  : {st.session_state.get('crop_confidence', 0.0):.2f}%
Predicted Yield  : {st.session_state.get('predicted_yield', 0.0):.2f} t/ha
Total Output     : {st.session_state.get('total_output', 0.0):.2f} Tonnes
Est. Farm Value  : ₹ {st.session_state.get('farm_value', 0.0):,.0f}
==================================================
Report Generated via AgriVision AI System
"""
        st.text_area("Full Summary Report Preview", rep_text, height=280)
        st.download_button(
            "📥 Download Executive Report (.txt)",
            rep_text,
            file_name=f"{p.get('farmer_id', 'farmer')}_agrivision_report.txt",
            type="primary",
        )

# ----------------------------------------------------------------------
# Floating Chatbot Popover Logic (Groq + Llama 3)
# ----------------------------------------------------------------------
def ai_assistant_reply(question, context):
    groq_api_key = st.secrets.get("GROQ_API_KEY", None)

    if not groq_api_key:
        return "❌ **Groq API Key Missing!** Add `GROQ_API_KEY` in Streamlit Cloud Secrets."

    prompt = f"""You are AgriVision AI, an expert agricultural advisor for Indian farmers.
Current Farmer Information:
{context}

Question:
{question}

Instructions:
- Give simple, practical farming advice.
- Use bullet points.
- Keep responses concise (under 200 words).
"""
    try:
        client = Groq(api_key=groq_api_key)
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=400,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"❌ Groq Error: {e}"


def render_ai_assistant_bubble():
    AVATARS = {"user": "🧑‍🌾", "assistant": "🌾"}

    with st.popover("🧑‍🌾", use_container_width=False, help="Ask AgriVision AI"):
        st.markdown("**🌾 AgriVision Assistant**")
        st.caption("Ask advice about crops, yields, or farm income.")

        chat_box = st.container(height=280)
        with chat_box:
            for role, msg in st.session_state.chat_history:
                with st.chat_message(role, avatar=AVATARS.get(role)):
                    st.write(msg)

        question = st.chat_input("Ask something...", key="floating_chat_input")
        if question:
            p = st.session_state.get("profile", {})
            context = f"Profile: {p}. Rec Crop: {st.session_state.get('recommended_crop', 'N/A')}. Pred Yield: {st.session_state.get('predicted_yield', 'N/A')}."

            st.session_state.chat_history.append(("user", question))
            reply = ai_assistant_reply(question, context)
            st.session_state.chat_history.append(("assistant", reply))
            st.rerun()


render_ai_assistant_bubble()
