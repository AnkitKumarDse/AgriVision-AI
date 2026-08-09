import base64
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from groq import Groq

# ----------------------------------------------------------------------
# Page Config & Initializations
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="AgriVision AI -- Decision Support System",
    page_icon="🌾",
    layout="wide",
)

if "profile" not in st.session_state:
    st.session_state.profile = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Helper for Base64 Farmer Avatar Icon
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
def load_ml_models():
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


ml_models = load_ml_models()

# ----------------------------------------------------------------------
# Custom Glassmorphism CSS & Dashboard Styling
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; max-width: 1280px; }
    
    /* Hero Header */
    .hero-card {
        background: linear-gradient(135deg, #064e3b 0%, #022c22 60%, #0f172a 100%);
        border: 1px solid #059669;
        border-radius: 18px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }
    .hero-card h1 { font-size: 2.3rem; color: #f0fdf4; margin-bottom: 0.2rem; font-weight: 700; }
    .hero-card p { color: #a7f3d0; font-size: 1.05rem; margin: 0; }
    .badge {
        display: inline-block; background: #065f46; color: #34d399;
        border-radius: 999px; padding: 4px 14px; font-size: 0.78rem; font-weight: 700;
        letter-spacing: 0.05em; margin-bottom: 0.8rem; border: 1px solid #059669;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 2px solid #064e3b; }
    .stTabs [data-baseweb="tab"] {
        background-color: #0f172a; border-radius: 8px 8px 0 0;
        padding: 12px 20px; color: #94a3b8; border: 1px solid #1e293b;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(180deg, #065f46 0%, #022c22 100%) !important;
        color: #34d399 !important; font-weight: 700; border-color: #059669 !important;
    }

    /* Streamlit Containers & Cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #0f172a; border: 1px solid #1e293b; border-radius: 14px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    div[data-testid="stMetric"] {
        background-color: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 14px;
    }

    /* Floating Chatbot Bubble */
    div[data-testid="stPopover"] {
        position: fixed !important; top: 62%; left: 24px; z-index: 9999;
        animation: av-float 3s ease-in-out infinite;
    }
    @keyframes av-float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-8px); }
    }
    div[data-testid="stPopover"] button {
        width: 66px; height: 66px; border-radius: 50%;
        background: linear-gradient(135deg, #10b981, #059669);
        border: 3px solid #f0fdf4; box-shadow: 0 8px 24px rgba(16, 185, 129, 0.4);
    }
    </style>

    <div class="hero-card">
        <div class="badge">NATIONAL AGRICULTURAL INTELLIGENCE PLATFORM</div>
        <h1>🌾 AgriVision AI</h1>
        <p>End-to-end decision support for Indian farmers -- news, crop suitability, yield forecasting, and farm evaluation.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Application Navigation (7 Logical Tabs)
# ----------------------------------------------------------------------
tabs = st.tabs(
    [
        "📰 Agri-Hub & News",
        "👤 Farmer Profile",
        "🌦️ Weather Dashboard",
        "🌱 Crop Recommendation",
        "📈 Yield Prediction",
        "💰 Farm Value & Income",
        "📄 Executive Report",
    ]
)

# ----------------------------------------------------------------------
# TAB 1: AGRI-HUB & NEWS
# ----------------------------------------------------------------------
with tabs[0]:
    with st.container(border=True):
        st.header("📰 Indian Agriculture & Innovation News")
        st.caption(
            "Live national policy updates, market schemes, and agtech developments."
        )

        n1, n2 = st.columns(2)
        with n1:
            st.subheader("🏛️ Schemes & Policy Updates")
            st.info(
                "**PM-Kisan Samman Nidhi 17th Installment Released**\n\n"
                "Over ₹20,000 crore transferred directly to 9.2 crore farmers. Ensure e-KYC is completed on PM-Kisan portal."
            )
            st.success(
                "**Digital Agriculture Mission Sanctioned**\n\n"
                "Union Cabinet approved ₹2,817 crore for Digital Public Infrastructure (DPI) in agriculture including Agristack & Krishi-DSS."
            )
        with n2:
            st.subheader("💡 Innovation & Market Trends")
            st.warning(
                "**Drone Subsidies for Farmers & FPOs**\n\n"
                "Government offering up to 80% financial assistance for purchasing agricultural drones for precision spraying."
            )
            st.success(
                "**MSP Updates for Kharif Season**\n\n"
                "Cabinet increases Minimum Support Price (MSP) for 14 Kharif crops including Paddy, Pulses, and Oilseeds."
            )

# ----------------------------------------------------------------------
# TAB 2: FARMER PROFILE
# ----------------------------------------------------------------------
with tabs[1]:
    with st.container(border=True):
        st.header("👤 Farmer Profile")
        st.caption("Shared state across all models and analysis tabs.")

        c1, c2 = st.columns(2)
        with c1:
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
                "State / Region",
                value=st.session_state.profile.get("region", "Punjab"),
            )

        with c2:
            total_land = st.number_input(
                "Total Land (Hectares)",
                0.1,
                500.0,
                float(st.session_state.profile.get("total_land_ha", 5.0)),
            )
            current_crop = st.text_input(
                "Current Primary Crop",
                value=st.session_state.profile.get("current_crop", "Wheat"),
            )
            non_agri_income = st.number_input(
                "Non-Agri Income (₹/month)",
                0,
                500000,
                int(st.session_state.profile.get("non_agri_income", 12000)),
            )
            distance_market = st.number_input(
                "Distance to Market (km)",
                0.0,
                500.0,
                float(
                    st.session_state.profile.get(
                        "distance_to_market_km", 15.0
                    )
                ),
            )

        if st.button("💾 Save Profile", type="primary", use_container_width=True):
            st.session_state.profile = {
                "farmer_id": farmer_id,
                "age": age,
                "education_level": education,
                "region": region,
                "total_land_ha": total_land,
                "current_crop": current_crop,
                "non_agri_income": non_agri_income,
                "distance_to_market_km": distance_market,
            }
            st.success("Profile saved successfully!")

# ----------------------------------------------------------------------
# TAB 3: WEATHER DASHBOARD
# ----------------------------------------------------------------------
with tabs[2]:
    with st.container(border=True):
        st.header("🌦️ Weather Intelligence Dashboard")
        city = st.text_input(
            "Enter City",
            value=st.session_state.profile.get("region", "Patna"),
        )

        if st.button("Fetch Weather", type="primary"):
            API_KEY = "fc99d46057bc6a25799d7a0577685b63"
            url = f"https://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={API_KEY}&units=metric"
            res = requests.get(url)

            if res.status_code == 200:
                w = res.json()
                st.session_state.weather = {
                    "temp": w["main"]["temp"],
                    "humidity": w["main"]["humidity"],
                    "wind": w["wind"]["speed"],
                    "desc": w["weather"][0]["description"],
                }
                st.success(f"Live Weather for {city}")
                m1, m2, m3 = st.columns(3)
                m1.metric("Temperature", f"{w['main']['temp']} °C")
                m2.metric("Humidity", f"{w['main']['humidity']}%")
                m3.metric("Wind Speed", f"{w['wind']['speed']} m/s")
            else:
                st.error("City not found or OpenWeather connection error.")

# ----------------------------------------------------------------------
# TAB 4: CROP RECOMMENDATION
# ----------------------------------------------------------------------
with tabs[3]:
    with st.container(border=True):
        st.header("🌱 AI Crop Recommendation")
        c1, c2 = st.columns(2)
        with c1:
            N = st.number_input("Nitrogen (N)", 0, 200, 90)
            P = st.number_input("Phosphorus (P)", 0, 200, 42)
            K = st.number_input("Potassium (K)", 0, 250, 43)
            ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)
        with c2:
            temp = st.number_input("Temperature (°C)", 0.0, 50.0, 27.0)
            hum = st.number_input("Humidity (%)", 0.0, 100.0, 80.0)
            rain = st.number_input("Rainfall (mm)", 0.0, 3000.0, 800.0)

        if st.button("🌱 Recommend Optimal Crop", type="primary", use_container_width=True):
            if ml_models["crop"]:
                npk = N + P + K
                feats = np.array(
                    [[
                        N, P, K, temp, hum, ph, rain, npk,
                        N / npk if npk else 0, P / npk if npk else 0, K / npk if npk else 0,
                        int(rain < 100), int(100 <= rain < 200), int(rain >= 200),
                        int(temp < 20), int(20 <= temp < 30), int(temp >= 30),
                    ]]
                )
                pred = ml_models["crop"].predict(feats)
                crop_name = ml_models["crop_label"].inverse_transform(pred)[0]
                conf = float(np.max(ml_models["crop"].predict_proba(feats)) * 100)

                st.session_state.recommended_crop = crop_name
                st.session_state.crop_confidence = conf

                st.success(f"Recommended Crop: **{crop_name.upper()}**")
                st.metric("Model Confidence", f"{conf:.2f}%")
            else:
                st.warning("`crop_model.pkl` not found. Demo mode active.")

# ----------------------------------------------------------------------
# TAB 5: YIELD PREDICTION
# ----------------------------------------------------------------------
with tabs[4]:
    with st.container(border=True):
        st.header("📈 AI Yield Prediction")
        l, r = st.columns(2)
        with l:
            crop_in = st.selectbox(
                "Crop",
                ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane", "Potato"],
            )
            season_in = st.selectbox(
                "Season",
                ["Kharif", "Rabi", "Summer", "Whole Year"],
            )
            state_in = st.text_input("State", "Punjab")
            year_in = st.number_input("Year", 1997, 2035, 2026)
        with r:
            area_in = st.number_input("Area (Hectares)", value=5.0)
            prod_in = st.number_input("Production (Tonnes)", value=20.0)
            rain_in = st.number_input("Rainfall (mm)", value=800.0)
            fert_in = st.number_input("Fertilizer", value=450.0)
            pest_in = st.number_input("Pesticide", value=8.0)

        if st.button("🚀 Predict Yield", type="primary", use_container_width=True):
            if ml_models["yield"]:
                try:
                    c_enc = ml_models["crop_enc"].transform([crop_in.strip()])[0]
                    s_classes = [x.strip() for x in ml_models["season_enc"].classes_]
                    s_enc = s_classes.index(season_in.strip())
                    st_classes = [x.strip() for x in ml_models["state_enc"].classes_]
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

                    st.success(f"Predicted Yield: **{y_pred:.2f} t/ha**")
                    st.metric("Total Estimated Output", f"{tot_out:.2f} Tonnes")
                except Exception as e:
                    st.error(f"Yield calculation error: {e}")

# ----------------------------------------------------------------------
# TAB 6: FARM VALUE & INCOME
# ----------------------------------------------------------------------
with tabs[5]:
    with st.container(border=True):
        st.header("💰 Farm Value Estimation")
        p = st.session_state.profile
        if not p:
            st.warning("Please save your Farmer Profile first.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                irr_area = st.number_input(
                    "Irrigated Area (ha)", 0.0, float(p.get("total_land_ha", 5.0)), 3.0
                )
            with col2:
                loan_amt = st.number_input("Loan Amount (₹)", 0.0, 10000000.0, 0.0)

            if st.button("🌾 Estimate Farm Value", type="primary", use_container_width=True):
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
                                "nonfarm_net_income": float(p.get("non_agri_income", 12000)) * 12,
                                "loan_amount": loan_amt,
                                "interest_rate": 7.0,
                                "irrigation_intensity": irr_area / land_s,
                                "loan_per_ha": loan_amt / land_s,
                                "crops_per_ha": 2.0 / land_s,
                                "household_density": 5.0 / land_s,
                                "irrigated_share": irr_area / land_s,
                                "interest_burden": loan_amt * 0.07,
                                "loan_land_interaction": loan_amt * land,
                                "land_size_squared": land ** 2,
                                "msp_sell_rate": 0.0,
                            }]
                        )
                        val = max(0.0, float(ml_models["farm_val"].predict(inp)[0]))
                        st.session_state.farm_value = val
                        st.success(f"Estimated Value: ₹ {val:,.0f}")
                    except Exception as e:
                        st.error(f"XGBoost calculation error: {e}")

# ----------------------------------------------------------------------
# TAB 7: EXECUTIVE REPORT
# ----------------------------------------------------------------------
with tabs[6]:
    with st.container(border=True):
        st.header("📄 Executive Summary Report")
        p = st.session_state.profile
        if not p:
            st.info("Complete farmer profile to view report.")
        else:
            rep = f"""==================================================
AGRIVISION AI - FARMER EXECUTIVE REPORT
==================================================
Farmer ID : {p.get('farmer_id')}
State     : {p.get('region')}
Land      : {p.get('total_land_ha')} ha

ANALYSIS RESULTS:
--------------------------------------------------
Recommended Crop : {st.session_state.get('recommended_crop', 'N/A')}
Predicted Yield  : {st.session_state.get('predicted_yield', 0.0):.2f} t/ha
Estimated Output : {st.session_state.get('total_output', 0.0):.2f} Tonnes
Est. Farm Value  : ₹ {st.session_state.get('farm_value', 0.0):,.0f}
=================================================="""

            st.text_area("Report Summary", rep, height=220)
            st.download_button(
                "📥 Download Summary (.txt)",
                rep,
                file_name=f"Report_{p.get('farmer_id')}.txt",
            )

# ----------------------------------------------------------------------
# Floating Chatbot Bubble Logic
# ----------------------------------------------------------------------
def ai_assistant_reply(question, context):
    key = st.secrets.get("GROQ_API_KEY", None)
    if not key:
        return "❌ Missing GROQ_API_KEY in Streamlit Secrets."
    try:
        c = Groq(api_key=key)
        res = c.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": f"You are AgriVision AI. Answer in simple bullet points under 200 words.\nContext:\n{context}\nQuestion:\n{question}",
                }
            ],
            temperature=0.6,
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"❌ Groq Error: {e}"


def render_ai_assistant_bubble():
    with st.popover("🧑‍🌾", help="Ask AgriVision AI"):
        st.markdown("**🌾 AgriVision Assistant**")
        chat = st.container(height=260)
        with chat:
            for role, m in st.session_state.chat_history:
                st.chat_message(role).write(m)

        q = st.chat_input("Ask advice...")
        if q:
            p = st.session_state.get("profile", {})
            ctx = f"Farmer Profile: {p}. Rec Crop: {st.session_state.get('recommended_crop','N/A')}. Pred Yield: {st.session_state.get('predicted_yield','N/A')}."
            st.session_state.chat_history.append(("user", q))
            ans = ai_assistant_reply(q, ctx)
            st.session_state.chat_history.append(("assistant", ans))
            st.rerun()


render_ai_assistant_bubble()
