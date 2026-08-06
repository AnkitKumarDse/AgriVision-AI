"""
AgriVision AI -- Streamlit frontend shell
Run with:  streamlit run app.py

This is the FRONTEND ONLY. All predictions currently run in DEMO MODE
(see utils.py) using transparent placeholder formulas so every screen is
fully clickable while the model team finishes training. Once a teammate
drops a .pkl into /models matching the contract documented at the top of
utils.py, the relevant tab switches from demo to real predictions
automatically -- no changes needed here.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import ai_assistant_reply, model_status, predict_income, predict_yield, recommend_crops

st.set_page_config(page_title="AgriVision AI", page_icon="🌾", layout="wide")

# ----------------------------------------------------------------------
# Custom styling -- hero banner, card containers, tab spacing.
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
# Session state -- this is how tabs share the farmer's profile/data
# ----------------------------------------------------------------------
if "profile" not in st.session_state:
    st.session_state.profile = {}
if "income_result" not in st.session_state:
    st.session_state.income_result = None
if "yield_result" not in st.session_state:
    st.session_state.yield_result = None
if "crop_result" not in st.session_state:
    st.session_state.crop_result = None

# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.title("🌾 AgriVision AI")
    st.caption("AI decision support for Indian agriculture")
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
        "🤖 AI Assistant",
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
            crop_yield = st.number_input("Crop Yield per Hectare (tons)", 0.0, 50.0, 10.2)
            rainfall = st.number_input("Rainfall (mm, seasonal avg)", 0.0, 3000.0, 800.0)
        with col2:
            land_override = st.number_input(
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
            st.session_state.income_result = predict_income(inputs)

        result = st.session_state.income_result
        if result:
            if result["demo"]:
                st.warning("Showing a DEMO estimate (formula-based) -- not a real model prediction yet.")
            c1, c2 = st.columns(2)
            c1.metric("Predicted Income (monthly)", f"₹ {result['value']:,.0f}")
            c2.metric("Model Confidence", f"{result['confidence']*100:.0f}%")

# ----------------------------------------------------------------------
# 3. Crop Recommendation
# ----------------------------------------------------------------------
with tabs[2]:
    with st.container(border=True):
        st.header("AI-Powered Crop Recommendation")
        p = st.session_state.profile
        col1, col2 = st.columns(2)
        with col1:
            soil_type = st.selectbox("Soil Type", ["Unknown", "Alluvial", "Black", "Red", "Laterite", "Sandy"])
            rainfall2 = st.number_input("Rainfall (mm)", 0.0, 3000.0, 800.0, key="rain2")
        with col2:
            temperature = st.number_input("Avg Temperature (°C)", 0.0, 50.0, 27.0)

        if st.button("Generate Crop Suggestions", type="primary"):
            inputs = {
                "region": p.get("region", "Punjab"),
                "soil_type": soil_type,
                "rainfall_mm": rainfall2,
                "temperature_c": temperature,
                "current_crop": p.get("current_crop", "Unknown"),
            }
            st.session_state.crop_result = recommend_crops(inputs)

        result = st.session_state.crop_result
        if result:
            if result["demo"]:
                st.warning("Showing DEMO suggestions (randomised) -- not a real model prediction yet.")
            for crop, score in result["crops"]:
                st.write(f"**{crop}** -- suitability score {score:.2f}")
                st.progress(min(max(score, 0.0), 1.0))

# ----------------------------------------------------------------------
# 4. Yield Prediction
# ----------------------------------------------------------------------
with tabs[3]:
    with st.container(border=True):
        st.header("Yield Prediction")
        col1, col2 = st.columns(2)
        with col1:
            crop_type = st.text_input("Crop Type", value=st.session_state.profile.get("current_crop", "Wheat"))
            land3 = st.number_input("Total Land (hectares)", 0.0, 500.0, 5.0, key="land3")
        with col2:
            rainfall3 = st.number_input("Rainfall (mm)", 0.0, 3000.0, 800.0, key="rain3")
            temp3 = st.number_input("Temperature (°C)", 0.0, 50.0, 27.0, key="temp3")
        input_costs = st.number_input("Input Costs (₹)", 0, 1000000, 20000)

        if st.button("Predict Yield", type="primary"):
            inputs = {
                "crop_type": crop_type,
                "total_land_ha": land3,
                "rainfall_mm": rainfall3,
                "temperature_c": temp3,
                "input_costs": input_costs,
            }
            st.session_state.yield_result = predict_yield(inputs)

        result = st.session_state.yield_result
        if result:
            if result["demo"]:
                st.warning("Showing a DEMO estimate -- not a real model prediction yet.")
            st.metric("Predicted Yield", f"{result['value']} tons/hectare")

# ----------------------------------------------------------------------
# 5. Weather Dashboard
# ----------------------------------------------------------------------
with tabs[4]:
    with st.container(border=True):
        st.header("Weather Dashboard")
        st.info(
            "Placeholder. Wire this up to a real weather API (e.g. Open-Meteo, IMD, or "
            "OpenWeatherMap) once you decide on a source -- pass the farmer's region from "
            "the profile tab as the query location."
        )
        demo_days = pd.date_range("2026-08-06", periods=7)
        demo_weather = pd.DataFrame(
            {
                "Date": demo_days,
                "Temp (°C)": [30, 31, 29, 28, 32, 33, 30],
                "Rainfall (mm)": [5, 0, 12, 20, 0, 0, 8],
            }
        )
        st.dataframe(demo_weather, use_container_width=True)
        st.line_chart(demo_weather.set_index("Date")[["Temp (°C)"]])
        st.bar_chart(demo_weather.set_index("Date")[["Rainfall (mm)"]])

# ----------------------------------------------------------------------
# 6. AI Assistant
# ----------------------------------------------------------------------
with tabs[5]:
    with st.container(border=True):
        st.header("AI Assistant")
        st.caption("Ask about your predicted income, crop choices, or general farming advice.")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for role, msg in st.session_state.chat_history:
            with st.chat_message(role):
                st.write(msg)

        question = st.chat_input("Ask something...")
        if question:
            st.session_state.chat_history.append(("user", question))
            p = st.session_state.profile
            context = {"summary": f"{p.get('current_crop', 'a crop')} farmer with {p.get('total_land_ha', '?')} ha in {p.get('region', 'India')}"}
            reply = ai_assistant_reply(question, context)
            st.session_state.chat_history.append(("assistant", reply))
            st.rerun()

# ----------------------------------------------------------------------
# 7. Reports
# ----------------------------------------------------------------------
with tabs[6]:
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
# 8. Final Dashboard
# ----------------------------------------------------------------------
with tabs[7]:
    with st.container(border=True):
        st.header("Final Dashboard")
        p = st.session_state.profile
        if not p:
            st.info("Fill in the Farmer Profile tab first.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Land Holding", f"{p.get('total_land_ha')} ha")
            c2.metric(
                "Predicted Income",
                f"₹{st.session_state.income_result['value']:,.0f}" if st.session_state.income_result else "Not run yet",
            )
            c3.metric(
                "Predicted Yield",
                f"{st.session_state.yield_result['value']} t/ha" if st.session_state.yield_result else "Not run yet",
            )

            if st.session_state.income_result:
                # Simple simulated seasonal trend around the predicted point, same idea
                # as the reference demo's "Simulated Income Trend Over Seasons" chart.
                base = st.session_state.income_result["value"]
                seasons = ["Season 1", "Season 2", "Season 3", "Season 4"]
                values = [base * f for f in (0.85, 1.05, 0.95, 1.0)]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=seasons, y=values, fill="tozeroy", line=dict(color="#2ecc71")))
                fig.update_layout(title="Simulated Income Trend Over Seasons", yaxis_title="₹", height=350)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("Run Income Estimation to see the seasonal trend chart here.")