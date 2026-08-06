import streamlit as st

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="AgriVision AI",
    page_icon="🌾",
    layout="wide"
)

# ----------------------------
# HEADER
# ----------------------------
st.title("🌾 AgriVision AI")
st.subheader("AI-Powered Decision Support System for Indian Agriculture")

st.markdown("---")

# ----------------------------
# HERO SECTION
# ----------------------------

left, right = st.columns([2,1])

with left:
    st.header("Welcome to AgriVision AI")

    st.write("""
AgriVision AI is an intelligent platform designed to help Indian farmers make smarter agricultural decisions using Artificial Intelligence and Machine Learning.

Our platform provides:

✅ Income Prediction

✅ Crop Recommendation

✅ Yield Prediction

✅ Weather Intelligence

✅ AI Farm Advisor

✅ Smart Reports
""")

    st.button("🚀 Start Analysis")

with right:

    st.info("""
🌾 Version : 1.0

🧠 AI Powered

📊 Built with Streamlit

🇮🇳 Made for Indian Farmers
""")

st.markdown("---")

# ----------------------------
# MODULES
# ----------------------------

st.header("Available Modules")

c1,c2,c3 = st.columns(3)

with c1:
    st.success("👨 Farmer Registration")

with c2:
    st.success("💰 Income Prediction")

with c3:
    st.success("🌾 Crop Recommendation")

c4,c5,c6 = st.columns(3)

with c4:
    st.success("📈 Yield Prediction")

with c5:
    st.success("🌦 Weather Intelligence")

with c6:
    st.success("🤖 AI Farm Advisor")

st.markdown("---")

st.caption("© 2026 AgriVision AI")

import streamlit as st

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="AgriVision AI",
    page_icon="🌾",
    layout="wide"
)

# ------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------

st.markdown("""
<style>

.main{
    background-color:#f5f8f2;
}

.hero{
    background:linear-gradient(90deg,#1B5E20,#2E7D32);
    padding:45px;
    border-radius:20px;
    color:white;
}

.feature{
    background:white;
    padding:25px;
    border-radius:15px;
    box-shadow:0px 3px 15px rgba(0,0,0,0.08);
    text-align:center;
    height:170px;
}

.metric{
    background:#E8F5E9;
    padding:18px;
    border-radius:15px;
    text-align:center;
}

.footer{
    text-align:center;
    color:grey;
    margin-top:50px;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------

st.sidebar.title("🌾 AgriVision AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "👨‍🌾 Farmer Registration",
        "💰 Income Prediction",
        "🌱 Crop Recommendation",
        "📈 Yield Prediction",
        "🌦 Weather Intelligence",
        "🤖 AI Assistant",
        "📊 Analytics"
    ]
)

# ------------------------------------------------
# HOME PAGE
# ------------------------------------------------

if page=="🏠 Home":

    st.markdown("""
    <div class='hero'>

    <h1>🌾 AgriVision AI</h1>

    <h3>AI Powered Agriculture Intelligence Platform</h3>

    <p>
    Empowering Indian farmers through Artificial Intelligence,
    Machine Learning, Weather Analytics and Smart Recommendations.
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    col1,col2,col3,col4=st.columns(4)

    with col1:
        st.metric("Registered Farmers","1,245")

    with col2:
        st.metric("Predictions","18,532")

    with col3:
        st.metric("Accuracy","94.8%")

    with col4:
        st.metric("Districts","210")

    st.write("")
    st.header("🚀 Platform Modules")

    c1,c2,c3,c4=st.columns(4)

    with c1:
        st.info("👨‍🌾 Farmer Registration")

    with c2:
        st.info("💰 Income Prediction")

    with c3:
        st.info("🌱 Crop Recommendation")

    with c4:
        st.info("📈 Yield Prediction")

    c5,c6,c7,c8=st.columns(4)

    with c5:
        st.info("🌦 Weather Intelligence")

    with c6:
        st.info("🤖 AI Assistant")

    with c7:
        st.info("📊 Analytics")

    with c8:
        st.info("🗺 GIS Mapping")

    st.write("")
    st.write("")

    st.subheader("📌 Vision")

    st.write("""
AgriVision AI is an end-to-end decision support platform for Indian agriculture.

Our objective is to combine Artificial Intelligence, Machine Learning,
Weather Intelligence, Remote Sensing and Government Agricultural Data
to improve farmers' income and productivity.
""")

    st.markdown("""
    <div class='footer'>

    © 2026 AgriVision AI

    </div>
    """, unsafe_allow_html=True)
    