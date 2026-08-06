

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
