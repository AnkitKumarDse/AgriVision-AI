import streamlit as st

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AgriVision AI",
    page_icon="🌾",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>

.hero{
background:#1b5e20;
padding:40px;
border-radius:15px;
color:white;
}

.footer{
text-align:center;
color:gray;
padding:20px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR
# -----------------------------
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

# =====================================================
# HOME PAGE
# =====================================================

if page == "🏠 Home":

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

    col1,col2,col3,col4 = st.columns(4)

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

    c1,c2,c3,c4 = st.columns(4)

    with c1:
        st.info("👨‍🌾 Farmer Registration")

    with c2:
        st.info("💰 Income Prediction")

    with c3:
        st.info("🌱 Crop Recommendation")

    with c4:
        st.info("📈 Yield Prediction")

    c5,c6,c7,c8 = st.columns(4)

    with c5:
        st.info("🌦 Weather Intelligence")

    with c6:
        st.info("🤖 AI Assistant")

    with c7:
        st.info("📊 Analytics")

    with c8:
        st.info("🗺 GIS Mapping")

    st.write("")
    st.subheader("📌 Vision")

    st.write("""
AgriVision AI is an end-to-end decision support platform for Indian agriculture.

Our objective is to combine Artificial Intelligence, Machine Learning,
Weather Intelligence, Remote Sensing and Government Agricultural Data
to improve farmers' income and productivity.
""")

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""
<div class='footer'>
© 2026 AgriVision AI
</div>
""", unsafe_allow_html=True)

# =====================================================
# OTHER PAGES
# =====================================================

elif page == "👨‍🌾 Farmer Registration":

    st.title("👨‍🌾 Farmer Registration")

    name = st.text_input("Farmer Name")

    age = st.number_input("Age",18,100)

    state = st.text_input("State")

    district = st.text_input("District")

    land = st.number_input("Land Size (Acres)",0.0)

    crop = st.text_input("Primary Crop")

    if st.button("Register Farmer"):
        st.success("Farmer Registered Successfully ✅")

elif page == "💰 Income Prediction":

    st.title("💰 Income Prediction")

    st.info("Income Prediction Model Coming Soon")

elif page == "🌱 Crop Recommendation":

    st.title("🌱 Crop Recommendation")

    st.info("Crop Recommendation Module Coming Soon")

elif page == "📈 Yield Prediction":

    st.title("📈 Yield Prediction")

    st.info("Yield Prediction Module Coming Soon")

elif page == "🌦 Weather Intelligence":

    st.title("🌦 Weather Intelligence")

    st.info("Weather Intelligence Module Coming Soon")

elif page == "🤖 AI Assistant":

    st.title("🤖 AI Assistant")

    question = st.text_area("Ask your farming question")

    if st.button("Ask AI"):
        st.success("AI Response will appear here.")

elif page == "📊 Analytics":

    st.title("📊 Analytics Dashboard")

    col1,col2,col3 = st.columns(3)

    col1.metric("Farmers",1245)

    col2.metric("Predictions",18532)

    col3.metric("Accuracy","94.8%")

    st.bar_chart({
        "Predictions":[120,180,240,310,450]
    })