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
