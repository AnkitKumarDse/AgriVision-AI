# ------------------------------------------------
# HOME PAGE
# ------------------------------------------------

if page == "🏠 Home":

    # ---------------- Hero ----------------

    st.title("🌾 AgriVision AI")

    st.subheader("AI-Powered Decision Support System for Indian Agriculture")

    st.markdown("""
Helping Indian farmers make smarter decisions using Artificial Intelligence,
Machine Learning, Weather Analytics and Data-Driven Insights.
""")

    st.divider()

    # ---------------- Statistics ----------------

    st.subheader("📊 Platform Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("👨‍🌾 Farmers", "1,245")

    with col2:
        st.metric("💰 Predictions", "18,532")

    with col3:
        st.metric("🎯 Accuracy", "94.8%")

    with col4:
        st.metric("📍 Districts", "210")

    st.divider()

    # ---------------- Modules ----------------

    st.subheader("🚀 Platform Modules")

    row1 = st.columns(4)

    row1[0].info("👨‍🌾 Farmer Registration")
    row1[1].info("💰 Income Prediction")
    row1[2].info("🌱 Crop Recommendation")
    row1[3].info("📈 Yield Prediction")

    row2 = st.columns(4)

    row2[0].info("🌦 Weather Intelligence")
    row2[1].info("🤖 AI Assistant")
    row2[2].info("📊 Analytics")
    row2[3].info("🗺 GIS Mapping")

    st.divider()

    # ---------------- Vision ----------------

    st.subheader("📌 Vision")

    st.write("""
AgriVision AI is an intelligent agriculture platform that combines Artificial Intelligence,
Machine Learning, Weather Intelligence and Government Agricultural Data to help Indian
farmers improve productivity, increase income and make better farming decisions.
""")

    st.divider()

    st.caption("© 2026 AgriVision AI")