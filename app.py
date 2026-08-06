import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Dashboard",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# SAMPLE DATA
# (We'll replace this with NSS later)
# -----------------------------
df = pd.DataFrame({
    "State": ["Punjab","Haryana","UP","Bihar","Maharashtra","Gujarat"],
    "Income": [420000,390000,280000,210000,470000,450000],
    "Yield": [5.8,5.3,4.1,3.8,6.0,5.5],
    "Crop": ["Wheat","Rice","Sugarcane","Maize","Cotton","Cotton"]
})

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>

.block-container{
    padding-top:2rem;
}

.card{
    background:#111827;
    padding:20px;
    border-radius:15px;
    border:1px solid #2d3748;
}

.metric-card{
    background:#161b22;
    border-radius:15px;
    padding:20px;
    text-align:center;
    border:1px solid #30363d;
}

.metric-card:hover{
    border:1px solid #00ff99;
}

</style>
""",unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.title("📊 AgriVision Dashboard")

st.caption("Real-time Agricultural Intelligence Platform")

st.divider()

# -----------------------------
# KPI CARDS
# -----------------------------

c1,c2,c3,c4=st.columns(4)

with c1:
    st.metric(
        "👨‍🌾 Farmers",
        "1,245",
        "+12%"
    )

with c2:
    st.metric(
        "💰 Avg Income",
        "₹3.7L",
        "+8%"
    )

with c3:
    st.metric(
        "🌾 Avg Yield",
        "5.1 Tons",
        "+5%"
    )

with c4:
    st.metric(
        "🎯 Accuracy",
        "94.8%",
        "+1.2%"
    )

st.divider()

# -----------------------------
# CHARTS
# -----------------------------

left,right=st.columns(2)

with left:

    fig=px.bar(
        df,
        x="State",
        y="Income",
        color="State",
        title="Income by State"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with right:

    fig2=px.pie(
        df,
        names="Crop",
        title="Crop Distribution"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

st.divider()

# -----------------------------
# TABLE + AI INSIGHTS
# -----------------------------

left,right=st.columns([2,1])

with left:

    st.subheader("🌾 Recent Farmer Records")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

with right:

    st.subheader("🤖 AI Insights")

    st.success("Punjab shows the highest average income.")

    st.info("Cotton is emerging as a profitable crop.")

    st.warning("Monitor rainfall in Bihar this week.")

    st.write("")

    st.subheader("🌦 Weather")

    st.metric(
        "Current",
        "28°C"
    )

    st.metric(
        "Humidity",
        "68%"
    )

    st.metric(
        "Rainfall",
        "860 mm"
    )

st.divider()

# -----------------------------
# QUICK ACTIONS
# -----------------------------

st.subheader("⚡ Quick Actions")

b1,b2,b3,b4=st.columns(4)

with b1:
    st.button(
        "👨‍🌾 Register Farmer",
        use_container_width=True
    )

with b2:
    st.button(
        "💰 Predict Income",
        use_container_width=True
    )

with b3:
    st.button(
        "🌱 Recommend Crop",
        use_container_width=True
    )

with b4:
    st.button(
        "🤖 Ask AI",
        use_container_width=True
    )

st.divider()

st.caption("© 2026 AgriVision AI")