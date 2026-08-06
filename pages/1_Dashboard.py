import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AgriVision Dashboard")
st.markdown("---")

farmers = pd.DataFrame({

    "State":[
        "Punjab","Punjab",
        "Haryana",
        "UP",
        "UP",
        "Maharashtra",
        "Bihar",
        "Gujarat"
    ],

    "Crop":[
        "Wheat",
        "Rice",
        "Wheat",
        "Sugarcane",
        "Rice",
        "Cotton",
        "Maize",
        "Cotton"
    ],

    "Income":[
        420000,
        350000,
        480000,
        270000,
        230000,
        510000,
        180000,
        450000
    ],

    "Yield":[
        5.4,
        4.2,
        6.0,
        7.1,
        4.5,
        5.8,
        3.6,
        6.2
    ],

    "Rainfall":[
        720,
        900,
        680,
        1050,
        980,
        640,
        1120,
        810
    ]
})

c1,c2,c3 = st.columns(3)

c1.metric(
    "👨‍🌾 Farmers",
    len(farmers)
)

c2.metric(
    "💰 Average Income",
    f"₹{farmers['Income'].mean():,.0f}"
)

c3.metric(
    "🌾 Average Yield",
    round(farmers["Yield"].mean(),2)
)

c4,c5,c6 = st.columns(3)

c4.metric(
    "🌦 Average Rainfall",
    round(farmers["Rainfall"].mean(),1)
)

c5.metric(
    "🌱 Crops",
    farmers["Crop"].nunique()
)

c6.metric(
    "📍 States",
    farmers["State"].nunique()
)

fig = px.bar(

    farmers,

    x="State",

    y="Income",

    color="State",

    title="Income by State"

)

st.plotly_chart(fig,use_container_width=True)

fig2 = px.pie(

    farmers,

    names="Crop",

    title="Crop Distribution"

)

st.plotly_chart(fig2,use_container_width=True)


st.subheader("🌾 Farmer Data")

st.dataframe(
    farmers,
    use_container_width=True
)
