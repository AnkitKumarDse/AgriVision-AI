import joblib
import numpy as np
model = joblib.load("models/crop_model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
N = st.number_input("Nitrogen (N)", 0, 150, 90)

P = st.number_input("Phosphorus (P)", 0, 150, 42)

K = st.number_input("Potassium (K)", 0, 250, 43)

temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 27.0)

humidity = st.number_input("Humidity (%)", 0.0, 100.0, 80.0)

ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)

rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 200.0)
if st.button("Generate Recommendation"):
    npk_total = N + P + K

n_ratio = N / npk_total

p_ratio = P / npk_total

k_ratio = K / npk_total
low = 1 if rainfall < 100 else 0

medium = 1 if 100 <= rainfall < 200 else 0

high = 1 if rainfall >= 200 else 0
cool = 1 if temperature < 20 else 0

moderate = 1 if 20 <= temperature < 30 else 0

hot = 1 if temperature >= 30 else 0
features = np.array([[
    N,
    P,
    K,
    temperature,
    humidity,
    ph,
    rainfall,
    npk_total,
    n_ratio,
    p_ratio,
    k_ratio,
    low,
    medium,
    high,
    cool,
    moderate,
    hot
]])
prediction = model.predict(features)

crop = label_encoder.inverse_transform(prediction)[0]
st.success(f"🌾 Recommended Crop: {crop}")