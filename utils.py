"""
utils.py
--------
Central place for:
  1) Loading trained models from /models (once your teammates hand them over)
  2) Prediction functions the UI calls

DESIGN GOAL: the Streamlit UI never talks to a model directly.
It always calls predict_income(), predict_yield(), recommend_crops(), etc.
Right now those functions run in DEMO MODE (simple formulas / randomised
placeholders) so the app is fully clickable today. The moment a real
.pkl model appears in /models with the expected filename, these functions
auto-switch to using it -- no UI changes required.

MODEL CONTRACT (give this section to your data teammates):
------------------------------------------------------------
models/income_model.pkl
    A single object with a `.predict(df)` method (plain sklearn/xgboost
    estimator, or better -- a full sklearn Pipeline that includes its own
    preprocessing/encoding). Input df columns, in this exact order:
        age                     int
        education_level        str  -> "None","Primary","Secondary","Graduate","Postgraduate"
        total_land_ha           float
        crop_yield_per_ha       float
        non_agri_income         float
        distance_to_market_km   float
        rainfall_mm             float
    Output: predicted monthly income in INR (float).

models/yield_model.pkl
    Same idea. Input columns:
        crop_type               str
        total_land_ha           float
        rainfall_mm             float
        temperature_c           float
        input_costs             float
    Output: predicted yield in tons/hectare (float).

models/crop_model.pkl
    Classifier. Input columns:
        region                  str
        soil_type               str (optional, default "Unknown")
        rainfall_mm             float
        temperature_c           float
        current_crop            str (optional, default "Unknown")
    Output: either a single crop label, or (better) .predict_proba so we
    can show a ranked top-3 list. Code below handles both.

Wrapping the preprocessing INSIDE the pickled pipeline is strongly
recommended -- that way this file never needs to know about encoders.
"""

from pathlib import Path
import random

import joblib
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# ----------------------------------------------------------------------
# Model loading (cached so the app doesn't reload from disk every rerun)
# ----------------------------------------------------------------------
_model_cache = {}


def _load(name: str):
    """Try to load models/<name>.pkl. Returns None if it doesn't exist yet
    or fails to load -- callers fall back to demo mode in that case."""
    if name in _model_cache:
        return _model_cache[name]

    path = MODELS_DIR / f"{name}.pkl"
    model = None
    if path.exists():
        try:
            model = joblib.load(path)
        except Exception as e:  # noqa: BLE001 - surface any load error, don't crash the app
            import streamlit as st

            st.warning(f"Found {path.name} but couldn't load it: {e}")
            model = None

    _model_cache[name] = model
    return model


def model_status() -> dict:
    """Used by the sidebar to show which real models are live vs. demo."""
    return {
        "income_model": _load("income_model") is not None,
        "yield_model": _load("yield_model") is not None,
        "crop_model": _load("crop_model") is not None,
    }


# ----------------------------------------------------------------------
# Prediction functions -- these are what every tab in app.py calls
# ----------------------------------------------------------------------
def predict_income(inputs: dict) -> dict:
    """inputs: age, education_level, total_land_ha, crop_yield_per_ha,
    non_agri_income, distance_to_market_km, rainfall_mm
    Returns: {"value": float, "confidence": float, "demo": bool}
    """
    model = _load("income_model")

    if model is not None:
        df = pd.DataFrame([inputs])
        value = float(model.predict(df)[0])
        confidence = 0.85  # swap for a real interval/uncertainty estimate if the model provides one
        return {"value": value, "confidence": confidence, "demo": False}

    # ---- DEMO MODE: simple transparent formula, NOT a real prediction ----
    base = (
        inputs["total_land_ha"] * inputs["crop_yield_per_ha"] * 1800
        + inputs["non_agri_income"]
        - inputs["distance_to_market_km"] * 50
        + inputs["rainfall_mm"] * 2
    )
    base = max(base, 3000)
    noise = random.uniform(0.95, 1.05)
    return {"value": round(base * noise, -2), "confidence": 0.5, "demo": True}


def predict_yield(inputs: dict) -> dict:
    """inputs: crop_type, total_land_ha, rainfall_mm, temperature_c, input_costs"""
    model = _load("yield_model")

    if model is not None:
        df = pd.DataFrame([inputs])
        value = float(model.predict(df)[0])
        return {"value": value, "demo": False}

    base = 2.5 + inputs["rainfall_mm"] / 400 + inputs["temperature_c"] / 50
    base -= inputs["input_costs"] / 50000
    base = max(base, 0.5)
    return {"value": round(base * random.uniform(0.9, 1.1), 2), "demo": True}


def recommend_crops(inputs: dict) -> dict:
    """inputs: region, soil_type, rainfall_mm, temperature_c, current_crop
    Returns: {"crops": [(name, score), ...], "demo": bool}
    """
    model = _load("crop_model")

    if model is not None:
        df = pd.DataFrame([inputs])
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df)[0]
            labels = model.classes_
            ranked = sorted(zip(labels, proba), key=lambda x: -x[1])[:3]
            return {"crops": ranked, "demo": False}
        label = model.predict(df)[0]
        return {"crops": [(label, 1.0)], "demo": False}

    # ---- DEMO MODE ----
    pool = ["Wheat", "Rice", "Cotton", "Sugarcane", "Maize", "Soybean", "Groundnut", "Pulses"]
    picks = random.sample(pool, 3)
    scores = sorted([random.uniform(0.5, 0.9) for _ in range(3)], reverse=True)
    return {"crops": list(zip(picks, scores)), "demo": True}


def ai_assistant_reply(question: str, context: dict) -> str:
    """Placeholder for the Gemini-powered assistant.
    Swap the body of this function for a real `google-generativeai` call once
    you have an API key -- read it from st.secrets["GEMINI_API_KEY"], never
    hardcode it. Keep the function signature the same so app.py doesn't change.
    """
    import streamlit as st

    api_key = st.secrets.get("GEMINI_API_KEY", None) if hasattr(st, "secrets") else None
    if api_key:
        # TODO: wire up google-generativeai here, e.g.:
        # import google.generativeai as genai
        # genai.configure(api_key=api_key)
        # model = genai.GenerativeModel("gemini-1.5-flash")
        # resp = model.generate_content(build_prompt(question, context))
        # return resp.text
        return "(Gemini key detected but call not wired up yet -- see ai_assistant_reply() in utils.py)"

    return (
        "DEMO MODE: I don't have a live AI connection yet. "
        f"Once GEMINI_API_KEY is set in secrets, I'll give real, personalised advice. "
        f"For now -- based on what you've entered so far ({context.get('summary', 'no profile yet')}), "
        "a generic tip: diversifying income with a secondary crop and reducing distance-to-market "
        "costs are usually the two biggest levers for farmer income."
    )