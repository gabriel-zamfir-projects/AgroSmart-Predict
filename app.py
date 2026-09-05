import streamlit as st
import torch
import numpy as np
import joblib
import os
from src.weather_api import get_live_forecast
from src.network_arch import SoilPredictorNet
from src.plots import generate_moisture_chart
from src.translations import TRANSLATIONS  # Clean architecture import

# Page configuration using dynamic language hooks
st.set_page_config(page_title="AgroSmart Europe", page_icon="🇪🇺", layout="wide")

# Sidebar Language Selection
st.sidebar.header("🌐 Language / Limbă / Sprache / Langue")
lang = st.sidebar.selectbox("Select Language", ["English", "Română", "Deutsch", "Français"])

# Pointer to the selected language sub-dictionary
lang_dict = TRANSLATIONS[lang]

# Main Headers
st.title(lang_dict["title"])
st.subheader(lang_dict["subtitle"])


# Resource Caching for PyTorch Model and Scaler
@st.cache_resource
def load_ml_assets():
    model = SoilPredictorNet(input_size=4)

    # Paths configured to point directly to the src/models pipeline footprint
    model_path = "src/models/soil_predictor.pth"
    scaler_path = "src/models/scaler.pkl"

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        scaler = joblib.load(scaler_path)
        return model, scaler
    else:
        return None, None


model, scaler = load_ml_assets()

# Sidebar UI Input Controls
st.sidebar.markdown("---")
st.sidebar.header(lang_dict["sidebar_header"])
lat = st.sidebar.number_input(lang_dict["lat_label"], value=44.6000, format="%.4f")
lon = st.sidebar.number_input(lang_dict["lon_label"], value=27.3000, format="%.4f")

crop_type = st.sidebar.selectbox(lang_dict["crop_label"],
                                 ["Corn/Maize", "Wheat", "Sunflower", "Vegetables", "Vineyards/Orchards"])
current_soil_moist = st.sidebar.slider(lang_dict["moist_label"], 10, 100, 45)

# Execution Logic Core
if model is None or scaler is None:
    st.warning("⚠️ Machine Learning assets not found. Please execute 'src/train_model.py' first.")
else:
    if st.button(lang_dict["btn_analyze"]):
        with st.spinner(lang_dict["spinner"]):
            try:
                # 1. Fetch live telemetry from our hourly-resampled Pandas pipeline
                weather_metrics = get_live_forecast(lat, lon)
                st.success(lang_dict["success_msg"])

                # Render incoming telemetry inside dynamic metrics blocks
                col1, col2, col3 = st.columns(3)
                col1.metric(lang_dict["metric_temp"], f"{weather_metrics['mean_temp']:.1f} °C")
                col2.metric(lang_dict["metric_rain"], f"{weather_metrics['rain']:.1f} mm")
                col3.metric(lang_dict["metric_et"], f"{weather_metrics['evapotranspiration']:.2f} mm")

                # 2. Structure array vector for PyTorch Neural Inference
                raw_input = np.array([[
                    weather_metrics['mean_temp'],
                    weather_metrics['rain'],
                    weather_metrics['evapotranspiration'],
                    float(current_soil_moist)
                ]])

                # Apply normalization transformation
                scaled_input = scaler.transform(raw_input)
                input_tensor = torch.tensor(scaled_input, dtype=torch.float32)

                with torch.no_grad():
                    predicted_moisture = model(input_tensor).item()

                # 3. AI Recommendation System Display
                st.markdown("---")
                st.subheader(lang_dict["engine_title"])
                st.metric(label=lang_dict["metric_pred"], value=f"{predicted_moisture:.1f} %")

                # Business rule customized for active crop choice
                critical_threshold = 35.0 if crop_type == "Corn/Maize" else 25.0

                if predicted_moisture < critical_threshold:
                    st.error(lang_dict["danger_msg"].format(threshold=critical_threshold, crop=crop_type))
                    water_deficit = critical_threshold - predicted_moisture
                    recommended_liters = int(water_deficit * 1.5)
                    st.info(lang_dict["recommendation"].format(liters=recommended_liters))
                else:
                    st.success(lang_dict["safe_msg"].format(crop=crop_type))

                # 4. Inject Matplotlib Visual Chart
                historical_tail = [current_soil_moist + 5, current_soil_moist + 2, current_soil_moist]
                fig = generate_moisture_chart(historical_tail, predicted_moisture)
                st.pyplot(fig)

            except Exception as error:
                st.error(f"An execution error occurred: {error}")