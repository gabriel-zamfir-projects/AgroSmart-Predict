import requests
import pandas as pd
import numpy as np


def get_live_forecast(lat: float, lon: float) -> dict:
    """
    Fetches live hourly forecast and structures metrics locally using Pandas
    to guarantee 100% alignment with the PyTorch model features.
    """
    # CRITICAL FIX: Using the correct, official API sub-domain endpoint
    url = "https://api.open-meteo.com/v1/forecast"

    # Querying stable hourly telemetry matrices to prevent server drops
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,precipitation",
        "timezone": "auto",
        "forecast_days": 1  # We only need tomorrow's continuous horizon
    }

    response = requests.get(url, params=params, timeout=15)

    if response.status_code != 200:
        raise Exception(f"Live Weather API error: HTTP {response.status_code}")

    data = response.json()
    if "hourly" not in data or "time" not in data["hourly"]:
        raise Exception("Invalid data structure received from the live weather server.")

    hourly_data = data["hourly"]

    # Wrap incoming metrics into a localized hourly Pandas DataFrame
    df_hourly = pd.DataFrame({
        "temperature": hourly_data["temperature_2m"],
        "precipitation": hourly_data["precipitation"]
    })

    # Process hourly matrices into single daily metrics for the AI core
    mean_temp = float(df_hourly["temperature"].mean())
    max_temp = float(df_hourly["temperature"].max())
    min_temp = float(df_hourly["temperature"].min())
    rain = float(df_hourly["precipitation"].sum())

    # Locally calculate Hargreaves Evapotranspiration to mirror the training metrics
    temp_range = max(max_temp - min_temp, 0.1)
    evapotranspiration = 0.0023 * (mean_temp + 17.8) * np.sqrt(temp_range) * 3.5
    evapotranspiration = float(np.clip(evapotranspiration, 0.2, 10.0))

    # Returns the clean vector array needed by app.py
    return {
        "mean_temp": mean_temp,
        "rain": rain,
        "evapotranspiration": evapotranspiration
    }