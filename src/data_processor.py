import requests
import pandas as pd
import numpy as np
import time
import os

# Stable European coordinate hubs
EUROPEAN_LOCATIONS = {
    "Bucharest_RO": {"lat": 44.4268, "lon": 26.1025},
    "Seville_ES": {"lat": 37.3891, "lon": -5.9845},
    "Nantes_FR": {"lat": 47.2184, "lon": -1.5536},
    "Hannover_DE": {"lat": 52.3759, "lon": 9.7320}
}


def fetch_location_data(lat: float, lon: float) -> pd.DataFrame:
    """Fetches core hourly metrics and aggregates them into daily slots using Pandas."""
    # We use the rock-solid forecast endpoint with past_days tracking
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,precipitation",
        "timezone": "auto",
        "past_days": 60,  # Safe 1-month window to get stable real training data
        "forecast_days": 0
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        if response.status_code != 200:
            return pd.DataFrame()

        data = response.json()
        if "hourly" not in data:
            return pd.DataFrame()

        hourly_data = data["hourly"]

        # Build hourly dataframe
        df_hourly = pd.DataFrame({
            "timestamp": pd.to_datetime(hourly_data["time"]),
            "temperature": hourly_data["temperature_2m"],
            "precipitation": hourly_data["precipitation"]
        })

        # Group by day using Pandas Resampling logic
        df_hourly.set_index("timestamp", inplace=True)
        daily_df = pd.DataFrame()
        daily_df["mean_temp"] = df_hourly["temperature"].resample("D").mean()
        daily_df["max_temp"] = df_hourly["temperature"].resample("D").max()
        daily_df["min_temp"] = df_hourly["temperature"].resample("D").min()
        daily_df["rain"] = df_hourly["precipitation"].resample("D").sum()
        daily_df.reset_index(inplace=True)

        # Calculate Evapotranspiration locally via Hargreaves method
        temp_range = np.clip(daily_df["max_temp"] - daily_df["min_temp"], 0.1, 50.0)
        daily_df["evapotranspiration"] = 0.0023 * (daily_df["mean_temp"] + 17.8) * np.sqrt(temp_range) * 3.5
        daily_df["evapotranspiration"] = np.clip(daily_df["evapotranspiration"], 0.2, 10.0)

        # Keep only the features needed by PyTorch
        return daily_df[["mean_temp", "rain", "evapotranspiration"]]

    except Exception:
        return pd.DataFrame()


def simulate_soil_moisture(df: pd.DataFrame, loc_name: str) -> pd.DataFrame:
    initial_moisture = 35.0 if "ES" in loc_name else 60.0
    moisture_signals = []
    current_moisture = initial_moisture

    for _, row in df.iterrows():
        water_change = row['rain'] - row['evapotranspiration']
        current_moisture += water_change
        current_moisture = np.clip(current_moisture, 10.0, 100.0)
        moisture_signals.append(current_moisture)

    df["soil_moisture"] = moisture_signals
    df["target_moisture_24h"] = df["soil_moisture"].shift(-1)
    df.dropna(inplace=True)
    return df


def build_dataset() -> pd.DataFrame:
    master_dfs = []
    for name, coords in EUROPEAN_LOCATIONS.items():
        print(f"Gathering data via Pandas Hourly Pipelines for: {name}...")
        df = fetch_location_data(coords["lat"], coords["lon"])
        if not df.empty:
            df = simulate_soil_moisture(df, name)
            master_dfs.append(df)
            print(f"-> Successfully processed {len(df)} rows.")
        time.sleep(1.0)

    if not master_dfs:
        raise ValueError("All location pipelines failed to fetch valid JSON data.")
    return pd.concat(master_dfs, ignore_index=True)


if __name__ == "__main__":
    print("Initializing Hourly Aggregator...")
    try:
        dataset = build_dataset()
        os.makedirs("data", exist_ok=True)
        dataset.to_csv("data/processed_european_data.csv", index=False)
        print(f"\n⚡ Success! Dataset saved. Total footprint: {len(dataset)} rows.")
    except Exception as ex:
        print(f"\n❌ Pipeline failed: {ex}")