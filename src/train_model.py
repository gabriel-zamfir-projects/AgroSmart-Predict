import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from network_arch import SoilPredictorNet


def train_pipeline(data_path: str, model_save_path: str, scaler_save_path: str):
    # 1. Load data
    df = pd.read_csv(data_path)

    # 2. Select Features (Inputs) and Target (Output)
    # Features: mean temperature, rain, evapotranspiration, and today's soil moisture
    feature_cols = ["mean_temp", "rain", "evapotranspiration", "soil_moisture"]
    X = df[feature_cols].values
    y = df["target_moisture_24h"].values.reshape(-1, 1)

    # 3. Train-Test Split (80% train, 20% validation)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Scale features (Crucial for neural networks to converge properly)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Save the scaler so we can reuse it in the Streamlit application later
    joblib.dump(scaler, scaler_save_path)
    print(f"Scaler saved successfully to {scaler_save_path}")

    # 5. Convert NumPy arrays into PyTorch Tensors
    device = torch.device("cpu")  # Forcing CPU computation for your Ryzen 7 setup

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    # 6. Initialize Network, Loss function, and Optimizer
    model = SoilPredictorNet(input_size=len(feature_cols)).to(device)
    criterion = nn.MSELoss()  # Mean Squared Error for regression tasks
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # 7. Training Loop
    epochs = 60
    batch_size = 32

    print("Starting training loop on CPU...")
    for epoch in range(epochs):
        model.train()

        # Simple batching mechanism using NumPy shuffling
        permutation = torch.randperm(X_train_tensor.size()[0])

        for i in range(0, X_train_tensor.size()[0], batch_size):
            optimizer.zero_grad()

            indices = permutation[i:i + batch_size]
            batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]

            # Forward pass
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)

            # Backward pass and optimization step
            loss.backward()
            optimizer.step()

        # Evaluation step every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                val_predictions = model(X_val_tensor)
                val_loss = criterion(val_predictions, y_val_tensor)
                # Calculate Mean Absolute Error (MAE) for human readability
                mae = torch.mean(torch.abs(val_predictions - y_val_tensor)).item()

            print(
                f"Epoch [{epoch + 1}/{epochs}] -> Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Val MAE: {mae:.2f}%")

    # 8. Save the trained weights
    torch.save(model.state_dict(), model_save_path)
    print(f"Model successfully trained and saved to {model_save_path}")


if __name__ == "__main__":
    train_pipeline(
        data_path="data/processed_european_data.csv",
        model_save_path="models/soil_predictor.pth",
        scaler_save_path="models/scaler.pkl"
    )