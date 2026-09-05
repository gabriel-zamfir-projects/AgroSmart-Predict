import matplotlib.pyplot as plt


def generate_moisture_chart(historical_data, predicted_value):
    fig, ax = plt.subplots(figsize=(10, 4))

    # Plot historical trajectory
    ax.plot(historical_data, label="Recent Soil Moisture Trend", color="#1f77b4", marker='o', linewidth=2)

    # Highlight prediction point
    prediction_index = len(historical_data)
    ax.scatter(prediction_index, predicted_value, color="#d62728", s=150, zorder=5, label="PyTorch AI Forecast (24h)")

    # Connect history to prediction with a dashed line
    ax.plot([prediction_index - 1, prediction_index], [historical_data[-1], predicted_value], color="#d62728",
            linestyle=":")

    # Standard agricultural warning bounds
    ax.axhline(y=30, color='#e377c2', linestyle='--', alpha=0.7, label='Universal Wilting Risk Boundary')

    ax.set_ylabel("Soil Moisture Content (%)", fontsize=10)
    ax.set_xlabel("Time Horizon (Days / Transition to Tomorrow)", fontsize=10)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.2)

    return fig