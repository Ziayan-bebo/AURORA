import joblib
import os
from sklearn.preprocessing import LabelEncoder

# Ensure model folder exists
os.makedirs("model", exist_ok=True)

# Define the 6 output classes
directions = ["forward", "left", "right", "sharp_left", "sharp_right", "stop"]

# Create encoder
encoder = LabelEncoder()
encoder.fit(directions)

# Save encoder
joblib.dump(encoder, "model/label_encoder_60points.pkl")

print("✔ Dummy label encoder created at: model/label_encoder_60points.pkl")
