import joblib
import os
from sklearn.preprocessing import LabelEncoder

# Ensure folder exists
os.makedirs("model", exist_ok=True)

# 6 movement classes
labels = ["forward", "left", "right", "sharp_left", "sharp_right", "stop"]

encoder = LabelEncoder()
encoder.fit(labels)

joblib.dump(encoder, "model/label_encoder_60points.pkl")

print("✔ Dummy .pkl label encoder created in /model")
