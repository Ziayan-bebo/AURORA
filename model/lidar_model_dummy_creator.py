import tensorflow as tf
import os

# Make sure model folder exists
os.makedirs("model", exist_ok=True)

# Create a VERY small dummy model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(60,)),
    tf.keras.layers.Dense(6, activation="softmax")  # 6 classes
])

# Compile (required for saving)
model.compile(optimizer="adam", loss="categorical_crossentropy")

# Save model
model.save("model/lidar_model_60points.keras")

print("✔ Dummy Keras model created at: model/lidar_model_60points.keras")
