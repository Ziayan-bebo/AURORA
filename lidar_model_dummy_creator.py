import tensorflow as tf
import os

os.makedirs("model", exist_ok=True)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(60,)),
    tf.keras.layers.Dense(6, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy")

model.save("model/lidar_model_60points.keras")

print("✔ Dummy Keras model created")
