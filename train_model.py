import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Make sure the model folder exists
os.makedirs('model', exist_ok=True)

# Load dataset
data = pd.read_csv('training_data.csv')
feature_columns = [f'point_{i}' for i in range(1, 61)]  # 60 LiDAR points
X = data[feature_columns]
y = data['direction']

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Define neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(60,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(set(y_encoded)), activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

# Save in Keras native format (.keras)
model.save('model/lidar_model_60points.keras')

# Save label encoder
joblib.dump(encoder, 'model/label_encoder_60points.pkl')

print("✅ Model trained and saved in Keras native format (.keras)")

