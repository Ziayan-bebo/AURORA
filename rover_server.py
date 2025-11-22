from flask import Flask, request, jsonify, render_template_string
import numpy as np
import tensorflow as tf
import joblib
import time
import os

# ===========================
# CONFIGURATION
# ===========================
CAMERA_IP = "192.168.222.251"  # Replace with your ESP32-CAM IP
MODEL_PATH = "model/lidar_model_60points.keras"
ENCODER_PATH = "model/label_encoder_60points.pkl"

# ===========================
# INITIALIZATION
# ===========================
app = Flask(__name__)

print("🔄 Loading ML model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model, encoder = None, None

latest_direction = "stop"
log_entries = []

# ===========================
# DASHBOARD HTML
# ===========================
dashboard_html = """
<!DOCTYPE html>
<html>
<head>
  <title>AURORA Rover Dashboard</title>
  <style>
    body {
      font-family: 'Segoe UI', sans-serif;
      background: #0a0a0a;
      color: #f0f0f0;
      text-align: center;
      margin: 0;
      padding: 0;
    }
    h1 {
      color: #00ffff;
      margin-top: 20px;
      font-size: 2.2em;
      letter-spacing: 1px;
    }
    #video {
      border: 3px solid #00ffff;
      border-radius: 12px;
      width: 720px;
      height: 540px;
      margin-top: 15px;
    }
    .controls {
      margin-top: 25px;
    }
    button {
      background: #00ffff;
      color: #111;
      border: none;
      border-radius: 6px;
      padding: 10px 20px;
      margin: 6px;
      cursor: pointer;
      font-weight: bold;
      font-size: 1em;
    }
    button:hover {
      background: #fff;
      color: #000;
    }
    .log {
      text-align: left;
      margin: 25px auto;
      width: 700px;
      background: #1b1b1b;
      padding: 12px;
      border-radius: 10px;
      height: 220px;
      overflow-y: scroll;
      border: 1px solid #00ffff40;
    }
    .direction {
      font-size: 1.4em;
      color: #00ff80;
      font-weight: bold;
    }
    iframe {
      background: #000;
    }
  </style>
  <script>
    async function sendCommand(cmd) {
      await fetch('/set_command', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command: cmd })
      });
    }

    async function updateDirection() {
      const res = await fetch('/get_command');
      const data = await res.json();
      document.getElementById('current_dir').innerText = data.command.toUpperCase();
      setTimeout(updateDirection, 1000);
    }

    async function updateLogs() {
      const res = await fetch('/get_logs');
      const text = await res.text();
      document.getElementById('logs').innerHTML = text;
      setTimeout(updateLogs, 2000);
    }

    window.onload = () => { updateDirection(); updateLogs(); };
  </script>
</head>
<body>
  <h1>🛰️ AURORA Rover Dashboard</h1>

  <!-- Embedded ESP32-CAM Stream -->
  <iframe id="video"
          src="http://{{camera_ip}}"
          allow="camera; autoplay"
          frameborder="0">
  </iframe>

  <h2>Current Direction: <span id="current_dir" class="direction">STOP</span></h2>

  <div class="controls">
    <button onclick="sendCommand('forward')">↑ Forward</button><br>
    <button onclick="sendCommand('left')">← Left</button>
    <button onclick="sendCommand('stop')">■ Stop</button>
    <button onclick="sendCommand('right')">→ Right</button><br>
    <button onclick="sendCommand('sharp_left')">⟲ Sharp Left</button>
    <button onclick="sendCommand('sharp_right')">⟳ Sharp Right</button>
    <button onclick="sendCommand('rotate')">↻ Rotate 180°</button>
  </div>

  <div class="log" id="logs"></div>
</body>
</html>
"""

# ===========================
# ROUTES
# ===========================

@app.route("/")
def dashboard():
    return render_template_string(dashboard_html, camera_ip=CAMERA_IP)


@app.route("/data", methods=["POST"])
def receive_data():
    global latest_direction
    data = request.json

    if not data or "lidar_points" not in data:
        return jsonify({"status": "error", "message": "Missing 'lidar_points'"}), 400

    lidar_points = data["lidar_points"]
    if len(lidar_points) != 60:
        return jsonify({"status": "error", "message": "Expected 60 LiDAR points"}), 400

    # Failsafe: Rotate if all LiDAR points are too close (<15 cm)
    if all(p < 15 for p in lidar_points):
        latest_direction = "rotate"
    else:
        arr = np.array(lidar_points).reshape(1, -1)
        preds = model.predict(arr)
        label_index = np.argmax(preds, axis=1)
        latest_direction = encoder.inverse_transform(label_index)[0]

    log_entries.append(f"[{time.strftime('%H:%M:%S')}] Direction → {latest_direction}")
    if len(log_entries) > 60:
        log_entries.pop(0)

    return jsonify({"status": "ok", "predicted_direction": latest_direction})


@app.route("/get_command", methods=["GET"])
def get_command():
    return jsonify({"command": latest_direction})


@app.route("/set_command", methods=["POST"])
def set_command():
    global latest_direction
    cmd = request.json.get("command", "")
    latest_direction = cmd
    log_entries.append(f"[{time.strftime('%H:%M:%S')}] Manual override → {cmd}")
    if len(log_entries) > 60:
        log_entries.pop(0)
    return jsonify({"status": "ok", "command": cmd})


@app.route("/get_logs", methods=["GET"])
def get_logs():
    return "<br>".join(log_entries[::-1])

# ===========================
# MAIN
# ===========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("🌐 AURORA Flask server is running...")
    print(f"🧠 Model: {MODEL_PATH}")
    print(f"🎥 Camera Stream: http://{CAMERA_IP}")
    print(f"💻 Open dashboard at: http://127.0.0.1:{port} or your local IP\n")
    app.run(host="0.0.0.0", port=port)

