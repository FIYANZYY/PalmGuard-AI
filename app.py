from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import RTDETR
import cv2
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Path hasil training yang akan dibuat oleh train_rtdetr.py
model_path = 'Sawit_Project/rtdetr_l_sawit/weights/best.pt'

if os.path.exists(model_path):
    model = RTDETR(model_path)
else:
    # Pakai model default jika belum selesai training
    model = RTDETR("rtdetr-l.pt")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    results = model(img)
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "label": model.names[int(box.cls)],
                "confidence": round(float(box.conf) * 100, 1),
                "bbox": box.xyxy.tolist() # Koordinat untuk Canvas
            })
    return jsonify(detections)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)