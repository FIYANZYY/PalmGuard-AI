from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import RTDETR
import cv2
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Path model hasil training PalmGuard AI
model_path = 'Sawit_Project/rtdetr_l_sawit/weights/best.pt'

if os.path.exists(model_path):
    print(f"Memuat model sukses: {model_path}")
    model = RTDETR(model_path)
else:
    print("Model training tidak ditemukan, menggunakan model default.")
    model = RTDETR("rtdetr-l.pt")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
        
    file = request.files['image']
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    # Prediksi menggunakan RT-DETR
    results = model(img)
    
    detections = []
    for r in results:
        # Ambil koordinat pixel asli (xyxy)
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy()
        
        for box, conf, cls in zip(boxes, confs, clss):
            detections.append({
                "label": model.names[int(cls)],
                "confidence": round(float(conf) * 100, 1),
                "bbox": box.tolist()  # Mengirim [x1, y1, x2, y2]
            })
            
    return jsonify(detections)

if __name__ == '__main__':
    # Aspire 7 lu bakal jalan di port 5000
    app.run(host='0.0.0.0', port=5000, debug=False)