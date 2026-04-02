import requests
import cv2
import numpy as np
import os

# 1. Konfigurasi
url = 'http://127.0.0.1:5000/predict'
# Pastiin file ini ada di folder images lu
image_path = 'images/daun-coklat-ujung.jpg' 

if not os.path.exists(image_path):
    print(f"Waduh bro, file {image_path} nggak ada! Cek lagi folder images lu.")
else:
    # 2. Kirim gambar ke API
    with open(image_path, 'rb') as img_file:
        print(f"Lagi ngetes gambar: {image_path}...")
        files = {'image': img_file}
        try:
            response = requests.post(url, files=files)
            detections = response.json()
            print("Hasil Deteksi dari Server:", detections)

            # 3. Visualisasi Hasil
            img = cv2.imread(image_path)
            
            if not detections:
                print("Model tidak mendeteksi penyakit/objek apapun.")
            
            for det in detections:
                # Ambil koordinat pixel asli
                x1, y1, x2, y2 = map(int, det['bbox'])
                label = f"{det['label']} {det['confidence']}%"
                
                # Gambar Bounding Box (Warna Hijau)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                # Kasih background label biar teks kebaca
                cv2.rectangle(img, (x1, y1 - 25), (x1 + 200, y1), (0, 255, 0), -1)
                cv2.putText(img, label, (x1 + 5, y1 - 7), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # 4. Tampilkan Gambar (Window bisa di-resize kalau kegedean)
            # 1. Bikin jendela yang bisa di-resize
            target_window_name = "Review Hasil Deteksi PalmGuard AI"
            cv2.namedWindow(target_window_name, cv2.WINDOW_NORMAL)
            
            # 2. Ambil ukuran asli gambar lu
            h_asli, w_asli = img.shape[:2]
            print(f"Ukuran Asli Gambar: {w_asli}x{h_asli}")

            # 3. Tentuin ukuran maksimal jendela di layar laptop lu
            # Kita set max lebar 1200px atau max tinggi 700px (biar pas di Full HD)
            mak_w = 1200
            mak_h = 700
            
            # Hitung rasio biar nggak gepeng
            rasio = min(mak_w / w_asli, mak_h / h_asli)
            new_w = int(w_asli * rasio)
            new_h = int(h_asli * rasio)
            
            # 4. Paksa jendela OpenCV ke ukuran baru yang manusiawi
            cv2.resizeWindow(target_window_name, new_w, new_h)

            # 5. Tampilkan gambar asli di dalam jendela yang sudah dikecilin
            cv2.imshow(target_window_name, img)
            
            print(f"Jendela dikecilkan jadi: {new_w}x{new_h} biar kelihatan semua.")
            print("Tekan tombol apa saja di gambar untuk tutup.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"Gagal konek ke server Flask! Error: {e}")