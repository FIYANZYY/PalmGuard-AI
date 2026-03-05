from ultralytics import RTDETR

if __name__ == "__main__":
    # Menggunakan model dasar RT-DETR-L
    model = RTDETR("rtdetr-l.pt")

    model.train(
        data="data.yaml",      # Mengacu pada file data.yaml di atas
        epochs=50,             # Melatih sebanyak 50 kali
        imgsz=640,             # Ukuran gambar standar
        device=0,              # Menggunakan GPU GTX NVIDIA
        batch=2,               # Aman untuk VRAM 4GB (Aspire 7)
        workers=2,             # Meringankan beban CPU i5
        amp=True,              # Menghemat memori GPU
        project="Sawit_Project",
        name="rtdetr_l_sawit"   
    )