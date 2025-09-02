import cv2

def test_cameras(max_devices=10):
    print("🔍 Mencoba mendeteksi kamera yang tersedia...\n")
    available_cams = []

    for i in range(max_devices):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            print(f"[X] Kamera index {i} tidak bisa dibuka.")
            continue

        ret, frame = cap.read()
        if ret:
            print(f"[✔] Kamera ditemukan di index {i}")
            available_cams.append(i)
        else:
            print(f"[ ] Kamera index {i} terbuka, tapi tidak ada frame.")
        cap.release()

    if not available_cams:
        print("\n❌ Tidak ada kamera yang terdeteksi.")
    else:
        print(f"\n✅ Kamera yang bisa digunakan: {available_cams}")

if __name__ == "__main__":
    test_cameras(max_devices=10)  # bisa ubah 10 jadi 30 kalau mau cek lebih banyak
