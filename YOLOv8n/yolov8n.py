from ultralytics import YOLO
import cv2

# Load model YOLOv8
model = YOLO("C:/Users/upgra/Documents/Nyoba Ngoding/Python/Yolov8/best(1).onnx")

# Buka kamera
cap = cv2.VideoCapture(0)

# Atur resolusi kamera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Prediksi dengan YOLOv8 (hilangkan output bawaan dengan verbose=False)
    results = model(frame, verbose=False)

    detected_labels = []  # simpan semua label objek yang terlihat di frame ini

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])      # ID kelas
            label = model.names[cls_id]   # Nama kelas
            detected_labels.append(label)

    # Cek mapping ruangan
    if "Rock" in detected_labels and "Scissors" in detected_labels:
        print("Anda berada di ruang makan 🍽️")
    else:
        for label in detected_labels:
            if label == "Scissors":
                print("Anda sedang di kamar mandi 🚿")
            elif label == "Rock":
                print("Anda sedang di kamar tidur 🛏️")
            elif label == "Paper":
                print("Anda sedang di ruang tamu 🛋️")

    # Ambil hasil frame dengan bounding box
    annotated_frame = results[0].plot()

    # Resize jendela tampilannya
    resized_frame = cv2.resize(annotated_frame, (1280, 720))

    # Tampilkan frame
    cv2.imshow("Kamera YOLOv8", resized_frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan resource
cap.release()
cv2.destroyAllWindows()


# from ultralytics import YOLO
# import cv2

# # Load model YOLOv8
# model = YOLO("C:/Users/upgra/Documents/Nyoba Ngoding/Python/Yolov8/best.pt")

# # Buka kamera
# cap = cv2.VideoCapture(0)

# # Atur resolusi kamera jika ingin
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # Lebar kamera
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)   # Tinggi kamera

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Prediksi dengan YOLOv8
#     results = model(frame)

#     # Ambil hasil frame dengan bounding box
#     annotated_frame = results[0].plot()

#     # Resize jendela tampilannya (misalnya 1280x720)
#     resized_frame = cv2.resize(annotated_frame, (1280, 720))

#     # Tampilkan frame
#     cv2.imshow("Kamera YOLOv8", resized_frame)

#     # Tekan 'q' untuk keluar
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Bersihkan resource
# cap.release()
# cv2.destroyAllWindows()