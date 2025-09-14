import os
import cv2
import re
import io
import time
import threading
import numpy as np
import pygame
from gtts import gTTS
import easyocr

# ======================
# Setup & Konfigurasi
# ======================
SAVE_DEBUG = True
os.makedirs("temp", exist_ok=True)

# OCR / Audio
OCR_LANGUAGES = ['id', 'en']
MIN_CONFIDENCE = 0.30
DESKEW_FOR_OCR = False  # False = OCR pakai orientasi asli (tanpa rotasi)

# Realtime (ringan)
FOCUS_THRESHOLD = 45.0
STABILITY_THRESHOLD = 0.93
STABLE_FRAMES_REQUIREMENT = 2

# ROI
ROI_W_RATIO = 0.80
ROI_H_RATIO = 0.60

# Filter hasil OCR
MIN_BBOX_HEIGHT = 12
MIN_AREA_RATIO  = 0.0006
SHORT_TOKEN_PROB = 0.55
ALLOWED_CHARS_RE = re.compile(r"[A-Za-zÀ-ÿ0-9 ,.'‘’“”\-–—():/?%@#&\[\]]+")

# ======================
# State bersama
# ======================
shared_data = {
    "frame": None,
    "status": "MENSTABILKAN...",
    "results": [],
    "combined_text": ""
}
lock = threading.Lock()
stop_event = threading.Event()
force_read_event = threading.Event()  # tekan 'r' untuk paksa baca

# ======================
# Util umum
# ======================
def calculate_focus_measure(gray_image):
    return cv2.Laplacian(gray_image, cv2.CV_64F).var()

def draw_text_above_bbox(img, tl_abs, br_abs, text, box_color=(255, 0, 0)):
    cv2.rectangle(img, tl_abs, br_abs, box_color, 2)
    text = text.strip()
    if len(text) > 50:
        text = text[:50] + "…"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thk = 2
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thk)
    x1, y1 = tl_abs
    y_label = y1 - 6
    if y_label - th - 4 < 0:
        y_label = y1 + th + 6
    x2 = min(x1 + tw + 6, img.shape[1] - 1)
    cv2.rectangle(img, (x1, y_label - th - 4), (x2, y_label + baseline), box_color, -1)
    cv2.putText(img, text, (x1 + 3, y_label - 2), font, font_scale, (255, 255, 255), thk, cv2.LINE_AA)

def _bbox_xyxy(bbox):
    xs = [p[0] for p in bbox]; ys = [p[1] for p in bbox]
    return min(xs), min(ys), max(xs), max(ys)

def _iou(b1, b2):
    x1,y1,x2,y2 = _bbox_xyxy(b1); X1,Y1,X2,Y2 = _bbox_xyxy(b2)
    ix1,iy1 = max(x1,X1), max(y1,Y1)
    ix2,iy2 = min(x2,X2), min(y2,Y2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    area1 = (x2-x1)*(y2-y1); area2 = (X2-X1)*(Y2-Y1)
    union = area1 + area2 - inter
    return inter/union if union > 0 else 0.0

def sort_ocr_results(results):
    if not results: return []
    vertical_tolerance = 12
    results.sort(key=lambda r: r[0][0][1])
    lines, current_line = [], []
    last_y = results[0][0][0][1]
    for res in results:
        current_y = res[0][0][1]
        if abs(current_y - last_y) < vertical_tolerance:
            current_line.append(res)
        else:
            current_line.sort(key=lambda r: r[0][0][0])
            lines.extend(current_line)
            current_line = [res]
            last_y = current_y
    current_line.sort(key=lambda r: r[0][0][0])
    lines.extend(current_line)
    return lines

def filter_ocr_results(results, roi_shape):
    H, W = roi_shape
    area_roi = H * W
    filtered = []
    for bbox, text, prob in results:
        x1,y1,x2,y2 = _bbox_xyxy(bbox)
        w = max(1, x2 - x1); h = max(1, y2 - y1)
        area = w * h

        if h < MIN_BBOX_HEIGHT:
            continue
        if area < area_roi * MIN_AREA_RATIO:
            continue

        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            continue
        if not ALLOWED_CHARS_RE.fullmatch(cleaned):
            continue
        if len(cleaned) <= 3 and prob < SHORT_TOKEN_PROB:
            continue
        if sum(ch.isalnum() for ch in cleaned) < 2:
            continue

        filtered.append((bbox, cleaned, prob))
    return filtered

def realtime_easyocr_light(reader, gray_img):
    """
    Realtime ringan (untuk tunanetra): 1x readtext, cepat.
    """
    den = cv2.medianBlur(gray_img, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    base = clahe.apply(den)
    binA = cv2.adaptiveThreshold(base, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 31, 7)
    scale = 1.3
    up = cv2.resize(binA, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    res = reader.readtext(up, paragraph=False,
                          min_size=10, text_threshold=0.30, low_text=0.25, link_threshold=0.20)
    res = [([(p[0]/scale, p[1]/scale) for p in r[0]], r[1], r[2]) for r in res if r[2] >= MIN_CONFIDENCE]
    merged = []
    for r in sorted(res, key=lambda x: x[2], reverse=True):
        if all(_iou(r[0], m[0]) < 0.5 for m in merged):
            merged.append(r)
    return merged

# ======================
# Heavy pipeline (referensi + robust)
# ======================
def grayscale(image_bgr):
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

def _ratio_white(img_bw: np.ndarray) -> float:
    return float((img_bw > 0).mean()) if img_bw.size else 0.0

def binarize_ref(gray: np.ndarray) -> np.ndarray:
    """
    Binarisasi tahan-banting:
      - CLAHE + blur
      - Otsu BINARY → bila ekstrem, Otsu INV → bila ekstrem, Adaptive Gaussian.
    Diakhir dipastikan background putih (teks hitam).
    """
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)
    g = cv2.GaussianBlur(g, (3, 3), 0)

    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    r = _ratio_white(bw)

    if r < 0.05 or r > 0.95:
        _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    r = _ratio_white(bw)
    if r < 0.05 or r > 0.95:
        h, w = gray.shape[:2]
        bs = max(31, min(151, (min(h, w) // 15) | 1))
        bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, bs, 9)

    # pastikan background putih
    if _ratio_white(bw) < 0.5:
        bw = cv2.bitwise_not(bw)
    return bw

def noise_removal_ref(image_bw: np.ndarray) -> np.ndarray:
    k3 = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(image_bw, cv2.MORPH_OPEN, k3, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k3, iterations=1)
    return cv2.medianBlur(closed, 3)

def thin_font(image_bw: np.ndarray) -> np.ndarray:
    img = cv2.bitwise_not(image_bw)
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    return cv2.bitwise_not(img)

def thick_font(image_bw: np.ndarray) -> np.ndarray:
    img = cv2.bitwise_not(image_bw)
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    return cv2.bitwise_not(img)

def get_skew_correction_angle(cvImage_bgr) -> float:
    """
    Hitung sudut KOREKSI yang benar (positif = CW).
    """
    gray = cv2.cvtColor(cvImage_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=2)
    cnts, _ = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    rect = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
    angle = rect[-1]            # (-90, 0]
    if angle < -45:             # normalisasi → (-45,45)
        angle = 90 + angle
    correction = -angle         # sudut koreksi
    return correction

def rotateImage(cvImage_bgr, angle: float):
    (h, w) = cvImage_bgr.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(cvImage_bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def deskew_ref(cvImage_bgr):
    return rotateImage(cvImage_bgr, get_skew_correction_angle(cvImage_bgr))

def preprocess_captured_reference(roi_bgr):
    """
    Jalankan urutan referensi (invert→grayscale→binarize→noise→thin/thick→deskew),
    simpan debug ke ./temp, lalu hasilkan gambar for_ocr_bgr (natural) untuk EasyOCR.
    """
    if SAVE_DEBUG:
        cv2.imwrite("temp/inverted.jpg", cv2.bitwise_not(roi_bgr))

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    if SAVE_DEBUG: cv2.imwrite("temp/gray.jpg", gray)

    im_bw = binarize_ref(gray)
    if SAVE_DEBUG:
        cv2.imwrite("temp/bw_image.jpg", im_bw)
        print(f"[DEBUG] bw white ratio={_ratio_white(im_bw):.3f}")

    no_noise = noise_removal_ref(im_bw)
    if SAVE_DEBUG:
        cv2.imwrite("temp/no_noise.jpg", no_noise)
        print(f"[DEBUG] no_noise white ratio={_ratio_white(no_noise):.3f}")

    eroded = thin_font(no_noise)
    dilated = thick_font(no_noise)
    if SAVE_DEBUG:
        cv2.imwrite("temp/eroded_image.jpg", eroded)
        cv2.imwrite("temp/dilated_image.jpg", dilated)

    # Deskew untuk referensi (disimpan); OCR sesuai flag
    deskewed_bgr = deskew_ref(roi_bgr)
    if SAVE_DEBUG:
        cv2.imwrite("temp/rotated_fixed.jpg", deskewed_bgr)

    base_for_ocr = deskewed_bgr if DESKEW_FOR_OCR else roi_bgr

    # Enhancement natural (CLAHE + unsharp) agar OCR stabil
    g2 = cv2.cvtColor(base_for_ocr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    eq = clahe.apply(g2)
    sharp = cv2.addWeighted(eq, 1.5, cv2.GaussianBlur(eq, (0,0), 1.2), -0.5, 0)
    for_ocr_bgr = cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)

    if SAVE_DEBUG:
        cv2.imwrite("temp/for_ocr.jpg", for_ocr_bgr)  # << inilah yang dimakan EasyOCR

    return for_ocr_bgr

def ocr_capture_text_reference(reader, img_for_ocr_bgr):
    res = reader.readtext(
        img_for_ocr_bgr,
        paragraph=False,
        min_size=8,
        text_threshold=0.25,
        low_text=0.20,
        link_threshold=0.20,
        contrast_ths=0.05, adjust_contrast=0.7
    )
    res = [(r[0], r[1], r[2]) for r in res if r[2] >= MIN_CONFIDENCE]

    merged = []
    for r in sorted(res, key=lambda x: x[2], reverse=True):
        if all(_iou(r[0], m[0]) < 0.5 for m in merged):
            merged.append(r)

    H, W = img_for_ocr_bgr.shape[:2]
    filtered = []
    for bbox, text, prob in merged:
        x1,y1,x2,y2 = _bbox_xyxy(bbox)
        h = max(1, y2 - y1); area = max(1, (x2 - x1) * h)
        if h < 10: 
            continue
        if area < (H*W) * 0.0003: 
            continue
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned: 
            continue
        if not ALLOWED_CHARS_RE.fullmatch(cleaned): 
            continue
        filtered.append((bbox, cleaned, prob))

    sorted_res = sort_ocr_results(filtered)
    combined_text = "\n".join([t for _, t, _ in sorted_res])

    if SAVE_DEBUG and sorted_res:
        canvas = img_for_ocr_bgr.copy()
        for (bbox, text, _) in sorted_res:
            tl, tr, br, bl = bbox
            cv2.rectangle(canvas, (int(tl[0]), int(tl[1])), (int(br[0]), int(br[1])), (0,255,255), 2)
        cv2.imwrite("temp/capture_annotated.jpg", canvas)

    return combined_text

def speak(text: str, lang='id'):
    if not text.strip():
        return
    try:
        mp3_fp = io.BytesIO()
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        pygame.mixer.music.load(mp3_fp, 'mp3')
        pygame.mixer.music.play()
    except Exception as e:
        print(f"❌ Error audio: {e}")

# ===============
# Thread kamera
# ===============
def camera_thread_func(cap):
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02)
            continue
        with lock:
            shared_data["frame"] = frame.copy()
        time.sleep(1/60)

# ===============
# Thread OCR Realtime (ringan)
# ===============
def ocr_thread_func_realtime(reader, roi_coords):
    stable_frames_count = 0
    prev_small = None

    while not stop_event.is_set():
        with lock:
            frame_to_process = shared_data["frame"].copy() if shared_data["frame"] is not None else None
        if frame_to_process is None:
            time.sleep(0.03)
            continue

        y1, y2, x1, x2 = roi_coords
        roi = frame_to_process[y1:y2, x1:x2]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Fokus
        if calculate_focus_measure(gray_roi) < FOCUS_THRESHOLD and not force_read_event.is_set():
            stable_frames_count = 0
            with lock: shared_data["status"] = "FOKUS BURAM"
            time.sleep(0.03)
            continue

        # Stabil ringan
        small_now = cv2.resize(gray_roi, (0, 0), fx=0.25, fy=0.25)
        if prev_small is not None:
            sim = 1.0 - (cv2.norm(small_now, prev_small, cv2.NORM_L1) / (small_now.size * 255.0))
            if sim >= STABILITY_THRESHOLD:
                stable_frames_count += 1
            else:
                stable_frames_count = 0
            with lock: shared_data["status"] = "MENSTABILKAN..."
        prev_small = small_now

        if stable_frames_count >= STABLE_FRAMES_REQUIREMENT or force_read_event.is_set():
            force_read_event.clear()
            with lock: shared_data["status"] = "MEMBACA..."

            start = time.perf_counter()
            merged_results = realtime_easyocr_light(reader, gray_roi)
            merged_results = filter_ocr_results(merged_results, gray_roi.shape)
            sorted_results = sort_ocr_results(merged_results)
            combined_text = "\n".join([r[1] for r in sorted_results])
            end = time.perf_counter()

            with lock:
                shared_data["results"] = merged_results
                shared_data["combined_text"] = combined_text
                shared_data["status"] = f"SIAP ({(end-start)*1000:.0f} ms)"

            stable_frames_count = 0
            time.sleep(0.03)

        time.sleep(0.02)

# ===============
# Main
# ===============
def main():
    print("Inisialisasi EasyOCR...")
    reader = easyocr.Reader(OCR_LANGUAGES, gpu=False, quantize=True, verbose=False)
    try:
        _ = reader.readtext(np.zeros((32, 128), dtype=np.uint8))
    except Exception:
        pass
    print("EasyOCR siap.")

    pygame.mixer.init()

    cap = cv2.VideoCapture(0)
    # gunakan 720p agar realtime lebih ringan
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("❌ Kamera tidak dapat diakses!")
        return

    ret, frame = cap.read()
    if not ret:
        cap.release()
        return

    H, W, _ = frame.shape
    roi_width, roi_height = int(W * ROI_W_RATIO), int(H * ROI_H_RATIO)
    roi_x1, roi_y1 = (W - roi_width) // 2, (H - roi_height) // 2
    roi_x2, roi_y2 = roi_x1 + roi_width, roi_y1 + roi_height
    roi_coords = (roi_y1, roi_y2, roi_x1, roi_x2)

    cam_thread = threading.Thread(target=camera_thread_func, args=(cap,))
    ocr_thread = threading.Thread(target=ocr_thread_func_realtime, args=(reader, roi_coords))
    cam_thread.start(); ocr_thread.start()

    print("🎥 Realtime ringan aktif.")
    print("Tombol: r=paksa baca • c=capture (heavy) • 1=lanjut (setelah capture) • 2/q=keluar")

    last_spoken_text = ""
    auto_read = True
    ui_mode = "realtime"
    decision_deadline = 0  # opsional timeout

    window_name = "Smart Reader"
    while not stop_event.is_set():
        with lock:
            frame_display = shared_data["frame"].copy() if shared_data["frame"] is not None else None
            results_display = list(shared_data["results"])
            current_text = shared_data["combined_text"]
            status = shared_data["status"]

        if frame_display is None:
            continue

        # ROI & status
        status_color = (0, 0, 255)
        if "SIAP" in status or "MEMBACA" in status: status_color = (0, 255, 0)
        elif "MENSTABILKAN" in status: status_color = (0, 255, 255)

        y1, y2, x1, x2 = roi_coords
        cv2.putText(frame_display, status, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.rectangle(frame_display, (x1, y1), (x2, y2), status_color, 2)

        # bbox realtime
        for (bbox, text, prob) in results_display:
            tl, tr, br, bl = bbox
            tl_abs = (int(tl[0]) + x1, int(tl[1]) + y1)
            br_abs = (int(br[0]) + x1, int(br[1]) + y1)
            draw_text_above_bbox(frame_display, tl_abs, br_abs, text, box_color=(255, 0, 0))

        # Auto-baca realtime
        if ui_mode == "realtime" and auto_read and current_text and current_text != last_spoken_text and not pygame.mixer.music.get_busy():
            last_spoken_text = current_text
            print(f"✅ Teks (realtime):\n---\n{current_text}\n---")
            speak(current_text, lang='id')

        cv2.imshow(window_name, frame_display)
        key = cv2.waitKey(1) & 0xFF

        # Keluar global
        if key == ord('q'):
            print("👋 Program dihentikan...")
            stop_event.set()
            break

        # MODE KEPUTUSAN (setelah capture)
        if ui_mode == "decision":
            if key == ord('1'):
                ui_mode = "realtime"
                auto_read = True
                print("🔁 Kembali ke realtime.")
                with lock:
                    shared_data["combined_text"] = ""
                    shared_data["results"] = []
            elif key == ord('2') or key == ord('q'):
                print("👋 Program dihentikan dari mode Capture.")
                stop_event.set()
                break
            elif decision_deadline and time.time() > decision_deadline:
                ui_mode = "realtime"
                auto_read = True
                print("⏳ Timeout — kembali ke realtime.")
            continue  # skip handler realtime saat decision

        # HANDLER REALTIME
        if key == ord('r'):
            force_read_event.set()   # paksa baca seketika
        elif key == ord('c'):
            # Ambil snapshot ROI & jalankan heavy pipeline (terminal + audio, tanpa window baru)
            with lock:
                snap = shared_data["frame"].copy() if shared_data["frame"] is not None else None
            if snap is not None:
                auto_read = False  # hentikan auto-baca realtime sementara
                roi_bgr = snap[y1:y2, x1:x2].copy()

                print("📸 Capture diambil. Memproses (invert→grayscale→binarize→noise→thin/thick→deskew→OCR)...")
                t0 = time.perf_counter()
                img_for_ocr_bgr = preprocess_captured_reference(roi_bgr)
                text_capture = ocr_capture_text_reference(reader, img_for_ocr_bgr)
                t1 = time.perf_counter()

                if text_capture.strip():
                    print("\n=== HASIL OCR (CAPTURE) ===\n" + text_capture)
                    print(f"⏱️ Durasi capture: {int((t1 - t0)*1000)} ms")
                    speak(text_capture, lang='id')
                else:
                    print("\n⚠️ Tidak ada teks terdeteksi pada mode capture. Coba dekatkan kamera, tambah cahaya, atau ubah sudut.")

                # Masuk decision mode (non-blocking)
                print("Tekan '1' untuk lanjut kamera, atau '2' untuk tutup.")
                ui_mode = "decision"
                decision_deadline = time.time() + 20  # timeout 20 dtk (opsional)

    cam_thread.join(); ocr_thread.join()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
