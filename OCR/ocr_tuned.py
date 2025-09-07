import cv2
import easyocr
from gtts import gTTS
import time
import threading
import numpy as np
import pygame
import io
import re

# ======================
# Konfigurasi & ambang
# ======================
OCR_LANGUAGES = ['id', 'en']
MIN_CONFIDENCE = 0.30

# Fokus & stabilisasi
FOCUS_THRESHOLD = 50.0
STABILITY_THRESHOLD = 0.95
STABLE_FRAMES_REQUIREMENT = 2   # dipercepat

# ROI (lebih kecil = lebih cepat)
ROI_W_RATIO = 0.80
ROI_H_RATIO = 0.65

# Filter hasil OCR (anti-sampah)
MIN_BBOX_HEIGHT = 12            # px absolut di ROI (direndahkan agar paragraf kecil lolos)
MIN_AREA_RATIO = 0.0006         # proporsi area bbox/ROI
SHORT_TOKEN_PROB = 0.55         # ambang prob. utk token sangat pendek (<=3)
ALLOWED_CHARS_RE = re.compile(r"[A-Za-zÀ-ÿ0-9 ,.'‘’“”\-–—():/?]+")

# ======================
# State bersama
# ======================
shared_data = {
    "frame": None,
    "status": "MENSTABILKAN...",
    "results": [],          # list[(bbox, text, prob)]
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

def sort_ocr_results(results):
    vertical_tolerance = 10
    results.sort(key=lambda r: r[0][0][1])
    lines, current_line = [], []
    if not results:
        return []
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

def draw_text_above_bbox(img, tl_abs, br_abs, text, box_color=(255, 0, 0)):
    cv2.rectangle(img, tl_abs, br_abs, box_color, 2)
    text = text.strip()
    if len(text) > 40:
        text = text[:40] + "…"
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

def _scale_bbox(bbox, sx, sy):
    return [(p[0]*sx, p[1]*sy) for p in bbox]

def _translate_bbox(bbox, dx, dy):
    return [(p[0]+dx, p[1]+dy) for p in bbox]

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

def _avg_height(res):
    if not res: return 0
    hs = []
    for b,_,_ in res:
        ys = [p[1] for p in b]
        hs.append(max(ys) - min(ys))
    return sum(hs)/len(hs)

def filter_ocr_results(results, roi_shape):
    """Singkirkan hasil kecil/sampah/karakter aneh."""
    H, W = roi_shape
    area_roi = H * W
    filtered = []
    for bbox, text, prob in results:
        x1,y1,x2,y2 = _bbox_xyxy(bbox)
        w = max(1, x2 - x1); h = max(1, y2 - y1)
        area = w * h

        # 1) ukuran minimum
        if h < MIN_BBOX_HEIGHT: 
            continue
        if area < area_roi * MIN_AREA_RATIO: 
            continue

        # 2) kebersihan karakter
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            continue
        if not ALLOWED_CHARS_RE.fullmatch(cleaned):
            continue

        # 3) token sangat pendek → perlu confidence lebih tinggi
        if len(cleaned) <= 3 and prob < SHORT_TOKEN_PROB:
            continue

        # 4) harus punya >=2 karakter alfanumerik total
        if sum(ch.isalnum() for ch in cleaned) < 2:
            continue

        filtered.append((bbox, cleaned, prob))
    return filtered


# ======================
# OCR 2-pass + Pass-C
# ======================
def run_easyocr_multiscale(reader, gray_img):
    """Pass-A cepat + Pass-B untuk kecil. Mengembalikan bbox di koordinat gray_img."""
    den = cv2.medianBlur(gray_img, 3)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    base = clahe.apply(den)

    results_all = []

    # Pass-A (cepat)
    binA = cv2.adaptiveThreshold(base, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 31, 7)
    fxA = fyA = 1.6
    big = cv2.resize(binA, None, fx=fxA, fy=fyA, interpolation=cv2.INTER_CUBIC)
    ra = reader.readtext(big, paragraph=False,
                         min_size=8, text_threshold=0.30, low_text=0.20, link_threshold=0.20)
    ra = [(_scale_bbox(r[0], 1.0/fxA, 1.0/fyA), r[1], r[2]) for r in ra if r[2] >= MIN_CONFIDENCE]
    results_all.extend(ra)

    # Jika teks sudah besar, cukup Pass-A
    if _avg_height(ra) >= 18:
        merged = []
        for r in sorted(results_all, key=lambda x: x[2], reverse=True):
            if all(_iou(r[0], m[0]) < 0.5 for m in merged):
                merged.append(r)
        return merged

    # Pass-B (khusus kecil, seluruh area)
    sharp = cv2.addWeighted(base, 1.5, cv2.GaussianBlur(base, (0, 0), 1.2), -0.5, 0)
    bh = cv2.morphologyEx(sharp, cv2.MORPH_BLACKHAT, np.ones((3, 3), np.uint8))
    _, binB = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binB = cv2.dilate(binB, np.ones((2, 2), np.uint8), 1)
    fxB = fyB = 2.4
    small = cv2.resize(binB, None, fx=fxB, fy=fyB, interpolation=cv2.INTER_CUBIC)
    rb = reader.readtext(
        small, paragraph=False,
        min_size=5, text_threshold=0.25, low_text=0.15, link_threshold=0.20
    )
    rb = [(_scale_bbox(r[0], 1.0/fxB, 1.0/fyB), r[1], r[2]) for r in rb if r[2] >= MIN_CONFIDENCE]

    # Merge awal
    merged = []
    for r in sorted(results_all + rb, key=lambda x: x[2], reverse=True):
        if all(_iou(r[0], m[0]) < 0.5 for m in merged):
            merged.append(r)
    return merged

def extract_text_stripes(gray):
    """Deteksi strip baris lebar untuk paragraf kecil (Pass-C)."""
    H, W = gray.shape
    g = cv2.GaussianBlur(gray, (3,3), 0)
    grad = cv2.Sobel(g, cv2.CV_16S, 1, 0, ksize=3)
    grad = cv2.convertScaleAbs(grad)
    _, th = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # hubungkan horizontal → blok baris
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    conn = cv2.morphologyEx(th, cv2.MORPH_CLOSE, h_kernel, iterations=1)
    conn = cv2.dilate(conn, v_kernel, 1)
    cnts, _ = cv2.findContours(conn, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    stripes = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        # baris harus lebar dan tidak tinggi
        if w < 0.45*W: 
            continue
        if h < 12 or h > 80:
            continue
        # hindari heading raksasa (terlalu tinggi)
        stripes.append((x,y,w,h))
    stripes.sort(key=lambda r: r[1])
    # batasi jumlah stripe buat performa
    return stripes[:12]

def read_small_stripes(reader, gray):
    """Baca hanya pada stripe baris → cepat & tajam."""
    res = []
    stripes = extract_text_stripes(gray)
    if not stripes:
        return res
    # Prepro dasar
    den = cv2.medianBlur(gray, 3)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    base = clahe.apply(den)
    for (x,y,w,h) in stripes:
        crop = base[y:y+h, x:x+w]
        sharp = cv2.addWeighted(crop, 1.5, cv2.GaussianBlur(crop,(0,0),1.2), -0.5, 0)
        bh = cv2.morphologyEx(sharp, cv2.MORPH_BLACKHAT, np.ones((3,3), np.uint8))
        _, binC = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binC = cv2.dilate(binC, np.ones((2,2), np.uint8), 1)
        scale = 3.0
        up = cv2.resize(binC, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        rc = reader.readtext(
            up, paragraph=False,
            min_size=4, text_threshold=0.22, low_text=0.12, link_threshold=0.18
        )
        for b, t, p in rc:
            if p < MIN_CONFIDENCE:
                continue
            b = _scale_bbox(b, 1.0/scale, 1.0/scale)
            b = _translate_bbox(b, x, y)  # kembali ke koordinat ROI
            res.append((b, t, p))
    # de-dup
    merged = []
    for r in sorted(res, key=lambda x: x[2], reverse=True):
        if all(_iou(r[0], m[0]) < 0.5 for m in merged):
            merged.append(r)
    return merged


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
# Thread OCR
# ===============
def ocr_thread_func(reader, roi_coords):
    stable_frames_count = 0
    prev_gray_roi = None

    while not stop_event.is_set():
        with lock:
            frame_to_process = shared_data["frame"].copy() if shared_data["frame"] is not None else None
        if frame_to_process is None:
            time.sleep(0.05)
            continue

        y1, y2, x1, x2 = roi_coords
        roi = frame_to_process[y1:y2, x1:x2]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Fokus
        if calculate_focus_measure(gray_roi) < FOCUS_THRESHOLD and not force_read_event.is_set():
            stable_frames_count = 0
            with lock: shared_data["status"] = "FOKUS BURAM"
            time.sleep(0.05)
            continue

        # Stabil ringan
        small_now = cv2.resize(gray_roi, (0, 0), fx=0.25, fy=0.25)
        if prev_gray_roi is not None:
            small_prev = cv2.resize(prev_gray_roi, (small_now.shape[1], small_now.shape[0]))
            sim = 1.0 - (cv2.norm(small_now, small_prev, cv2.NORM_L1) / (small_now.size * 255.0))
            if sim >= STABILITY_THRESHOLD:
                stable_frames_count += 1
            else:
                stable_frames_count = 0
            with lock: shared_data["status"] = "MENSTABILKAN..."
        prev_gray_roi = gray_roi

        if stable_frames_count >= STABLE_FRAMES_REQUIREMENT or force_read_event.is_set():
            force_read_event.clear()
            with lock: shared_data["status"] = "MEMBACA..."

            # (Opsional) Koreksi perspektif ringan
            transformed_roi = roi
            blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)
            thresh_persp = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY_INV, 21, 10)
            contours, _ = cv2.findContours(thresh_persp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                page_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(page_contour) > 5000:
                    peri = cv2.arcLength(page_contour, True)
                    approx = cv2.approxPolyDP(page_contour, 0.02 * peri, True)
                    if len(approx) == 4:
                        rect = np.float32([p[0] for p in approx])
                        (tl, tr, br, bl) = rect
                        widthA = np.hypot(br[0]-bl[0], br[1]-bl[1])
                        widthB = np.hypot(tr[0]-tl[0], tr[1]-tl[1])
                        heightA = np.hypot(tr[0]-br[0], tr[1]-br[1])
                        heightB = np.hypot(tl[0]-bl[0], tl[1]-bl[1])
                        maxWidth = max(int(widthA), int(widthB))
                        maxHeight = max(int(heightA), int(heightB))
                        if maxWidth > 0 and maxHeight > 0:
                            dst = np.array([[0, 0], [maxWidth-1, 0], [maxWidth-1, maxHeight-1], [0, maxHeight-1]], dtype="float32")
                            M = cv2.getPerspectiveTransform(rect, dst)
                            transformed_roi = cv2.warpPerspective(roi, M, (maxWidth, maxHeight))

            # OCR: Pass-A/B (umum)
            final_gray = cv2.cvtColor(transformed_roi, cv2.COLOR_BGR2GRAY)
            merged_results = run_easyocr_multiscale(reader, final_gray)

            # Tambah Pass-C (hanya stripe baris) jika hasil sedikit (indikasi paragraf kecil)
            if len(merged_results) < 6:
                small_line_results = read_small_stripes(reader, final_gray)
                merged_results = merged_results + small_line_results
                # de-dup lagi
                dedup = []
                for r in sorted(merged_results, key=lambda x: x[2], reverse=True):
                    if all(_iou(r[0], m[0]) < 0.5 for m in dedup):
                        dedup.append(r)
                merged_results = dedup

            # FILTER hasil
            merged_results = filter_ocr_results(merged_results, final_gray.shape)

            # Gabung teks baris
            sorted_results = sort_ocr_results(merged_results)
            combined_text = "\n".join([r[1] for r in sorted_results])

            with lock:
                shared_data["results"] = merged_results
                shared_data["combined_text"] = combined_text
                shared_data["status"] = "SIAP"

            stable_frames_count = 0
            time.sleep(0.05)  # cooldown singkat

        time.sleep(0.05)


# ===============
# Main
# ===============
def main():
    print("Inisialisasi EasyOCR...")
    # quantize=True agar CPU lebih cepat
    reader = easyocr.Reader(OCR_LANGUAGES, gpu=False, quantize=True, verbose=False)

    # WARM-UP: panggil sekali agar inferensi pertama tak lama
    try:
        _ = reader.readtext(np.zeros((32, 128), dtype=np.uint8))
    except Exception:
        pass
    print("EasyOCR siap.")

    pygame.mixer.init()

    cap = cv2.VideoCapture(0)
    # coba 1080p; jika webcam tak sanggup, akan fallback ke nilai driver
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
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
    ocr_thread = threading.Thread(target=ocr_thread_func, args=(reader, roi_coords))
    cam_thread.start(); ocr_thread.start()

    print("🎥 Kamera siap. Arahkan area fokus ke teks dan tahan hingga stabil.")
    print("Tip: tekan tombol 'r' untuk paksa baca segera.")

    last_spoken_text = ""

    while not stop_event.is_set():
        with lock:
            frame_display = shared_data["frame"].copy() if shared_data["frame"] is not None else None
            results_display = list(shared_data["results"])
            current_text = shared_data["combined_text"]
            status = shared_data["status"]

        if frame_display is None:
            continue

        status_color = (0, 0, 255)
        if "SIAP" in status or "MEMBACA" in status: status_color = (0, 255, 0)
        elif "MENSTABILKAN" in status: status_color = (0, 255, 255)

        cv2.putText(frame_display, status, (roi_x1, roi_y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.rectangle(frame_display, (roi_x1, roi_y1), (roi_x2, roi_y2), status_color, 2)

        # Gambar bbox + label
        for (bbox, text, prob) in results_display:
            tl, tr, br, bl = bbox
            tl_abs = (int(tl[0]) + roi_x1, int(tl[1]) + roi_y1)
            br_abs = (int(br[0]) + roi_x1, int(br[1]) + roi_y1)
            draw_text_above_bbox(frame_display, tl_abs, br_abs, text, box_color=(255, 0, 0))

        # TTS
        if current_text and current_text != last_spoken_text and not pygame.mixer.music.get_busy():
            last_spoken_text = current_text
            print(f"✅ Teks terdeteksi:\n---\n{current_text}\n---")
            try:
                mp3_fp = io.BytesIO()
                tts = gTTS(text=current_text, lang='id', slow=False)
                tts.write_to_fp(mp3_fp)
                mp3_fp.seek(0)
                print("🔊 Membacakan teks...")
                pygame.mixer.music.load(mp3_fp, 'mp3')
                pygame.mixer.music.play()
            except Exception as e:
                print(f"❌ Error audio: {e}")
                last_spoken_text = ""

        cv2.imshow("Smart Reader (Optimized) - Tekan 'q'", frame_display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("👋 Program dihentikan...")
            stop_event.set()
            break
        elif key == ord('r'):
            # paksa baca seketika
            force_read_event.set()

    cam_thread.join(); ocr_thread.join()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
