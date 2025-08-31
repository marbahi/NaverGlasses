import cv2
import easyocr
from gtts import gTTS
from playsound import playsound
import os
import time

reader = easyocr.Reader(['id', 'en'])

AUDIO_FILE = "temp_audio.mp3"

def main():
    cap = cv2.VideoCapture(0)
    
    last_detected_text = ""

    print("🎥 Kamera siap. Arahkan ke teks yang ingin dibaca.")
    print("Tekan 'q' pada jendela kamera untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Gagal mengakses kamera.")
            break

        cv2.imshow("Smart Reader - Tekan 'q' untuk keluar", frame)

        results = reader.readtext(frame, paragraph=True)

        
        if results:
            
            current_text = " ".join([res[1] for res in results])
            
            if current_text and current_text != last_detected_text:
                last_detected_text = current_text
                print(f"✅ Teks terdeteksi: {current_text}")

                try:
                    tts = gTTS(text=current_text, lang='id', slow=False)
                    tts.save(AUDIO_FILE)

                    print("🔊 Membacakan teks...")
                    playsound(AUDIO_FILE)
                    
                    time.sleep(1) 

                except Exception as e:
                    print(f"❌ Terjadi error saat memproses audio: {e}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 Program dihentikan.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if os.path.exists(AUDIO_FILE):
        os.remove(AUDIO_FILE)

if __name__ == "__main__":
    main()