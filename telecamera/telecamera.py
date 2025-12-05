# Questo script gestisce l'acquisizione e lo streaming del feed video dalla telecamera.
# È configurato per catturare video a una risoluzione specifica e a 20 FPS,
# simulando l'input in tempo reale che verrà utilizzato per l'elaborazione AI.
# Il codice include funzionalità per avviare/fermare la registrazione, convertire
# i flussi video grezzi in formato MP4 e visualizzare un'anteprima del feed.
# Supporta l'interazione utente tramite tastiera per il controllo della registrazione.

# file per il setting della telecamera

# picam2 = picamera()                                    crea l'oggetto telecamera

# config = picam2.create_preview_configuration()         serve per l'anteprima
# config = picam2.create_still_configuration()           serve per catturare immagine (alta qualita')
# config = picam2.create_video_configuration             serve per catturare video

# picam2.configure(config)                               serve per configurare

# picam2.start()                                         avvia telecamera
# picam2.stop()                                          ferma telecamera
# picam2.close()                                         chiude la telecamera

# array = picam2.capture_array()                         salva la foto in array di numpy, cosi' non lo salva in file ma in memoria

# cv2.destroyAllWindows()                                distrugge tutte le schermate cv2
# cv2.cvtColor(array, cv2.COLOR_RGB2BGR)                 converte i colori in bgr, dato che cv2 legge cosi'
# cv2.imshow(frame_bgr)                                  mostra anteprima ('nome', frame)
# key = cv2.waitKey(1) & 0xFF                            ogni 1 ms, controlla se premuto q, 0xff, cosi' funziona per x32 e x64

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput
import cv2
import time
import os
import subprocess
from datetime import datetime
import threading

qualita_larghezza = 1920
qualita_altezza = 1080
schermo_larghezza = 768
schermo_altezza = 576

picam2 = Picamera2()

# configurazione della qualita'
config = picam2.create_preview_configuration(
    main={"size": (qualita_larghezza, qualita_altezza)}  
)
picam2.configure(config)
picam2.set_controls({"FrameRate": 20})
picam2.start()

time.sleep(2)

print(f"Qualita': {qualita_larghezza}x{qualita_altezza}")
print(f"Schermo: {schermo_larghezza}x{schermo_altezza}")
print("Q - avvia registrazione | X/Z - ferma | E/ESC - esci")

frame_time = 1 / 20
frame_count = 0
tempo_totale = 0
recording = False
encoder = H264Encoder(bitrate=25000000, repeat=True)
output = None
last_h264_path = None
last_mp4_path = None
comandi = {"start": False, "stop": False}
def ascolta_input():
    while True:
        try:
            cmd = input().strip().lower()
        except:
            break
        if cmd == 'q':
            comandi["start"] = True
        elif cmd == 'x' or cmd == 'z':
            comandi["stop"] = True
        elif cmd == 'e':
            break
threading.Thread(target=ascolta_input, daemon=True).start()

try:
    while True:
        tempo_inizio = time.time()
        array = picam2.capture_array()
        frame_bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        frame_grande = cv2.resize(
            frame_bgr,
            (schermo_larghezza, schermo_altezza),
            interpolation=cv2.INTER_NEAREST
        )
        cv2.imshow(f'Qualita {qualita_larghezza}x{qualita_altezza} - Schermo {schermo_larghezza}x{schermo_altezza}', frame_grande)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            comandi["start"] = True
        elif key == ord('x') or key == ord('z'):
            comandi["stop"] = True
        elif key == 27 or key == ord('e'):
            break
        if comandi["start"] and not recording:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_dir = os.path.dirname(os.path.abspath(__file__))
            last_h264_path = os.path.join(base_dir, f"video_{ts}.h264")
            last_mp4_path = os.path.join(base_dir, f"video_{ts}.mp4")
            output = FileOutput(last_h264_path)
            picam2.start_recording(encoder, output)
            recording = True
            comandi["start"] = False
            print(f"Registrazione avviata: {last_h264_path}")
        if comandi["stop"] and recording:
            picam2.stop_recording()
            os.sync()
            # Conversione in MP4 con ffmpeg (contenitore compatibile Windows)
            if last_h264_path and last_mp4_path:
                try:
                    ret = subprocess.run([
                        "ffmpeg", "-y", "-f", "h264", "-r", "20", "-i", last_h264_path, "-c:v", "copy", "-movflags", "+faststart", last_mp4_path
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                    if (not os.path.exists(last_mp4_path)) or (os.path.getsize(last_mp4_path) < 1024) or ret.returncode != 0:
                        subprocess.run([
                            "ffmpeg", "-y", "-f", "h264", "-r", "20", "-i", last_h264_path, "-c:v", "libx264", "-preset", "slow", "-crf", "14", "-pix_fmt", "yuv420p", "-movflags", "+faststart", last_mp4_path
                        ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                    print(f"File MP4 pronto: {last_mp4_path}")
                    try:
                        os.remove(last_h264_path)
                    except:
                        pass
                except Exception as e:
                    print(f"Errore conversione MP4: {e}")
            recording = False
            output = None
            comandi["stop"] = False
            last_h264_path = None
            last_mp4_path = None
            print("Registrazione terminata")
        tempo_trascorso = time.time() - tempo_inizio
        delay = max(0, frame_time - tempo_trascorso)
        time.sleep(delay)
        frame_count += 1
        tempo_totale += tempo_trascorso
        print(f"Frame: {frame_count}, Tempo impiegato: {tempo_trascorso:.4f} secondi, Somma cumulativa: {tempo_totale:.4f} secondi")
        if frame_count == 20:
            frame_count = 0
            tempo_totale = 0

finally:
    if recording:
        picam2.stop_recording()
    os.sync()
    cv2.destroyAllWindows()
    picam2.stop()
    picam2.close()