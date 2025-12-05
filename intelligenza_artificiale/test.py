import tensorflow as tf
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # Imposta il backend prima di importare pyplot
import matplotlib.pyplot as plt
import random

# Controlla la dipendenza da SciPy
try:
    from scipy.ndimage import rotate
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# --- Costanti Globali ---
IMG_HEIGHT = 96
IMG_WIDTH = 128
BATCH_SIZE = 32

# Percorsi
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SESSIONS_DIR = os.path.join(SCRIPT_DIR, 'training_sessions')
DATASET_DIR = os.path.join(SCRIPT_DIR, 'dataset')


def clear_screen():
    """Pulisce lo schermo della console."""
    os.system('cls' if os.name == 'nt' else 'clear')

def select_session_dir():
    """Elenca le sessioni di training disponibili e chiede all'utente di sceglierne una."""
    if not os.path.isdir(SESSIONS_DIR):
        print(f"❌ Errore: La cartella delle sessioni '{SESSIONS_DIR}' non è stata trovata.")
        return None
    
    session_dirs = [
        d for d in os.listdir(SESSIONS_DIR)
        if os.path.isdir(os.path.join(SESSIONS_DIR, d)) and d.startswith("run_")
    ]
    
    if not session_dirs:
        print(f"❌ Errore: Nessuna sessione di training trovata in '{SESSIONS_DIR}'.")
        return None
        
    # Ordina le sessioni dalla più recente alla meno recente
    session_dirs.sort(key=lambda d: os.path.getmtime(os.path.join(SESSIONS_DIR, d)), reverse=True)
    
    print("\n--- Seleziona una sessione di training da testare ---")
    for i, dir_name in enumerate(session_dirs):
        label = "(più recente)" if i == 0 else ""
        print(f"  {i+1}: {dir_name} {label}")
        
    while True:
        try:
            scelta = input(f"\nScegli un numero (1-{len(session_dirs)}) o premi Invio per usare la più recente: ")
            if not scelta:
                return os.path.join(SESSIONS_DIR, session_dirs[0])
            
            scelta_idx = int(scelta) - 1
            if 0 <= scelta_idx < len(session_dirs):
                return os.path.join(SESSIONS_DIR, session_dirs[scelta_idx])
            else:
                print("Scelta non valida. Riprova.")
        except ValueError:
            print("Inserisci un numero valido. Riprova.")

def load_model_and_classes(session_dir):
    """Carica il modello e i nomi delle classi dalla cartella di sessione fornita."""
    print(f"\n--- Caricamento modello e classi da: {os.path.basename(session_dir)} ---")

    model_path_keras = os.path.join(session_dir, 'line_detection_model.keras')
    model_path_h5 = os.path.join(session_dir, 'line_detection_model.h5')
    classes_path = os.path.join(session_dir, 'class_names.txt')

    model_path_to_load = None
    if os.path.exists(model_path_keras):
        model_path_to_load = model_path_keras
    elif os.path.exists(model_path_h5):
        model_path_to_load = model_path_h5
    
    if not model_path_to_load:
        print(f"❌ Errore: Nessun file modello (.keras o .h5) trovato nella cartella di sessione.")
        return None, None

    if not os.path.exists(classes_path):
        print(f"❌ Errore: File delle classi non trovato in '{classes_path}'.")
        return None, None

    try:
        model = tf.keras.models.load_model(model_path_to_load)
        with open(classes_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        print(f"✓ Modello e classi ({class_names}) caricati con successo.")
        return model, class_names
    except Exception as e:
        print(f"❌ Errore durante il caricamento del modello '{os.path.basename(model_path_to_load)}': {e}")
        return None, None

def evaluate_model(model, val_ds):
    """Valuta il modello sul dataset di validazione."""
    print("\n--- Valutazione del modello sul set di validazione ---")
    loss, accuracy = model.evaluate(val_ds)
    print(f"  - Accuratezza sul set di validazione: {accuracy * 100:.2f}%")
    print(f"  - Loss sul set di validazione: {loss:.4f}")

def preprocess_dataset_item(image, label):
    """Applica il preprocessing specifico di MobileNetV3Small all\'immagine."""
    image = tf.keras.applications.mobilenet_v3.preprocess_input(image)
    return image, label

# --- Funzioni di Augmentation Personalizzate ---
def apply_random_rotation(image_np, factor=0.25):
    """Applica una rotazione casuale usando SciPy e restituisce l\'angolo in gradi."""
    angle_deg = np.random.uniform(-factor * 180, factor * 180)
    rotated_image = rotate(image_np, angle_deg, reshape=False, mode='reflect')
    return rotated_image, angle_deg

def apply_random_zoom(image_tensor, factor=0.35):
    """Applica uno zoom casuale all\'immagine e restituisce il fattore di zoom."""
    zoom_val = tf.random.uniform([], 1 - factor, 1 + factor)
    h, w = tf.shape(image_tensor)[0], tf.shape(image_tensor)[1]
    new_h = tf.cast(tf.cast(h, tf.float32) / zoom_val, tf.int32)
    new_w = tf.cast(tf.cast(w, tf.float32) / zoom_val, tf.int32)
    image_float = tf.cast(image_tensor, tf.float32)
    resized_image = tf.image.resize(image_float, [new_h, new_w], method=tf.image.ResizeMethod.BILINEAR)
    
    if new_h < h: # Zoom out, quindi facciamo padding
        pad_top = (h - new_h) // 2
        pad_bottom = h - new_h - pad_top
        pad_left = (w - new_w) // 2
        pad_right = w - new_w - pad_left
        zoomed_image = tf.pad(resized_image, [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')
    else: # Zoom in, quindi facciamo crop
        crop_h = (new_h - h) // 2
        crop_w = (new_w - w) // 2
        zoomed_image = tf.image.crop_to_bounding_box(resized_image, crop_h, crop_w, h, w)
        
    return tf.cast(zoomed_image, tf.uint8), zoom_val

def apply_random_brightness(image_tensor, factor=0.25):
    """Applica una variazione casuale di luminosità e restituisce il delta."""
    delta = tf.random.uniform([], -factor, factor)
    image_float = tf.cast(image_tensor, tf.float32)
    bright_image = tf.image.adjust_brightness(image_float, delta * 255)
    bright_image = tf.clip_by_value(bright_image, 0, 255)
    return tf.cast(bright_image, tf.uint8), delta

def interactive_test(model, class_names, val_ds):
    """Permette di testare immagini casuali in modo interattivo."""
    print("\nPreparazione per il test interattivo...")
    raw_val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR, validation_split=0.2, subset="validation", seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=1, label_mode='binary'
    )
    linea_images, no_linea_images = [], []
    linea_index, no_linea_index = class_names.index('linea'), class_names.index('no_linea')
    for img, label in raw_val_ds.unbatch().as_numpy_iterator():
        if int(label[0]) == linea_index:
            linea_images.append((img, label))
        else:
            no_linea_images.append((img, label))
    print(f"✓ Dati di validazione pronti: {len(linea_images)} 'linea', {len(no_linea_images)} 'no_linea'.")

    while True:
        print("\n--- Test Interattivo ---")
        scelta = input("Cosa vuoi testare? (1: linea / 2: no_linea / s: casuale / n: esci): ").strip().lower()
        if scelta == 'n': break
        img_to_test_data = None
        if scelta == '1' and linea_images: img_to_test_data = random.choice(linea_images)
        elif scelta == '2' and no_linea_images: img_to_test_data = random.choice(no_linea_images)
        elif scelta in ('s', ''): img_to_test_data = random.choice(linea_images + no_linea_images)
        else:
            print("Scelta non valida o categoria vuota. Riprova.")
            continue
            
        original_img_np, true_label_index_raw = img_to_test_data
        true_label_name = class_names[int(true_label_index_raw[0])]

        # Augmentation: 1. Rotazione (NumPy) -> 2. Zoom/Luminosità (TensorFlow)
        rotated_img_np, rotation_angle = apply_random_rotation(original_img_np)
        img_tensor = tf.convert_to_tensor(rotated_img_np)
        augmented_img_tensor, zoom_factor = apply_random_zoom(img_tensor)
        augmented_img_tensor, brightness_delta = apply_random_brightness(augmented_img_tensor)
        augmented_img_np = augmented_img_tensor.numpy()

        img_for_prediction = tf.keras.applications.mobilenet_v3.preprocess_input(augmented_img_np)
        prediction_score = model.predict(np.expand_dims(img_for_prediction, axis=0))[0][0]
        predicted_class_name = class_names[no_linea_index if prediction_score > 0.5 else linea_index]

        aug_info = f"Rot: {rotation_angle:.1f}° | Zoom: {zoom_factor:.2f}x | Bright: {brightness_delta:.2f}"
        plt.imshow(augmented_img_np.astype("uint8"))
        plt.title(f"Reale: {true_label_name} | Predetto: {predicted_class_name} (Score: {prediction_score:.2f})\n{aug_info}", fontsize=10)
        plt.axis("off")
        plt.show()

def main():
    """Funzione principale per orchestrare il test."""
    clear_screen()
    print(f"\n{'='*50}\nSCRIPT DI TEST PER RICONOSCIMENTO LINEA\n{'='*50}")

    if not SCIPY_AVAILABLE:
        print("❌ Errore: La libreria 'SciPy' non è installata ma è necessaria per la rotazione delle immagini.")
        print("✅ Per favore, installala eseguendo questo comando e poi riavvia lo script:")
        print("   pip install scipy")
        sys.exit(1)

    # Chiede all'utente quale sessione testare
    chosen_session_dir = select_session_dir()
    if not chosen_session_dir:
        return # Esce se non ci sono sessioni

    model, class_names = load_model_and_classes(chosen_session_dir)
    if not model or not class_names:
        return

    model.summary()

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR, validation_split=0.2, subset="validation", seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, label_mode='binary'
    ).map(preprocess_dataset_item, num_parallel_calls=tf.data.AUTOTUNE).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    evaluate_model(model, val_ds)
    interactive_test(model, class_names, val_ds)
    
    print("\nArrivederci!")

if __name__ == "__main__":
    main()
