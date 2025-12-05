import tensorflow as tf
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # Imposta il backend prima di importare pyplot
import matplotlib.pyplot as plt
import random

# --- Costanti Globali ---
IMG_HEIGHT = 96
IMG_WIDTH = 128
BATCH_SIZE = 32

# Percorsi
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, 'modello_linea')
MODEL_PATH = os.path.join(MODEL_DIR, 'line_detection_model.h5')
CLASSES_PATH = os.path.join(MODEL_DIR, 'class_names.txt')
DATASET_DIR = os.path.join(SCRIPT_DIR, 'dataset')

def clear_screen():
    """Pulisce lo schermo della console."""
    os.system('cls' if os.name == 'nt' else 'clear')

def load_model_and_classes():
    """Carica il modello e i nomi delle classi."""
    print("--- Caricamento modello e classi ---")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Errore: Modello non trovato in '{MODEL_PATH}'.")
        print("Assicurati di aver eseguito prima lo script di training (train.py).")
        return None, None

    if not os.path.exists(CLASSES_PATH):
        print(f"❌ Errore: File delle classi non trovato in '{CLASSES_PATH}'.")
        return None, None

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(CLASSES_PATH, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        print(f"✓ Modello e classi ({class_names}) caricati con successo.")
        return model, class_names
    except Exception as e:
        print(f"❌ Errore durante il caricamento: {e}")
        return None, None

def evaluate_model(model, val_ds):
    """Valuta il modello sul dataset di validazione."""
    print("\n--- Valutazione del modello sul set di validazione ---")
    loss, accuracy = model.evaluate(val_ds)
    print(f"  - Accuratezza sul set di validazione: {accuracy * 100:.2f}%")
    print(f"  - Loss sul set di validazione: {loss:.4f}")

def interactive_test(model, class_names, val_ds):
    """Permette di testare immagini casuali dal set di validazione."""
    
    # Converte il dataset in una lista di tuple (immagine, etichetta) per un accesso casuale facile
    print("\nPreparazione per il test interattivo...")
    validation_data = list(val_ds.unbatch().as_numpy_iterator())
    print("✓ Dati di validazione pronti.")

    while True:
        print("\n--- Test Interattivo ---")
        
        # Scegli un'immagine e la sua etichetta a caso
        img, true_label_index = random.choice(validation_data)
        true_label_name = class_names[int(true_label_index)]

        # L'immagine è già un array numpy, basta aggiungere la dimensione del batch
        img_array_expanded = np.expand_dims(img, axis=0)
        
        # Predizione
        prediction_score = model.predict(img_array_expanded)[0][0]
        predicted_class_index = 1 if prediction_score > 0.5 else 0
        predicted_class_name = class_names[predicted_class_index]

        # Mostra i risultati
        plt.imshow(img.astype("uint8"))
        plt.title(f"Reale: {true_label_name} | Predetto: {predicted_class_name} (Score: {prediction_score:.2f})")
        plt.axis("off")
        plt.show()

        # Chiedi all'utente se vuole continuare
        scelta = input("\nVuoi testare un'altra immagine? (s/n): ").strip().lower()
        if scelta != 's':
            break

def main():
    """Funzione principale per orchestrare il test."""
    clear_screen()
    print(f"\n{'='*50}\nSCRIPT DI TEST PER RICONOSCIMENTO LINEA\n{'='*50}")

    model, class_names = load_model_and_classes()
    if not model or not class_names:
        return

    model.summary()

    # Carica il dataset di validazione una sola volta
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,  # DEVE essere lo stesso seed del training
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode='binary'
    ).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    evaluate_model(model, val_ds)
    
    interactive_test(model, class_names, val_ds)
    
    print("\nArrivederci!")

if __name__ == "__main__":
    main()