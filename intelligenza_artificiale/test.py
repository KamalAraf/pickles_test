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

CLASSES_PATH = os.path.join(MODEL_DIR, 'class_names.txt')
DATASET_DIR = os.path.join(SCRIPT_DIR, 'dataset')

def clear_screen():
    """Pulisce lo schermo della console."""
    os.system('cls' if os.name == 'nt' else 'clear')

def load_model_and_classes():
    """Carica il modello e i nomi delle classi, cercando prima il formato .keras e poi .h5."""
    print("--- Caricamento modello e classi ---")
    
    model_path_keras = os.path.join(MODEL_DIR, 'line_detection_model.keras')
    model_path_h5 = os.path.join(MODEL_DIR, 'line_detection_model.h5')
    
    model_path_to_load = None
    if os.path.exists(model_path_keras):
        model_path_to_load = model_path_keras
        print(f"✓ Trovato modello in formato .keras: {model_path_to_load}")
    elif os.path.exists(model_path_h5):
        model_path_to_load = model_path_h5
        print(f"✓ Trovato modello in formato .h5: {model_path_to_load}")
    else:
        print(f"❌ Errore: Nessun modello trovato. Controllati i percorsi:")
        print(f"  - {model_path_keras}")
        print(f"  - {model_path_h5}")
        print("Assicurati di aver eseguito prima lo script di training (train.py).")
        return None, None

    if not os.path.exists(CLASSES_PATH):
        print(f"❌ Errore: File delle classi non trovato in '{CLASSES_PATH}'.")
        return None, None

    try:
        model = tf.keras.models.load_model(model_path_to_load)
        with open(CLASSES_PATH, 'r') as f:
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
    """Applica il preprocessing specifico di MobileNetV3Small all'immagine."""
    image = tf.keras.applications.mobilenet_v3.preprocess_input(image)
    return image, label

def interactive_test(model, class_names, val_ds):
    """Permette di testare immagini casuali in modo interattivo, scegliendo la categoria e applicando augmentation."""
    
    print("\nPreparazione per il test interattivo...")
    raw_val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=1,
        label_mode='binary'
    )
    
    # Separa le immagini di validazione in due liste per categoria
    linea_images = []
    no_linea_images = []
    # Keras assegna le etichette in ordine alfabetico: linea (0), no_linea (1)
    linea_index = class_names.index('linea')
    no_linea_index = class_names.index('no_linea')

    for img, label in raw_val_ds.unbatch().as_numpy_iterator():
        if int(label) == linea_index:
            linea_images.append((img, label))
        else:
            no_linea_images.append((img, label))

    print(f"✓ Dati di validazione pronti: {len(linea_images)} immagini 'linea', {len(no_linea_images)} immagini 'no_linea'.")

    # Definisci il layer di augmentation qui per applicarlo alle immagini di test
    test_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.3), # Stessi fattori di train.py
        tf.keras.layers.RandomZoom(0.3),     # Stessi fattori di train.py
        tf.keras.layers.RandomBrightness(factor=0.2), # Stesso fattore di train.py
    ], name='test_augmentation')

    while True:
        print("\n--- Test Interattivo ---")
        scelta = input("Cosa vuoi testare? (1: linea / 2: no_linea / s: casuale / n: esci): ").strip().lower()

        img_to_test_data = None
        if scelta == '1':
            if not linea_images:
                print("❌ Nessuna immagine 'linea' nel set di validazione.")
                continue
            img_to_test_data = random.choice(linea_images)
        elif scelta == '2':
            if not no_linea_images:
                print("❌ Nessuna immagine 'no_linea' nel set di validazione.")
                continue
            img_to_test_data = random.choice(no_linea_images)
        elif scelta == 's' or scelta == '':
            all_images = linea_images + no_linea_images
            if not all_images:
                print("❌ Nessuna immagine nel set di validazione.")
                continue
            img_to_test_data = random.choice(all_images)
        elif scelta == 'n':
            break
        else:
            print("Scelta non valida. Riprova.")
            continue
            
        original_img_np, true_label_index_raw = img_to_test_data
        true_label_name = class_names[int(true_label_index_raw)]

        # Applica l'augmentation all'immagine originale per visualizzazione e predizione
        # Converti a tf.Tensor prima di applicare l'augmentation
        augmented_img_tensor = test_augmentation(tf.expand_dims(original_img_np, axis=0), training=False)
        augmented_img_np = augmented_img_tensor[0].numpy()

        # Applica il preprocessing specifico di MobileNetV3Small all'immagine aumentata
        img_for_prediction = tf.keras.applications.mobilenet_v3.preprocess_input(augmented_img_np)
        img_array_expanded = np.expand_dims(img_for_prediction, axis=0)
        
        # Predizione
        prediction_score = model.predict(img_array_expanded)[0][0]
        predicted_class_index = no_linea_index if prediction_score > 0.5 else linea_index
        predicted_class_name = class_names[predicted_class_index]

        # Mostra i risultati usando l'immagine aumentata
        plt.imshow(augmented_img_np.astype("uint8"))
        plt.title(f"Reale: {true_label_name} | Predetto: {predicted_class_name} (Score: {prediction_score:.2f})")
        plt.axis("off")
        plt.show()

def main():
    """Funzione principale per orchestrare il test."""
    clear_screen()
    print(f"\n{'='*50}\nSCRIPT DI TEST PER RICONOSCIMENTO LINEA\n{'='*50}")

    model, class_names = load_model_and_classes()
    if not model or not class_names:
        return

    model.summary()

    # Carica il dataset di validazione una sola volta e applica il preprocessing
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,  # DEVE essere lo stesso seed del training
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode='binary'
    ).map(preprocess_dataset_item, num_parallel_calls=tf.data.AUTOTUNE).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    evaluate_model(model, val_ds)
    
    # interactive_test ora riceve il val_ds, ma per la visualizzazione carica le immagini raw
    interactive_test(model, class_names, val_ds)
    
    print("\nArrivederci!")

if __name__ == "__main__":
    main()