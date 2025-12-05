# Questo script è dedicato al testing dei modelli di intelligenza artificiale addestrati.
# Permette di valutare le prestazioni del modello su immagini di test, sia in modalità
# manuale (caricando immagini casuali) che automatica. Il suo scopo è fornire un'interfaccia
# interattiva per verificare l'accuratezza del modello e generare log dettagliati delle sessioni di test.
# Il codice gestisce la selezione della modalità (test/bot), il caricamento del modello,
# la pre-elaborazione delle immagini e la registrazione dei risultati.

import os
import sys
import time
import random
import msvcrt
from datetime import datetime
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# --- Variabili Globali e Costanti ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(SCRIPT_DIR, 'logs', 'logs_test')
# Le classi devono corrispondere all'ordine usato durante il training
CLASSI = ['cane', 'gallina', 'gatto', 'mucca', 'pecora']
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 96

# Nuove variabili globali per gestire il dataset TFDS in memoria
TF_FLOWERS_DATA = None
TF_CLASS_NAMES = None

def load_benchmark_dataset_into_memory():
    """
    Carica il set di test di CIFAR-10 in memoria per il testing interattivo.
    """
    global TF_FLOWERS_DATA, TF_CLASS_NAMES
    if TF_FLOWERS_DATA is not None:
        print("✓ Dataset 'cifar10' già in memoria.")
        return

    print("\nCaricamento del dataset 'cifar10' per il test in corso...")
    try:
        ds_test, ds_info = tfds.load(
            'cifar10',
            split='test', # Carica solo lo split di test
            shuffle_files=False,
            as_supervised=True,
            with_info=True,
        )
        
        # Salva i nomi delle classi
        TF_CLASS_NAMES = ds_info.features['label'].names
        
        # Applica il resize alle immagini per conformarle all'input del modello
        def resize_image(image, label):
            return tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH]), label
        
        ds_test = ds_test.map(resize_image)

        # Pre-carica l'intero set di test in una lista per un accesso casuale veloce
        TF_FLOWERS_DATA = list(ds_test.as_numpy_iterator())
        
        print(f"✓ Dataset 'cifar10' caricato: {len(TF_FLOWERS_DATA)} immagini di test pronte.")
        time.sleep(1.5)

    except Exception as e:
        print(f"\nErrore critico durante il caricamento di cifar10: {e}")
        print("Impossibile procedere con il test per questo modello.")
        time.sleep(3)
        # Resetta per permettere di tornare al menu principale
        TF_FLOWERS_DATA = [] 
        TF_CLASS_NAMES = []



def clear_screen():
    """Pulisce lo schermo della console."""
    os.system('cls' if os.name == 'nt' else 'clear')

def select_model():
    """
    Trova tutti i modelli disponibili, li elenca e chiede all'utente di sceglierne uno.
    Restituisce il percorso completo al modello e la modalità inferita ('bot' o 'tf').
    """
    base_dir = os.path.join(SCRIPT_DIR, 'modelli_salvati')
    model_options = [] # Lista di tuple (percorso_completo, percorso_relativo_per_display)

    if not os.path.exists(base_dir):
        print(f"\nErrore: La cartella base dei modelli '{base_dir}' non è stata trovata.")
        sys.exit(1)

    # Cerca in tutte le sottocartelle di modelli_salvati (es. modelli_salvati_bot, modelli_salvati_tf)
    for model_type_dir in sorted(os.listdir(base_dir)):
        full_model_type_dir = os.path.join(base_dir, model_type_dir)
        if os.path.isdir(full_model_type_dir):
            for session_dir in sorted(os.listdir(full_model_type_dir)):
                full_session_path = os.path.join(full_model_type_dir, session_dir)
                if os.path.isdir(full_session_path):
                    # Cerca qualsiasi file .keras o .h5
                    for file_name in os.listdir(full_session_path):
                        if file_name.endswith(('.keras', '.h5')):
                            full_path = os.path.join(full_session_path, file_name)
                            display_path = os.path.join(model_type_dir, session_dir, file_name)
                            model_options.append((full_path, display_path))

    if not model_options:
        print(f"\nErrore: Nessun modello (.h5 o .keras) trovato in '{base_dir}'.")
        sys.exit(1)

    while True:
        clear_screen()
        print("\nScegli quale modello testare:")
        for i, (_, display_path) in enumerate(model_options):
            print(f"  {i + 1}. {display_path}")

        try:
            scelta = input(f"Inserisci un numero (1-{len(model_options)}): ").strip()
            scelta_idx = int(scelta) - 1
            if 0 <= scelta_idx < len(model_options):
                selected_path, display_name = model_options[scelta_idx]
                
                # Inferisce la modalità dal percorso
                mode = 'tf' if 'tf' in display_name else 'bot'
                
                print(f"Modello selezionato: {display_name} (Modalità: {mode})")
                return selected_path, mode
            else:
                print(f"\nErrore: Inserisci un numero tra 1 e {len(model_options)}.")
                time.sleep(1.5)
        except ValueError:
            print("\nErrore: Inserisci solo un numero valido.")
            time.sleep(1.5)

def load_model(model_path):
    """
    Carica il modello Keras dal percorso specificato.
    Restituisce l'oggetto modello caricato.
    """
    print("\nCaricamento del modello in corso...")
    try:
        model = tf.keras.models.load_model(model_path)
        print("✓ Modello caricato con successo!")
        time.sleep(1.5)
        return model
    except Exception as e:
        print(f"\nErrore critico durante il caricamento del modello: {e}")
        sys.exit(1)

def get_random_image_path(mode):
    """
    Ottiene il percorso di un'immagine casuale dal set di validazione corretto,
    usando la stessa logica di splitting di train.py.
    Restituisce il percorso dell'immagine e la sua vera classe, o (None, None) se non vengono trovate immagini.
    """
    if mode != 'bot':
        print("\nInfo: La ricerca di immagini è disponibile solo per la modalità 'bot'.")
        return None, None

    base_path = os.path.join(SCRIPT_DIR, 'dataset', f"dataset_{mode}")
    
    if not os.path.exists(base_path) or not any(os.scandir(base_path)):
        print(f"\nErrore: La cartella del dataset '{base_path}' non è stata trovata o è vuota.")
        return None, None

    # Usa la stessa funzione di train.py per ottenere il set di validazione
    try:
        val_ds = tf.keras.utils.image_dataset_from_directory(
            base_path,
            labels='inferred',
            label_mode='int',
            class_names=CLASSI,
            image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
            shuffle=False, # Non serve mescolare qui, vogliamo solo i percorsi
            seed=123,
            validation_split=0.2,
            subset='validation'
        )
    except tf.errors.InvalidArgumentError as e:
         print(f"\nErrore: Nessuna immagine trovata in '{base_path}'. Assicurati che le immagini siano nelle sottocartelle di classe (es. 'cane', 'gatto'...).")
         print(f"Dettaglio errore: {e}")
         return None, None


    file_paths = val_ds.file_paths
    if not file_paths:
        print(f"\nErrore: Nessuna immagine trovata nel set di validazione per il dataset '{base_path}'.")
        return None, None

    random_path = random.choice(file_paths)
    # Estrai la classe dal percorso (es. .../cane/img.jpg -> cane)
    true_class = os.path.basename(os.path.dirname(random_path))
    
    return random_path, true_class

def preprocess_image(image_path):
    """
    Carica e pre-elabora un'immagine per renderla pronta per il modello.
    Restituisce l'immagine elaborata come array numpy.
    """
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Crea un batch di 1
    # Applica la stessa pre-elaborazione usata in training per EfficientNet
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array

def log_and_exit(start_time, total_images_tested, correct_predictions, mode):
    """
    Registra i dettagli della sessione nel file di log corretto ed esce.
    """
    end_time = datetime.now()
    run_time = end_time - start_time
    accuracy = (correct_predictions / total_images_tested * 100) if total_images_tested > 0 else 0

    # Imposta la directory dei log in base alla modalità inferita ('bot' o 'tf')
    log_subdir = f'logs_{mode}'
    current_log_dir = os.path.join(SCRIPT_DIR, 'logs', 'logs_test', log_subdir)
    os.makedirs(current_log_dir, exist_ok=True)
    
    # Trova il prossimo numero sequenziale per il file di log
    next_num = 1
    while True:
        log_file_name = f"logs_{mode}_{next_num}.txt"
        log_file_path = os.path.join(current_log_dir, log_file_name)
        if not os.path.exists(log_file_path):
            break
        next_num += 1

    log_entry = f"""
----------------------------------------
SESSIONE DI TEST
----------------------------------------
Modalita': {mode.upper()}
Avviato il: {start_time.strftime('%Y-%m-%d %H:%M:%S')}
Finito il:  {end_time.strftime('%Y-%m-%d %H:%M:%S')}
Durata:     {str(run_time).split('.')[0]}

--- Statistiche ---
Immagini testate: {total_images_tested}
Accuratezza totale: {accuracy:.2f}%
----------------------------------------
"""
    
    print(f"\nSalvataggio log in '{log_file_path}'...")
    with open(log_file_path, 'a') as f:
        f.write(log_entry)
    
    print("Uscita in corso. Arrivederci!")
    time.sleep(2)
    sys.exit(0)

def main_loop(model, mode, start_time):
    """
    Il loop interattivo principale per testare il modello.
    """
    total_images_tested = 0
    correct_predictions = 0
    last_prediction_result = "N/A"
    last_predicted_class = "N/A"
    last_true_class = "N/A"

    # Determina la lista di classi corretta in base alla modalità
    class_list = TF_CLASS_NAMES if mode == 'tf' else CLASSI

    while True:
        clear_screen()
        accuracy = (correct_predictions / total_images_tested * 100) if total_images_tested > 0 else 0
        
        print(f"--- INTERFACCIA DI TEST (Modalità: {mode.upper()}) ---")
        print(f"Accuratezza generale: {accuracy:.2f}% ({correct_predictions}/{total_images_tested})")
        print("-" * 27)
        print(f"Ultimo risultato: {last_prediction_result}")
        if last_prediction_result != "N/A":
            print(f"  - Predetto: {last_predicted_class}")
            print(f"  - Reale:    {last_true_class}")
        
        print("-" * 27)
        print("\nScegli un'opzione:")
        print("  1. Carica immagine casuale")
        print("  2. Avvia test automatico")
        print("  3. Esci e salva log")

        scelta = input("\n> ").strip()

        if scelta in ['1', '2']:
            # Logica per test singolo o automatico
            is_auto_test = (scelta == '2')
            intervallo = 1.0 # Default per test singolo

            if is_auto_test:
                try:
                    intervallo_str = input("Inserisci l'intervallo in secondi (es. 2.5): ")
                    intervallo = float(intervallo_str)
                    if intervallo <= 0: raise ValueError
                except ValueError:
                    print("Intervallo non valido. Ritorno al menu.")
                    time.sleep(1.5)
                    continue
                print("\nAvvio test automatico... Premi 'ESC' per fermare.")
                time.sleep(2)

            # Loop per test automatico (gira una sola volta per test singolo)
            while True:
                if is_auto_test and msvcrt.kbhit() and ord(msvcrt.getch()) == 27:
                    print("\nTest automatico interrotto. Ritorno al menu...")
                    time.sleep(1.5)
                    break

                processed_image = None
                true_class = None

                if mode == 'bot':
                    image_path, true_class_str = get_random_image_path(mode)
                    if image_path:
                        processed_image = preprocess_image(image_path)
                        true_class = true_class_str
                
                elif mode == 'tf':
                    if TF_FLOWERS_DATA:
                        image_tensor, label_idx = random.choice(TF_FLOWERS_DATA)
                        true_class = class_list[label_idx]
                        # Pre-elaborazione per il tensore già in memoria
                        img_array = np.expand_dims(image_tensor, axis=0)
                        processed_image = tf.keras.applications.efficientnet.preprocess_input(img_array)

                if processed_image is not None and true_class is not None:
                    prediction = model.predict(processed_image)
                    predicted_class_idx = np.argmax(prediction)
                    predicted_class = class_list[predicted_class_idx]

                    total_images_tested += 1
                    last_true_class = true_class
                    last_predicted_class = predicted_class

                    if predicted_class == true_class:
                        correct_predictions += 1
                        last_prediction_result = "Giusto"
                    else:
                        last_prediction_result = "Sbagliato"
                else:
                    last_prediction_result = "Errore caricamento immagine"
                    print("Errore nel caricamento dell'immagine, salto il test.")

                if not is_auto_test:
                    break # Esce dal loop se è un test singolo
                
                # Aggiorna UI per test automatico
                clear_screen()
                accuracy = (correct_predictions / total_images_tested * 100)
                print("--- TEST AUTOMATICO IN CORSO (Premi ESC per fermare) ---")
                print(f"Accuratezza generale: {accuracy:.2f}% ({correct_predictions}/{total_images_tested})")
                print("-" * 27)
                print(f"Risultato: {last_prediction_result} (Predetto: {last_predicted_class}, Reale: {last_true_class})")
                print(f"Prossimo test in {intervallo} secondi...")
                time.sleep(intervallo)

        elif scelta == '3':
            log_and_exit(start_time, total_images_tested, correct_predictions, mode)
        
        else:
            print("Scelta non valida. Riprova.")
            time.sleep(1)

def main():
    """
    Funzione principale per orchestrare il processo di test.
    """
    start_time = datetime.now()
    clear_screen()
    print("=" * 50)
    print("--- SCRIPT DI TEST PER MODELLI ---")
    print("=" * 50)

    # 1. Seleziona il modello (la modalità viene inferita)
    model_path, mode = select_model()

    # 2. Se la modalità è 'tf', carica il dataset di benchmark in memoria
    if mode == 'tf':
        load_benchmark_dataset_into_memory()
        # Se il caricamento fallisce, TF_FLOWERS_DATA sarà vuoto e il loop di test lo gestirà
    
    # 3. Carica il modello Keras
    model = load_model(model_path)

    # 4. Avvia il loop di test
    main_loop(model, mode, start_time)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nUscita forzata. Arrivederci!")
        sys.exit(0)