# ========================================
# SISTEMA DI TRAINING ULTRA-OTTIMIZZATO - VERSIONE CORRETTA
# ========================================

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import time
import threading
import json
import tensorflow_model_optimization as tfmot

# --- Variabili Globali e Costanti ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_HEIGHT = 96
IMG_WIDTH = 128
BATCH_SIZE = 64 # Ridotto per stabilità
CLASSI = ['cane', 'gallina', 'gatto', 'mucca', 'pecora']


# ========================================
# FUNZIONI UTILITY
# ========================================

def clear_screen():
    """Pulisce lo schermo della console."""
    os.system('cls' if os.name == 'nt' else 'clear')


stop_training_flag = False


def listen_for_quit():
    """Attende l'input 'q' per fermare il training."""
    global stop_training_flag
    print("\nPremi 'q' e poi Invio in qualsiasi momento per fermare il training alla fine dell'epoca corrente.")
    while True:
        try:
            if input() == 'q':
                print("\n[INFO] Richiesta di interruzione ricevuta. Il training si fermerà alla fine di questa epoca.")
                stop_training_flag = True
                break
        except (EOFError, KeyboardInterrupt):
            # Gestisce i casi in cui lo stream di input è chiuso o interrotto
            print("\n[INFO] Stream di input interrotto, avvio arresto training.")
            stop_training_flag = True
            break


class QuitCallback(tf.keras.callbacks.Callback):
    """Callback per fermare il training se il flag globale è impostato."""
    def on_epoch_end(self, epoch, logs=None):
        global stop_training_flag
        if stop_training_flag:
            print("\n🛑 Interruzione del training in corso...")
            self.model.stop_training = True


class CustomEarlyStopping(tf.keras.callbacks.Callback):
    """
    Callback personalizzata che replica EarlyStopping e aggiunge un log sulla pazienza.
    """
    def __init__(self, monitor='val_loss', patience=10, restore_best_weights=True, verbose=1):
        super(CustomEarlyStopping, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        
        # Imposta la funzione di comparazione in base al monitor
        if 'acc' in self.monitor or 'accuracy' in self.monitor:
            self.monitor_op = np.greater
            self.best = -np.inf
        else:
            self.monitor_op = np.less
            self.best = np.inf

    def on_train_begin(self, logs=None):
        # Resetta le variabili all'inizio del training
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.inf if self.monitor_op == np.less else -np.inf
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return

        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
        
        # Stampa sempre lo stato della pazienza
        print(f" - EarlyStopping: Pazienza {self.wait}/{self.patience}")

        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            if self.restore_best_weights and self.best_weights is not None:
                if self.verbose > 0:
                    print("Ripristino dei pesi del modello dal miglior epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print(f"Epoch {self.stopped_epoch + 1}: Early stopping")


class WarmupExponentialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Learning rate schedule che combina un warmup lineare con un decadimento esponenziale.
    """
    def __init__(self, initial_learning_rate, decay_steps, decay_rate, warmup_steps, staircase=True, name=None):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate, decay_steps, decay_rate, staircase=staircase
        )
        self.warmup_steps = float(warmup_steps)
        self.name = name

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        
        # Calcola il learning rate del warmup
        warmup_percent_done = step / self.warmup_steps
        warmup_learning_rate = self.initial_learning_rate * warmup_percent_done
        
        # Scegli tra warmup e decay in base allo step corrente
        is_warmup = step < self.warmup_steps
        learning_rate = tf.cond(
            is_warmup,
            lambda: warmup_learning_rate,
            lambda: self.decay_schedule(step - self.warmup_steps)
        )
        return learning_rate

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_schedule.decay_steps,
            "decay_rate": self.decay_schedule.decay_rate,
            "warmup_steps": self.warmup_steps,
            "staircase": self.decay_schedule.staircase,
            "name": self.name,
        }


# ========================================
# CONFIGURAZIONE TENSORFLOW
# ========================================

def configura_tensorflow():
    """Ottimizza TensorFlow per massime performance"""
    # Ottimizzazioni CPU
    tf.config.threading.set_inter_op_parallelism_threads(4)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    
    print("✓ TensorFlow configurato per CPU")


# ========================================
# CARICAMENTO DATI CON TENSORFLOW DATASETS (TFDS)
# ========================================

def carica_e_prepara_benchmark_dataset(img_height, img_width, batch_size):
    """
    Carica, splitta e pre-processa il dataset 'cifar10' usando TFDS.
    """
    print("\n- - - Caricamento Benchmark Dataset 'cifar10' con TFDS - - -")
    # Carica il dataset con informazioni, usando gli split predefiniti
    (ds_train, ds_val), ds_info = tfds.load(
        'cifar10',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    
    num_classes = ds_info.features['label'].num_classes
    class_names = ds_info.features['label'].names
    
    print(f"✓ Dataset caricato: {ds_info.name}")
    print(f"  - Esempi di training: {ds_info.splits['train'].num_examples}")
    print(f"  - Esempi di validazione: {ds_info.splits['test'].num_examples}")
    print(f"  - Numero di classi: {num_classes} ({', '.join(class_names)})")

    # Funzione di pre-processing per ridimensionare e fare one-hot encoding
    def preprocess_and_one_hot(image, label):
        image = tf.image.resize(image, [img_height, img_width])
        label = tf.one_hot(label, num_classes)
        return image, label

    # Pipeline di dati
    AUTOTUNE = tf.data.AUTOTUNE
    ds_train = ds_train.map(preprocess_and_one_hot, num_parallel_calls=AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples // 10)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.prefetch(buffer_size=AUTOTUNE)

    ds_val = ds_val.map(preprocess_and_one_hot, num_parallel_calls=AUTOTUNE)
    ds_val = ds_val.batch(batch_size)
    ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(buffer_size=AUTOTUNE)
    
    print("✓ Pipeline dati creata (training e validazione con one-hot encoding)")
    return ds_train, ds_val, ds_info


# ========================================
# CARICAMENTO DATI CUSTOM
# ========================================

def carica_dataset_custom(mode, img_height, img_width, batch_size):
    """
    Crea i dataset di training e validazione dal file system locale.
    """
    print(f"\n- - - Caricamento Dataset Custom (Modalità: {mode.upper()}) - - -")
    
    base_path = os.path.join(SCRIPT_DIR, 'dataset', f"dataset_{mode}")
    
    # Crea dataset di training
    train_ds = tf.keras.utils.image_dataset_from_directory(
        base_path,
        labels='inferred',
        label_mode='int',
        class_names=CLASSI,
        image_size=(img_height, img_width),
        interpolation='bilinear',
        batch_size=batch_size,
        shuffle=True,
        seed=123,
        validation_split=0.2,
        subset='training'
    )

    # Crea dataset di validazione
    val_ds = tf.keras.utils.image_dataset_from_directory(
        base_path,
        labels='inferred',
        label_mode='int',
        class_names=CLASSI,
        image_size=(img_height, img_width),
        interpolation='bilinear',
        batch_size=batch_size,
        shuffle=False, # La validazione non necessita di shuffle
        seed=123,
        validation_split=0.2,
        subset='validation'
    )

    num_classes = len(train_ds.class_names)
    print(f"✓ Trovate {num_classes} classi: {train_ds.class_names}")

    def to_one_hot(image, label):
        return image, tf.one_hot(label, num_classes)
    
    AUTOTUNE = tf.data.AUTOTUNE
    # Applica one-hot encoding
    train_ds = train_ds.map(to_one_hot, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(to_one_hot, num_parallel_calls=AUTOTUNE)

    # Pipeline ottimizzata: cache, shuffle per ogni epoca, prefetch
    train_ds = train_ds.cache().shuffle(1000, reshuffle_each_iteration=True).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    print("✓ Pipeline dati custom creata (training e validazione con shuffle per epoca e one-hot encoding)")
    
    # Creiamo un oggetto "info" fittizio per compatibilità
    class DummyInfo:
        def __init__(self, num_classes):
            self.features = {'label': self}
            self.num_classes = num_classes
            self.name = f"custom_{mode}"

    ds_info = DummyInfo(num_classes)
    
    return train_ds, val_ds, ds_info


# ========================================
# MODELLO E ARCHITETTURA
# ========================================

def crea_augmentation_layer():
    """Layer di data augmentation potenziato."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        tf.keras.layers.RandomBrightness(factor=0.2),
        tf.keras.layers.RandomContrast(factor=0.2),
    ], name='data_augmentation')


def crea_modello(num_classes):
    """
    Crea il modello di transfer learning con MobileNetV3Small per efficienza.
    """
    print("\n- - - Creazione Modello - - -")
    
    # Input del modello
    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    
    # Data Augmentation
    x = crea_augmentation_layer()(inputs)
    
    # Preprocessing specifico del modello
    x = tf.keras.applications.mobilenet_v3.preprocess_input(x)

    # Modello base
    try:
        base_model = tf.keras.applications.MobileNetV3Small(
            include_top=False,
            weights='imagenet',
            input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
        )
    except ValueError:
        print("⚠ Errore di shape mismatch, applico workaround per caricamento pesi.")
        base_model = tf.keras.applications.MobileNetV3Small(
            include_top=False, weights=None, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
        )
        print("Caricamento pesi 'imagenet' su un modello base appena creato.")
        temp_model_for_weights = tf.keras.applications.MobileNetV3Small(
            include_top=False, weights='imagenet'
        )
        base_model.set_weights(temp_model_for_weights.get_weights())


    base_model.trainable = False # Inizia con il base model congelato
    print(f"✓ Modello base MobileNetV3Small caricato. Trainable: {base_model.trainable}")

    # Collega il modello base
    x = base_model(x, training=False)

    # Testa di classificazione
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    
    modello = tf.keras.Model(inputs, outputs)
    return modello, base_model


# ========================================
# TRAINING
# ========================================

def addestra_modello(ds_train, ds_val, ds_info):
    """
    Orchestra le due fasi di training (head e fine-tuning) usando model.fit.
    Applica il PRUNING al modello.
    """
    # --- Creazione Modello Denso ---
    modello_denso, base_model = crea_modello(num_classes=ds_info.features['label'].num_classes)
    
    # --- Configurazione Pruning ---
    cardinality_train = tf.data.experimental.cardinality(ds_train).numpy()
    if cardinality_train == tf.data.experimental.UNKNOWN_CARDINALITY:
        print("⚠️ Attenzione: Cardinalità del dataset sconosciuta. Stima degli step per il pruning.")
        num_examples = ds_info.splits['train'].num_examples if 'cifar10' in ds_info.name else BATCH_SIZE * 50
        cardinality_train = num_examples // BATCH_SIZE
    
    epochs_totali_stimati = 80  # Stima conservativa delle epoche totali
    end_step = cardinality_train * epochs_totali_stimati
    
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.0,
                                                               final_sparsity=0.50,
                                                               begin_step=cardinality_train * 2, # Inizia dopo 2 epoche
                                                               end_step=end_step)
    }
    
    # --- Applica il wrapper di Pruning al modello denso ---
    modello = tfmot.sparsity.keras.prune_low_magnitude(modello_denso, **pruning_params)
    print("\n✓ Wrapper di Pruning applicato al modello.")

    # --- Inizio Training ---
    history_fase1 = None
    history_fase2 = None
    
    quit_thread = threading.Thread(target=listen_for_quit, daemon=True)
    quit_thread.start()

    steps_per_epoch = cardinality_train

    # ---------------------------------
    # FASE 1: Training della Testa
    # ---------------------------------
    print("\n\n{'='*70}")
    print("🚀 FASE 1: Training della sola testa di classificazione (con Pruning)")
    print(f"{'='*70}")
    
    lr_schedule_fase1 = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3, decay_steps=steps_per_epoch * 5, decay_rate=0.9, staircase=True)

    callbacks_fase1 = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        QuitCallback(),
        CustomEarlyStopping(monitor='val_loss', patience=7, verbose=1, restore_best_weights=True),
    ]
    
    modello.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule_fase1),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    
    history_fase1 = modello.fit(ds_train, epochs=30, validation_data=ds_val, callbacks=callbacks_fase1)
    interrupted = stop_training_flag

    # Salva checkpoint con wrapper di pruning
    session_dir = salva_risultati(modello, history_fase1, None, ds_info, current_phase='phase1', base_model_trainable=base_model.trainable)

    if not interrupted:
        # ---------------------------------
        # FASE 2: Fine-Tuning
        # ---------------------------------
        print("\n\n{'='*70}")
        print("🔥 FASE 2: Fine-tuning del modello base (con Pruning)")
        print(f"{'='*70}")

        base_model.trainable = True
        fine_tune_at = -40
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        
        trainable_params = np.sum([tf.keras.backend.count_params(w) for w in modello.trainable_weights])
        print(f"✓ Modello base parzialmente scongelato (ultimi {abs(fine_tune_at)} layer). Parametri trainabili: {trainable_params:,}")

        warmup_epochs = 2
        warmup_steps = warmup_epochs * steps_per_epoch
        lr_schedule_fase2 = WarmupExponentialDecay(
            initial_learning_rate=1e-4, decay_steps=steps_per_epoch * 5, decay_rate=0.9, warmup_steps=warmup_steps, staircase=True)

        callbacks_fase2 = [
            tfmot.sparsity.keras.UpdatePruningStep(),
            QuitCallback(),
            CustomEarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True),
        ]

        modello.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule_fase2),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy']
        )

        initial_epoch_fase2 = len(history_fase1.epoch) if history_fase1 else 0
        history_fase2 = modello.fit(ds_train, epochs=80, validation_data=ds_val, initial_epoch=initial_epoch_fase2, callbacks=callbacks_fase2)
        
        if stop_training_flag: interrupted = True
        
        final_phase = 'phase0' if not interrupted else 'phase2'
        salva_risultati(modello, history_fase1, history_fase2, ds_info, session_dir=session_dir, current_phase=final_phase, base_model_trainable=base_model.trainable)

    return modello, history_fase1, history_fase2, interrupted


def continua_addestramento(session_dir):
    """
    Carica uno stato di training con pruning e continua l'addestramento.
    """
    print(f"\n- - - Continuazione Training dalla Sessione: {os.path.basename(session_dir)} - - -")
    
    # --- Carica stato di training ---
    state_path = os.path.join(session_dir, 'training_state.json')
    try:
        with open(state_path, 'r') as f: state = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ Errore critico: Impossibile caricare il file di stato '{state_path}'. {e}"); return

    last_epoch = state['last_epoch']
    dataset_name = state['dataset_name']
    model_path = os.path.join(SCRIPT_DIR, state['model_relative_path'])
    previous_history = state['history']
    current_phase = state.get('current_phase', 'phase1')
    is_pruned = state.get('is_pruned', False) # Controlla se il modello è pruned

    print(f"✓ Stato caricato. Ultima epoca: {last_epoch}, Dataset: {dataset_name}, Fase: {current_phase.upper()}, Pruning: {'Sì' if is_pruned else 'No'}")

    # --- Carica dataset ---
    if 'cifar10' in dataset_name: ds_train, ds_val, ds_info = carica_e_prepara_benchmark_dataset(IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE)
    else: mode = dataset_name.replace('custom_', ''); ds_train, ds_val, ds_info = carica_dataset_custom(mode, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE)

    steps_per_epoch = tf.data.experimental.cardinality(ds_train).numpy()

    # --- Carica modello (con o senza wrapper di pruning) ---
    print(f"Caricamento modello da '{model_path}'...")
    try:
        if is_pruned:
            with tfmot.sparsity.keras.prune_scope():
                modello = tf.keras.models.load_model(model_path)
            print("✓ Modello con Pruning caricato.")
        else:
            modello = tf.keras.models.load_model(model_path)
            print("✓ Modello standard (non-pruned) caricato.")
    except Exception as e:
        print(f"❌ Errore critico durante il caricamento del modello: {e}"); return

    base_model = next((layer for layer in modello.layers if 'mobilenet' in layer.name.lower() or 'efficientnet' in layer.name.lower()), None)
    if base_model is None: print("❌ Errore critico: Impossibile trovare il layer del modello base."); return
    
    global stop_training_flag; stop_training_flag = False
    quit_thread = threading.Thread(target=listen_for_quit, daemon=True); quit_thread.start()

    history_fase1_continuazione, history_fase2_continuazione = None, None
    interrupted = False

    # --- Setup Callbacks ---
    pruning_callbacks = [tfmot.sparsity.keras.UpdatePruningStep()] if is_pruned else []

    # ---------------------------------
    # LOGICA DI CONTINUAZIONE FASE 1
    # ---------------------------------
    if current_phase == 'phase1':
        print(f"\n\n{'='*70}\n🚀 Continuazione FASE 1 (da epoca {last_epoch})\n{'='*70}")
        base_model.trainable = False
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(1e-3, steps_per_epoch * 5, 0.9, staircase=True)
        callbacks = pruning_callbacks + [QuitCallback(), CustomEarlyStopping(patience=7, verbose=1)]
        modello.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=['accuracy'])
        history_fase1_continuazione = modello.fit(ds_train, epochs=30, validation_data=ds_val, initial_epoch=last_epoch, callbacks=callbacks)
        interrupted = stop_training_flag
        last_epoch = len(previous_history.get('accuracy', [])) + len(history_fase1_continuazione.epoch)

    # ---------------------------------
    # LOGICA DI CONTINUAZIONE FASE 2 (o transizione da Fase 1)
    # ---------------------------------
    if not interrupted and (current_phase == 'phase2' or (current_phase == 'phase1' and not stop_training_flag)):
        print(f"\n\n{'='*70}\n🔥 Continuazione FASE 2 (Fine-tuning, da epoca {last_epoch})\n{'='*70}")
        base_model.trainable = True
        fine_tune_at = -40
        for layer in base_model.layers[:fine_tune_at]: layer.trainable = False
        
        warmup_steps = 2 * steps_per_epoch
        lr_schedule = WarmupExponentialDecay(1e-4, steps_per_epoch * 5, 0.9, warmup_steps, staircase=True)
        callbacks = pruning_callbacks + [QuitCallback(), CustomEarlyStopping(patience=10, verbose=1)]
        modello.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=['accuracy'])
        history_fase2_continuazione = modello.fit(ds_train, epochs=80, validation_data=ds_val, initial_epoch=last_epoch, callbacks=callbacks)
        if stop_training_flag: interrupted = True
    
    # --- Determina fase finale e salva ---
    if not interrupted and (current_phase in ['phase1', 'phase2']): final_phase = 'phase0'
    else: final_phase = 'phase2' if (current_phase == 'phase2' or (current_phase == 'phase1' and not interrupted)) else 'phase1'
    
    salva_risultati(modello, history_fase1_continuazione, history_fase2_continuazione, ds_info, session_dir, previous_history, final_phase, base_model.trainable, is_pruned)
    if final_phase == 'phase0': print("\n✅ Training completato con successo! Modello finale (stripped) salvato.")
    else: print("\n✅ Processo interrotto/completato. Risultati parziali salvati.")


# ========================================
# SALVATAGGIO E LOGGING
# ========================================

def salva_modello_definitivo(modello_pruned, session_dir, model_name):
    """Rimuove i wrapper di pruning e salva il modello finale, pronto per l'inferenza."""
    print("\n- - - Creazione Modello Finale (Stripped) - - -")
    try:
        modello_stripped = tfmot.sparsity.keras.strip_pruning(modello_pruned)
        final_model_name = model_name.replace('.keras', '_pruned_final.keras')
        final_model_path = os.path.join(session_dir, final_model_name)
        modello_stripped.save(final_model_path)
        print(f"✓ Modello finale salvato: {final_model_path}")
        return final_model_path
    except Exception as e:
        print(f"❌ Errore durante lo stripping/salvataggio del modello finale: {e}")
        return None

def salva_risultati(modello, history1, history2, ds_info, session_dir=None, previous_history=None, current_phase='unknown', base_model_trainable=False, is_pruned=False):
    """
    Salva il modello (con wrapper), il grafico, un log e lo stato di training.
    Se il training è completato, salva anche una versione "stripped" del modello.
    """
    print("\n- - - Salvataggio Risultati - - -")
    
    # --- Determina la modalità e i nomi dei file ---
    if 'cifar10' in ds_info.name:
        mode = 'tf' 
        model_save_name = 'modello_cifar10.keras'
    else:
        mode = ds_info.name.replace('custom_', '')
        model_save_name = 'modello.keras'

    # --- Gestisce la cartella di sessione ---
    if session_dir is None:
        model_base_dir = os.path.join(SCRIPT_DIR, 'modelli_salvati', f"modelli_salvati_{mode}")
        os.makedirs(model_base_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_name = f"{mode}_{timestamp}"
        session_dir = os.path.join(model_base_dir, session_name)
        os.makedirs(session_dir)
        print(f"Creata nuova cartella sessione: {session_dir}")
    
    # Salva modello di checkpoint (CON wrapper, se presente)
    model_path = os.path.join(session_dir, model_save_name)
    modello.save(model_path)
    print(f"✓ Modello di checkpoint salvato: {model_path}")

    # Combina history per grafico e log
    if previous_history is None: previous_history = {}
    def combine_metric(metric_name):
        return previous_history.get(metric_name, []) + \
               (history1.history.get(metric_name, []) if history1 else []) + \
               (history2.history.get(metric_name, []) if history2 else [])
    combined_history = {k: combine_metric(k) for k in ['accuracy', 'val_accuracy', 'loss', 'val_loss', 'learning_rate']}
    total_epochs = len(combined_history['accuracy'])

    # --- Crea e salva grafico ---
    if total_epochs > 0:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(combined_history['accuracy'], label='Training Accuracy')
        plt.plot(combined_history['val_accuracy'], label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        plt.subplot(1, 2, 2)
        plt.plot(combined_history['loss'], label='Training Loss')
        plt.plot(combined_history['val_loss'], label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        graph_path = os.path.join(session_dir, 'grafico_training.png')
        plt.savefig(graph_path)
        plt.close()
        print(f"✓ Grafico salvato: {graph_path}")

    # --- Salva log testuale ---
    training_status = "COMPLETATO" if current_phase == 'phase0' else f"IN CORSO {'(PRUNING)' if is_pruned else ''}"
    log_path = os.path.join(session_dir, f"log_sessione_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(log_path, 'w', encoding='utf-8') as f: f.write(f"SESSIONE DI TRAINING '{ds_info.name}' {'(CON PRUNING)' if is_pruned else ''}\n{'='*50}\n")
    f.write(f"Stato: {training_status}\nModello Base: MobileNetV3Small\nEpoche totali: {total_epochs}\n")
    if total_epochs > 0:
        final_val_acc = combined_history['val_accuracy'][-1]
        f.write(f"Accuratezza Validazione Finale: {final_val_acc:.4f}\n")
    print(f"✓ Log salvato: {log_path}")

    # Salva stato per continuazione
    if total_epochs > 0:
        training_state = {
            "last_epoch": total_epochs, "dataset_name": ds_info.name,
            "model_relative_path": os.path.relpath(model_path, SCRIPT_DIR).replace('\\', '/'),
            "history": combined_history, "current_phase": current_phase,
            "training_completed": current_phase == 'phase0', "is_pruned": is_pruned
        }
        state_path = os.path.join(session_dir, 'training_state.json')
        with open(state_path, 'w', encoding='utf-8') as f: json.dump(training_state, f, indent=4)
        print(f"✓ Stato di training salvato: {state_path}")
    
    # Se il training è completo, salva il modello finale senza wrapper
    if current_phase == 'phase0' and is_pruned:
        salva_modello_definitivo(modello, session_dir, model_save_name)
        
    return session_dir


def seleziona_sessione_da_continuare(mode):
    """
    Cerca e permette all'utente di selezionare una sessione di training da cui continuare.
    """
    print(f"\n- - - Ricerca sessioni di training da continuare (Modalità: {mode.upper()}) - - -")
    base_dir = os.path.join(SCRIPT_DIR, 'modelli_salvati', f"modelli_salvati_{mode}")
    
    if not os.path.exists(base_dir):
        print(f"⚠ La cartella '{base_dir}' non esiste. Impossibile continuare."); time.sleep(2); return None

    sessions = []
    for session_name in sorted(os.listdir(base_dir), reverse=True):
        state_file = os.path.join(base_dir, session_name, 'training_state.json')
        if os.path.isfile(state_file):
            try:
                with open(state_file, 'r') as f: state = json.load(f)
                is_pruned = state.get('is_pruned', False)
                sessions.append({
                    "path": os.path.join(base_dir, session_name), "name": session_name,
                    "epochs": state.get("last_epoch", "N/A"),
                    "val_acc": state.get("history", {}).get("val_accuracy", [])[-1] if state.get("history", {}).get("val_accuracy") else "N/A",
                    "completed": state.get('training_completed', False), "pruned": is_pruned
                })
            except (json.JSONDecodeError, IndexError): continue
    
    if not sessions: print("Nessuna sessione valida trovata."); time.sleep(2); return None

    while True:
        clear_screen()
        print("\nScegli la sessione da cui continuare:\nLegenda: [✓] Completato | [►] In corso | [P] Pruning\n")
        for i, s in enumerate(sessions):
            val_acc = f"{s['val_acc']:.4f}" if isinstance(s['val_acc'], float) else s['val_acc']
            status = "[✓]" if s['completed'] else "[►]"
            pruning_tag = "[P]" if s['pruned'] else ""
            print(f"  {i+1}. {status} {s['name']} {pruning_tag}\n      Epoche: {s['epochs']} | Val Acc: {val_acc}")
        
        print(f"\n  {len(sessions) + 1}. Annulla")
        try:
            scelta = int(input(f"\nInserisci un numero (1-{len(sessions) + 1}): ").strip()) - 1
            if 0 <= scelta < len(sessions): return sessions[scelta]["path"]
            elif scelta == len(sessions): return None
        except ValueError: print("\nErrore: Inserisci solo un numero."); time.sleep(1.5)


# ========================================
# MAIN
# ========================================

def main():
    """Funzione principale per orchestrare il processo di training."""
    while True:
        clear_screen()
        print(f"\n{'='*70}\n🤖 SISTEMA DI TRAINING v5 - CON PRUNING\n{'='*70}")
        print("\nScegli un'opzione:")
        print("  1. Nuovo training (Benchmark - CIFAR-10)")
        print("  2. Nuovo training (Dataset Bot)")
        print("  3. Continua training (Benchmark - CIFAR-10)")
        print("  4. Continua training (Dataset Bot)")
        print("  5. Esci")
        
        scelta = input("\nInserisci la tua scelta (1-5): ").strip()

        if scelta in ['1', '2']:
            mode = 'tf' if scelta == '1' else 'bot'
            print(f"\n--- Avvio Nuovo Training (Modalità: {mode.upper()}) ---")
            configura_tensorflow()
            if mode == 'tf': ds_train, ds_val, ds_info = carica_e_prepara_benchmark_dataset(IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE)
            else: ds_train, ds_val, ds_info = carica_dataset_custom('bot', IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE)
            addestra_modello(ds_train, ds_val, ds_info)
            input("\nPremi Invio per tornare al menu principale...")

        elif scelta in ['3', '4']:
            mode = 'tf' if scelta == '3' else 'bot'
            session_dir = seleziona_sessione_da_continuare(mode)
            if session_dir:
                continua_addestramento(session_dir)
                input("\nPremi Invio per tornare al menu principale...")

        elif scelta == '5': print("Arrivederci!"); sys.exit(0)
        else: print("\nErrore: Scelta non valida."); time.sleep(1.5)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n\n❌ ERRORE CRITICO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)