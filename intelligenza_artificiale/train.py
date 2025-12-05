import tensorflow as tf
import os
import matplotlib.pyplot as plt
from datetime import datetime

# --- Costanti Globali ---
IMG_HEIGHT = 96
IMG_WIDTH = 128
BATCH_SIZE = 32

# --- Percorsi ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, 'dataset')
SESSIONS_DIR = os.path.join(SCRIPT_DIR, 'training_sessions')

def configure_tensorflow():
    """Configura TensorFlow per l'uso della CPU."""
    tf.config.threading.set_inter_op_parallelism_threads(4)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    print("✓ TensorFlow configurato per l'uso ottimale della CPU.")

def load_and_prepare_dataset():
    """Carica e prepara i dataset di training e validazione."""
    print("\n--- Caricamento del dataset ---")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode='binary'
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode='binary'
    )
    class_names = train_ds.class_names
    print(f"✓ Classi trovate: {class_names}")
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    print("✓ Dataset di training e validazione pronti.")
    return train_ds, val_ds, class_names

def create_model():
    """Crea un modello di classificazione basato su MobileNetV3Small."""
    print("\n--- Creazione del modello ---")
    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = True
    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = base_model(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, outputs)
    print("✓ Modello creato con successo.")
    return model

def plot_history(history, save_path):
    """Salva i grafici di accuratezza e loss."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig(save_path)
    plt.close()
    print(f"✓ Grafici salvati in: {save_path}")

class OverfittingDetector(tf.keras.callbacks.Callback):
    def __init__(self, tolerance_factor=1.1, patience=3):
        super().__init__()
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.tolerance_factor = tolerance_factor
        self.patience = patience

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        if train_loss is None or val_loss is None: return
        if val_loss >= self.best_val_loss:
            self.epochs_no_improve += 1
        else:
            self.best_val_loss = val_loss
            self.epochs_no_improve = 0
        if self.epochs_no_improve >= self.patience and val_loss > train_loss * self.tolerance_factor:
            print(f"\n⚠️ EPOCH {epoch+1}: Possibile overfitting! Val Loss ({val_loss:.4f}) non migliora da {self.patience} epoche e supera la Train Loss ({train_loss:.4f}).")

def preprocess_dataset_item(image, label):
    """Applica il preprocessing specifico di MobileNetV3Small."""
    return tf.keras.applications.mobilenet_v3.preprocess_input(image), label

def save_results(model, class_names, session_dir):
    """Salva il modello, i grafici e i nomi delle classi nella cartella di sessione."""
    print(f"\n--- Salvataggio dei risultati in: {session_dir} ---")
    model_path = os.path.join(session_dir, 'line_detection_model.keras')
    model.save(model_path)
    print(f"✓ Modello salvato in: {model_path}")
    class_path = os.path.join(session_dir, 'class_names.txt')
    with open(class_path, 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    print(f"✓ Nomi delle classi salvati in: {class_path}")
    history = model.history
    if history and history.history:
        plot_path = os.path.join(session_dir, 'training_history.png')
        plot_history(history, plot_path)
    else:
        print("✗ Dati di history non trovati, il grafico non sarà generato.")

def main():
    """Funzione principale per orchestrare il processo."""
    clear_screen()
    print(f"\n{'='*50}\nTRAINING PER RICONOSCIMENTO LINEA\n{'='*50}")

    # Creazione di una cartella di sessione univoca
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(SESSIONS_DIR, f"run_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)
    print(f"📂 Sessione di training creata in: {session_dir}")

    configure_tensorflow()
    train_ds_raw, val_ds_raw, class_names = load_and_prepare_dataset()
    
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.25, fill_mode="reflect"),
        tf.keras.layers.RandomZoom(0.35, fill_mode="reflect"),
        tf.keras.layers.RandomBrightness(factor=0.25),
    ], name='data_augmentation')

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds_raw.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE).map(preprocess_dataset_item, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds_raw.map(preprocess_dataset_item, num_parallel_calls=AUTOTUNE)
    
    model = create_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        OverfittingDetector(tolerance_factor=1.1, patience=3)
    ]

    try:
        print("\n--- Avvio del training ---")
        print("Premi Ctrl+C per interrompere e salvare il modello migliore.")
        model.fit(train_ds, epochs=50, validation_data=val_ds, callbacks=callbacks)
        print("✓ Training completato.")
    except KeyboardInterrupt:
        print("\n❗️ Training interrotto. Salvataggio del modello migliore in corso...")
    
    save_results(model, class_names, session_dir)
    
    print(f"\n{'='*50}\nPROCESSO COMPLETATO\n{'='*50}")

def clear_screen():
    """Pulisce lo schermo della console."""
    os.system('cls' if os.name == 'nt' else 'clear')

if __name__ == "__main__":
    main()
