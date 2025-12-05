# Documentazione Interna Gemini

Questo documento `GEMINI.md` serve come memoria interna e registro di contesto per l'agente Gemini. Contiene una descrizione dello stato attuale del progetto, delle decisioni chiave, della struttura, delle convenzioni di denominazione e dei dettagli tecnici cruciali per lo sviluppo.

**Importante:** Questo file `GEMINI.md` deve essere aggiornato in caso di modifiche alla struttura del progetto, allo scopo dei file, ai percorsi o alle convenzioni di formattazione.

---

## Stato Attuale del Progetto

Questa sezione riassume lo stato corrente della pipeline di addestramento e test dell'IA, focalizzata sulla classificazione binaria di immagini (`linea` vs. `no_linea`).

### 1. Obiettivo Principale e Dataset
-   **Scopo:** Classificare immagini in due categorie: quelle che contengono una linea (`linea`) e quelle che non la contengono (`no_linea`).
-   **Dataset:** I dati si trovano in `intelligenza_artificiale/dataset/`. La struttura prevede due cartelle principali, `linea` e `no_linea`, che a loro volta contengono sotto-cartelle di immagini.

### 2. Script di Addestramento (`train.py`)
-   **Caricamento Dati:** Utilizza `tf.keras.utils.image_dataset_from_directory` per caricare le immagini in modo efficiente.
    -   **Modalità:** `label_mode='binary'`.
    -   **Split:** Suddivide automaticamente il dataset in 80% per il training e 20% per la validazione (`validation_split=0.2`), usando un `seed` fisso per garantire la riproducibilità.
    -   **Ottimizzazione:** La pipeline dati è ottimizzata con `.cache()` e `.prefetch()`.
-   **Architettura Modello:**
    -   **Base:** `MobileNetV3Small` pre-addestrato, per un buon compromesso tra efficienza e performance.
    -   **Head:** Un head di classificazione specifico per il task binario: `GlobalAveragePooling2D`, `Dropout`, e un singolo neurone `Dense` con attivazione `sigmoid`.
    -   **Loss:** `binary_crossentropy`.
-   **Data Augmentation:**
    -   Un layer `Sequential` applica `RandomFlip`, `RandomRotation`, `RandomZoom`, e `RandomBrightness` per aumentare la variabilità del training set e ridurre l'overfitting.
-   **Processo di Addestramento:**
    -   Esegue un singolo ciclo di `model.fit()`.
    -   **EarlyStopping:** Monitora la `val_loss` e interrompe l'addestramento se non ci sono miglioramenti, ripristinando i pesi della migliore epoca (`restore_best_weights=True`).
    -   **OverfittingDetector:** Una callback personalizzata che monitora le loss di training e validazione. Se la `val_loss` peggiora o non migliora per un certo numero di epoche (patience) e diventa significativamente maggiore della `train_loss`, un avviso di possibile overfitting viene stampato in console.
    -   **Graceful Shutdown:** L'addestramento può essere interrotto manualmente con `Ctrl+C`. Grazie al callback `EarlyStopping`, il modello migliore trovato fino a quel momento viene comunque salvato.
-   **Artefatti di Output:**
    -   Tutti i file vengono salvati in una cartella dedicata: `intelligenza_artificiale/modello_linea/`.
    -   `line_detection_model.keras`: Il modello addestrato nel formato Keras v3.
    -   `class_names.txt`: File di testo con i nomi delle due classi (`linea`, `no_linea`).
    -   `training_history.png`: Grafico con le curve di accuratezza e loss per training e validazione.

### 3. Script di Test (`test.py`)
-   **Allineamento:** Lo script è perfettamente allineato con `train.py`.
-   **Caricamento Modello:** Carica il modello (`.keras` o `.h5` per retrocompatibilità) e i nomi delle classi dalla cartella `modello_linea`.
-   **Valutazione Automatica:** Carica il 20% del dataset usato per la validazione (usando lo stesso `seed` di `train.py`) e valuta le performance del modello, stampando loss e accuratezza.
-   **Test Interattivo:**
    -   Offre un'interfaccia a riga di comando per testare il modello su immagini specifiche.
    -   **Selezione Categoria:** L'utente può scegliere se testare un'immagine dalla categoria `linea`, `no_linea`, oppure un'immagine casuale.
    -   **Augmentation in Test:** Prima della predizione, all'immagine selezionata viene applicata la **stessa data augmentation** usata nel training. Questo permette di testare la robustezza del modello a variazioni di zoom, luminosità e rotazione.
    -   Mostra l'immagine (aumentata), l'etichetta reale e la predizione del modello.

### Conclusione
Il progetto possiede una pipeline MLOps pulita, semplice e robusta, specificamente progettata per il task di classificazione binaria. Gli script sono coerenti, facili da usare e pronti per l'addestramento e la valutazione affidabile del modello.
