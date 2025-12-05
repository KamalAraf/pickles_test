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
    -   Un layer `Sequential` applica `RandomFlip`, `RandomRotation(0.25, fill_mode="reflect")`, `RandomZoom(0.35, fill_mode="reflect")`, e `RandomBrightness(factor=0.25)`.
    -   L'uso di `fill_mode="reflect"` evita la creazione di bordi neri durante le trasformazioni, che potrebbero confondere il modello.
-   **Processo di Addestramento:**
    -   Esegue un singolo ciclo di `model.fit()`.
    -   **EarlyStopping:** Monitora la `val_loss` e interrompe l'addestramento se non ci sono miglioramenti, ripristinando i pesi della migliore epoca (`restore_best_weights=True`).
    -   **OverfittingDetector:** Una callback personalizzata che monitora le loss di training e validazione. Se la `val_loss` peggiora o non migliora per un certo numero di epoche (patience) e diventa significativamente maggiore della `train_loss`, un avviso di possibile overfitting viene stampato in console.
    -   **Graceful Shutdown:** L'addestramento può essere interrotto manualmente con `Ctrl+C`. Grazie al callback `EarlyStopping`, il modello migliore trovato fino a quel momento viene comunque salvato.
-   **Artefatti di Output:**
    -   Ogni sessione di training viene salvata in una cartella univoca con timestamp dentro `intelligenza_artificiale/training_sessions/`. Esempio: `run_20231027_153000`.
    -   All'interno di questa cartella vengono salvati:
        -   `line_detection_model.keras`: Il modello addestrato.
        -   `class_names.txt`: I nomi delle classi.
        -   `training_history.png`: Il grafico delle performance.
    -   La generazione del grafico è robusta e funziona anche se il training viene interrotto.

### 3. Script di Test (`test.py`)
-   **Dipendenze:** Richiede la libreria `SciPy` per la rotazione delle immagini. Se non è installata, lo script fornirà le istruzioni per installarla.
-   **Allineamento:** Lo script è perfettamente allineato con `train.py`.
-   **Caricamento Modello:** All'avvio, lo script elenca tutte le sessioni di training disponibili e chiede all'utente di sceglierne una. L'utente può inserire un numero per testare un modello specifico o premere Invio per usare l'opzione predefinita (la sessione più recente).
-   **Valutazione Automatica:** Carica il 20% del dataset usato per la validazione (usando lo stesso `seed` di `train.py`) e valuta le performance del modello, stampando loss e accuratezza.
-   **Test Interattivo:**
    -   Offre un'interfaccia a riga di comando per testare il modello su immagini specifiche.
    -   **Selezione Categoria:** L'utente può scegliere se testare un'immagine dalla categoria `linea`, `no_linea`, oppure un'immagine casuale.
    -   **Augmentation in Test:** Prima della predizione, all'immagine selezionata viene applicata una **data augmentation manuale**, che replica quella usata nel training (`Rotation`, `Zoom`, `Brightness`). Anche qui viene usato il **riempimento a specchio (`reflect`)** per evitare bordi neri. I **parametri esatti** di rotazione, zoom e luminosità applicati vengono mostrati nel titolo del grafico.
    -   Mostra l'immagine (aumentata), l'etichetta reale e la predizione del modello.

### Conclusione
Il progetto possiede una pipeline MLOps pulita, semplice e robusta, specificamente progettata per il task di classificazione binaria. Gli script sono coerenti, facili da usare e pronti per l'addestramento e la valutazione affidabile del modello.
