# Documentazione Interna Gemini

Questo documento `GEMINI.md` serve come memoria interna e registro di contesto per l'agente Gemini. Contiene un riepilogo delle sessioni precedenti, decisioni chiave, struttura del progetto, convenzioni di denominazione, percorsi dei file e dettagli tecnici specifici che sono cruciali per la comprensione e la continuazione dello sviluppo del progetto Pickles Bot. È mantenuto in inglese per coerenza con la lingua di interazione dell'agente e per facilitare la comprensione dei dettagli tecnici.

**Important:** This `GEMINI.md` file should be updated whenever there are changes to the project structure, file purposes, paths, or formatting conventions.

## Riepilogo della Sessione Attuale (Per la Prossima Sessione)

This session focused on implementing model interpretability and further optimizing the training process. The journey was complex, involving deep debugging of Keras model loading and graph execution, ultimately leading to a successful implementation of Occlusion Sensitivity.

**1. Training Optimization (Initial Request & `train.py`):**
-   **Objective:** To improve training speed and effectiveness.
-   **`MobileNetV3Small` Integration:** Replaced `EfficientNetB0` with `MobileNetV3Small` as the base model in `train.py` for faster inference on resource-constrained devices, such as a robot.
    -   Updated preprocessing from `efficientnet.preprocess_input` to `mobilenet_v3.preprocess_input`.
    -   Adjusted fine-tuning layers from 80 to 40 (`fine_tune_at = -40`) to better suit the smaller `MobileNetV3Small` architecture.

**2. Model Interpretability (Initial Grad-CAM Attempt & `heatmap.py`):**
-   **Objective:** Implement a Grad-CAM visualization tool (`heatmap.py`) to understand model decisions.
-   **Initial Implementation & Debugging:**
    -   Created `intelligenza_artificiale/debug/heatmap.py`.
    -   **Path Issues:** Corrected `SCRIPT_DIR` logic after file relocation to `debug/`.
    -   **Dynamic Preprocessing:** Implemented logic to apply correct preprocessing based on model name (`EfficientNet` or `MobileNet`).
    -   **Class Name Retrieval:** Added robust logic to retrieve class names from `tfds` (for CIFAR-10) or hardcoded lists (for custom datasets).
    -   **Random Image Selection:** Implemented functionality to automatically select and process a random image from the validation set, with real vs. predicted labels shown.
    -   **Grad-CAM Layer Finding Bugs (Repeated `KeyError`):** Faced persistent `KeyError` when attempting to build a `tf.keras.Model` to extract intermediate layer outputs for Grad-CAM. This proved to be an **insurmountable incompatibility** with the loaded Keras model's internal graph, even for newly trained models and when using the `tf-keras-vis` library. The problem was not in the script's logic, but fundamentally with Keras's serialization and deserialization, making Grad-CAM impossible in this environment.
-   **Decision:** The Grad-CAM implementation was declared unfixable due to deep TensorFlow/Keras graph issues. The `heatmap.py` script, its `debug` folder, and the `tf-keras-vis` dependency were removed.

**3. Model Interpretability (Occlusion Sensitivity & `heatmap.py` - Second Attempt):**
-   **Objective:** To provide a simpler, more robust interpretability solution that works.
-   **Implementation:**
    -   Recreated `intelligenza_artificiale/debug/heatmap.py` with **Occlusion Sensitivity**.
    -   This method avoids complex graph traversal by systematically occluding parts of the image and observing prediction changes.
    -   The script reuses all previously developed utility functions (`seleziona_modello`, `carica_immagine_da_path`, `applica_preprocessing`, `carica_dataset_di_validazione`, `sovrapponi_heatmap`).
-   **Debugging Occlusion Sensitivity:**
    -   **Misidentified Base Model:** Initially, `base_model` detection found the `data_augmentation` layer (a `Sequential` model) instead of the actual base model (`MobileNet` or `EfficientNet`). Fixed by excluding 'augmentation' from the search.
    -   **Incorrect Dtype (`uint8` vs `float32`):** `tf-keras-vis` warned about `uint8` dtype. Fixed by explicitly casting image arrays to `float32` in `applica_preprocessing`.
    -   **Plot Not Showing:** User reported plot not appearing. Fixed by adding `matplotlib.use('TkAgg')` to explicitly force an interactive backend.
    -   **Slow Generation:** Occlusion Sensitivity is computationally intensive. Reduced `patch_size` to 24 and `stride` to 12 in `genera_occlusion_heatmap` for faster, interactive feedback (trading resolution for speed).
-   **Current State:** The `heatmap.py` script with Occlusion Sensitivity is now functional and ready for use.

**Stato Attuale:** The project's training pipeline in `train.py` is updated for better performance on edge devices. The `heatmap.py` script now provides a working model interpretability solution using Occlusion Sensitivity, which is simpler and more robust than Grad-CAM in this environment.

---
## (Previous session summaries are kept below for historical context)

This session focused on implementing several "real-world", advanced training techniques to improve the model's effectiveness and stability, followed by a documentation update.

**1. Advanced Training Strategy Implementation (`train.py`):**
-   **Aggressive Fine-Tuning:** The fine-tuning strategy was made more aggressive by unfreezing the last **80 layers** of `EfficientNetB0` (up from 20), allowing the model to adapt more significantly to the target dataset.
-   **Dynamic Learning Rate Schedule:** The previous `ReduceLROnPlateau` callback was replaced with a smoother `ExponentialDecay` schedule for both training phases.
-   **Label Smoothing:** Implemented label smoothing as a regularization technique. This involved:
    -   Modifying both data loaders (`carica_e_prepara_benchmark_dataset` and `carica_dataset_custom`) to **one-hot encode** the labels.
    -   Changing the loss function in all `model.compile()` calls to `tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)`.
-   **Learning Rate Warmup:** To stabilize the beginning of the aggressive fine-tuning phase, a custom `WarmupExponentialDecay` learning rate schedule was created and implemented for Phase 2. It warms up the learning rate linearly over 2 epochs before starting the decay.
-   **Data Pipeline Correction:** Fixed the `carica_dataset_custom` pipeline to ensure the training data is shuffled properly before every epoch (`.shuffle(1000, reshuffle_each_iteration=True)`).

**2. GPU/Mixed Precision (Implemented and Reverted):**
-   Initially, to address performance, conditional GPU detection and **mixed-precision training** (`mixed_float16`) were implemented in `configura_tensorflow`.
-   Per the user's explicit request, as they do not have a GPU, this functionality was **reverted**, and the script was returned to a **CPU-only** optimized configuration.

**3. Documentation Update:**
-   **`documentazione.md`:** The user-facing technical documentation was completely updated to reflect all the new training strategies: the 80-layer fine-tuning, `ExponentialDecay` and `WarmupExponentialDecay` schedules, and the implementation of Label Smoothing.
-   **`GEMINI.md`:** This internal memory file has been updated with this summary.

**Stato Attuale:** The `train.py` script is now highly optimized for **CPU-based training**, incorporating modern techniques like label smoothing and learning rate warmup to improve model stability and generalization. The "continue training" feature is fully compatible with these new techniques. The project documentation is fully aligned with the current state of the code.

---
## (Previous session summaries are kept below for historical context)

This session focused on a major refactoring and alignment of the training and testing scripts, significant user experience improvements, and a complete overhaul of the project's documentation to ensure consistency and accuracy.

**1. User Experience (`train.py`):**
-   **Graceful Shutdown:** Implemented a feature to allow the user to gracefully stop the training by pressing 'q' followed by Enter. This is managed by a background `threading` process and a custom `QuitCallback`.
-   **Live Training Feedback:** Created a `CustomEarlyStopping` callback to provide real-time feedback on the early stopping patience counter (e.g., `Pazienza: 3/15`), giving the user a clear indication of how close the training is to stopping automatically.

**2. Performance Tuning & Strategy (`train.py`):**
-   **Cautious Fine-Tuning:** The fine-tuning strategy was adjusted for more stable learning. The number of unfrozen layers in Phase 2 was set to the last **20** of the EfficientNetB0 base model, a more conservative approach to prevent catastrophic forgetting.
-   **Aggressive Data Augmentation:** The `crea_augmentation_layer` function was enhanced with `RandomTranslation`, `RandomBrightness`, and `RandomContrast` to improve model generalization.
-   **Benchmark Dataset Update:** The standard benchmark dataset was switched from `tf_flowers` to the more robust **`cifar10`**.

**3. Bug Fixing & Critical Alignments:**
-   **Data Loading Inconsistency (`test.py`):** Resolved a major bug where `test.py` and `train.py` were using different datasets for validation/testing. `test.py` was modified to use the exact same `validation_split` logic as `train.py`, ensuring test results are consistent and comparable to training validation metrics.
-   **Benchmark Testing (`test.py`):**
    -   Enabled full interactive testing for models trained on the `cifar10` dataset.
    -   Fixed a critical `ValueError` related to input shape by adding the missing **image resize transformation** when loading the `cifar10` data (from 32x32 to the model's expected 96x128).
-   **NumPy 2.0 Compatibility (`train.py`):** Fixed a crashing `AttributeError` by replacing all instances of the deprecated `np.Inf` with the correct `np.inf`.
-   **Sequential Log File Naming (`test.py`):** Log files in `test.py` are now saved with sequential numbers (e.g., `logs_tf_1.txt`, `logs_tf_2.txt`) to prevent overwriting and maintain a history of test sessions.

**4. Documentation Overhaul:**
-   **Comprehensive Update:** All user-facing and internal documentation was updated to reflect the latest code.
-   **Advanced Documentation (`documentazione.md`):** The technical guide was corrected to show the new `cifar10` benchmark, the 20-layer fine-tuning, and the new UX features.
-   **READMEs (`/` and `/intelligenza_artificiale`):** Both README files were updated. The root `README.md`'s detailed development history was rewritten to accurately narrate the debugging and refactoring process.

**5. Training Continuation Feature (`train.py`):**
-   **New Menu Options:** The `train.py` script now offers a comprehensive menu allowing users to "Start New Training" or "Continue Training" for both CIFAR-10 and custom "Bot" datasets.
-   **State Saving (`training_state.json`):**
    -   A new `training_state.json` file is saved within each session directory upon interruption or completion of a training run.
    -   This file stores critical information required for seamless continuation, including: `last_epoch`, `dataset_name`, `model_relative_path`, `learning_rate`, the full `history` of metrics, the **`current_phase`** (`phase1` or `phase2`), and the `base_model_trainable` status.
    -   The `training_state.json` is automatically removed if a continued training session completes naturally, signifying its finalization.
-   **Phase-Aware Continuation (`continua_addestramento`):**
    -   The `continua_addestramento` function intelligently loads the saved state and resumes training from the exact point of interruption.
    -   It is **phase-aware**: if interrupted during Phase 1, it continues Phase 1 with its specific callbacks and learning rate. Upon natural completion of Phase 1, it automatically transitions to Phase 2. If interrupted during Phase 2, it continues Phase 2 directly.
    -   The model is re-compiled with the appropriate learning rate and the `base_model`'s trainable status is correctly set for the resumed phase.
-   **Consistent Session Logging:** The `addestra_modello` function now ensures that a unique session directory is created at the start of training (even if interrupted during Phase 1) and that all subsequent saves (e.g., after Phase 2 completion or further interruptions) use this same session directory. This guarantees that all artifacts (model, graph, logs, state file) for a single training run are grouped together.
-   **Timestamped Logging:** The `salva_risultati` function now creates a new, uniquely timestamped log file (e.g., `log_sessione_YYYYMMDD_HHMMSS.txt`) within the session directory upon each save. This ensures that a historical record of all saves (interruptions or completions) is preserved, rather than overwriting a single log file.

**Stato Attuale:** Il progetto è ora significativamente più robusto, coerente e ben documentato. Gli script di training e testing sono allineati, risolvendo bug critici che rendevano i test precedenti inaffidabili. Il sistema è pronto per un addestramento affidabile sul dataset finale dell'utente, con la flessibilità aggiuntiva di poter interrompere e riprendere il training in qualsiasi momento, mantenendo lo stato e lo storico completi.

---
## Strategie di Ottimizzazione Future

Questa sezione elenca ulteriori strategie per migliorare le performance del modello, da considerare se l'accuratezza desiderata (90-95%) non viene raggiunta con il dataset finale e le attuali ottimizzazioni.

### 1. Provare un Modello Base più Potente
-   **Descrizione:** `EfficientNetB0` è un ottimo punto di partenza. Se necessitiamo di maggiore potenza di calcolo e capacità di apprendimento, possiamo passare a modelli leggermente più grandi della stessa famiglia o ad architetture alternative.
-   **Esempi:** `EfficientNetB1`, `EfficientNetB2`, o `MobileNetV3Large`.
-   **Considerazioni:** Questi modelli sono più pesanti, richiedono più tempo di training e maggiori risorse hardware (RAM/VRAM).

### 2. Usare uno Scheduler del Learning Rate Diverso
-   **Descrizione:** Attualmente utilizziamo `ReduceLROnPlateau`, che è reattivo ai plateau di performance. Un'alternativa è uno scheduler che segue una curva predefinita, come `CosineDecay`. Questo approccio riduce il learning rate in modo fluido e continuo durante il training.
-   **Vantaggi:** Può aiutare il modello a convergere verso un minimo globale migliore, specialmente nelle fasi finali del training.
-   **Considerazioni:** Richiede una configurazione più attenta del numero totale di step/epoche.

### 3. Ottimizzazione degli Iperparametri (Grid Search / Random Search)
-   **Descrizione:** Questo è un approccio sistematico per trovare la migliore combinazione di iperparametri (es. learning rate iniziale, tassi di dropout, numero di layer scongelati, dimensione del batch). Si tratta di eseguire molteplici sessioni di training con diverse configurazioni.
-   **Vantaggi:** Può portare a un'ottimizzazione fine delle performance.
-   **Considerazioni:** È estremamente dispendioso in termini di tempo e risorse computazionali, poiché richiede l'esecuzione di numerosi training completi. Da considerare come ultima risorsa.

---
## (Previous session summaries are kept below for historical context)
...