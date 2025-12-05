# Modulo di Intelligenza Artificiale - Pickles Bot

Questo modulo contiene il cuore del progetto: gli script per addestrare e testare i modelli di classificazione delle immagini.

## Workflow Rapido

1.  **Prepara i Dati**:
    -   Crea una cartella per ogni classe in `intelligenza_artificiale/dataset/dataset_bot/`.
    -   Esempio: `.../dataset_bot/cane/`, `.../dataset_bot/gatto/`.
    -   Metti le immagini corrispondenti in ogni cartella. Non sono necessarie sottocartelle `training` o `test`.

2.  **Addestra il Modello**:
    -   Esegui `python intelligenza_artificiale/train.py`.
    -   Scegli l'opzione **2. Dataset Bot**.
    -   Attendi che il training si completi o **interrompilo quando vuoi** (`q` + Invio). Un nuovo modello verrà salvato in `modelli_salvati/modelli_salvati_bot/`.

3.  **Testa il Modello**:
    -   Esegui `python intelligenza_artificiale/test.py`.
    -   Scegli il modello appena creato dalla lista.
    -   Usa le opzioni interattive per valutare l'accuratezza.

## Script Principali

-   **`train.py`**:
    Script di addestramento potenziato. Utilizza un modello pre-addestrato (EfficientNetB0) e lo adatta ai tuoi dati con una tecnica a due fasi. Include data augmentation, arresto automatico (Early Stopping) e la possibilità di **interruzione manuale sicura**.

-   **`test.py`**:
    Script di testing interattivo. Carica un modello e lo valuta usando lo **stesso identico set di validazione** creato da `train.py`, garantendo un confronto diretto e affidabile con le metriche di training.

---

*Per una spiegazione tecnica dettagliata dell'architettura, del workflow e del funzionamento degli script, consulta la [**Documentazione Avanzata**](./documentazione.md).*