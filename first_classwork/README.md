# IDM-2025: Customer Behavior Analysis

Questo progetto analizza i pattern di acquisto dei clienti basandosi su un dataset transazionale di 2.3 milioni di righe. L'obiettivo è estrarre conoscenza tramite analisi di frequenza, regole di associazione e tecniche di clustering.

## Struttura del Progetto
- `src/main.py`:  Punto di ingresso che coordina l'esecuzione delle 5 Task.
- `src/data_loader.py`: Caricamento e pulizia dei dati.
- `src/frequency_analysis.py`: Task 1 & 2 - Analisi delle frequenze assolute/relative e stratificazione temporale (mensile e oraria).
- `src/association_rules.py`: Task 3 & 4 - Mining di regole di associazione tramite algoritmi Apriori e FP-Growth.
- `src/clustering_analysis.py`: Task 5 - Segmentazione della clientela tramite riduzione dimensionale e clustering.

## Requisiti
Le dipendenze Python necessarie sono:
- `pandas`
- `matplotlib`
- `scikit-learn`
- `mlxtend`
- `numpy`

```bash
pip install pandas matplotlib scikit-learn mlxtend numpy

