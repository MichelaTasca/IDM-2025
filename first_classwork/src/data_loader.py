import pandas as pd
import numpy as np
import re # Usato per pulizia estrema

class DataLoader:
    """
    Gestisce il caricamento del dataset e la pulizia/preparazione iniziale.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_data(self):
        """Carica il dataset da file CSV/TSV."""
        try:
            # Uso il punto e virgola (;) come separatore e una codifica robusta
            self.df = pd.read_csv(self.file_path, sep=',', encoding='cp1252', on_bad_lines='skip')
            print(f"Dataset caricato. Righe iniziali: {len(self.df)}")
            return self.df
        except FileNotFoundError:
            print(f"ERRORE: File non trovato al percorso specificato: {self.file_path}")
            return None
        except Exception as e:
            print(f"ERRORE durante il caricamento del file: {e}")
            return None

    def clean_and_prepare(self):
        """
        Esegue la pulizia iniziale dei dati, uniformando i nomi delle colonne
        e filtrando gli shoppers (Task 1).
        """
        if self.df is None:
            return None

        # 1. Pulizia ESTREMA dei nomi delle colonne
        self.df.columns = self.df.columns.str.strip()
        
        # Pulizia estrema con regex: rimuove tutti i caratteri non-parola e non-spazio, 
        # rende minuscolo e converte gli spazi in underscore.
        self.df.columns = (
            self.df.columns
            .str.lower()
            .str.replace(r'[^\w\s]', '_', regex=True) # Sostituisce non-parola/non-spazio con underscore
            .str.replace(r'\s+', '_', regex=True)    # Sostituisce spazi multipli con un singolo underscore
            .str.strip('_')
        )
        
        # 2. Controllo di coerenza sulla colonna chiave 'descr_prod' (causa del NoneType)
        
        # Cerchiamo colonne che contengono 'descr_prod' (per catturare eventuali errori di codifica)
        possible_cols = [col for col in self.df.columns if 'descr_prod' in col]   
        
        # 3. Escludere 'shoppers' dalla colonna 'descr_prod' (Task 1)
        initial_rows = len(self.df)
        self.df = self.df[~self.df['descr_prod'].str.contains('shopper', case=False, na=False)]
        print(f"Esclusi {initial_rows - len(self.df)} 'shoppers' dal dataset.")

        # 4. Conversione tipi di dati e pulizia valori
        
        if 'data' in self.df.columns:
            self.df['data'] = pd.to_datetime(self.df['data'])
        if 'ora' in self.df.columns:
            self.df['ora'] = self.df['ora'].astype(str).str.strip()

        merch_cols_clean = ['descr_liv1', 'descr_liv2', 'descr_liv3', 'descr_liv4'] 
        existing_merch_cols = [col for col in merch_cols_clean if col in self.df.columns]
        
        # Rimuovi NaNs dai livelli di merchandising e strip i valori
        if existing_merch_cols:
            self.df.dropna(subset=existing_merch_cols, inplace=True)
            for col in existing_merch_cols:
                self.df[col] = self.df[col].astype(str).str.strip()

        return self.df

    def get_data(self):
        return self.df