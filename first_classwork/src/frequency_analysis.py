import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os

RESULTS_DIR = '../results/'

class FrequencyAnalysis:
    """
    Esegue l'analisi di frequenza e la visualizzazione per i livelli di merchandising.
    """
    def __init__(self, df):
        self.df = df
        # Usiamo i nomi delle colonne pulite
        self.merch_levels = ['descr_liv1', 'descr_liv2', 'descr_liv3', 'descr_liv4'] 
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
    def compute_frequency(self, level_column, df_subset=None):
        """Calcola la frequenza (sul subset o sull'intero DF)."""
        df_to_use = df_subset if df_subset is not None else self.df
        
        frequency = df_to_use[level_column].value_counts().sort_values(ascending=False)
        total = frequency.sum()
        
        freq_df = pd.DataFrame({
            'Assoluta': frequency,
            'Relativa (%)': (frequency / total) * 100
        })
        
        return freq_df

    def create_bar_plots(self, level_column, freq_df, prefix=""):
        """Crea e salva i grafici per i 5 più e meno frequenti."""
        
        level_name = level_column.replace('descr_', '').upper()
        
        # 1. Top 5 
        top_5 = freq_df.head(5)
        
        plt.figure(figsize=(10, 6))
        top_5['Assoluta'].plot(kind='bar', color='skyblue')
        plt.title(f'Top 5 Categorie {level_name} {prefix}'.strip())
        plt.ylabel('Frequenza Assoluta (Righe Prodotto)')
        plt.xlabel('Categoria')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'{prefix}_top5_freq_{level_name}.png'))
        plt.close()
        
        # 2. Bottom 5 
        bottom_5 = freq_df[freq_df['Assoluta'] > 0].tail(5) 
        
        plt.figure(figsize=(10, 6))
        bottom_5['Assoluta'].plot(kind='bar', color='salmon')
        plt.title(f'Bottom 5 Categorie {level_name} {prefix}'.strip())
        plt.ylabel('Frequenza Assoluta (Righe Prodotto)')
        plt.xlabel('Categoria')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'{prefix}_bottom5_freq_{level_name}.png'))
        plt.close()

    def run_task1(self):
        """Esegue l'intera Task 1 per tutti i livelli."""
        print("\n--- Esecuzione Task 1: Analisi di Frequenza Globale ---")
        
        for level in self.merch_levels:
            if level in self.df.columns:
                print(f"\nAnalisi per la colonna: {level}")
                freq_df = self.compute_frequency(level)
                
                print(f"Top 5 di {level}:\n{freq_df.head(5)}")
                print(f"\nBottom 5 di {level}:\n{freq_df[freq_df['Assoluta'] > 0].tail(5)}")
                
                self.create_bar_plots(level, freq_df, prefix="GLOBALE")
            else:
                print(f"ERRORE: Colonna '{level}' non trovata nel dataset.")


    def stratify_by_month(self):
        """Stratifica il dataset in 3 intervalli di mesi (Task 2)."""
        df = self.df.copy()
        df['month'] = df['data'].dt.month
        
        is_range_1 = (df['month'].isin([1, 2, 3, 4])) | \
                     ((df['month'] == 5) & (df['data'].dt.day <= 15))
                     
        is_range_2 = ((df['month'] == 5) & (df['data'].dt.day > 15)) | \
                     (df['month'].isin([6, 7, 8, 9]))
                     
        is_range_3 = df['month'].isin([10, 11, 12])

        ranges = {
            'RANGE_1_GEN_MAG': df[is_range_1],
            'RANGE_2_MAG_SET': df[is_range_2],
            'RANGE_3_OTT_DIC': df[is_range_3]
        }
        return ranges

    def stratify_by_hour(self):
        """Stratifica il dataset in 3 fasce orarie (Task 2)."""
        df = self.df.copy()
        
        slot1_start = pd.to_datetime('08:30:00').time()
        slot1_end   = pd.to_datetime('12:30:00').time()
        slot2_start = pd.to_datetime('12:30:00').time()
        slot2_end   = pd.to_datetime('16:30:00').time()
        slot3_start = pd.to_datetime('16:30:00').time()
        slot3_end   = pd.to_datetime('20:30:00').time()

        try:
            df['time_only'] = pd.to_datetime(df['data'].dt.date.astype(str) + ' ' + df['ora']).dt.time
        except Exception:
            df['time_only'] = pd.to_datetime(df['ora'], format='%H:%M:%S', errors='coerce').dt.time
            df.dropna(subset=['time_only'], inplace=True)

        def get_slot(t):
            if slot1_start <= t < slot1_end:  
                return 'SLOT_1_MATTINA'
            elif slot2_start <= t < slot2_end: 
                return 'SLOT_2_PRANZO'
            elif slot3_start <= t <= slot3_end:
                return 'SLOT_3_SERA'
            return 'FUORI_ORARIO'

        df['time_slot'] = df['time_only'].apply(get_slot)
        
        slots = {
            slot: df[df['time_slot'] == slot] 
            for slot in ['SLOT_1_MATTINA', 'SLOT_2_PRANZO', 'SLOT_3_SERA']
        }
        
        return slots

    def run_task2(self):
        """Esegue l'intera Task 2: stratificazione mensile e oraria."""
        print("\n--- Esecuzione Task 2: Stratificazione per Mese ---")
        monthly_ranges = self.stratify_by_month()
        
        for name, subset_df in monthly_ranges.items():
            print(f"\nAnalisi per Range Mensile: {name} (Righe: {len(subset_df)})")
            for level in self.merch_levels:
                if len(subset_df) > 0:
                    freq_df = self.compute_frequency(level, subset_df)
                    self.create_bar_plots(level, freq_df, prefix=name)
                    
        print("\n--- Esecuzione Task 2: Stratificazione per Orario ---")
        hourly_slots = self.stratify_by_hour()
        
        for name, subset_df in hourly_slots.items():
            print(f"\nAnalisi per Fascia Oraria: {name} (Righe: {len(subset_df)})")
            for level in self.merch_levels:
                if len(subset_df) > 0:
                    freq_df = self.compute_frequency(level, subset_df)
                    self.create_bar_plots(level, freq_df, prefix=name)