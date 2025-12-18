import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import os

class AssociationRules:
    """
    Gestisce la trasformazione del dataset e l'applicazione degli algoritmi APRIORI e FP-Growth
    per creare le regole di associazione al livello di merchandising 4.
    """
    def __init__(self, df):
        self.df = df
        self.level_col = 'descr_liv4' # Livello richiesto per le regole di associazione
        self.results_dir = '../results/'
        os.makedirs(self.results_dir, exist_ok=True)

    def prepare_transactions(self):
        """
        Trasforma il DataFrame in un formato transazionale: una lista di prodotti (descr_liv4)
        acquistati per ogni scontrino (scontrino_id).
        """
        print("\nPreparazione del dataset in formato transazionale (Scontrino -> Prodotti)...")
        
        if self.level_col not in self.df.columns or 'scontrino_id' not in self.df.columns:
            raise KeyError("Colonne 'descr_liv3' o 'scontrino_id' non trovate. Verifica la pulizia dei nomi.")
            
        transactions_df = self.df[['scontrino_id', self.level_col]].copy()

        # Raggruppa i prodotti per scontrino in una lista
        transaction_list = (
            transactions_df.groupby('scontrino_id')[self.level_col]
            .apply(list)
            .tolist()
        )
        print(f"Numero totale di transazioni (scontrini unici): {len(transaction_list)}")
        
        # Converte la lista di transazioni in formato One-Hot Encoding (richiesto da mlxtend)
        te = TransactionEncoder()
        te_ary = te.fit(transaction_list).transform(transaction_list)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
        
        return df_encoded

    def apply_algorithm(self, df_encoded, algorithm='apriori', min_support=0.01):
        """
        Applica APRIORI o FP-Growth per trovare i set di oggetti frequenti (Frequent Itemsets).
        """
        if algorithm == 'apriori':
            frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
        elif algorithm == 'fpgrowth':
            frequent_itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)
        else:
            raise ValueError("Algoritmo specificato non supportato.")

        return frequent_itemsets

    def generate_rules(self, frequent_itemsets, min_confidence=0.5):
        """
        Genera le regole di associazione (Antecedents -> Consequents).
        """
        # Calcola le metriche, usando Lift > 1.0 come soglia minima di interesse
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
        
        # Filtra sulla confidenza minima richiesta
        rules = rules[rules['confidence'] >= min_confidence]
        
        # Ordina per Lift (forza dell'associazione), Confidenza e Supporto
        rules = rules.sort_values(['lift', 'confidence', 'support'], ascending=[False, False, False])
        
        return rules

    def run_task3_4(self, min_support=0.05, min_confidence=0.5):
        """Esegue le Task 3 (APRIORI) e 4 (FP-Growth)."""
        
        # Nota sul Supporto: min_support=0.001 (0.1%) è il valore di partenza. 
        # Potrebbe essere necessario aumentarlo (es. 0.005) se il calcolo è troppo lento o la memoria non è sufficiente.
        
        try:
            df_encoded = self.prepare_transactions()
        except KeyError as e:
            print(f"ERRORE: Preparazione transazioni fallita. {e}")
            return
        except Exception as e:
             print(f"ERRORE inatteso durante la preparazione delle transazioni: {e}")
             return
        
        # --- Task 3: APRIORI ---
        print("\n--- Esecuzione Task 3: Algoritmo APRIORI ---")
        try:
            frequent_apriori = self.apply_algorithm(df_encoded, algorithm='apriori', min_support=min_support)
            
            if frequent_apriori.empty:
                print(f"ATTENZIONE: Nessun itemset frequente APRIORI trovato con min_support={min_support}.")
            else:
                rules_apriori = self.generate_rules(frequent_apriori, min_confidence=min_confidence)
                print(f"Regole APRIORI trovate ({len(rules_apriori)}). Salvataggio in results/rules_apriori.csv")
                rules_apriori.to_csv(os.path.join(self.results_dir, 'rules_apriori.csv'), index=False)
        except MemoryError:
            print(f"ERRORE MEMORIA: APRIORI ha esaurito la RAM. Prova ad aumentare min_support.")
        
        # --- Task 4: FP-GROWTH ---
        print("\n--- Esecuzione Task 4: Algoritmo FP-GROWTH ---")
        try:
            frequent_fpgrowth = self.apply_algorithm(df_encoded, algorithm='fpgrowth', min_support=min_support)

            if frequent_fpgrowth.empty:
                print(f"ATTENZIONE: Nessun itemset frequente FP-GROWTH trovato con min_support={min_support}.")
            else:
                rules_fpgrowth = self.generate_rules(frequent_fpgrowth, min_confidence=min_confidence)
                print(f"Regole FP-GROWTH trovate ({len(rules_fpgrowth)}). Salvataggio in results/rules_fpgrowth.csv")
                rules_fpgrowth.to_csv(os.path.join(self.results_dir, 'rules_fpgrowth.csv'), index=False)
        except MemoryError:
            print(f"ERRORE MEMORIA: FP-GROWTH ha esaurito la RAM. Prova ad aumentare min_support.")