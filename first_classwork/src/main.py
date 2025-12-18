from data_loader import DataLoader
from frequency_analysis import FrequencyAnalysis
from association_rules import AssociationRules
from clustering_analysis import ClusteringAnalysis

# Dipendenze:
# Devi installare pandas, matplotlib, scikit-learn, mlxtend
# In terminale: pip install pandas matplotlib scikit-learn mlxtend
# pip install hdbscan

def main():
    file_path = "../AnonymizedFidelity.csv" 
    
    # 1. Caricamento e Pulizia Dati
    loader = DataLoader(file_path)
    data = loader.load_data()
    
    if data is None:
        return

    cleaned_data = loader.clean_and_prepare()


    # 2. Esecuzione Task 1 & 2: Analisi di Frequenza
    freq_analyzer = FrequencyAnalysis(cleaned_data)
    freq_analyzer.run_task1()
    print("\n--- FINE Esecuzione Task 1 ---")
    freq_analyzer.run_task2() 
    print("\n--- FINE Esecuzione Task 2 ---")

    # 3. Esecuzione Task 3 & 4: Regole di Associazione (Livello 3)
    # IMPORTANTE: Us0 min_support=0.02 per evitare problemi di memoria
    assoc_analyzer = AssociationRules(cleaned_data)
    assoc_analyzer.run_task3_4(min_support=0.02, min_confidence=0.5) 
    print("\n--- FINE Esecuzione Task 3/4 ---")
    
    # 4. Esecuzione Task 5: PCA e Clustering (Cliente x Prodotto)
    cluster_analyzer = ClusteringAnalysis(cleaned_data)
    cluster_analyzer.run_task5()
    print("\n--- FINE Esecuzione Task 5 ---")

if __name__ == "__main__":
    main()