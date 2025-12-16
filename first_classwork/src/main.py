from data_loader import DataLoader
from frequency_analysis import FrequencyAnalysis
from association_rules import AssociationRules

# Dipendenze:
# Devi installare pandas e matplotlib
# In terminale: pip install pandas matplotlib

def main():
    file_path = "../AnonymizedFidelity.csv" 
    
    # 1. Caricamento e Pulizia Dati
    loader = DataLoader(file_path)
    data = loader.load_data()
    
    if data is None:
        return

    cleaned_data = loader.clean_and_prepare()


    # 2. Esecuzione Task 1: Analisi di Frequenza
    freq_analyzer = FrequencyAnalysis(cleaned_data)
    freq_analyzer.run_task1()

    
    print("\n--- FINE Esecuzione Task 1 ---")

    freq_analyzer.run_task2() 
    
    print("\n--- FINE Esecuzione Task 2 ---")

    assoc_analyzer = AssociationRules(cleaned_data)
    assoc_analyzer.run_task3_4(min_support=0.01, min_confidence=0.5)

    print("\n--- FINE Esecuzione Task 3 ---")


if __name__ == "__main__":
    main()