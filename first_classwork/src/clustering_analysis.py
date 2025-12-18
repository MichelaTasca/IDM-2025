import pandas as pd
from sklearn.preprocessing import MaxAbsScaler # Migliore per matrici sparse
from sklearn.decomposition import TruncatedSVD 
import hdbscan 
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')

RESULTS_DIR = '../results/'

class ClusteringAnalysis:
    def __init__(self, df):
        self.df = df[df['tessera'].notna() & (df['tessera'] != '')].copy()
        self.results_dir = RESULTS_DIR
        self.customer_id_col = 'tessera'
        self.product_id_col = 'cod_prod' 
        os.makedirs(self.results_dir, exist_ok=True)

    def create_customer_product_matrix(self):
        print("\nCreazione della matrice Cliente x Prodotto...")
        frequency_df = self.df.groupby([self.customer_id_col, self.product_id_col]).size().reset_index(name='freq')
        # Creiamo una matrice sparsa per risparmiare memoria
        customer_product_matrix = frequency_df.pivot(
            index=self.customer_id_col, 
            columns=self.product_id_col, 
            values='freq'
        ).fillna(0)
        return customer_product_matrix

    def apply_svd(self, data_matrix, n_components=100):
        """Applica Truncated SVD (LSA) al posto della PCA."""
        print(f"Esecuzione Truncated SVD (n={n_components})...")
        
        # Per dati sparsi si usa MaxAbsScaler invece di StandardScaler
        scaler = MaxAbsScaler()
        data_scaled = scaler.fit_transform(data_matrix)
        
        # TruncatedSVD non centra i dati, preservando la sparsità
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        svd_result = svd.fit_transform(data_scaled)
        
        explained_variance = np.sum(svd.explained_variance_ratio_)
        print(f"Varianza spiegata dalla SVD: {explained_variance:.2f}")
        
        return pd.DataFrame(svd_result, index=data_matrix.index)

    def apply_clustering(self, data_df):
        print(f"Applicazione Clustering HDBSCAN sui risultati SVD...")
        # Proviamo ad abbassare min_cluster_size se HDBSCAN non trova nulla
        hdb = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5, prediction_data=True)
        clusters = hdb.fit_predict(data_df)
        
        data_df['Cluster'] = clusters
        n_clusters = len([c for c in np.unique(clusters) if c != -1])
        n_noise = list(clusters).count(-1)
        print(f"Risultato: {n_clusters} cluster trovati e {n_noise} punti di rumore.")
        
        # Visualizzazione
        plt.figure(figsize=(10, 8))
        plt.scatter(data_df.iloc[:, 0], data_df.iloc[:, 1], c=data_df['Cluster'], cmap='plasma', s=10, alpha=0.5)
        plt.title('SVD + HDBSCAN Clustering')
        plt.savefig(os.path.join(self.results_dir, 'svd_hdbscan_plot.png'))
        plt.close()
        
        data_df[['Cluster']].to_csv(os.path.join(self.results_dir, 'client_clusters_svd.csv'))

    def run_task5(self):
        matrix = self.create_customer_product_matrix()
        if matrix is not None:
            reduced_data = self.apply_svd(matrix)
            self.apply_clustering(reduced_data)