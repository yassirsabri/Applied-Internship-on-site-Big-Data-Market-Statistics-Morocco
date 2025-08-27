# Analyse Statistique Avancée des Marchés Publics Marocains
# Version améliorée avec graphiques sans chevauchement

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import warnings
warnings.filterwarnings('ignore')

# Configuration matplotlib pour l'affichage français
plt.rcParams['font.size'] = 11
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")
sns.set_palette("husl")

class AnalyseMarchesPublicsAmeliore:
    """Classe améliorée pour l'analyse des marchés publics avec graphiques optimisés"""
    
    def __init__(self, data_path=None, df=None):
        """Initialisation avec données"""
        if df is not None:
            self.df = df.copy()
        elif data_path:
            self.df = pd.read_excel(data_path)
        else:
            raise ValueError("Fournir soit data_path soit df")
        
        self.df_encoded = None
        self.pca = None
        self.scaler = None
        self.label_encoders = {}
        
    def preprocess_data(self):
        """Préparation et encodage des données"""
        print("=== PRÉPARATION DES DONNÉES ===")
        
        # Nettoyage des données
        self.df['Référence'] = pd.to_datetime(self.df['Référence'], errors='coerce')
        self.df['Publié le'] = pd.to_datetime(self.df['Publié le'], errors='coerce')
        
        # Création de variables dérivées
        self.df['Année'] = self.df['Référence'].dt.year
        self.df['Mois'] = self.df['Référence'].dt.month
        self.df['Délai_publication'] = (self.df['Référence'] - self.df['Publié le']).dt.days
        
        # Variables pour l'analyse
        categorical_vars = ['Procédure', 'Catégorie', 'Type de dépense personnalisé', 
                           'region', 'Acheteur public']
        
        # Encodage des variables catégorielles
        df_for_analysis = self.df.copy()
        
        for var in categorical_vars:
            if var in df_for_analysis.columns:
                le = LabelEncoder()
                df_for_analysis[f'{var}_encoded'] = le.fit_transform(
                    df_for_analysis[var].fillna('Inconnu').astype(str)
                )
                self.label_encoders[var] = le
        
        # Sélection des variables pour l'analyse
        analysis_vars = [
            'Procédure_encoded', 'Catégorie_encoded', 'Type de dépense personnalisé_encoded',
            'region_encoded', 'Procedure_AOO', 'Procedure_AOS', 'Procedure_CONCA',
            'Procedure_APPEL', 'Procedure_AVIS', 'Categorie_Services',
            'Categorie_Travaux', 'Categorie_Fournitures', 'Année', 'Mois'
        ]
        
        # Filtrer les variables existantes
        analysis_vars = [var for var in analysis_vars if var in df_for_analysis.columns]
        
        self.df_encoded = df_for_analysis[analysis_vars].fillna(0)
        
        print(f"✅ Données préparées: {self.df_encoded.shape[0]} observations, {self.df_encoded.shape[1]} variables")
        
        return self.df_encoded
    
    def plot_pca_scree(self, var_explained, cumvar_explained):
        """Graphique d'éboulis des valeurs propres - Version améliorée"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. Éboulis des valeurs propres
        n_components = len(var_explained)
        x_pos = range(1, n_components + 1)
        
        bars = ax1.bar(x_pos, var_explained, alpha=0.7, color='steelblue', 
                      edgecolor='navy', linewidth=1.2)
        line = ax1.plot(x_pos, var_explained, 'ro-', linewidth=2.5, 
                       markersize=8, color='darkred')
        
        # Ajouter les valeurs sur les barres
        for i, (x, y) in enumerate(zip(x_pos, var_explained)):
            if i < 5:  # Afficher seulement les 5 premières pour éviter l'encombrement
                ax1.text(x, y + 0.005, f'{y:.3f}', ha='center', va='bottom', 
                        fontweight='bold', fontsize=10)
        
        ax1.set_xlabel('Composantes Principales', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Variance Expliquée', fontsize=12, fontweight='bold')
        ax1.set_title('📊 Éboulis des Valeurs Propres\n(Méthode du Coude)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(x_pos)
        
        # 2. Variance cumulée
        line_cum = ax2.plot(x_pos, cumvar_explained, 'go-', linewidth=3, 
                           markersize=10, color='forestgreen', label='Variance cumulée')
        ax2.axhline(y=0.8, color='red', linestyle='--', alpha=0.8, linewidth=2,
                   label='Seuil 80%')
        ax2.axhline(y=0.9, color='orange', linestyle='--', alpha=0.8, linewidth=2,
                   label='Seuil 90%')
        
        # Marquer le point 80%
        idx_80 = np.where(cumvar_explained >= 0.8)[0]
        if len(idx_80) > 0:
            first_80 = idx_80[0]
            ax2.scatter(first_80 + 1, cumvar_explained[first_80], 
                       s=150, color='red', zorder=5)
            ax2.annotate(f'CP{first_80 + 1}\n{cumvar_explained[first_80]:.1%}', 
                        xy=(first_80 + 1, cumvar_explained[first_80]),
                        xytext=(first_80 + 1.5, cumvar_explained[first_80] - 0.1),
                        arrowprops=dict(arrowstyle='->', color='red'),
                        fontsize=10, fontweight='bold')
        
        ax2.set_xlabel('Nombre de Composantes', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Variance Expliquée Cumulée', fontsize=12, fontweight='bold')
        ax2.set_title('📈 Variance Expliquée Cumulée\n(Critère de Kaiser)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)
        ax2.set_xticks(x_pos)
        
        plt.tight_layout()
        plt.show()
        
        # Interprétation textuelle
        print(f"\n📊 INTERPRÉTATION DE L'ACP:")
        print(f"   • Nombre de composantes pour 80% de variance: {sum(cumvar_explained < 0.8) + 1}")
        print(f"   • Nombre de composantes pour 90% de variance: {sum(cumvar_explained < 0.9) + 1}")
        print(f"   • Variance de la 1ère composante: {var_explained[0]:.1%}")
        print(f"   • Variance des 3 premières composantes: {sum(var_explained[:3]):.1%}")
    
    def plot_pca_projections(self, X_pca, var_explained):
        """Projections des individus et cercle des corrélations - Version améliorée"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # 1. Projection des marchés publics
        scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, s=60, 
                             c=range(len(X_pca)), cmap='viridis', edgecolors='black', linewidth=0.5)
        
        ax1.set_xlabel(f'CP1 ({var_explained[0]:.1%} de variance)', fontsize=12, fontweight='bold')
        ax1.set_ylabel(f'CP2 ({var_explained[1]:.1%} de variance)', fontsize=12, fontweight='bold')
        ax1.set_title('🎯 Projection des Marchés Publics\n(Plan Factoriel CP1-CP2)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax1.grid(True, alpha=0.3)
        
        # Ajouter des lignes de référence
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Statistiques sur la projection
        x_range = X_pca[:, 0].max() - X_pca[:, 0].min()
        y_range = X_pca[:, 1].max() - X_pca[:, 1].min()
        ax1.text(0.02, 0.98, f'Dispersion CP1: {x_range:.2f}\nDispersion CP2: {y_range:.2f}', 
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10)
        
        # 2. Cercle des corrélations (si peu de variables)
        if len(self.df_encoded.columns) <= 15:
            correlations = self.pca.components_[:2].T * np.sqrt(self.pca.explained_variance_[:2])
            
            # Cercle unitaire
            circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='--', alpha=0.7, linewidth=2)
            ax2.add_patch(circle)
            
            # Variables
            colors = plt.cm.Set3(np.linspace(0, 1, len(self.df_encoded.columns)))
            
            for i, (var, color) in enumerate(zip(self.df_encoded.columns, colors)):
                # Flèche
                ax2.arrow(0, 0, correlations[i, 0], correlations[i, 1], 
                         head_width=0.04, head_length=0.04, fc=color, ec=color, linewidth=2)
                
                # Étiquette avec fond
                ax2.text(correlations[i, 0]*1.15, correlations[i, 1]*1.15, 
                        var.replace('_encoded', '').replace('_', ' '), 
                        fontsize=9, ha='center', va='center', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                 edgecolor=color, alpha=0.8))
            
            ax2.set_xlim(-1.3, 1.3)
            ax2.set_ylim(-1.3, 1.3)
            ax2.set_xlabel(f'CP1 ({var_explained[0]:.1%})', fontsize=12, fontweight='bold')
            ax2.set_ylabel(f'CP2 ({var_explained[1]:.1%})', fontsize=12, fontweight='bold')
            ax2.set_title('🔄 Cercle des Corrélations\n(Contribution des Variables)', 
                         fontsize=14, fontweight='bold', pad=20)
            ax2.grid(True, alpha=0.3)
            ax2.set_aspect('equal')
            
            # Ajouter des lignes de référence
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
        else:
            # Contributions des variables (pour beaucoup de variables)
            contrib_cp1 = (self.pca.components_[0]**2) / np.sum(self.pca.components_[0]**2) * 100
            contrib_cp2 = (self.pca.components_[1]**2) / np.sum(self.pca.components_[1]**2) * 100
            
            y_pos = np.arange(len(self.df_encoded.columns))
            
            ax2.barh(y_pos - 0.2, contrib_cp1, 0.4, alpha=0.8, label='CP1', color='skyblue')
            ax2.barh(y_pos + 0.2, contrib_cp2, 0.4, alpha=0.8, label='CP2', color='lightcoral')
            
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels([col.replace('_encoded', '').replace('_', ' ') 
                                for col in self.df_encoded.columns], fontsize=9)
            ax2.set_xlabel('Contribution (%)', fontsize=12, fontweight='bold')
            ax2.set_title('📊 Contributions des Variables\n(aux 2 premières CP)', 
                         fontsize=14, fontweight='bold', pad=20)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def perform_pca(self, n_components=None):
        """Analyse en Composantes Principales avec visualisations améliorées"""
        print("\n=== ANALYSE EN COMPOSANTES PRINCIPALES (ACP) ===")
        
        if self.df_encoded is None:
            self.preprocess_data()
        
        # Standardisation
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(self.df_encoded)
        
        # ACP
        if n_components is None:
            n_components = min(10, self.df_encoded.shape[1])
        
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Variance expliquée
        var_explained = self.pca.explained_variance_ratio_
        cumvar_explained = np.cumsum(var_explained)
        
        print(f"📊 Variance expliquée par les {n_components} premières composantes:")
        for i, (var, cumvar) in enumerate(zip(var_explained[:5], cumvar_explained[:5])):
            print(f"   CP{i+1}: {var:.3f} ({cumvar:.3f} cumulé)")
        
        # Visualisations séparées pour éviter le chevauchement
        self.plot_pca_scree(var_explained, cumvar_explained)
        self.plot_pca_projections(X_pca, var_explained)
        
        return X_pca
    
    def plot_clustering_analysis(self, X_scaled, inertias, silhouette_scores, k_range):
        """Analyse de clustering avec graphiques séparés"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. Méthode du coude
        ax1.plot(k_range, inertias, 'bo-', linewidth=3, markersize=10, color='navy')
        
        # Marquer le coude optimal
        if len(inertias) >= 3:
            optimal_k = self._find_elbow(k_range, inertias)
            optimal_idx = optimal_k - k_range[0]
            ax1.scatter(optimal_k, inertias[optimal_idx], s=200, color='red', 
                       zorder=5, marker='*')
            ax1.annotate(f'Optimal K={optimal_k}', 
                        xy=(optimal_k, inertias[optimal_idx]),
                        xytext=(optimal_k + 0.5, inertias[optimal_idx] + max(inertias)*0.1),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2),
                        fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        ax1.set_xlabel('Nombre de Clusters (K)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Inertie (Within-Cluster Sum of Squares)', fontsize=12, fontweight='bold')
        ax1.set_title('📈 Méthode du Coude\n(Détermination du K optimal)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(k_range)
        
        # 2. Scores de silhouette
        k_sil = list(k_range)[1:]
        ax2.plot(k_sil, silhouette_scores, 'ro-', linewidth=3, markersize=10, color='darkgreen')
        
        # Marquer le meilleur score
        best_sil_idx = np.argmax(silhouette_scores)
        best_k = k_sil[best_sil_idx]
        ax2.scatter(best_k, silhouette_scores[best_sil_idx], s=200, color='gold', 
                   zorder=5, marker='*')
        ax2.annotate(f'Meilleur Score\nK={best_k}\nScore={silhouette_scores[best_sil_idx]:.3f}', 
                    xy=(best_k, silhouette_scores[best_sil_idx]),
                    xytext=(best_k + 0.5, silhouette_scores[best_sil_idx] - 0.05),
                    arrowprops=dict(arrowstyle='->', color='gold', lw=2),
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        ax2.set_xlabel('Nombre de Clusters (K)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Score de Silhouette', fontsize=12, fontweight='bold')
        ax2.set_title('🎯 Scores de Silhouette\n(Qualité des Clusters)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(k_sil)
        
        plt.tight_layout()
        plt.show()
    
    def plot_cluster_results(self, X_scaled, cluster_labels):
        """Visualisation des résultats de clustering"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # 1. Projection des clusters
        if self.pca is not None:
            X_pca = self.pca.transform(X_scaled)
            scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, 
                                 cmap='Set1', alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
            ax1.set_xlabel(f'CP1 ({self.pca.explained_variance_ratio_[0]:.1%})', 
                          fontsize=12, fontweight='bold')
            ax1.set_ylabel(f'CP2 ({self.pca.explained_variance_ratio_[1]:.1%})', 
                          fontsize=12, fontweight='bold')
            
            # Centres des clusters
            unique_labels = np.unique(cluster_labels)
            for label in unique_labels:
                mask = cluster_labels == label
                center_x = X_pca[mask, 0].mean()
                center_y = X_pca[mask, 1].mean()
                ax1.scatter(center_x, center_y, s=300, c='black', marker='x', linewidth=4)
                ax1.text(center_x, center_y + 0.3, f'C{label}', ha='center', va='center',
                        fontsize=12, fontweight='bold', color='white',
                        bbox=dict(boxstyle='circle', facecolor='black'))
        else:
            scatter = ax1.scatter(X_scaled[:, 0], X_scaled[:, 1], c=cluster_labels, 
                                 cmap='Set1', alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
            ax1.set_xlabel('Variable 1 (standardisée)', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Variable 2 (standardisée)', fontsize=12, fontweight='bold')
        
        ax1.set_title('🎨 Visualisation des Clusters\n(Projection sur Plan Factoriel)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax1.grid(True, alpha=0.3)
        
        # Légende des clusters
        unique_labels = np.unique(cluster_labels)
        for i, label in enumerate(unique_labels):
            ax1.scatter([], [], c=scatter.cmap(scatter.norm(label)), s=100, 
                       label=f'Cluster {label}')
        ax1.legend(title='Clusters', title_fontsize=12, fontsize=11)
        
        # 2. Distribution des clusters
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        bars = ax2.bar(unique_labels, counts, alpha=0.8, color=colors, 
                      edgecolor='black', linewidth=1.5)
        
        # Ajouter les pourcentages et valeurs
        total = len(cluster_labels)
        for i, (bar, label, count) in enumerate(zip(bars, unique_labels, counts)):
            percentage = count / total * 100
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total * 0.01, 
                    f'{count}\n({percentage:.1f}%)', 
                    ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax2.set_xlabel('Cluster', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Nombre de Marchés', fontsize=12, fontweight='bold')
        ax2.set_title('📊 Distribution des Clusters\n(Taille des Groupes)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax2.set_xticks(unique_labels)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
    
    def _find_elbow(self, k_range, inertias):
        """Trouve le coude dans la courbe d'inertie"""
        if len(inertias) < 3:
            return k_range[0]
        
        # Méthode de la différence seconde
        diff1 = np.diff(inertias)
        diff2 = np.diff(diff1)
        
        elbow_idx = np.argmax(diff2) + 2
        return k_range[elbow_idx] if elbow_idx < len(k_range) else k_range[-1]
    
    def perform_clustering(self, method='kmeans', n_clusters_range=(2, 8)):
        """Clustering avec visualisations améliorées"""
        print("\n=== ANALYSE DE CLUSTERING ===")
        
        if self.df_encoded is None:
            self.preprocess_data()
        
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(self.df_encoded)
        else:
            X_scaled = self.scaler.transform(self.df_encoded)
        
        if method == 'kmeans':
            # Détermination du nombre optimal de clusters
            inertias = []
            silhouette_scores = []
            k_range = range(n_clusters_range[0], n_clusters_range[1] + 1)
            
            print("🔍 Recherche du nombre optimal de clusters...")
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_scaled)
                
                inertias.append(kmeans.inertia_)
                if k > 1:
                    sil_score = silhouette_score(X_scaled, labels)
                    silhouette_scores.append(sil_score)
                    print(f"   K={k}: Inertie={kmeans.inertia_:.2f}, Silhouette={sil_score:.3f}")
            
            # Visualisation de l'analyse du nombre optimal
            self.plot_clustering_analysis(X_scaled, inertias, silhouette_scores, k_range)
            
            # Clustering final avec le nombre optimal
            optimal_k = self._find_elbow(k_range, inertias)
            print(f"🎯 Nombre optimal de clusters: {optimal_k}")
            
            kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            cluster_labels = kmeans_final.fit_predict(X_scaled)
            
            # Ajout des labels au DataFrame
            self.df['Cluster_KMeans'] = cluster_labels
            
            # Visualisation des résultats
            self.plot_cluster_results(X_scaled, cluster_labels)
            
            # Analyse détaillée des clusters
            self._analyze_clusters_detailed(cluster_labels, method='kmeans')
            
            return cluster_labels
        
        else:
            raise NotImplementedError("Seule la méthode K-means est implémentée dans cette version")
    
    def _analyze_clusters_detailed(self, cluster_labels, method):
        """Analyse détaillée des clusters avec visualisations"""
        print(f"\n=== ANALYSE DÉTAILLÉE DES CLUSTERS ({method.upper()}) ===")
        
        n_clusters = len(np.unique(cluster_labels))
        
        # Analyse par cluster
        cluster_analysis = []
        
        for cluster_id in range(n_clusters):
            mask = cluster_labels == cluster_id
            cluster_data = self.df[mask]
            
            print(f"\n🏷️  CLUSTER {cluster_id}:")
            print(f"   📊 Taille: {mask.sum():,} marchés ({mask.sum()/len(self.df)*100:.1f}%)")
            
            # Analyse par région
            if 'region' in cluster_data.columns:
                region_dist = cluster_data['region'].value_counts().head(3)
                print(f"   🌍 Top 3 régions:")
                for region, count in region_dist.items():
                    print(f"      • {region}: {count} ({count/len(cluster_data)*100:.1f}%)")
            
            # Analyse par catégorie
            if 'Catégorie' in cluster_data.columns:
                cat_dist = cluster_data['Catégorie'].value_counts()
                print(f"   📁 Répartition par catégorie:")
                for cat, count in cat_dist.items():
                    print(f"      • {cat}: {count} ({count/len(cluster_data)*100:.1f}%)")
            
            # Analyse par procédure
            if 'Procédure' in cluster_data.columns:
                proc_dist = cluster_data['Procédure'].value_counts().head(2)
                print(f"   ⚙️  Top 2 procédures:")
                for proc, count in proc_dist.items():
                    print(f"      • {proc}: {count} ({count/len(cluster_data)*100:.1f}%)")
        
        # Graphique de comparaison des clusters
        self._plot_cluster_comparison(cluster_labels)
    
    def _plot_cluster_comparison(self, cluster_labels):
        """Graphique de comparaison entre clusters"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
        
        n_clusters = len(np.unique(cluster_labels))
        colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
        
        # 1. Répartition par région
        if 'region' in self.df.columns:
            region_cluster = pd.crosstab(self.df['region'], cluster_labels, normalize='columns') * 100
            region_cluster.plot(kind='bar', ax=ax1, color=colors, alpha=0.8)
            ax1.set_title('🌍 Répartition Régionale par Cluster (%)', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Région', fontweight='bold')
            ax1.set_ylabel('Pourcentage', fontweight='bold')
            ax1.legend(title='Cluster', title_fontsize=10)
            ax1.tick_params(axis='x', rotation=45)
        
        # 2. Répartition par catégorie
        if 'Catégorie' in self.df.columns:
            cat_cluster = pd.crosstab(self.df['Catégorie'], cluster_labels, normalize='columns') * 100
            cat_cluster.plot(kind='bar', ax=ax2, color=colors, alpha=0.8)
            ax2.set_title('📁 Répartition par Catégorie (%)', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Catégorie', fontweight='bold')
            ax2.set_ylabel('Pourcentage', fontweight='bold')
            ax2.legend(title='Cluster', title_fontsize=10)
            ax2.tick_params(axis='x', rotation=45)
        
        # 3. Évolution temporelle par cluster
        if 'Année' in self.df.columns:
            temp_cluster = pd.crosstab(self.df['Année'], cluster_labels)
            temp_cluster.plot(kind='line', ax=ax3, marker='o', linewidth=2, markersize=6, color=colors)
            ax3.set_title('📈 Évolution Temporelle par Cluster', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Année', fontweight='bold')
            ax3.set_ylabel('Nombre de Marchés', fontweight='bold')
            ax3.legend(title='Cluster', title_fontsize=10)
            ax3.grid(True, alpha=0.3)
        
        # 4. Répartition par procédure
        if 'Procédure' in self.df.columns:
            proc_cluster = pd.crosstab(self.df['Procédure'], cluster_labels, normalize='columns') * 100
            proc_cluster.plot(kind='bar', ax=ax4, color=colors, alpha=0.8)
            ax4.set_title('⚙️ Répartition par Procédure (%)', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Procédure', fontweight='bold')
            ax4.set_ylabel('Pourcentage', fontweight='bold')
            ax4.legend(title='Cluster', title_fontsize=10)
            ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def cross_analysis_improved(self):
        """Analyse croisée avec visualisations améliorées"""
        print("\n=== ANALYSE CROISÉE AMÉLIORÉE ===")
        
        # 1. Heatmap Région × Type de dépense
        if 'region' in self.df.columns and 'Type de dépense personnalisé' in self.df.columns:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            crosstab_region_type = pd.crosstab(
                self.df['region'], 
                self.df['Type de dépense personnalisé']
            )
            
            # Heatmap des valeurs absolues
            sns.heatmap(crosstab_region_type, annot=True, fmt='d', cmap='YlOrRd', 
                       ax=ax1, cbar_kws={'label': 'Nombre de marchés'})
            ax1.set_title('🔥 Heatmap: Région × Type de Dépense\n(Valeurs absolues)', 
                         fontsize=14, fontweight='bold')
            ax1.set_xlabel('Type de Dépense', fontweight='bold')
            ax1.set_ylabel('Région', fontweight='bold')
            
            # Heatmap des pourcentages
            crosstab_pct = pd.crosstab(
                self.df['region'], 
                self.df['Type de dépense personnalisé'], 
                normalize='index'
            ) * 100
            
            sns.heatmap(crosstab_pct, annot=True, fmt='.1f', cmap='Blues', 
                       ax=ax2, cbar_kws={'label': 'Pourcentage (%)'})
            ax2.set_title('📊 Heatmap: Région × Type de Dépense\n(Pourcentages par région)', 
                         fontsize=14, fontweight='bold')
            ax2.set_xlabel('Type de Dépense', fontweight='bold')
            ax2.set_ylabel('Région', fontweight='bold')
            
            plt.tight_layout()
            plt.show()
        
        # 2. Analyse temporelle détaillée
        if 'Année' in self.df.columns and 'region' in self.df.columns:
            self._plot_temporal_analysis()
        
        # 3. Top acheteurs par région
        if 'Acheteur public' in self.df.columns and 'region' in self.df.columns:
            self._plot_top_acheteurs()
    
    def _plot_temporal_analysis(self):
        """Analyse temporelle détaillée"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
        
        # 1. Évolution par région
        temporal_region = self.df.groupby(['Année', 'region']).size().unstack(fill_value=0)
        
        for region in temporal_region.columns:
            ax1.plot(temporal_region.index, temporal_region[region], 
                    marker='o', linewidth=2.5, markersize=6, label=region)
        
        ax1.set_title('📈 Évolution Temporelle par Région', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Année', fontweight='bold')
        ax1.set_ylabel('Nombre de Marchés', fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # 2. Évolution par catégorie
        if 'Catégorie' in self.df.columns:
            temporal_cat = self.df.groupby(['Année', 'Catégorie']).size().unstack(fill_value=0)
            temporal_cat.plot(kind='bar', ax=ax2, alpha=0.8, width=0.8)
            ax2.set_title('📁 Évolution par Catégorie', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Année', fontweight='bold')
            ax2.set_ylabel('Nombre de Marchés', fontweight='bold')
            ax2.legend(title='Catégorie', fontsize=9)
            ax2.tick_params(axis='x', rotation=45)
        
        # 3. Évolution mensuelle (dernière année)
        if 'Mois' in self.df.columns:
            last_year = self.df['Année'].max()
            monthly_data = self.df[self.df['Année'] == last_year].groupby('Mois').size()
            
            ax3.bar(monthly_data.index, monthly_data.values, alpha=0.8, color='skyblue')
            ax3.set_title(f'📅 Répartition Mensuelle ({last_year})', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Mois', fontweight='bold')
            ax3.set_ylabel('Nombre de Marchés', fontweight='bold')
            ax3.set_xticks(range(1, 13))
            ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Croissance annuelle
        annual_growth = temporal_region.sum(axis=1).pct_change() * 100
        ax4.bar(annual_growth.index[1:], annual_growth.values[1:], 
               alpha=0.8, color=['green' if x > 0 else 'red' for x in annual_growth.values[1:]])
        ax4.set_title('📊 Croissance Annuelle (%)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Année', fontweight='bold')
        ax4.set_ylabel('Croissance (%)', fontweight='bold')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_top_acheteurs(self):
        """Analyse des top acheteurs par région"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 1. Top 10 acheteurs globaux
        top_acheteurs = self.df['Acheteur public'].value_counts().head(10)
        
        bars1 = ax1.barh(range(len(top_acheteurs)), top_acheteurs.values, alpha=0.8, color='lightblue')
        ax1.set_yticks(range(len(top_acheteurs)))
        ax1.set_yticklabels([name[:50] + '...' if len(name) > 50 else name 
                            for name in top_acheteurs.index], fontsize=10)
        ax1.set_xlabel('Nombre de Marchés', fontweight='bold')
        ax1.set_title('🏛️ Top 10 Acheteurs Publics', fontsize=14, fontweight='bold')
        
        # Ajouter les valeurs sur les barres
        for i, (bar, value) in enumerate(zip(bars1, top_acheteurs.values)):
            ax1.text(bar.get_width() + max(top_acheteurs.values)*0.01, bar.get_y() + bar.get_height()/2, 
                    f'{value}', ha='left', va='center', fontweight='bold')
        
        # 2. Répartition par région des top acheteurs
        acheteur_region = self.df.groupby(['region', 'Acheteur public']).size().reset_index(name='count')
        top_by_region = acheteur_region.loc[acheteur_region.groupby('region')['count'].idxmax()]
        
        bars2 = ax2.bar(range(len(top_by_region)), top_by_region['count'], alpha=0.8, color='lightcoral')
        ax2.set_xticks(range(len(top_by_region)))
        ax2.set_xticklabels(top_by_region['region'], rotation=45, ha='right')
        ax2.set_ylabel('Nombre de Marchés', fontweight='bold')
        ax2.set_title('🌍 Principal Acheteur par Région', fontsize=14, fontweight='bold')
        
        # Ajouter les noms des acheteurs
        for i, (bar, acheteur, count) in enumerate(zip(bars2, top_by_region['Acheteur public'], top_by_region['count'])):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(top_by_region['count'])*0.01, 
                    f'{count}\n{acheteur[:20]}...', ha='center', va='bottom', fontsize=8, rotation=0)
        
        plt.tight_layout()
        plt.show()
    
    def generate_comprehensive_report(self):
        """Rapport de synthèse complet et détaillé"""
        print("\n" + "="*80)
        print("                    RAPPORT DE SYNTHÈSE COMPLET")
        print("               Analyse Statistique des Marchés Publics Marocains")
        print("                           Version Améliorée 2025")
        print("="*80)
        
        # Statistiques générales
        print(f"\n📊 STATISTIQUES GÉNÉRALES:")
        print(f"   • Nombre total de marchés analysés: {len(self.df):,}")
        print(f"   • Période d'analyse: {self.df['Année'].min():.0f} - {self.df['Année'].max():.0f}")
        print(f"   • Nombre de régions couvertes: {self.df['region'].nunique()}")
        print(f"   • Nombre d'acheteurs publics uniques: {self.df['Acheteur public'].nunique()}")
        
        # Analyse ACP
        if self.pca is not None:
            print(f"\n🔍 RÉSULTATS DE L'ANALYSE EN COMPOSANTES PRINCIPALES:")
            print(f"   • Variance expliquée par la 1ère composante: {self.pca.explained_variance_ratio_[0]:.1%}")
            print(f"   • Variance expliquée par les 2 premières: {sum(self.pca.explained_variance_ratio_[:2]):.1%}")
            print(f"   • Variance expliquée par les 3 premières: {sum(self.pca.explained_variance_ratio_[:3]):.1%}")
            print(f"   • Nombre de composantes pour 80% de variance: {sum(self.pca.explained_variance_ratio_.cumsum() < 0.8) + 1}")
            print(f"   • Dimensionnalité réduite de {self.df_encoded.shape[1]} à {len(self.pca.explained_variance_ratio_)} variables")
        
        # Analyse clustering
        if 'Cluster_KMeans' in self.df.columns:
            cluster_counts = self.df['Cluster_KMeans'].value_counts().sort_index()
            print(f"\n🎯 RÉSULTATS DU CLUSTERING K-MEANS:")
            print(f"   • Nombre optimal de clusters identifiés: {len(cluster_counts)}")
            print(f"   • Répartition des marchés par cluster:")
            for cluster, count in cluster_counts.items():
                print(f"     - Cluster {cluster}: {count:,} marchés ({count/len(self.df)*100:.1f}%)")
            
            # Silhouette score
            if hasattr(self, 'final_silhouette_score'):
                print(f"   • Score de silhouette final: {self.final_silhouette_score:.3f}")
        
        # Répartition par catégorie
        print(f"\n📁 RÉPARTITION PAR CATÉGORIE:")
        cat_counts = self.df['Catégorie'].value_counts()
        for cat, count in cat_counts.items():
            print(f"   • {cat}: {count:,} marchés ({count/len(self.df)*100:.1f}%)")
        
        # Répartition par région
        print(f"\n🌍 RÉPARTITION GÉOGRAPHIQUE:")
        region_counts = self.df['region'].value_counts().head(5)
        print("   Top 5 des régions les plus actives:")
        for region, count in region_counts.items():
            print(f"   • {region}: {count:,} marchés ({count/len(self.df)*100:.1f}%)")
        
        # Top acheteurs
        print(f"\n🏛️ PRINCIPAUX ACHETEURS PUBLICS:")
        top_acheteurs = self.df['Acheteur public'].value_counts().head(5)
        for i, (acheteur, count) in enumerate(top_acheteurs.items(), 1):
            acheteur_display = acheteur[:60] + "..." if len(acheteur) > 60 else acheteur
            print(f"   {i}. {acheteur_display}")
            print(f"      → {count:,} marchés ({count/len(self.df)*100:.1f}%)")
        
        # Analyse temporelle
        if 'Année' in self.df.columns:
            annual_dist = self.df['Année'].value_counts().sort_index()
            print(f"\n📈 ÉVOLUTION TEMPORELLE:")
            print("   Répartition par année:")
            for year, count in annual_dist.items():
                growth = ""
                if year > annual_dist.index.min():
                    prev_count = annual_dist.get(year-1, 0)
                    if prev_count > 0:
                        growth_rate = ((count - prev_count) / prev_count) * 100
                        growth = f" ({growth_rate:+.1f}%)"
                print(f"   • {year:.0f}: {count:,} marchés{growth}")
        
        # Insights et recommandations
        print(f"\n💡 INSIGHTS PRINCIPAUX:")
        
        # Concentration géographique
        top_3_regions_pct = (self.df['region'].value_counts().head(3).sum() / len(self.df)) * 100
        print(f"   • Concentration géographique: Les 3 premières régions représentent {top_3_regions_pct:.1f}% des marchés")
        
        # Concentration des acheteurs
        top_10_acheteurs_pct = (self.df['Acheteur public'].value_counts().head(10).sum() / len(self.df)) * 100
        print(f"   • Concentration des acheteurs: Les 10 premiers acheteurs représentent {top_10_acheteurs_pct:.1f}% des marchés")
        
        # Diversité des procédures
        if 'Procédure' in self.df.columns:
            proc_diversity = self.df['Procédure'].nunique()
            print(f"   • Diversité des procédures: {proc_diversity} types de procédures utilisées")
        
        print(f"\n🎯 RECOMMANDATIONS:")
        print("   • Analyser les facteurs de concentration géographique")
        print("   • Étudier les patterns de clustering pour optimiser les processus")
        print("   • Surveiller l'évolution temporelle pour anticiper les tendances")
        print("   • Approfondir l'analyse des acheteurs les plus actifs")
        
        print("\n" + "="*80)
        print("                          FIN DU RAPPORT")
        print("="*80)

# Fonction principale améliorée
def main_improved():
    """Fonction principale pour exécuter l'analyse complète améliorée"""
    
    print("🚀 ANALYSE STATISTIQUE AVANCÉE - VERSION AMÉLIORÉE")
    print("📋 Marchés Publics Marocains - Graphiques Optimisés")
    print("-" * 70)
    
    try:
        # Chargement des données (remplacez par votre fichier)
        df = pd.read_excel('C:/Users/lenovo/Downloads/stage_Application MP/base_de_donnee_finale_encodee.xlsx')
        
        # Création de l'analyseur amélioré
        analyser = AnalyseMarchesPublicsAmeliore(df=df)
        
        print("🔧 Phase 1: Préparation des données...")
        analyser.preprocess_data()
        
        print("📊 Phase 2: Analyse en Composantes Principales...")
        X_pca = analyser.perform_pca(n_components=10)
        
        print("🎯 Phase 3: Clustering K-means...")
        cluster_labels = analyser.perform_clustering(method='kmeans', n_clusters_range=(2, 8))
        
        print("🔀 Phase 4: Analyses croisées...")
        analyser.cross_analysis_improved()
        
        print("📝 Phase 5: Génération du rapport final...")
        analyser.generate_comprehensive_report()
        
        print("\n✅ ANALYSE TERMINÉE AVEC SUCCÈS!")
        print("📊 Tous les graphiques ont été affichés sans chevauchement.")
        print("📋 Consultez le rapport détaillé ci-dessus.")
        
        return analyser
        
    except Exception as e:
        print(f"❌ Erreur lors de l'analyse: {e}")
        print("💡 Vérifiez le chemin du fichier et les colonnes requises.")
        return None

# Instructions d'utilisation améliorées
def print_improved_instructions():
    """Instructions détaillées pour la version améliorée"""
    
    instructions = """
    📋 GUIDE D'UTILISATION - VERSION AMÉLIORÉE
    ==========================================
    
    🎯 NOUVEAUTÉS DE CETTE VERSION:
    • Graphiques séparés sans chevauchement
    • Visualisations améliorées avec légendes claires
    • Analyses plus détaillées avec interprétations
    • Rapport de synthèse complet
    • Code optimisé et commenté
    
    🔧 UTILISATION RAPIDE:
    1. Remplacez le chemin du fichier dans main_improved()
    2. Exécutez: analyser = main_improved()
    3. Les graphiques s'affichent automatiquement sans chevauchement
    
    📊 GRAPHIQUES GÉNÉRÉS:
    • ACP: Éboulis + Variance cumulée (séparés)
    • ACP: Projections + Cercle corrélations (séparés)
    • Clustering: Coude + Silhouette (séparés)
    • Clustering: Résultats + Distribution (séparés)
    • Analyses croisées: Heatmaps et comparaisons
    
    💡 CONSEILS:
    • Ajustez les paramètres selon vos besoins
    • Sauvegardez les graphiques avec plt.savefig()
    • Exportez le rapport dans un fichier texte
    """
    
    print(instructions)

# Test avec données d'exemple
def test_improved():
    """Test rapide de la version améliorée"""
    
    # Données d'exemple plus réalistes
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = {
        'Référence': pd.date_range('2021-01-01', periods=n_samples, freq='D'),
        'Procédure': np.random.choice(['AOO', 'AOS', 'APPEL'], n_samples, p=[0.6, 0.3, 0.1]),
        'Catégorie': np.random.choice(['Services', 'Travaux', 'Fournitures'], n_samples, p=[0.5, 0.3, 0.2]),
        'Type de dépense personnalisé': np.random.choice(['Autre', 'Transport', 'Informatique', 'Maintenance'], n_samples),
        'region': np.random.choice(['Casablanca-Settat', 'Souss-Massa', 'Oriental', 'Rabat-Salé'], n_samples),
        'Acheteur public': np.random.choice([f'Ministère {i}' for i in range(1, 21)], n_samples),
        'Procedure_AOO': np.random.choice([0, 1], n_samples),
        'Procedure_AOS': np.random.choice([0, 1], n_samples),
        'Categorie_Services': np.random.choice([0, 1], n_samples),
        'Categorie_Travaux': np.random.choice([0, 1], n_samples),
        'Categorie_Fournitures': np.random.choice([0, 1], n_samples)
    }
    
    df_test = pd.DataFrame(sample_data)
    
    print("🧪 TEST DE LA VERSION AMÉLIORÉE")
    print("-" * 50)
    
    try:
        analyser = AnalyseMarchesPublicsAmeliore(df=df_test)
        
        analyser.preprocess_data()
        analyser.perform_pca(n_components=8)
        analyser.perform_clustering(method='kmeans', n_clusters_range=(2, 6))
        analyser.cross_analysis_improved()
        analyser.generate_comprehensive_report()
        
        print("✅ Test réussi! Graphiques affichés sans chevauchement.")
        return analyser
        
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        return None
analyser = main_improved()
if __name__ == "__main__":
    print_improved_instructions()
    
    # Décommentez pour exécuter:
    # analyser = main_improved()  # Analyse avec vos vraies données
    # analyser = test_improved()  # Test rapide avec données simulées
