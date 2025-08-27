import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configuration de l'affichage
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# ====================================================================
# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES
# ====================================================================

def load_and_prepare_data(file_path):
    """
    Charge et prépare les données des marchés publics
    """
    # Détection du type de fichier et chargement approprié
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        # Pour les fichiers Excel
        df = pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        # Pour les fichiers CSV, test de différents séparateurs
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except:
            try:
                df = pd.read_csv(file_path, encoding='latin-1')
            except:
                df = pd.read_csv(file_path, encoding='cp1252')
    else:
        # Pour les fichiers texte (.txt), test de différents séparateurs
        try:
            df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
        except:
            try:
                df = pd.read_csv(file_path, sep='\t', encoding='latin-1')
            except:
                try:
                    df = pd.read_csv(file_path, sep=';', encoding='utf-8')
                except:
                    df = pd.read_csv(file_path, sep=';', encoding='latin-1')
    
    print(f"✅ Données chargées avec succès: {len(df)} lignes, {len(df.columns)} colonnes")
    
    # Affichage des colonnes disponibles
    print(f"📋 Colonnes disponibles: {list(df.columns)}")
    
    # Nettoyage des noms de colonnes
    df.columns = df.columns.str.strip()
    
    # Conversion de la colonne de date (adaptation selon le nom réel de la colonne)
    date_column = None
    for col in df.columns:
        if 'publié' in col.lower() or 'date' in col.lower():
            date_column = col
            break
    
    if date_column:
        df['Date_Publication'] = pd.to_datetime(df[date_column], errors='coerce')
        df['Annee'] = df['Date_Publication'].dt.year
        df['Mois'] = df['Date_Publication'].dt.month
        df['Trimestre'] = df['Date_Publication'].dt.quarter
        print(f"✅ Date de publication identifiée: {date_column}")
    else:
        print("⚠️ Colonne de date non trouvée, création d'une date par défaut")
        df['Date_Publication'] = pd.to_datetime('2025-01-01')
        df['Annee'] = 2025
        df['Mois'] = 1
        df['Trimestre'] = 1
    
    # Nettoyage des données textuelles (adaptation selon les colonnes disponibles)
    text_columns = []
    for possible_col in ['Objet', 'Acheteur public', 'Lieu d\'exécution', 'region', 'Catégorie', 'Procédure']:
        for actual_col in df.columns:
            if possible_col.lower() in actual_col.lower():
                text_columns.append(actual_col)
                break
    
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    
    # Affichage d'un aperçu des données
    print(f"\n📊 Aperçu des premières lignes:")
    print(df.head(3))
    
    return df

# ====================================================================
# 2. STATISTIQUES DESCRIPTIVES GLOBALES
# ====================================================================

def analyse_descriptive_globale(df):
    """
    Génère les statistiques descriptives globales
    """
    print("="*60)
    print("📊 ANALYSE DESCRIPTIVE GLOBALE")
    print("="*60)
    
    # Informations générales
    print(f"📈 Nombre total de marchés: {len(df):,}")
    
    # Recherche automatique des colonnes importantes
    date_col = None
    acheteur_col = None
    region_col = None
    lieu_col = None
    procedure_col = None
    category_col = None
    
    for col in df.columns:
        if 'publié' in col.lower() or 'date' in col.lower():
            date_col = col
        elif 'acheteur' in col.lower():
            acheteur_col = col
        elif 'region' in col.lower():
            region_col = col
        elif 'lieu' in col.lower() or 'exécution' in col.lower():
            lieu_col = col
        elif 'procédure' in col.lower():
            procedure_col = col
        elif 'catégorie' in col.lower():
            category_col = col
    
    if date_col and 'Date_Publication' in df.columns:
        print(f"📅 Période couverte: {df['Date_Publication'].min()} à {df['Date_Publication'].max()}")
    
    if acheteur_col:
        print(f"🏛️ Nombre d'acheteurs publics: {df[acheteur_col].nunique()}")
    
    if region_col:
        print(f"🌍 Nombre de régions: {df[region_col].nunique()}")
    
    if lieu_col:
        print(f"🏙️ Nombre de lieux d'exécution: {df[lieu_col].nunique()}")
    
    # Répartition par procédure
    procedure_counts = None
    if procedure_col:
        print(f"\n📋 RÉPARTITION PAR TYPE DE PROCÉDURE:")
        procedure_counts = df[procedure_col].value_counts()
        for proc, count in procedure_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   {proc}: {count:,} ({percentage:.1f}%)")
    
    # Répartition par catégorie
    category_counts = None
    if category_col:
        print(f"\n🏗️ RÉPARTITION PAR CATÉGORIE:")
        category_counts = df[category_col].value_counts()
        for cat, count in category_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   {cat}: {count:,} ({percentage:.1f}%)")
    
    # CORRECTION: Vérification correcte des objets pandas Series
    # Changement de "if not procedure_counts and not category_counts:" en:
    if (procedure_counts is None or procedure_counts.empty) and (category_counts is None or category_counts.empty):
        print(f"\n⚠️ Colonnes 'Procédure' ou 'Catégorie' non trouvées.")
        print(f"📋 Colonnes disponibles: {list(df.columns)}")
        print(f"📊 Aperçu des 5 premières colonnes:")
        for col in df.columns[:5]:
            print(f"   {col}: {df[col].nunique()} valeurs uniques")
            print(f"      Exemples: {list(df[col].dropna().unique()[:3])}")
    
    return procedure_counts, category_counts

# ====================================================================
# 3. ANALYSE TEMPORELLE
# ====================================================================

def analyse_temporelle(df):
    """
    Analyse l'évolution temporelle des marchés publics
    """
    print("\n" + "="*60)
    print("📅 ANALYSE TEMPORELLE")
    print("="*60)
    
    # Création des graphiques temporels
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Évolution Temporelle des Marchés Publics Marocains', fontsize=16, fontweight='bold')
    
    # 1. Évolution annuelle
    annual_counts = df.groupby('Annee').size()
    axes[0,0].plot(annual_counts.index, annual_counts.values, marker='o', linewidth=2, markersize=8)
    axes[0,0].set_title('Évolution Annuelle du Nombre de Marchés')
    axes[0,0].set_xlabel('Année')
    axes[0,0].set_ylabel('Nombre de Marchés')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Répartition mensuelle
    monthly_counts = df.groupby('Mois').size()
    mois_labels = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun', 
                   'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']
    axes[0,1].bar(range(1, 13), [monthly_counts.get(i, 0) for i in range(1, 13)])
    axes[0,1].set_title('Répartition Mensuelle des Publications')
    axes[0,1].set_xlabel('Mois')
    axes[0,1].set_ylabel('Nombre de Marchés')
    axes[0,1].set_xticks(range(1, 13))
    axes[0,1].set_xticklabels(mois_labels, rotation=45)
    
    # 3. Évolution par catégorie
    category_evolution = df.groupby(['Annee', 'Catégorie']).size().unstack(fill_value=0)
    category_evolution.plot(kind='area', stacked=True, ax=axes[1,0], alpha=0.7)
    axes[1,0].set_title('Évolution par Catégorie de Marché')
    axes[1,0].set_xlabel('Année')
    axes[1,0].set_ylabel('Nombre de Marchés')
    axes[1,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 4. Heatmap mensuelle par année
    monthly_yearly = df.groupby(['Annee', 'Mois']).size().unstack(fill_value=0)
    sns.heatmap(monthly_yearly, annot=True, fmt='d', cmap='YlOrRd', ax=axes[1,1])
    axes[1,1].set_title('Heatmap: Marchés par Mois et Année')
    axes[1,1].set_xlabel('Mois')
    axes[1,1].set_ylabel('Année')
    
    plt.tight_layout()
    plt.show()
    
    return annual_counts, monthly_counts

# ====================================================================
# 4. ANALYSE GÉOGRAPHIQUE
# ====================================================================

def analyse_geographique(df):
    """
    Analyse la répartition géographique des marchés
    """
    print("\n" + "="*60)
    print("🌍 ANALYSE GÉOGRAPHIQUE")
    print("="*60)
    
    # Top 10 des régions
    top_regions = df['region'].value_counts().head(10)
    print("🏆 TOP 10 DES RÉGIONS:")
    for i, (region, count) in enumerate(top_regions.items(), 1):
        percentage = (count / len(df)) * 100
        print(f"   {i:2}. {region}: {count:,} marchés ({percentage:.1f}%)")
    
    # Visualisations géographiques
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Analyse Géographique des Marchés Publics', fontsize=16, fontweight='bold')
    
    # 1. Top 10 régions
    top_regions.plot(kind='barh', ax=axes[0,0])
    axes[0,0].set_title('Top 10 des Régions par Nombre de Marchés')
    axes[0,0].set_xlabel('Nombre de Marchés')
    
    # 2. Répartition par catégorie et région (top 5 régions)
    top5_regions = top_regions.head(5).index
    region_category = df[df['region'].isin(top5_regions)].groupby(['region', 'Catégorie']).size().unstack(fill_value=0)
    region_category.plot(kind='bar', stacked=True, ax=axes[0,1])
    axes[0,1].set_title('Répartition par Catégorie (Top 5 Régions)')
    axes[0,1].set_xlabel('Région')
    axes[0,1].set_ylabel('Nombre de Marchés')
    axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. Top 15 lieux d'exécution
    top_lieux = df['Lieu d\'exécution'].value_counts().head(15)
    top_lieux.plot(kind='barh', ax=axes[1,0])
    axes[1,0].set_title('Top 15 des Lieux d\'Exécution')
    axes[1,0].set_xlabel('Nombre de Marchés')
    
    # 4. Camembert des régions (top 8 + autres)
    top8_regions = top_regions.head(8)
    autres = top_regions.iloc[8:].sum()
    pie_data = list(top8_regions) + [autres]
    pie_labels = list(top8_regions.index) + ['Autres']
    
    axes[1,1].pie(pie_data, labels=pie_labels, autopct='%1.1f%%', startangle=90)
    axes[1,1].set_title('Répartition Régionale des Marchés Publics')
    
    plt.tight_layout()
    plt.show()
    
    return top_regions, top_lieux

# ====================================================================
# 5. ANALYSE DES ACHETEURS PUBLICS
# ====================================================================

def analyse_acheteurs(df):
    """
    Analyse des acheteurs publics les plus actifs
    """
    print("\n" + "="*60)
    print("🏛️ ANALYSE DES ACHETEURS PUBLICS")
    print("="*60)
    
    # Top 15 des acheteurs
    top_acheteurs = df['Acheteur public'].value_counts().head(15)
    print("🏆 TOP 15 DES ACHETEURS PUBLICS:")
    for i, (acheteur, count) in enumerate(top_acheteurs.items(), 1):
        percentage = (count / len(df)) * 100
        print(f"   {i:2}. {acheteur[:50]}{'...' if len(acheteur) > 50 else ''}: {count:,} ({percentage:.1f}%)")
    
    # Visualisation
    plt.figure(figsize=(14, 8))
    top_acheteurs.plot(kind='barh')
    plt.title('Top 15 des Acheteurs Publics les Plus Actifs', fontsize=14, fontweight='bold')
    plt.xlabel('Nombre de Marchés Publics')
    plt.ylabel('Acheteurs Publics')
    plt.tight_layout()
    plt.show()
    
    return top_acheteurs

# ====================================================================
# 6. ANALYSE DES TYPES DE DÉPENSES
# ====================================================================

def analyse_types_depenses(df):
    """
    Analyse des types de dépenses personnalisés
    """
    print("\n" + "="*60)
    print("💰 ANALYSE DES TYPES DE DÉPENSES")
    print("="*60)
    
    # Répartition des types de dépenses
    type_depenses = df['Type de dépense personnalisé'].value_counts()
    print("📊 RÉPARTITION PAR TYPE DE DÉPENSE:")
    for type_dep, count in type_depenses.items():
        percentage = (count / len(df)) * 100
        print(f"   {type_dep}: {count:,} ({percentage:.1f}%)")
    
    # Analyse croisée catégorie vs type de dépense
    crosstab = pd.crosstab(df['Catégorie'], df['Type de dépense personnalisé'], margins=True)
    print("\n📋 TABLEAU CROISÉ CATÉGORIE vs TYPE DE DÉPENSE:")
    print(crosstab)
    
    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Graphique en barres des types de dépenses
    type_depenses.plot(kind='bar', ax=axes[0])
    axes[0].set_title('Répartition par Type de Dépense')
    axes[0].set_xlabel('Type de Dépense')
    axes[0].set_ylabel('Nombre de Marchés')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Heatmap du tableau croisé (sans les marges)
    sns.heatmap(crosstab.iloc[:-1, :-1], annot=True, fmt='d', cmap='Blues', ax=axes[1])
    axes[1].set_title('Heatmap: Catégorie vs Type de Dépense')
    
    plt.tight_layout()
    plt.show()
    
    return type_depenses, crosstab

# ====================================================================
# 7. ANALYSE TEXTUELLE BASIQUE DES OBJETS
# ====================================================================

def analyse_textuelle_objets(df, top_n=20):
    """
    Analyse textuelle basique des objets de marchés
    """
    print("\n" + "="*60)
    print("📝 ANALYSE TEXTUELLE DES OBJETS")
    print("="*60)
    
    # Mots les plus fréquents dans les objets
    all_text = ' '.join(df['Objet'].astype(str)).upper()
    
    # Nettoyage basique
    import re
    # Suppression des caractères spéciaux et mots vides courants
    mots_vides = ['DE', 'LA', 'LE', 'DU', 'DES', 'ET', 'À', 'AU', 'AUX', 'POUR', 'DANS', 'SUR', 'AVEC', 'PAR', 'EN']
    words = re.findall(r'\b[A-ZÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞŸ]{3,}\b', all_text)
    words = [word for word in words if word not in mots_vides and len(word) > 2]
    
    # Comptage des mots
    from collections import Counter
    word_counts = Counter(words)
    top_words = dict(word_counts.most_common(top_n))
    
    print(f"🔤 TOP {top_n} DES MOTS LES PLUS FRÉQUENTS:")
    for i, (word, count) in enumerate(top_words.items(), 1):
        print(f"   {i:2}. {word}: {count:,} occurrences")
    
    # Visualisation
    plt.figure(figsize=(12, 8))
    words_df = pd.Series(top_words)
    words_df.plot(kind='barh')
    plt.title(f'Top {top_n} des Mots les Plus Fréquents dans les Objets de Marchés')
    plt.xlabel('Fréquence')
    plt.ylabel('Mots')
    plt.tight_layout()
    plt.show()
    
    return top_words

# ====================================================================
# 8. DASHBOARD INTERACTIF AVEC PLOTLY
# ====================================================================

def create_interactive_dashboard(df):
    """
    Crée un dashboard interactif avec Plotly
    """
    print("\n" + "="*60)
    print("📊 CRÉATION DU DASHBOARD INTERACTIF")
    print("="*60)
    
    # 1. Graphique temporel interactif
    fig_temporal = px.line(
        df.groupby('Date_Publication').size().reset_index(),
        x='Date_Publication', 
        y=0,
        title='Évolution Temporelle des Marchés Publics'
    )
    fig_temporal.update_layout(yaxis_title='Nombre de Marchés')
    fig_temporal.show()
    
    # 2. Répartition géographique interactive
    region_counts = df['region'].value_counts().head(10)
    fig_geo = px.bar(
        x=region_counts.values,
        y=region_counts.index[::-1],
        orientation='h',
        title='Top 10 des Régions par Nombre de Marchés',
        labels={'x': 'Nombre de Marchés', 'y': 'Région'}
    )
    fig_geo.show()
    
    # 3. Treemap des catégories et types
    category_type = df.groupby(['Catégorie', 'Type de dépense personnalisé']).size().reset_index(name='count')
    fig_treemap = px.treemap(
        category_type,
        path=['Catégorie', 'Type de dépense personnalisé'],
        values='count',
        title='Répartition Hiérarchique: Catégories et Types de Dépenses'
    )
    fig_treemap.show()
    
    return fig_temporal, fig_geo, fig_treemap

# ====================================================================
# 9. FONCTION PRINCIPALE
# ====================================================================

def main_analysis(file_path):
    """
    Fonction principale qui exécute toute l'analyse exploratoire
    """
    print("🚀 DÉMARRAGE DE L'ANALYSE EXPLORATOIRE")
    print("=" * 80)
    
    # Chargement des données
    df = load_and_prepare_data(file_path)
    
    # Exécution de toutes les analyses
    procedure_counts, category_counts = analyse_descriptive_globale(df)
    annual_counts, monthly_counts = analyse_temporelle(df)
    top_regions, top_lieux = analyse_geographique(df)
    top_acheteurs = analyse_acheteurs(df)
    type_depenses, crosstab = analyse_types_depenses(df)
    top_words = analyse_textuelle_objets(df)
    
    # Dashboard interactif
    create_interactive_dashboard(df)
    
    print("\n" + "="*80)
    print("✅ ANALYSE EXPLORATOIRE TERMINÉE AVEC SUCCÈS!")
    print("="*80)
    
    # Retour des résultats principaux
    return {
        'dataframe': df,
        'procedure_counts': procedure_counts,
        'category_counts': category_counts,
        'annual_counts': annual_counts,
        'top_regions': top_regions,
        'top_acheteurs': top_acheteurs,
        'top_words': top_words
    }

# ====================================================================
# UTILISATION
# ====================================================================

if __name__ == "__main__":
    # Remplacez par le chemin vers votre fichier de données
    file_path = "C:/Users/lenovo/Downloads/stage_Application MP/base_de_donnee_finale_encodee.xlsx"  # ou .txt selon votre format
    
    # Exécution de l'analyse complète
    results = main_analysis(file_path)
    
    # Vous pouvez accéder aux résultats individuels:
    # print(results['top_regions'])
    # print(results['category_counts'])
    # etc.
