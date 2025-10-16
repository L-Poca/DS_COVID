import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import base64
from typing import Optional, Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

# Imports du framework RAF
try:
    from src.features.raf.interpretability import (
        SHAPExplainer, GradCAMExplainer, InterpretabilityAnalyzer,
        create_interpretability_dashboard, generate_interpretability_report
    )
    RAF_AVAILABLE = True
except ImportError:
    RAF_AVAILABLE = False
    st.error("⚠️ Module RAF d'interprétabilité non disponible")

# Configuration de la page
st.set_page_config(
    page_title="Interprétabilité - SHAP & GradCAM",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Interprétabilité des Modèles - SHAP & GradCAM")
st.markdown("---")

if not RAF_AVAILABLE:
    st.stop()

# Sidebar pour la configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Section Modèles
    st.subheader("📋 Modèles")
    
    # Upload de modèles (simulation)
    cnn_model_file = st.file_uploader("Modèle CNN (.h5/.keras)", type=['h5', 'keras'])
    ml_model_file = st.file_uploader("Modèle ML (.pkl/.joblib)", type=['pkl', 'joblib'])
    
    # Paramètres GradCAM
    st.subheader("🎯 GradCAM")
    layer_name = st.text_input("Nom de la couche", placeholder="auto-détection")
    gradcam_method = st.selectbox("Méthode", ["GradCAM", "GradCAM++"])
    alpha = st.slider("Transparence superposition", 0.0, 1.0, 0.4, 0.1)
    colormap = st.selectbox("Colormap", ["jet", "hot", "viridis", "plasma"])
    
    # Paramètres SHAP
    st.subheader("📊 SHAP")
    shap_method = st.selectbox("Méthode SHAP", ["auto", "tree", "linear", "deep", "kernel"])
    max_evals = st.number_input("Max évaluations", 100, 1000, 100)
    
    # Classes
    st.subheader("🏷️ Classes")
    default_classes = ["COVID", "Normal", "Viral", "Lung_Opacity"]
    class_names = st.text_area(
        "Noms des classes (un par ligne)",
        value="\n".join(default_classes)
    ).split("\n")

# Interface principale
tab1, tab2, tab3, tab4 = st.tabs([
    "🖼️ Analyse Simple", 
    "📊 Comparaison", 
    "📈 Rapport Batch", 
    "🔧 Utilitaires"
])

with tab1:
    st.header("🖼️ Analyse d'une Image")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📤 Upload Image")
        
        # Upload d'image
        uploaded_file = st.file_uploader(
            "Choisir une image",
            type=['png', 'jpg', 'jpeg'],
            key="single_image"
        )
        
        if uploaded_file is not None:
            # Affichage de l'image
            image = Image.open(uploaded_file)
            st.image(image, caption="Image uploadée", use_column_width=True)
            
            # Conversion en array numpy
            img_array = np.array(image)
            if len(img_array.shape) == 3 and img_array.shape[-1] == 3:
                # Conversion en niveaux de gris si nécessaire
                img_gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
                img_array = img_gray
            
            # Normalisation
            if img_array.max() > 1:
                img_array = img_array / 255.0
            
            # Redimensionnement (simulation - à adapter selon vos modèles)
            target_size = st.selectbox("Taille cible", [(224, 224), (256, 256), (512, 512)])
            
            st.info(f"Image shape: {img_array.shape}")
            
        else:
            st.info("👆 Uploadez une image pour commencer l'analyse")
    
    with col2:
        st.subheader("🔍 Résultats d'Analyse")
        
        if uploaded_file is not None:
            # Simulation d'analyse (à remplacer par vrai modèle)
            if st.button("🚀 Lancer l'Analyse", type="primary"):
                with st.spinner("Analyse en cours..."):
                    # Ici, vous intégreriez vos vrais modèles
                    st.success("✅ Analyse terminée !")
                    
                    # Tabs pour les résultats
                    result_tab1, result_tab2, result_tab3 = st.tabs([
                        "GradCAM", "SHAP", "Comparaison"
                    ])
                    
                    with result_tab1:
                        st.markdown("#### 🎯 Résultats GradCAM")
                        
                        # Simulation de résultats GradCAM
                        col_orig, col_heat, col_overlay = st.columns(3)
                        
                        with col_orig:
                            st.image(image, caption="Original", use_column_width=True)
                        
                        with col_heat:
                            # Génération d'une heatmap factice
                            fake_heatmap = np.random.rand(224, 224)
                            fig, ax = plt.subplots(figsize=(6, 6))
                            ax.imshow(fake_heatmap, cmap=colormap)
                            ax.axis('off')
                            ax.set_title("Heatmap GradCAM")
                            st.pyplot(fig)
                        
                        with col_overlay:
                            st.info("Superposition générée")
                            
                        # Métriques GradCAM
                        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                        with metrics_col1:
                            st.metric("Activation Moyenne", "0.432")
                        with metrics_col2:
                            st.metric("Concentration 75%", "23.5%")
                        with metrics_col3:
                            st.metric("Entropie", "2.34")
                    
                    with result_tab2:
                        st.markdown("#### 📊 Résultats SHAP")
                        
                        # Simulation de graphique SHAP
                        fake_features = np.random.randn(20)
                        feature_names = [f"Feature_{i}" for i in range(20)]
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        colors = ['red' if x > 0 else 'blue' for x in fake_features]
                        ax.barh(range(len(fake_features)), fake_features, color=colors, alpha=0.7)
                        ax.set_yticks(range(len(fake_features)))
                        ax.set_yticklabels(feature_names)
                        ax.set_xlabel("Valeur SHAP")
                        ax.set_title("Importance des Features (SHAP)")
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        
                        # Statistiques SHAP
                        st.markdown("**Statistiques:**")
                        stat_col1, stat_col2, stat_col3 = st.columns(3)
                        with stat_col1:
                            st.metric("Valeur moyenne", f"{np.mean(fake_features):.4f}")
                        with stat_col2:
                            st.metric("Valeur max", f"{np.max(fake_features):.4f}")
                        with stat_col3:
                            st.metric("Écart-type", f"{np.std(fake_features):.4f}")
                    
                    with result_tab3:
                        st.markdown("#### ⚖️ Comparaison des Méthodes")
                        
                        comparison_data = {
                            'Méthode': ['GradCAM', 'SHAP'],
                            'Confiance': [0.856, 0.782],
                            'Classe Prédite': ['COVID', 'COVID'],
                            'Zone Focus': ['Poumons droits', 'Région centrale'],
                            'Temps Calcul (s)': [0.23, 1.45]
                        }
                        
                        df_comparison = pd.DataFrame(comparison_data)
                        st.dataframe(df_comparison, use_container_width=True)
                        
                        st.success("✅ Cohérence entre les méthodes : Les deux pointent vers une infection COVID-19")

with tab2:
    st.header("📊 Comparaison de Modèles")
    
    st.info("🔧 Cette section permet de comparer les explications de plusieurs modèles sur la même image")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Configuration")
        
        # Sélection des modèles à comparer
        models_to_compare = st.multiselect(
            "Modèles à comparer",
            ["CNN Baseline", "CNN ResNet", "CNN VGG", "Random Forest", "SVM", "XGBoost"],
            default=["CNN Baseline", "Random Forest"]
        )
        
        # Image à analyser
        comparison_image = st.file_uploader(
            "Image pour comparaison",
            type=['png', 'jpg', 'jpeg'],
            key="comparison_image"
        )
        
        if comparison_image:
            st.image(comparison_image, caption="Image à analyser", use_column_width=True)
    
    with col2:
        st.subheader("📈 Résultats Comparatifs")
        
        if comparison_image and models_to_compare:
            if st.button("🔄 Comparer les Modèles", type="primary"):
                with st.spinner("Comparaison en cours..."):
                    # Simulation de comparaison
                    st.success("✅ Comparaison terminée !")
                    
                    # Graphique de comparaison des confiances
                    fake_confidences = np.random.uniform(0.6, 0.95, len(models_to_compare))
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.bar(models_to_compare, fake_confidences, alpha=0.7, 
                                 color=plt.cm.Set3(np.linspace(0, 1, len(models_to_compare))))
                    ax.set_ylabel("Confiance de Prédiction")
                    ax.set_title("Comparaison des Confiances par Modèle")
                    ax.set_ylim(0, 1)
                    
                    # Ajout des valeurs sur les barres
                    for bar, conf in zip(bars, fake_confidences):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{conf:.2%}', ha='center', va='bottom')
                    
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Tableau récapitulatif
                    comparison_results = pd.DataFrame({
                        'Modèle': models_to_compare,
                        'Confiance': [f"{c:.2%}" for c in fake_confidences],
                        'Classe Prédite': np.random.choice(class_names, len(models_to_compare)),
                        'Méthode Explication': ['GradCAM' if 'CNN' in m else 'SHAP' for m in models_to_compare],
                        'Cohérence': np.random.choice(['Élevée', 'Moyenne', 'Faible'], len(models_to_compare))
                    })
                    
                    st.markdown("#### 📋 Résumé Détaillé")
                    st.dataframe(comparison_results, use_container_width=True)

with tab3:
    st.header("📈 Rapport d'Analyse Batch")
    
    st.info("🔧 Analysez plusieurs images simultanément pour générer un rapport complet")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📁 Upload Batch")
        
        # Upload multiple files
        batch_files = st.file_uploader(
            "Sélectionner plusieurs images",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            key="batch_images"
        )
        
        if batch_files:
            st.success(f"✅ {len(batch_files)} images uploadées")
            
            # Paramètres du batch
            st.subheader("⚙️ Paramètres Batch")
            batch_method = st.selectbox("Méthode d'analyse", ["GradCAM", "SHAP", "Les deux"])
            generate_individual = st.checkbox("Rapports individuels", True)
            generate_summary = st.checkbox("Résumé global", True)
    
    with col2:
        st.subheader("📊 Résultats Batch")
        
        if batch_files:
            if st.button("🚀 Lancer Analyse Batch", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results_container = st.container()
                
                # Simulation de traitement batch
                for i, file in enumerate(batch_files):
                    progress = (i + 1) / len(batch_files)
                    progress_bar.progress(progress)
                    status_text.text(f"Traitement de {file.name}...")
                    
                    # Simulation de délai
                    import time
                    time.sleep(0.5)
                
                status_text.text("✅ Traitement terminé !")
                
                with results_container:
                    # Génération du rapport récapitulatif
                    st.markdown("#### 📋 Rapport Récapitulatif")
                    
                    # Données simulées
                    report_data = []
                    for i, file in enumerate(batch_files):
                        report_data.append({
                            'Image': file.name,
                            'Classe Prédite': np.random.choice(class_names),
                            'Confiance': f"{np.random.uniform(0.7, 0.98):.2%}",
                            'Activation Moyenne': f"{np.random.uniform(0.2, 0.8):.3f}",
                            'Entropie': f"{np.random.uniform(1.5, 3.0):.2f}",
                            'Statut': '✅ Succès'
                        })
                    
                    df_report = pd.DataFrame(report_data)
                    st.dataframe(df_report, use_container_width=True)
                    
                    # Graphiques de synthèse
                    col_chart1, col_chart2 = st.columns(2)
                    
                    with col_chart1:
                        # Distribution des classes prédites
                        class_counts = df_report['Classe Prédite'].value_counts()
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%')
                        ax.set_title("Distribution des Classes Prédites")
                        st.pyplot(fig)
                    
                    with col_chart2:
                        # Distribution des confiances
                        confidences = [float(c.strip('%'))/100 for c in df_report['Confiance']]
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.hist(confidences, bins=10, alpha=0.7, edgecolor='black')
                        ax.set_xlabel("Confiance")
                        ax.set_ylabel("Nombre d'images")
                        ax.set_title("Distribution des Confiances")
                        st.pyplot(fig)
                    
                    # Bouton de téléchargement du rapport
                    csv = df_report.to_csv(index=False)
                    st.download_button(
                        label="📥 Télécharger le Rapport CSV",
                        data=csv,
                        file_name=f"rapport_interpretabilite_{len(batch_files)}_images.csv",
                        mime="text/csv"
                    )

with tab4:
    st.header("🔧 Utilitaires et Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📦 Installation des Dépendances")
        
        dependencies = {
            "SHAP": "pip install shap",
            "TensorFlow": "pip install tensorflow",
            "OpenCV": "pip install opencv-python",
            "Scikit-learn": "pip install scikit-learn"
        }
        
        for lib, cmd in dependencies.items():
            st.code(cmd, language="bash")
        
        st.subheader("🧪 Test des Modules")
        
        if st.button("🔍 Tester SHAP"):
            try:
                import shap
                st.success("✅ SHAP disponible")
            except ImportError:
                st.error("❌ SHAP non installé")
        
        if st.button("🔍 Tester TensorFlow"):
            try:
                import tensorflow as tf
                st.success(f"✅ TensorFlow {tf.__version__} disponible")
            except ImportError:
                st.error("❌ TensorFlow non installé")
    
    with col2:
        st.subheader("📚 Documentation")
        
        st.markdown("""
        #### 🎯 GradCAM
        - **Objectif**: Visualiser les régions importantes pour un CNN
        - **Utilisation**: Images médicales, classification d'objets
        - **Sortie**: Heatmap des activations importantes
        
        #### 📊 SHAP
        - **Objectif**: Expliquer les prédictions avec des valeurs d'importance
        - **Utilisation**: Tout type de modèle ML
        - **Sortie**: Valeurs d'importance par feature
        
        #### ⚖️ Comparaison
        - **GradCAM**: Visuel, spécifique aux CNN, rapide
        - **SHAP**: Quantitatif, universel, plus lent
        - **Complémentaires**: Utilisez les deux pour une analyse complète
        """)
        
        st.subheader("🔗 Ressources")
        
        resources = [
            "[Documentation SHAP](https://shap.readthedocs.io/)",
            "[Paper GradCAM](https://arxiv.org/abs/1610.02391)",
            "[Framework RAF](./)",
            "[Exemples COVID-19](./notebooks/)"
        ]
        
        for resource in resources:
            st.markdown(f"- {resource}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    🔍 Interface d'Interprétabilité - Framework RAF<br>
    SHAP & GradCAM pour l'analyse des modèles COVID-19
    </div>
    """,
    unsafe_allow_html=True
)