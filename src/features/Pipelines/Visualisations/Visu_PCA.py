from matplotlib import pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pandas as pd


def afficher_pca(pca_ref, data_x, data_flat, data_pca):
    """Affiche les résultats de l'analyse PCA.
    
    Args:
        pca_ref: Objet PCA fitted
        data_x: Images originales
        data_flat: Images aplaties avant PCA
        data_pca: Images transformées par PCA
        max_samples: Nombre maximum d'échantillons à analyser (None = tous)
    """

    
    print(f"\n🔬 Analyse en Composantes Principales (PCA)")
    print(f"📊 Nombre de composantes: {pca_ref.n_components}")
    print(f"📈 Échantillons analysés: {len(data_x)}")
    
    # Variance expliquée par chaque composante
    explained_variance_ratio = pca_ref.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # Créer un DataFrame pour les composantes PCA
    n_show = min(20, pca_ref.n_components)  # Augmenté à 20 pour plus de détails
    
    pca_df = pd.DataFrame({
        'Composante': [f'PC{i+1}' for i in range(n_show)],
        'Variance_Expliquée': explained_variance_ratio[:n_show],
        'Variance_Expliquée_%': (explained_variance_ratio[:n_show] * 100),
        'Variance_Cumulée': cumulative_variance[:n_show],
        'Variance_Cumulée_%': (cumulative_variance[:n_show] * 100)
    })
    
    # Formatage pour un affichage plus propre
    pca_df = pca_df.round({
        'Variance_Expliquée': 4,
        'Variance_Expliquée_%': 2,
        'Variance_Cumulée': 4,
        'Variance_Cumulée_%': 2
    })
    
    print(f"\n📈 Variance expliquée par composante:")
    
    # Utiliser display() pour un affichage interactif du DataFrame dans les notebooks
    try:
        from IPython.display import display
        display(pca_df.head(n_show))
    except ImportError:
        # Fallback si IPython n'est pas disponible (ex: script Python classique)
        print(pca_df.to_string(index=False, 
                               col_space={'Composante': 12, 
                                        'Variance_Expliquée': 18,
                                        'Variance_Expliquée_%': 20,
                                        'Variance_Cumulée': 16,
                                        'Variance_Cumulée_%': 18}))
    
    if pca_ref.n_components > n_show:
        print(f"\n... ({pca_ref.n_components - n_show} composantes supplémentaires)")
    
    print(f"\n📈 Variance totale expliquée: {cumulative_variance[-1]:.4f} ({cumulative_variance[-1]*100:.2f}%)")
            
    # Graphique de la variance expliquée
    plt.figure(figsize=(12, 4))
    
    # Subplot 1: Variance par composante
    plt.subplot(1, 2, 1)
    plt.bar(range(1, min(21, pca_ref.n_components + 1)), 
            explained_variance_ratio[:min(20, pca_ref.n_components)], 
            alpha=0.7, color='steelblue')
    plt.xlabel('Composante Principale')
    plt.ylabel('Variance Expliquée')
    plt.title('Variance Expliquée par Composante')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Variance cumulée
    plt.subplot(1, 2, 2)
    plt.plot(range(1, pca_ref.n_components + 1), 
             cumulative_variance, 
             'o-', color='red', linewidth=2, markersize=4)
    plt.axhline(y=0.95, color='green', linestyle='--', alpha=0.7, label='95% variance')
    plt.axhline(y=0.90, color='orange', linestyle='--', alpha=0.7, label='90% variance')
    plt.xlabel('Nombre de Composantes')
    plt.ylabel('Variance Cumulée')
    plt.title('Variance Cumulée Expliquée')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

          
def create_interactive_pca_plot(pca_ref, data_x, data_pca, labels=None):
    """Crée un graphique interactif Plotly avec images au survol et couleurs par classe.
    
    Args:
        pca_ref: Objet PCA fitted
        data_x: Images originales
        data_pca: Données transformées par PCA
        labels: Labels des classes (optionnel)
    
    """
    original_size = len(data_x)
    
    indices = np.arange(len(data_x))
    print(f"📊 Affichage de tous les {len(data_x)} points")
    
    explained_variance_ratio = pca_ref.explained_variance_ratio_
    
    # Gérer les labels
    if labels is None:
        print("⚠️  Aucun label fourni, utilisation d'un gradient par défaut")
        color_values = np.arange(len(data_pca))
        color_labels = [f"Point {i}" for i in range(len(data_pca))]
        colorscale = 'Viridis'
        showscale = True
    else:
        # Créer un mapping couleur pour chaque classe
        unique_labels = list(set(labels))
        color_map = {label: i for i, label in enumerate(unique_labels)}
        color_values = [color_map[label] for label in labels]
        color_labels = labels
        
        # Définir une palette de couleurs distinctes
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
        if len(unique_labels) <= len(colors):
            if len(unique_labels) == 1:
                colorscale = [[0, colors[0]], [1, colors[0]]]
            else:
                colorscale = [[i/(len(unique_labels)-1), colors[i]] for i in range(len(unique_labels))]
        else:
            colorscale = 'Set3'
        showscale = False
        
    # Graphique 3D si disponible
    if pca_ref.n_components >= 3:
        print(f"\n🎲 Visualisation 3D de l'espace PCA (colorée par classes):")

        fig_3d = go.Figure(data=[go.Scatter3d(
            x=data_pca[:, 0],
            y=data_pca[:, 1],
            z=data_pca[:, 2],
            mode='markers',
            marker=dict(
                size=6,
                color=color_values,
                colorscale=colorscale,
                showscale=showscale,
                line=dict(width=0.5, color='DarkSlateGrey'),
                opacity=0.8
            ),
            hovertemplate='<b>Image %{text}</b><br>' +
                         'Classe: %{customdata}<br>' +
                         'PC1: %{x:.3f}<br>' +
                         'PC2: %{y:.3f}<br>' +
                         'PC3: %{z:.3f}<br>' +
                         '<extra></extra>',
            text=indices,
            customdata=color_labels,
            name='Images par classe'
        )])        

        fig_3d.update_layout(
            title=f'🎲 Espace PCA 3D - Distribution par Classes ({len(data_pca)} points)',
            scene=dict(
                xaxis_title=f'PC1 ({explained_variance_ratio[0]:.1%})',
                yaxis_title=f'PC2 ({explained_variance_ratio[1]:.1%})',
                zaxis_title=f'PC3 ({explained_variance_ratio[2]:.1%})'
            ),
            width=900,
            height=700
        )
        
        # Ajouter une légende manuelle si on a des classes
        if labels is not None:
            # Créer des traces invisibles pour la légende
            for i, label in enumerate(unique_labels):
                fig_3d.add_trace(go.Scatter3d(
                    x=[None], y=[None], z=[None],
                    mode='markers',
                    marker=dict(size=10, color=colors[i] if i < len(colors) else f'rgb({i*50%256},{i*80%256},{i*110%256})'),
                    name=label,
                    showlegend=True
                ))
            
        fig_3d.show()
    
    # Graphique 2D également coloré par classes
    if pca_ref.n_components >= 2:
        print(f"\n📊 Visualisation 2D de l'espace PCA (colorée par classes):")
        
        fig_2d = go.Figure(data=[go.Scatter(
            x=data_pca[:, 0],
            y=data_pca[:, 1],
            mode='markers',
            marker=dict(
                size=8,
                color=color_values,
                colorscale=colorscale,
                showscale=showscale,
                line=dict(width=1, color='DarkSlateGrey'),
                opacity=0.8
            ),
            hovertemplate='<b>Image %{text}</b><br>' +
                         'Classe: %{customdata}<br>' +
                         'PC1: %{x:.3f}<br>' +
                         'PC2: %{y:.3f}<br>' +
                         '<extra></extra>',
            text=indices,
            customdata=color_labels,
            name='Images par classe'
        )])
        
        fig_2d.update_layout(
            title=f'📊 Espace PCA 2D - Distribution par Classes ({len(data_pca)} points)',
            xaxis_title=f'PC1 ({explained_variance_ratio[0]:.1%})',
            yaxis_title=f'PC2 ({explained_variance_ratio[1]:.1%})',
            width=800,
            height=600
        )
        
        # Ajouter une légende manuelle pour le 2D aussi
        if labels is not None:
            for i, label in enumerate(unique_labels):
                fig_2d.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(size=10, color=colors[i] if i < len(colors) else f'rgb({i*50%256},{i*80%256},{i*110%256})'),
                    name=label,
                    showlegend=True
                ))
        
        fig_2d.show()
        
    # Affichage des statistiques par classe
    if labels is not None:
        print(f"\n📈 Statistiques par classe (échantillon affiché):")
        label_counts = pd.Series(labels).value_counts()
        for label, count in label_counts.items():
            percentage = (count / len(labels)) * 100
            print(f"  🏷️  {label}: {count} images ({percentage:.1f}%)")

    print("\n✅ Visualisations interactives créées avec couleurs par classe !")
    print("\n💡 Survolez les points pour voir les détails de chaque image\n")
