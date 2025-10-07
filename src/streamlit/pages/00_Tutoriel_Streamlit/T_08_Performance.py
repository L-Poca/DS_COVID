import streamlit as st
import time
import pandas as pd
import numpy as np
import psutil
import gc
from datetime import datetime, timedelta
import threading
import plotly.express as px
import plotly.graph_objects as go

st.header("⚡ T_08 - Optimisation des Performances")

st.markdown("**📋 Objectif :** Maîtriser l'optimisation complète des applications Streamlit : cache intelligent, monitoring système, gestion mémoire, et techniques avancées pour créer des applications ultra-performantes et scalables.")

st.markdown("---")

# ================================
# 1. STRATÉGIES DE CACHE AVANCÉES
# ================================
st.subheader("1️⃣ Stratégies de cache avancées")

st.markdown("""
**📖 Description :**
Le cache est l'arme secrète des applications performantes. Au-delà du simple stockage,
une stratégie de cache intelligente peut transformer une application lente en solution
ultra-réactive. Streamlit offre des outils puissants pour optimiser chaque aspect.

**🎯 Types de cache et leurs usages :**
- **`@st.cache_data`** : Données sérialisables (DataFrames, listes, résultats calculs)
- **`@st.cache_resource`** : Ressources non-sérialisables (connexions DB, modèles ML)
- **Cache conditionnel** : Invalidation basée sur paramètres métier
- **Cache hiérarchique** : Différents niveaux de cache selon la criticité
- **Cache partagé** : Optimisation pour applications multi-utilisateurs

**💡 Stratégies d'optimisation :**
- TTL adaptatif selon la fréquence de mise à jour des données
- Pré-calcul des résultats les plus fréquents
- Cache en cascade pour optimiser les dépendances
- Monitoring des hit/miss ratios pour ajustements
- Éviction intelligente des données périmées
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code('''
# Cache de données avec TTL adaptatif
@st.cache_data(
    ttl=3600,  # 1 heure pour données statiques
    max_entries=100,
    show_spinner="Chargement des données..."
)
def load_business_data(data_type, filters=None):
    """Cache intelligent pour données métier"""
    start_time = time.time()
    
    # Simulation chargement selon le type
    if data_type == "sales":
        time.sleep(1.5)  # Simulation requête DB lourde
        np.random.seed(42)
        data = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=365),
            'Ventes': np.cumsum(np.random.randn(365) * 100) + 10000,
            'Produit': np.random.choice(['A', 'B', 'C'], 365)
        })
    elif data_type == "customers":
        time.sleep(1.0)
        data = pd.DataFrame({
            'ID': range(1000),
            'Age': np.random.randint(18, 70, 1000),
            'Segment': np.random.choice(['Premium', 'Standard', 'Basic'], 1000)
        })
    else:
        time.sleep(0.5)
        data = pd.DataFrame({'empty': []})
    
    # Appliquer les filtres si fournis
    if filters and not data.empty:
        for filter_col, filter_val in filters.items():
            if filter_col in data.columns:
                data = data[data[filter_col] == filter_val]
    
    processing_time = time.time() - start_time
    st.info(f"⏱️ Temps de traitement: {processing_time:.2f}s")
    
    return data

# Cache de ressources pour modèles/connexions
@st.cache_resource
def initialize_ml_model():
    """Cache pour ressources lourdes"""
    time.sleep(2)  # Simulation chargement modèle
    return {
        'model_type': 'RandomForest',
        'accuracy': 0.95,
        'features': ['feature_1', 'feature_2', 'feature_3'],
        'loaded_at': datetime.now().isoformat()
    }

# Cache conditionnel avec hash_funcs personnalisés
@st.cache_data(
    ttl=600,  # 10 minutes
    max_entries=50,
    hash_funcs={dict: lambda x: str(sorted(x.items()))}
)
def compute_advanced_metrics(data_dict, calculation_type):
    """Cache pour calculs métier complexes"""
    time.sleep(1)  # Simulation calcul intensif
    
    if calculation_type == "aggregation":
        result = {
            'total': sum(data_dict.values()),
            'average': np.mean(list(data_dict.values())),
            'max': max(data_dict.values()),
            'min': min(data_dict.values())
        }
    elif calculation_type == "forecast":
        values = list(data_dict.values())
        trend = np.polyfit(range(len(values)), values, 1)[0]
        result = {
            'trend': trend,
            'next_month': values[-1] + trend,
            'confidence': 0.85
        }
    else:
        result = {'error': 'Unknown calculation type'}
    
    return result

# Interface de démonstration
st.markdown("**🚀 Test des Caches Avancés**")

# Sélection du type de données
data_type = st.selectbox(
    "Type de données à charger:",
    ["sales", "customers", "analytics"],
    help="Chaque type a un temps de chargement différent"
)

# Options de filtrage
with st.expander("🔍 Filtres avancés"):
    filter_options = {}
    if data_type == "sales":
        product_filter = st.selectbox("Produit:", ["Tous", "A", "B", "C"])
        if product_filter != "Tous":
            filter_options["Produit"] = product_filter
    elif data_type == "customers":
        segment_filter = st.selectbox("Segment:", ["Tous", "Premium", "Standard", "Basic"])
        if segment_filter != "Tous":
            filter_options["Segment"] = segment_filter

# Boutons de test de cache
col_cache1, col_cache2, col_cache3 = st.columns(3)

with col_cache1:
    if st.button("📊 Charger Données", type="primary"):
        start_load = time.time()
        data = load_business_data(data_type, filter_options if filter_options else None)
        end_load = time.time()
        
        st.success(f"✅ Données chargées en {end_load - start_load:.2f}s")
        st.dataframe(data.head(), use_container_width=True)
        st.info(f"📈 {len(data)} lignes chargées")

with col_cache2:
    if st.button("🤖 Initialiser Modèle"):
        model_info = initialize_ml_model()
        st.json(model_info)
        st.success("🎯 Modèle prêt (mis en cache pour réutilisation)")

with col_cache3:
    if st.button("📊 Calculs Avancés"):
        # Données d'exemple pour calculs
        sample_data = {
            'jan': 1000, 'feb': 1200, 'mar': 1100, 
            'apr': 1300, 'may': 1250, 'jun': 1400
        }
        
        calc_type = st.radio("Type de calcul:", ["aggregation", "forecast"], horizontal=True)
        
        results = compute_advanced_metrics(sample_data, calc_type)
        st.json(results)

# Gestion du cache
st.markdown("**🛠️ Gestion du Cache**")
cache_cols = st.columns(3)

with cache_cols[0]:
    if st.button("🗑️ Vider Cache Données"):
        st.cache_data.clear()
        st.success("Cache données vidé!")

with cache_cols[1]:
    if st.button("🗑️ Vider Cache Ressources"):
        st.cache_resource.clear()
        st.success("Cache ressources vidé!")

with cache_cols[2]:
    # Statistiques de cache (simulation)
    st.metric("Cache Hits", "🔥 87%", "↗️ +5%")
''', language="python")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Démonstration simplifiée des caches
    @st.cache_data(ttl=300)
    def demo_load_data(data_size):
        time.sleep(0.5)  # Simulation charge
        np.random.seed(42)
        return pd.DataFrame({
            'ID': range(data_size),
            'Valeur': np.random.randn(data_size) * 100,
            'Catégorie': np.random.choice(['A', 'B', 'C'], data_size)
        })
    
    @st.cache_resource  
    def demo_init_resource():
        time.sleep(0.3)
        return {'status': 'ready', 'version': '1.0'}
    
    st.markdown("**⚡ Tests Cache**")
    
    demo_size = st.slider("Taille des données", 10, 1000, 100, key="demo_cache_size")
    
    if st.button("🔄 Charger (avec cache)", key="demo_cache_load"):
        start = time.time()
        data = demo_load_data(demo_size)
        end = time.time()
        
        st.success(f"✅ Chargé en {end-start:.3f}s")
        st.write(f"📊 {len(data)} lignes")
        st.dataframe(data.head(3))
        
        if end - start < 0.1:
            st.info("⚡ Données servies depuis le cache!")
    
    if st.button("🎛️ Initialiser Ressource", key="demo_resource"):
        resource = demo_init_resource()
        st.json(resource)
    
    # Stats de cache simplifié
    demo_stats_col1, demo_stats_col2 = st.columns(2)
    with demo_stats_col1:
        st.metric("Cache Data", "Actif", "✅")
    with demo_stats_col2:
        st.metric("Cache Resource", "Actif", "✅")

st.divider()

# ================================
# 2. MONITORING SYSTÈME ET PERFORMANCE
# ================================
st.subheader("2️⃣ Monitoring système et performance")

st.markdown("""
**📖 Description :**
Le monitoring en temps réel est essentiel pour maintenir des performances optimales.
Il permet de détecter les goulots d'étranglement, d'anticiper les problèmes de capacité,
et d'optimiser l'allocation des ressources pour une expérience utilisateur fluide.

**🎯 Métriques critiques à surveiller :**
- **CPU et Mémoire** : Utilisation des ressources système
- **Temps de réponse** : Latence des opérations critiques
- **Cache hit ratio** : Efficacité du système de cache
- **Concurrent users** : Charge utilisateur simultanée
- **Error rates** : Taux d'erreur et stabilité

**💡 Techniques d'optimisation :**
- Alertes proactives sur les seuils critiques
- Auto-scaling basé sur la charge
- Profilage des fonctions les plus coûteuses
- Optimisation des requêtes et calculs
- Load balancing intelligent
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code('''
# Monitoring des ressources système
def get_system_metrics():
    """Collecte des métriques système"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    return {
        'cpu_percent': cpu_percent,
        'memory_percent': memory.percent,
        'memory_available_gb': memory.available / (1024**3),
        'disk_percent': disk.percent,
        'disk_free_gb': disk.free / (1024**3)
    }

# Profilage de performance
def profile_function_performance(func, *args, **kwargs):
    """Profile les performances d'une fonction"""
    import tracemalloc
    
    # Démarrer le monitoring mémoire
    tracemalloc.start()
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss
    
    # Exécuter la fonction
    result = func(*args, **kwargs)
    
    # Mesurer les ressources
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    metrics = {
        'execution_time': end_time - start_time,
        'memory_delta_mb': (end_memory - start_memory) / (1024*1024),
        'peak_memory_mb': peak / (1024*1024),
        'success': True
    }
    
    return result, metrics

# Dashboard de monitoring en temps réel
def create_performance_dashboard():
    """Crée un dashboard de monitoring"""
    
    # Métriques système actuelles
    metrics = get_system_metrics()
    
    # Affichage des métriques principales
    metric_cols = st.columns(4)
    
    with metric_cols[0]:
        cpu_color = "🔴" if metrics['cpu_percent'] > 80 else "🟡" if metrics['cpu_percent'] > 60 else "🟢"
        st.metric(
            f"{cpu_color} CPU", 
            f"{metrics['cpu_percent']:.1f}%",
            help="Utilisation CPU actuelle"
        )
    
    with metric_cols[1]:
        mem_color = "🔴" if metrics['memory_percent'] > 85 else "🟡" if metrics['memory_percent'] > 70 else "🟢"
        st.metric(
            f"{mem_color} Mémoire", 
            f"{metrics['memory_percent']:.1f}%",
            f"({metrics['memory_available_gb']:.1f}GB libre)"
        )
    
    with metric_cols[2]:
        disk_color = "🔴" if metrics['disk_percent'] > 90 else "🟡" if metrics['disk_percent'] > 80 else "🟢"
        st.metric(
            f"{disk_color} Disque", 
            f"{metrics['disk_percent']:.1f}%",
            f"({metrics['disk_free_gb']:.1f}GB libre)"
        )
    
    with metric_cols[3]:
        # Simulation charge utilisateur
        current_users = np.random.randint(5, 25)
        user_color = "🔴" if current_users > 20 else "🟡" if current_users > 15 else "🟢"
        st.metric(f"{user_color} Utilisateurs", current_users)
    
    return metrics

# Fonction de test pour le profilage
@st.cache_data
def heavy_computation(n_iterations, use_cache=True):
    """Simulation calcul intensif pour tests de performance"""
    result = 0
    for i in range(n_iterations):
        result += np.sum(np.random.randn(1000))
    return result

# Interface de monitoring
st.markdown("**📊 Dashboard de Monitoring**")

# Actualisation automatique
auto_refresh = st.checkbox("🔄 Actualisation automatique (5s)", value=False)

if auto_refresh:
    # Placeholder pour refresh automatique
    dashboard_placeholder = st.empty()
    
    with dashboard_placeholder.container():
        metrics = create_performance_dashboard()
        st.caption(f"Dernière mise à jour: {datetime.now().strftime('%H:%M:%S')}")
else:
    if st.button("🔄 Actualiser Métriques"):
        metrics = create_performance_dashboard()

# Test de performance avec profilage
st.markdown("**⚙️ Test de Performance avec Profilage**")

test_cols = st.columns(2)

with test_cols[0]:
    iterations = st.slider("Nombre d'itérations", 100, 10000, 1000)
    use_cache_test = st.checkbox("Utiliser le cache", value=True)

with test_cols[1]:
    if st.button("🚀 Lancer Test de Performance"):
        with st.spinner("Test en cours..."):
            result, perf_metrics = profile_function_performance(
                heavy_computation, iterations, use_cache_test
            )
        
        st.success("✅ Test terminé!")
        
        # Affichage des résultats de performance
        perf_result_cols = st.columns(3)
        
        with perf_result_cols[0]:
            st.metric("Temps d'exécution", f"{perf_metrics['execution_time']:.3f}s")
        
        with perf_result_cols[1]:
            st.metric("Mémoire utilisée", f"{perf_metrics['memory_delta_mb']:.1f}MB")
        
        with perf_result_cols[2]:
            st.metric("Pic mémoire", f"{perf_metrics['peak_memory_mb']:.1f}MB")
        
        # Recommandations basées sur les métriques
        if perf_metrics['execution_time'] > 2.0:
            st.warning("⚠️ Temps d'exécution élevé - Considérez l'optimisation")
        if perf_metrics['memory_delta_mb'] > 100:
            st.warning("⚠️ Consommation mémoire importante - Vérifiez les fuites")
        if perf_metrics['execution_time'] < 0.1:
            st.info("⚡ Excellent! Optimisation réussie")
''', language="python")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Monitoring simplifié pour la démo
    def demo_get_metrics():
        return {
            'cpu': np.random.uniform(20, 80),
            'memory': np.random.uniform(40, 85),
            'users': np.random.randint(3, 15)
        }
    
    st.markdown("**📊 Monitoring Demo**")
    
    if st.button("📈 Vérifier Système", key="demo_monitoring"):
        demo_metrics = demo_get_metrics()
        
        demo_metric_cols = st.columns(3)
        
        with demo_metric_cols[0]:
            cpu_status = "🟢" if demo_metrics['cpu'] < 60 else "🟡" if demo_metrics['cpu'] < 80 else "🔴"
            st.metric(f"{cpu_status} CPU", f"{demo_metrics['cpu']:.1f}%")
        
        with demo_metric_cols[1]:
            mem_status = "🟢" if demo_metrics['memory'] < 70 else "🟡" if demo_metrics['memory'] < 85 else "🔴"
            st.metric(f"{mem_status} RAM", f"{demo_metrics['memory']:.1f}%")
        
        with demo_metric_cols[2]:
            user_status = "🟢" if demo_metrics['users'] < 10 else "🟡"
            st.metric(f"{user_status} Users", demo_metrics['users'])
    
    # Test de performance simplifié
    st.markdown("**⚡ Test Performance**")
    
    demo_iterations = st.slider("Itérations de test", 10, 100, 50, key="demo_perf_iterations")
    
    if st.button("🧪 Tester", key="demo_perf_test"):
        start_demo = time.time()
        
        # Simulation calcul
        result = sum(range(demo_iterations * 100))
        
        end_demo = time.time()
        execution_time = end_demo - start_demo
        
        st.success(f"✅ Terminé en {execution_time:.3f}s")
        
        # Évaluation performance
        if execution_time < 0.01:
            st.info("⚡ Excellent!")
        elif execution_time < 0.1:
            st.info("👍 Bon")
        else:
            st.warning("⚠️ Optimisation possible")

st.divider()

# ================================
# 3. OPTIMISATION MÉMOIRE ET GARBAGE COLLECTION
# ================================
st.subheader("3️⃣ Optimisation mémoire et garbage collection")

st.markdown("""
**📖 Description :**
La gestion intelligente de la mémoire est cruciale pour maintenir des performances stables,
surtout dans les applications avec de gros volumes de données ou des sessions utilisateur longues.
Une stratégie d'optimisation mémoire évite les fuites et maintient la réactivité.

**🎯 Techniques d'optimisation mémoire :**
- **Garbage Collection manuel** : Libération proactive de la mémoire
- **Weak references** : Références qui n'empêchent pas le GC
- **Memory profiling** : Détection des fuites et optimisations
- **Lazy loading** : Chargement à la demande pour économiser la RAM
- **Data chunking** : Traitement par blocs pour gros datasets

**💡 Stratégies avancées :**
- Session state cleanup automatique
- Cache size limits adaptatifs
- Monitoring des memory leaks
- Optimisation des structures de données
- Compression en mémoire pour gros volumes
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💻 Code")
    st.code('''
# Utilitaires de gestion mémoire
def get_memory_usage():
    """Analyse détaillée de l'usage mémoire"""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_percent = process.memory_percent()
    
    return {
        'rss_mb': memory_info.rss / (1024 * 1024),  # Resident Set Size
        'vms_mb': memory_info.vms / (1024 * 1024),  # Virtual Memory Size
        'percent': memory_percent,
        'available_mb': psutil.virtual_memory().available / (1024 * 1024)
    }

def optimize_session_state():
    """Nettoyage intelligent du session state"""
    keys_to_remove = []
    total_size = 0
    
    for key, value in st.session_state.items():
        # Calculer la taille approximative
        try:
            import sys
            size = sys.getsizeof(value)
            total_size += size
            
            # Marquer pour suppression si trop ancien ou gros
            if key.startswith('temp_') or size > 10 * 1024 * 1024:  # 10MB
                keys_to_remove.append(key)
        except:
            pass
    
    # Nettoyer les clés identifiées
    for key in keys_to_remove:
        if key in st.session_state:
            del st.session_state[key]
    
    return len(keys_to_remove), total_size

def process_large_dataset_chunked(data, chunk_size=1000):
    """Traitement par chunks pour économiser la mémoire"""
    results = []
    
    for i in range(0, len(data), chunk_size):
        chunk = data.iloc[i:i+chunk_size] if hasattr(data, 'iloc') else data[i:i+chunk_size]
        
        # Traitement du chunk
        processed_chunk = chunk.copy() if hasattr(chunk, 'copy') else list(chunk)
        
        # Simulation traitement
        time.sleep(0.01)  # Simulation calcul
        
        results.append(processed_chunk)
        
        # Libération mémoire explicite
        del chunk, processed_chunk
        
        # Force garbage collection périodique
        if i % (chunk_size * 5) == 0:
            gc.collect()
    
    return results

# Interface de gestion mémoire
st.markdown("**🧠 Gestionnaire de Mémoire**")

# Monitoring mémoire en temps réel
if st.button("📊 Analyser Mémoire Actuelle"):
    memory_stats = get_memory_usage()
    
    mem_cols = st.columns(4)
    
    with mem_cols[0]:
        st.metric("RSS", f"{memory_stats['rss_mb']:.1f} MB", 
                 help="Mémoire physique réellement utilisée")
    
    with mem_cols[1]:
        st.metric("VMS", f"{memory_stats['vms_mb']:.1f} MB",
                 help="Mémoire virtuelle totale")
    
    with mem_cols[2]:
        st.metric("Utilisation", f"{memory_stats['percent']:.1f}%",
                 help="Pourcentage de mémoire système utilisée")
    
    with mem_cols[3]:
        st.metric("Disponible", f"{memory_stats['available_mb']:.0f} MB",
                 help="Mémoire système disponible")

# Test de traitement par chunks
st.markdown("**⚙️ Test Traitement par Chunks**")

chunk_test_cols = st.columns(2)

with chunk_test_cols[0]:
    data_size = st.slider("Taille du dataset", 1000, 50000, 10000, step=1000)
    chunk_size = st.slider("Taille des chunks", 100, 5000, 1000, step=100)

with chunk_test_cols[1]:
    if st.button("🚀 Test Chunked Processing"):
        # Créer un dataset de test
        test_data = pd.DataFrame({
            'id': range(data_size),
            'value': np.random.randn(data_size),
            'category': np.random.choice(['A', 'B', 'C'], data_size)
        })
        
        # Mesurer la mémoire avant
        memory_before = get_memory_usage()
        
        # Traitement par chunks
        with st.spinner("Traitement en cours..."):
            start_time = time.time()
            results = process_large_dataset_chunked(test_data, chunk_size)
            end_time = time.time()
        
        # Mesurer la mémoire après
        memory_after = get_memory_usage()
        
        st.success(f"✅ Traité en {end_time - start_time:.2f}s")
        
        # Afficher les stats
        processing_cols = st.columns(3)
        
        with processing_cols[0]:
            st.metric("Chunks traités", len(results))
        
        with processing_cols[1]:
            memory_delta = memory_after['rss_mb'] - memory_before['rss_mb']
            st.metric("Delta mémoire", f"{memory_delta:+.1f} MB")
        
        with processing_cols[2]:
            throughput = data_size / (end_time - start_time)
            st.metric("Débit", f"{throughput:.0f} lignes/s")

# Nettoyage et optimisation
st.markdown("**🧹 Nettoyage et Optimisation**")

cleanup_cols = st.columns(3)

with cleanup_cols[0]:
    if st.button("🗑️ Nettoyer Session State"):
        removed_count, total_size = optimize_session_state()
        st.success(f"✅ {removed_count} clés supprimées")
        st.info(f"📊 Taille totale analysée: {total_size / (1024*1024):.1f} MB")

with cleanup_cols[1]:
    if st.button("♻️ Force Garbage Collection"):
        # Mesurer avant
        memory_before_gc = get_memory_usage()
        
        # Forcer le garbage collection
        collected = gc.collect()
        
        # Mesurer après
        memory_after_gc = get_memory_usage()
        memory_freed = memory_before_gc['rss_mb'] - memory_after_gc['rss_mb']
        
        st.success(f"✅ {collected} objets collectés")
        if memory_freed > 0:
            st.info(f"🆓 {memory_freed:.1f} MB libérés")
        else:
            st.info("💚 Mémoire déjà optimisée")

with cleanup_cols[2]:
    # Statistiques du garbage collector
    gc_stats = gc.get_stats()
    if gc_stats:
        st.metric("GC Génération 0", gc_stats[0]['collections'])
''', language="python")

with col2:
    st.markdown("#### 🎯 Résultat")
    
    # Simulation monitoring mémoire
    st.markdown("**🧠 Monitoring Mémoire**")
    
    if st.button("🔍 Analyser", key="demo_memory"):
        # Simulation métriques mémoire
        demo_rss = np.random.uniform(50, 150)
        demo_percent = np.random.uniform(30, 80)
        
        demo_mem_cols = st.columns(2)
        
        with demo_mem_cols[0]:
            st.metric("Mémoire utilisée", f"{demo_rss:.1f} MB")
        
        with demo_mem_cols[1]:
            mem_status = "🟢" if demo_percent < 70 else "🟡" if demo_percent < 85 else "🔴"
            st.metric(f"{mem_status} Utilisation", f"{demo_percent:.1f}%")
    
    # Test de chunking simplifié
    st.markdown("**⚙️ Test Chunking**")
    
    demo_data_size = st.slider("Taille test", 100, 1000, 500, key="demo_chunk_size")
    
    if st.button("🧪 Test Chunking", key="demo_chunk_test"):
        start_chunk = time.time()
        
        # Simulation traitement par chunks
        chunks_count = demo_data_size // 100
        
        progress_bar = st.progress(0)
        for i in range(chunks_count):
            progress_bar.progress((i + 1) / chunks_count)
            time.sleep(0.05)
        
        end_chunk = time.time()
        
        st.success(f"✅ {chunks_count} chunks traités en {end_chunk - start_chunk:.2f}s")
        st.info("💡 Mémoire optimisée par traitement séquentiel")
    
    # Nettoyage demo
    st.markdown("**🧹 Nettoyage**")
    
    demo_cleanup_cols = st.columns(2)
    
    with demo_cleanup_cols[0]:
        if st.button("🗑️ Nettoyer", key="demo_cleanup"):
            st.success("✅ Session nettoyée")
    
    with demo_cleanup_cols[1]:
        if st.button("♻️ GC", key="demo_gc"):
            st.success("✅ Mémoire optimisée")

st.markdown("---")

st.success("🎉 **Félicitations !** Vous maîtrisez maintenant l'optimisation complète des performances Streamlit !")

st.markdown("""
**🚀 Points clés à retenir :**

**⚡ Cache Avancé :**
- Utilisez `@st.cache_data` pour les données et `@st.cache_resource` pour les ressources
- Configurez TTL et max_entries selon vos besoins
- Implémentez des stratégies de cache hiérarchique
- Monitorer les hit/miss ratios pour optimiser

**📊 Monitoring Système :**
- Surveillez CPU, mémoire, et charge utilisateur
- Profilez les fonctions critiques pour identifier les goulots
- Implémentez des alertes proactives sur les seuils
- Optimisez basé sur les métriques réelles

**🧠 Gestion Mémoire :**
- Nettoyez régulièrement le session state
- Utilisez le traitement par chunks pour gros volumes
- Forcez le garbage collection si nécessaire
- Monitoring continu pour éviter les fuites

**🔗 Prochaine étape :** Découvrez T_09_Astuces pour les techniques avancées et les secrets de développement !
""")
