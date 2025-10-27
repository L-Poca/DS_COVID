#!/usr/bin/env python3
"""
Script de test pour vérifier les imports des modules legacy.
Aide au debugging des problèmes d'importation.
"""

import sys
from pathlib import Path

# Ajout des chemins nécessaires
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'src' / 'features'))

def test_imports():
    """Test des imports des modules legacy."""
    
    print("🔍 Test des imports des modules legacy...")
    print(f"Project root: {project_root}")
    print(f"Python path: {sys.path[:3]}...")
    print()
    
    # Test Pipeline_Sklearn
    print("📊 Test Pipeline_Sklearn...")
    try:
        from src.features.Pipelines.Pipeline_Sklearn import PipelineManager as SklearnPipelineManager
        print("✅ Pipeline_Sklearn importé avec succès!")
        print(f"   Classe: {SklearnPipelineManager}")
    except ImportError as e:
        print(f"❌ Erreur Pipeline_Sklearn: {e}")
        
        # Test chemins alternatifs
        try:
            sys.path.append(str(project_root / 'src' / 'features' / 'Pipelines'))
            from Pipeline_Sklearn import PipelineManager as SklearnPipelineManager
            print("✅ Pipeline_Sklearn importé avec chemin alternatif!")
        except ImportError as e2:
            print(f"❌ Échec total Pipeline_Sklearn: {e2}")
    
    print()
    
    # Test Pipeline_TensorFlow
    print("🧠 Test Pipeline_TensorFlow...")
    try:
        from src.features.Pipelines.Pipeline_TensorFlow import TensorFlowPipelineManager
        print("✅ Pipeline_TensorFlow importé avec succès!")
        print(f"   Classe: {TensorFlowPipelineManager}")
    except ImportError as e:
        print(f"❌ Erreur Pipeline_TensorFlow: {e}")
        
        # Test chemins alternatifs
        try:
            sys.path.append(str(project_root / 'src' / 'features' / 'Pipelines'))
            from Pipeline_TensorFlow import TensorFlowPipelineManager
            print("✅ Pipeline_TensorFlow importé avec chemin alternatif!")
        except ImportError as e2:
            print(f"❌ Échec total Pipeline_TensorFlow: {e2}")
    
    print()
    
    # Test covid_data_loader
    print("🦠 Test covid_data_loader...")
    try:
        from src.features.Data_Loaders import covid_data_loader
        print("✅ covid_data_loader importé avec succès!")
        print(f"   Module: {covid_data_loader}")
    except ImportError as e:
        print(f"❌ Erreur covid_data_loader: {e}")
        
        # Test chemins alternatifs
        try:
            sys.path.append(str(project_root / 'src' / 'features' / 'Data_Loaders'))
            import covid_data_loader
            print("✅ covid_data_loader importé avec chemin alternatif!")
        except ImportError as e2:
            print(f"❌ Échec total covid_data_loader: {e2}")
    
    print()
    
    # Test des adaptateurs
    print("🔧 Test des adaptateurs Clean Architecture...")
    try:
        from src.features.Widget_Streamlit.core.adapters.sklearn_pipeline_adapter import SklearnPipelineAdapter
        print("✅ SklearnPipelineAdapter importé!")
    except ImportError as e:
        print(f"❌ Erreur SklearnPipelineAdapter: {e}")
    
    try:
        from src.features.Widget_Streamlit.core.adapters.tensorflow_pipeline_adapter import TensorFlowPipelineAdapter
        print("✅ TensorFlowPipelineAdapter importé!")
    except ImportError as e:
        print(f"❌ Erreur TensorFlowPipelineAdapter: {e}")
    
    try:
        from src.features.Widget_Streamlit.core.adapters.covid_data_adapter import CovidDataLoaderAdapter
        print("✅ CovidDataLoaderAdapter importé!")
    except ImportError as e:
        print(f"❌ Erreur CovidDataLoaderAdapter: {e}")
    
    print()
    
    # Test du DI Container
    print("🏗️ Test du DI Container...")
    try:
        from src.features.Widget_Streamlit.core.di_container import AppContainerFactory
        container = AppContainerFactory.create_container()
        print("✅ DI Container créé avec succès!")
        print(f"   Container: {container}")
    except Exception as e:
        print(f"❌ Erreur DI Container: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎯 Test terminé!")


if __name__ == "__main__":
    test_imports()