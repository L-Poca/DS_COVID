#!/usr/bin/env python3
"""
Test d'importation des nouveaux pipelines
Vérifie que tous les modules peuvent être importés correctement
"""

import sys
from pathlib import Path

# Ajouter le projet au path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_imports():
    """Test des imports des pipelines"""
    success_count = 0
    total_tests = 0
    
    # Test Pipeline Sklearn
    total_tests += 1
    try:
        from src.features.Pipelines.Pipeline_Sklearn import PipelineManager
        print("✅ Pipeline_Sklearn importé avec succès")
        success_count += 1
    except Exception as e:
        print(f"❌ Erreur Pipeline_Sklearn: {e}")
    
    # Test Pipeline TensorFlow
    total_tests += 1
    try:
        from src.features.Pipelines.Pipeline_TensorFlow import TensorFlowPipelineManager
        print("✅ Pipeline_TensorFlow importé avec succès")
        success_count += 1
    except Exception as e:
        print(f"❌ Erreur Pipeline_TensorFlow: {e}")
    
    # Test Pipeline Data Augmentation
    total_tests += 1
    try:
        from src.features.Pipelines.Pipeline_DataAugmentation import DataAugmentationPipeline
        print("✅ Pipeline_DataAugmentation importé avec succès")
        success_count += 1
    except Exception as e:
        print(f"❌ Erreur Pipeline_DataAugmentation: {e}")
    
    # Test COVID Data Loader
    total_tests += 1
    try:
        from src.features.Data_Loaders.covid_data_loader import (
            load_covid_dataset, prepare_data_for_sklearn, prepare_data_for_tensorflow,
            get_data_paths, check_data_availability
        )
        print("✅ covid_data_loader importé avec succès")
        success_count += 1
    except Exception as e:
        print(f"❌ Erreur covid_data_loader: {e}")
    
    # Test configurations
    total_tests += 1
    try:
        config_sklearn = project_root / "src" / "features" / "Pipelines" / "Pipeline_Sklearn_config.json"
        config_tf = project_root / "src" / "features" / "Pipelines" / "Pipeline_TensorFlow_config.json"
        config_aug = project_root / "src" / "features" / "Pipelines" / "Pipeline_DataAugmentation_config.json"
        
        if config_sklearn.exists() and config_tf.exists() and config_aug.exists():
            print("✅ Fichiers de configuration trouvés")
            success_count += 1
        else:
            print(f"❌ Fichiers de configuration manquants:")
            print(f"  - Sklearn: {config_sklearn.exists()}")
            print(f"  - TensorFlow: {config_tf.exists()}")
            print(f"  - Augmentation: {config_aug.exists()}")
    except Exception as e:
        print(f"❌ Erreur vérification configs: {e}")
    
    # Test disponibilité des données
    total_tests += 1
    try:
        data_path = project_root / "data" / "raw" / "COVID-19_Radiography_Dataset"
        if data_path.exists():
            print("✅ Dossier de données COVID-19 trouvé")
            success_count += 1
        else:
            print(f"❌ Dossier de données manquant: {data_path}")
    except Exception as e:
        print(f"❌ Erreur vérification données: {e}")
    
    print(f"\n📊 Résultats: {success_count}/{total_tests} tests réussis")
    
    if success_count == total_tests:
        print("🎉 Tous les tests d'importation ont réussi!")
        return True
    else:
        print("⚠️ Certains imports ont échoué")
        return False

if __name__ == "__main__":
    print("🔍 Test d'importation des pipelines COVID-19")
    print("=" * 50)
    test_imports()