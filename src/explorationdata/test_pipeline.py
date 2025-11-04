#!/usr/bin/env python
"""
Test script to validate pipeline structure and imports
Can run without actual dataset
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")

    try:
        from src.explorationdata.pipeline import (
            data_loader,
            embedding_extractor,
            dimensionality_reducer,
            clustering_analyzer,
            visualizer,
            advanced_analysis,
            pipeline_runner
        )
        print("✓ All pipeline modules imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_class_initialization():
    """Test that classes can be initialized with mock data"""
    print("\nTesting class initialization...")

    try:
        from src.explorationdata.pipeline.data_loader import DatasetLoader
        from src.explorationdata.pipeline.visualizer import Visualizer
        from src.explorationdata.pipeline.dimensionality_reducer import DimensionalityReducer
        from src.explorationdata.pipeline.clustering_analyzer import ClusteringAnalyzer
        from src.explorationdata.pipeline.advanced_analysis import AdvancedAnalyzer

        # These should not fail even with invalid paths
        # They only check initialization
        loader = DatasetLoader(
            base_path="/fake/path",
            metadata_path="/fake/metadata"
        )
        print("✓ DatasetLoader initialized")

        visualizer = Visualizer(random_state=42)
        print("✓ Visualizer initialized")

        dim_reducer = DimensionalityReducer(random_state=42)
        print("✓ DimensionalityReducer initialized")

        clustering = ClusteringAnalyzer(random_state=42)
        print("✓ ClusteringAnalyzer initialized")

        advanced = AdvancedAnalyzer(random_state=42)
        print("✓ AdvancedAnalyzer initialized")

        return True

    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return False


def test_dependencies():
    """Test that required dependencies are available"""
    print("\nTesting dependencies...")

    dependencies = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'sklearn': 'scikit-learn',
        'PIL': 'Pillow',
        'cv2': 'opencv-python',
    }

    optional_deps = {
        'torch': 'torch',
        'torchvision': 'torchvision',
        'umap': 'umap-learn',
    }

    all_ok = True

    print("\nRequired dependencies:")
    for module, package in dependencies.items():
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - MISSING (required)")
            all_ok = False

    print("\nOptional dependencies (for full functionality):")
    for module, package in optional_deps.items():
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ⚠ {package} - missing (optional, limits functionality)")

    return all_ok


def test_mock_pipeline():
    """Test pipeline with mock data"""
    print("\nTesting pipeline with mock data...")

    try:
        import numpy as np
        import pandas as pd
        from src.explorationdata.pipeline.dimensionality_reducer import DimensionalityReducer
        from src.explorationdata.pipeline.clustering_analyzer import ClusteringAnalyzer

        # Create mock embeddings
        n_samples = 100
        n_features = 50
        mock_embeddings = np.random.randn(n_samples, n_features)
        mock_labels = ['class_A'] * 50 + ['class_B'] * 50

        print(f"  Created mock embeddings: {mock_embeddings.shape}")

        # Test PCA
        dim_reducer = DimensionalityReducer(random_state=42)
        pca_embeddings, pca_model = dim_reducer.fit_pca(
            mock_embeddings,
            n_components=10
        )
        print(f"  ✓ PCA completed: {pca_embeddings.shape}")

        # Test clustering
        clustering = ClusteringAnalyzer(random_state=42)
        kmeans_labels = clustering.fit_kmeans(mock_embeddings, n_clusters=2)
        print(f"  ✓ KMeans completed: {len(kmeans_labels)} labels")

        # Test evaluation
        metrics = clustering.evaluate_clustering(
            mock_labels,
            kmeans_labels,
            "KMeans"
        )
        print(f"  ✓ Clustering evaluation: ARI={metrics['ari']:.3f}")

        return True

    except Exception as e:
        print(f"  ✗ Mock pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_script_exists():
    """Test that all scripts exist and are executable"""
    print("\nTesting script files...")

    scripts = [
        'src/explorationdata/run_eda_pipeline.py',
        'src/explorationdata/generate_report.py',
        'notebooks/Complete_EDA_COVID_Dataset.ipynb',
        'src/explorationdata/README_EDA_PIPELINE.md'
    ]

    all_exist = True
    for script in scripts:
        path = Path(script)
        if path.exists():
            print(f"  ✓ {script}")
        else:
            print(f"  ✗ {script} - MISSING")
            all_exist = False

    return all_exist


def main():
    """Run all tests"""
    print("="*60)
    print("EDA Pipeline Test Suite")
    print("="*60)

    results = []

    results.append(("Imports", test_imports()))
    results.append(("Dependencies", test_dependencies()))
    results.append(("Class Initialization", test_class_initialization()))
    results.append(("Mock Pipeline", test_mock_pipeline()))
    results.append(("Script Files", test_script_exists()))

    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    all_passed = True
    for test_name, passed in results:
        status = "PASSED" if passed else "FAILED"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {test_name}: {status}")
        if not passed:
            all_passed = False

    print("="*60)

    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    exit(main())
