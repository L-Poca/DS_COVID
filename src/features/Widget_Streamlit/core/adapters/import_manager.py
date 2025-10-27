"""
Gestionnaire d'imports robuste pour les modules legacy.
Fournit des fallbacks sécurisés en cas d'échec d'importation.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Any, Type

logger = logging.getLogger(__name__)


class ImportManager:
    """Gestionnaire centralisé des imports legacy avec fallbacks."""
    
    def __init__(self):
        self._sklearn_pipeline_manager = None
        self._tensorflow_pipeline_manager = None  
        self._covid_data_loader = None
        self._setup_paths()
        
    def _setup_paths(self):
        """Configure les chemins d'import pour les modules legacy."""
        try:
            # Calcul des chemins depuis le répertoire actuel
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent.parent.parent
            
            # Ajout des chemins nécessaires
            paths_to_add = [
                str(project_root),
                str(project_root / 'src'),
                str(project_root / 'src' / 'features'),
                str(project_root / 'src' / 'features' / 'Pipelines'),
                str(project_root / 'src' / 'features' / 'Data_Loaders')
            ]
            
            for path in paths_to_add:
                if path not in sys.path:
                    sys.path.insert(0, path)
                    
            logger.info(f"Chemins d'import configurés: {len(paths_to_add)} chemins ajoutés")
            
        except Exception as e:
            logger.error(f"Erreur lors de la configuration des chemins: {e}")
    
    def get_sklearn_pipeline_manager(self) -> Optional[Type]:
        """
        Import sécurisé du SklearnPipelineManager.
        
        Returns:
            Classe SklearnPipelineManager ou None si indisponible
        """
        if self._sklearn_pipeline_manager is not None:
            return self._sklearn_pipeline_manager
            
        try:
            # Tentative d'import principal
            from src.features.Pipelines.Pipeline_Sklearn import PipelineManager
            self._sklearn_pipeline_manager = PipelineManager
            logger.info("✅ SklearnPipelineManager importé avec succès (chemin principal)")
            return self._sklearn_pipeline_manager
            
        except ImportError:
            try:
                # Fallback import
                from Pipeline_Sklearn import PipelineManager
                self._sklearn_pipeline_manager = PipelineManager
                logger.info("✅ SklearnPipelineManager importé avec succès (fallback)")
                return self._sklearn_pipeline_manager
                
            except ImportError as e:
                logger.warning(f"⚠️ SklearnPipelineManager non disponible: {e}")
                self._sklearn_pipeline_manager = MockPipelineManager  # Fallback mock
                return self._sklearn_pipeline_manager
    
    def get_tensorflow_pipeline_manager(self) -> Optional[Type]:
        """
        Import sécurisé du TensorFlowPipelineManager.
        
        Returns:
            Classe TensorFlowPipelineManager ou None si indisponible
        """
        if self._tensorflow_pipeline_manager is not None:
            return self._tensorflow_pipeline_manager
            
        try:
            # Tentative d'import principal
            from src.features.Pipelines.Pipeline_TensorFlow import TensorFlowPipelineManager
            self._tensorflow_pipeline_manager = TensorFlowPipelineManager
            logger.info("✅ TensorFlowPipelineManager importé avec succès (chemin principal)")
            return self._tensorflow_pipeline_manager
            
        except ImportError:
            try:
                # Fallback import
                from Pipeline_TensorFlow import TensorFlowPipelineManager
                self._tensorflow_pipeline_manager = TensorFlowPipelineManager
                logger.info("✅ TensorFlowPipelineManager importé avec succès (fallback)")
                return self._tensorflow_pipeline_manager
                
            except ImportError as e:
                logger.warning(f"⚠️ TensorFlowPipelineManager non disponible: {e}")
                self._tensorflow_pipeline_manager = MockPipelineManager  # Fallback mock
                return self._tensorflow_pipeline_manager
    
    def get_covid_data_loader(self) -> Optional[Any]:
        """
        Import sécurisé du covid_data_loader.
        
        Returns:
            Module covid_data_loader ou None si indisponible
        """
        if self._covid_data_loader is not None:
            return self._covid_data_loader
            
        try:
            # Tentative d'import principal
            from src.features.Data_Loaders import covid_data_loader
            self._covid_data_loader = covid_data_loader
            logger.info("✅ covid_data_loader importé avec succès (chemin principal)")
            return self._covid_data_loader
            
        except ImportError:
            try:
                # Fallback import
                import covid_data_loader
                self._covid_data_loader = covid_data_loader
                logger.info("✅ covid_data_loader importé avec succès (fallback)")
                return self._covid_data_loader
                
            except ImportError as e:
                logger.warning(f"⚠️ covid_data_loader non disponible: {e}")
                self._covid_data_loader = MockDataLoader()  # Fallback mock
                return self._covid_data_loader
    
    def check_availability(self) -> dict:
        """
        Vérifie la disponibilité de tous les modules.
        
        Returns:
            Dict avec le statut de chaque module
        """
        return {
            'sklearn_pipeline': self.get_sklearn_pipeline_manager() is not None,
            'tensorflow_pipeline': self.get_tensorflow_pipeline_manager() is not None,
            'covid_data_loader': self.get_covid_data_loader() is not None
        }


class MockPipelineManager:
    """Mock pipeline manager pour les cas où les vrais modules ne sont pas disponibles."""
    
    def __init__(self, *args, **kwargs):
        self._is_mock = True
        logger.warning("🎭 MockPipelineManager initialisé - fonctionnalités limitées")
    
    def train_pipeline(self, *args, **kwargs):
        return {
            'success': False,
            'error': 'Pipeline manager non disponible (mode mock)',
            'is_mock': True
        }
    
    def predict(self, *args, **kwargs):
        return {
            'success': False, 
            'error': 'Pipeline manager non disponible (mode mock)',
            'is_mock': True
        }
    
    def evaluate_pipeline(self, *args, **kwargs):
        return {
            'success': False,
            'error': 'Pipeline manager non disponible (mode mock)', 
            'is_mock': True
        }
    
    def get_available_pipelines(self):
        return []


class MockDataLoader:
    """Mock data loader pour les cas où le vrai module n'est pas disponible."""
    
    def __init__(self):
        self._is_mock = True
        logger.warning("🎭 MockDataLoader initialisé - fonctionnalités limitées")
    
    def load_data(self, *args, **kwargs):
        return None, None, {'error': 'Data loader non disponible (mode mock)', 'is_mock': True}
    
    def preprocess_data(self, *args, **kwargs):
        return None, {'error': 'Data loader non disponible (mode mock)', 'is_mock': True}


# Instance globale du gestionnaire d'imports
import_manager = ImportManager()