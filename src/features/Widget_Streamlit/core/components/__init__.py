"""
Module des composants UI réutilisables pour Streamlit.
Composants atomiques focalisés sur des responsabilités spécifiques.
"""

from .forms import ConfigurationForm, FileUploadForm, ProgressTracker

# Export des composants principaux
__all__ = [
    'ConfigurationForm',
    'FileUploadForm',
    'ProgressTracker'
]

# Documentation des composants UI
COMPONENT_DOCUMENTATION = {
    'ConfigurationForm': {
        'description': 'Formulaires de configuration réutilisables',
        'methods': [
            'render_data_config_form() - Configuration des données',
            'render_training_config_form() - Configuration d\'entraînement',
            'render_data_config_inline() - Version compacte'
        ],
        'features': [
            'Validation intégrée des formulaires',
            'Persistance d\'état via StateManager',
            'Interface cohérente et intuitive',
            'Support des configurations par défaut'
        ],
        'use_cases': [
            'Paramétrage des pipelines ML',
            'Configuration des datasets',
            'Saisie des hyperparamètres'
        ]
    },
    
    'FileUploadForm': {
        'description': 'Composant de téléchargement de fichiers avec validation',
        'methods': [
            'render_single_file_upload() - Upload d\'un fichier',
            'render_multiple_files_upload() - Upload multiple'
        ],
        'features': [
            'Validation des extensions de fichiers',
            'Contrôle de taille des fichiers',
            'Messages d\'erreur informatifs',
            'Support upload simple ou multiple'
        ],
        'use_cases': [
            'Upload d\'images pour prédiction',
            'Import de datasets personnalisés',
            'Chargement de modèles pré-entraînés'
        ]
    },
    
    'ProgressTracker': {
        'description': 'Composant d\'affichage de progression pour tâches longues',
        'methods': [
            'update() - Mise à jour progression',
            'complete() - Fin de tâche',
            'error() - Affichage d\'erreur'
        ],
        'features': [
            'Barre de progression visuelle',
            'Messages de statut dynamiques',
            'Gestion des états d\'erreur',
            'Interface claire et informative'
        ],
        'use_cases': [
            'Progression d\'entraînement de modèles',
            'Chargement de datasets volumineux',
            'Traitement batch d\'images'
        ]
    }
}


def get_component_info(component_name: str = None) -> dict:
    """
    Retourne la documentation d'un composant ou de tous les composants.
    
    Args:
        component_name: Nom du composant (optionnel)
        
    Returns:
        dict: Documentation du/des composant(s)
    """
    if component_name:
        return COMPONENT_DOCUMENTATION.get(component_name, {})
    return COMPONENT_DOCUMENTATION


# Configuration par défaut des composants
DEFAULT_COMPONENT_CONFIG = {
    'forms': {
        'auto_save': True,
        'validation_enabled': True,
        'show_help_text': True,
        'compact_mode': False
    },
    'file_upload': {
        'max_file_size_mb': 10,
        'allowed_image_extensions': ['.png', '.jpg', '.jpeg'],
        'allowed_model_extensions': ['.pkl', '.h5', '.pb'],
        'show_file_details': True
    },
    'progress': {
        'show_percentage': True,
        'show_eta': False,
        'update_frequency_ms': 100,
        'smooth_animation': True
    }
}


# Factory pour créer des composants configurés
class ComponentFactory:
    """Factory pour créer des composants UI avec configuration."""
    
    @staticmethod
    def create_configuration_form(state_manager, form_id: str = "default") -> ConfigurationForm:
        """Crée un formulaire de configuration."""
        return ConfigurationForm(state_manager, form_id)
    
    @staticmethod
    def create_file_upload(file_types: list = None, max_size_mb: int = 10) -> FileUploadForm:
        """Crée un composant de téléchargement de fichiers."""
        return FileUploadForm(file_types, max_size_mb)
    
    @staticmethod
    def create_progress_tracker(total_steps: int, description: str = "Progression") -> ProgressTracker:
        """Crée un tracker de progression."""
        return ProgressTracker(total_steps, description)


# Utilitaires pour les composants
class ComponentUtils:
    """Utilitaires partagés pour les composants UI."""
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Formate une taille de fichier en unités lisibles."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"
    
    @staticmethod
    def validate_json_string(json_str: str) -> tuple:
        """
        Valide une chaîne JSON.
        
        Returns:
            tuple: (is_valid, parsed_data_or_error_message)
        """
        try:
            import json
            data = json.loads(json_str)
            return True, data
        except json.JSONDecodeError as e:
            return False, str(e)
    
    @staticmethod
    def generate_color_palette(n_colors: int) -> list:
        """Génère une palette de couleurs pour les graphiques."""
        import colorsys
        
        colors = []
        for i in range(n_colors):
            hue = i / n_colors
            rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
            hex_color = '#%02x%02x%02x' % tuple(int(c * 255) for c in rgb)
            colors.append(hex_color)
        
        return colors
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 50) -> str:
        """Tronque un texte avec ellipse."""
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + "..."