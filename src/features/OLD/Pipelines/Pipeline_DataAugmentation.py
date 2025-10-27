"""Module de pipelines d'augmentation de données pour la détection COVID-19.

Ce module implémente des pipelines complets d'augmentation de données utilisant 
TensorFlow/Keras et d'autres bibliothèques pour enrichir le dataset d'images 
médicales COVID-19. Il utilise les configurations JSON pour créer dynamiquement 
différents types de stratégies d'augmentation.

Auteur: L-Poca
Date: 2025
Compatible avec: TensorFlow, Keras, PIL, OpenCV, Albumentation
"""

import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array, load_img
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("⚠️ Albumentations non disponible - certaines augmentations avancées seront désactivées")


class DataAugmentationPipeline:
    """Pipeline complet d'augmentation de données pour images médicales COVID-19."""
    
    def __init__(self, config_path="Pipeline_DataAugmentation_config.json"):
        """
        Args:
            config_path (str): Chemin vers le fichier de configuration JSON
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.stats = {
            'original_counts': {},
            'augmented_counts': {},
            'processing_time': {},
            'generated_files': []
        }
        
    def _load_config(self):
        """Charge la configuration depuis le fichier JSON."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Fichier de configuration non trouvé: {self.config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Erreur dans le fichier JSON: {e}")
    
    def _create_keras_generator(self, aug_config: Dict) -> ImageDataGenerator:
        """Crée un générateur Keras ImageDataGenerator."""
        params = aug_config.get('params', {})
        
        # Paramètres de base Keras
        keras_params = {
            'rotation_range': params.get('rotation_range', 0),
            'width_shift_range': params.get('width_shift_range', 0.0),
            'height_shift_range': params.get('height_shift_range', 0.0),
            'shear_range': params.get('shear_range', 0.0),
            'zoom_range': params.get('zoom_range', 0.0),
            'horizontal_flip': params.get('horizontal_flip', False),
            'vertical_flip': params.get('vertical_flip', False),
            'fill_mode': params.get('fill_mode', 'nearest'),
            'brightness_range': params.get('brightness_range', None),
            'channel_shift_range': params.get('channel_shift_range', 0.0),
            'rescale': params.get('rescale', None)
        }
        
        # Filtrer les paramètres None
        keras_params = {k: v for k, v in keras_params.items() if v is not None}
        
        return ImageDataGenerator(**keras_params)
    
    def _create_albumentations_transform(self, aug_config: Dict):
        """Crée une transformation Albumentations."""
        if not ALBUMENTATIONS_AVAILABLE:
            raise ImportError("Albumentations n'est pas installé")
        
        params = aug_config.get('params', {})
        transforms = []
        
        # Transformations géométriques
        if params.get('rotation_range', 0) > 0:
            transforms.append(A.Rotate(
                limit=params['rotation_range'], 
                p=params.get('rotation_prob', 0.5)
            ))
        
        if params.get('horizontal_flip', False):
            transforms.append(A.HorizontalFlip(p=params.get('horizontal_flip_prob', 0.5)))
        
        if params.get('vertical_flip', False):
            transforms.append(A.VerticalFlip(p=params.get('vertical_flip_prob', 0.5)))
        
        # Transformations de couleur/intensité
        if params.get('brightness_contrast', False):
            transforms.append(A.RandomBrightnessContrast(
                brightness_limit=params.get('brightness_limit', 0.2),
                contrast_limit=params.get('contrast_limit', 0.2),
                p=params.get('brightness_contrast_prob', 0.5)
            ))
        
        # Transformations spécifiques aux images médicales
        if params.get('gaussian_noise', False):
            transforms.append(A.GaussNoise(
                var_limit=params.get('noise_var_limit', (10, 50)),
                p=params.get('noise_prob', 0.3)
            ))
        
        if params.get('gaussian_blur', False):
            transforms.append(A.GaussianBlur(
                blur_limit=params.get('blur_limit', (1, 3)),
                p=params.get('blur_prob', 0.3)
            ))
        
        # Transformations élastiques pour simuler des variations anatomiques
        if params.get('elastic_transform', False):
            transforms.append(A.ElasticTransform(
                alpha=params.get('elastic_alpha', 1),
                sigma=params.get('elastic_sigma', 50),
                alpha_affine=params.get('elastic_alpha_affine', 50),
                p=params.get('elastic_prob', 0.3)
            ))
        
        return A.Compose(transforms)
    
    def _apply_custom_medical_augmentation(self, image: np.ndarray, aug_config: Dict) -> np.ndarray:
        """Applique des augmentations personnalisées pour images médicales."""
        params = aug_config.get('params', {})
        augmented = image.copy()
        
        # Amélioration du contraste adaptatif (CLAHE)
        if params.get('clahe', False):
            if len(augmented.shape) == 3:
                # Convertir en niveaux de gris pour CLAHE
                gray = cv2.cvtColor(augmented, cv2.COLOR_RGB2GRAY)
                clahe = cv2.createCLAHE(
                    clipLimit=params.get('clahe_clip_limit', 2.0),
                    tileGridSize=params.get('clahe_grid_size', (8, 8))
                )
                augmented = clahe.apply(gray)
                if len(image.shape) == 3:
                    augmented = cv2.cvtColor(augmented, cv2.COLOR_GRAY2RGB)
        
        # Égalisation d'histogramme
        if params.get('histogram_equalization', False):
            if len(augmented.shape) == 3:
                # Pour les images couleur, égaliser chaque canal
                augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2YUV)
                augmented[:,:,0] = cv2.equalizeHist(augmented[:,:,0])
                augmented = cv2.cvtColor(augmented, cv2.COLOR_YUV2RGB)
            else:
                augmented = cv2.equalizeHist(augmented)
        
        # Filtrage pour réduire le bruit
        if params.get('denoise', False):
            if len(augmented.shape) == 3:
                augmented = cv2.fastNlMeansDenoisingColored(
                    augmented,
                    None,
                    params.get('denoise_h', 10),
                    params.get('denoise_h_color', 10),
                    params.get('denoise_template_window_size', 7),
                    params.get('denoise_search_window_size', 21)
                )
            else:
                augmented = cv2.fastNlMeansDenoising(
                    augmented,
                    None,
                    params.get('denoise_h', 10),
                    params.get('denoise_template_window_size', 7),
                    params.get('denoise_search_window_size', 21)
                )
        
        return augmented
    
    def process_class_directory(self, 
                              class_name: str, 
                              input_dir: str, 
                              output_dir: str,
                              aug_config_name: str,
                              target_count: Optional[int] = None,
                              preserve_originals: bool = True) -> Dict:
        """
        Traite un dossier de classe avec augmentation de données.
        
        Args:
            class_name (str): Nom de la classe (COVID, Normal, etc.)
            input_dir (str): Dossier d'entrée contenant les images originales
            output_dir (str): Dossier de sortie pour les images augmentées
            aug_config_name (str): Nom de la configuration d'augmentation
            target_count (int, optional): Nombre cible d'images (None = selon config)
            preserve_originals (bool): Conserver les images originales
        
        Returns:
            Dict: Statistiques du traitement
        """
        print(f"\n{'='*70}")
        print(f"🔄 AUGMENTATION DE DONNÉES - CLASSE: {class_name}")
        print(f"{'='*70}")
        
        start_time = datetime.now()
        
        # Vérifications
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Dossier d'entrée non trouvé: {input_dir}")
        
        if aug_config_name not in self.config['augmentation_strategies']:
            raise ValueError(f"Configuration d'augmentation non trouvée: {aug_config_name}")
        
        # Configuration
        aug_config = self.config['augmentation_strategies'][aug_config_name]
        
        # Créer le dossier de sortie
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Lister les images d'entrée
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        input_images = [f for f in os.listdir(input_dir) 
                       if f.lower().endswith(image_extensions)]
        
        original_count = len(input_images)
        print(f"📊 Images originales trouvées: {original_count}")
        
        if original_count == 0:
            print("❌ Aucune image trouvée dans le dossier d'entrée")
            return {'error': 'No images found'}
        
        # Déterminer le nombre cible
        if target_count is None:
            target_count = aug_config.get('target_count', original_count * 2)
        
        augmentations_needed = max(0, target_count - (original_count if preserve_originals else 0))
        augmentations_per_image = max(1, augmentations_needed // original_count)
        
        print(f"🎯 Objectif: {target_count} images")
        print(f"🔢 Augmentations par image: {augmentations_per_image}")
        print(f"📝 Stratégie: {aug_config['description']}")
        
        # Copier les originaux si demandé
        generated_files = []
        if preserve_originals:
            print(f"📋 Copie des images originales...")
            for img_file in input_images:
                src_path = os.path.join(input_dir, img_file)
                dst_path = os.path.join(output_dir, f"original_{img_file}")
                shutil.copy2(src_path, dst_path)
                generated_files.append(dst_path)
        
        # Préparer l'augmentation selon le type
        aug_type = aug_config.get('type', 'keras')
        
        if aug_type == 'keras':
            generator = self._create_keras_generator(aug_config)
        elif aug_type == 'albumentations' and ALBUMENTATIONS_AVAILABLE:
            transform = self._create_albumentations_transform(aug_config)
        elif aug_type == 'custom_medical':
            pass  # Utilise la fonction personnalisée
        else:
            print(f"⚠️ Type d'augmentation non supporté: {aug_type}, utilisation de Keras")
            generator = self._create_keras_generator(aug_config)
            aug_type = 'keras'
        
        # Appliquer les augmentations
        print(f"🚀 Génération des images augmentées...")
        progress_count = 0
        
        for i, img_file in enumerate(input_images):
            img_path = os.path.join(input_dir, img_file)
            base_name = os.path.splitext(img_file)[0]
            
            # Charger l'image
            try:
                if aug_type == 'keras':
                    img = load_img(img_path, target_size=aug_config.get('target_size'))
                    img_array = img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0)
                else:
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"⚠️ Impossible de charger: {img_file}")
                        continue
                    
                    target_size = aug_config.get('target_size')
                    if target_size:
                        img = cv2.resize(img, target_size)
                
            except Exception as e:
                print(f"⚠️ Erreur lors du chargement de {img_file}: {e}")
                continue
            
            # Générer les augmentations
            for aug_idx in range(augmentations_per_image):
                try:
                    if aug_type == 'keras':
                        # Utiliser le générateur Keras
                        aug_iter = generator.flow(img_array, batch_size=1)
                        augmented_batch = next(aug_iter)
                        augmented = augmented_batch[0].astype(np.uint8)
                        
                    elif aug_type == 'albumentations':
                        # Utiliser Albumentations
                        augmented = transform(image=img)['image']
                        
                    elif aug_type == 'custom_medical':
                        # Utiliser les augmentations médicales personnalisées
                        augmented = self._apply_custom_medical_augmentation(img, aug_config)
                    
                    # Sauvegarder l'image augmentée
                    output_filename = f"{base_name}_aug_{aug_idx+1:03d}.png"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    if len(augmented.shape) == 3:
                        cv2.imwrite(output_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
                    else:
                        cv2.imwrite(output_path, augmented)
                    
                    generated_files.append(output_path)
                    progress_count += 1
                    
                except Exception as e:
                    print(f"⚠️ Erreur lors de l'augmentation de {img_file} (aug {aug_idx+1}): {e}")
                    continue
            
            # Affichage du progrès
            if (i + 1) % 50 == 0 or i == len(input_images) - 1:
                print(f"📈 Progression: {i+1}/{len(input_images)} images traitées")
        
        # Statistiques finales
        processing_time = datetime.now() - start_time
        final_count = len([f for f in os.listdir(output_dir) 
                          if f.lower().endswith(image_extensions)])
        
        results = {
            'class_name': class_name,
            'augmentation_strategy': aug_config_name,
            'original_count': original_count,
            'target_count': target_count,
            'final_count': final_count,
            'generated_count': progress_count,
            'processing_time': str(processing_time),
            'input_directory': input_dir,
            'output_directory': output_dir,
            'generated_files': generated_files
        }
        
        # Mise à jour des statistiques globales
        self.stats['original_counts'][class_name] = original_count
        self.stats['augmented_counts'][class_name] = final_count
        self.stats['processing_time'][class_name] = processing_time
        self.stats['generated_files'].extend(generated_files)
        
        print(f"✅ Augmentation terminée pour {class_name}")
        print(f"📊 Images générées: {progress_count}")
        print(f"📊 Total final: {final_count}")
        print(f"⏱️ Temps de traitement: {processing_time}")
        
        return results
    
    def process_all_classes(self, 
                           input_base_dir: str,
                           output_base_dir: str,
                           aug_config_name: str,
                           class_mapping: Optional[Dict[str, str]] = None) -> List[Dict]:
        """
        Traite toutes les classes avec augmentation de données.
        
        Args:
            input_base_dir (str): Dossier de base d'entrée
            output_base_dir (str): Dossier de base de sortie  
            aug_config_name (str): Nom de la configuration d'augmentation
            class_mapping (dict, optional): Mapping des noms de classes
        
        Returns:
            List[Dict]: Résultats pour chaque classe
        """
        print(f"\n{'='*80}")
        print(f"🎯 TRAITEMENT COMPLET - AUGMENTATION DE DONNÉES")
        print(f"{'='*80}")
        
        # Mapping par défaut des classes
        if class_mapping is None:
            class_mapping = self.config['default_settings'].get('class_mapping', {
                'COVID': 'COVID',
                'Normal': 'Normal', 
                'Lung_Opacity': 'Lung_Opacity',
                'Viral Pneumonia': 'Viral Pneumonia'
            })
        
        results = []
        total_start_time = datetime.now()
        
        # Traiter chaque classe
        for class_key, class_dir in class_mapping.items():
            input_dir = os.path.join(input_base_dir, class_dir, 'images')
            output_dir = os.path.join(output_base_dir, class_key, 'augmented')
            
            if os.path.exists(input_dir):
                try:
                    class_results = self.process_class_directory(
                        class_name=class_key,
                        input_dir=input_dir,
                        output_dir=output_dir,
                        aug_config_name=aug_config_name
                    )
                    results.append(class_results)
                except Exception as e:
                    print(f"❌ Erreur pour la classe {class_key}: {e}")
                    results.append({
                        'class_name': class_key,
                        'error': str(e)
                    })
            else:
                print(f"⚠️ Dossier non trouvé pour {class_key}: {input_dir}")
        
        # Statistiques globales
        total_time = datetime.now() - total_start_time
        print(f"\n{'='*80}")
        print(f"📊 RÉSUMÉ GLOBAL")
        print(f"{'='*80}")
        print(f"⏱️ Temps total: {total_time}")
        
        # Générer un rapport de synthèse
        self._generate_summary_report(results, output_base_dir)
        
        return results
    
    def _generate_summary_report(self, results: List[Dict], output_dir: str):
        """Génère un rapport de synthèse des augmentations."""
        report_data = []
        
        for result in results:
            if 'error' not in result:
                report_data.append({
                    'Classe': result['class_name'],
                    'Images_Originales': result['original_count'],
                    'Images_Cibles': result['target_count'],
                    'Images_Finales': result['final_count'],
                    'Images_Générées': result['generated_count'],
                    'Temps_Traitement': result['processing_time'],
                    'Strategie': result['augmentation_strategy']
                })
        
        if report_data:
            df = pd.DataFrame(report_data)
            
            # Sauvegarder le rapport CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(output_dir, f"augmentation_report_{timestamp}.csv")
            df.to_csv(report_path, index=False)
            
            print(f"📄 Rapport sauvegardé: {report_path}")
            print("\n📊 TABLEAU DE SYNTHÈSE:")
            print(df.to_string(index=False))
            
            # Visualisation simple
            if len(report_data) > 1:
                self._create_augmentation_charts(df, output_dir, timestamp)
    
    def _create_augmentation_charts(self, df: pd.DataFrame, output_dir: str, timestamp: str):
        """Crée des graphiques de synthèse des augmentations."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Rapport d\'Augmentation de Données COVID-19', fontsize=16)
            
            # Graphique 1: Comparaison avant/après
            axes[0, 0].bar(df['Classe'], df['Images_Originales'], alpha=0.7, label='Originales')
            axes[0, 0].bar(df['Classe'], df['Images_Finales'], alpha=0.7, label='Après augmentation')
            axes[0, 0].set_title('Comparaison Avant/Après Augmentation')
            axes[0, 0].set_ylabel('Nombre d\'images')
            axes[0, 0].legend()
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Graphique 2: Images générées par classe
            axes[0, 1].bar(df['Classe'], df['Images_Générées'])
            axes[0, 1].set_title('Images Générées par Classe')
            axes[0, 1].set_ylabel('Images générées')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Graphique 3: Distribution finale
            axes[1, 0].pie(df['Images_Finales'], labels=df['Classe'], autopct='%1.1f%%')
            axes[1, 0].set_title('Distribution Finale des Classes')
            
            # Graphique 4: Temps de traitement
            axes[1, 1].barh(df['Classe'], [float(t.split(':')[1]) for t in df['Temps_Traitement']])
            axes[1, 1].set_title('Temps de Traitement par Classe (minutes)')
            axes[1, 1].set_xlabel('Minutes')
            
            plt.tight_layout()
            
            # Sauvegarder
            chart_path = os.path.join(output_dir, f"augmentation_charts_{timestamp}.png")
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📈 Graphiques sauvegardés: {chart_path}")
            
        except Exception as e:
            print(f"⚠️ Erreur lors de la création des graphiques: {e}")
    
    def preview_augmentations(self, 
                            image_path: str, 
                            aug_config_name: str, 
                            num_samples: int = 6) -> None:
        """
        Affiche un aperçu des augmentations pour une image donnée.
        
        Args:
            image_path (str): Chemin vers l'image de test
            aug_config_name (str): Configuration d'augmentation à tester
            num_samples (int): Nombre d'échantillons à générer
        """
        print(f"🔍 APERÇU DES AUGMENTATIONS - {aug_config_name}")
        
        if not os.path.exists(image_path):
            print(f"❌ Image non trouvée: {image_path}")
            return
        
        if aug_config_name not in self.config['augmentation_strategies']:
            print(f"❌ Configuration non trouvée: {aug_config_name}")
            return
        
        aug_config = self.config['augmentation_strategies'][aug_config_name]
        
        # Charger l'image originale
        original_img = cv2.imread(image_path)
        if original_img is None:
            print(f"❌ Impossible de charger l'image: {image_path}")
            return
        
        # Redimensionner si nécessaire
        target_size = aug_config.get('target_size')
        if target_size:
            original_img = cv2.resize(original_img, target_size)
        
        # Créer la figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Aperçu Augmentations - {aug_config_name}', fontsize=16)
        
        # Image originale
        axes[0, 0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Image Originale')
        axes[0, 0].axis('off')
        
        # Générer les augmentations
        aug_type = aug_config.get('type', 'keras')
        
        try:
            if aug_type == 'keras':
                generator = self._create_keras_generator(aug_config)
                img_array = np.expand_dims(original_img, axis=0)
                
                for i in range(num_samples - 1):
                    row, col = divmod(i + 1, 3)
                    aug_iter = generator.flow(img_array, batch_size=1)
                    augmented_batch = next(aug_iter)
                    augmented = augmented_batch[0].astype(np.uint8)
                    
                    axes[row, col].imshow(augmented)
                    axes[row, col].set_title(f'Augmentation {i+1}')
                    axes[row, col].axis('off')
                    
            elif aug_type == 'albumentations' and ALBUMENTATIONS_AVAILABLE:
                transform = self._create_albumentations_transform(aug_config)
                
                for i in range(num_samples - 1):
                    row, col = divmod(i + 1, 3)
                    augmented = transform(image=original_img)['image']
                    
                    axes[row, col].imshow(cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB))
                    axes[row, col].set_title(f'Augmentation {i+1}')
                    axes[row, col].axis('off')
                    
            elif aug_type == 'custom_medical':
                for i in range(num_samples - 1):
                    row, col = divmod(i + 1, 3)
                    augmented = self._apply_custom_medical_augmentation(original_img, aug_config)
                    
                    axes[row, col].imshow(cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB))
                    axes[row, col].set_title(f'Augmentation {i+1}')
                    axes[row, col].axis('off')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"❌ Erreur lors de la génération de l'aperçu: {e}")
    
    def get_available_strategies(self) -> List[Dict]:
        """Retourne la liste des stratégies d'augmentation disponibles."""
        strategies = []
        for name, config in self.config['augmentation_strategies'].items():
            strategies.append({
                'name': name,
                'description': config['description'],
                'type': config.get('type', 'keras'),
                'target_count': config.get('target_count', 'Auto')
            })
        return strategies
    
    def validate_configuration(self) -> bool:
        """Valide la configuration chargée."""
        required_sections = ['augmentation_strategies', 'default_settings']
        
        for section in required_sections:
            if section not in self.config:
                print(f"❌ Section manquante dans la configuration: {section}")
                return False
        
        print("✅ Configuration validée avec succès")
        return True


# Fonctions utilitaires
def balance_dataset_with_augmentation(input_dirs: Dict[str, str], 
                                    output_base_dir: str,
                                    target_count: int,
                                    aug_config_name: str = 'balanced_medical') -> Dict:
    """
    Équilibre un dataset en augmentant les classes sous-représentées.
    
    Args:
        input_dirs (dict): Dictionnaire {classe: dossier_images}
        output_base_dir (str): Dossier de base de sortie
        target_count (int): Nombre cible d'images par classe
        aug_config_name (str): Configuration d'augmentation à utiliser
        
    Returns:
        Dict: Statistiques du rééquilibrage
    """
    pipeline = DataAugmentationPipeline()
    
    # Analyser les comptes actuels
    current_counts = {}
    for class_name, input_dir in input_dirs.items():
        if os.path.exists(input_dir):
            images = [f for f in os.listdir(input_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            current_counts[class_name] = len(images)
        else:
            current_counts[class_name] = 0
    
    print(f"📊 Comptes actuels: {current_counts}")
    print(f"🎯 Objectif: {target_count} images par classe")
    
    # Traiter chaque classe
    results = {}
    for class_name, input_dir in input_dirs.items():
        output_dir = os.path.join(output_base_dir, class_name)
        
        if current_counts[class_name] < target_count:
            print(f"\n🔄 Augmentation de {class_name}: {current_counts[class_name]} → {target_count}")
            
            class_result = pipeline.process_class_directory(
                class_name=class_name,
                input_dir=input_dir,
                output_dir=output_dir,
                aug_config_name=aug_config_name,
                target_count=target_count
            )
            results[class_name] = class_result
        else:
            print(f"✅ {class_name} déjà équilibrée ({current_counts[class_name]} images)")
    
    return results


# Exemple d'utilisation
if __name__ == "__main__":
    print("🧪 Test du pipeline d'augmentation de données")
    
    # Configuration de test
    config_path = "Pipeline_DataAugmentation_config.json"
    
    # Créer le pipeline
    pipeline = DataAugmentationPipeline(config_path)
    
    # Valider la configuration
    if not pipeline.validate_configuration():
        exit(1)
    
    # Afficher les stratégies disponibles
    print("\n📋 Stratégies d'augmentation disponibles:")
    strategies = pipeline.get_available_strategies()
    for strategy in strategies[:3]:  # Afficher les 3 premières
        print(f"- {strategy['name']}: {strategy['description']}")
        print(f"  Type: {strategy['type']}, Cible: {strategy['target_count']}")
    
    # Exemple de traitement (avec des chemins de test)
    # Décommenter pour tester avec vos données
    """
    base_input = "data/raw/COVID-19_Radiography_Dataset/COVID-19_Radiography_Dataset"
    base_output = "data/processed/augmented"
    
    # Traiter toutes les classes
    results = pipeline.process_all_classes(
        input_base_dir=base_input,
        output_base_dir=base_output,
        aug_config_name='medical_basic'
    )
    
    print("✅ Augmentation terminée!")
    """