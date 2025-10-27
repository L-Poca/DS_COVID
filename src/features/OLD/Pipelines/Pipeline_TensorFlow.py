"""Module de pipelines TensorFlow/Keras pour la détection COVID-19.

Ce module implémente des pipelines complets utilisant TensorFlow et Keras pour 
l'entraînement de modèles de deep learning sur des images médicales COVID-19.
Il supporte différentes architectures (CNN custom, Transfer Learning, ensembles)
et utilise les configurations JSON pour une création dynamique.

Auteur: L-Poca
Date: 2025
Compatible avec: TensorFlow 2.x, Keras, scikit-learn
"""

import json
import os
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                           confusion_matrix, f1_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import (VGG16, VGG19, ResNet50, ResNet101,
                                          InceptionV3, DenseNet121, EfficientNetB0,
                                          MobileNetV2, Xception)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# Supprimer les warnings TensorFlow moins importants
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class TensorFlowPipelineManager:
    """Gestionnaire de pipelines TensorFlow/Keras pour COVID-19."""
    
    def __init__(self, config_path="Pipeline_TensorFlow_config.json"):
        """
        Args:
            config_path (str): Chemin vers le fichier de configuration JSON
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.models = {}
        self.histories = {}
        self.results = {}
        self.data_generators = {}
        
        # Configuration GPU si disponible
        self._setup_gpu()
        
    def _load_config(self):
        """Charge la configuration depuis le fichier JSON."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Fichier de configuration non trouvé: {self.config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Erreur dans le fichier JSON: {e}")
    
    def _setup_gpu(self):
        """Configure l'utilisation du GPU si disponible."""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Activer la croissance mémoire dynamique
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"🚀 GPU détecté et configuré: {len(gpus)} GPU(s)")
            except RuntimeError as e:
                print(f"⚠️ Erreur configuration GPU: {e}")
        else:
            print("💻 Utilisation du CPU")
    
    def _create_custom_cnn(self, model_config: Dict) -> tf.keras.Model:
        """Crée un CNN personnalisé."""
        params = model_config['architecture_params']
        input_shape = tuple(params['input_shape'])
        num_classes = params['num_classes']
        
        model = models.Sequential(name=model_config['name'])
        
        # Couches d'entrée
        model.add(layers.Input(shape=input_shape))
        
        # Normalisation/Preprocessing
        if params.get('rescaling', True):
            model.add(layers.Rescaling(1./255))
        
        if params.get('data_augmentation', False):
            # Augmentation intégrée au modèle
            model.add(layers.RandomFlip("horizontal"))
            model.add(layers.RandomRotation(0.1))
            model.add(layers.RandomZoom(0.1))
        
        # Blocs convolutionnels
        conv_blocks = params.get('conv_blocks', [
            {'filters': 32, 'kernel_size': 3, 'pool_size': 2},
            {'filters': 64, 'kernel_size': 3, 'pool_size': 2},
            {'filters': 128, 'kernel_size': 3, 'pool_size': 2}
        ])
        
        for i, block in enumerate(conv_blocks):
            # Convolution
            model.add(layers.Conv2D(
                filters=block['filters'],
                kernel_size=block['kernel_size'],
                activation=params.get('activation', 'relu'),
                padding='same',
                name=f'conv2d_{i+1}'
            ))
            
            # Batch Normalization
            if params.get('batch_normalization', True):
                model.add(layers.BatchNormalization())
            
            # Pooling
            if block.get('pool_size'):
                model.add(layers.MaxPooling2D(
                    pool_size=block['pool_size'],
                    name=f'maxpool_{i+1}'
                ))
            
            # Dropout
            if params.get('conv_dropout', 0) > 0:
                model.add(layers.Dropout(params['conv_dropout']))
        
        # Aplatissement
        model.add(layers.GlobalAveragePooling2D())
        
        # Couches denses
        dense_layers = params.get('dense_layers', [512, 256])
        for i, units in enumerate(dense_layers):
            model.add(layers.Dense(
                units,
                activation=params.get('activation', 'relu'),
                name=f'dense_{i+1}'
            ))
            
            if params.get('dense_dropout', 0) > 0:
                model.add(layers.Dropout(params['dense_dropout']))
        
        # Couche de sortie
        if num_classes == 2:
            model.add(layers.Dense(1, activation='sigmoid', name='output'))
        else:
            model.add(layers.Dense(num_classes, activation='softmax', name='output'))
        
        return model
    
    def _create_transfer_learning_model(self, model_config: Dict) -> tf.keras.Model:
        """Crée un modèle de transfer learning."""
        params = model_config['architecture_params']
        base_model_name = params['base_model']
        input_shape = tuple(params['input_shape'])
        num_classes = params['num_classes']
        
        # Mapping des modèles pré-entraînés
        base_models = {
            'VGG16': VGG16,
            'VGG19': VGG19,
            'ResNet50': ResNet50,
            'ResNet101': ResNet101,
            'InceptionV3': InceptionV3,
            'DenseNet121': DenseNet121,
            'EfficientNetB0': EfficientNetB0,
            'MobileNetV2': MobileNetV2,
            'Xception': Xception
        }
        
        if base_model_name not in base_models:
            raise ValueError(f"Modèle de base non supporté: {base_model_name}")
        
        # Créer le modèle de base
        base_model = base_models[base_model_name](
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        # Geler/Dégeler les couches
        base_model.trainable = params.get('fine_tuning', False)
        
        if params.get('fine_tuning', False):
            # Fine-tuning: geler les premières couches
            freeze_layers = params.get('freeze_layers', len(base_model.layers) // 2)
            for layer in base_model.layers[:freeze_layers]:
                layer.trainable = False
        
        # Construire le modèle complet
        inputs = keras.Input(shape=input_shape)
        
        # Preprocessing
        if params.get('rescaling', True):
            x = layers.Rescaling(1./255)(inputs)
        else:
            x = inputs
        
        # Augmentation de données
        if params.get('data_augmentation', False):
            x = layers.RandomFlip("horizontal")(x)
            x = layers.RandomRotation(0.1)(x)
            x = layers.RandomZoom(0.1)(x)
        
        # Modèle de base
        x = base_model(x, training=False)
        
        # Pooling global
        pooling_type = params.get('global_pooling', 'avg')
        if pooling_type == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        else:
            x = layers.GlobalMaxPooling2D()(x)
        
        # Régularisation
        if params.get('dropout', 0) > 0:
            x = layers.Dropout(params['dropout'])(x)
        
        # Couches personnalisées
        dense_layers = params.get('dense_layers', [128])
        for i, units in enumerate(dense_layers):
            x = layers.Dense(units, activation='relu', name=f'custom_dense_{i+1}')(x)
            if params.get('dense_dropout', 0) > 0:
                x = layers.Dropout(params['dense_dropout'])(x)
        
        # Couche de sortie
        if num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid', name='predictions')(x)
        else:
            outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
        
        model = keras.Model(inputs, outputs, name=f"{base_model_name}_transfer")
        
        return model
    
    def _create_ensemble_model(self, model_config: Dict) -> tf.keras.Model:
        """Crée un modèle d'ensemble."""
        params = model_config['architecture_params']
        input_shape = tuple(params['input_shape'])
        num_classes = params['num_classes']
        
        # Créer les modèles de base
        base_models = []
        for i, base_config in enumerate(params['base_models']):
            if base_config['type'] == 'cnn':
                base_model = self._create_custom_cnn({'architecture_params': base_config})
            elif base_config['type'] == 'transfer':
                base_model = self._create_transfer_learning_model({'architecture_params': base_config})
            else:
                raise ValueError(f"Type de modèle non supporté: {base_config['type']}")
            
            # Renommer le modèle
            base_model._name = f"base_model_{i+1}"
            base_models.append(base_model)
        
        # Créer l'architecture d'ensemble
        inputs = keras.Input(shape=input_shape)
        
        # Obtenir les prédictions de chaque modèle
        model_outputs = []
        for base_model in base_models:
            output = base_model(inputs)
            model_outputs.append(output)
        
        # Stratégie d'ensemble
        ensemble_strategy = params.get('ensemble_strategy', 'average')
        
        if ensemble_strategy == 'average':
            # Moyenne des prédictions
            ensemble_output = layers.Average()(model_outputs)
        elif ensemble_strategy == 'weighted_average':
            # Moyenne pondérée (poids appris)
            weights = params.get('ensemble_weights', [1.0] * len(base_models))
            weighted_outputs = [layers.Lambda(lambda x: x * w)(output) 
                              for output, w in zip(model_outputs, weights)]
            ensemble_output = layers.Add()(weighted_outputs)
        elif ensemble_strategy == 'concatenate':
            # Concaténation + couches denses
            concatenated = layers.Concatenate()(model_outputs)
            x = layers.Dense(128, activation='relu')(concatenated)
            x = layers.Dropout(0.5)(x)
            if num_classes == 2:
                ensemble_output = layers.Dense(1, activation='sigmoid')(x)
            else:
                ensemble_output = layers.Dense(num_classes, activation='softmax')(x)
        else:
            raise ValueError(f"Stratégie d'ensemble non supportée: {ensemble_strategy}")
        
        model = keras.Model(inputs=inputs, outputs=ensemble_output, name="ensemble_model")
        
        return model
    
    def create_model(self, config_name: str) -> tf.keras.Model:
        """Crée un modèle à partir de la configuration."""
        if config_name not in self.config['model_configs']:
            raise ValueError(f"Configuration non trouvée: {config_name}")
        
        model_config = self.config['model_configs'][config_name]
        architecture_type = model_config['architecture_type']
        
        print(f"🏗️ Création du modèle: {config_name}")
        print(f"📐 Architecture: {architecture_type}")
        
        if architecture_type == 'custom_cnn':
            model = self._create_custom_cnn(model_config)
        elif architecture_type == 'transfer_learning':
            model = self._create_transfer_learning_model(model_config)
        elif architecture_type == 'ensemble':
            model = self._create_ensemble_model(model_config)
        else:
            raise ValueError(f"Type d'architecture non supporté: {architecture_type}")
        
        # Compilation du modèle
        compile_config = model_config.get('compile_config', {})
        
        # Optimiseur
        optimizer_name = compile_config.get('optimizer', 'adam')
        learning_rate = compile_config.get('learning_rate', 0.001)
        
        optimizers_map = {
            'adam': optimizers.Adam(learning_rate=learning_rate),
            'sgd': optimizers.SGD(learning_rate=learning_rate, momentum=0.9),
            'rmsprop': optimizers.RMSprop(learning_rate=learning_rate),
            'adamax': optimizers.Adamax(learning_rate=learning_rate)
        }
        
        optimizer = optimizers_map.get(optimizer_name, optimizers.Adam(learning_rate=learning_rate))
        
        # Fonction de perte
        num_classes = model_config['architecture_params']['num_classes']
        if num_classes == 2:
            loss = 'binary_crossentropy'
            metrics = ['accuracy', 'precision', 'recall']
        else:
            loss = compile_config.get('loss', 'categorical_crossentropy')
            metrics = ['accuracy', 'top_2_accuracy']
        
        # Compilation
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        # Sauvegarde du modèle
        self.models[config_name] = model
        
        # Affichage du résumé
        print(f"📊 Résumé du modèle {config_name}:")
        model.summary()
        
        return model
    
    def setup_data_generators(self, 
                            train_dir: str,
                            validation_dir: str = None,
                            test_dir: str = None,
                            config_name: str = 'default') -> Dict:
        """Configure les générateurs de données."""
        print(f"\n📁 Configuration des générateurs de données")
        
        model_config = self.config['model_configs'].get(config_name, {})
        data_config = model_config.get('data_config', self.config['default_data_settings'])
        
        # Paramètres des générateurs
        target_size = tuple(data_config['target_size'])
        batch_size = data_config['batch_size']
        class_mode = data_config.get('class_mode', 'categorical')
        color_mode = data_config.get('color_mode', 'rgb')
        
        # Générateur d'entraînement avec augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=data_config.get('rotation_range', 20),
            width_shift_range=data_config.get('width_shift_range', 0.2),
            height_shift_range=data_config.get('height_shift_range', 0.2),
            shear_range=data_config.get('shear_range', 0.2),
            zoom_range=data_config.get('zoom_range', 0.2),
            horizontal_flip=data_config.get('horizontal_flip', True),
            fill_mode=data_config.get('fill_mode', 'nearest'),
            validation_split=data_config.get('validation_split', 0.2) if validation_dir is None else 0.0
        )
        
        # Générateur de validation/test (sans augmentation)
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        generators = {}
        
        # Générateur d'entraînement
        if validation_dir is None:
            # Utiliser validation_split
            train_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=target_size,
                batch_size=batch_size,
                class_mode=class_mode,
                color_mode=color_mode,
                subset='training',
                shuffle=True
            )
            
            val_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=target_size,
                batch_size=batch_size,
                class_mode=class_mode,
                color_mode=color_mode,
                subset='validation',
                shuffle=False
            )
        else:
            # Dossiers séparés
            train_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=target_size,
                batch_size=batch_size,
                class_mode=class_mode,
                color_mode=color_mode,
                shuffle=True
            )
            
            val_generator = val_test_datagen.flow_from_directory(
                validation_dir,
                target_size=target_size,
                batch_size=batch_size,
                class_mode=class_mode,
                color_mode=color_mode,
                shuffle=False
            )
        
        generators['train'] = train_generator
        generators['validation'] = val_generator
        
        # Générateur de test si disponible
        if test_dir and os.path.exists(test_dir):
            test_generator = val_test_datagen.flow_from_directory(
                test_dir,
                target_size=target_size,
                batch_size=batch_size,
                class_mode=class_mode,
                color_mode=color_mode,
                shuffle=False
            )
            generators['test'] = test_generator
        
        # Informations sur les classes
        class_indices = train_generator.class_indices
        num_classes = len(class_indices)
        
        print(f"📊 Classes détectées: {num_classes}")
        print(f"📋 Mapping des classes: {class_indices}")
        print(f"🔢 Échantillons d'entraînement: {train_generator.samples}")
        print(f"🔢 Échantillons de validation: {val_generator.samples}")
        
        # Calcul des poids de classe pour déséquilibre
        if data_config.get('compute_class_weights', True):
            class_weights = self._compute_class_weights(train_generator)
            generators['class_weights'] = class_weights
            print(f"⚖️ Poids des classes: {class_weights}")
        
        # Sauvegarde
        self.data_generators[config_name] = generators
        
        return generators
    
    def _compute_class_weights(self, generator) -> Dict[int, float]:
        """Calcule les poids des classes pour gérer le déséquilibre."""
        # Obtenir les labels de toutes les données
        labels = []
        for i in range(len(generator)):
            batch_x, batch_y = generator[i]
            if len(batch_y.shape) > 1:
                # Categorical
                batch_labels = np.argmax(batch_y, axis=1)
            else:
                # Binary
                batch_labels = batch_y.astype(int)
            labels.extend(batch_labels)
        
        # Calculer les poids
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(labels),
            y=labels
        )
        
        return {i: weight for i, weight in enumerate(class_weights)}
    
    def train_model(self,
                   config_name: str,
                   generators: Dict,
                   epochs: int = None,
                   callbacks_list: List = None) -> Dict:
        """Entraîne un modèle."""
        print(f"\n{'='*70}")
        print(f"🚀 ENTRAÎNEMENT DU MODÈLE: {config_name}")
        print(f"{'='*70}")
        
        # Vérifications
        if config_name not in self.models:
            self.create_model(config_name)
        
        model = self.models[config_name]
        model_config = self.config['model_configs'][config_name]
        
        # Paramètres d'entraînement
        training_config = model_config.get('training_config', {})
        if epochs is None:
            epochs = training_config.get('epochs', 50)
        
        print(f"📝 Description: {model_config['description']}")
        print(f"🔄 Époques: {epochs}")
        
        # Callbacks par défaut si non fournis
        if callbacks_list is None:
            callbacks_list = self._create_default_callbacks(config_name, training_config)
        
        print(f"🔧 Callbacks: {[type(cb).__name__ for cb in callbacks_list]}")
        
        # Générateurs
        train_gen = generators['train']
        val_gen = generators['validation']
        class_weights = generators.get('class_weights')
        
        # Entraînement
        start_time = datetime.now()
        
        try:
            history = model.fit(
                train_gen,
                epochs=epochs,
                validation_data=val_gen,
                class_weight=class_weights,
                callbacks=callbacks_list,
                verbose=1
            )
            
            training_time = datetime.now() - start_time
            
            print(f"✅ Entraînement terminé!")
            print(f"⏱️ Temps d'entraînement: {training_time}")
            
            # Sauvegarde de l'historique
            self.histories[config_name] = history
            
            # Évaluation sur le jeu de test si disponible
            results = {
                'config_name': config_name,
                'training_time': str(training_time),
                'final_epoch': len(history.history['loss']),
                'best_val_accuracy': max(history.history.get('val_accuracy', [0])),
                'best_val_loss': min(history.history.get('val_loss', [float('inf')])),
                'history': history.history
            }
            
            # Test si disponible
            if 'test' in generators:
                print(f"\n📊 ÉVALUATION SUR LE JEU DE TEST")
                test_results = self.evaluate_model(config_name, generators['test'])
                results.update(test_results)
            
            # Sauvegarde des résultats
            self.results[config_name] = results
            
            # Génération des graphiques d'entraînement
            self._plot_training_history(config_name, history)
            
            return results
            
        except Exception as e:
            print(f"❌ Erreur pendant l'entraînement: {e}")
            raise e
    
    def _create_default_callbacks(self, config_name: str, training_config: Dict) -> List:
        """Crée les callbacks par défaut."""
        callbacks_list = []
        
        # Model checkpoint
        if training_config.get('save_best_model', True):
            checkpoint_dir = self.config['model_saving']['output_directory']
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = os.path.join(checkpoint_dir, f"{config_name}_best.h5")
            checkpoint = callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
            callbacks_list.append(checkpoint)
        
        # Early stopping
        if training_config.get('early_stopping', True):
            early_stop = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=training_config.get('patience', 10),
                restore_best_weights=True,
                verbose=1
            )
            callbacks_list.append(early_stop)
        
        # Reduce learning rate
        if training_config.get('reduce_lr', True):
            reduce_lr = callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=training_config.get('lr_factor', 0.2),
                patience=training_config.get('lr_patience', 5),
                min_lr=training_config.get('min_lr', 1e-7),
                verbose=1
            )
            callbacks_list.append(reduce_lr)
        
        # TensorBoard
        if training_config.get('tensorboard', False):
            log_dir = os.path.join('logs', config_name, datetime.now().strftime("%Y%m%d-%H%M%S"))
            tensorboard = callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True,
                write_images=True
            )
            callbacks_list.append(tensorboard)
        
        return callbacks_list
    
    def evaluate_model(self, config_name: str, test_generator) -> Dict:
        """Évalue un modèle sur un jeu de test."""
        if config_name not in self.models:
            raise ValueError(f"Modèle non trouvé: {config_name}")
        
        model = self.models[config_name]
        
        print(f"📊 Évaluation du modèle {config_name}")
        
        # Évaluation basique
        test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)[:2]
        
        # Prédictions détaillées
        predictions = model.predict(test_generator, verbose=0)
        
        # Vraies étiquettes
        true_labels = test_generator.classes[:len(predictions)]
        
        # Prédictions de classe
        if predictions.shape[1] == 1:
            # Binaire
            predicted_classes = (predictions > 0.5).astype(int).flatten()
        else:
            # Multi-classe
            predicted_classes = np.argmax(predictions, axis=1)
        
        # Métriques détaillées
        f1 = f1_score(true_labels, predicted_classes, average='weighted')
        
        # Rapport de classification
        class_names = list(test_generator.class_indices.keys())
        class_report = classification_report(
            true_labels, 
            predicted_classes, 
            target_names=class_names,
            output_dict=True
        )
        
        # Matrice de confusion
        cm = confusion_matrix(true_labels, predicted_classes)
        
        results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'test_f1_score': float(f1),
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'class_names': class_names
        }
        
        print(f"✅ Accuracy: {test_accuracy:.4f}")
        print(f"✅ F1-Score: {f1:.4f}")
        print(f"✅ Loss: {test_loss:.4f}")
        
        # Sauvegarder la matrice de confusion
        self._plot_confusion_matrix(config_name, cm, class_names)
        
        return results
    
    def _plot_training_history(self, config_name: str, history):
        """Génère les graphiques d'historique d'entraînement."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Historique d\'entraînement - {config_name}', fontsize=16)
            
            # Accuracy
            axes[0, 0].plot(history.history['accuracy'], label='Train')
            if 'val_accuracy' in history.history:
                axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
            axes[0, 0].set_title('Accuracy')
            axes[0, 0].set_xlabel('Époque')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Loss
            axes[0, 1].plot(history.history['loss'], label='Train')
            if 'val_loss' in history.history:
                axes[0, 1].plot(history.history['val_loss'], label='Validation')
            axes[0, 1].set_title('Loss')
            axes[0, 1].set_xlabel('Époque')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # Learning rate si disponible
            if 'lr' in history.history:
                axes[1, 0].plot(history.history['lr'])
                axes[1, 0].set_title('Learning Rate')
                axes[1, 0].set_xlabel('Époque')
                axes[1, 0].set_ylabel('LR')
                axes[1, 0].set_yscale('log')
                axes[1, 0].grid(True)
            
            # Métriques supplémentaires
            if 'precision' in history.history:
                axes[1, 1].plot(history.history['precision'], label='Precision')
            if 'recall' in history.history:
                axes[1, 1].plot(history.history['recall'], label='Recall')
            if len(axes[1, 1].lines) > 0:
                axes[1, 1].set_title('Precision & Recall')
                axes[1, 1].set_xlabel('Époque')
                axes[1, 1].legend()
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            # Sauvegarder
            output_dir = self.config['model_saving']['output_directory']
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(output_dir, f"{config_name}_training_history_{timestamp}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📈 Graphiques d'entraînement sauvegardés: {plot_path}")
            
        except Exception as e:
            print(f"⚠️ Erreur lors de la création des graphiques: {e}")
    
    def _plot_confusion_matrix(self, config_name: str, cm: np.ndarray, class_names: List[str]):
        """Génère la matrice de confusion."""
        try:
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names)
            plt.title(f'Matrice de Confusion - {config_name}')
            plt.xlabel('Prédiction')
            plt.ylabel('Vérité terrain')
            
            # Sauvegarder
            output_dir = self.config['model_saving']['output_directory']
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(output_dir, f"{config_name}_confusion_matrix_{timestamp}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"🔄 Matrice de confusion sauvegardée: {plot_path}")
            
        except Exception as e:
            print(f"⚠️ Erreur lors de la création de la matrice de confusion: {e}")
    
    def compare_models(self, results_dict: Dict = None) -> pd.DataFrame:
        """Compare les performances de plusieurs modèles."""
        if results_dict is None:
            results_dict = self.results
        
        if not results_dict:
            print("❌ Aucun résultat à comparer")
            return None
        
        print(f"\n{'='*80}")
        print(f"📊 COMPARAISON DES MODÈLES")
        print(f"{'='*80}")
        
        comparison_data = []
        for config_name, results in results_dict.items():
            row = {
                'Modèle': config_name,
                'Temps_Entraînement': results.get('training_time', 'N/A'),
                'Époques': results.get('final_epoch', 'N/A'),
                'Meilleure_Val_Accuracy': f"{results.get('best_val_accuracy', 0):.4f}",
                'Meilleure_Val_Loss': f"{results.get('best_val_loss', float('inf')):.4f}"
            }
            
            if 'test_accuracy' in results:
                row['Test_Accuracy'] = f"{results['test_accuracy']:.4f}"
                row['Test_F1'] = f"{results['test_f1_score']:.4f}"
                row['Test_Loss'] = f"{results['test_loss']:.4f}"
            
            comparison_data.append(row)
        
        df_comparison = pd.DataFrame(comparison_data)
        print(df_comparison.to_string(index=False))
        
        # Identifier le meilleur modèle
        if 'Test_Accuracy' in df_comparison.columns:
            best_model_idx = df_comparison['Test_Accuracy'].astype(float).idxmax()
            best_model = df_comparison.loc[best_model_idx, 'Modèle']
            print(f"\n🏆 MEILLEUR MODÈLE: {best_model}")
        
        return df_comparison
    
    def save_model(self, config_name: str, format_type: str = 'h5'):
        """Sauvegarde un modèle entraîné."""
        if config_name not in self.models:
            raise ValueError(f"Modèle non trouvé: {config_name}")
        
        model = self.models[config_name]
        output_dir = self.config['model_saving']['output_directory']
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format_type == 'h5':
            model_path = os.path.join(output_dir, f"{config_name}_{timestamp}.h5")
            model.save(model_path)
        elif format_type == 'savedmodel':
            model_path = os.path.join(output_dir, f"{config_name}_{timestamp}")
            model.save(model_path, save_format='tf')
        else:
            raise ValueError(f"Format non supporté: {format_type}")
        
        print(f"💾 Modèle sauvegardé: {model_path}")
        return model_path
    
    def get_available_configs(self) -> List[Dict]:
        """Retourne les configurations disponibles."""
        configs = []
        for name, config in self.config['model_configs'].items():
            configs.append({
                'name': name,
                'description': config['description'],
                'architecture_type': config['architecture_type'],
                'input_shape': config['architecture_params']['input_shape'],
                'num_classes': config['architecture_params']['num_classes']
            })
        return configs


# Fonction utilitaire pour charger un modèle sauvegardé
def load_trained_model(model_path: str) -> tf.keras.Model:
    """Charge un modèle TensorFlow sauvegardé."""
    return tf.keras.models.load_model(model_path)


# Exemple d'utilisation
if __name__ == "__main__":
    print("🧪 Test du gestionnaire de pipelines TensorFlow")
    
    # Créer le gestionnaire
    manager = TensorFlowPipelineManager("Pipeline_TensorFlow_config.json")
    
    # Afficher les configurations disponibles
    print("\n📋 Configurations disponibles:")
    configs = manager.get_available_configs()
    for config in configs[:3]:  # Afficher les 3 premières
        print(f"- {config['name']}: {config['description']}")
        print(f"  Architecture: {config['architecture_type']}")
        print(f"  Input: {config['input_shape']}, Classes: {config['num_classes']}")
    
    # Exemple de création de modèle (décommenter pour tester)
    """
    # Créer un modèle de test
    model = manager.create_model('basic_cnn')
    
    # Configurer les données (remplacer par vos chemins)
    data_dir = "data/processed/organized"
    generators = manager.setup_data_generators(
        train_dir=data_dir,
        config_name='basic_cnn'
    )
    
    # Entraîner le modèle
    results = manager.train_model(
        config_name='basic_cnn',
        generators=generators,
        epochs=10
    )
    
    print("✅ Entraînement terminé!")
    """