"""Module de pipelines sklearn pour la détection COVID-19.

Ce module implémente des pipelines complets utilisant scikit-learn pour 
l'entraînement de modèles de classification d'images médicales COVID-19.
Il utilise les configurations JSON pour créer dynamiquement différents
types de pipelines.

Auteur: L-Poca
Date: 2025
"""

import json
import os
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier,
                              VotingClassifier)
from sklearn.feature_selection import SelectKBest, VarianceThreshold, chi2, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                           confusion_matrix, f1_score)
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import SVC


class ImagePreprocessor(BaseEstimator, TransformerMixin):
    """Transformateur personnalisé pour le preprocessing d'images médicales."""
    
    def __init__(self, flatten=True, normalize=True, resize=None):
        """
        Args:
            flatten (bool): Aplatir les images en vecteurs 1D
            normalize (bool): Normaliser les valeurs entre 0 et 1
            resize (tuple): Taille cible (width, height) ou None
        """
        self.flatten = flatten
        self.normalize = normalize
        self.resize = resize
    
    def fit(self, X, y=None):
        """Ajustement du transformateur (pas d'apprentissage nécessaire)."""
        return self
    
    def transform(self, X):
        """Application des transformations aux données."""
        X_transformed = X.copy()
        
        # Normalisation
        if self.normalize:
            X_transformed = X_transformed.astype(np.float32) / 255.0
        
        # Aplatissement
        if self.flatten and len(X_transformed.shape) > 2:
            n_samples = X_transformed.shape[0]
            X_transformed = X_transformed.reshape(n_samples, -1)
        
        return X_transformed


class MedicalImagePreprocessor(BaseEstimator, TransformerMixin):
    """Transformateur spécialisé pour images médicales radiographiques."""
    
    def __init__(self, enhance_contrast=True, noise_reduction=True, 
                 histogram_equalization=True):
        """
        Args:
            enhance_contrast (bool): Améliorer le contraste
            noise_reduction (bool): Réduction du bruit
            histogram_equalization (bool): Égalisation d'histogramme
        """
        self.enhance_contrast = enhance_contrast
        self.noise_reduction = noise_reduction
        self.histogram_equalization = histogram_equalization
    
    def fit(self, X, y=None):
        """Ajustement du transformateur."""
        return self
    
    def transform(self, X):
        """Application des transformations médicales."""
        X_transformed = X.copy()
        
        # Pour l'instant, on fait juste une normalisation basique
        # Dans une vraie implémentation, on utiliserait OpenCV ou PIL
        X_transformed = X_transformed.astype(np.float32)
        
        if self.histogram_equalization:
            # Simulation d'égalisation d'histogramme
            X_transformed = (X_transformed - X_transformed.min()) / (X_transformed.max() - X_transformed.min())
        
        # Aplatissement pour sklearn
        if len(X_transformed.shape) > 2:
            n_samples = X_transformed.shape[0]
            X_transformed = X_transformed.reshape(n_samples, -1)
        
        return X_transformed


class PipelineManager:
    """Gestionnaire de pipelines sklearn pour COVID-19."""
    
    def __init__(self, config_path="Pipeline_Sklearn_config.json"):
        """
        Args:
            config_path (str): Chemin vers le fichier de configuration JSON
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.pipelines = {}
        self.results = {}
    
    def _load_config(self):
        """Charge la configuration depuis le fichier JSON."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Fichier de configuration non trouvé: {self.config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Erreur dans le fichier JSON: {e}")
    
    def _get_transformer(self, transformer_name, params):
        """Retourne l'instance du transformateur sklearn."""
        transformers = {
            'StandardScaler': StandardScaler,
            'MinMaxScaler': MinMaxScaler,
            'RobustScaler': RobustScaler,
            'SimpleImputer': SimpleImputer,
            'PCA': PCA,
            'SelectKBest': SelectKBest,
            'VarianceThreshold': VarianceThreshold,
            'RandomForestClassifier': RandomForestClassifier,
            'SVC': SVC,
            'GradientBoostingClassifier': GradientBoostingClassifier,
            'LogisticRegression': LogisticRegression,
            'VotingClassifier': VotingClassifier,
            'ImagePreprocessor': ImagePreprocessor,
            'MedicalImagePreprocessor': MedicalImagePreprocessor
        }
        
        if transformer_name not in transformers:
            raise ValueError(f"Transformateur non supporté: {transformer_name}")
        
        # Gestion spéciale pour SelectKBest
        if transformer_name == 'SelectKBest' and 'score_func' in params:
            score_func_name = params.pop('score_func')
            score_funcs = {
                'f_classif': f_classif,
                'chi2': chi2,
                'mutual_info_classif': f_classif  # Simplification
            }
            params['score_func'] = score_funcs.get(score_func_name, f_classif)
        
        # Gestion spéciale pour VotingClassifier
        if transformer_name == 'VotingClassifier' and 'estimators' in params:
            estimators = []
            for est_name, est_class, est_params in params['estimators']:
                est_instance = self._get_transformer(est_class, est_params)
                estimators.append((est_name, est_instance))
            params['estimators'] = estimators
        
        return transformers[transformer_name](**params)
    
    def create_pipeline(self, config_name):
        """Crée un pipeline à partir de la configuration."""
        if config_name not in self.config['pipeline_configs']:
            raise ValueError(f"Configuration non trouvée: {config_name}")
        
        pipeline_config = self.config['pipeline_configs'][config_name]
        
        # Construction des étapes du pipeline
        steps = []
        for step in pipeline_config['steps']:
            transformer = self._get_transformer(
                step['transformer'], 
                step['params']
            )
            steps.append((step['name'], transformer))
        
        # Création du pipeline
        pipeline = Pipeline(steps)
        
        # Ajout de GridSearch si nécessaire
        if pipeline_config.get('grid_search', False):
            grid_params = pipeline_config.get('grid_params', {})
            cv_folds = pipeline_config.get('cv_folds', 5)
            scoring = pipeline_config.get('scoring', 'accuracy')
            n_jobs = self.config['default_settings'].get('n_jobs', -1)
            
            pipeline = GridSearchCV(
                pipeline,
                param_grid=grid_params,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=self.config['default_settings'].get('verbose', 1)
            )
        
        self.pipelines[config_name] = pipeline
        return pipeline
    
    def train_pipeline(self, config_name, X_train, y_train, X_test=None, y_test=None):
        """Entraîne un pipeline et l'évalue."""
        print(f"\n{'='*60}")
        print(f"🚀 ENTRAÎNEMENT DU PIPELINE : {config_name}")
        print(f"{'='*60}")
        
        # Créer le pipeline si pas déjà fait
        if config_name not in self.pipelines:
            self.create_pipeline(config_name)
        
        pipeline = self.pipelines[config_name]
        pipeline_config = self.config['pipeline_configs'][config_name]
        
        print(f"Description: {pipeline_config['description']}")
        print(f"Données d'entraînement: {X_train.shape}")
        
        # Entraînement
        start_time = datetime.now()
        pipeline.fit(X_train, y_train)
        training_time = datetime.now() - start_time
        
        print(f"⏱️ Temps d'entraînement: {training_time}")
        
        # Résultats d'entraînement
        results = {
            'config_name': config_name,
            'pipeline': pipeline,
            'training_time': training_time,
            'config': pipeline_config
        }
        
        # Si GridSearch, récupérer les meilleurs paramètres
        if hasattr(pipeline, 'best_params_'):
            results['best_params'] = pipeline.best_params_
            results['best_score'] = pipeline.best_score_
            print(f"🎯 Meilleur score CV: {pipeline.best_score_:.4f}")
            print(f"🔧 Meilleurs paramètres: {pipeline.best_params_}")
        
        # Évaluation sur les données de test si disponibles
        if X_test is not None and y_test is not None:
            print(f"\n📊 ÉVALUATION SUR LE JEU DE TEST")
            print(f"Données de test: {X_test.shape}")
            
            # Prédictions
            y_pred = pipeline.predict(X_test)
            
            # Métriques
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            results.update({
                'test_accuracy': accuracy,
                'test_f1': f1,
                'y_pred': y_pred,
                'y_test': y_test
            })
            
            print(f"✅ Précision: {accuracy:.4f}")
            print(f"✅ F1-Score: {f1:.4f}")
            
            # Rapport de classification détaillé
            if self.config['evaluation_metrics'].get('detailed_report', True):
                report = classification_report(y_test, y_pred)
                results['classification_report'] = report
                print(f"\n📋 RAPPORT DE CLASSIFICATION:")
                print(report)
            
            # Matrice de confusion
            if self.config['evaluation_metrics'].get('confusion_matrix', True):
                cm = confusion_matrix(y_test, y_pred)
                results['confusion_matrix'] = cm
                print(f"\n🔄 MATRICE DE CONFUSION:")
                print(cm)
        
        # Cross-validation sur les données d'entraînement
        print(f"\n🔄 VALIDATION CROISÉE")
        cv_folds = pipeline_config.get('cv_folds', 5)
        
        # Utiliser le pipeline de base si c'est un GridSearch
        base_pipeline = pipeline.best_estimator_ if hasattr(pipeline, 'best_estimator_') else pipeline
        
        cv_scores = cross_val_score(base_pipeline, X_train, y_train, cv=cv_folds, scoring='accuracy')
        results['cv_scores'] = cv_scores
        results['cv_mean'] = cv_scores.mean()
        results['cv_std'] = cv_scores.std()
        
        print(f"📈 Score CV: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Sauvegarde des résultats
        self.results[config_name] = results
        
        # Sauvegarde du pipeline si configuré
        if self.config['model_saving'].get('save_pipeline', True):
            self._save_pipeline(config_name, pipeline)
        
        print(f"\n✅ Pipeline {config_name} entraîné avec succès!")
        return results
    
    def _save_pipeline(self, config_name, pipeline):
        """Sauvegarde un pipeline entraîné."""
        output_dir = self.config['model_saving'].get('output_directory', 'models/pipelines/')
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Nom de fichier avec timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        naming = self.config['model_saving'].get('naming_convention', 
                                                'pipeline_{config_name}_{timestamp}')
        filename = naming.format(config_name=config_name, timestamp=timestamp)
        filepath = Path(output_dir) / f"{filename}.pkl"
        
        # Sauvegarde
        with open(filepath, 'wb') as f:
            pickle.dump(pipeline, f)
        
        print(f"💾 Pipeline sauvegardé: {filepath}")
        return filepath
    
    def compare_pipelines(self, results_dict=None):
        """Compare les performances de plusieurs pipelines."""
        if results_dict is None:
            results_dict = self.results
        
        if not results_dict:
            print("❌ Aucun résultat à comparer")
            return None
        
        print(f"\n{'='*80}")
        print(f"📊 COMPARAISON DES PIPELINES")
        print(f"{'='*80}")
        
        # Création du tableau de comparaison
        comparison_data = []
        for config_name, results in results_dict.items():
            row = {
                'Pipeline': config_name,
                'CV_Score': f"{results.get('cv_mean', 0):.4f} (±{results.get('cv_std', 0):.4f})",
                'Training_Time': str(results.get('training_time', 'N/A')),
            }
            
            if 'test_accuracy' in results:
                row['Test_Accuracy'] = f"{results['test_accuracy']:.4f}"
                row['Test_F1'] = f"{results['test_f1']:.4f}"
            
            if 'best_score' in results:
                row['Best_GridSearch'] = f"{results['best_score']:.4f}"
            
            comparison_data.append(row)
        
        # Affichage du tableau
        df_comparison = pd.DataFrame(comparison_data)
        print(df_comparison.to_string(index=False))
        
        # Meilleur pipeline
        if comparison_data:
            best_pipeline = max(comparison_data, 
                              key=lambda x: float(x.get('Test_Accuracy', x['CV_Score'].split()[0])))
            print(f"\n🏆 MEILLEUR PIPELINE: {best_pipeline['Pipeline']}")
        
        return df_comparison
    
    def get_available_configs(self):
        """Retourne la liste des configurations disponibles."""
        configs = []
        for name, config in self.config['pipeline_configs'].items():
            configs.append({
                'name': name,
                'description': config['description'],
                'grid_search': config.get('grid_search', False),
                'cv_folds': config.get('cv_folds', 5)
            })
        return configs


def load_pipeline(filepath):
    """Charge un pipeline sauvegardé."""
    with open(filepath, 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline


def predict_with_pipeline(pipeline, X_new):
    """Fait des prédictions avec un pipeline."""
    predictions = pipeline.predict(X_new)
    
    # Probabilités si disponibles
    if hasattr(pipeline, 'predict_proba'):
        probabilities = pipeline.predict_proba(X_new)
        return predictions, probabilities
    elif hasattr(pipeline, 'decision_function'):
        scores = pipeline.decision_function(X_new)
        return predictions, scores
    
    return predictions, None


# Exemple d'utilisation
if __name__ == "__main__":
    # Exemple avec des données simulées
    print("🧪 Test du gestionnaire de pipelines")
    
    # Données simulées (remplacer par vos vraies données)
    np.random.seed(42)
    n_samples, n_features = 1000, 16384  # Simule des images 128x128
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(0, 4, n_samples)  # 4 classes: COVID, Normal, Viral, Lung
    
    # Division train/test
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Gestionnaire de pipelines
    manager = PipelineManager("Pipeline_Sklearn_config.json")
    
    # Afficher les configurations disponibles
    print("\n📋 Configurations disponibles:")
    configs = manager.get_available_configs()
    for config in configs[:3]:  # Afficher les 3 premières
        print(f"- {config['name']}: {config['description']}")
    
    # Tester quelques pipelines
    test_configs = ['basic_rf', 'fast_prototype']
    
    for config_name in test_configs:
        try:
            results = manager.train_pipeline(
                config_name, 
                X_train, y_train, 
                X_test, y_test
            )
        except Exception as e:
            print(f"❌ Erreur avec {config_name}: {e}")
    
    # Comparaison des résultats
    if manager.results:
        manager.compare_pipelines()