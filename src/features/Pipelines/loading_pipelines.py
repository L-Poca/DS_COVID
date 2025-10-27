"""
Module pour le chargement et la création de pipelines de traitement de données.

Ce module fournit des utilitaires pour charger les chemins d'images et de masques
depuis une structure de répertoires, et pour créer dynamiquement des pipelines
scikit-learn à partir de configurations JSON.
"""

import os
import glob
import json
import importlib
import inspect
from typing import List, Tuple, Dict, Any, Optional
from sklearn.pipeline import Pipeline


def load_paths_data_raw(root_dir: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Charge les chemins des fichiers d'images et de masques depuis un répertoire structuré.

    Args:
        root_dir (str): Répertoire racine contenant les sous-répertoires de classes.
                       Structure attendue : root_dir/nom_classe/images/*.png
                                          root_dir/nom_classe/masks/*.png

    Returns:
        Tuple[List[str], List[str], List[str]]: Trois listes contenant :
            - chemins_images : Liste des chemins des fichiers d'images
            - chemins_masques : Liste des chemins des masques correspondants
            - etiquettes : Liste des étiquettes de classe (en minuscules)

    Raises:
        OSError: Si root_dir n'existe pas ou n'est pas accessible.
    """
    chemins_images = []
    chemins_masques = []
    etiquettes = []

    if not os.path.exists(root_dir):
        raise OSError(f"Le répertoire racine '{root_dir}' n'existe pas")

    for etiquette in os.listdir(root_dir):
        rep_img_classe = os.path.join(root_dir, etiquette, 'images')
        rep_masque_classe = os.path.join(root_dir, etiquette, 'masks')

        if os.path.isdir(rep_img_classe):
            for chemin_img in glob.glob(os.path.join(rep_img_classe, '*.png')):
                nom_fichier = os.path.basename(chemin_img)
                chemin_masque = os.path.join(rep_masque_classe, nom_fichier)

                if os.path.exists(chemin_masque):
                    chemins_images.append(chemin_img)
                    chemins_masques.append(chemin_masque)
                    etiquettes.append(etiquette.lower())

    return chemins_images, chemins_masques, etiquettes


def discover_available_pipelines(configs_dir: str = None) -> Dict[str, Dict[str, Any]]:
    """
    Découvre automatiquement tous les pipelines disponibles en scannant le dossier de configurations.

    Args:
        configs_dir (str, optional): Chemin vers le dossier des configurations.
                                   Si None, utilise le chemin par défaut.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionnaire des pipelines disponibles avec leurs métadonnées.
                                 Format: {nom_pipeline: {file: str, name: str, description: str, category: str}}

    Raises:
        OSError: Si le dossier de configurations n'existe pas
    """
    if configs_dir is None:
        configs_dir = os.path.join(
            os.path.dirname(__file__), 
            "Configs_Pipelines"
        )
    
    if not os.path.exists(configs_dir):
        raise OSError(f"Le dossier de configurations '{configs_dir}' n'existe pas")

    pipelines_disponibles = {}
    
    # Scanner tous les fichiers JSON dans le dossier
    pattern_json = os.path.join(configs_dir, "*.json")
    
    for chemin_fichier in glob.glob(pattern_json):
        nom_fichier = os.path.basename(chemin_fichier)
        
        # Ignorer les fichiers de configuration système
        if nom_fichier in ["pipeline_config.json", "config.json", "settings.json"]:
            continue
            
        try:
            # Charger et analyser le fichier de configuration
            with open(chemin_fichier, 'r', encoding='utf-8') as fichier:
                config = json.load(fichier)
            
            # Extraire le nom du pipeline (nom du fichier sans extension)
            nom_pipeline = os.path.splitext(nom_fichier)[0]
            
            # Nettoyer le nom (retirer les préfixes comme "pipeline_")
            if nom_pipeline.startswith("pipeline_"):
                nom_pipeline_clean = nom_pipeline[9:]  # Retirer "pipeline_"
            else:
                nom_pipeline_clean = nom_pipeline
                
            # Déterminer la catégorie automatiquement
            category = _detect_pipeline_category(config, nom_pipeline)
            
            # Créer l'entrée du registre
            pipelines_disponibles[nom_pipeline_clean] = {
                "file": chemin_fichier,
                "name": config.get("name", nom_pipeline_clean.replace("_", " ").title()),
                "description": config.get("description", f"Pipeline {nom_pipeline_clean}"),
                "category": category,
                "original_filename": nom_fichier
            }
            
        except (json.JSONDecodeError, KeyError, OSError) as e:
            print(f"⚠️  Impossible de charger {nom_fichier}: {e}")
            continue
    
    return pipelines_disponibles


def _detect_pipeline_category(config: Dict[str, Any], nom_pipeline: str) -> str:
    """
    Détecte automatiquement la catégorie d'un pipeline basé sur sa configuration.

    Args:
        config (Dict[str, Any]): Configuration du pipeline
        nom_pipeline (str): Nom du pipeline

    Returns:
        str: Catégorie détectée
    """
    # Vérifier si la catégorie est explicitement définie
    if "category" in config:
        return config["category"]
    
    # Détection basée sur le nom
    nom_lower = nom_pipeline.lower()
    
    if "composite" in nom_lower:
        return "composite"
    elif "tensorflow" in nom_lower or "keras" in nom_lower or "transfer" in nom_lower:
        return "deep_learning"
    elif "augment" in nom_lower:
        return "data_augmentation"
    elif "feature" in nom_lower:
        return "feature_engineering"
    elif "simple" in nom_lower or "basic" in nom_lower:
        return "basic"
    
    # Détection basée sur les étapes du pipeline
    steps = config.get("steps", [])
    step_classes = [step.get("class", "").lower() for step in steps]
    
    if any("tensorflow" in cls or "keras" in cls for cls in step_classes):
        return "deep_learning"
    elif any("augment" in cls for cls in step_classes):
        return "data_augmentation"
    elif any("feature" in cls or "pca" in cls for cls in step_classes):
        return "feature_engineering"
    elif config.get("type") == "composite":
        return "composite"
    else:
        return "data_processing"


def load_pipeline_by_name(nom_pipeline: str, masques: Optional[List[str]] = None, 
                         configs_dir: str = None) -> Pipeline:
    """
    Charge un pipeline par son nom en utilisant la découverte automatique.

    Args:
        nom_pipeline (str): Nom du pipeline à charger
        masques (Optional[List[str]]): Liste optionnelle de chemins de masques à injecter
        configs_dir (str, optional): Chemin vers le dossier des configurations

    Returns:
        Pipeline: Objet Pipeline scikit-learn configuré

    Raises:
        ValueError: Si nom_pipeline n'est pas trouvé
        FileNotFoundError: Si les fichiers de configuration ne sont pas trouvés
    """
    # Découvrir automatiquement les pipelines disponibles
    pipelines_disponibles = discover_available_pipelines(configs_dir)

    if nom_pipeline not in pipelines_disponibles:
        disponibles = list(pipelines_disponibles.keys())
        raise ValueError(
            f"Pipeline '{nom_pipeline}' non trouvé. Disponibles : {disponibles}"
        )

    pipeline_info = pipelines_disponibles[nom_pipeline]
    fichier_pipeline = pipeline_info["file"]
    config_pipeline = load_pipeline_config(fichier_pipeline)

    print(f"Chargement du pipeline : {pipeline_info['name']}")
    print(f"Description : {pipeline_info['description']}")
    print(f"Catégorie : {pipeline_info['category']}")

    return create_pipeline_from_config(config_pipeline, masques)


def get_default_pipeline(masques: Optional[List[str]] = None, configs_dir: str = None) -> Pipeline:
    """
    Charge le pipeline par défaut (simple) ou le premier disponible.

    Args:
        masques (Optional[List[str]]): Liste optionnelle de chemins de masques à injecter
        configs_dir (str, optional): Chemin vers le dossier des configurations

    Returns:
        Pipeline: Objet Pipeline scikit-learn configuré

    Raises:
        ValueError: Si aucun pipeline n'est disponible
    """
    pipelines_disponibles = discover_available_pipelines(configs_dir)
    
    if not pipelines_disponibles:
        raise ValueError("Aucun pipeline disponible")
    
    # Essayer de trouver le pipeline "simple" en priorité
    for nom_prefere in ["simple", "basic", "default"]:
        if nom_prefere in pipelines_disponibles:
            return load_pipeline_by_name(nom_prefere, masques, configs_dir)
    
    # Sinon, prendre le premier disponible
    premier_pipeline = list(pipelines_disponibles.keys())[0]
    print(f"⚠️  Aucun pipeline par défaut trouvé, utilisation de '{premier_pipeline}'")
    return load_pipeline_by_name(premier_pipeline, masques, configs_dir)


def get_transformer_class(nom_classe: str) -> type:
    """
    Récupère une classe de transformateur depuis différents contextes.

    Cherche d'abord dans le scope global du module appelant,
    puis dans les modules de transformateurs.

    Args:
        nom_classe (str): Nom de la classe de transformateur

    Returns:
        type: La classe de transformateur

    Raises:
        NameError: Si nom_classe n'est pas trouvé
    """
    # 1. Chercher dans le scope global du module appelant
    frame = inspect.currentframe()
    try:
        # Remonter la pile d'appels pour trouver le bon contexte
        caller_frame = frame.f_back
        while caller_frame:
            caller_globals = caller_frame.f_globals
            if nom_classe in caller_globals:
                return caller_globals[nom_classe]
            caller_frame = caller_frame.f_back
    finally:
        del frame

    # 2. Chercher dans les modules de transformateurs et sklearn
    modules_transformateurs = [
        'src.features.Pipelines.Transformateurs.image_loaders',
        'src.features.Pipelines.Transformateurs.image_preprocessing',
        'src.features.Pipelines.Transformateurs.image_features',
        'src.features.Pipelines.Transformateurs.image_augmentation',
        'src.features.Pipelines.Transformateurs.utilities',
        'src.features.Pipelines.Transformateurs.tensorflow_transformers',
        'sklearn.linear_model',
        'sklearn.ensemble',
        'sklearn.svm',
        'sklearn.tree',
        'sklearn.naive_bayes',
        'sklearn.neighbors'
    ]

    for module_name in modules_transformateurs:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, nom_classe):
                return getattr(module, nom_classe)
        except ImportError:
            continue

    # 3. Si rien n'est trouvé, lever une exception
    raise NameError(f"Classe '{nom_classe}' non trouvée dans le scope global ou les modules")


def load_pipeline_config(fichier_pipeline: str) -> Dict[str, Any]:
    """
    Charge la configuration d'un pipeline depuis un fichier JSON.

    Args:
        fichier_pipeline (str): Chemin vers le fichier de configuration JSON

    Returns:
        Dict[str, Any]: Dictionnaire de configuration du pipeline

    Raises:
        FileNotFoundError: Si le fichier de configuration n'existe pas
        json.JSONDecodeError: Si le fichier contient du JSON invalide
    """
    try:
        with open(fichier_pipeline, 'r', encoding='utf-8') as fichier:
            return json.load(fichier)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Fichier de config pipeline '{fichier_pipeline}' non trouvé"
        ) from exc
    except json.JSONDecodeError as exc:
        raise json.JSONDecodeError(
            f"JSON invalide dans le fichier de config '{fichier_pipeline}'",
            exc.doc,
            exc.pos
        ) from exc


def create_pipeline_from_config(config: Dict[str, Any],
                               masques: Optional[List[str]] = None) -> Pipeline:
    """
    Crée un pipeline scikit-learn à partir d'une configuration JSON.
    Gère les pipelines simples et composites.

    Args:
        config (Dict[str, Any]): Dictionnaire de configuration du pipeline
        masques (Optional[List[str]]): Chemins de masques optionnels à injecter

    Returns:
        Pipeline: Objet Pipeline scikit-learn configuré

    Raises:
        ValueError: Si le type de pipeline n'est pas supporté
    """
    type_pipeline = config.get("type", "simple")

    if type_pipeline == "composite":
        return create_composite_pipeline(config, masques)
    if type_pipeline == "simple":
        return create_simple_pipeline(config, masques)
    raise ValueError(f"Type de pipeline non supporté : {type_pipeline}")


def create_simple_pipeline(config: Dict[str, Any],
                          masques: Optional[List[str]] = None) -> Pipeline:
    """
    Crée un pipeline simple à partir des définitions d'étapes.

    Args:
        config (Dict[str, Any]): Configuration du pipeline contenant les étapes
        masques (Optional[List[str]]): Chemins de masques optionnels à injecter

    Returns:
        Pipeline: Objet Pipeline scikit-learn configuré

    Raises:
        KeyError: Si les clés de configuration requises sont manquantes
        NameError: Si les classes de transformateur ne sont pas trouvées
    """
    etapes = []

    for etape in config["steps"]:
        classe_transformateur = get_transformer_class(etape["class"])
        parametres = etape["params"].copy()  # Copie pour éviter de modifier l'original

        # Injecter les chemins de masques si nécessaire
        if ("mask_paths" in parametres and
            parametres["mask_paths"] == "masks" and
            masques is not None):
            parametres["mask_paths"] = masques

        etapes.append((etape["name"], classe_transformateur(**parametres)))

    return Pipeline(etapes, verbose=True)


def create_composite_pipeline(config: Dict[str, Any],
                             masques: Optional[List[str]] = None) -> Pipeline:
    """
    Crée un pipeline composite en combinant d'autres pipelines.

    Args:
        config (Dict[str, Any]): Configuration du pipeline composite
        masques (Optional[List[str]]): Chemins de masques optionnels à injecter

    Returns:
        Pipeline: Objet Pipeline scikit-learn configuré

    Raises:
        KeyError: Si les clés de configuration requises sont manquantes
        FileNotFoundError: Si les fichiers de sous-pipelines ne sont pas trouvés
        NameError: Si les classes de transformateur ne sont pas trouvées
    """
    toutes_etapes = []

    for etape in config["steps"]:
        if etape.get("type") == "pipeline":
            # Charger le sous-pipeline
            sous_config = load_pipeline_config(etape["pipeline_file"])

            # Filtrer les étapes si nécessaire
            sous_etapes = sous_config["steps"]
            if "include_steps" in etape:
                sous_etapes = [s for s in sous_etapes if s["name"] in etape["include_steps"]]
            elif "exclude_steps" in etape:
                sous_etapes = [s for s in sous_etapes if s["name"] not in etape["exclude_steps"]]

            # Ajouter les étapes du sous-pipeline
            for sous_etape in sous_etapes:
                classe_transformateur = get_transformer_class(sous_etape["class"])
                parametres = sous_etape["params"].copy()  # Copie pour éviter de modifier l'original

                # Injecter les chemins de masques si nécessaire
                if ("mask_paths" in parametres and
                    parametres["mask_paths"] == "masks" and
                    masques is not None):
                    parametres["mask_paths"] = masques

                nom_etape = f"{etape['name']}_{sous_etape['name']}"
                toutes_etapes.append((nom_etape, classe_transformateur(**parametres)))

        else:
            # Étape normale
            classe_transformateur = get_transformer_class(etape["class"])
            parametres = etape["params"].copy()  # Copie pour éviter de modifier l'original

            # Injecter les chemins de masques si nécessaire
            if ("mask_paths" in parametres and
                parametres["mask_paths"] == "masks" and
                masques is not None):
                parametres["mask_paths"] = masques

            toutes_etapes.append((etape["name"], classe_transformateur(**parametres)))

    return Pipeline(toutes_etapes, verbose=True)


def print_available_pipelines(configs_dir: str = None) -> None:
    """
    Affiche tous les pipelines disponibles organisés par catégorie.

    Args:
        configs_dir (str, optional): Chemin vers le dossier des configurations
    """
    try:
        pipelines_disponibles = discover_available_pipelines(configs_dir)
        
        if not pipelines_disponibles:
            print("❌ Aucun pipeline disponible")
            return
        
        print("🔍 Pipelines découverts automatiquement:")
        print("=" * 60)
        
        # Organiser par catégorie
        categories = {}
        for nom, info in pipelines_disponibles.items():
            category = info.get("category", "other")
            if category not in categories:
                categories[category] = []
            categories[category].append((nom, info))
        
        # Afficher par catégorie
        for category, pipelines in sorted(categories.items()):
            print(f"\n📁 Catégorie: {category.upper()}")
            print("-" * 40)
            
            for nom, info in sorted(pipelines):
                print(f"  📄 {nom}")
                print(f"     Nom: {info['name']}")
                print(f"     Description: {info['description']}")
                print(f"     Fichier: {info['original_filename']}")
                print()
        
        print(f"📊 Total: {len(pipelines_disponibles)} pipeline(s) trouvé(s)")
        
    except Exception as e:
        print(f"❌ Erreur lors de la découverte des pipelines: {e}")


def list_pipelines_by_category(category: str = None, configs_dir: str = None) -> List[str]:
    """
    Liste les noms des pipelines disponibles, optionnellement filtrés par catégorie.

    Args:
        category (str, optional): Catégorie à filtrer (ex: 'deep_learning', 'basic')
        configs_dir (str, optional): Chemin vers le dossier des configurations

    Returns:
        List[str]: Liste des noms de pipelines

    Raises:
        ValueError: Si la catégorie spécifiée n'existe pas
    """
    pipelines_disponibles = discover_available_pipelines(configs_dir)
    
    if category is None:
        return list(pipelines_disponibles.keys())
    
    # Filtrer par catégorie
    pipelines_filtres = [
        nom for nom, info in pipelines_disponibles.items()
        if info.get("category", "").lower() == category.lower()
    ]
    
    if not pipelines_filtres and category:
        categories_disponibles = set(
            info.get("category", "") for info in pipelines_disponibles.values()
        )
        raise ValueError(
            f"Catégorie '{category}' non trouvée. "
            f"Disponibles: {sorted(categories_disponibles)}"
        )
    
    return pipelines_filtres


def set_pipeline_random_state(pipeline, random_state):
    """Applique une seed à tous les composants du pipeline qui l'acceptent"""
    print(f"\n🎲 Application de la seed {random_state} au pipeline:\n")
    for step_name, step in pipeline.steps:
        if hasattr(step, 'random_state'):
            step.random_state = random_state
            print(f"  ✅ Seed {random_state} appliquée à {step_name}")
        elif hasattr(step, 'set_params'):
            try:
                step.set_params(random_state=random_state)
                print(f"  ✅ Seed {random_state} appliquée à {step_name} via set_params")
            except:
                print(f"  ⚠️  {step_name} ne supporte pas random_state")

def display_pipeline_structure(registry_config: Dict[str, Any]) -> None:
    """Affiche la structure du pipeline de manière lisible"""
    print("Pipelines disponibles:")
    print("=" * 50)

    # Organiser par catégorie
    categories = {}
    for name, info in registry_config.items():
        # Ignorer les métadonnées
        if name in ["description"]:
            continue
        
        if isinstance(info, dict) and "category" in info:
            category = info.get("category", "other")
            if category not in categories:
                categories[category] = []
            categories[category].append((name, info))

    # Afficher par catégorie
    for category, pipelines in sorted(categories.items()):
        print(f"\nCatégorie: {category.upper()}\n")
        for name, info in pipelines:
            status = "(par défaut)" if name == "default" else ""
            print(f"  {name}: {info['description']} {status}")
            print(f"     Fichier: {info['original_filename']}\n")