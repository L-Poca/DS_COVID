

import glob
import importlib
import inspect
import os


def discover_transformers(project_root):
    """Découvre automatiquement tous les transformateurs disponibles"""
    transformateurs_path = os.path.join(project_root, "src", "features", "Pipelines", "Transformateurs")
    transformers_found = {}
    transformers_with_seed = {}
    
    # Scanner tous les fichiers Python dans le dossier Transformateurs
    for file_path in glob.glob(os.path.join(transformateurs_path, "*.py")):
        file_name = os.path.basename(file_path)
        
        # Ignorer __init__.py et __pycache__
        if file_name.startswith("__"):
            continue
            
        module_name = file_name[:-3]  # Retirer .py
        
        try:
            # Importer le module dynamiquement
            module = importlib.import_module(f"src.features.Pipelines.Transformateurs.{module_name}")
            
            # Chercher toutes les classes qui héritent de BaseEstimator
            for name in dir(module):
                obj = getattr(module, name)
                if (inspect.isclass(obj) and 
                    hasattr(obj, 'fit') and 
                    hasattr(obj, 'transform') and
                    name not in ['BaseEstimator', 'TransformerMixin', 'ClassifierMixin']):
                    
                    if module_name not in transformers_found:
                        transformers_found[module_name] = []
                        transformers_with_seed[module_name] = []
                    
                    transformers_found[module_name].append(name)
                    
                    # Vérifier si le transformateur supporte random_state
                    seed_support = check_seed_support(obj)
                    if seed_support['has_seed']:
                        transformers_with_seed[module_name].append({
                            'name': name,
                            'method': seed_support['method']
                        })
                    
        except Exception as e:
            print(f"⚠️  Erreur lors de l'import de {module_name}: {e}")
    
    return transformers_found, transformers_with_seed

def check_seed_support(transformer_class):
    """Vérifie si un transformateur supporte random_state"""
    seed_info = {'has_seed': False, 'method': None}
    
    try:
        # Méthode 1: Vérifier dans __init__
        init_signature = inspect.signature(transformer_class.__init__)
        if 'random_state' in init_signature.parameters:
            seed_info = {'has_seed': True, 'method': 'constructor'}
            return seed_info
            
        # Méthode 2: Vérifier les attributs de classe
        if hasattr(transformer_class, 'random_state'):
            seed_info = {'has_seed': True, 'method': 'attribute'}
            return seed_info
            
        # Méthode 3: Vérifier si c'est un wrapper sklearn
        try:
            # Créer une instance temporaire pour tester
            temp_instance = transformer_class()
            if hasattr(temp_instance, 'set_params'):
                # Tester si set_params accepte random_state
                try:
                    temp_instance.set_params(random_state=42)
                    seed_info = {'has_seed': True, 'method': 'set_params'}
                except (TypeError, ValueError):
                    pass
        except:
            pass  # Ignore si on ne peut pas créer d'instance
            
    except Exception:
        pass
    
    return seed_info

def print_transformers(project_root):
    
    print("🔍 Découverte automatique des transformateurs:")
    print("=" * 50)

    transformers_by_module, transformers_with_seed_by_module = discover_transformers(project_root)

    total_transformers = 0
    total_with_seed = 0

    for module_name, transformer_names in transformers_by_module.items():
        print(f"\n📁 Module: {module_name}")
        print("-" * 30)
        
        # Transformateurs avec seed
        seed_transformers = transformers_with_seed_by_module.get(module_name, [])
        seed_names = [t['name'] for t in seed_transformers]
        
        for transformer_name in transformer_names:
            if transformer_name in seed_names:
                # Trouver la méthode de seed
                method = next(t['method'] for t in seed_transformers if t['name'] == transformer_name)
                print(f"  ✅ {transformer_name} 🎲 (seed via {method})")
                total_with_seed += 1
            else:
                print(f"  ✅ {transformer_name}")
            total_transformers += 1

    print(f"\n📊 Résumé:")
    print(f"  Total transformateurs: {total_transformers}")
    print(f"  Avec support seed: {total_with_seed}")
    print(f"  Sans seed: {total_transformers - total_with_seed}")
