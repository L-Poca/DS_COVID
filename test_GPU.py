import tensorflow as tf
import os

# Désactiver les messages oneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print()
print("TensorFlow version:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())

# Utiliser la nouvelle API
print("GPU devices:", tf.config.list_physical_devices('GPU'))

# Configuration GPU si disponible
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Permettre la croissance de mémoire GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU configuré: {gpus[0]}")
    except RuntimeError as e:
        print(f"Erreur GPU: {e}")
else:
    print("Aucun GPU détecté")