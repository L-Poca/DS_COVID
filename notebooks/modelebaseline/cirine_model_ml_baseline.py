import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
from PIL import Image

def build_baseline_cnn(input_shape=(256, 256, 1),num_classes=4):
     
    model = models.Sequential([
    
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
    
        layers.Flatten(),
        layers.Dropout(0.5),

        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
         
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',   # pour labels en int (0,1,2,3)
        metrics=['accuracy']
    )
    
    return model

SEED=42
IMG_SIZE = (299, 299)
SAMPLES_PER_CLASS = 1000

# Définition des classes
classes = ['COVID', 'NORMAL', 'VIRAL', 'LUNG']
class_labels = {0: 'COVID', 1: 'NORMAL', 2: 'VIRAL', 3: 'LUNG'}

# Définition des chemins par classe
class_paths = {
    0: r"data\raw\COVID-19_Radiography_Dataset\COVID-19_Radiography_Dataset\COVID\images",
    1: r"data\raw\COVID-19_Radiography_Dataset\COVID-19_Radiography_Dataset\Normal\images",
    2: r"data\raw\COVID-19_Radiography_Dataset\COVID-19_Radiography_Dataset\Viral Pneumonia\images",
    3: r"data\raw\COVID-19_Radiography_Dataset\COVID-19_Radiography_Dataset\Lung_Opacity\images",
}


def load_images_flat(folder_path, label, img_size=(299, 299), max_images=None):
    """Charge les images en niveaux de gris et les aplatit."""
    data, labels = [], []
    files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    random.seed(SEED)
    random.shuffle(files)

    if max_images:
        files = files[:max_images]

    print(f"[INFO] Chargement de {len(files)} images pour la classe {class_labels[label]}")

    for file in files:
        try:
            img = Image.open(os.path.join(folder_path, file)).convert('L')
            img = img.resize(img_size)
            img=img.astype('float32')
            img_norm=(img/127.5)-1
            data.append(np.array(img_norm).flatten())
            labels.append(label)
        except Exception as e:
            print(f"[WARNING] Erreur avec {file} : {e}")

    return data, labels


if __name__ == "__main__":
    print("[INFO] Démarrage du chargement des données...")

    data, labels = [], []
    for label, path in class_paths.items():
        d, l = load_images_flat(path, label, img_size=IMG_SIZE, max_images=SAMPLES_PER_CLASS)
        data += d
        labels += l

    print(f"[INFO] Nombre total d'images chargées : {len(data)}")

    # Conversion en numpy arrays
    X = np.array(data)
    y = np.array(labels)

# Découpage en training / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

input_shape = (256, 256, 1)  
num_classes = 4              

model = build_multiclass_cnn(input_shape, num_classes)
model.summary()

history = model.fit(
    X_train,          
    y_train,          
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test)
)