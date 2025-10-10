"""
Command Line Interface for DS-COVID package
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import os


def train_model():
    """CLI command for training models"""
    parser = argparse.ArgumentParser(
        description="Train COVID-19 classification model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--dataset-path", 
        type=str, 
        required=True,
        help="Path to COVID-19 dataset"
    )
    parser.add_argument(
        "--output-path", 
        type=str, 
        default="./models",
        help="Output directory for trained model"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--img-size", 
        type=int, 
        nargs=2,
        default=[256, 256],
        help="Image size (height width)"
    )
    parser.add_argument(
        "--max-samples", 
        type=int, 
        help="Maximum samples per class (for testing)"
    )
    parser.add_argument(
        "--validation-split", 
        type=float, 
        default=0.2,
        help="Validation split ratio"
    )
    
    args = parser.parse_args()
    
    # Import here to avoid loading heavy dependencies if not needed
    from .models import build_baseline_cnn
    from .features import load_images_flat, DEFAULT_CLASS_PATHS, DEFAULT_CLASS_LABELS
    from sklearn.model_selection import train_test_split
    import numpy as np
    
    print("🚀 Starting COVID-19 model training...")
    print(f"📁 Dataset: {args.dataset_path}")
    print(f"🖼️  Image size: {args.img_size}")
    print(f"🔄 Epochs: {args.epochs}")
    print(f"📦 Batch size: {args.batch_size}")
    
    # Load data
    data, labels = [], []
    class_paths = {
        0: f"{args.dataset_path}/COVID/images",
        1: f"{args.dataset_path}/Normal/images",
        2: f"{args.dataset_path}/Viral Pneumonia/images",
        3: f"{args.dataset_path}/Lung_Opacity/images",
    }
    
    for label, path in class_paths.items():
        if os.path.exists(path):
            d, l = load_images_flat(
                path, label, 
                img_size=tuple(args.img_size), 
                max_images=args.max_samples
            )
            data += d
            labels += l
        else:
            print(f"⚠️  Warning: Path not found - {path}")
    
    if not data:
        print("❌ No data loaded! Check your dataset path.")
        return 1
    
    print(f"✅ Loaded {len(data)} images total")
    
    # Convert to numpy
    X = np.array(data)
    y = np.array(labels)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.validation_split, stratify=y, random_state=42
    )
    
    # Reshape for CNN
    img_h, img_w = args.img_size
    X_train = X_train.reshape(-1, img_h, img_w, 1)
    X_test = X_test.reshape(-1, img_h, img_w, 1)
    
    print(f"📊 Training shape: {X_train.shape}")
    print(f"📊 Test shape: {X_test.shape}")
    
    # Build and train model
    model = build_baseline_cnn(input_shape=(img_h, img_w, 1), num_classes=4)
    print(f"🧠 Model created with {model.count_params():,} parameters")
    
    # Train
    history = model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Save model
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / "covid_model.h5"
    model.save(model_path)
    
    print(f"✅ Model saved to: {model_path}")
    print(f"🎯 Final accuracy: {history.history['accuracy'][-1]:.4f}")
    
    return 0


def predict():
    """CLI command for making predictions"""
    parser = argparse.ArgumentParser(
        description="Make predictions on COVID-19 images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model-path", 
        type=str, 
        required=True,
        help="Path to trained model (.h5 file)"
    )
    parser.add_argument(
        "--image-path", 
        type=str, 
        required=True,
        help="Path to image for prediction"
    )
    parser.add_argument(
        "--img-size", 
        type=int, 
        nargs=2,
        default=[256, 256],
        help="Image size (height width)"
    )
    
    args = parser.parse_args()
    
    # Import dependencies
    import tensorflow as tf
    from PIL import Image
    import numpy as np
    from .features import DEFAULT_CLASS_LABELS
    
    print(f"🔮 Making prediction...")
    print(f"🧠 Model: {args.model_path}")
    print(f"🖼️  Image: {args.image_path}")
    
    # Load model
    try:
        model = tf.keras.models.load_model(args.model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return 1
    
    # Load and preprocess image
    try:
        img = Image.open(args.image_path).convert('L')
        img = img.resize(tuple(args.img_size))
        img_array = np.array(img).astype('float32')
        img_norm = (img_array / 127.5) - 1  # Normalize to [-1, 1]
        img_batch = img_norm.reshape(1, args.img_size[0], args.img_size[1], 1)
        
        print("✅ Image preprocessed successfully")
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return 1
    
    # Make prediction
    try:
        predictions = model.predict(img_batch, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        class_name = DEFAULT_CLASS_LABELS.get(predicted_class, f"Class_{predicted_class}")
        
        print(f"\n🎯 PREDICTION RESULTS:")
        print(f"   Class: {class_name}")
        print(f"   Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        print(f"\n📊 All probabilities:")
        for i, prob in enumerate(predictions[0]):
            label = DEFAULT_CLASS_LABELS.get(i, f"Class_{i}")
            print(f"   {label}: {prob:.4f} ({prob*100:.2f}%)")
        
    except Exception as e:
        print(f"❌ Error making prediction: {e}")
        return 1
    
    return 0


def apply_masks():
    """CLI command for applying masks"""
    parser = argparse.ArgumentParser(
        description="Apply masks to COVID-19 images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--dataset-path", 
        type=str, 
        required=True,
        help="Path to COVID-19 dataset"
    )
    parser.add_argument(
        "--category", 
        type=str, 
        default="COVID",
        choices=["COVID", "Normal", "Lung_Opacity", "Viral Pneumonia"],
        help="Category to process"
    )
    parser.add_argument(
        "--method", 
        type=str, 
        default="overlay",
        choices=["overlay", "multiply", "extract"],
        help="Mask application method"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="./results",
        help="Output directory"
    )
    parser.add_argument(
        "--max-samples", 
        type=int, 
        default=10,
        help="Maximum samples to process"
    )
    parser.add_argument(
        "--alpha", 
        type=float, 
        default=0.5,
        help="Alpha value for overlay method"
    )
    
    args = parser.parse_args()
    
    from .models import MaskApplicator
    from .features import get_image_mask_pairs
    import cv2
    from tqdm import tqdm
    
    print(f"🎭 Applying masks...")
    print(f"📁 Dataset: {args.dataset_path}")
    print(f"📂 Category: {args.category}")
    print(f"🔧 Method: {args.method}")
    print(f"💾 Output: {args.output}")
    
    # Get image/mask pairs
    pairs = get_image_mask_pairs(args.dataset_path, args.category)
    
    if not pairs:
        print(f"❌ No image/mask pairs found for {args.category}")
        return 1
    
    # Limit samples
    pairs = pairs[:args.max_samples]
    print(f"🖼️  Processing {len(pairs)} image pairs...")
    
    # Create output directory
    output_dir = Path(args.output) / f"{args.category}_{args.method}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Apply masks
    mask_applicator = MaskApplicator()
    
    for img_path, mask_path in tqdm(pairs, desc="Applying masks"):
        try:
            result, _, _ = mask_applicator.apply_mask(
                img_path, mask_path, method=args.method, alpha=args.alpha
            )
            
            output_path = output_dir / f"{img_path.stem}_{args.method}.png"
            cv2.imwrite(str(output_path), result)
            
        except Exception as e:
            print(f"❌ Error with {img_path.name}: {e}")
    
    print(f"✅ Results saved to: {output_dir}")
    return 0


def run_streamlit():
    """CLI command for running Streamlit app"""
    import subprocess
    import sys
    
    try:
        # Find streamlit app path
        app_path = Path(__file__).parent / "streamlit" / "app.py"
        
        if not app_path.exists():
            print("❌ Streamlit app not found!")
            return 1
        
        print(f"🚀 Starting Streamlit app...")
        print(f"📱 App path: {app_path}")
        
        # Run streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])
        
    except Exception as e:
        print(f"❌ Error starting Streamlit: {e}")
        return 1
    
    return 0


def setup_colab():
    """Setup function for Google Colab"""
    print("🔧 Setting up DS-COVID for Google Colab...")
    
    # This function can be called from Colab to setup the environment
    import subprocess
    import sys
    
    commands = [
        "pip install opencv-python-headless",
        "pip install matplotlib seaborn tqdm pandas scikit-learn",
        "pip install tensorflow pillow numpy plotly scikit-image"
    ]
    
    for cmd in commands:
        print(f"Installing: {cmd}")
        subprocess.run(cmd.split(), check=True)
    
    print("✅ Colab setup complete!")


if __name__ == "__main__":
    print("DS-COVID CLI - Use specific commands:")
    print("  ds-covid-train     - Train model")
    print("  ds-covid-predict   - Make predictions")
    print("  ds-covid-apply-masks - Apply masks")
    print("  ds-covid-streamlit - Run Streamlit app")