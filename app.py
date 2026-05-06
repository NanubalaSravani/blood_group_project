#!/usr/bin/env python3
"""
🩸 Fingerprint-Based Blood Group Prediction (Academic Prototype)
⚠️  EXPERIMENTAL ONLY - NOT FOR MEDICAL USE

This project combines:
- Training pipeline (load images, preprocess, train model)
- Prediction engine (preprocess and predict blood groups)
- Flask web application (modern UI for predictions)

All in ONE FILE for simplicity!
"""

import os
import sys
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB0

from flask import Flask, render_template_string, request, jsonify
from PIL import Image
import io
from base64 import b64encode

# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
DATASET_PATH = "dataset_blood_group"
MODEL_PATH = "best_model.keras"
ENCODER_PATH = "label_encoder.pkl"
BLOOD_GROUPS = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def preprocess_image(image_path):
    """
    Preprocess image: Load, convert to grayscale, resize, enhance, normalize
    """
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Resize to 224x224
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Gaussian blur (noise reduction)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    
    # Normalize to 0-1
    img = img.astype(np.float32) / 255.0
    
    return img

def preprocess_image_pil(image_bytes):
    """
    Preprocess image from PIL/uploaded file
    """
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    img = cv2.cvtColor(np.array(img), cv2.COLOR_GRAY2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Gaussian blur
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    
    # Normalize
    img = img.astype(np.float32) / 255.0
    
    return img

def load_dataset():
    """
    Load images from dataset_blood_group folders
    Returns: X (images), y (labels)
    """
    print("\n📂 Loading dataset...")
    X, y = [], []
    
    dataset_root = Path(DATASET_PATH)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_PATH}")
    
    for blood_group in BLOOD_GROUPS:
        blood_path = dataset_root / blood_group
        if not blood_path.exists():
            print(f"   ⚠️  {blood_group}: folder not found")
            continue
        
        image_count = 0
        for img_file in blood_path.glob("*"):
            if img_file.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
                try:
                    img = preprocess_image(str(img_file))
                    X.append(img)
                    y.append(blood_group)
                    image_count += 1
                except Exception as e:
                    print(f"   ⚠️  Error loading {img_file}: {e}")
        
        print(f"   ✓ {blood_group}: {image_count} images")
    
    if not X:
        raise ValueError("No images found in dataset!")
    
    X = np.array(X)
    print(f"\n✓ Total images loaded: {len(X)}")
    print(f"✓ Shape: {X.shape}")
    
    return X, np.array(y)

def build_model():
    """
    Build EfficientNetB0-based transfer learning model
    """
    print("\n🧠 Building model...")
    print("   Using EfficientNetB0 transfer learning")
    
    # Load pre-trained EfficientNetB0
    base_model = EfficientNetB0(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model weights
    base_model.trainable = False
    
    # Build custom head
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.Lambda(lambda x: tf.repeat(x, 3, axis=-1)),  # Convert grayscale to RGB
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(len(BLOOD_GROUPS), activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model_pipeline():
    """
    Complete training pipeline
    """
    print("\n" + "="*60)
    print("🔴 BLOOD GROUP PREDICTION - TRAINING PIPELINE")
    print("⚠️  EXPERIMENTAL ACADEMIC PROTOTYPE ONLY")
    print("="*60)
    
    # Load dataset
    X, y = load_dataset()
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = keras.utils.to_categorical(y_encoded, num_classes=len(BLOOD_GROUPS))
    
    # Train-validation-test split (70-15-15)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_categorical, test_size=0.30, random_state=42, stratify=y_encoded
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )
    
    print(f"\n📊 Data Split:")
    print(f"   Training: {len(X_train)} ({100*len(X_train)/(len(X)):.1f}%)")
    print(f"   Validation: {len(X_val)} ({100*len(X_val)/(len(X)):.1f}%)")
    print(f"   Testing: {len(X_test)} ({100*len(X_test)/(len(X)):.1f}%)")
    
    # Data augmentation
    print("\n🔄 Applying data augmentation...")
    datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Build model
    model = build_model()
    print(f"\n✓ Model built")
    print(f"✓ Total parameters: {model.count_params():,}")
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train
    print("\n🚀 Starting training...\n")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Load best model
    model = keras.models.load_model(MODEL_PATH)
    
    # Evaluate
    print("\n📊 Evaluating model...")
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\n✓ Training Accuracy: {train_acc*100:.2f}%")
    print(f"✓ Validation Accuracy: {val_acc*100:.2f}%")
    print(f"✓ Test Accuracy: {test_acc*100:.2f}%")
    
    # Classification report
    y_test_pred = model.predict(X_test, verbose=0)
    y_test_pred_labels = np.argmax(y_test_pred, axis=1)
    y_test_true_labels = np.argmax(y_test, axis=1)
    
    print("\n📊 Classification Report:")
    print(classification_report(
        y_test_true_labels,
        y_test_pred_labels,
        target_names=le.classes_
    ))
    
    # Confusion matrix
    cm = confusion_matrix(y_test_true_labels, y_test_pred_labels)
    print("\n📊 Confusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap='Blues', interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks(range(len(le.classes_)), le.classes_, rotation=45)
    plt.yticks(range(len(le.classes_)), le.classes_)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("✓ Confusion matrix saved to confusion_matrix.png")
    plt.close()
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("✓ Training history saved to training_history.png")
    plt.close()
    
    # Save encoder
    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(le, f)
    print(f"✓ Label encoder saved to {ENCODER_PATH}")
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETED SUCCESSFULLY")
    print("="*60 + "\n")

def predict_image(image_bytes):
    """
    Predict blood group from image bytes
    Returns: blood_group, confidence, top_3, all_scores
    """
    # Load model and encoder
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        return None, None, None, None, "Model not found. Train first: python app.py train"
    
    try:
        model = keras.models.load_model(MODEL_PATH)
        with open(ENCODER_PATH, 'rb') as f:
            le = pickle.load(f)
        
        # Preprocess image
        img = preprocess_image_pil(image_bytes)
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        
        # Predict
        predictions = model.predict(img, verbose=0)[0]
        
        # Get results
        pred_idx = np.argmax(predictions)
        confidence = float(predictions[pred_idx]) * 100
        blood_group = le.classes_[pred_idx]
        
        # Top 3
        top_3_idx = np.argsort(predictions)[::-1][:3]
        top_3 = [(le.classes_[i], float(predictions[i])*100) for i in top_3_idx]
        
        # All scores
        all_scores = {le.classes_[i]: float(predictions[i])*100 for i in range(len(le.classes_))}
        
        return blood_group, confidence, top_3, all_scores, None
    
    except Exception as e:
        return None, None, None, None, str(e)

# ============================================================
# FLASK WEB APPLICATION
# ============================================================

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🩸 Blood Group Prediction</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            max-width: 600px;
            width: 100%;
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px 20px;
            text-align: center;
        }
        
        .header h1 { font-size: 28px; margin-bottom: 5px; }
        .header p { font-size: 14px; opacity: 0.9; }
        
        .warning {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px 20px;
            margin: 20px;
            border-radius: 5px;
            font-size: 14px;
            color: #856404;
        }
        
        .content {
            padding: 30px 20px;
        }
        
        .upload-area {
            border: 2px dashed #667eea;
            border-radius: 10px;
            padding: 40px 20px;
            text-align: center;
            background: #f8f9ff;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .upload-area:hover { background: #eef0ff; border-color: #764ba2; }
        .upload-area.dragover { background: #e8ebff; border-color: #764ba2; }
        
        .upload-area input { display: none; }
        .upload-area p { color: #666; margin-top: 10px; }
        
        .preview {
            display: none;
            margin: 20px 0;
            text-align: center;
        }
        
        .preview img {
            max-width: 100%;
            max-height: 300px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }
        
        button {
            width: 100%;
            padding: 12px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            margin-top: 20px;
            transition: transform 0.2s;
        }
        
        button:hover { transform: scale(1.02); }
        button:disabled { opacity: 0.6; cursor: not-allowed; }
        
        .results {
            display: none;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 2px solid #eee;
        }
        
        .result-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 20px;
        }
        
        .blood-group { font-size: 48px; font-weight: bold; }
        .confidence { font-size: 24px; margin-top: 10px; opacity: 0.9; }
        
        .low-confidence-warning {
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
            padding: 12px;
            border-radius: 5px;
            margin-bottom: 15px;
            display: none;
        }
        
        .score-bars {
            margin-top: 20px;
        }
        
        .score-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .score-label { font-weight: bold; color: #333; width: 50px; }
        .score-bar {
            flex-grow: 1;
            height: 25px;
            background: #e0e0e0;
            border-radius: 5px;
            margin: 0 10px;
            overflow: hidden;
        }
        
        .score-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.5s ease;
        }
        
        .score-percent { color: #333; font-weight: bold; width: 50px; text-align: right; }
        
        .loading { display: none; text-align: center; color: #667eea; }
        .spinner { border: 3px solid #f3f3f3; border-top: 3px solid #667eea; border-radius: 50%; width: 30px; height: 30px; animation: spin 1s linear infinite; margin: 10px auto; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        
        .error { background: #f8d7da; color: #721c24; padding: 12px; border-radius: 5px; margin-top: 10px; }
        
        .disclaimer {
            background: #d1ecf1;
            border: 1px solid #bee5eb;
            color: #0c5460;
            padding: 12px;
            border-radius: 5px;
            margin-top: 20px;
            font-size: 13px;
            line-height: 1.5;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🩸 Blood Group Prediction</h1>
            <p>Fingerprint-Based Academic Prototype</p>
        </div>
        
        <div class="warning">
            ⚠️ <strong>EXPERIMENTAL PROTOTYPE:</strong> This is for academic demonstration only. NOT suitable for medical diagnosis. Always confirm blood group through medical blood testing.
        </div>
        
        <div class="content">
            <div class="upload-area" id="uploadArea">
                <p>📷 <strong>Drag fingerprint image here or click to browse</strong></p>
                <p style="font-size: 12px; margin-top: 5px;">Supported: PNG, JPG, BMP, TIFF (Max 10MB)</p>
                <input type="file" id="fileInput" accept=".png,.jpg,.jpeg,.bmp,.tiff">
            </div>
            
            <div class="preview" id="preview">
                <img id="previewImg" alt="Preview">
                <p style="color: #666; margin-top: 10px; font-size: 14px;" id="fileName"></p>
            </div>
            
            <button id="predictBtn" onclick="predict()">🔮 Predict Blood Group</button>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Analyzing fingerprint...</p>
            </div>
            
            <div class="results" id="results">
                <div class="low-confidence-warning" id="lowConfidenceWarning">
                    ⚠️ <strong>Low Confidence:</strong> Prediction accuracy is below 70%. Results may be unreliable.
                </div>
                
                <div class="result-card">
                    <p style="opacity: 0.9; margin-bottom: 10px;">Predicted Blood Group:</p>
                    <div class="blood-group" id="bloodGroup">-</div>
                    <div class="confidence" id="confidence">-</div>
                </div>
                
                <div id="top3" style="margin-bottom: 20px;"></div>
                
                <p style="font-weight: bold; color: #333; margin-bottom: 10px;">Confidence Scores (All Blood Groups):</p>
                <div class="score-bars" id="scoresBars"></div>
                
                <div class="disclaimer">
                    <strong>Medical Disclaimer:</strong> This prediction is experimental and NOT suitable for medical use. Blood group determination requires laboratory testing by qualified healthcare professionals.
                </div>
            </div>
            
            <div class="error" id="error" style="display: none;"></div>
        </div>
    </div>
    
    <script>
        let selectedFile = null;
        
        // Drag and drop
        const uploadArea = document.getElementById('uploadArea');
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        });
        
        uploadArea.addEventListener('click', () => {
            document.getElementById('fileInput').click();
        });
        
        document.getElementById('fileInput').addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });
        
        function handleFile(file) {
            selectedFile = file;
            const reader = new FileReader();
            reader.onload = (e) => {
                document.getElementById('previewImg').src = e.target.result;
                document.getElementById('preview').style.display = 'block';
                document.getElementById('fileName').textContent = file.name;
                document.getElementById('error').style.display = 'none';
            };
            reader.readAsDataURL(file);
        }
        
        function predict() {
            if (!selectedFile) {
                showError('Please select an image first');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', selectedFile);
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').style.display = 'none';
            document.getElementById('error').style.display = 'none';
            document.getElementById('predictBtn').disabled = true;
            
            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('predictBtn').disabled = false;
                
                if (data.error) {
                    showError(data.error);
                } else {
                    displayResults(data);
                }
            })
            .catch(error => {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('predictBtn').disabled = false;
                showError('Error: ' + error);
            });
        }
        
        function displayResults(data) {
            document.getElementById('bloodGroup').textContent = data.blood_group;
            document.getElementById('confidence').textContent = data.confidence.toFixed(1) + '% Confidence';
            
            // Low confidence warning
            if (data.confidence < 70) {
                document.getElementById('lowConfidenceWarning').style.display = 'block';
            } else {
                document.getElementById('lowConfidenceWarning').style.display = 'none';
            }
            
            // Top 3
            let top3Html = '<p style="font-weight: bold; color: #333; margin-bottom: 10px;">Top 3 Predictions:</p>';
            data.top_3.forEach((item, idx) => {
                top3Html += `<p style="color: #666; margin: 5px 0;"><strong>${idx+1}. ${item[0]}</strong>: ${item[1].toFixed(1)}%</p>`;
            });
            document.getElementById('top3').innerHTML = top3Html;
            
            // Score bars
            let barsHtml = '';
            data.all_scores.forEach(item => {
                const bg = item[0] === data.blood_group ? '#667eea' : '#ccc';
                barsHtml += `
                    <div class="score-item">
                        <span class="score-label">${item[0]}</span>
                        <div class="score-bar">
                            <div class="score-fill" style="width: ${item[1]}%; background: ${bg};"></div>
                        </div>
                        <span class="score-percent">${item[1].toFixed(1)}%</span>
                    </div>
                `;
            });
            document.getElementById('scoresBars').innerHTML = barsHtml;
            
            document.getElementById('results').style.display = 'block';
        }
        
        function showError(msg) {
            document.getElementById('error').textContent = msg;
            document.getElementById('error').style.display = 'block';
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file type
        allowed = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        if not any(file.filename.lower().endswith(ext) for ext in allowed):
            return jsonify({'error': 'Invalid file format. Use PNG, JPG, BMP, or TIFF'}), 400
        
        # Read file
        image_bytes = file.read()
        if not image_bytes:
            return jsonify({'error': 'Empty file'}), 400
        
        # Predict
        blood_group, confidence, top_3, all_scores, error = predict_image(image_bytes)
        
        if error:
            return jsonify({'error': error}), 400
        
        # Format response
        response = {
            'blood_group': blood_group,
            'confidence': confidence,
            'top_3': top_3,
            'all_scores': [[bg, score] for bg, score in all_scores.items()]
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌐 FLASK WEB APP STARTING")
    print("="*60)
    
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        # Training mode
        train_model_pipeline()
    else:
        # Web app mode
        print("📍 Open browser: http://localhost:5000")
        print("⚠️  Remember: This is an experimental academic prototype")
        print("="*60 + "\n")
        app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
