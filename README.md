# 🩸 Fingerprint-Based Blood Group Prediction (Academic Prototype)

⚠️ **MEDICAL DISCLAIMER**: This is an **experimental academic prototype ONLY**. Fingerprint patterns show weak statistical associations with blood groups and this prediction is **NOT suitable for medical use**. Always confirm blood group through medical blood testing.

---

## 📦 Project: 3 Files Only

```
blood_group_project/
├── app.py              # Complete app (training + web + prediction)
├── requirements.txt    # Dependencies
├── README.md          # This file
└── dataset_blood_group/  # Create this folder
    ├── A+/ ├── A-/ ├── B+/ ├── B-/
    ├── AB+/ ├── AB-/ ├── O+/ └── O-/
```

---

## 🚀 Quick Start

```bash
# 1. Virtual environment
python -m venv venv
venv\Scripts\activate

# 2. Install
pip install -r requirements.txt

# 3. Create dataset folders
mkdir -p dataset_blood_group/{A+,A-,B+,B-,AB+,AB-,O+,O-}

# 4. Add fingerprint images to each folder (50+ per group)

# 5. Train model
python app.py train

# 6. Run web app
python app.py
# Open: http://localhost:5000
```

---

## ✨ `app.py` Features (Everything in One File!)

### Training Pipeline
- ✅ Load fingerprint images from dataset folders
- ✅ Convert to grayscale & resize to 224×224
- ✅ Gaussian blur + CLAHE enhancement
- ✅ Data split: 70% train, 15% val, 15% test
- ✅ Data augmentation: rotation, zoom, shift, brightness
- ✅ EfficientNetB0 transfer learning
- ✅ Early stopping + checkpointing + LR reduction
- ✅ Classification report + confusion matrix

### Prediction Engine
- ✅ Image preprocessing (identical to training)
- ✅ Blood group prediction with confidence
- ✅ Top 3 predictions
- ✅ Confidence scores for all 8 blood groups
- ✅ Low confidence warnings (<70%)

### Flask Web App
- ✅ Modern, responsive UI (embedded HTML/CSS/JS)
- ✅ Drag-and-drop file upload
- ✅ Real-time image preview
- ✅ Beautiful result display
- ✅ Confidence bars for all blood groups
- ✅ Medical disclaimers (4 levels)
- ✅ Error handling

---

## 📊 Expected Output

### Training
```
✓ Training Accuracy: 85-90%
✓ Validation Accuracy: 80-85%
✓ Test Accuracy: 75-82%
✓ Confusion matrix saved
✓ Training history saved
```

### Web Interface
- Predicted blood group (large)
- Confidence percentage
- ⚠️ Warning if confidence <70%
- Top 3 predictions
- Confidence bars for all 8 groups
- Medical disclaimer

---

## 🔧 Troubleshooting

### "Model not found"
```bash
python app.py train  # Train first
python app.py        # Then run app
```

### "No images found"
- Create: `dataset_blood_group/A+/`, `dataset_blood_group/A-/`, etc.
- Add fingerprint images to each folder

### Import errors
```bash
pip install --upgrade tensorflow opencv-python keras numpy scikit-learn
```

### Low accuracy
- Add more images (100+ per group)
- Use clearer fingerprint images
- Increase training epochs

### OutOfMemory
- Reduce batch size in app.py (line ~350)
- Or use MobileNetV2 instead of EfficientNetB0

### Port 5000 in use
```bash
# Change in app.py last line:
app.run(port=5001)
```

---

## ⚠️ Important Notes

❌ **NOT a medical diagnostic tool**  
❌ **Fingerprints don't determine blood groups**  
✅ **For academic/research demonstration only**  
✅ **Use medical blood testing for real results**  

---

## 🎓 Learn About

- Transfer Learning
- Image Preprocessing
- Data Augmentation
- Deep Learning (TensorFlow/Keras)
- Flask Web Development
- Model Evaluation
- Ethical AI

---

**Happy Learning!** 🎓
