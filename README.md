# Fruit and Vegetable Classification

A deep learning project for automated classification of 36 different types of fruits and vegetables using computer vision and transfer learning techniques. This system achieves 94.31% accuracy on test data and includes advanced visualization features for model interpretability.

## 🎬 Project Demo Video

[![Watch Demo Video](https://img.shields.io/badge/Watch-Demo%20Video-red?style=for-the-badge&logo=youtube)](https://drive.google.com/file/d/1wx3V4cPXXMXwgf7M_WhylRdPiJNWLFQz/view?usp=sharing)

## 📓 Project Notebook

[![View Notebook](https://img.shields.io/badge/View-Jupyter%20Notebook-orange?style=for-the-badge&logo=jupyter)](https://github.com/LearnCode801/Fruit-and-Vegetable-Classification/blob/main/fruit-and-vegetable-classification.ipynb)

## 📊 Project Overview

This project addresses the challenge of automated food recognition by developing a robust classification system capable of identifying 36 different fruits and vegetables from images. The solution leverages transfer learning with MobileNetV2 and advanced data augmentation techniques to achieve high accuracy while maintaining computational efficiency.

## 🎯 Key Features

- **High Accuracy**: 94.31% classification accuracy on test dataset
- **Transfer Learning**: MobileNetV2 pre-trained on ImageNet for feature extraction  
- **Data Augmentation**: Comprehensive augmentation pipeline for robustness
- **Model Interpretability**: Grad-CAM visualization for decision explanation
- **36 Categories**: Complete classification across fruits and vegetables
- **Production Ready**: Saved model with clean inference pipeline

## 📈 Dataset Information

**Source**: Fruit and Vegetable Image Recognition Dataset  
**Total Images**: 3,861 images  
**Categories**: 36 different fruits and vegetables  
**Image Distribution**:
- Training: 2,780 images (100 per category)
- Testing: 334 images (10 per category)  
- Validation: 334 images (10 per category)

### Categories Included:

**Fruits (10)**:
banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango

**Vegetables (26)**:
cucumber, carrot, capsicum, onion, potato, lemon, tomato, raddish, beetroot, cabbage, lettuce, spinach, soy beans, cauliflower, bell pepper, chilli pepper, turnip, corn, sweetcorn, sweet potato, paprika, jalapeño, ginger, garlic, peas, eggplant

## 🔧 Step-by-Step Problem Approach

### Problem Statement
Develop an automated system to classify 36 different types of fruits and vegetables from images with high accuracy, addressing the challenge of food recognition in real-world applications.

### Step 1: Data Understanding and Preparation
```python
# Dataset Structure Analysis
- Total Images: 3,861 images
- Categories: 36 different fruits and vegetables
- Distribution:
  * Training: 2,780 images (100 per category)
  * Testing: 334 images (10 per category) 
  * Validation: 334 images (10 per category)
```

**Data Organization:**
- Created file path lists for train/test/validation sets
- Generated DataFrame structure with filepath and label columns
- Implemented data shuffling for better training

### Step 2: Exploratory Data Analysis
```python
def proc_img(filepath):
    """Process image paths and extract labels"""
    labels = [str(filepath[i]).split("/")[-2] for i in range(len(filepath))]
    filepath = pd.Series(filepath, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')
    df = pd.concat([filepath, labels], axis=1)
    return df.sample(frac=1).reset_index(drop=True)
```

**Key Findings:**
- 36 unique categories identified
- Balanced dataset with equal samples per class
- Visual inspection showed diverse image quality and backgrounds

### Step 3: Data Preprocessing and Augmentation
```python
# Image Data Generators with Augmentation
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)
```

**Augmentation Strategy:**
- **Rotation**: ±30 degrees for orientation invariance
- **Zoom**: 15% range for scale variation
- **Shifts**: 20% width/height for position robustness
- **Horizontal Flipping**: For symmetry handling
- **Shear Transformation**: For perspective variation

### Step 4: Model Architecture Selection
```python
# Transfer Learning with MobileNetV2
pretrained_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)
pretrained_model.trainable = False
```

**Architecture Design:**
- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Input Shape**: 224×224×3 RGB images
- **Feature Extraction**: Frozen pretrained weights
- **Custom Head**: Two Dense layers (128 neurons each) + output layer

### Step 5: Model Construction and Compilation
```python
# Custom Classification Head
inputs = pretrained_model.input
x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
x = tf.keras.layers.Dense(128, activation='relu')(x)
outputs = tf.keras.layers.Dense(36, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

**Design Rationale:**
- **Transfer Learning**: Leverage ImageNet features
- **Dense Layers**: 128-128-36 configuration for feature refinement
- **Activation**: ReLU for hidden layers, Softmax for classification
- **Optimizer**: Adam for adaptive learning rate

### Step 6: Training Strategy and Execution
```python
# Training Configuration
history = model.fit(
    train_images,
    validation_data=val_images,
    batch_size=32,
    epochs=5,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=2,
            restore_best_weights=True
        )
    ]
)
```

**Training Results:**
- **Epochs**: 5 (with early stopping)
- **Final Training Accuracy**: 95.76%
- **Validation Accuracy**: 94.31%
- **Loss Reduction**: From 2.60 to 0.15

### Step 7: Model Evaluation and Testing
```python
# Comprehensive Evaluation
pred = model.predict(test_images)
pred = np.argmax(pred, axis=1)
test_accuracy = accuracy_score(y_test, pred)
print(f'Test Accuracy: {100*test_accuracy:.2f}%')
```

**Performance Metrics:**
- **Test Accuracy**: 94.31%
- **Confusion Matrix**: Normalized visualization across all classes
- **Class-wise Performance**: Consistent across categories

### Step 8: Model Interpretability Analysis
```python
# Grad-CAM Visualization Implementation
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, pred_index]
    
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    heatmap = last_conv_layer_output[0] @ pooled_grads[..., tf.newaxis]
    return tf.squeeze(heatmap)
```

**Interpretability Features:**
- **Grad-CAM Heatmaps**: Visual explanation of model focus areas
- **Class Activation Maps**: Understanding decision-making process
- **Feature Visualization**: Highlighting important image regions

### Step 9: Model Deployment Preparation
```python
# Model Persistence
model.save('FruitModel.h5')

# Prediction Function
def output(location):
    img = load_img(location, target_size=(224,224,3))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, [0])
    
    prediction = model.predict(img)
    class_idx = prediction.argmax(axis=-1)[0]
    return labels[class_idx]
```

**Deployment Ready:**
- **Model Format**: Saved as HDF5 file
- **Inference Function**: Simple prediction interface
- **Preprocessing Pipeline**: Consistent with training

### Step 10: Results Analysis and Validation

**Quantitative Results:**
- **Overall Accuracy**: 94.31% on test set
- **Training Convergence**: Achieved in 5 epochs
- **Validation Performance**: Consistent with training accuracy
- **Class Coverage**: All 36 categories successfully classified

**Qualitative Analysis:**
- **Visual Predictions**: Sample predictions match ground truth
- **Grad-CAM Analysis**: Model focuses on relevant fruit/vegetable regions
- **Error Analysis**: Misclassifications mainly on visually similar items


## 📊 Model Performance

### Training History
| Epoch | Training Accuracy | Training Loss | Validation Accuracy | Validation Loss |
|-------|------------------|---------------|-------------------|-----------------|
| 1     | 35.07%          | 2.5997        | 87.13%           | 0.4605         |
| 2     | 81.63%          | 0.5901        | 91.92%           | 0.3033         |
| 3     | 88.78%          | 0.3389        | 90.72%           | 0.2562         |
| 4     | 92.18%          | 0.2461        | 96.11%           | 0.2050         |
| 5     | 95.76%          | 0.1470        | 94.31%           | 0.2043         |

### Model Architecture Summary
```
Model: MobileNetV2 + Custom Classification Head
├── Input Layer: (224, 224, 3)
├── MobileNetV2 Base: Pre-trained on ImageNet (frozen)
├── Global Average Pooling: 1280 features
├── Dense Layer 1: 128 neurons, ReLU activation
├── Dense Layer 2: 128 neurons, ReLU activation
└── Output Layer: 36 neurons, Softmax activation

Total Parameters: ~3.5M
Trainable Parameters: ~70K
```

## 📊 Advanced Visualizations

### Grad-CAM Implementation
```python
def visualize_gradcam(model, image_path, class_index=None):
    """Generate Grad-CAM heatmap for model interpretability"""
    
    # Preprocess image
    img_array = preprocess_input(get_img_array(image_path, size=(224, 224)))
    
    # Generate heatmap
    heatmap = make_gradcam_heatmap(img_array, model, "Conv_1", class_index)
    
    # Overlay heatmap on original image
    cam_path = save_and_display_gradcam(image_path, heatmap)
    
    return cam_path
```

### Confusion Matrix Analysis
The normalized confusion matrix shows strong diagonal performance with minimal off-diagonal confusion, indicating robust classification across all 36 categories.

## 🎯 Applications & Use Cases

### Food Industry Applications
- **Automated Sorting**: Industrial fruit/vegetable sorting systems
- **Quality Control**: Automated quality assessment in processing plants
- **Retail Systems**: Self-checkout assistance in grocery stores
- **Inventory Management**: Automated stock counting and categorization

### Consumer Applications
- **Mobile Apps**: Food identification and nutritional information
- **Smart Kitchen**: Recipe suggestions based on available ingredients
- **Educational Tools**: Interactive learning about fruits and vegetables
- **Health Tracking**: Dietary monitoring and nutrition analysis

### Research Applications
- **Agricultural Research**: Crop monitoring and yield assessment
- **Nutritional Studies**: Automated food diary categorization
- **Computer Vision Research**: Benchmark for classification algorithms
- **Transfer Learning**: Base model for specialized food classification

## 📊 Technical Achievements

1. **High Accuracy**: Achieved 94.31% test accuracy with efficient architecture
2. **Transfer Learning**: Successfully leveraged pre-trained ImageNet features
3. **Data Augmentation**: Comprehensive augmentation pipeline for robustness
4. **Model Interpretability**: Implemented Grad-CAM for explainable AI
5. **Balanced Performance**: Consistent accuracy across all 36 categories
6. **Efficient Training**: Converged in just 5 epochs with early stopping
7. **Production Ready**: Clean inference pipeline suitable for deployment

## 🐛 Troubleshooting

### Common Issues
1. **Low Accuracy on Custom Images**:
   - Ensure images are properly preprocessed (224x224, normalized)
   - Check image quality and lighting conditions
   - Verify the image contains the target fruit/vegetable clearly

2. **Memory Issues During Training**:
   - Reduce batch size from 32 to 16 or 8
   - Use mixed precision training
   - Consider using data generators with lower resolution

3. **Model Loading Errors**:
   - Ensure TensorFlow version compatibility
   - Verify model file integrity
   - Check custom object dependencies

---

**Note**: This model is trained on a specific dataset and may require retraining or fine-tuning for optimal performance on different image distributions or lighting conditions.
