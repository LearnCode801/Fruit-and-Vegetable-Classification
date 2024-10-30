---

# Fruit and Vegetable Classification

## Project Overview
This project classifies images of 36 different fruits and vegetables using a Convolutional Neural Network (CNN) model. It utilizes data augmentation, training-validation splitting, and visualization to optimize model performance, aiming for practical applications in inventory or grocery management.
###
![IMG](https://github.com/LearnCode801/Fruit-and-Vegetable-Classification/blob/main/Screenshot%202024-10-30%20165244.png)
## Dataset
- **Images**: ~3,861 labeled images of 36 classes (fruits and vegetables).
- **Structure**:
  - **train/**: 100 images per class
  - **test/**: 10 images per class
  - **validation/**: 10 images per class

## Table of Contents
1. Loading and Preprocessing
2. Data Augmentation
3. Model Training (CNN)
4. Result Visualization
5. Class Activation Heatmap
6. Model Deployment

## Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd fruit-vegetable-classification
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Load Data**: Load and preprocess data by running the initial cells.
2. **Train Model**: Execute the model training cell.
3. **Visualize Results**: Review accuracy/loss plots and class activation heatmaps.
4. **Deploy Model**: Follow deployment instructions for real-time classification.

## Model Architecture
The project employs a Convolutional Neural Network (CNN) with the following architecture:
- **Input Layer**: Accepts images of fixed size (e.g., 128x128x3).
- **Convolutional Layers**: Series of Conv2D and MaxPooling layers to extract spatial features.
- **Fully Connected Layers**: Dense layers with dropout to prevent overfitting.
- **Output Layer**: Softmax activation for multi-class classification (36 classes).

### Data Augmentation
Data augmentation includes transformations such as rotation, zoom, and flip to improve model robustness.

### Training Configuration
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Epochs**: Typically 20-30, based on validation performance
- **Batch Size**: 32 (adjustable)

## Results

### Training and Validation Accuracy
The following graph illustrates model accuracy over training epochs, showcasing generalization on the validation set:

![Training and Validation Accuracy](training_validation_accuracy.png)

### Loss
A similar loss plot (training and validation) helps visualize convergence.

### Confusion Matrix
The confusion matrix provides insights into model performance across all classes, highlighting accurate classifications and areas for improvement.

### Sample Predictions
Sample predictions include both true and predicted labels, allowing for qualitative evaluation.

### Class Activation Heatmap
The heatmap visualizes image areas that influenced the model’s classification, improving interpretability.

## Deployment
Use the deployment cell for real-time model classification, suitable for production integration.

---
