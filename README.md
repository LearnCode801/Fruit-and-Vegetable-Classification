---

# Fruit and Vegetable Classification
## 🎬 Project Demo Video

[![Watch Demo Video](https://drive.google.com/file/d/1wx3V4cPXXMXwgf7M_WhylRdPiJNWLFQz/view?usp=sharing)

## Project Overview
This project utilizes a Convolutional Neural Network (CNN) to classify images of 36 different fruits and vegetables. The model architecture is supported by MobileNetV2 preprocessing, along with robust data augmentation techniques to enhance the model's generalizability.
![IMG](https://github.com/LearnCode801/Fruit-and-Vegetable-Classification/blob/main/Screenshot%202024-10-30%20165244.png)
## Dataset
- **Images**: ~3,861 labeled images for training, validation, and testing.
- **Structure**:
  - **train/**: 100 images per class
  - **test/**: 10 images per class
  - **validation/**: 10 images per class

## Techniques

### Data Generators and Augmentation
The model utilizes `ImageDataGenerator` to create data pipelines for training, validation, and testing. The training and validation generators apply **MobileNetV2 preprocessing** to normalize images to the input scale required by the model. Additionally, data augmentation techniques, such as **rotation, zoom, width and height shift, shear, and horizontal flip**, are used to improve the model's robustness by exposing it to variations in data.

Key augmentation parameters:
- **Rotation Range**: Up to 30 degrees
- **Zoom Range**: Up to 15%
- **Shift Range**: Up to 20% for width and height
- **Shear Range**: Up to 15%
- **Horizontal Flip**: Random horizontal flips

These augmentations help the model generalize better by preventing overfitting and making it adaptable to variations in image orientations and sizes.

### Model Architecture
This project uses a CNN optimized for image classification with the following architecture:
- **Convolutional Layers**: Extract spatial features from input images.
- **Pooling Layers**: Reduce spatial dimensions, helping with computational efficiency.
- **Fully Connected Layers**: For higher-level feature learning.
- **Softmax Output Layer**: For multi-class classification (36 classes).

### Training Configuration
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Batch Size**: 32
- **Image Size**: Resized to 224x224 pixels for MobileNetV2 compatibility

## Results

### Accuracy and Loss Graphs
The graphs below show the training and validation accuracy, as well as the loss across epochs:

1. **Accuracy Graph**: This graph shows that both training and validation accuracy improve steadily over epochs, reaching over 90% by the end of training.
   
![IMG](https://github.com/LearnCode801/Fruit-and-Vegetable-Classification/blob/main/result%201.png)

2. **Loss Graph**: The loss graph indicates a consistent decrease in both training and validation loss, demonstrating good convergence and minimal overfitting.
   
![IMG](https://github.com/LearnCode801/Fruit-and-Vegetable-Classification/blob/main/result%202.png)

The model achieved high accuracy on the validation set, indicating strong performance on unseen data. 

### Confusion Matrix (Optional)
The notebook may include a confusion matrix to visualize the classification performance per category, revealing insights into which classes are classified accurately and where misclassifications may occur.

### Class Activation Heatmap
Class activation heatmaps are generated to highlight areas in the images that influence the model’s decision, aiding interpretability and providing insights into the model’s focus during classification.

## Deployment
Instructions for deploying the trained model for real-time classification are provided in the notebook, allowing users to apply the model in practical applications.

--- 
