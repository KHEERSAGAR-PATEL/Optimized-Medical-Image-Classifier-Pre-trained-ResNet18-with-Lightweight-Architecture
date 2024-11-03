# Optimized-Medical-Image-Classifier-Pre-trained-ResNet18-with-Lightweight-Architecture
Developed a medical image classifier using a pre-trained ResNet18 for feature extraction and a custom lightweight neural network for binary classification. This approach achieves efficient, accurate classification with low computational cost, utilizing AdamW optimization and cross-entropy loss. Evaluated using Accuracy and ROC AUC.
Here's an enhanced, structured version for a GitHub README that emphasizes clarity and organization for a professional presentation:

---


## Key Components

### Model Architecture
- **ResNet18 (Pre-trained)**: Utilized a ResNet18 model trained on ImageNet for feature extraction. The final fully connected layers were removed, allowing the network to output feature maps that capture complex spatial hierarchies in medical images.
- **Lightweight Neural Network**: A custom, fully connected lightweight network with ReLU activations processes the extracted features for binary classification.

### Training Process
- **Frozen Feature Extractor**: ResNet18’s parameters are frozen, serving only as a feature extractor while fine-tuning occurs on the lightweight classification head.
- **Optimization**: The AdamW optimizer, with weight decay, was used to prevent overfitting and enable smooth convergence.
- **Loss Function**: Cross-Entropy Loss was employed, appropriate for binary classification tasks, to minimize the error between predicted probabilities and actual labels.

### Evaluation Metrics
- **Accuracy**: To assess overall correctness of the model’s classifications.
- **ROC AUC (Area Under the Curve)**: Provides insight into the model’s capacity to distinguish between classes, critical for medical diagnostics.
- **Test Set Validation**: Batch processing was facilitated using PyTorch’s `DataLoader`, ensuring efficient handling of large test datasets.

### Data Preprocessing
- **Image Transformation**: Using PyTorch’s `transforms` utilities, images were resized to 224x224, normalized to ImageNet statistics, and converted to tensor format.
- **Dataset Handling**: `ImageFolder` was used to load and preprocess images, simplifying data preparation for training and evaluation.

## Getting Started

### Prerequisites
- Python 3.x
- PyTorch
- Torchvision
- Other dependencies listed in `requirements.txt`

## Results
This model achieves high accuracy and ROC AUC scores, validating its effectiveness for binary classification of medical images.
