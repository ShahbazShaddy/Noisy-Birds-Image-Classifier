# Noisy Birds Classification Project

## Overview
This project addresses an image classification challenge involving four categories: budgie, rubber duck, canary, and duckling. The dataset includes both labeled and unlabeled images, with noise added to all samples, making traditional noise removal techniques ineffective. The goal is to develop a robust model that leverages semi-supervised learning to utilize both labeled and unlabeled data, achieving high classification accuracy while keeping the model size under 70 MB.

## Dataset
- **Source**: Hugging Face dataset (`RayanAi/Noisy_birds`)
- **Classes**: Budgie, Rubber Duck, Canary, Duckling
- **Total Images**: 1,358 (approximately 40 labeled images per class, remaining unlabeled)
- **Image Size**: 128x128 pixels
- **Structure**: Labeled images are stored in respective class folders; unlabeled images are in an "unlabeled" folder.
- **Challenge**: Noise is present in all images, and the test set is inaccessible, preventing noise removal preprocessing.

## Project Structure
- **rayyan-q1-v2.ipynb**: Main Jupyter notebook containing the implementation.
- **Noisy_birds/**: Directory for the downloaded dataset.
- **best_model.pth**: Saved weights of the best-performing model (DenseNet121).

## Methodology
- **Data Preprocessing**:
  - Applied data augmentation (RandomHorizontalFlip, RandomRotation, RandomResizedCrop, ColorJitter) to enhance training data robustness.
  - Normalized images using ImageNet mean and standard deviation (`mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`).
  - Split labeled dataset into 60% training and 40% validation sets.
- **Model Selection**:
  - Evaluated multiple pretrained CNN architectures: DenseNet121, ResNet50, EfficientNet-B0, MobileNetV3-Large, and ResNet18.
  - Used transfer learning by fine-tuning the final layers for 4-class classification.
- **Training**:
  - Employed Adam optimizer with an initial learning rate of 0.001.
  - Used CrossEntropyLoss and ReduceLROnPlateau scheduler to optimize training.
  - Trained for 30 epochs, saving the model with the lowest validation loss.
- **Evaluation**:
  - Metrics: F1 score (macro), accuracy, and AUC (one-vs-rest).
  - Best model (DenseNet121) results: F1 Score: 0.7535, Accuracy: 0.7500, AUC: 0.9216.

## Requirements
- Python 3.11
- PyTorch
- Torchvision
- Hugging Face Hub
- PIL (Pillow)
- NumPy
- Scikit-learn
- Matplotlib

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd noisy-birds-classification
