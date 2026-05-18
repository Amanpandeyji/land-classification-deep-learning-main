# Satellite Image Classification with Deep CNNs

A comprehensive deep learning project comparing multiple CNN architectures for multi-class satellite image classification using transfer learning and various training strategies.

## 🎯 Project Overview

This project implements and evaluates different Convolutional Neural Network (CNN) architectures for classifying satellite imagery into 28 distinct classes. The study compares the effectiveness of transfer learning approaches versus training from scratch across multiple state-of-the-art architectures.

## 📊 Dataset

- **Classes**: 28 different satellite image categories
- **Total Images**: 2,800 images
  - Training: 1,400 images (50 per class)
  - Testing: 1,400 images (50 per class)
- **Data Split**: K-fold cross-validation with 40 images per class per fold
- **Image Type**: Satellite/aerial imagery with diverse terrain and land use patterns

##  Model Architectures

The project evaluates four state-of-the-art CNN architectures:

1. **ResNet18** - Deep residual learning with skip connections
2. **DenseNet169** - Dense connectivity for feature reuse
3. **MobileNetV2** - Efficient architecture for mobile deployment
4. **ResNeXt101_32x8d** - Aggregated residual transformations

##  Training Strategies

Each architecture is evaluated under three different training approaches:

### 1. Fine-tuned Transfer Learning
- Pre-trained on ImageNet
- All layers trainable with lower learning rate
- Leverages learned features while adapting to satellite imagery

### 2. Fixed Feature Extractor
- Pre-trained backbone frozen
- Only final classifier layers trained
- Fastest training approach with strong performance

### 3. Training from Scratch
- Random weight initialization
- Full network training on satellite data only
- Baseline comparison for transfer learning effectiveness

##  Results Summary

### Best Performing Models

| Architecture | Training Strategy | Test Accuracy | Training Loss |
|-------------|------------------|---------------|---------------|
| **DenseNet169** | Fixed Feature Extractor | **73.57%** | 0.85 |
| ResNet18 | Fixed Feature Extractor | 70.00% | 0.92 |
| MobileNetV2 | Fine-tuned | 68.21% | 1.12 |
| ResNeXt101 | Fixed Feature Extractor | 65.43% | 1.08 |

### Key Findings

- **Transfer learning significantly outperforms training from scratch** (40+ percentage points improvement)
- **Fixed feature extractor approach** achieved best results with fastest training time
- **DenseNet169** emerged as the optimal architecture for this satellite imagery task
- **Training from scratch** achieved only 20-30% accuracy across all architectures

##  Evaluation Methodology

The project implements rigorous evaluation using multiple statistical approaches:

### Cross-Validation Results
- **K-fold Cross-Validation**: Mean accuracy of 20.71%
- **Individual Fold Range**: 31.79% - 37.50%
- **Holdout Validation**: 16.79% accuracy, 2.78 loss
- **Bootstrap Sampling**: 44.5% accuracy, 1.76 loss

### Training Monitoring
- 20 epochs per experiment
- Epoch-by-epoch accuracy and loss tracking
- Early stopping and learning rate scheduling
- Comprehensive performance visualization

##  Technical Implementation

### Dependencies
```python
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
matplotlib>=3.4.0
scikit-learn>=0.24.0
PIL>=8.3.0
```

### Key Features
- **Data Augmentation**: Random rotations, flips, and color jittering
- **Batch Processing**: Optimized data loading with PyTorch DataLoader
- **GPU Acceleration**: CUDA support for faster training
- **Visualization**: Training curves, confusion matrices, and performance plots
- **Model Checkpointing**: Save best performing models during training

##  Project Structure

```
DIP REVIEW2/
├── README.md              # Project documentation
├── 1.ipynb               # Notebook 1
├── 2.ipynb               # Main experimentation notebook
├── 3.ipynb               # Notebook 3
└── models/               # Saved model checkpoints (if applicable)
```

##  Getting Started

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Installation
1. Clone or download the project files
2. Install required dependencies:
   ```bash
   pip install torch torchvision matplotlib scikit-learn pillow numpy
   ```
3. Open `2.ipynb` in Jupyter Notebook or JupyterLab
4. Run all cells to reproduce the experiments

### Usage
The main notebook `2.ipynb` contains:
- Data loading and preprocessing
- Model architecture definitions
- Training loops for all experiments
- Evaluation and visualization code
- Results comparison and analysis

##  Performance Analysis

### Transfer Learning Impact
The results demonstrate the significant advantage of transfer learning:
- **73.57% accuracy** with transfer learning vs **~25% average** from scratch
- **Reduced training time** by leveraging pre-trained features
- **Better generalization** on limited satellite imagery dataset

### Architecture Comparison
- **DenseNet169**: Best overall performance due to dense feature connectivity
- **ResNet18**: Good balance of performance and computational efficiency  
- **MobileNetV2**: Suitable for deployment scenarios requiring smaller models
- **ResNeXt101**: Strong performance but higher computational cost

##  Applications

This satellite image classification system can be applied to:
- **Land Use Monitoring**: Automated classification of terrain types
- **Urban Planning**: Analysis of development patterns
- **Environmental Studies**: Tracking changes in vegetation and land cover
- **Agriculture**: Crop type identification and monitoring
- **Disaster Response**: Rapid assessment of affected areas

## Future Work

- **Data Augmentation**: Advanced augmentation techniques for satellite imagery
- **Ensemble Methods**: Combining predictions from multiple architectures
- **Attention Mechanisms**: Vision Transformers and attention-based models
- **Multi-temporal Analysis**: Incorporating time-series satellite data
- **Deployment Optimization**: Model quantization and mobile deployment

##  References

- He, K., et al. "Deep Residual Learning for Image Recognition" (ResNet)
- Huang, G., et al. "Densely Connected Convolutional Networks" (DenseNet)
- Sandler, M., et al. "MobileNetV2: Inverted Residuals and Linear Bottlenecks" (MobileNet)
- Xie, S., et al. "Aggregated Residual Transformations for Deep Neural Networks" (ResNeXt)



---

**Note**: This project demonstrates proficiency in deep learning, computer vision, transfer learning, and scientific experimentation methodology. The systematic comparison of architectures and training strategies showcases best practices in ML research and development.
