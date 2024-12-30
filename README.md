# Ocular Disease Prediction

## Introduction

This project explores the prediction of ocular diseases using advanced machine learning techniques. By implementing models like ResNet and GoogLeNet, we improved our accuracy and conducted thorough research into scholarly documentation and citation standards.

### Key Results:
- Improved accuracy from 63% to **64.5%** using Microsoft ResNet.
- Benchmarked with Google's GoogLeNet, achieving **62%** accuracy.
- Drastically reduced computational time by leveraging GPU resources.

---

## Dataset Description

### ODIR-5K Dataset

"The Ocular Disease Intelligent Recognition (ODIR-5K) dataset is a comprehensive collection of retinal fundus images designed to facilitate the development and evaluation of automated systems for ocular disease diagnosis." – [Kaggle](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k)

**Key Characteristics:**
- **Patients:** 3196
- **Images:** 6392 fundus photographs (left and right eyes)
- **Annotations:** Eight diagnostic categories:
  - Normal (N)
  - Diabetes (D)
  - Glaucoma (G)
  - Cataract (C)
  - Age-related Macular Degeneration (AMD)
  - Hypertension (H)
  - Myopia (M)
  - Other diseases/abnormalities (O)

The dataset simulates real-world multi-label classification scenarios where patients may present with multiple concurrent ocular conditions.

**Source:** Developed by Peking University and released during the "Intelligent Eye" competition in 2019.

---

## Methodology

### Models Used:

1. **ResNet:**
   - Validation Accuracy: **65%**
   - Efficient feature extraction for embedding analysis.

2. **GoogLeNet:**
   - Validation Accuracy: **59%**
   - Multi-scale feature extraction using "Inception" modules.

### Dataset Splits:
- Part A: **50% of the dataset**, used for tuning models.
- Part B: **50% of the dataset**, used for final evaluation.

### Statistical Metrics:
To address class imbalance, we prioritized:
- **F1-Score**
- **Balanced Accuracy**
- **AUROC**

---

## Results

### ResNet Performance:
| Metric    | Accuracy | AUROC  | F1-Score | Balanced Accuracy |
|-----------|----------|--------|----------|-------------------|
| ResNet50  | 64.5%    | 85.6%  | 52.7%    | 50.4%            |

### GoogLeNet Performance:
| Metric    | Accuracy | AUROC  | F1-Score | Balanced Accuracy |
|-----------|----------|--------|----------|-------------------|
| GoogLeNet | 62%      | 83.6%  | 50.2%    | 49.3%            |

---

## Files Submitted

### Report and Dataset:
- **Phase_3_Advanced_Report.pdf**: Detailed report containing all project details.
- **Data_used.csv**: Cleaned dataset for training and testing.

### Python Scripts:
- `extract_embeddings_resnet.py`: Extract embeddings using ResNet.
- `extract_embeddings_googlenet.py`: Extract embeddings using GoogLeNet.
- `merge_embeddings.py`: Merge embeddings with patient data.
- `train_resnet.py`: Train ResNet.
- `train_googlenet.py`: Train GoogLeNet.



---

## References

1. [ODIR-5K Dataset on Kaggle](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k)
2. [XVisionHelper Documentation](https://github.com/moayadeldin/X-vision-helper)
3. [AUROC and AUPRC Analysis for Class Imbalance](https://arxiv.org/abs/2110.02099)
4. [GoogLeNet on Wikipedia](https://en.wikipedia.org/wiki/Inception_%28deep_learning_architecture%29)

---
